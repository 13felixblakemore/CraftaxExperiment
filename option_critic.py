import json
import sys
from pathlib import Path
from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax.serialization import to_bytes
from flax.training.train_state import TrainState

import wandb
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn

from logz.batch_logging import create_log_dict, batch_log
from ppo_shared import LogWrapper
from wrappers import OptimisticResetVecEnvWrapper, AutoResetEnvWrapper, BatchEnvWrapper


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray
    option: jnp.ndarray
    b: jnp.ndarray
    option_boundary: jnp.ndarray
    remaining: jnp.ndarray


# could combine HiPPO duration mechanism with termination function
# can compare different duration distributions e.g. fixed, random, learned
# use stochastic manager instead of e greedy pi(w given s)

# to do:
# - add termination function / duration toggle

class SharedOptionCritic(nn.Module):
    num_options: int
    action_dim: int
    dim: int

    @nn.compact
    def __call__(self, s):
        # s is shape (num_envs, obs_shape)

        s = nn.Dense(self.dim)(s)
        s = nn.relu(s)
        s = nn.Dense(self.dim)(s)
        s = nn.relu(s)

        q_w = nn.Dense(self.num_options)(s) # q_w shape: (num_envs, num_options)
        b = nn.Dense(self.num_options, bias_init=nn.initializers.constant(-2.0))(s) # b shape: (n) -- terminate the active option i with probability b_i (sigmoid)
        actions = nn.Dense(self.num_options * self.action_dim)(s) # actions shape: (n * action_dim)
        actions = actions.reshape((s.shape[0], self.num_options, self.action_dim)) # actions shape: (n, action_dim)

        return q_w, b, actions


class SingleOption(nn.Module):
    action_dim: int
    dim: int

    @nn.compact
    def __call__(self, obs):
        features = nn.Dense(self.dim)(obs)
        features = nn.relu(features)
        features = nn.Dense(self.dim)(features)
        features = nn.relu(features)

        action_logits = nn.Dense(self.action_dim)(features)
        q_option = nn.Dense(1)(features).squeeze(-1)
        beta_logit = nn.Dense(
            1,
            bias_init=nn.initializers.constant(-2.0),
        )(features).squeeze(-1)

        return q_option, beta_logit, action_logits


class SeparateOptionCritic(nn.Module):
    num_options: int
    action_dim: int
    dim: int

    @nn.compact
    def __call__(self, obs):
        q_values = []
        beta_logits = []
        action_logits = []

        for option in range(self.num_options):
            q, beta, logits = SingleOption(
                action_dim=self.action_dim,
                dim=self.dim,
                name=f"option_{option}",
            )(obs)

            q_values.append(q)
            beta_logits.append(beta)
            action_logits.append(logits)

        return (
            jnp.stack(q_values, axis=-1),
            jnp.stack(beta_logits, axis=-1),
            jnp.stack(action_logits, axis=1),
        )


def make_option_critic_network(config, action_dim):
    """Construct the configured Option-Critic architecture.

    Both implementations expose the same outputs:
        q_w:          (batch_size, num_options)
        beta_logits:  (batch_size, num_options)
        action_logits:(batch_size, num_options, action_dim)
    """
    network_cls = (
        SharedOptionCritic
        if config["SHARED_NETWORK"]
        else SeparateOptionCritic
    )

    return network_cls(
        num_options=config["NUM_OPTIONS"],
        action_dim=action_dim,
        dim=config["LAYER_SIZE"],
    )


def make_train(config):
    # Temporal abstraction mode:
    #   learned: sample termination from beta_o(s)
    #   fixed:   execute every option for FIXED_OPTION_LENGTH steps
    #   uniform: sample each option length uniformly from
    #            [MIN_OPTION_LENGTH, MAX_OPTION_LENGTH]

    duration_mode = config["OPTION_DURATION_MODE"].lower()
    valid_duration_modes = {"learned", "fixed", "uniform"}
    if duration_mode not in valid_duration_modes:
        raise ValueError(
            "OPTION_DURATION_MODE must be one of "
            f"{sorted(valid_duration_modes)}, got {duration_mode!r}"
        )
    if config["FIXED_OPTION_LENGTH"] < 1:
        raise ValueError("FIXED_OPTION_LENGTH must be at least 1")
    if config["MIN_OPTION_LENGTH"] < 1:
        raise ValueError("MIN_OPTION_LENGTH must be at least 1")
    if config["MAX_OPTION_LENGTH"] < config["MIN_OPTION_LENGTH"]:
        raise ValueError(
            "MAX_OPTION_LENGTH must be greater than or equal to "
            "MIN_OPTION_LENGTH"
        )

    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = env.default_params
    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"]*config["NUM_STEPS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"]*config["NUM_STEPS"]) // config["NUM_MINIBATCHES"]

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        network = make_option_critic_network(
            config=config,
            action_dim=env.action_space(env_params).n,
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        obs, env_states = env.reset(_rng, env_params)

        # choose w according to Q_w(obs)
        q_w, b, action_logits = network.apply(train_state.params, obs)

        # change this (?)
        def epsilon_greedy_options(rng, q_w, epsilon):
            batch_size = q_w.shape[0]

            greedy_option = jnp.argmax(q_w, axis=-1)

            rng_random, rng_uniform = jax.random.split(rng)
            random_option = jax.random.randint(
                rng_random,
                shape=(batch_size,),
                minval=0,
                maxval=q_w.shape[-1],
            )

            choose_random = jax.random.uniform(rng_uniform, shape=(batch_size,)) < epsilon

            return jnp.where(choose_random, random_option, greedy_option)

        def sample_option_duration(rng, batch_size):
            """Sample a new commitment length for each environment."""
            if duration_mode == "fixed":
                return jnp.full(
                    (batch_size,),
                    config["FIXED_OPTION_LENGTH"],
                    dtype=jnp.int32,
                )
            if duration_mode == "uniform":
                return jax.random.randint(
                    rng,
                    shape=(batch_size,),
                    minval=config["MIN_OPTION_LENGTH"],
                    maxval=config["MAX_OPTION_LENGTH"] + 1,
                    dtype=jnp.int32,
                )
            # The learned-beta mode does not use the timer. Keeping a
            # placeholder gives every mode the same scan carry structure.
            return jnp.zeros((batch_size,), dtype=jnp.int32)

        rng, _rng = jax.random.split(rng)
        option = epsilon_greedy_options(_rng, q_w, config["OPTION_POLICY_EPS"])

        rng, _rng = jax.random.split(rng)
        remaining = sample_option_duration(_rng, config["NUM_ENVS"])

        def update_step(run_state, _):
            def rollout_step(carry, _):
                train_state, obs, env_states, rng, option, remaining = carry

                # Execute the intra-option policy for the active option.
                rng, action_key = jax.random.split(rng)
                values, beta_logits, action_logits = network.apply(
                    train_state.params, obs
                )
                env_idx = jnp.arange(config["NUM_ENVS"])
                logits_o = action_logits[env_idx, option, :]
                policy = distrax.Categorical(logits=logits_o)
                actions = policy.sample(seed=action_key)
                log_probs = policy.log_prob(actions)

                rng, env_key = jax.random.split(rng)
                next_obs, next_env_states, rewards, dones, infos = env.step(
                    env_key,
                    env_states,
                    actions,
                    env_params,
                )

                q_w_next, beta_next_logits, _ = network.apply(
                    train_state.params, next_obs
                )

                if duration_mode == "learned":
                    beta_next = nn.sigmoid(beta_next_logits)
                    beta_next_o = beta_next[env_idx, option]
                    rng, termination_key = jax.random.split(rng)
                    option_boundary = dones | jax.random.bernoulli(
                        termination_key, beta_next_o
                    )
                    remaining_after = remaining
                else:
                    # `remaining` includes the action just taken. A value of
                    # one therefore reaches an option boundary after this step.
                    remaining_after = remaining - 1
                    option_boundary = dones | (remaining_after <= 0)

                # The original Option-Critic outer policy is epsilon-greedy
                # over Q_Omega and is consulted only at an option boundary.
                rng, option_key = jax.random.split(rng)
                candidate_option = epsilon_greedy_options(
                    option_key,
                    q_w_next,
                    config["OPTION_POLICY_EPS"],
                )
                next_option = jnp.where(
                    option_boundary,
                    candidate_option,
                    option,
                )

                if duration_mode == "learned":
                    next_remaining = remaining
                else:
                    rng, duration_key = jax.random.split(rng)
                    sampled_duration = sample_option_duration(
                        duration_key, config["NUM_ENVS"]
                    )
                    next_remaining = jnp.where(
                        option_boundary,
                        sampled_duration,
                        remaining_after,
                    )

                transition = Transition(
                    done=dones,
                    action=actions,
                    value=values,
                    reward=rewards,
                    log_prob=log_probs,
                    obs=obs,
                    next_obs=next_obs,
                    info=infos,
                    option=option,
                    b=beta_logits,
                    option_boundary=option_boundary,
                    remaining=remaining,
                )

                new_carry = (
                    train_state,
                    next_obs,
                    next_env_states,
                    rng,
                    next_option,
                    next_remaining,
                )
                return new_carry, transition

            (
                train_state,
                obs,
                env_states,
                rng,
                option,
                remaining,
                update_idx,
            ) = run_state

            rollout_state = (
                train_state,
                obs,
                env_states,
                rng,
                option,
                remaining,
            )

            rollout_state, rollout = jax.lax.scan(
                rollout_step,
                rollout_state,
                xs=None,
                length=config["NUM_STEPS"],
            )

            (
                train_state,
                obs,
                env_states,
                rng,
                option,
                remaining,
            ) = rollout_state

            def compute_gae(rollout, last_q, last_b, option):
                def gae_step(carry, transition):
                    last_gae, next_value, next_b_logits = carry
                    (
                        reward,
                        value,
                        done,
                        current_option,
                        rollout_b_logits,
                        option_boundary,
                    ) = transition

                    env_idx = jnp.arange(config["NUM_ENVS"])
                    next_non_terminal = 1.0 - done.astype(jnp.float32)

                    q_continue = next_value[env_idx, current_option]
                    q_switch = jnp.max(next_value, axis=-1)

                    if duration_mode == "learned":
                        # Standard expected Option-Critic continuation target.
                        next_beta = nn.sigmoid(next_b_logits)
                        beta_next_o = next_beta[env_idx, current_option]
                        bootstrap = (
                            (1.0 - beta_next_o) * q_continue
                            + beta_next_o * q_switch
                        )
                    else:
                        # Forced-duration modes switch only when the timer ends.
                        bootstrap = jnp.where(
                            option_boundary,
                            q_switch,
                            q_continue,
                        )

                    selected_value = value[env_idx, current_option]
                    delta = (
                        reward
                        + config["GAMMA"] * bootstrap * next_non_terminal
                        - selected_value
                    )

                    last_gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * next_non_terminal
                        * last_gae
                    )

                    return (
                        last_gae,
                        value,
                        rollout_b_logits,
                    ), last_gae

                initial_carry = (
                    jnp.zeros_like(option, dtype=jnp.float32),
                    last_q,
                    last_b,
                )

                _, advantages = jax.lax.scan(
                    gae_step,
                    initial_carry,
                    (
                        rollout.reward,
                        rollout.value,
                        rollout.done,
                        rollout.option,
                        rollout.b,
                        rollout.option_boundary,
                    ),
                    reverse=True,
                    unroll=16,
                )

                values = rollout.value[
                    jnp.arange(config["NUM_STEPS"])[:, None],
                    jnp.arange(config["NUM_ENVS"])[None, :],
                    rollout.option.astype(jnp.int32),
                ]

                returns = advantages + values
                return advantages, returns

            q_w_next, b_next, _ = network.apply(train_state.params, obs)

            advantages, returns = compute_gae(rollout, q_w_next, b_next, option)

            def update_epoch(update_state, _):
                def update_minibatch(train_state, batch_info):
                    rollout, advantages, returns = batch_info
                    def loss_fn(params, rollout, gae, returns):
                        q_w, b, action_logits = network.apply(params, rollout.obs)
                        q_w_next, b_next_logits, _ = network.apply(params, rollout.next_obs)

                        B = rollout.obs.shape[0]
                        batch_idx = jnp.arange(B)
                        options = rollout.option.astype(jnp.int32)

                        logits_o = action_logits[batch_idx, options, :]

                        policy = distrax.Categorical(logits=logits_o)

                        new_log_probs = policy.log_prob(rollout.action)
                        old_log_probs = rollout.log_prob
                        log_ratio = new_log_probs - old_log_probs
                        ratio = jnp.exp(log_ratio)

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss1 = ratio * gae
                        loss2 = (
                                jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"])
                                * gae
                        )
                        actor_loss = -jnp.mean(jnp.minimum(loss1, loss2))

                        entropy = policy.entropy().mean()

                        B = rollout.obs.shape[0]
                        batch_idx = jnp.arange(B)
                        options = rollout.option.astype(jnp.int32)

                        values = q_w[batch_idx, options]
                        critic_loss = jnp.mean((values - returns) ** 2)

                        if duration_mode == "learned":
                            b_next = nn.sigmoid(b_next_logits)
                            beta_next_o = b_next[batch_idx, options]
                            q_next_o = q_w_next[batch_idx, options]

                            # Greedy outer-policy baseline from original OC.
                            v_next = jnp.max(q_w_next, axis=-1)
                            termination_advantage = q_next_o - v_next
                            termination_advantage = jax.lax.stop_gradient(
                                termination_advantage + config["DELIB_COST"]
                            )

                            nonterminal = (
                                1.0
                                - rollout.done.astype(jnp.float32)
                            )
                            termination_loss = jnp.mean(
                                nonterminal
                                * beta_next_o
                                * termination_advantage
                            )
                        else:
                            # Beta heads remain in the model for a controlled
                            # architecture comparison, but receive no loss.
                            termination_loss = jnp.zeros(
                                (), dtype=actor_loss.dtype
                            )

                        total_loss = actor_loss + config["VF_COEF"] * critic_loss - config["ENT_COEF"] * entropy + termination_loss

                        aux = {
                            "actor_loss": actor_loss,
                            "critic_loss": critic_loss,
                            "entropy": entropy,
                            "termination_loss": termination_loss,
                        }

                        return total_loss, aux

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (total_loss, losses), grads = grad_fn(
                        train_state.params, rollout, advantages, returns
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, rollout, advantages, returns, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                        batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"

                permutation = jax.random.permutation(_rng, batch_size)
                batch = (rollout, advantages, returns)

                # flatten rollout batch into num_envs * steps for each item (actions, states, rewards, etc)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )

                # shuffle the flattened batch with the permutation
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                # reshape into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    update_minibatch, train_state, minibatches
                )
                update_state = (train_state, rollout, advantages, returns, rng)
                return update_state, total_loss

            update_state = train_state, rollout, advantages, returns, rng

            update_state, total_loss = jax.lax.scan(
                update_epoch,
                update_state,
                None,
                config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * rollout.info["returned_episode"]).sum()
                          / rollout.info["returned_episode"].sum(),
                rollout.info,
            )

            rng = update_state[-1]

            if config["DEBUG"] and config["USE_WANDB"]:
                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(
                    callback,
                    metric,
                    update_idx,
                )

            runner_state = (
                train_state,
                obs,
                env_states,
                rng,
                option,
                remaining,
                update_idx + 1,
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        run_state = (
            train_state,
            obs,
            env_states,
            _rng,
            option,
            remaining,
            jnp.array(0),
        )

        run_state, metric = jax.lax.scan(
            update_step,
            run_state,
            None,
            length=config["NUM_UPDATES"],
        )

        return {"runner_state": run_state}
    return train