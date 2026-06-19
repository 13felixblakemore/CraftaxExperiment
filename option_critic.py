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

import configs
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn

from configs import debug_config
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


class OptionCritic(nn.Module):
    num_options: int
    action_dim: int
    dim: int

    @nn.compact
    def __call__(self, s):
        # s is shape (b, obs_shape)
        # maybe dense instead of conv for symbolic

        s = nn.Dense(self.dim)(s)
        s = nn.relu(s)
        s = nn.Dense(self.dim)(s)
        s = nn.relu(s)

        q_w = nn.Dense(self.num_options)(s) # q_w shape: (n) -- choose policy with epsilon greedy
        b = nn.Dense(self.num_options, bias_init=nn.initializers.constant(-2.0))(s) # b shape: (n) -- terminate the active option i with probability b_i (sigmoid)
        actions = nn.Dense(self.num_options * self.action_dim)(s) # actions shape: (n * action_dim)
        actions = actions.reshape((s.shape[0], self.num_options, self.action_dim)) # actions shape: (n, action_dim)

        return q_w, b, actions


def make_train(config):
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
        network = OptionCritic(config["NUM_OPTIONS"], env.action_space(env_params).n, config["LAYER_SIZE"])
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

        rng, _rng = jax.random.split(rng)
        option = epsilon_greedy_options(_rng, q_w, config["OPTION_POLICY_EPS"])

        def update_step(run_state, _):
            def rollout_step(carry, transition):
                train_state, obs, env_states, rng, option = carry

                # choose action for each env
                # calc log_probs of each action
                rng, _rng = jax.random.split(rng)
                values, b, action_logits = network.apply(train_state.params, obs)
                logits_o = action_logits[jnp.arange(config["NUM_ENVS"]), option, :]
                policy = distrax.Categorical(logits=logits_o)
                actions = policy.sample(seed=_rng)
                log_probs = policy.log_prob(actions)

                rng, _rng = jax.random.split(_rng)
                next_obs, next_env_states, rewards, dones, infos = env.step(
                    _rng,
                    env_states,
                    actions,
                    env_params,
                )

                q_w, b_next, _ = network.apply(train_state.params, next_obs)
                b_next = nn.sigmoid(b_next)

                b_next_o = b_next[jnp.arange(config["NUM_ENVS"]), option]

                rng, _rng = jax.random.split(_rng)
                terminate = dones | jax.random.bernoulli(_rng, b_next_o)

                rng, _rng = jax.random.split(_rng)
                new_option = epsilon_greedy_options(_rng, q_w, config["OPTION_POLICY_EPS"])
                new_option = jnp.where((terminate == 1), new_option, option)

                transition = Transition(dones, actions, values, rewards, log_probs, obs, next_obs, infos, option, b)
                new_carry = train_state, next_obs, next_env_states, rng, new_option
                return new_carry, transition


            train_state, obs, env_states, rng, option, update_idx = run_state

            rollout_state = (train_state, obs, env_states, rng, option)

            rollout_state, rollout = jax.lax.scan(
                rollout_step,
                rollout_state,
                xs=None,
                length=config["NUM_STEPS"],
            )

            train_state, obs, env_states, rng, option = rollout_state

            def compute_gae(rollout, last_q, last_b, option):
                def gae_step(carry, transition):
                    last_gae, next_value, next_b = carry
                    reward, value, done, option, rollout_b_logits = transition
                    next_b = nn.sigmoid(next_b)

                    next_non_terminal = 1.0 - done.astype(jnp.float32)

                    bootstrap = ((1 - next_b[jnp.arange(config["NUM_ENVS"]), option]) * next_value[jnp.arange(config["NUM_ENVS"]), option] + next_b[jnp.arange(config["NUM_ENVS"]), option] * jnp.max(next_value, axis=-1))

                    delta = reward + config["GAMMA"] * bootstrap * next_non_terminal - value[jnp.arange(config["NUM_ENVS"]), option]

                    last_gae = (
                            delta
                            + config["GAMMA"]
                            * config["GAE_LAMBDA"]
                            * next_non_terminal
                            * last_gae
                    )

                    return (last_gae, value, rollout_b_logits), last_gae

                initial_carry = (
                    jnp.zeros_like(option, dtype=jnp.float32),
                    last_q,
                    last_b
                )

                _, advantages = jax.lax.scan(
                    gae_step,
                    initial_carry,
                    (rollout.reward, rollout.value, rollout.done, rollout.option, rollout.b),
                    reverse=True,
                    unroll=16,
                )

                values = rollout.value[
                    jnp.arange(config["NUM_STEPS"])[:, None],
                    jnp.arange(config["NUM_ENVS"])[None, :],
                    rollout.option.astype(jnp.int32)
                ]

                returns = advantages + values

                return advantages, returns

            q_w_next, b_next, action_logits_next = network.apply(train_state.params, obs)

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

                        b_next = nn.sigmoid(b_next_logits)
                        beta_next_o = b_next[batch_idx, options]
                        q_next_o = q_w_next[batch_idx, options]

                        # Greedy baseline version
                        v_next = jnp.max(q_w_next, axis=-1)

                        termination_advantage = q_next_o - v_next
                        termination_advantage = jax.lax.stop_gradient(
                            termination_advantage + config["DELIB_COST"]
                        )

                        nonterminal = 1.0 - rollout.done.astype(jnp.float32)

                        termination_loss = jnp.mean(
                            nonterminal * beta_next_o * termination_advantage
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
                update_idx + 1,
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        run_state = (train_state, obs, env_states, _rng, option, jnp.array(0))

        run_state, metric = jax.lax.scan(
            update_step,
            run_state,
            None,
            length=config["NUM_UPDATES"],
        )

        return {"runner_state": run_state}
    return train