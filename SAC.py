# Modern discrete SAC-style implementation with two critics and no value network.
from __future__ import annotations

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from craftax.craftax_env import make_craftax_env_from_name
from dqn import ReplayBuffer, Transition
from logz.batch_logging import batch_log, create_log_dict
from wrappers import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper, OptimisticResetVecEnvWrapper
import gymnax


# Separate nets for clarity
class Actor(nn.Module):
    dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class Critic(nn.Module):
    dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


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

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (
        config["NUM_STEPS"] * config["NUM_ENVS"]
    )

    def train(rng):
        action_dim = env.action_space(env_params).n

        actor_net = Actor(config["LAYER_SIZE"], action_dim)
        q1_net = Critic(config["LAYER_SIZE"], action_dim)
        q2_net = Critic(config["LAYER_SIZE"], action_dim)

        rng, actor_key, q1_key, q2_key = jax.random.split(rng, 4)

        init = jnp.zeros((1, *env.observation_space(env_params).shape))

        actor_params = actor_net.init(actor_key, init)
        q1_params = q1_net.init(q1_key, init)
        q2_params = q2_net.init(q2_key, init)

        rb = ReplayBuffer.create(
            capacity=config["BUFFER_CAPACITY"],
            batch_size=config["BATCH_SIZE"],
            warmup=config["WARMUP"],
            state_shape=env.observation_space(env_params).shape,
            action_shape=(),
            action_dtype=jnp.int32,
        )

        def linear_schedule(count):
            # count is optimizer-step count, not env-step count.
            # This is approximate, but better than integer-dividing by TOTAL_TIMESTEPS.
            frac = 1.0 - (count / max(config["NUM_UPDATES"] * config["NUM_UPDATE_STEPS"], 1))
            frac = jnp.clip(frac, 0.0, 1.0)
            return config["LR"] * frac

        if config["ANNEAL_LR"]:
            tx_actor = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )

            tx_critic = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )

            tx_alpha = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(
                    learning_rate=config.get("ALPHA_LR", 3e-5),
                    eps=1e-5,
                ),
            )
        else:
            tx_actor = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

            tx_critic = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"] * 0.5, eps=1e-5),
            )

            tx_alpha = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config.get("ALPHA_LR", 3e-5), eps=1e-5),
            )

        actor_state = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_params,
            tx=tx_actor,
        )

        q1_state = TrainState.create(
            apply_fn=q1_net.apply,
            params=q1_params,
            tx=tx_critic,
        )

        q2_state = TrainState.create(
            apply_fn=q2_net.apply,
            params=q2_params,
            tx=tx_critic,
        )

        alpha_state = TrainState.create(
            apply_fn=None,
            params={
                "log_alpha": jnp.array(
                    jnp.log(config.get("ENT_TEMP", 0.2)),
                    dtype=jnp.float32,
                )
            },
            tx=tx_alpha,
        )

        train_state = (actor_state, q1_state, q2_state, alpha_state)

        target_q1_params = q1_params
        target_q2_params = q2_params

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)

        def train_loop(run_state, _):
            (
                train_state,
                target_q1_params,
                target_q2_params,
                rb,
                obs,
                env_state,
                rng,
                update_idx,
            ) = run_state

            actor_state, q1_state, q2_state, alpha_state = train_state

            def collect_transitions(carry, _):
                actor_state, obs, env_state, rng = carry

                rng, policy_key, env_key = jax.random.split(rng, 3)

                action_logits = actor_net.apply(actor_state.params, obs)
                policy = distrax.Categorical(logits=action_logits)
                action = policy.sample(seed=policy_key)

                next_obs, next_env_state, reward, done, info = env.step(
                    env_key,
                    env_state,
                    action,
                    env_params,
                )

                transition = Transition(obs, action, reward, next_obs, done)

                next_carry = actor_state, next_obs, next_env_state, rng
                return next_carry, (transition, info)

            rng, _rng = jax.random.split(rng)

            initial_carry = actor_state, obs, env_state, _rng

            rollout_state, (transition, rollout_info) = jax.lax.scan(
                collect_transitions,
                initial_carry,
                xs=None,
                length=config["NUM_STEPS"],
            )

            flat_transition = jax.tree.map(
                lambda x: x.reshape((-1, *x.shape[2:])),
                transition,
            )

            rb = rb.add_batch(flat_transition)

            next_actor_state, next_obs, next_env_state, rng = rollout_state
            next_train_state = (next_actor_state, q1_state, q2_state, alpha_state)

            def update(carry, _):
                def loss_fn(
                    actor_params,
                    q1_params,
                    q2_params,
                    alpha_params,
                    target_q1_params,
                    target_q2_params,
                    transition,
                ):
                    log_alpha = alpha_params["log_alpha"]

                    # Keep alpha positive. Do not optimise raw alpha directly.
                    alpha = jnp.exp(log_alpha)
                    alpha_sg = jax.lax.stop_gradient(alpha)

                    # For Craftax with 43 actions, near-uniform entropy is too high.
                    # log(43) ~= 3.76, so a target around 1.0 is usually more sensible.
                    target_entropy = jnp.asarray(
                        config.get("TARGET_ENTROPY", 1.0),
                        dtype=jnp.float32,
                    )

                    # ------------------------------------------------------------
                    # Critic loss
                    # ------------------------------------------------------------
                    q1_values = q1_net.apply(q1_params, transition.state)
                    q2_values = q2_net.apply(q2_params, transition.state)

                    batch_idx = jnp.arange(transition.state.shape[0])
                    action = transition.action.astype(jnp.int32)

                    q1_selected = q1_values[batch_idx, action]
                    q2_selected = q2_values[batch_idx, action]

                    next_action_logits = actor_net.apply(
                        actor_params,
                        transition.next_state,
                    )
                    next_probs = jax.nn.softmax(next_action_logits, axis=-1)
                    next_log_probs = jax.nn.log_softmax(next_action_logits, axis=-1)

                    target_q1 = q1_net.apply(
                        target_q1_params,
                        transition.next_state,
                    )
                    target_q2 = q2_net.apply(
                        target_q2_params,
                        transition.next_state,
                    )
                    target_q = jnp.minimum(target_q1, target_q2)

                    # Correct discrete SAC soft value:
                    #
                    # V(s') = sum_a pi(a|s') * [Q_target(s', a) - alpha * log pi(a|s')]
                    #
                    # Since log pi <= 0, this is equivalent to expected_q + alpha * entropy.
                    next_v = jnp.sum(
                        next_probs * (target_q - alpha_sg * next_log_probs),
                        axis=-1,
                    )

                    target = transition.reward + config["GAMMA"] * (
                        1.0 - transition.done.astype(jnp.float32)
                    ) * next_v

                    target = jax.lax.stop_gradient(target)

                    q1_loss = 0.5 * jnp.mean((q1_selected - target) ** 2)
                    q2_loss = 0.5 * jnp.mean((q2_selected - target) ** 2)

                    # ------------------------------------------------------------
                    # Actor loss
                    # ------------------------------------------------------------
                    action_logits = actor_net.apply(actor_params, transition.state)
                    probs = jax.nn.softmax(action_logits, axis=-1)
                    log_probs = jax.nn.log_softmax(action_logits, axis=-1)

                    # Stop critic gradients inside actor loss.
                    q_values = jax.lax.stop_gradient(
                        jnp.minimum(
                            q1_net.apply(q1_params, transition.state),
                            q2_net.apply(q2_params, transition.state),
                        )
                    )

                    # Equivalent to maximising E[Q + alpha * H].
                    actor_loss = jnp.mean(
                        jnp.sum(
                            probs * (alpha_sg * log_probs - q_values),
                            axis=-1,
                        )
                    )

                    """
                    q_gap = (q_values.max(axis=-1) - q_values.min(axis=-1)).mean()
                    q_std = q_values.std()
                    max_action_prob = probs.max(axis=-1).mean()
                    min_action_prob = probs.min(axis=-1).mean()"""

                    entropy = -jnp.sum(probs * log_probs, axis=-1)
                    expected_q = jnp.sum(probs * q_values, axis=-1)

                    # ------------------------------------------------------------
                    # Alpha loss
                    # ------------------------------------------------------------
                    # Behaviour:
                    # entropy > target -> alpha decreases
                    # entropy < target -> alpha increases
                    alpha_loss = jnp.mean(
                        alpha * jax.lax.stop_gradient(entropy - target_entropy)
                    )

                    total_loss = actor_loss + q1_loss + q2_loss + alpha_loss

                    return total_loss, (
                        actor_loss,
                        q1_loss,
                        q2_loss,
                        alpha_loss,
                        entropy.mean(),
                        alpha,
                        expected_q.mean(),
                    )

                train_state, target_q1_params, target_q2_params, rng = carry
                actor_state, q1_state, q2_state, alpha_state = train_state

                rng, _rng = jax.random.split(rng)
                transition_batch = rb.sample(_rng)

                (total_loss, losses), grads = jax.value_and_grad(
                    loss_fn,
                    has_aux=True,
                    argnums=(0, 1, 2, 3),
                )(
                    actor_state.params,
                    q1_state.params,
                    q2_state.params,
                    alpha_state.params,
                    target_q1_params,
                    target_q2_params,
                    transition_batch,
                )

                actor_grads, q1_grads, q2_grads, alpha_grads = grads

                actor_state = actor_state.apply_gradients(grads=actor_grads)
                q1_state = q1_state.apply_gradients(grads=q1_grads)
                q2_state = q2_state.apply_gradients(grads=q2_grads)
                alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

                train_state = (actor_state, q1_state, q2_state, alpha_state)

                target_q1_params = jax.tree.map(
                    lambda target, online: (1.0 - config["TAU"]) * target
                    + config["TAU"] * online,
                    target_q1_params,
                    q1_state.params,
                )

                target_q2_params = jax.tree.map(
                    lambda target, online: (1.0 - config["TAU"]) * target
                    + config["TAU"] * online,
                    target_q2_params,
                    q2_state.params,
                )

                return (
                    train_state,
                    target_q1_params,
                    target_q2_params,
                    rng,
                ), (total_loss, losses)

            rng, _rng = jax.random.split(rng)

            ready = rb.size >= max(config["WARMUP"], config["BATCH_SIZE"])

            def do_updates(carry):
                train_state, target_q1_params, target_q2_params, rng = carry

                (
                    train_state,
                    target_q1_params,
                    target_q2_params,
                    rng,
                ), (total_loss, losses) = jax.lax.scan(
                    update,
                    (train_state, target_q1_params, target_q2_params, rng),
                    xs=None,
                    length=config["NUM_UPDATE_STEPS"],
                )

                mean_loss = total_loss.mean()
                mean_losses = jax.tree.map(lambda x: x.mean(), losses)

                return (
                    train_state,
                    target_q1_params,
                    target_q2_params,
                    rng,
                    mean_loss,
                    mean_losses,
                )

            def skip_updates(carry):
                train_state, target_q1_params, target_q2_params, rng = carry

                mean_loss = jnp.array(jnp.nan, dtype=jnp.float32)
                loss = jnp.array(jnp.nan, dtype=jnp.float32)

                # actor_loss, q1_loss, q2_loss, alpha_loss, entropy, alpha, expected_q
                losses = (loss, loss, loss, loss, loss, loss, loss)

                return (
                    train_state,
                    target_q1_params,
                    target_q2_params,
                    rng,
                    mean_loss,
                    losses,
                )

            (
                train_state,
                target_q1_params,
                target_q2_params,
                rng,
                mean_loss,
                losses,
            ) = jax.lax.cond(
                ready,
                do_updates,
                skip_updates,
                operand=(next_train_state, target_q1_params, target_q2_params, _rng),
            )

            episode_mask = rollout_info["returned_episode"].astype(jnp.float32)
            num_completed_episodes = episode_mask.sum()

            episode_metric = jax.tree.map(
                lambda x: jnp.where(
                    num_completed_episodes > 0,
                    (x * episode_mask).sum() / num_completed_episodes,
                    jnp.nan,
                ),
                rollout_info,
            )

            global_step = update_idx * config["NUM_ENVS"] * config["NUM_STEPS"]

            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, global_step, losses):
                    to_log = create_log_dict(metric, config)

                    to_log.update(
                        {
                            "global_step": int(global_step),
                            "loss/actor_loss": float(losses[0]),
                            "loss/q1_loss": float(losses[1]),
                            "loss/q2_loss": float(losses[2]),
                            "loss/alpha_loss": float(losses[3]),
                            "entropy": float(losses[4]),
                            "alpha": float(losses[5]),
                            "expected_q": float(losses[6]),
                        }
                    )

                    batch_log(global_step, to_log, config)

                jax.debug.callback(callback, episode_metric, global_step, losses)

            run_state = (
                train_state,
                target_q1_params,
                target_q2_params,
                rb,
                next_obs,
                next_env_state,
                rng,
                update_idx + 1,
            )

            return run_state, episode_metric

        run_state = (
            train_state,
            target_q1_params,
            target_q2_params,
            rb,
            obs,
            env_state,
            rng,
            jnp.array(0, dtype=jnp.int32),
        )

        run_state, _ = jax.lax.scan(
            train_loop,
            run_state,
            xs=None,
            length=config["NUM_UPDATES"],
        )

        return run_state

    return train