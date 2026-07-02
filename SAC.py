# Modern discrete SAC-style implementation with two critics and no value network.
from __future__ import annotations

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

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

    # Create environment
    env, env_params = gymnax.make("CartPole-v1")
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

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"])

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
            frac = (
                1.0
                - (count // (config["TOTAL_TIMESTEPS"]))
            )
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
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
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
                optax.adam(config["LR"] * 0.5, eps=1e-5),
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
            params={"log_alpha": jnp.array(jnp.log(config.get("ENT_TEMP", 0.2)), dtype=jnp.float32)},
            tx=tx_alpha,
        )

        train_state = (actor_state, q1_state, q2_state, alpha_state)

        target_q1_params = q1_params
        target_q2_params = q2_params

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)

        def train_loop(run_state, _):
            train_state, target_q1_params, target_q2_params, rb, obs, env_state, rng, update_idx = run_state
            actor_state, q1_state, q2_state, alpha_state = train_state

            def collect_transitions(carry, _):
                actor_state, obs, env_state, rng = carry

                rng, logits_key, env_key = jax.random.split(rng, 3)
                action_logits = actor_net.apply(actor_state.params, obs)
                policy = distrax.Categorical(logits=action_logits)
                action = policy.sample(seed=logits_key)

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
                def loss_fn(actor_params, q1_params, q2_params, alpha_params, target_q1_params, target_q2_params, transition):
                    log_alpha = alpha_params["log_alpha"]
                    alpha = jnp.exp(jnp.clip(log_alpha, -20.0, 2.0))
                    target_entropy = -jnp.log(1.0 / action_dim)

                    q1_values = q1_net.apply(q1_params, transition.state)
                    q2_values = q2_net.apply(q2_params, transition.state)
                    q1_selected = q1_values[
                        jnp.arange(transition.state.shape[0]),
                        transition.action.astype(jnp.int32),
                    ]
                    q2_selected = q2_values[
                        jnp.arange(transition.state.shape[0]),
                        transition.action.astype(jnp.int32),
                    ]

                    next_action_logits = actor_net.apply(actor_params, transition.next_state)
                    next_probs = jax.nn.softmax(next_action_logits, axis=-1)
                    next_log_probs = jax.nn.log_softmax(next_action_logits, axis=-1)
                    next_q1 = q1_net.apply(target_q1_params, transition.next_state)
                    next_q2 = q2_net.apply(target_q2_params, transition.next_state)
                    next_q = jnp.minimum(next_q1, next_q2)
                    next_q_target = jnp.sum(next_probs * next_q, axis=-1)
                    next_entropy = -jnp.sum(next_probs * next_log_probs, axis=-1)

                    target = transition.reward + config["GAMMA"] * (1 - transition.done) * (
                        next_q_target - alpha * next_entropy
                    )
                    target = jax.lax.stop_gradient(target)

                    q1_loss = 0.5 * jnp.mean((q1_selected - target) ** 2)
                    q2_loss = 0.5 * jnp.mean((q2_selected - target) ** 2)

                    action_logits = actor_net.apply(actor_params, transition.state)
                    probs = jax.nn.softmax(action_logits, axis=-1)
                    log_probs = jax.nn.log_softmax(action_logits, axis=-1)
                    q_values = jnp.minimum(
                        q1_net.apply(q1_params, transition.state),
                        q2_net.apply(q2_params, transition.state),
                    )
                    expected_q = jnp.sum(probs * q_values, axis=-1)
                    entropy = -jnp.sum(probs * log_probs, axis=-1)

                    actor_loss = -jnp.mean(expected_q + alpha * entropy)
                    alpha_loss = -jnp.mean(log_alpha * jax.lax.stop_gradient(entropy + target_entropy))
                    total_loss = actor_loss + q1_loss + q2_loss + alpha_loss

                    return total_loss, (actor_loss, q1_loss, q2_loss, alpha_loss, entropy, expected_q)

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
                    lambda target, online: (1 - config["TAU"]) * target + config["TAU"] * online,
                    target_q1_params,
                    q1_state.params,
                )
                target_q2_params = jax.tree.map(
                    lambda target, online: (1 - config["TAU"]) * target + config["TAU"] * online,
                    target_q2_params,
                    q2_state.params,
                )

                return (train_state, target_q1_params, target_q2_params, rng), (total_loss, losses)

            rng, _rng = jax.random.split(rng)
            ready = rb.size >= max(config["WARMUP"], config["BATCH_SIZE"])

            def do_updates(carry):
                train_state, target_q1_params, target_q2_params, rng = carry
                (train_state, target_q1_params, target_q2_params, rng), (total_loss, losses) = jax.lax.scan(
                    update,
                    (train_state, target_q1_params, target_q2_params, rng),
                    xs=None,
                    length=config["NUM_UPDATE_STEPS"],
                )
                mean_loss = total_loss.mean()
                mean_losses = jax.tree.map(lambda x: x.mean(), losses)
                return train_state, target_q1_params, target_q2_params, rng, mean_loss, mean_losses

            def skip_updates(carry):
                train_state, target_q1_params, target_q2_params, rng = carry
                mean_loss = jnp.array(jnp.nan, dtype=jnp.float32)
                loss = jnp.array(jnp.nan, dtype=jnp.float32)
                losses = (loss, loss, loss, loss, loss, loss)
                return train_state, target_q1_params, target_q2_params, rng, mean_loss, losses

            train_state, target_q1_params, target_q2_params, rng, mean_loss, losses = jax.lax.cond(
                ready,
                do_updates,
                skip_updates,
                operand=(next_train_state, target_q1_params, target_q2_params, _rng),
            )

            episode_mask = rollout_info["returned_episode"].astype(jnp.float32)
            num_completed_episodes = episode_mask.sum()
            denominator = jnp.maximum(num_completed_episodes, 1.0)
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
                        }
                    )
                    batch_log(global_step, to_log, config)

                jax.debug.callback(callback, episode_metric, global_step, losses)

            run_state = train_state, target_q1_params, target_q2_params, rb, next_obs, next_env_state, rng, update_idx + 1
            return run_state, episode_metric

        run_state = train_state, target_q1_params, target_q2_params, rb, obs, env_state, rng, jnp.array(0, dtype=jnp.int32)
        run_state, _ = jax.lax.scan(train_loop, run_state, xs=None, length=config["NUM_UPDATES"])
        return run_state

    return train