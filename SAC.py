# Value network: estimates expected return under current policy
# Q network: estimates expected return from (s, a) pair
# Actor: chooses action based on state, s
# Replay Buffer
from __future__ import annotations
from typing import Sequence
import distrax
import jax.numpy
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn, struct
import jax.numpy as jnp
from flax.training.train_state import TrainState
from logz.batch_logging import create_log_dict, batch_log
from wrappers import LogWrapper, OptimisticResetVecEnvWrapper, AutoResetEnvWrapper, BatchEnvWrapper
from dqn import Transition, ReplayBuffer

import optax

# Estimate expected return from (s, a) pair under current policy
class Actor(nn.Module):
    dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x

class Critic(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)

        x = nn.Dense(1)(x)
        return x


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

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"]*config["NUM_ENVS"])

    def train(rng):
        actor_net = Actor(config["LAYER_SIZE"], env.action_space(env_params).n)
        q_net = Actor(config["LAYER_SIZE"], env.action_space(env_params).n)
        critic_net = Critic(config["LAYER_SIZE"])

        rng, _rng = jax.random.split(rng, 2)
        init = jnp.zeros((1, *env.observation_space(env_params).shape))
        actor_params = actor_net.init(_rng, init)
        q_params = q_net.init(_rng, init)
        critic_params = critic_net.init(_rng, init)

        rb = ReplayBuffer.create(
            capacity=config["BUFFER_CAPACITY"],
            batch_size=config["BATCH_SIZE"],
            warmup=config["WARMUP"],
            state_shape=env.observation_space(env_params).shape,
            action_shape=(),
            action_dtype=jnp.float32,
        )

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

        train_state = TrainState.create(
            apply_fn=None,
            params={"a": actor_params, "q": q_params, "v": critic_params},
            tx=tx,
        )

        target_params = train_state.params["v"]

        rng, _rng = jax.random.split(rng, 2)
        obs, env_state = env.reset(_rng, env_params)
        def train_loop(run_state, _):
            train_state, target_params, rb, obs, env_state, rng, update_idx = run_state
            def collect_transitions(carry, _):
                train_state, obs, env_state, rng = carry

                rng, _rng = jax.random.split(rng)
                action_logits = actor_net.apply(train_state.params["a"], obs)
                policy = distrax.Categorical(logits=action_logits)
                action = policy.sample(seed=_rng)
                print(action.shape)

                rng, _rng = jax.random.split(rng)
                next_obs, next_env_state, reward, done, info = env.step(
                    _rng,
                    env_state,
                    action,
                    env_params)

                transition = Transition(obs, action, reward, next_obs, done)

                rng, _rng = jax.random.split(rng)
                next_carry  = train_state, next_obs, next_env_state, _rng
                return next_carry, (transition, info)
            
            rng, _rng = jax.random.split(rng)
            initial_carry = train_state, obs, env_state, _rng
            rollout_state, (transition, rollout_info) = jax.lax.scan(collect_transitions,
                                                    initial_carry,
                                                    xs=None,
                                                    length=config["NUM_STEPS"])

            flat_transition = jax.tree.map(
                lambda x: x.reshape((-1, *x.shape[2:])),
                transition,
            )

            rb = rb.add_batch(flat_transition)

            next_train_state, next_obs, next_env_state, rng = rollout_state
            
            def update(carry, _):
                def loss_fn(params, target_params, transition):
                    
                    def value_loss(v_params, q_params, a_params, target_params, transition):
                        value_estimate = critic_net.apply(v_params, transition.state)
                        q_values = q_net.apply(q_params, transition.state)
                        action_logits = actor_net.apply(a_params, transition.state)
                        policy = distrax.Categorical(logits=action_logits)
                        probs = jax.nn.softmax(action_logits, axis=-1)
                        expected_return = jnp.sum(probs * q_values, axis=-1)
                        target = jax.lax.stop_gradient(expected_return - config["ENT_TEMP"] * policy.entropy())
                        target = jnp.expand_dims(target, axis=-1)
                        loss = 0.5 * jnp.mean((value_estimate - target) ** 2)
                        return loss

                    def actor_loss(a_params, q_params, transition):
                        q_values = q_net.apply(q_params, transition.state)
                        action_logits = actor_net.apply(a_params, transition.state)
                        policy = distrax.Categorical(logits=action_logits)
                        probs = jax.nn.softmax(action_logits, axis=-1)
                        expected_return = jnp.sum(probs * q_values, axis=-1)

                        entropy = policy.entropy()

                        actor_target = (config["ENT_TEMP"] * entropy - expected_return)
                        loss = actor_target.mean()
                        return loss, entropy, expected_return

                    def q_func_loss(q_params, target_params, transition):
                        q_values = q_net.apply(q_params, transition.state)
                        expected_next_state_value = critic_net.apply(target_params, transition.next_state)
                        chosen_q_values = q_values[jnp.arange(config["BATCH_SIZE"]), transition.action.astype(jnp.int32)]
                        target = transition.reward + config["GAMMA"] * (1 - transition.done) * expected_next_state_value
                        target = jax.lax.stop_gradient(target)
                        loss = 0.5 * jnp.mean((chosen_q_values - target) ** 2)
                        return loss

                    a_loss, entropy, expected_return = actor_loss(params["a"], jax.lax.stop_gradient(params["q"]), transition)

                    v_loss = value_loss(params["v"], jax.lax.stop_gradient(params["q"]), jax.lax.stop_gradient(params["a"]), target_params, transition)

                    q_loss = q_func_loss(params["q"], target_params, transition)

                    total_loss = a_loss + v_loss + q_loss

                    return total_loss, (a_loss, v_loss, q_loss, entropy, expected_return)

                train_state, target_params, rng = carry
                rng, _rng = jax.random.split(rng)

                transition_batch = rb.sample(_rng)

                (total_loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    train_state.params,
                    target_params,
                    transition_batch,
                )   
                
                target_params = jax.tree.map(
                    lambda target, online:
                    (1 - config["TAU"]) * target + config["TAU"] * online,
                    target_params,
                    train_state.params["v"],
                )

                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, target_params, rng), (total_loss, losses)

            rng, _rng = jax.random.split(rng)

            ready = rb.size >= max(config["WARMUP"], config["BATCH_SIZE"])

            def do_updates(carry):
                train_state, target_params, rng = carry

                (train_state, target_params, rng), (total_loss, losses) = jax.lax.scan(
                    update,
                    (train_state, target_params, rng),
                    xs=None,
                    length=config["NUM_UPDATE_STEPS"],
                )

                mean_loss = total_loss.mean()

                mean_losses = jax.tree.map(lambda x: x.mean(), losses)

                return train_state, target_params, rng, mean_loss, mean_losses

            def skip_updates(carry):
                train_state, target_params, rng = carry
                mean_loss = jnp.array(jnp.nan, dtype=jnp.float32)
                loss = jnp.array(jnp.nan, dtype=jnp.float32)
                losses = (loss, loss, loss, loss, loss)

                return train_state, target_params, rng, mean_loss, losses

            train_state, target_params, rng, mean_loss, losses = jax.lax.cond(
                ready,
                do_updates,
                skip_updates,
                operand=(next_train_state, target_params, _rng),
            )

            episode_mask = rollout_info["returned_episode"].astype(jnp.float32)
            num_completed_episodes = episode_mask.sum()

            denominator = jnp.maximum(num_completed_episodes, 1.0)

            episode_metric = jax.tree.map(
                lambda x: (
                                    x * episode_mask
                            ).sum() / denominator,
                rollout_info,
            )

            global_step = update_idx * config["NUM_ENVS"] * config["NUM_STEPS"]

            if config["DEBUG"] and config["USE_WANDB"]:
                def callback(
                        metric,
                        global_step,
                        losses,
                ):
                    to_log = create_log_dict(metric, config)

                    to_log.update({
                        "global_step": int(global_step),
                        "loss/actor_loss": float(losses[0]),
                        "loss/critic_loss": float(losses[1]),
                        "loss/q_loss": float(losses[2]),
                        "loss/entropy": float(losses[3]),
                        "loss/expected_return": float(losses[4]),
                    })

                    batch_log(
                        global_step,
                        to_log,
                        config,
                    )

                jax.debug.callback(
                    callback,
                    episode_metric,
                    global_step,
                    losses,
                )

            run_state = train_state, target_params, rb, next_obs, next_env_state, rng, update_idx+1
            return run_state, episode_metric

        run_state = train_state, target_params, rb, obs, env_state, rng, jnp.array(0, dtype=jnp.int32)
        run_state, _ = jax.lax.scan(train_loop, run_state, xs=None, length=config["NUM_UPDATES"])
        return run_state
    
    return train