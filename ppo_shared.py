# Used for reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
import json
import sys
import time
from functools import partial
from pathlib import Path
from pickletools import uint8
from typing import NamedTuple, Optional, Tuple, Union

import chex
import distrax
import imageio
import jax
import numpy as np
import optax
import jax.numpy as jnp
from flax.serialization import to_bytes
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper, LogEnvState

import wandb
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn
from flax.training.train_state import TrainState
import configs
from test_environment import ChainEnv
from typing import NamedTuple, Any

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class LogEnvState(NamedTuple):
    env_state: Any
    episode_return: jnp.ndarray
    episode_length: jnp.ndarray
    returned_episode_return: jnp.ndarray
    returned_episode_length: jnp.ndarray
    timestep: jnp.ndarray

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)

        state = LogEnvState(
            env_state=env_state,
            episode_return=jnp.array(0.0),
            episode_length=jnp.array(0),
            returned_episode_return=jnp.array(0.0),
            returned_episode_length=jnp.array(0),
            timestep=jnp.array(0),
        )

        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key,
            state.env_state,
            action,
            params,
        )

        episode_return = state.episode_return + reward
        episode_length = state.episode_length + 1
        timestep = state.timestep + 1

        info["returned_episode"] = done
        info["returned_episode_returns"] = episode_return
        info["returned_episode_lengths"] = episode_length
        info["timestep"] = timestep

        new_state = LogEnvState(
            env_state=env_state,
            episode_return=episode_return * (1.0 - done),
            episode_length=episode_length * (1 - done),
            returned_episode_return=episode_return,
            returned_episode_length=episode_length,
            timestep=timestep,
        )

        return obs, new_state, reward, done, info


class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256,
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=nn.initializers.constant(0.0),
                    )(x)
        x = nn.tanh(x)

        x = nn.Dense(256,
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=nn.initializers.constant(0.0),
                    )(x)
        x = nn.tanh(x)

        actor_logits = nn.Dense(self.action_dim,
                    kernel_init=nn.initializers.orthogonal(0.01),
                    bias_init=nn.initializers.constant(0.0),
                    )(x)

        critic = nn.Dense(1,
                     kernel_init=nn.initializers.orthogonal(1.0),
                     bias_init=nn.initializers.constant(0.0)
                     )(x)

        policy = distrax.Categorical(logits=actor_logits)

        return policy, jnp.squeeze(critic, axis=-1)

def make_train(config):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)

    batch_size = config["actors"] * config["num_steps"]
    num_updates = config["total_timesteps"] // batch_size
    config["minibatch_size"] = batch_size // config["num_minibatches"]
    num_optim_steps = num_updates * config["update_epochs"] * config["num_minibatches"]

    def train(rng):
        network = ActorCritic(env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        schedule = optax.linear_schedule(
            init_value=config["learning_rate"],
            end_value=config["end_learning_rate"],
            transition_steps=num_optim_steps
        )

        optimiser = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(schedule, eps=config["adam_epsilon"]),
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=optimiser,)

        rng, _rng = jax.random.split(rng)
        keys = jax.random.split(_rng, config["actors"])

        obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)

        def log_callback(log_data, global_step):
            log_data = {k: float(v) for k, v in log_data.items()}
            wandb.log(log_data, step=int(global_step))

        def update_step(run_state, _):
            def rollout_step(carry, _):
                train_state, obs, env_states, rng = carry

                # choose action for each env
                # calc log_probs of each action
                rng, _rng = jax.random.split(rng)
                policy, values = network.apply(train_state.params, obs)
                actions = policy.sample(seed = _rng)
                log_probs = policy.log_prob(actions)

                # env step
                rng, _rng = jax.random.split(rng)
                rngs = jax.random.split(_rng, config["actors"])
                next_obs, env_states, rewards, dones, infos = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rngs,
                    env_states,
                    actions,
                    env_params,
                )

                transition = Transition(dones, actions, values, rewards, log_probs, obs, infos)
                new_carry = train_state, next_obs, env_states, rng
                return new_carry, transition

            train_state, obs, env_states, rng, update_idx = run_state
            global_step = update_idx * config["actors"] * config["num_steps"]

            rollout_state = (train_state, obs, env_states, rng)

            rollout_state, rollout = jax.lax.scan(
                rollout_step,
                rollout_state,
                xs=None,
                length=config["num_steps"],
            )

            train_state, obs, env_states, rng = rollout_state

            def compute_gae(rollout, last_val):
                def gae_step(carry, transition):
                    last_gae, next_value = carry
                    reward, value, done = transition

                    next_non_terminal = 1.0 - done.astype(jnp.float32)

                    delta = reward + config["gamma"] * next_value * next_non_terminal - value
                    last_gae = (
                            delta
                            + config["gamma"]
                            * config["lambda"]
                            * next_non_terminal
                            * last_gae
                    )

                    return (last_gae, value), last_gae

                initial_carry = (
                    jnp.zeros_like(last_val),
                    last_val,
                )

                _, advantages = jax.lax.scan(
                    gae_step,
                    initial_carry,
                    (rollout.reward, rollout.value, rollout.done),
                    reverse=True,
                    unroll=16,
                )

                returns = advantages + rollout.value
                return advantages, returns

            _, last_val = network.apply(train_state.params, obs)

            advantages, returns = compute_gae(rollout, last_val)

            def update_epoch(update_state, _):
                def update_minibatch(carry, batch_info):
                    train_state = carry
                    rollout, advantages, returns = batch_info
                    def loss_fn(params, rollout, gae, returns):
                        policy, values = network.apply(params, rollout.obs)

                        new_log_probs = policy.log_prob(rollout.action)
                        old_log_probs = rollout.log_prob
                        log_ratio = new_log_probs - old_log_probs
                        ratio = jnp.exp(log_ratio)

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss1 = ratio * gae
                        loss2 = (
                                jnp.clip(ratio, 1 - config["epsilon"], 1 + config["epsilon"])
                                * gae
                        )
                        actor_loss = -jnp.mean(jnp.minimum(loss1, loss2))

                        entropy = policy.entropy().mean()

                        # clip this ?
                        critic_loss = jnp.mean((values - returns) ** 2)

                        total_loss = actor_loss + config["vf_coef"] * critic_loss - config["ent_coef"] * entropy

                        return total_loss, (actor_loss, critic_loss, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, rollout, advantages, returns
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, rollout, advantages, returns, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["minibatch_size"] * config["num_minibatches"]
                assert (
                        batch_size == config["num_steps"] * config["actors"]
                ), "batch size must be equal to number of steps * number of envs"

                permutation = jax.random.permutation(_rng, batch_size)
                batch = (rollout, advantages, returns)

                # flatten rollout batch into actors * steps for each item (actions, states, rewards, etc)
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
                        x, [config["num_minibatches"], -1] + list(x.shape[1:])
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
                config["update_epochs"]
            )

            train_state = update_state[0]
            returned = rollout.info["returned_episode"]
            num_returned = returned.sum()

            episode_return = jnp.where(
                num_returned > 0,
                (rollout.info["returned_episode_returns"] * returned).sum() / num_returned,
                jnp.nan,
            )

            episode_length = jnp.where(
                num_returned > 0,
                (rollout.info["returned_episode_lengths"] * returned).sum() / num_returned,
                jnp.nan,
            )
            log_data = {
                "episode_return": episode_return,
                "episode_length": episode_length,
            }

            jax.debug.callback(log_callback, log_data, global_step)
            if config["log"]:
                jax.lax.cond(
                    update_idx % config["log_every"] == 0,
                    lambda _: jax.debug.callback(log_callback, log_data, update_idx),
                    lambda _: None,
                    operand=None,
                )

            rng = update_state[-1]

            runner_state = (
                train_state,
                obs,
                env_states,
                rng,
                update_idx + 1,
            )

            return runner_state, None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, obs, env_states, _rng, jnp.array(0))
        num_updates = config["total_timesteps"] // (config["actors"]*config["num_steps"])
        runner_state, _ = jax.lax.scan(
            update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state}

    return train

if __name__ == "__main__":
    wandb.init(
        project="craftax",
        name="ppo-1",
        config=configs.large_run
    )

    config = wandb.config

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    final_train_state = out["runner_state"][0]
    params = final_train_state.params

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    params_path = save_dir / "final_params.msgpack"
    config_path = save_dir / "config.json"

    with open(params_path, "wb") as f:
        f.write(to_bytes(params))

    with open(config_path, "w") as f:
        json.dump(dict(wandb.config), f, indent=2)

    artifact = wandb.Artifact(
        name=f"{wandb.run.name}-final-model",
        type="model",
    )

    artifact.add_file(str(params_path))
    artifact.add_file(str(config_path))

    wandb.log_artifact(artifact)
