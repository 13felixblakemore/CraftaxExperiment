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
from logz.batch_logging import create_log_dict, batch_log
from typing import NamedTuple, Any
import gymnax

from wrappers import LogWrapper, OptimisticResetVecEnvWrapper, AutoResetEnvWrapper, BatchEnvWrapper


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

class ActorCritic(nn.Module):
    action_dim: int
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim,
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=nn.initializers.constant(0.0),
                    )(x)
        x = nn.tanh(x)

        x = nn.Dense(self.dim,
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
    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = env.default_params
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
        network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
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
                next_obs, env_states, rewards, dones, infos = env.step(
                    _rng,
                    env_states,
                    actions,
                    env_params,
                )

                transition = Transition(dones, actions, values, rewards, log_probs, obs, infos)
                new_carry = train_state, next_obs, env_states, rng
                return new_carry, transition

            train_state, obs, env_states, rng, update_idx = run_state
            rollout_state = (train_state, obs, env_states, rng)

            rollout_state, rollout = jax.lax.scan(
                rollout_step,
                rollout_state,
                xs=None,
                length=config["NUM_STEPS"],
            )

            train_state, obs, env_states, rng = rollout_state

            def compute_gae(rollout, last_val):
                def gae_step(carry, transition):
                    last_gae, next_value = carry
                    reward, value, done = transition

                    next_non_terminal = 1.0 - done.astype(jnp.float32)

                    delta = reward + config["GAMMA"] * next_value * next_non_terminal - value
                    last_gae = (
                            delta
                            + config["GAMMA"]
                            * config["GAE_LAMBDA"]
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
                                jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"])
                                * gae
                        )
                        actor_loss = -jnp.mean(jnp.minimum(loss1, loss2))

                        entropy = policy.entropy().mean()

                        # clip this ?
                        critic_loss = jnp.mean((values - returns) ** 2)

                        total_loss = actor_loss + config["VF_COEF"] * critic_loss - config["ENT_COEF"] * entropy

                        return total_loss, (actor_loss, critic_loss, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
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


            global_step = update_idx * config["NUM_ENVS"] * config["NUM_STEPS"]

            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, global_step):
                    to_log = create_log_dict(metric, config)
                    to_log.update({
                        "global_step": global_step,
                    })
                    batch_log(global_step, to_log, config)
                """
                jax.debug.callback(
                    callback,
                    metric,
                    global_step,
                )"""

            runner_state = (
                train_state,
                obs,
                env_states,
                rng,
                update_idx + 1,
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, obs, env_states, _rng, jnp.array(0))
        runner_state, metric = jax.lax.scan(
            update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train