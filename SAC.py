# Value network: estimates expected return under current policy
# Q network: estimates expected return from (s, a) pair
# Actor: chooses action based on state, s
# Replay Buffer
from __future__ import annotations
from typing import Sequence

import jax.numpy
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn, struct
import jax.numpy as jnp
from wrappers import LogWrapper, OptimisticResetVecEnvWrapper, AutoResetEnvWrapper, BatchEnvWrapper
from dqn import Transition, ReplayBuffer

class Q(nn.Module):
    dim: int
    def __call__(self, x):
        x = nn.Dense(self.dim)(x)

class Actor(nn.Module):
    dim: int
    action_dim: int

    def __call__(self, x):
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
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

    def train(rng):
        q_net = Q_Network(config["LAYER_SIZE"], env.action_space(env_params).n)

        rng, _rng = jax.random.split(rng, 2)
        init = jnp.zeros((1, env.observation_space(env_params).n))
        q_net.init(_rng, init)

        rb = ReplayBuffer.create(
            capacity=config["BUFFER_CAPACITY"],
            batch_size=config["BATCH_SIZE"],
            state_shape=env.observation_space(env_params).shape,
            action_shape=(env.action_space(env_params).n,),
            action_dtype=jnp.float32,
        )

        rng, _rng = jax.random.split(rng, 2)
        obs, env_state = env.reset(_rng, env_params)
        def rollout(obs, transition):
            action_logits = q_net.apply(obs)


            return next_obs, transition
        # rollout + save to replay buffer

        # update: sample batch of transitions from replay buffer
        # Calculate loss with entropy etc, update params
        #
        return None
    return train