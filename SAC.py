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

@struct.dataclass
class Transition:
    state: jax.Array
    action: jax.Array
    reward: jax.Array
    next_state: jax.Array
    done: jax.Array

@struct.dataclass
class ReplayBuffer:
    states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_states: jax.Array
    dones: jax.Array

    write_index: jax.Array
    size: jax.Array

    # Static metadata: these are not transformed as JAX arrays.
    capacity: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        capacity: int,
        batch_size: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int] = (),
        action_dtype=jnp.float32,
        state_dtype=jnp.float32,
    ) -> ReplayBuffer:
        """Allocate an empty replay buffer."""

        return cls(
            states=jnp.zeros(
                (capacity, *state_shape),
                dtype=state_dtype,
            ),
            actions=jnp.zeros(
                (capacity, *action_shape),
                dtype=action_dtype,
            ),
            rewards=jnp.zeros(
                (capacity,),
                dtype=jnp.float32,
            ),
            next_states=jnp.zeros(
                (capacity, *state_shape),
                dtype=state_dtype,
            ),
            dones=jnp.zeros(
                (capacity,),
                dtype=jnp.bool_,
            ),
            write_index=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
            capacity=capacity,
            batch_size=batch_size,
        )

    def add_batch(self, batch: Transition) -> ReplayBuffer:
        """
        Add B transitions.

        batch.state has shape:
            (B, *state_shape)

        Assumes B <= capacity.
        """
        num_added = batch.state.shape[0]

        if num_added > self.capacity:
            raise ValueError(
                "The inserted batch cannot exceed replay capacity."
            )

        indices = (
            self.write_index
            + jnp.arange(num_added, dtype=jnp.int32)
        ) % self.capacity

        return self.replace(
            states=self.states.at[indices].set(batch.state),
            actions=self.actions.at[indices].set(batch.action),
            rewards=self.rewards.at[indices].set(batch.reward),
            next_states=self.next_states.at[indices].set(
                batch.next_state
            ),
            dones=self.dones.at[indices].set(
                batch.done
            ),
            write_index=(
                self.write_index + num_added
            ) % self.capacity,
            size=jnp.minimum(
                self.size + num_added,
                self.capacity,
            ),
        )

    def add(self, transition: Transition) -> ReplayBuffer:
        """Add a single transition."""

        batch = jax.tree.map(
            lambda x: jnp.expand_dims(x, axis=0),
            transition,
        )

        return self.add_batch(batch)

    def sample(
        self,
        rng: jax.Array,
    ) -> Transition:
        """
        Sample with replacement.

        Only call once size > 0, and preferably once
        size >= batch_size.
        """
        indices = jax.random.randint(
            rng,
            shape=(self.batch_size,),
            minval=0,
            maxval=self.size,
            dtype=jnp.int32,
        )

        return Transition(
            state=self.states[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            next_state=self.next_states[indices],
            done=self.dones[indices],
        )

    def can_sample(self) -> jax.Array:
        return self.size >= self.batch_size

class Critic(nn.Module):
    dim: int

    def __call__(self, x):
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)

        x = nn.Dense(1)(x)
        return x

class Q_Network(nn.Module):
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