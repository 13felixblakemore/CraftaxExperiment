from __future__ import annotations

import jax
import optax
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn
import jax.numpy as jnp
from flax.training.train_state import TrainState

from wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper, OptimisticResetVecEnvWrapper

from typing import Sequence
from flax import struct


Array = jax.Array


@struct.dataclass
class Transition:
    state: Array
    goal: Array
    action: Array
    reward: Array
    next_state: Array
    discount: Array


@struct.dataclass
class ReplayBuffer:
    states: Array
    goals: Array
    actions: Array
    rewards: Array
    next_states: Array
    discounts: Array

    write_index: Array
    size: Array

    # Static metadata: these are not transformed as JAX arrays.

    capacity: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        capacity: int,
        batch_size: int,
        state_shape: Sequence[int],
        goal_shape: Sequence[int],
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
            goals=jnp.zeros(
                (capacity, *goal_shape),
                dtype=jnp.float32,
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
            discounts=jnp.zeros(
                (capacity,),
                dtype=jnp.float32,
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
            goals=self.goals.at[indices].set(batch.goal),
            actions=self.actions.at[indices].set(batch.action),
            rewards=self.rewards.at[indices].set(batch.reward),
            next_states=self.next_states.at[indices].set(
                batch.next_state
            ),
            discounts=self.discounts.at[indices].set(
                batch.discount
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
        rng: Array,
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
            goal=self.goals[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            next_state=self.next_states[indices],
            discount=self.discounts[indices],
        )

    def can_sample(self) -> Array:
        return self.size >= self.batch_size

class Critic(nn.Module):
    dim: int
    action_dim: int

    @nn.compact
    def call(self, s, g):
        x = jnp.concatenate([s, g], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        out = nn.Dense(self.action_dim)(x)
        return out

class Actor(nn.Module):
    num_levels: int
    action_dim: int
    dim: int
    goal_dim: int
    layer_num: int

    @nn.compact
    def __call__(self, s, g):
        x = jnp.concatenate([s, g], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)

        if self.layer_num == 0:
            action = nn.Dense(self.action_dim)(x)
            return action

        subgoal = nn.Dense(self.goal_dim)(x)
        return subgoal

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

    def start_training(rng):
        num_levels = config["NUM_LEVELS"]

        actors = tuple(
            Actor(
                config["NUM_LEVELS"],
                env.action_space(env_params).n,
                *env.observation_space(env_params).shape,
                config["GOAL_DIM"],
                level,
            )
            for level in range(num_levels)
        )

        critics = tuple(
            Critic(
                *env.observation_space(env_params).shape,
                env.action_space(env_params).n,
            )
            for _ in range(num_levels)
        )

        dummy_obs = jnp.zeros(
            (1, *env.observation_space(env_params).shape),
            dtype=jnp.float32,
        )

        # One retained key, one key per actor, one per critic.
        keys = jax.random.split(rng, 1 + 2 * num_levels)

        rng = keys[0]
        actor_keys = keys[1: 1 + num_levels]
        critic_keys = keys[1 + num_levels:]

        actor_states = []
        critic_states = []

        for level in range(num_levels):
            actor_variables = actors[level].init(
                actor_keys[level],
                dummy_obs,
            )

            critic_variables = critics[level].init(
                critic_keys[level],
                dummy_obs,
            )

            actor_states.append(
                TrainState.create(
                    apply_fn=actors[level].apply,
                    params=actor_variables["params"],
                    tx=optax.adam(config["ACTOR_LR"]),
                )
            )

            critic_states.append(
                TrainState.create(
                    apply_fn=critics[level].apply,
                    params=critic_variables["params"],
                    tx=optax.adam(config["CRITIC_LR"]),
                )
            )

        buffers = [ReplayBuffer.create(
            capacity=config["BUFFER_CAPACITY"],
            batch_size=config["BATCH_SIZE"],
            state_shape=env.observation_space(env_params).shape,
            goal_shape=(config["GOAL_DIM"],),
            action_shape=(config["GOAL_DIM"],),
            action_dtype=jnp.float32,
        ) for _ in range(num_levels)]

        buffers[0].actions = jnp.zeros(
                (config["BUFFER_CAPACITY"], *()),
                dtype=jnp.float32,
            )

        HER_buffers = [ReplayBuffer.create(
            capacity=config["BUFFER_CAPACITY"],
            batch_size=config["BATCH_SIZE"],
            state_shape=env.observation_space(env_params).shape,
            goal_shape=(config["GOAL_DIM"],),
            action_shape=(config["GOAL_DIM"],),
            action_dtype=jnp.float32,
        ) for _ in range(num_levels)]

        HER_buffers[0].actions = jnp.zeros(
                (config["BUFFER_CAPACITY"], *()),
                dtype=jnp.float32,
            )

        def episode(carry, transition):
            rng = carry
            # initialise starting state and goal
            rng, _rng = jax.random.split(rng)
            obs, env_state = env.reset(_rng, env_params)
            # init goal here
            def train_level(carry, transition):
                obs, env_state, g, level = carry

                attempts = 0

                while attempts < config["MAX_ATTEMPTS"]:
                    subgoal = actors[level].apply_fn(obs, g)

                    def subgoal_test(subgoal):
                        return None

                    subgoal_test(subgoal)

                    if level > 0:
                        next_carry = obs, env_state, subgoal, level - 1
                        reward = jnp.array(-config["MAX_ATTEMPTS"]) # shape may be weird
                        next_obs = train_level(next_carry, transition)
                    else:
                        next_obs, next_env_state, reward, done, info = env.step(
                            _rng,
                            env_state,
                            subgoal,
                            env_params,
                        )

                    def check_reached_subgoal(state, subgoal):
                        return True

                    reached = check_reached_subgoal(next_obs, subgoal)
                    tested = True

                    reward = jnp.where(level>0 and not reached, 0, reward)
                    discount = jnp.where


                    reward = jnp.where(reached, 0, -1)
                    discount = jnp.where(reached, 0, config["GAMMA"])

                    buffers[level].add(Transition(state=obs,
                                                action=subgoal,
                                                reward=reward,
                                                next_state=next_obs,
                                                goal=g,
                                                discount=discount))

                    TBD = None

                    HER_buffers[level].add(Transition(state=obs,
                                               action=subgoal,
                                               reward=TBD,
                                               next_state=next_obs,
                                               goal=TBD,
                                               discount=discount))
                    obs = next_obs
                    attempts += 1
                # change goal from tbd to next_obs, and reward to ?
                HER_buffers[level]...
                buffers[level] = Transition(HER)
                return next_obs
            return None

    return start_training