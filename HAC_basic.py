from __future__ import annotations

import jax
import optax
import gymnax
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn
import jax.numpy as jnp
from flax.training.train_state import TrainState

from wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper, OptimisticResetVecEnvWrapper

from typing import Sequence
from flax import struct


Array = jax.Array

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

    @nn.compact
    def __call__(self, s, g):
        x = jnp.concatenate([s, g], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)

        action = nn.Dense(1)(x)
        return action

from typing import NamedTuple
import jax
import jax.numpy as jnp

@struct.dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    goal: jnp.ndarray
    discount: jnp.ndarray
    done: jnp.ndarray
    valid: jnp.ndarray


def expand_mask(mask, x):
    # mask: [B]
    # x:    [B, ...]
    return mask.reshape((mask.shape[0],) + (1,) * (x.ndim - 1))


def batch_where(mask, x_true, x_false):
    return jnp.where(expand_mask(mask, x_true), x_true, x_false)


def tree_where(mask, true_tree, false_tree):
    return jax.tree.map(
        lambda true, false: jnp.where(
            expand_mask(mask, true),
            true,
            false,
        ),
        true_tree,
        false_tree,
    )
    return jax.tree.map(
        lambda x_true, x_false: jnp.where(
            expand_mask(mask, x_true),
            x_true,
            x_false,
        ),
        true_tree,
        false_tree,
    )

def make_train(config):
    env, env_params = gymnax.make("PointRobot-misc")
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
        actor = Actor(
                config["NUM_LEVELS"],
                env.action_space(env_params).n,
                *env.observation_space(env_params).shape,
                config["GOAL_DIM"],
            )
        critic = Critic(
                *env.observation_space(env_params).shape,
                env.action_space(env_params).n,
            )

        dummy_obs = jnp.zeros(
            (1, *env.observation_space(env_params).shape),
            dtype=jnp.float32,
        )

        rng, actor_key, critic_key = jax.random.split(rng, 3)

        actor_variables = actor.init(
            actor_key,
            dummy_obs,
        )

        critic_variables = critic.init(
            critic_key,
            dummy_obs,
        )

        actor_state = TrainState.create(
                apply_fn=actor.apply,
                params=actor_variables["params"],
                tx=optax.adam(config["ACTOR_LR"]),
            )

        critic_state = TrainState.create(
                apply_fn=critic.apply,
                params=critic_variables["params"],
                tx=optax.adam(config["CRITIC_LR"]),
            )

        buffer = ReplayBuffer.create(
            capacity=config["BUFFER_CAPACITY"],
            batch_size=config["BATCH_SIZE"],
            state_shape=env.observation_space(env_params).shape,
            goal_shape=(config["GOAL_DIM"],),
            action_shape=(config["GOAL_DIM"],),
            action_dtype=jnp.float32,
        )

        HER_buffer = ReplayBuffer.create(
            capacity=config["BUFFER_CAPACITY"],
            batch_size=config["BATCH_SIZE"],
            state_shape=env.observation_space(env_params).shape,
            goal_shape=(config["GOAL_DIM"],),
            action_shape=(config["GOAL_DIM"],),
            action_dtype=jnp.float32,
        )

        train_state = (actor_state, critic_state)

        target_params = critic_state.params

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)

        already_done = jnp.zeros((config["NUM_ENVS"],), dtype=bool)

        def collect_transitions(carry, _):
            actor_state, obs, goal, env_state, already_done, testing, rng = carry

            rng, action_key, noise_key, step_key = jax.random.split(rng, 4)

            active = ~already_done  # [NUM_ENVS]

            # Deterministic DDPG actor.
            # Expected shape: [NUM_ENVS, ACTION_DIM]
            action = actor_state.apply(actor_state.params, obs, goal)

            # Exploration noise for continuous actions.
            noise = (
                jax.random.normal(noise_key, shape=action.shape)
                * config.get("ACTION_NOISE_STD", 0.1)
            )

            noisy_action = action + noise

            # testing=True means no exploration noise.
            action = jnp.where(
                expand_mask(testing, action),
                action,
                noisy_action,
            )

            # Clip to env action range.
            action = jnp.clip(
                action,
                config.get("ACTION_LOW", -1.0),
                config.get("ACTION_HIGH", 1.0),
            )

            # Inactive envs should not move.
            action = batch_where(
                active,
                action,
                jnp.zeros_like(action),
            )

            next_obs, next_env_state, env_reward, env_done, info = env.step(
                step_key,
                env_state,
                action,
                env_params,
            )

            # You define this.
            # Should return shape [NUM_ENVS].
            achieved = subgoal_reached(next_obs, goal)

            # HAC-style sparse goal reward.
            reward = jnp.where(achieved, 0.0, -1.0)

            done = achieved | env_done

            discount = jnp.where(done, 0.0, config["GAMMA"])

            valid = active

            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                goal=goal,
                discount=discount,
                done=done,
                valid=valid,
            )

            new_already_done = already_done | done

            # Freeze envs that had already reached the subgoal.
            next_obs = batch_where(active, next_obs, obs)
            next_env_state = tree_where(active, next_env_state, env_state)

            next_carry = (
                actor_state,
                next_obs,
                goal,
                next_env_state,
                new_already_done,
                testing,
                rng,
            )

            return next_carry, (transition, info)
        
        carry = (
            actor_state,
            obs,
            goal,
            env_state,
            already_done,
            testing,  # sampled by higher level, shape [NUM_ENVS]
            rng,
        )

        carry, (transitions, infos) = jax.lax.scan(
            collect_transitions,
            carry,
            xs=None,
            length=config["SUBGOAL_HORIZON"],
        )

        actor_state, final_obs, goal, final_env_state, already_done, testing, rng = carry

        def ddpg_update(
            actor_state,
            critic_state,
            target_actor_params,
            target_critic_params,
            batch,
            config,
        ):
            gamma = config["GAMMA"]
            tau = config.get("TAU", 0.005)

            valid = batch.valid.astype(jnp.float32)
            valid_sum = jnp.maximum(valid.sum(), 1.0)

            def critic_loss_fn(critic_params):
                next_action = actor_state.apply(
                    target_actor_params,
                    batch.next_obs,
                    batch.goal,
                )

                next_action = jnp.clip(
                    next_action,
                    config.get("ACTION_LOW", -1.0),
                    config.get("ACTION_HIGH", 1.0),
                )

                target_q = critic_state.apply(
                    target_critic_params,
                    batch.next_obs,
                    batch.goal,
                    next_action,
                )

                target_q = jax.lax.stop_gradient(target_q)

                target = batch.reward + batch.discount * target_q

                q = critic_state.apply(
                    critic_params,
                    batch.obs,
                    batch.goal,
                    batch.action,
                )

                td_error = q - target

                loss = (valid * td_error ** 2).sum() / valid_sum

                return loss, {
                    "critic_loss": loss,
                    "q_mean": (valid * q).sum() / valid_sum,
                    "target_q_mean": (valid * target_q).sum() / valid_sum,
                }

            (critic_loss, critic_info), critic_grads = jax.value_and_grad(
                critic_loss_fn,
                has_aux=True,
            )(critic_state.params)

            critic_state = critic_state.apply_gradients(grads=critic_grads)

            def actor_loss_fn(actor_params):
                action = actor_state.apply(
                    actor_params,
                    batch.obs,
                    batch.goal,
                )

                action = jnp.clip(
                    action,
                    config.get("ACTION_LOW", -1.0),
                    config.get("ACTION_HIGH", 1.0),
                )

                q = critic_state.apply(
                    critic_state.params,
                    batch.obs,
                    batch.goal,
                    action,
                )

                # DDPG actor maximizes Q, so minimize -Q.
                loss = -(valid * q).sum() / valid_sum

                return loss, {
                    "actor_loss": loss,
                    "actor_q_mean": (valid * q).sum() / valid_sum,
                }

            (actor_loss, actor_info), actor_grads = jax.value_and_grad(
                actor_loss_fn,
                has_aux=True,
            )(actor_state.params)

            actor_state = actor_state.apply_gradients(grads=actor_grads)

            target_actor_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_actor_params,
                actor_state.params,
            )

            target_critic_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_critic_params,
                critic_state.params,
            )

            info = {
                **critic_info,
                **actor_info,
            }

            return (
                actor_state,
                critic_state,
                target_actor_params,
                target_critic_params,
                info,
            )

        return run_state

    return start_training