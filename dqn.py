from __future__ import annotations
from typing import Sequence
import jax
import numpy as np
import jax.numpy as jnp
import optax
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn, struct
from flax.training.train_state import TrainState

from wrappers import LogWrapper, AutoResetEnvWrapper, OptimisticResetVecEnvWrapper, BatchEnvWrapper


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
    warmup: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        capacity: int,
        batch_size: int,
        warmup: int,
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
            warmup=warmup,
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
        return self.size >= self.batch_size and self.size >= self.warmup

class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        num_actions = 43
        x = nn.Dense(num_actions)(x)
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

    config["NUM_EPOCHS"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"]

    def train(rng):
        q_net = QNetwork()

        rng, q_key, t_key = jax.random.split(rng, 3)
        init = jnp.zeros((1, *env.observation_space(env_params).shape))
        q_params = q_net.init(q_key, init)

        rb = ReplayBuffer.create(
            capacity=config["BUFFER_CAPACITY"],
            batch_size=config["BATCH_SIZE"],
            warmup=config["WARMUP"],
            state_shape=env.observation_space(env_params).shape,
            action_shape=(),
            action_dtype=jnp.int32,
        )

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

        # change t.
        train_state = TrainState.create(
            apply_fn=q_net.apply,
            params=q_params,
            tx=tx,
        )

        target_params = train_state.params

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        def train_loop(run_state, _):
            train_state, target_params, rb, obs, env_state, rng = run_state
            def collect_transitions(carry, _):
                train_state, rb, obs, env_state, rng = carry
                action_logits = q_net.apply(train_state.params, obs)

                rng, eps_key, action_key = jax.random.split(rng, 3)

                random_values = jax.random.uniform(
                    eps_key,
                    shape=(config["NUM_ENVS"],),
                )

                random_actions = jax.random.randint(
                    action_key,
                    shape=(config["NUM_ENVS"],),
                    minval=0,
                    maxval=env.action_space(env_params).n,
                )

                greedy_actions = jnp.argmax(action_logits, axis=-1)

                action = jnp.where(
                    random_values < config["EPSILON"],
                    random_actions,
                    greedy_actions,
                )

                rng, _rng = jax.random.split(rng)
                next_obs, next_env_state, reward, done, info = env.step(
                    _rng,
                    env_state,
                    action,
                    env_params)

                transition = Transition(obs, action, reward, next_obs, done)
                rb = rb.add_batch(transition)

                rng, _rng = jax.random.split(rng)
                next_carry  = train_state, rb, next_obs, next_env_state, _rng
                return next_carry, _

            rng, _rng = jax.random.split(rng)
            initial_carry = train_state, rb, obs, env_state, _rng
            rollout_state, rollout = jax.lax.scan(collect_transitions,
                                                  initial_carry,
                                                  xs=None,
                                                  length=config["NUM_STEPS"])

            next_train_state, rb, next_obs, next_env_state, rng = rollout_state

            def update(carry, _):
                def loss_fn(params, target_params, transition):
                    q_values = q_net.apply(params,
                        transition.state
                    )

                    q_selected = q_values[
                        jnp.arange(q_values.shape[0]), transition.action,
                    ]

                    next_q_values = q_net.apply(target_params,
                        transition.next_state,
                    )

                    target = (
                            transition.reward
                            + config["GAMMA"]
                            * (1 - transition.done)
                            * jnp.max(next_q_values, axis=-1)
                    )

                    target = jax.lax.stop_gradient(target)

                    return jnp.mean((q_selected - target) ** 2)

                train_state, target_params, rng = carry
                rng, _rng = jax.random.split(rng)

                transition_batch = rb.sample(_rng)

                loss, grads = jax.value_and_grad(loss_fn)(
                    train_state.params,
                    target_params,
                    transition_batch,
                )

                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, target_params, rng), _

            rng, _rng = jax.random.split(rng)

            ready = rb.size >= max(config["WARMUP"], config["BATCH_SIZE"])

            def do_updates(carry):
                train_state, target_params, rng = carry

                (train_state, target_params, rng), losses = jax.lax.scan(
                    update,
                    (train_state, target_params, rng),
                    xs=None,
                    length=config["NUM_UPDATE_STEPS"],
                )

                return train_state, target_params, rng

            def skip_updates(carry):
                return carry

            train_state, target_params, rng = jax.lax.cond(
                ready,
                do_updates,
                skip_updates,
                operand=(train_state, target_params, _rng),
            )

            target_params = jax.tree.map(
                lambda target, online:
                (1 - config["TAU"]) * target + config["TAU"] * online,
                target_params,
                train_state.params,
            )
            run_state = train_state, target_params, rb, next_obs, next_env_state, rng
            return run_state, _


        run_state = train_state, target_params, rb, obs, env_state, rng
        run_state, _ = jax.lax.scan(train_loop, run_state, xs=None, length=config["NUM_EPOCHS"])
        return run_state
    return train