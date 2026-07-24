from flax import linen as nn
import jax.numpy as jnp
import gymnax
from wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper, OptimisticResetVecEnvWrapper
import jax
import optax
from flax.training.train_state import TrainState
import flashbax as fbx
from functools import partial


def make_example_transition(state_dim: int, action_dim: int):
    return {
        "obs": jnp.zeros((state_dim,), dtype=jnp.float32),
        "action": jnp.zeros((action_dim,), dtype=jnp.float32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "next_obs": jnp.zeros((state_dim,), dtype=jnp.float32),
        "goal": jnp.zeros((state_dim,), dtype=jnp.float32),
        "achieved_goal": jnp.zeros((state_dim,), dtype=jnp.float32),
        "done": jnp.zeros((), dtype=jnp.bool_),
        "discount": jnp.zeros((), dtype=jnp.float32),
    }

def make_level_buffer(
    state_dim: int,
    action_dim: int,
    num_envs: int,
    buffer_time_length: int = 100_000,
    min_time_length: int = 1_000,
    sample_batch_size: int = 256,
    sample_sequence_length: int = 8,
):
    buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=buffer_time_length,
        min_length_time_axis=min_time_length,
        sample_batch_size=sample_batch_size,
        add_batch_size=num_envs,
        sample_sequence_length=sample_sequence_length,
        period=1,
    )

    example_transition = make_example_transition(state_dim, action_dim)
    buffer_state = buffer.init(example_transition)

    return buffer, buffer_state

class Actor(nn.Module):
    state_dim: int
    action_dim: int
    action_bounds: float
    offset: float

    @nn.compact
    def __call__(self, state, goal):
        # actor
        x = jnp.concatenate([state, goal], axis=-1)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)

        x = x * self.action_bounds + self.offset
        return x
        
class Critic(nn.Module):
    state_dim: int
    action_dim: int
    H: float

    def __call__(self, state, action, goal):
        x = jnp.concatenate([state, action, goal], axis=-1)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        x = - x * self.H
        return x
    
class Transition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array
    goal: jax.Array


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

    def train(rng):
        num_levels = config["NUM_LEVELS"]
        H = config["MAX_HORIZON"]
        test_freq = config["TEST_FREQ"]

        state_dim = env.observation_space(env_params).shape

        actors = tuple(
            Actor(
                config["NUM_LEVELS"],
                env.action_space(env_params).n,
                *state_dim,
                config["GOAL_DIM"],
                level,
            )
            for level in range(num_levels)
        )

        critics = tuple(
            Critic(
                *state_dim,
                env.action_space(env_params).n,
            )
            for _ in range(num_levels)
        )

        dummy_obs = jnp.zeros(
            (1, *state_dim),
            dtype=jnp.float32,
        )

        keys = jax.random.split(rng, 1 + 2 * num_levels)

        rng = keys[0]
        actor_keys = keys[1: 1 + num_levels]
        critic_keys = keys[1 + num_levels:]

        actor_states = []
        critic_states = []
        buffers = []
        buffer_states = []

        low_level_action_dim = config["LOW_LEVEL_ACTION_DIM"]
        high_level_action_dim = config["HIGH_LEVEL_ACTION_DIM"]

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

            action_dim = low_level_action_dim if level == 0 else high_level_action_dim

            buffer, buffer_state = make_level_buffer(
                state_dim=state_dim,
                action_dim=action_dim,
                num_envs=config["NUM_ENVS"],
                buffer_time_length=config["BUFFER_CAPACITY"],
                min_time_length=["WARMUP"],
                sample_batch_size=["BATCH_SIZE"],
                sample_sequence_length=8,
            )

            buffers.append(buffer)
            buffer_states.append(buffer_state)

            level_fns.append(make_level_fn(level))

        def primitive_rollout(obs, env_state, goal, rng, train_state, buffer, buffer_state, lower_function=None):
            def step(carry, _):
                obs, env_state, rng = carry

                rng, env_key, action_key = jax.random.split(rng)
                action = actor_state.apply(
                        level=level,
                        obs=obs,
                        goal=goal,
                        rng=action_key,
                    )
                
                next_obs, next_env_state, reward, done, info = env.step(
                    env_key,
                    env_state,
                    action,
                    env_params,
                )

                transition = Transition(obs, action, reward, next_obs, done, goal)

                carry = next_obs, next_env_state, rng

                return carry, transition

            actor_state, critic_state = train_state

            init = obs, env_state, rng
            carry, transitions = jax.lax.scan(step, init, xs=None, length=H)

            buffer_state = buffer.add(
                buffer_state,
                transitions,
            )

            def update():
                pass

            def skip_update():
                pass

            actor_state, critic_state = jax.lax.cond(num_steps % config["UPDATE_FREQUENCY"] == 0, update, skip_update)

            return carry, transitions

        def outer_rollout(obs, env_state, goal, rng, actor_state, lower_function):
            def step(carry, _):
                obs, env_state, rng = carry
                rng, action_key = jax.random.split(rng)
                subgoal = actor_state.apply(
                    obs=obs,
                    goal=goal,
                    rng = action_key
                )
                carry, transitions = lower_function(obs, env_state, subgoal, rng, actor_state)
                return carry, transitions
            
            init = obs, env_state, rng
            carry, transitions = jax.lax.scan(step, init, xs=None, length=H)
            return carry, transitions

        func = primitive_rollout(actor_state=actor_states[0])
        for level in range(1, num_levels):
            func = outer_rollout(actor_state=actor_states[level], lower_function=func)


        train_state = (actor_states, critic_states)

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        goal = None

        iterations = config["TOTAL_TIMESTEPS"] // H**(num_levels - 1)

        init = obs, env_state, goal, rng
        carry, transitions = jax.lax.scan(func, init, xs=None, length=iterations)

        return train_state
        
    return train
