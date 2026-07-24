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
        level_fns = []

        low_level_action_dim = config["LOW_LEVEL_ACTION_DIM"]
        high_level_action_dim = config["HIGH_LEVEL_ACTION_DIM"]

        def make_level_fn(level, lower_level_fn=None):
            if level == 0:
                def run_level(
                    rng,
                    obs,
                    env_state,
                    goal,
                    params,
                    config,
                    is_subgoal_test,
                ):
                    # level 0 actor outputs primitive env action
                    action = actor_apply(
                        params=params["actors"][0],
                        obs=obs,
                        goal=goal,
                    )

                    rng, step_rng = jax.random.split(rng)

                    next_obs, next_env_state, env_reward, env_done, info = env_step(
                        step_rng,
                        env_state,
                        action,
                    )

                    reached = goal_reached(
                        next_obs,
                        goal,
                        threshold=config["GOAL_THRESHOLDS"][0],
                    )

                    reward = jnp.where(reached, 0.0, -1.0)
                    done = env_done | reached
                    discount = jnp.where(done, 0.0, config["GAMMA"])

                    transition = {
                        "obs": obs,
                        "action": action,
                        "reward": reward,
                        "next_obs": next_obs,
                        "goal": goal,
                        "done": done,
                        "discount": discount,
                    }

                    return rng, next_obs, next_env_state, transition

                return run_level

            else:
                def run_level(
                    rng,
                    obs,
                    env_state,
                    goal,
                    params,
                    config,
                    is_subgoal_test,
                ):
                    start_obs = obs
                    start_env_state = env_state

                    # high-level actor outputs subgoal
                    subgoal = actor_apply(
                        params=params["actors"][level],
                        obs=obs,
                        goal=goal,
                    )

                    rng, subgoal = add_hac_subgoal_exploration(
                        rng=rng,
                        action=subgoal,
                        is_subgoal_test=is_subgoal_test,
                        exploration_state_noise=config["EXPLORATION_STATE_NOISE"],
                        state_clip_low=config["STATE_CLIP_LOW"],
                        state_clip_high=config["STATE_CLIP_HIGH"],
                    )

                    def lower_attempt_step(carry, _):
                        rng, obs, env_state = carry

                        # The important HAC bit:
                        # the current level's action becomes the lower level's goal.
                        rng, next_obs, next_env_state, lower_transition = lower_level_fn(
                            rng,
                            obs,
                            env_state,
                            subgoal,
                            params,
                            config,
                            is_subgoal_test,
                        )

                        return (rng, next_obs, next_env_state), lower_transition

                    init_carry = rng, obs, env_state

                    final_carry, lower_transitions = jax.lax.scan(
                        lower_attempt_step,
                        init_carry,
                        xs=None,
                        length=config["H"],
                    )

                    rng, final_obs, final_env_state = final_carry

                    reached = goal_reached(
                        final_obs,
                        subgoal,
                        threshold=config["GOAL_THRESHOLDS"][level],
                    )

                    reward = jnp.where(reached, 0.0, -1.0)
                    done = reached
                    discount = jnp.where(done, 0.0, config["GAMMA"])

                    high_transition = {
                        "obs": start_obs,
                        "action": subgoal,
                        "reward": reward,
                        "next_obs": final_obs,
                        "goal": goal,
                        "done": done,
                        "discount": discount,
                        "is_subgoal_test": is_subgoal_test,
                    }

                    return rng, final_obs, final_env_state, {
                        "this_level": high_transition,
                        "lower": lower_transitions,
                    }

                return run_level

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

        train_state = (actor_states, critic_states)

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)

        goal = None

        level = num_levels - 1

        def train_loop(carry, _):
            obs, env_state, goal, rng = carry
            def run(carry, _):
                obs, env_state, goal, level, rng = carry

                rng, _rng = jax.random.split(rng)
                is_subgoal_test = (
                    jax.random.uniform(_rng, shape=(obs.shape[0],))
                    < test_freq
                )

                def run_all_levels(obs, goal, exploration_state_noise, state_clip_high, state_clip_low, is_subgoal_testing, rng):
                    actions = []
                    for level, actor_state in enumerate(actor_states):
                        rng, _rng = jax.random.split(rng)
                        action = actor_state.apply(
                                level=level,
                                obs=obs,
                                goal=goal,
                                rng=_rng,
                            )
                        actions.append(action)
                    
                    B = config["BATCH_SIZE"]

                    rng, prob_rng, noise_rng, random_rng = jax.random.split(rng, 4)

                    use_noisy_action = (
                        jax.random.uniform(prob_rng, shape=(B,)) > 0.2
                    )

                    noisy_actions = actions + (
                        jax.random.normal(noise_rng, shape=action.shape)
                        * exploration_state_noise
                    )

                    noisy_actions = jnp.clip(
                        noisy_actions,
                        state_clip_low,
                        state_clip_high,
                    )

                    random_actions = jax.random.uniform(
                        random_rng,
                        shape=actions.shape,
                        minval=state_clip_low,
                        maxval=state_clip_high,
                    )

                    exploratory_action = jnp.where(
                        use_noisy_action[:, None],
                        noisy_actions,
                        random_actions,
                    )

                    # Works if is_subgoal_test is scalar or shape (B,)
                    test_mask = jnp.broadcast_to(is_subgoal_test, (B,))

                    final_actions = jnp.where(
                        test_mask[:, None],
                        action,              # no noise during subgoal testing
                        exploratory_action,
                    )

                    return [actions]

                actions = run_all_levels(obs, goal, config["EXPLORATION_NOISE"], config["STATE_CLIP_HIGH"], config["STATE_CLIP_LOW"], is_subgoal_testing, rng)
                current_action = actions[level]

                def high_level(obs, current_action, goal, level):
                    goal = current_action
                    level -= 1
                    return obs, env_state, goal, level

                def low_level(obs, current_action, goal, level):
                    obs, env_state = env.step(current_action)
                    level = num_levels - 1
                    return obs, env_state, goal, level

                obs, env_state, goal, level = jax.lax.cond(level > 0, high_level, low_level, (obs, current_action, goal, level))

                rng, _rng = jax.random.split(rng)
                carry = obs, env_state, goal, level, _rng

                return carry, None

            def goal_reached(state, goal, threshold=0.05):
                distance = jnp.linalg.norm(state - goal, axis=-1)
                return distance < threshold
            
            carry = obs, env_state, goal, rng 
            a, b = jax.lax.scan(run, carry, xs = None, length=H**(num_levels-1))
            return train_state
        
        train_state = train_loop...
        

