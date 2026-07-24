from __future__ import annotations

from functools import partial
from typing import Sequence

import gymnax
import distrax
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from craftax.craftax_env import make_craftax_env_from_name
from logz.batch_logging import batch_log, create_log_dict
from wrappers import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper, OptimisticResetVecEnvWrapper


@struct.dataclass
class HACTransition:
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.ndarray
    goal: jnp.ndarray
    done: jnp.ndarray


@struct.dataclass
class HACReplayBuffer:
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_states: jnp.ndarray
    goals: jnp.ndarray
    dones: jnp.ndarray
    write_index: jnp.ndarray
    size: jnp.ndarray
    capacity: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        *,
        capacity: int,
        batch_size: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int] | tuple | int,
        goal_shape: Sequence[int] | tuple | int,
        action_dtype=jnp.float32,
    ):
        if isinstance(action_shape, int):
            action_shape = (action_shape,)
        if isinstance(goal_shape, int):
            goal_shape = (goal_shape,)

        return cls(
            states=jnp.zeros((capacity, *state_shape), dtype=jnp.float32),
            actions=jnp.zeros((capacity, *action_shape), dtype=action_dtype),
            rewards=jnp.zeros((capacity,), dtype=jnp.float32),
            next_states=jnp.zeros((capacity, *state_shape), dtype=jnp.float32),
            goals=jnp.zeros((capacity, *goal_shape), dtype=jnp.float32),
            dones=jnp.zeros((capacity,), dtype=jnp.float32),
            write_index=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
            capacity=capacity,
            batch_size=batch_size,
        )

    def add_batch(self, transition: HACTransition):
        batch_n = transition.reward.shape[0]
        idx = (jnp.arange(batch_n, dtype=jnp.int32) + self.write_index) % self.capacity

        return self.replace(
            states=self.states.at[idx].set(transition.state.astype(jnp.float32)),
            actions=self.actions.at[idx].set(transition.action.astype(self.actions.dtype)),
            rewards=self.rewards.at[idx].set(transition.reward.astype(jnp.float32)),
            next_states=self.next_states.at[idx].set(transition.next_state.astype(jnp.float32)),
            goals=self.goals.at[idx].set(transition.goal.astype(jnp.float32)),
            dones=self.dones.at[idx].set(transition.done.astype(jnp.float32)),
            write_index=(self.write_index + batch_n) % self.capacity,
            size=jnp.minimum(self.size + batch_n, self.capacity),
        )

    def sample(self, rng) -> HACTransition:
        idx = jax.random.randint(rng, (self.batch_size,), minval=0, maxval=self.size)
        return HACTransition(
            state=self.states[idx],
            action=self.actions[idx],
            reward=self.rewards[idx],
            next_state=self.next_states[idx],
            goal=self.goals[idx],
            done=self.dones[idx],
        )


@struct.dataclass
class CollectCarry:
    actor_states: tuple
    rbs: tuple
    obs: jnp.ndarray
    env_state: object
    rng: jnp.ndarray


class DiscreteGoalActor(nn.Module):
    dim: int
    action_dim: int

    @nn.compact
    def __call__(self, state, goal):
        x = jnp.concatenate([state.reshape((state.shape[0], -1)), goal], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class SubgoalActor(nn.Module):
    dim: int
    goal_dim: int
    subgoal_scale: float

    @nn.compact
    def __call__(self, state, goal):
        x = jnp.concatenate([state.reshape((state.shape[0], -1)), goal], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        return jnp.tanh(nn.Dense(self.goal_dim)(x)) * self.subgoal_scale


class DiscreteGoalCritic(nn.Module):
    dim: int
    action_dim: int
    q_bound: float
    q_init: float = -0.067

    @nn.compact
    def __call__(self, state, goal):
        x = jnp.concatenate([state.reshape((state.shape[0], -1)), goal], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        q_offset = -jnp.log(self.q_bound / abs(self.q_init) - 1.0)
        raw_q = nn.Dense(
            self.action_dim,
            bias_init=nn.initializers.constant(q_offset),
        )(x)
        # HAC uses sparse non-positive rewards, so bound Q in [-H, 0].
        return -self.q_bound * jax.nn.sigmoid(raw_q)


class ContinuousGoalCritic(nn.Module):
    dim: int
    q_bound: float
    q_init: float = -0.067

    @nn.compact
    def __call__(self, state, goal, action):
        x = jnp.concatenate([state.reshape((state.shape[0], -1)), goal, action], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        q_offset = -jnp.log(self.q_bound / abs(self.q_init) - 1.0)
        raw_q = nn.Dense(1, bias_init=nn.initializers.constant(q_offset))(x)
        return (-self.q_bound * jax.nn.sigmoid(raw_q)).squeeze(-1)


def _replace_tuple(xs: tuple, index: int, value):
    return xs[:index] + (value,) + xs[index + 1 :]


def _as_batch_bool(x, batch_size: int):
    x = jnp.asarray(x)
    if x.ndim == 0:
        return jnp.full((batch_size,), x, dtype=jnp.bool_)
    return x.astype(jnp.bool_)


def _tree_where(mask, new_tree, old_tree):
    def select(new, old):
        m = mask
        while m.ndim < new.ndim:
            m = m[..., None]
        return jnp.where(m, new, old)

    return jax.tree.map(select, new_tree, old_tree)


def make_train(config):
    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = env.default_params#
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

    obs_shape = env.observation_space(env_params).shape
    flat_obs_dim = int(jnp.prod(jnp.asarray(obs_shape)))
    goal_indices_cfg = config.get("GOAL_INDICES", None)
    if goal_indices_cfg is None:
        goal_indices = tuple(range(flat_obs_dim))
    else:
        goal_indices = tuple(goal_indices_cfg)
    goal_dim = len(goal_indices)
    goal_indices_jnp = jnp.asarray(goal_indices, dtype=jnp.int32)
    end_goal_cfg = config.get("END_GOAL", None)
    if end_goal_cfg is None:
        end_goal = jnp.zeros((goal_dim,), dtype=jnp.float32)
    else:
        end_goal = jnp.asarray(end_goal_cfg, dtype=jnp.float32)
        if end_goal.shape != (goal_dim,):
            raise ValueError(
                f"END_GOAL must have shape ({goal_dim},), matching len(GOAL_INDICES). "
                f"Got {end_goal.shape}."
            )

    num_levels = int(config.get("NUM_LEVELS", 2))
    time_scale = int(config.get("TIME_SCALE", config.get("HORIZON", 10)))
    atomic_steps_per_top_step = int(time_scale ** max(num_levels - 1, 0))
    q_bound = float(config.get("Q_BOUND", time_scale))

    config["ATOMIC_STEPS_PER_TOP_STEP"] = atomic_steps_per_top_step
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (
        config["NUM_STEPS"] * config["NUM_ENVS"] * atomic_steps_per_top_step
    )

    def project_goal(obs):
        flat = obs.reshape((obs.shape[0], -1))
        return flat[:, goal_indices_jnp]

    def goal_achieved(obs, goal, threshold):
        projected = project_goal(obs)
        return jnp.max(jnp.abs(projected - goal), axis=-1) <= threshold

    def make_top_goal(obs):
        return jnp.broadcast_to(end_goal, (obs.shape[0], goal_dim))

    def train(rng):
        action_dim = env.action_space(env_params).n
        actor_nets = []
        q1_nets = []
        q2_nets = []
        for level in range(num_levels):
            if level == 0:
                actor_nets.append(DiscreteGoalActor(config["LAYER_SIZE"], action_dim))
                q1_nets.append(DiscreteGoalCritic(config["LAYER_SIZE"], action_dim, q_bound))
                q2_nets.append(DiscreteGoalCritic(config["LAYER_SIZE"], action_dim, q_bound))
            else:
                actor_nets.append(
                    SubgoalActor(
                        config["LAYER_SIZE"],
                        goal_dim,
                        float(config.get("SUBGOAL_SCALE", 1.0)),
                    )
                )
                q1_nets.append(ContinuousGoalCritic(config["LAYER_SIZE"], q_bound))
                q2_nets.append(ContinuousGoalCritic(config["LAYER_SIZE"], q_bound))

        init_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        init_goal = jnp.zeros((1, goal_dim), dtype=jnp.float32)
        init_subgoal_action = jnp.zeros((1, goal_dim), dtype=jnp.float32)

        def linear_schedule(count):
            frac = 1.0 - (count / max(config["NUM_UPDATES"] * config["NUM_UPDATE_STEPS"], 1))
            frac = jnp.clip(frac, 0.0, 1.0)
            return config["LR"] * frac

        if config["ANNEAL_LR"]:
            tx_actor = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            tx_critic = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx_actor = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            tx_critic = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"] * float(config.get("CRITIC_LR_SCALE", 0.5)), eps=1e-5),
            )

        actor_states = []
        q1_states = []
        q2_states = []

        for level in range(num_levels):
            rng, actor_key, q1_key, q2_key = jax.random.split(rng, 4)
            actor_params = actor_nets[level].init(actor_key, init_obs, init_goal)
            if level == 0:
                q1_params = q1_nets[level].init(q1_key, init_obs, init_goal)
                q2_params = q2_nets[level].init(q2_key, init_obs, init_goal)
            else:
                q1_params = q1_nets[level].init(q1_key, init_obs, init_goal, init_subgoal_action)
                q2_params = q2_nets[level].init(q2_key, init_obs, init_goal, init_subgoal_action)

            actor_states.append(
                TrainState.create(
                    apply_fn=actor_nets[level].apply,
                    params=actor_params,
                    tx=tx_actor,
                )
            )
            q1_states.append(
                TrainState.create(
                    apply_fn=q1_nets[level].apply,
                    params=q1_params,
                    tx=tx_critic,
                )
            )
            q2_states.append(
                TrainState.create(
                    apply_fn=q2_nets[level].apply,
                    params=q2_params,
                    tx=tx_critic,
                )
            )

        actor_states = tuple(actor_states)
        q1_states = tuple(q1_states)
        q2_states = tuple(q2_states)
        train_state = (actor_states, q1_states, q2_states)

        target_actor_params = tuple(s.params for s in actor_states)
        target_q1_params = tuple(s.params for s in q1_states)
        target_q2_params = tuple(s.params for s in q2_states)

        rbs = []
        for level in range(num_levels):
            if level == 0:
                rbs.append(
                    HACReplayBuffer.create(
                        capacity=config["BUFFER_CAPACITY"],
                        batch_size=config["BATCH_SIZE"],
                        state_shape=obs_shape,
                        action_shape=(),
                        action_dtype=jnp.int32,
                        goal_shape=(goal_dim,),
                    )
                )
            else:
                rbs.append(
                    HACReplayBuffer.create(
                        capacity=config["BUFFER_CAPACITY"],
                        batch_size=config["BATCH_SIZE"],
                        state_shape=obs_shape,
                        action_shape=(goal_dim,),
                        action_dtype=jnp.float32,
                        goal_shape=(goal_dim,),
                    )
                )
        rbs = tuple(rbs)

        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)

        def sample_level_action(level: int, actor_state, obs, goal, rng, subgoal_test):
            batch_size = obs.shape[0]
            subgoal_test = _as_batch_bool(subgoal_test, batch_size)

            if level == 0:
                rng, policy_key, random_key, explore_key = jax.random.split(rng, 4)
                logits = actor_nets[level].apply(actor_state.params, obs, goal)
                policy = distrax.Categorical(logits=logits)
                sampled_action = policy.sample(seed=policy_key)
                greedy_action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
                random_action = jax.random.randint(
                    random_key,
                    shape=(batch_size,),
                    minval=0,
                    maxval=action_dim,
                    dtype=jnp.int32,
                )
                random_mask = (
                    jax.random.uniform(explore_key, (batch_size,))
                    < float(config.get("ATOMIC_RANDOM_PROB", 0.2))
                )
                exploratory_action = jnp.where(random_mask, random_action, sampled_action)
                action = jnp.where(subgoal_test, greedy_action, exploratory_action)
                return action.astype(jnp.int32), rng

            rng, noise_key, random_key, explore_key = jax.random.split(rng, 4)
            clean_subgoal = actor_nets[level].apply(actor_state.params, obs, goal)
            noise = (
                jax.random.normal(noise_key, clean_subgoal.shape)
                * float(config.get("SUBGOAL_NOISE", 0.1))
            )
            noisy_subgoal = clean_subgoal + noise
            random_subgoal = jax.random.uniform(
                random_key,
                clean_subgoal.shape,
                minval=-float(config.get("SUBGOAL_SCALE", 1.0)),
                maxval=float(config.get("SUBGOAL_SCALE", 1.0)),
            )
            random_mask = (
                jax.random.uniform(explore_key, (batch_size,))
                < float(config.get("SUBGOAL_RANDOM_PROB", 0.2))
            )
            while random_mask.ndim < noisy_subgoal.ndim:
                random_mask = random_mask[..., None]
            exploratory_subgoal = jnp.where(random_mask, random_subgoal, noisy_subgoal)
            exploratory_subgoal = jnp.clip(
                exploratory_subgoal,
                -float(config.get("SUBGOAL_SCALE", 1.0)),
                float(config.get("SUBGOAL_SCALE", 1.0)),
            )
            action = jnp.where(subgoal_test[:, None], clean_subgoal, exploratory_subgoal)
            return action.astype(jnp.float32), rng

        def add_transition_to_level(carry: CollectCarry, level: int, transition: HACTransition):
            rb = carry.rbs[level].add_batch(transition)
            return carry.replace(rbs=_replace_tuple(carry.rbs, level, rb))

        def run_level(level: int, carry: CollectCarry, goal, subgoal_test):
            old_obs = carry.obs
            batch_size = old_obs.shape[0]
            subgoal_test = _as_batch_bool(subgoal_test, batch_size)

            action, rng = sample_level_action(
                level,
                carry.actor_states[level],
                old_obs,
                goal,
                carry.rng,
                subgoal_test,
            )
            carry = carry.replace(rng=rng)

            if level == 0:
                rng, env_key = jax.random.split(carry.rng)
                next_obs, next_env_state, env_reward, env_done, info = env.step(
                    env_key,
                    carry.env_state,
                    action,
                    env_params,
                )
                carry = carry.replace(obs=next_obs, env_state=next_env_state, rng=rng)

                achieved = goal_achieved(
                    next_obs,
                    goal,
                    float(config.get("SUBGOAL_THRESHOLD", config.get("GOAL_THRESHOLD", 0.05))),
                )
                reward = jnp.where(achieved, 0.0, -1.0)
                done = jnp.logical_or(achieved, env_done).astype(jnp.float32)

                transition = HACTransition(
                    state=old_obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    goal=goal,
                    done=done,
                )
                carry = add_transition_to_level(carry, level, transition)

                if bool(config.get("HINDSIGHT_GOAL_REPLAY", True)):
                    hindsight_goal = project_goal(next_obs)
                    hindsight_transition = HACTransition(
                        state=old_obs,
                        action=action,
                        reward=jnp.zeros_like(reward),
                        next_state=next_obs,
                        goal=hindsight_goal,
                        done=jnp.ones_like(done),
                    )
                    carry = add_transition_to_level(carry, level, hindsight_transition)

                return carry, info

            rng, test_key = jax.random.split(carry.rng)
            lower_test_random = (
                jax.random.uniform(test_key, (batch_size,))
                < float(config.get("SUBGOAL_TEST_PROB", 0.3))
            )
            # If this level itself is being tested, its lower level should also act greedily.
            lower_subgoal_test = jnp.logical_or(subgoal_test, lower_test_random)
            carry = carry.replace(rng=rng)

            def lower_attempt(lower_carry, _):
                return run_level(level - 1, lower_carry, action, lower_subgoal_test)

            carry, lower_info = jax.lax.scan(
                lower_attempt,
                carry,
                xs=None,
                length=time_scale,
            )

            next_obs = carry.obs
            achieved_lower = goal_achieved(
                next_obs,
                action,
                float(config.get("SUBGOAL_THRESHOLD", config.get("GOAL_THRESHOLD", 0.05))),
            )
            achieved_current = goal_achieved(
                next_obs,
                goal,
                float(config.get("GOAL_THRESHOLD", 0.05)),
            )

            achieved_projection = project_goal(next_obs)
            hindsight_action = jnp.where(achieved_lower[:, None], action, achieved_projection)
            reward = jnp.where(achieved_current, 0.0, -1.0)
            done = achieved_current.astype(jnp.float32)

            action_replay_transition = HACTransition(
                state=old_obs,
                action=hindsight_action,
                reward=reward,
                next_state=next_obs,
                goal=goal,
                done=done,
            )
            carry = add_transition_to_level(carry, level, action_replay_transition)

            if bool(config.get("SUBGOAL_TESTING", True)):
                penalty_mask = jnp.logical_and(lower_subgoal_test, jnp.logical_not(achieved_lower))
                penalty_reward = jnp.where(
                    penalty_mask,
                    float(config.get("SUBGOAL_PENALTY", -time_scale)),
                    reward,
                )
                penalty_done = jnp.where(penalty_mask, 1.0, done)
                penalty_transition = HACTransition(
                    state=old_obs,
                    action=action,
                    reward=penalty_reward,
                    next_state=next_obs,
                    goal=goal,
                    done=penalty_done,
                )
                carry = add_transition_to_level(carry, level, penalty_transition)

            if bool(config.get("HINDSIGHT_GOAL_REPLAY", True)):
                hindsight_goal = project_goal(next_obs)
                hindsight_goal_transition = HACTransition(
                    state=old_obs,
                    action=hindsight_action,
                    reward=jnp.zeros_like(reward),
                    next_state=next_obs,
                    goal=hindsight_goal,
                    done=jnp.ones_like(done),
                )
                carry = add_transition_to_level(carry, level, hindsight_goal_transition)

            return carry, lower_info

        def update_level(
            level: int,
            actor_states,
            q1_states,
            q2_states,
            target_actor_params,
            target_q1_params,
            target_q2_params,
            rb,
            rng,
        ):
            actor_state = actor_states[level]
            q1_state = q1_states[level]
            q2_state = q2_states[level]

            rng, sample_key = jax.random.split(rng)
            batch = rb.sample(sample_key)

            if level == 0:

                def loss_fn(actor_params, q1_params, q2_params):
                    action = batch.action.astype(jnp.int32)
                    batch_idx = jnp.arange(action.shape[0])

                    q1_values = q1_nets[level].apply(q1_params, batch.state, batch.goal)
                    q2_values = q2_nets[level].apply(q2_params, batch.state, batch.goal)
                    q1_selected = q1_values[batch_idx, action]
                    q2_selected = q2_values[batch_idx, action]

                    next_logits = actor_nets[level].apply(
                        target_actor_params[level],
                        batch.next_state,
                        batch.goal,
                    )
                    next_probs = jax.nn.softmax(next_logits, axis=-1)
                    next_target_q1 = q1_nets[level].apply(
                        target_q1_params[level],
                        batch.next_state,
                        batch.goal,
                    )
                    next_target_q2 = q2_nets[level].apply(
                        target_q2_params[level],
                        batch.next_state,
                        batch.goal,
                    )
                    next_target_q = jnp.minimum(next_target_q1, next_target_q2)
                    next_v = jnp.sum(next_probs * next_target_q, axis=-1)

                    target = batch.reward + config["GAMMA"] * (1.0 - batch.done) * next_v
                    target = jnp.clip(target, -q_bound, 0.0)
                    target = jax.lax.stop_gradient(target)

                    q1_loss = 0.5 * jnp.mean((q1_selected - target) ** 2)
                    q2_loss = 0.5 * jnp.mean((q2_selected - target) ** 2)

                    logits = actor_nets[level].apply(actor_params, batch.state, batch.goal)
                    probs = jax.nn.softmax(logits, axis=-1)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    q_for_actor = jax.lax.stop_gradient(
                        jnp.minimum(
                            q1_nets[level].apply(q1_params, batch.state, batch.goal),
                            q2_nets[level].apply(q2_params, batch.state, batch.goal),
                        )
                    )
                    entropy_coef = float(config.get("LEVEL0_ENT_COEF", 0.0))
                    actor_loss = jnp.mean(
                        jnp.sum(probs * (entropy_coef * log_probs - q_for_actor), axis=-1)
                    )
                    entropy = -jnp.sum(probs * log_probs, axis=-1).mean()
                    q_mean = q_for_actor.mean()
                    target_mean = target.mean()
                    total_loss = actor_loss + q1_loss + q2_loss
                    metrics = jnp.array(
                        [actor_loss, q1_loss, q2_loss, q_mean, target_mean, entropy],
                        dtype=jnp.float32,
                    )
                    return total_loss, metrics

            else:

                def loss_fn(actor_params, q1_params, q2_params):
                    next_action = actor_nets[level].apply(
                        target_actor_params[level],
                        batch.next_state,
                        batch.goal,
                    )
                    next_q1 = q1_nets[level].apply(
                        target_q1_params[level],
                        batch.next_state,
                        batch.goal,
                        next_action,
                    )
                    next_q2 = q2_nets[level].apply(
                        target_q2_params[level],
                        batch.next_state,
                        batch.goal,
                        next_action,
                    )
                    next_q = jnp.minimum(next_q1, next_q2)
                    target = batch.reward + config["GAMMA"] * (1.0 - batch.done) * next_q
                    target = jnp.clip(target, -q_bound, 0.0)
                    target = jax.lax.stop_gradient(target)

                    q1_value = q1_nets[level].apply(q1_params, batch.state, batch.goal, batch.action)
                    q2_value = q2_nets[level].apply(q2_params, batch.state, batch.goal, batch.action)
                    q1_loss = 0.5 * jnp.mean((q1_value - target) ** 2)
                    q2_loss = 0.5 * jnp.mean((q2_value - target) ** 2)

                    policy_action = actor_nets[level].apply(actor_params, batch.state, batch.goal)
                    actor_q = q1_nets[level].apply(q1_params, batch.state, batch.goal, policy_action)
                    actor_loss = -jnp.mean(actor_q)
                    entropy = jnp.array(0.0, dtype=jnp.float32)
                    q_mean = actor_q.mean()
                    target_mean = target.mean()
                    total_loss = actor_loss + q1_loss + q2_loss
                    metrics = jnp.array(
                        [actor_loss, q1_loss, q2_loss, q_mean, target_mean, entropy],
                        dtype=jnp.float32,
                    )
                    return total_loss, metrics

            (total_loss, metrics), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
                argnums=(0, 1, 2),
            )(
                actor_state.params,
                q1_state.params,
                q2_state.params,
            )

            actor_grads, q1_grads, q2_grads = grads
            actor_state = actor_state.apply_gradients(grads=actor_grads)
            q1_state = q1_state.apply_gradients(grads=q1_grads)
            q2_state = q2_state.apply_gradients(grads=q2_grads)

            tau = float(config.get("TAU", 0.005))
            new_target_actor_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_actor_params[level],
                actor_state.params,
            )
            new_target_q1_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_q1_params[level],
                q1_state.params,
            )
            new_target_q2_params = jax.tree.map(
                lambda target, online: (1.0 - tau) * target + tau * online,
                target_q2_params[level],
                q2_state.params,
            )

            actor_states = _replace_tuple(actor_states, level, actor_state)
            q1_states = _replace_tuple(q1_states, level, q1_state)
            q2_states = _replace_tuple(q2_states, level, q2_state)
            target_actor_params = _replace_tuple(target_actor_params, level, new_target_actor_params)
            target_q1_params = _replace_tuple(target_q1_params, level, new_target_q1_params)
            target_q2_params = _replace_tuple(target_q2_params, level, new_target_q2_params)

            return (
                actor_states,
                q1_states,
                q2_states,
                target_actor_params,
                target_q1_params,
                target_q2_params,
                rng,
            ), metrics

        def train_loop(run_state, _):
            (
                train_state,
                target_actor_params,
                target_q1_params,
                target_q2_params,
                rbs,
                obs,
                env_state,
                rng,
                update_idx,
            ) = run_state
            actor_states, q1_states, q2_states = train_state

            def collect_top_step(carry: CollectCarry, _):
                top_goal = make_top_goal(carry.obs)
                subgoal_test = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.bool_)
                return run_level(num_levels - 1, carry, top_goal, subgoal_test)

            rng, collect_key = jax.random.split(rng)
            collect_carry = CollectCarry(
                actor_states=actor_states,
                rbs=rbs,
                obs=obs,
                env_state=env_state,
                rng=collect_key,
            )

            collect_carry, rollout_info = jax.lax.scan(
                collect_top_step,
                collect_carry,
                xs=None,
                length=config["NUM_STEPS"],
            )

            rbs = collect_carry.rbs
            next_obs = collect_carry.obs
            next_env_state = collect_carry.env_state
            rng = collect_carry.rng

            def one_update(carry, _):
                (
                    actor_states,
                    q1_states,
                    q2_states,
                    target_actor_params,
                    target_q1_params,
                    target_q2_params,
                    rng,
                ) = carry

                level_metrics = []
                for level in range(num_levels):
                    ready = rbs[level].size >= max(config["WARMUP"], config["BATCH_SIZE"])

                    def do_update(args):
                        return update_level(level, *args)

                    def skip_update(args):
                        (
                            actor_states,
                            q1_states,
                            q2_states,
                            target_actor_params,
                            target_q1_params,
                            target_q2_params,
                            _rb,
                            rng,
                        ) = args
                        nan_metrics = jnp.full((6,), jnp.nan, dtype=jnp.float32)
                        return (
                            actor_states,
                            q1_states,
                            q2_states,
                            target_actor_params,
                            target_q1_params,
                            target_q2_params,
                            rng,
                        ), nan_metrics

                    (
                        actor_states,
                        q1_states,
                        q2_states,
                        target_actor_params,
                        target_q1_params,
                        target_q2_params,
                        rng,
                    ), metrics = jax.lax.cond(
                        ready,
                        do_update,
                        skip_update,
                        operand=(
                            actor_states,
                            q1_states,
                            q2_states,
                            target_actor_params,
                            target_q1_params,
                            target_q2_params,
                            rbs[level],
                            rng,
                        ),
                    )
                    level_metrics.append(metrics)

                return (
                    actor_states,
                    q1_states,
                    q2_states,
                    target_actor_params,
                    target_q1_params,
                    target_q2_params,
                    rng,
                ), jnp.stack(level_metrics)

            (
                actor_states,
                q1_states,
                q2_states,
                target_actor_params,
                target_q1_params,
                target_q2_params,
                rng,
            ), update_metrics = jax.lax.scan(
                one_update,
                (
                    actor_states,
                    q1_states,
                    q2_states,
                    target_actor_params,
                    target_q1_params,
                    target_q2_params,
                    rng,
                ),
                xs=None,
                length=config["NUM_UPDATE_STEPS"],
            )

            # Shape: [num_update_steps, num_levels, 6]
            mean_update_metrics = jnp.nanmean(update_metrics, axis=0)

            episode_mask = rollout_info["returned_episode"].astype(jnp.float32)
            num_completed_episodes = episode_mask.sum()
            episode_metric = jax.tree.map(
                lambda x: jnp.where(
                    num_completed_episodes > 0,
                    (x * episode_mask).sum() / num_completed_episodes,
                    jnp.nan,
                ),
                rollout_info,
            )

            global_step = (
                update_idx
                * config["NUM_ENVS"]
                * config["NUM_STEPS"]
                * atomic_steps_per_top_step
            )

            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, global_step, losses, rb_sizes):
                    to_log = create_log_dict(metric, config)
                    to_log.update({"global_step": int(global_step)})

                    names = [
                        "actor_loss",
                        "q1_loss",
                        "q2_loss",
                        "q_mean",
                        "target_q_mean",
                        "entropy",
                    ]
                    for level in range(losses.shape[0]):
                        for i, name in enumerate(names):
                            to_log[f"hac/level_{level}/{name}"] = float(losses[level, i])
                        to_log[f"hac/level_{level}/buffer_size"] = int(rb_sizes[level])

                    to_log["hac/atomic_steps_per_top_step"] = int(atomic_steps_per_top_step)
                    batch_log(global_step, to_log, config)

                rb_sizes = jnp.asarray([rb.size for rb in rbs], dtype=jnp.int32)
                jax.debug.callback(callback, episode_metric, global_step, mean_update_metrics, rb_sizes)

            train_state = (actor_states, q1_states, q2_states)
            run_state = (
                train_state,
                target_actor_params,
                target_q1_params,
                target_q2_params,
                rbs,
                next_obs,
                next_env_state,
                rng,
                update_idx + 1,
            )
            return run_state, episode_metric

        run_state = (
            train_state,
            target_actor_params,
            target_q1_params,
            target_q2_params,
            rbs,
            obs,
            env_state,
            rng,
            jnp.array(0, dtype=jnp.int32),
        )

        run_state, _ = jax.lax.scan(
            train_loop,
            run_state,
            xs=None,
            length=config["NUM_UPDATES"],
        )
        return run_state

    return train
