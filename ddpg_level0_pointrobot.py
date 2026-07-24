from __future__ import annotations

from typing import Sequence

import numpy as np
import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState

try:
    from logz.batch_logging import batch_log
except Exception:  # pragma: no cover - only used when user's logging package is available.
    batch_log = None

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

from wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)


Array = jax.Array


@struct.dataclass
class Transition:
    obs: Array
    action: Array
    reward: Array
    next_obs: Array
    goal: Array
    discount: Array
    done: Array
    valid: Array


@struct.dataclass
class ReplayBuffer:
    states: Array
    goals: Array
    actions: Array
    rewards: Array
    next_states: Array
    discounts: Array
    dones: Array
    valids: Array

    write_index: Array
    size: Array

    capacity: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        capacity: int,
        batch_size: int,
        state_shape: Sequence[int],
        goal_shape: Sequence[int],
        action_shape: Sequence[int],
        action_dtype=jnp.float32,
        state_dtype=jnp.float32,
    ) -> ReplayBuffer:
        return cls(
            states=jnp.zeros((capacity, *state_shape), dtype=state_dtype),
            goals=jnp.zeros((capacity, *goal_shape), dtype=jnp.float32),
            actions=jnp.zeros((capacity, *action_shape), dtype=action_dtype),
            rewards=jnp.zeros((capacity,), dtype=jnp.float32),
            next_states=jnp.zeros((capacity, *state_shape), dtype=state_dtype),
            discounts=jnp.zeros((capacity,), dtype=jnp.float32),
            dones=jnp.zeros((capacity,), dtype=bool),
            valids=jnp.zeros((capacity,), dtype=bool),
            write_index=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
            capacity=capacity,
            batch_size=batch_size,
        )

    def add_batch(self, batch: Transition) -> ReplayBuffer:
        """Add a flat batch of transitions with leading dimension B."""
        num_added = batch.obs.shape[0]

        if num_added > self.capacity:
            raise ValueError("Inserted batch cannot exceed replay capacity.")

        indices = (
            self.write_index + jnp.arange(num_added, dtype=jnp.int32)
        ) % self.capacity

        return self.replace(
            states=self.states.at[indices].set(batch.obs),
            goals=self.goals.at[indices].set(batch.goal),
            actions=self.actions.at[indices].set(batch.action),
            rewards=self.rewards.at[indices].set(batch.reward),
            next_states=self.next_states.at[indices].set(batch.next_obs),
            discounts=self.discounts.at[indices].set(batch.discount),
            dones=self.dones.at[indices].set(batch.done),
            valids=self.valids.at[indices].set(batch.valid),
            write_index=(self.write_index + num_added) % self.capacity,
            size=jnp.minimum(self.size + num_added, self.capacity),
        )

    def sample(self, rng: Array) -> Transition:
        indices = jax.random.randint(
            rng,
            shape=(self.batch_size,),
            minval=0,
            maxval=self.size,
            dtype=jnp.int32,
        )

        return Transition(
            obs=self.states[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            next_obs=self.next_states[indices],
            goal=self.goals[indices],
            discount=self.discounts[indices],
            done=self.dones[indices],
            valid=self.valids[indices],
        )

    def can_sample(self) -> Array:
        return self.size >= self.batch_size


class Actor(nn.Module):
    dim: int
    action_dim: int

    @nn.compact
    def __call__(self, obs: Array, goal: Array) -> Array:
        x = jnp.concatenate([obs, goal], axis=-1)
        x = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        action = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        # DDPG actor is deterministic. tanh keeps output in [-1, 1].
        return nn.tanh(action)


class Critic(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, obs: Array, goal: Array, action: Array) -> Array:
        x = jnp.concatenate([obs, goal, action], axis=-1)
        x = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.relu(x)
        q = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return jnp.squeeze(q, axis=-1)


def expand_mask(mask: Array, x: Array) -> Array:
    """Expand a [B] mask so it broadcasts over a [B, ...] tensor."""
    return mask.reshape((mask.shape[0],) + (1,) * (x.ndim - 1))


def batch_where(mask: Array, x_true: Array, x_false: Array) -> Array:
    return jnp.where(expand_mask(mask, x_true), x_true, x_false)


def tree_where(mask: Array, true_tree, false_tree):
    return jax.tree.map(
        lambda true, false: jnp.where(expand_mask(mask, true), true, false),
        true_tree,
        false_tree,
    )


def flatten_transitions(transitions: Transition) -> Transition:
    """Flatten [T, B, ...] rollout transitions to [T * B, ...]."""
    return jax.tree.map(
        lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]),
        transitions,
    )


def soft_update(target_params, online_params, tau: float):
    return jax.tree.map(
        lambda target, online: (1.0 - tau) * target + tau * online,
        target_params,
        online_params,
    )


def _host_scalar(x):
    """Convert a JAX value to a Python scalar for logging.

    If a metric accidentally has more than one element, log its mean instead of
    passing the full array to wandb/batch_log. This prevents debug callbacks from
    failing with giant arrays in the traceback.
    """
    arr = np.asarray(jax.device_get(x))
    if arr.shape == () or arr.size == 1:
        return arr.item()
    return float(arr.mean())


def _as_batch_vector(x: Array) -> Array:
    """Return shape [B] from [B] or [B, 1].

    DDPG critics should output one scalar per transition. This helper avoids
    accidental [B, 1] broadcasting against [B] masks, which can otherwise create
    [B, B] arrays and enormous traceback/log outputs.
    """
    x = jnp.asarray(x)
    if x.ndim >= 2 and x.shape[-1] == 1:
        x = jnp.squeeze(x, axis=-1)
    return x


def _per_env_abs_mean(x: Array) -> Array:
    """Mean absolute value per environment, shape [B]."""
    x = jnp.asarray(x)
    if x.ndim == 1:
        return jnp.abs(x)
    return jnp.mean(jnp.abs(x).reshape((x.shape[0], -1)), axis=-1)


def make_train(config):
    """
    Level-0 goal-conditioned DDPG on a continuous gymnax env.

    The training structure and logging mirror the PPO style used in ppo_shared:
    make_train(config) computes NUM_UPDATES from TOTAL_TIMESTEPS, train(rng)
    scans update_step, update_step scans rollout_step, and logging uses
    global_step = update_idx * NUM_ENVS * NUM_STEPS.

    This is not PPO: updates remain DDPG critic/actor updates from replay.
    """
    config = dict(config)

    if "NUM_STEPS" not in config:
        config["NUM_STEPS"] = config.get("SUBGOAL_HORIZON", 20)
    config["SUBGOAL_HORIZON"] = config["NUM_STEPS"]

    if "TOTAL_TIMESTEPS" in config:
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (
            config["NUM_ENVS"] * config["NUM_STEPS"]
        )
    else:
        config["NUM_UPDATES"] = config.get("NUM_UPDATES", 1_000)
        config["TOTAL_TIMESTEPS"] = (
            config["NUM_UPDATES"] * config["NUM_ENVS"] * config["NUM_STEPS"]
        )

    config["NUM_UPDATE_STEPS"] = config.get(
        "NUM_UPDATE_STEPS",
        config.get("UPDATES_PER_ROLLOUT", 1),
    )

    env, env_params = gymnax.make(config.get("ENV_NAME", "PointRobot-misc"))
    env = LogWrapper(env)

    if config.get("USE_OPTIMISTIC_RESETS", False):
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(
                config.get("OPTIMISTIC_RESET_RATIO", config["NUM_ENVS"]),
                config["NUM_ENVS"],
            ),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    obs_shape = env.observation_space(env_params).shape
    obs_dim = obs_shape[-1]

    action_space = env.action_space(env_params)
    action_shape = tuple(getattr(action_space, "shape", ()))
    action_dim = action_shape[-1] if len(action_shape) > 0 else 1
    squeeze_action_for_env = len(action_shape) == 0

    goal_dim = config.get("GOAL_DIM", obs_dim)
    actor_dim = config.get("ACTOR_DIM", config.get("LAYER_SIZE", config.get("DIM", 256)))
    critic_dim = config.get("CRITIC_DIM", config.get("LAYER_SIZE", config.get("DIM", 256)))

    action_low = config.get("ACTION_LOW", -1.0)
    action_high = config.get("ACTION_HIGH", 1.0)
    goal_low = config.get("GOAL_LOW", -1.0)
    goal_high = config.get("GOAL_HIGH", 1.0)
    goal_threshold = config.get("GOAL_THRESHOLD", 0.1)

    def sample_goals(rng: Array) -> Array:
        return jax.random.uniform(
            rng,
            shape=(config["NUM_ENVS"], goal_dim),
            minval=goal_low,
            maxval=goal_high,
            dtype=jnp.float32,
        )

    def achieved_goal(obs: Array) -> Array:
        # For PointRobot this is normally the xy position. Adjust for other envs.
        return obs[..., :goal_dim]

    def subgoal_reached(obs: Array, goal: Array) -> Array:
        dist = jnp.linalg.norm(achieved_goal(obs) - goal, axis=-1)
        return dist <= goal_threshold

    def ddpg_update(
        actor_state: TrainState,
        critic_state: TrainState,
        target_actor_params,
        target_critic_params,
        batch: Transition,
    ):
        tau = config.get("TAU", 0.005)
        valid = batch.valid.astype(jnp.float32)
        valid_sum = jnp.maximum(valid.sum(), 1.0)

        def critic_loss_fn(critic_params):
            next_action = actor_state.apply_fn(
                {"params": target_actor_params},
                batch.next_obs,
                batch.goal,
            )
            next_action = jnp.clip(next_action, action_low, action_high)

            target_q = critic_state.apply_fn(
                {"params": target_critic_params},
                batch.next_obs,
                batch.goal,
                next_action,
            )
            target_q = _as_batch_vector(target_q)
            target_q = jax.lax.stop_gradient(target_q)

            reward = _as_batch_vector(batch.reward)
            discount = _as_batch_vector(batch.discount)
            target = reward + discount * target_q

            q = critic_state.apply_fn(
                {"params": critic_params},
                batch.obs,
                batch.goal,
                batch.action,
            )
            q = _as_batch_vector(q)

            td_error = q - target
            loss = (valid * td_error**2).sum() / valid_sum

            info = {
                "losses/critic_loss": loss,
                "values/q_mean": (valid * q).sum() / valid_sum,
                "values/target_q_mean": (valid * target_q).sum() / valid_sum,
                "values/td_error_abs": (valid * jnp.abs(td_error)).sum() / valid_sum,
            }
            return loss, info

        (critic_loss, critic_info), critic_grads = jax.value_and_grad(
            critic_loss_fn,
            has_aux=True,
        )(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grads)

        def actor_loss_fn(actor_params):
            action = actor_state.apply_fn(
                {"params": actor_params},
                batch.obs,
                batch.goal,
            )
            action = jnp.clip(action, action_low, action_high)

            q = critic_state.apply_fn(
                {"params": critic_state.params},
                batch.obs,
                batch.goal,
                action,
            )
            q = _as_batch_vector(q)

            loss = -(valid * q).sum() / valid_sum
            action_abs = _per_env_abs_mean(action)
            info = {
                "losses/actor_loss": loss,
                "values/actor_q_mean": (valid * q).sum() / valid_sum,
                "policy/action_abs_mean": (valid * action_abs).sum() / valid_sum,
            }
            return loss, info

        (actor_loss, actor_info), actor_grads = jax.value_and_grad(
            actor_loss_fn,
            has_aux=True,
        )(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        target_actor_params = soft_update(
            target_actor_params,
            actor_state.params,
            tau,
        )
        target_critic_params = soft_update(
            target_critic_params,
            critic_state.params,
            tau,
        )

        info = {**critic_info, **actor_info}
        return actor_state, critic_state, target_actor_params, target_critic_params, info

    def train(rng: Array):
        actor = Actor(dim=actor_dim, action_dim=action_dim)
        critic = Critic(dim=critic_dim)

        dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        dummy_goal = jnp.zeros((1, goal_dim), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        rng, actor_key, critic_key, reset_key = jax.random.split(rng, 4)

        actor_variables = actor.init(actor_key, dummy_obs, dummy_goal)
        critic_variables = critic.init(critic_key, dummy_obs, dummy_goal, dummy_action)

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

        target_actor_params = actor_state.params
        target_critic_params = critic_state.params

        buffer = ReplayBuffer.create(
            capacity=config["BUFFER_CAPACITY"],
            batch_size=config["BATCH_SIZE"],
            state_shape=obs_shape,
            goal_shape=(goal_dim,),
            action_shape=(action_dim,),
            action_dtype=jnp.float32,
        )

        obs, env_state = env.reset(reset_key, env_params)

        def update_step(run_state, _):
            (
                actor_state,
                critic_state,
                target_actor_params,
                target_critic_params,
                buffer,
                obs,
                env_state,
                rng,
                update_idx,
            ) = run_state

            rng, goal_key, test_key = jax.random.split(rng, 3)
            goal = sample_goals(goal_key)
            testing = jax.random.bernoulli(
                test_key,
                p=config.get("SUBGOAL_TEST_RATE", 0.0),
                shape=(config["NUM_ENVS"],),
            )
            already_done = jnp.zeros((config["NUM_ENVS"],), dtype=bool)

            def rollout_step(carry, _):
                obs, env_state, already_done, rng = carry

                rng, noise_key, step_key = jax.random.split(rng, 3)
                active = ~already_done

                action = actor_state.apply_fn(
                    {"params": actor_state.params},
                    obs,
                    goal,
                )

                noise = (
                    jax.random.normal(noise_key, shape=action.shape)
                    * config.get("ACTION_NOISE_STD", 0.1)
                )
                noisy_action = action + noise

                # testing=True means no exploration noise for that env.
                action = jnp.where(expand_mask(testing, action), action, noisy_action)
                action = jnp.clip(action, action_low, action_high)

                # Inactive envs are frozen after the step; zero action avoids nonsense actions.
                action = batch_where(active, action, jnp.zeros_like(action))

                env_action = jnp.squeeze(action, axis=-1) if squeeze_action_for_env else action

                next_obs, next_env_state, env_reward, env_done, info = env.step(
                    step_key,
                    env_state,
                    env_action,
                    env_params,
                )

                achieved = subgoal_reached(next_obs, goal)
                done = achieved | env_done
                reward = jnp.where(achieved, 0.0, -1.0)
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

                # Freeze envs that had already completed the subgoal before this step.
                next_obs = batch_where(active, next_obs, obs)
                next_env_state = tree_where(active, next_env_state, env_state)

                next_carry = (next_obs, next_env_state, new_already_done, rng)
                return next_carry, (transition, info)

            rollout_state = (obs, env_state, already_done, rng)
            rollout_state, (rollout, infos) = jax.lax.scan(
                rollout_step,
                rollout_state,
                xs=None,
                length=config["NUM_STEPS"],
            )
            final_obs, final_env_state, already_done, rng = rollout_state

            flat_rollout = flatten_transitions(rollout)
            buffer = buffer.add_batch(flat_rollout)

            def zero_update_info():
                return {
                    "losses/critic_loss": jnp.array(0.0),
                    "values/q_mean": jnp.array(0.0),
                    "values/target_q_mean": jnp.array(0.0),
                    "values/td_error_abs": jnp.array(0.0),
                    "losses/actor_loss": jnp.array(0.0),
                    "values/actor_q_mean": jnp.array(0.0),
                    "policy/action_abs_mean": jnp.array(0.0),
                }

            def update_once(update_state, _):
                (
                    actor_state,
                    critic_state,
                    target_actor_params,
                    target_critic_params,
                    rng,
                ) = update_state

                rng, sample_key = jax.random.split(rng)

                def do_update(_):
                    batch = buffer.sample(sample_key)
                    return ddpg_update(
                        actor_state,
                        critic_state,
                        target_actor_params,
                        target_critic_params,
                        batch,
                    )

                def skip_update(_):
                    return (
                        actor_state,
                        critic_state,
                        target_actor_params,
                        target_critic_params,
                        zero_update_info(),
                    )

                (
                    actor_state,
                    critic_state,
                    target_actor_params,
                    target_critic_params,
                    update_info,
                ) = jax.lax.cond(buffer.can_sample(), do_update, skip_update, operand=None)

                update_state = (
                    actor_state,
                    critic_state,
                    target_actor_params,
                    target_critic_params,
                    rng,
                )
                return update_state, update_info

            update_state = (
                actor_state,
                critic_state,
                target_actor_params,
                target_critic_params,
                rng,
            )
            update_state, update_infos = jax.lax.scan(
                update_once,
                update_state,
                xs=None,
                length=config["NUM_UPDATE_STEPS"],
            )
            (
                actor_state,
                critic_state,
                target_actor_params,
                target_critic_params,
                rng,
            ) = update_state

            valid = rollout.valid.astype(jnp.float32)
            valid_sum = jnp.maximum(valid.sum(), 1.0)
            rollout_return = (rollout.reward * valid).sum(axis=0)
            success = (rollout.done & rollout.valid).any(axis=0)
            final_distance = jnp.linalg.norm(achieved_goal(final_obs) - goal, axis=-1)

            # PPO-style optional LogWrapper metrics, if the wrapper provides these keys.
            env_metric = {}
            if isinstance(infos, dict) and "returned_episode" in infos:
                returned = infos["returned_episode"].astype(jnp.float32)
                returned_sum = jnp.maximum(returned.sum(), 1.0)
                for key in ("returned_episode_return", "returned_episode_length"):
                    if key in infos:
                        env_metric[f"env/{key}"] = (infos[key] * returned).sum() / returned_sum
                env_metric["env/returned_episode_count"] = returned.sum()

            update_last = jax.tree.map(lambda x: x[-1], update_infos)

            global_step = update_idx * config["NUM_ENVS"] * config["NUM_STEPS"]

            metrics = {
                "global_step": global_step,
                "charts/update": update_idx,
                "charts/buffer_size": buffer.size,
                "charts/valid_steps": valid_sum,
                "rollout/mean_return": rollout_return.mean(),
                "rollout/success_rate": success.astype(jnp.float32).mean(),
                "rollout/mean_final_distance": final_distance.mean(),
                "rollout/min_final_distance": final_distance.min(),
                "rollout/testing_rate": testing.astype(jnp.float32).mean(),
                **update_last,
                **env_metric,
            }

            if config.get("DEBUG", False) and config.get("USE_WANDB", False):

                def callback(metric, step):
                    step = int(_host_scalar(step))
                    to_log = {k: _host_scalar(v) for k, v in metric.items()}
                    to_log["global_step"] = step
                    if batch_log is not None:
                        batch_log(step, to_log, config)
                    elif wandb is not None:
                        wandb.log(to_log, step=step)
                    else:
                        print({"step": step, **to_log})

                jax.debug.callback(callback, metrics, global_step)

            runner_state = (
                actor_state,
                critic_state,
                target_actor_params,
                target_critic_params,
                buffer,
                final_obs,
                final_env_state,
                rng,
                update_idx + 1,
            )
            return runner_state, metrics

        rng, train_key = jax.random.split(rng)
        runner_state = (
            actor_state,
            critic_state,
            target_actor_params,
            target_critic_params,
            buffer,
            obs,
            env_state,
            train_key,
            jnp.array(0, dtype=jnp.int32),
        )

        runner_state, metrics = jax.lax.scan(
            update_step,
            runner_state,
            xs=None,
            length=config["NUM_UPDATES"],
        )

        return {"runner_state": runner_state, "metrics": metrics, "config": config}

    return train


DEFAULT_CONFIG = {
    "ENV_NAME": "PointRobot-misc",
    "TOTAL_TIMESTEPS": 640_000,
    "NUM_ENVS": 32,
    "NUM_STEPS": 20,
    "NUM_UPDATE_STEPS": 4,
    "BUFFER_CAPACITY": 100_000,
    "BATCH_SIZE": 256,
    "GAMMA": 0.98,
    "TAU": 0.005,
    "ACTOR_LR": 3e-4,
    "CRITIC_LR": 3e-4,
    "LAYER_SIZE": 256,
    "GOAL_DIM": 2,
    "GOAL_LOW": -1.0,
    "GOAL_HIGH": 1.0,
    "GOAL_THRESHOLD": 0.1,
    "ACTION_LOW": -1.0,
    "ACTION_HIGH": 1.0,
    "ACTION_NOISE_STD": 0.1,
    "SUBGOAL_TEST_RATE": 0.0,
    "USE_OPTIMISTIC_RESETS": False,
    "OPTIMISTIC_RESET_RATIO": 16,
    "DEBUG": False,
    "USE_WANDB": False,
}


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    train = make_train(DEFAULT_CONFIG)
    out = jax.jit(train(rng))
    last_metrics = jax.tree.map(
        lambda x: x[-1] if hasattr(x, "ndim") and x.ndim > 0 else x,
        out["metrics"],
    )
    print(last_metrics)
