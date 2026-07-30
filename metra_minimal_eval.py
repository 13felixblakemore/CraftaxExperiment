from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple
import csv
import json
import numpy as np
import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from mujoco_playground import registry
from mujoco_playground import wrapper
from pathlib import Path

import mediapy as media
from logz.batch_logging import batch_log


LOG_2PI = jnp.log(2.0 * jnp.pi)
LOG_2 = jnp.log(2.0)


class Encoder(nn.Module):
    z_dim: int
    layer_size: int

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        x = nn.Dense(self.layer_size)(state)
        x = nn.relu(x)
        x = nn.Dense(self.layer_size)(x)
        x = nn.relu(x)
        return nn.Dense(self.z_dim)(x)


class Actor(nn.Module):
    """Tanh-squashed diagonal Gaussian policy."""

    dim: int
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(
        self,
        obs: jax.Array,
        z: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([obs, z], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(-0.5),
        )(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class Critic(nn.Module):
    """Continuous-action Q(s, z, a) critic."""

    dim: int

    @nn.compact
    def __call__(
        self,
        obs: jax.Array,
        z: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        x = jnp.concatenate([obs, z, action], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    next_obs: jax.Array
    z: jax.Array
    done: jax.Array


def _select_observation(obs, obs_key: str) -> jax.Array:
    """Selects the actor observation from Playground dict observations."""
    if isinstance(obs, Mapping):
        if obs_key not in obs:
            raise KeyError(
                f"Observation key {obs_key!r} not found. "
                f"Available keys: {tuple(obs.keys())}"
            )
        return obs[obs_key]
    return obs


def get_metra_obs(env_state):
    policy_obs = _select_observation(env_state.obs, "state")

    # Walker planar root coordinates are x, z, pitch, followed by joints.
    root_x = env_state.data.qpos[..., 0:1]

    return jnp.concatenate(
        [policy_obs, root_x],
        axis=-1,
    )

def _get_policy_observation(env_state, config) -> jax.Array:
    """Constructs exactly the observation used by actor, critics, and phi."""
    obs = _select_observation(
        env_state.obs,
        config.get("OBS_KEY", "state"),
    )

    # Recommended for METRA on WalkerWalk because its standard observation
    # omits global horizontal position.
    if config.get("INCLUDE_ROOT_X", False):
        root_x = env_state.data.qpos[..., 0:1]
        obs = jnp.concatenate([obs, root_x], axis=-1)

    return obs


def make_train(config):
    """Builds a JAX METRA + continuous SAC training function for Playground."""

    env_name = config.get("ENV_NAME", "Go1JoystickFlatTerrain")
    env_config = registry.get_default_config(env_name)

    env_config_overrides = dict(config.get("ENV_CONFIG_OVERRIDES", {}))
    if "PLAYGROUND_IMPL" in config:
        env_config_overrides["impl"] = config["PLAYGROUND_IMPL"]

    env = registry.load(
        env_name,
        config=env_config,
        config_overrides=env_config_overrides or None,
    )

    episode_length = int(
        config.get("EPISODE_LENGTH", getattr(env_config, "episode_length", 1000))
    )
    action_repeat = int(
        config.get("ACTION_REPEAT", getattr(env_config, "action_repeat", 1))
    )

    # This adds vmap, episode limits, and auto-reset. With full_reset=False,
    # each parallel environment resets to its cached initial state.
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        full_reset=config.get("FULL_RESET", False),
    )

    obs_key = config.get("OBS_KEY", "state")
    num_envs = config["NUM_ENVS"]

    def train(rng):
        action_dim = env.action_size

        def sample_z(key: jax.Array) -> jax.Array:
            z = jax.random.normal(key, (num_envs, config["Z_DIM"]))
            if config.get("UNIT_Z", True):
                z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
            return z

        def sample_z_discrete(key: jax.Array) -> jax.Array:
            skill_ids = jax.random.randint(
                key,
                shape=(num_envs,),
                minval=0,
                maxval=config["Z_DIM"],
            )

            return jax.nn.one_hot(
                skill_ids,
                config["Z_DIM"],
                dtype=jnp.float32,
            )

        def sample_action(
            actor_params,
            obs: jax.Array,
            z: jax.Array,
            key: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            """Samples tanh(N(mean, std)) and returns its corrected log-prob."""
            mean, log_std = actor.apply(actor_params, obs, z)
            std = jnp.exp(log_std)
            noise = jax.random.normal(key, mean.shape)
            pre_tanh = mean + std * noise
            action = jnp.tanh(pre_tanh)

            normal_log_prob = -0.5 * (
                jnp.square(noise) + 2.0 * log_std + LOG_2PI
            )
            normal_log_prob = normal_log_prob.sum(axis=-1)

            # Stable log(1 - tanh(x)^2) correction.
            log_det_jacobian = 2.0 * (
                LOG_2 - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh)
            )
            log_prob = normal_log_prob - log_det_jacobian.sum(axis=-1)
            return action, log_prob, mean, log_std

        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, num_envs)
        env_state = env.reset(reset_keys)
        obs = get_metra_obs(env_state)

        if obs.ndim != 2:
            raise ValueError(
                "This script expects flat state observations with shape "
                f"(NUM_ENVS, OBS_DIM), but received {obs.shape}."
            )

        state_dim = obs.shape[-1]

        actor = Actor(
            dim=config["LAYER_SIZE"],
            action_dim=action_dim,
            log_std_min=config.get("LOG_STD_MIN", -5.0),
            log_std_max=config.get("LOG_STD_MAX", 2.0),
        )
        q1 = Critic(config["LAYER_SIZE"])
        q2 = Critic(config["LAYER_SIZE"])
        phi = Encoder(config["Z_DIM"], config["LAYER_SIZE"])

        dummy_obs = jnp.zeros((1, state_dim), dtype=jnp.float32)
        dummy_z = jnp.zeros((1, config["Z_DIM"]), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        rng, actor_key, q1_key, q2_key, phi_key = jax.random.split(rng, 5)

        actor_params = actor.init(actor_key, dummy_obs, dummy_z)
        q1_params = q1.init(q1_key, dummy_obs, dummy_z, dummy_action)
        q2_params = q2.init(q2_key, dummy_obs, dummy_z, dummy_action)
        phi_params = phi.init(phi_key, dummy_obs)

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=optax.adam(config.get("ACTOR_LR", config["LR"])),
        )

        q1_state = TrainState.create(
            apply_fn=q1.apply,
            params=q1_params,
            tx=optax.adam(config.get("CRITIC_LR", config["LR"])),
        )

        q2_state = TrainState.create(
            apply_fn=q2.apply,
            params=q2_params,
            tx=optax.adam(config.get("CRITIC_LR", config["LR"])),
        )

        phi_state = TrainState.create(
            apply_fn=phi.apply,
            params=phi_params,
            tx=optax.adam(config.get("PHI_LR", config["LR"])),
        )

        log_alpha_state = TrainState.create(
            apply_fn=lambda params: params["log_alpha"],
            params={
                "log_alpha": jnp.array(
                    jnp.log(config.get("ALPHA_INIT", 0.001)),
                    dtype=jnp.float32,
                )
            },
            tx=optax.adam(config.get("ALPHA_LR", config["LR"])),
        )

        log_lambda_state = TrainState.create(
            apply_fn=lambda params: params["log_lambda"],
            params={
                "log_lambda": jnp.array(
                    jnp.log(config.get("LAMBDA_INIT", 30.0)),
                    dtype=jnp.float32,
                )
            },
            tx=optax.adam(config.get("LAMBDA_LR", config["LR"])),
        )

        target_q1_params = q1_params
        target_q2_params = q2_params

        rng, z_key = jax.random.split(rng)

        if config["DISCRETE"]:
            z = sample_z_discrete(z_key)
        else:
            z = sample_z(z_key)

        buffer = fbx.make_item_buffer(
            config["BUFFER_CAPACITY"],
            config["WARMUP"],
            config["BATCH_SIZE"],
            add_sequences=False,
            add_batches=True,
        )

        single_obs = obs[0]
        init_transition = Transition(
            obs=single_obs,
            action=jnp.zeros((action_dim,), dtype=jnp.float32),
            next_obs=single_obs,
            z=jnp.zeros((config["Z_DIM"],), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.bool_),
        )
        buffer_state = buffer.init(init_transition)

        # Playground's built-in reward is logged only. METRA/SAC trains on the
        # intrinsic reward produced by phi(s') - phi(s).
        running_episode_return = jnp.zeros((num_envs,), dtype=jnp.float32)
        running_episode_length = jnp.zeros((num_envs,), dtype=jnp.int32)

        def train_loop(carry, iteration):
            (
                train_state,
                target_q1_params,
                target_q2_params,
                log_lambda_state,
                log_alpha_state,
                obs,
                env_state,
                buffer_state,
                z,
                running_episode_return,
                running_episode_length,
                rng,
            ) = carry

            actor_state, q1_state, q2_state, phi_state = train_state

            def collect_rollout(carry, _):
                def step(carry, _):
                    (
                        obs,
                        env_state,
                        z,
                        running_episode_return,
                        running_episode_length,
                        rng,
                    ) = carry

                    rng, action_key = jax.random.split(rng)
                    action, _, _, _ = sample_action(
                        actor_state.params,
                        obs,
                        z,
                        action_key,
                    )

                    next_env_state = env.step(env_state, action)
                    next_obs = get_metra_obs(next_env_state)
                    env_reward = next_env_state.reward.astype(jnp.float32)
                    done = next_env_state.done > 0.0

                    transition = Transition(
                        obs=obs,
                        action=action,
                        next_obs=next_obs,
                        z=z,
                        done=done,
                    )

                    updated_episode_return = running_episode_return + env_reward
                    updated_episode_length = running_episode_length + 1

                    returned_episode_returns = jnp.where(
                        done,
                        updated_episode_return,
                        jnp.zeros_like(updated_episode_return),
                    )
                    returned_episode_lengths = jnp.where(
                        done,
                        updated_episode_length,
                        jnp.zeros_like(updated_episode_length),
                    )

                    running_episode_return = jnp.where(
                        done,
                        jnp.zeros_like(updated_episode_return),
                        updated_episode_return,
                    )
                    running_episode_length = jnp.where(
                        done,
                        jnp.zeros_like(updated_episode_length),
                        updated_episode_length,
                    )

                    # Keep one skill fixed for an episode, then resample it.
                    rng, z_key = jax.random.split(rng)
                    if config["DISCRETE"]:
                        new_z = sample_z_discrete(z_key)
                    else:
                        new_z = sample_z(z_key)
                    z = jnp.where(done[:, None], new_z, z)

                    step_info = {
                        "returned_episode": done,
                        "returned_episode_returns": returned_episode_returns,
                        "returned_episode_lengths": returned_episode_lengths,
                        "env_reward": env_reward,
                    }

                    carry = (
                        next_obs,
                        next_env_state,
                        z,
                        running_episode_return,
                        running_episode_length,
                        rng,
                    )
                    return carry, (transition, step_info)

                (
                    obs,
                    env_state,
                    buffer_state,
                    z,
                    running_episode_return,
                    running_episode_length,
                    rng,
                ) = carry

                state, (transitions, info) = jax.lax.scan(
                    step,
                    (
                        obs,
                        env_state,
                        z,
                        running_episode_return,
                        running_episode_length,
                        rng,
                    ),
                    xs=None,
                    length=config["NUM_STEPS"],
                )

                (
                    obs,
                    env_state,
                    z,
                    running_episode_return,
                    running_episode_length,
                    rng,
                ) = state

                # (NUM_STEPS, NUM_ENVS, ...) -> (NUM_STEPS * NUM_ENVS, ...)
                transitions = jax.tree.map(
                    lambda x: x.reshape((-1, *x.shape[2:])),
                    transitions,
                )
                buffer_state = buffer.add(buffer_state, transitions)

                carry = (
                    obs,
                    env_state,
                    buffer_state,
                    z,
                    running_episode_return,
                    running_episode_length,
                    rng,
                )
                return carry, info

            rollout_init = (
                obs,
                env_state,
                buffer_state,
                z,
                running_episode_return,
                running_episode_length,
                rng,
            )

            rollout_carry, info = jax.lax.scan(
                collect_rollout,
                rollout_init,
                xs=None,
                length=config["NUM_TRAJECTORIES"],
            )

            (
                obs,
                env_state,
                buffer_state,
                z,
                running_episode_return,
                running_episode_length,
                rng,
            ) = rollout_carry

            returned = info["returned_episode"].astype(jnp.float32)
            num_episodes = returned.sum()

            episode_return_sum = (
                info["returned_episode_returns"] * returned
            ).sum()
            episode_length_sum = (
                info["returned_episode_lengths"].astype(jnp.float32) * returned
            ).sum()

            episode_return = jnp.where(
                num_episodes > 0,
                episode_return_sum / num_episodes,
                jnp.nan,
            )
            episode_length_metric = jnp.where(
                num_episodes > 0,
                episode_length_sum / num_episodes,
                jnp.nan,
            )

            rollout_metric = {
                "episode_return": episode_return,
                "episode_length": episode_length_metric,
                "num_episodes": num_episodes,
                "env_reward": jnp.mean(info["env_reward"]),
            }

            def update(carry, _):
                (
                    train_state,
                    target_q1_params,
                    target_q2_params,
                    log_lambda_state,
                    log_alpha_state,
                    rng,
                ) = carry

                actor_state, q1_state, q2_state, phi_state = train_state

                rng, sample_key = jax.random.split(rng)
                transition = buffer.sample(buffer_state, sample_key).experience
                obs_b, action_b, next_obs_b, z_b, done_b = transition

                nonterminal = 1.0 - done_b.astype(jnp.float32)
                valid_denom = jnp.maximum(nonterminal.sum(), 1.0)

                def metra_components(current_phi_params):
                    phi_obs = phi.apply(current_phi_params, obs_b)
                    phi_next = phi.apply(current_phi_params, next_obs_b)
                    phi_diff = phi_next - phi_obs

                    if config.get("DISCRETE", False):
                        num_skills = config["Z_DIM"]

                        skill_mask = (
                                             z_b - jnp.mean(z_b, axis=-1, keepdims=True)
                                     ) * num_skills / max(num_skills - 1, 1)

                        raw_r = jnp.sum(phi_diff * skill_mask, axis=-1)
                    else:
                        raw_r = jnp.sum(phi_diff * z_b, axis=-1)

                    r = raw_r * nonterminal

                    # This preserves your dual_dist='one' implementation.
                    sq_dist_unmasked = jnp.mean(jnp.square(phi_diff), axis=-1)
                    sq_dist = sq_dist_unmasked * nonterminal

                    cst_dist = jnp.ones_like(sq_dist_unmasked)
                    cst_penalty = cst_dist - sq_dist_unmasked
                    cst_penalty = jnp.minimum(
                        cst_penalty,
                        config.get("LAGRANGE_EPS", 0.001),
                    )
                    cst_penalty = cst_penalty * nonterminal

                    phi_delta_norm = jnp.linalg.norm(phi_diff, axis=-1)

                    return {
                        "raw_r": raw_r,
                        "r": r,
                        "sq_dist_unmasked": sq_dist_unmasked,
                        "sq_dist": sq_dist,
                        "cst_penalty": cst_penalty,
                        "phi_delta_norm": phi_delta_norm,
                    }

                log_lambda = log_lambda_state.params["log_lambda"]

                def phi_loss_fn(current_phi_params):
                    comp = metra_components(current_phi_params)
                    lambda_ = jnp.exp(log_lambda)
                    objective = (
                        comp["r"]
                        + jax.lax.stop_gradient(lambda_)
                        * comp["cst_penalty"]
                    )
                    loss = -objective.sum() / valid_denom

                    aux = {
                        "metra_reward": comp["r"].sum() / valid_denom,
                        "raw_metra_reward": jnp.mean(comp["raw_r"]),
                        "abs_metra_reward": (
                            jnp.abs(comp["r"]).sum() / valid_denom
                        ),
                        "positive_metra_reward_frac": (
                            (
                                (comp["r"] > 0).astype(jnp.float32)
                                * nonterminal
                            ).sum()
                            / valid_denom
                        ),
                        "phi_sq_dist": comp["sq_dist"].sum() / valid_denom,
                        "phi_sq_dist_unmasked": jnp.mean(
                            comp["sq_dist_unmasked"]
                        ),
                        "phi_delta_norm": (
                            (comp["phi_delta_norm"] * nonterminal).sum()
                            / valid_denom
                        ),
                        "cst_penalty": (
                            comp["cst_penalty"].sum() / valid_denom
                        ),
                    }
                    return loss, aux

                (phi_loss, phi_info), phi_grad = jax.value_and_grad(
                    phi_loss_fn,
                    has_aux=True,
                )(phi_state.params)
                phi_state = phi_state.apply_gradients(grads=phi_grad)

                def lambda_loss_fn(log_lambda_params):
                    current_log_lambda = log_lambda_params["log_lambda"]
                    comp = metra_components(phi_state.params)
                    mean_cst_penalty = (
                        jax.lax.stop_gradient(comp["cst_penalty"]).sum()
                        / valid_denom
                    )
                    return current_log_lambda * mean_cst_penalty

                lambda_loss, lambda_grad = jax.value_and_grad(lambda_loss_fn)(
                    log_lambda_state.params
                )
                log_lambda_state = log_lambda_state.apply_gradients(
                    grads=lambda_grad
                )

                rng, next_action_key = jax.random.split(rng)
                log_alpha = log_alpha_state.params["log_alpha"]

                def critic_loss_fn(current_q1_params, current_q2_params):
                    alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))

                    q1_value = q1.apply(
                        current_q1_params,
                        obs_b,
                        z_b,
                        action_b,
                    )
                    q2_value = q2.apply(
                        current_q2_params,
                        obs_b,
                        z_b,
                        action_b,
                    )

                    next_action, next_log_prob, _, _ = sample_action(
                        actor_state.params,
                        next_obs_b,
                        z_b,
                        next_action_key,
                    )
                    target_q1 = q1.apply(
                        target_q1_params,
                        next_obs_b,
                        z_b,
                        next_action,
                    )
                    target_q2 = q2.apply(
                        target_q2_params,
                        next_obs_b,
                        z_b,
                        next_action,
                    )
                    next_v = (
                        jnp.minimum(target_q1, target_q2)
                        - alpha * next_log_prob
                    )

                    comp = metra_components(phi_state.params)
                    intrinsic_r = config.get("METRA_REWARD_SCALE", 10.0) * jax.lax.stop_gradient(comp["r"])
                    target = (
                        intrinsic_r
                        + config["GAMMA"] * nonterminal * next_v
                    )
                    target = jax.lax.stop_gradient(target)

                    q1_loss = 0.5 * jnp.mean(jnp.square(q1_value - target))
                    q2_loss = 0.5 * jnp.mean(jnp.square(q2_value - target))

                    aux = {
                        "q1_mean": jnp.mean(q1_value),
                        "q2_mean": jnp.mean(q2_value),
                        "target_q_mean": jnp.mean(target),
                        "q_abs_error": jnp.mean(jnp.abs(q1_value - target)),
                        "target_log_prob": jnp.mean(next_log_prob),
                    }
                    return q1_loss + q2_loss, aux

                (critic_loss, critic_info), critic_grad = jax.value_and_grad(
                    critic_loss_fn,
                    argnums=(0, 1),
                    has_aux=True,
                )(q1_state.params, q2_state.params)

                q1_grad, q2_grad = critic_grad
                q1_state = q1_state.apply_gradients(grads=q1_grad)
                q2_state = q2_state.apply_gradients(grads=q2_grad)

                rng, actor_action_key = jax.random.split(rng)

                def actor_loss_fn(current_actor_params):
                    sampled_action, log_prob, mean, log_std = sample_action(
                        current_actor_params,
                        obs_b,
                        z_b,
                        actor_action_key,
                    )
                    q1_value = q1.apply(
                        q1_state.params,
                        obs_b,
                        z_b,
                        sampled_action,
                    )
                    q2_value = q2.apply(
                        q2_state.params,
                        obs_b,
                        z_b,
                        sampled_action,
                    )
                    q_value = jnp.minimum(q1_value, q2_value)
                    alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))
                    loss = jnp.mean(alpha * log_prob - q_value)

                    aux = {
                        "entropy": -jnp.mean(log_prob),
                        "expected_q": jnp.mean(q_value),
                        "action_abs_mean": jnp.mean(jnp.abs(sampled_action)),
                        "action_saturation_frac": jnp.mean(
                            (jnp.abs(sampled_action) > 0.95).astype(jnp.float32)
                        ),
                        "policy_mean_abs": jnp.mean(jnp.abs(jnp.tanh(mean))),
                        "policy_std": jnp.mean(jnp.exp(log_std)),
                        "policy_log_prob": jnp.mean(log_prob),
                    }
                    return loss, aux

                (actor_loss, actor_info), actor_grad = jax.value_and_grad(
                    actor_loss_fn,
                    has_aux=True,
                )(actor_state.params)
                actor_state = actor_state.apply_gradients(grads=actor_grad)

                rng, alpha_action_key = jax.random.split(rng)

                def alpha_loss_fn(log_alpha_params):
                    current_log_alpha = log_alpha_params["log_alpha"]
                    _, log_prob, _, _ = sample_action(
                        actor_state.params,
                        obs_b,
                        z_b,
                        alpha_action_key,
                    )
                    target_entropy = jnp.asarray(
                        config.get("TARGET_ENTROPY", -0.5 * action_dim),
                        dtype=jnp.float32,
                    )
                    return -jnp.mean(
                        current_log_alpha
                        * jax.lax.stop_gradient(log_prob + target_entropy)
                    )

                alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
                    log_alpha_state.params
                )
                log_alpha_state = log_alpha_state.apply_gradients(
                    grads=alpha_grad
                )

                target_q1_params = jax.tree.map(
                    lambda target, online: (
                        (1.0 - config["TAU"]) * target
                        + config["TAU"] * online
                    ),
                    target_q1_params,
                    q1_state.params,
                )
                target_q2_params = jax.tree.map(
                    lambda target, online: (
                        (1.0 - config["TAU"]) * target
                        + config["TAU"] * online
                    ),
                    target_q2_params,
                    q2_state.params,
                )

                train_state = actor_state, q1_state, q2_state, phi_state
                update_carry = (
                    train_state,
                    target_q1_params,
                    target_q2_params,
                    log_lambda_state,
                    log_alpha_state,
                    rng,
                )

                update_metric = {
                    "metra_reward": phi_info["metra_reward"],
                    "raw_metra_reward": phi_info["raw_metra_reward"],
                    "abs_metra_reward": phi_info["abs_metra_reward"],
                    "positive_metra_reward_frac": phi_info[
                        "positive_metra_reward_frac"
                    ],
                    "phi_sq_dist": phi_info["phi_sq_dist"],
                    "phi_sq_dist_unmasked": phi_info[
                        "phi_sq_dist_unmasked"
                    ],
                    "phi_delta_norm": phi_info["phi_delta_norm"],
                    "cst_penalty": phi_info["cst_penalty"],
                    "phi_loss": phi_loss,
                    "lambda_loss": lambda_loss,
                    "critic_loss": critic_loss,
                    "actor_loss": actor_loss,
                    "alpha_loss": alpha_loss,
                    "entropy": actor_info["entropy"],
                    "expected_q": actor_info["expected_q"],
                    "action_abs_mean": actor_info["action_abs_mean"],
                    "action_saturation_frac": actor_info[
                        "action_saturation_frac"
                    ],
                    "policy_mean_abs": actor_info["policy_mean_abs"],
                    "policy_std": actor_info["policy_std"],
                    "policy_log_prob": actor_info["policy_log_prob"],
                    "q1_mean": critic_info["q1_mean"],
                    "q2_mean": critic_info["q2_mean"],
                    "target_q_mean": critic_info["target_q_mean"],
                    "target_log_prob": critic_info["target_log_prob"],
                    "q_abs_error": critic_info["q_abs_error"],
                    "alpha": jnp.exp(
                        log_alpha_state.params["log_alpha"]
                    ),
                    "lambda": jnp.exp(
                        log_lambda_state.params["log_lambda"]
                    ),
                    "valid_batch_frac": jnp.mean(nonterminal),
                }
                return update_carry, update_metric

            update_init = (
                train_state,
                target_q1_params,
                target_q2_params,
                log_lambda_state,
                log_alpha_state,
                rng,
            )

            update_carry, update_metric = jax.lax.scan(
                update,
                update_init,
                xs=None,
                length=config["NUM_UPDATE_STEPS"],
            )
            update_metric = jax.tree.map(jnp.mean, update_metric)

            global_step = (
                (iteration + 1)
                * config["NUM_TRAJECTORIES"]
                * config["NUM_STEPS"]
                * num_envs
            )

            metric = {
                **rollout_metric,
                **update_metric,
            }

            if config.get("DEBUG", False) and config.get("USE_WANDB", False):

                def callback(metric, global_step):
                    to_log = {}
                    for key, value in metric.items():
                        value = float(value)
                        if value == value:
                            to_log[key] = value
                    to_log["global_step"] = int(global_step)
                    batch_log(int(global_step), to_log, config)

                jax.debug.callback(callback, metric, global_step)

            (
                train_state,
                target_q1_params,
                target_q2_params,
                log_lambda_state,
                log_alpha_state,
                rng,
            ) = update_carry

            outer_carry = (
                train_state,
                target_q1_params,
                target_q2_params,
                log_lambda_state,
                log_alpha_state,
                obs,
                env_state,
                buffer_state,
                z,
                running_episode_return,
                running_episode_length,
                rng,
            )
            return outer_carry, metric

        train_state = actor_state, q1_state, q2_state, phi_state

        init = (
            train_state,
            target_q1_params,
            target_q2_params,
            log_lambda_state,
            log_alpha_state,
            obs,
            env_state,
            buffer_state,
            z,
            running_episode_return,
            running_episode_length,
            rng,
        )

        iterations = config["TOTAL_TIMESTEPS"] // (
            config["NUM_TRAJECTORIES"]
            * config["NUM_STEPS"]
            * num_envs
        )

        carry, metric = jax.lax.scan(
            train_loop,
            init,
            xs=jnp.arange(iterations),
        )

        train_state = carry[0]

        return train_state

    return train


def _make_eval_latents(config) -> jax.Array:
    z_dim = int(config["Z_DIM"])

    if config.get("DISCRETE", False):
        return jnp.eye(z_dim, dtype=jnp.float32)

    if z_dim == 1:
        return jnp.array([[-1.0], [1.0]], dtype=jnp.float32)

    if z_dim == 2:
        n = int(config.get("NUM_EVAL_SKILLS", 8))
        angles = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    n = int(config.get("NUM_EVAL_SKILLS", 8))
    key = jax.random.PRNGKey(int(config.get("EVAL_SEED", 0)))
    z = jax.random.normal(key, (n, z_dim))
    if config.get("UNIT_Z", True):
        z /= jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8
    return z.astype(jnp.float32)


def _make_raw_eval_env(config):
    env_name = config.get("ENV_NAME", "WalkerWalk")
    env_config = registry.get_default_config(env_name)
    overrides = dict(config.get("ENV_CONFIG_OVERRIDES", {}))
    if "PLAYGROUND_IMPL" in config:
        overrides["impl"] = config["PLAYGROUND_IMPL"]
    env = registry.load(
        env_name,
        config=env_config,
        config_overrides=overrides or None,
    )
    return env_name, env


def _eval_reward_vectors(eval_zs: jax.Array, config) -> jax.Array:
    if not config.get("DISCRETE", False):
        return eval_zs
    k = int(config["Z_DIM"])
    return (eval_zs - eval_zs.mean(axis=-1, keepdims=True)) * k / max(k - 1, 1)


def _qpos(state, index: int) -> float:
    return float(np.asarray(jax.device_get(state.data.qpos)).reshape(-1)[index])


def _write_csv(path: Path, rows: list[dict], z_dim: int) -> None:
    if not rows:
        return
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k != "z"}
        for i, value in enumerate(np.asarray(row["z"]).reshape(-1)):
            flat[f"z_{i}"] = float(value)
        flat_rows.append(flat)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_rows)


def evaluate_skill_diagnostics(
    config,
    actor_state: TrainState,
    phi_state: TrainState | None = None,
) -> dict:
    """Small deterministic evaluation focused on actual skill diversity."""
    env_name, env = _make_raw_eval_env(config)
    actor = Actor(
        config["LAYER_SIZE"],
        env.action_size,
        config.get("LOG_STD_MIN", -5.0),
        config.get("LOG_STD_MAX", 2.0),
    )
    encoder = Encoder(config["Z_DIM"], config["LAYER_SIZE"])

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    @jax.jit
    def act(params, obs, z):
        mean, _ = actor.apply(params, obs, z)
        return jnp.tanh(mean)

    @jax.jit
    def encode(params, obs):
        return encoder.apply(params, obs)

    eval_zs = _make_eval_latents(config)
    reward_vectors = _eval_reward_vectors(eval_zs, config)
    num_skills = int(eval_zs.shape[0])
    num_rollouts = int(config.get("NUM_EVAL_ROLLOUTS", 3))
    horizon = int(config.get("EVAL_EPISODE_LENGTH", config.get("EPISODE_LENGTH", 200)))
    action_repeat = int(config.get("ACTION_REPEAT", 1))
    root_x_index = int(config.get("ROOT_X_QPOS_INDEX", 0))
    root_z_index = int(config.get("ROOT_Z_QPOS_INDEX", 1))
    root_pitch_index = int(config.get("ROOT_PITCH_QPOS_INDEX", 2))
    base_key = jax.random.PRNGKey(int(config.get("EVAL_SEED", config.get("SEED", 0))))

    rows = []
    skill_mean_phi_diffs = []
    probe_obs = []

    for skill_i in range(num_skills):
        z = eval_zs[skill_i]
        rollouts = []
        all_phi_diffs = []

        for rollout_i in range(num_rollouts):
            # Same reset keys for every skill.
            state = reset(jax.random.fold_in(base_key, rollout_i))
            initial_x = _qpos(state, root_x_index)
            previous_x = initial_x
            xs = [initial_x]
            heights = [_qpos(state, root_z_index)]
            pitches = [_qpos(state, root_pitch_index)]
            velocities = []
            actions = []
            env_return = 0.0
            terminated = False
            phi_rewards = []

            previous_phi = None
            if phi_state is not None:
                previous_phi = encode(phi_state.params, get_metra_obs(state))

            for t in range(horizon):
                obs = get_metra_obs(state)
                action = act(actor_state.params, obs, z)
                actions.append(np.asarray(jax.device_get(action), dtype=np.float64))

                if rollout_i == 0 and t % max(horizon // 16, 1) == 0:
                    probe_obs.append(np.asarray(jax.device_get(obs), dtype=np.float32))

                for _ in range(action_repeat):
                    state = step(state, action)
                    env_return += float(jax.device_get(state.reward))

                    current_x = _qpos(state, root_x_index)
                    velocities.append((current_x - previous_x) / float(env.dt))
                    previous_x = current_x
                    xs.append(current_x)
                    heights.append(_qpos(state, root_z_index))
                    pitches.append(_qpos(state, root_pitch_index))

                    if bool(jax.device_get(state.done)):
                        terminated = True
                        break

                if phi_state is not None:
                    next_phi = encode(phi_state.params, get_metra_obs(state))
                    phi_diff = np.asarray(jax.device_get(next_phi - previous_phi), dtype=np.float64)
                    all_phi_diffs.append(phi_diff)
                    phi_rewards.append(float(phi_diff @ np.asarray(jax.device_get(reward_vectors[skill_i]))))
                    previous_phi = next_phi

                if terminated:
                    break

            xs = np.asarray(xs)
            velocities = np.asarray(velocities)
            actions = np.asarray(actions)
            heights = np.asarray(heights)
            pitches = np.asarray(pitches)

            rollouts.append({
                "final_x": xs[-1],
                "delta_x": xs[-1] - xs[0],
                "mean_vx": velocities.mean() if velocities.size else np.nan,
                "mean_abs_vx": np.abs(velocities).mean() if velocities.size else np.nan,
                "std_vx": velocities.std() if velocities.size else np.nan,
                "x_range": xs.max() - xs.min(),
                "x_path_length": np.abs(np.diff(xs)).sum(),
                "mean_height": heights.mean(),
                "std_height": heights.std(),
                "mean_abs_pitch": np.abs(pitches).mean(),
                "std_pitch": pitches.std(),
                "action_energy": np.square(actions).mean() if actions.size else np.nan,
                "mean_action_abs": np.abs(actions).mean() if actions.size else np.nan,
                "env_return": env_return,
                "termination": float(terminated),
                "mean_phi_reward": np.mean(phi_rewards) if phi_rewards else np.nan,
            })

        mean = lambda name: float(np.nanmean([r[name] for r in rollouts]))
        std = lambda name: float(np.nanstd([r[name] for r in rollouts]))
        row = {
            "skill": skill_i,
            "z": np.asarray(jax.device_get(z)),
            "final_x": mean("final_x"),
            "delta_x": mean("delta_x"),
            "delta_x_std": std("delta_x"),
            "mean_vx": mean("mean_vx"),
            "mean_abs_vx": mean("mean_abs_vx"),
            "std_vx": mean("std_vx"),
            "x_range": mean("x_range"),
            "x_path_length": mean("x_path_length"),
            "mean_height": mean("mean_height"),
            "std_height": mean("std_height"),
            "mean_abs_pitch": mean("mean_abs_pitch"),
            "std_pitch": mean("std_pitch"),
            "action_energy": mean("action_energy"),
            "mean_action_abs": mean("mean_action_abs"),
            "env_return": mean("env_return"),
            "termination_rate": mean("termination"),
            "mean_phi_reward": mean("mean_phi_reward"),
        }
        rows.append(row)
        skill_mean_phi_diffs.append(
            np.mean(all_phi_diffs, axis=0) if all_phi_diffs else np.zeros(config["Z_DIM"])
        )

        print(
            f"Skill {skill_i:02d} | dx={row['delta_x']:+.3f}±{row['delta_x_std']:.3f} "
            f"vx={row['mean_vx']:+.3f} | |vx|={row['mean_abs_vx']:.3f} "
            f"range={row['x_range']:.3f} | path={row['x_path_length']:.3f} "
            f"height={row['mean_height']:.3f}±{row['std_height']:.3f} "
            f"pitch={row['mean_abs_pitch']:.3f} | energy={row['action_energy']:.3f} "
            f"phi_r={row['mean_phi_reward']:+.3f}"
        )

    # Does each skill's trajectory align most with its own reward vector?
    phi_diff_matrix = np.stack(skill_mean_phi_diffs)
    reward_matrix = phi_diff_matrix @ np.asarray(jax.device_get(reward_vectors)).T
    diagonal = np.diag(reward_matrix)
    off_diagonal = (reward_matrix.sum(axis=1) - diagonal) / max(num_skills - 1, 1)
    margins = diagonal - off_diagonal
    for i, row in enumerate(rows):
        row["own_skill_reward"] = float(diagonal[i])
        row["own_skill_reward_margin"] = float(margins[i])

    # Hold observations fixed and vary only z: near-zero means actor ignores z.
    if probe_obs:
        obs = jnp.asarray(np.stack(probe_obs))
        obs_grid = jnp.repeat(obs[:, None, :], num_skills, axis=1)
        z_grid = jnp.repeat(eval_zs[None, :, :], obs.shape[0], axis=0)
        actions = act(
            actor_state.params,
            obs_grid.reshape((-1, obs.shape[-1])),
            z_grid.reshape((-1, eval_zs.shape[-1])),
        ).reshape((obs.shape[0], num_skills, env.action_size))
        counterfactual_action_std = float(jax.device_get(actions.std(axis=1).mean()))
    else:
        counterfactual_action_std = float("nan")

    summary = {
        "delta_x_range": float(np.ptp([r["delta_x"] for r in rows])),
        "mean_x_range": float(np.mean([r["x_range"] for r in rows])),
        "mean_x_path_length": float(np.mean([r["x_path_length"] for r in rows])),
        "counterfactual_action_std": counterfactual_action_std,
        "reward_matrix_diagonal": float(diagonal.mean()),
        "reward_matrix_off_diagonal": float(off_diagonal.mean()),
        "reward_matrix_margin": float(margins.mean()),
    }

    output_dir = Path(config.get("EVAL_VIDEO_DIR", "eval_videos"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "eval_skills.csv", rows, int(config["Z_DIM"]))
    np.save(output_dir / "eval_reward_matrix.npy", reward_matrix)
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "\nSkill-diversity summary"
        f"\n  dx range: {summary['delta_x_range']:.3f}"
        f"\n  mean x range: {summary['mean_x_range']:.3f}"
        f"\n  mean x path length: {summary['mean_x_path_length']:.3f}"
        f"\n  counterfactual action std: {counterfactual_action_std:.4f}"
        f"\n  reward-matrix margin: {summary['reward_matrix_margin']:.4f}"
    )
    return {"skills": rows, "summary": summary, "reward_matrix": reward_matrix}


def record_eval_videos(
    config,
    actor_state: TrainState,
    phi_state: TrainState | None = None,
) -> list[Path]:
    """Runs compact diagnostics, then records one deterministic video per skill."""
    if config.get("RUN_EVAL_DIAGNOSTICS", True):
        evaluate_skill_diagnostics(config, actor_state, phi_state)

    env_name, env = _make_raw_eval_env(config)
    actor = Actor(
        config["LAYER_SIZE"],
        env.action_size,
        config.get("LOG_STD_MIN", -5.0),
        config.get("LOG_STD_MAX", 2.0),
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    @jax.jit
    def act(params, obs, z):
        mean, _ = actor.apply(params, obs, z)
        return jnp.tanh(mean)

    eval_zs = _make_eval_latents(config)
    horizon = int(config.get("EVAL_EPISODE_LENGTH", config.get("EPISODE_LENGTH", 200)))
    action_repeat = int(config.get("ACTION_REPEAT", 1))
    root_x_index = int(config.get("ROOT_X_QPOS_INDEX", 0))
    output_dir = Path(config.get("EVAL_VIDEO_DIR", "eval_videos"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_key = jax.random.PRNGKey(int(config.get("EVAL_SEED", config.get("SEED", 0))))
    camera = config.get(
        "EVAL_CAMERA",
        "side" if env_name in {"WalkerWalk", "WalkerRun", "WalkerStand"} else None,
    )
    paths = []

    for skill_i, z in enumerate(eval_zs):
        # Shared reset makes videos directly comparable.
        state = reset(jax.random.fold_in(base_key, int(config.get("EVAL_VIDEO_RESET_INDEX", 0))))
        rollout = [state]
        initial_x = _qpos(state, root_x_index)
        env_return = 0.0

        for _ in range(horizon):
            action = act(actor_state.params, get_metra_obs(state), z)
            done = False
            for _ in range(action_repeat):
                state = step(state, action)
                env_return += float(jax.device_get(state.reward))
                done = bool(jax.device_get(state.done))
                if done:
                    break
            rollout.append(state)
            if done:
                break

        final_x = _qpos(state, root_x_index)
        kwargs = {
            "width": int(config.get("EVAL_WIDTH", 640)),
            "height": int(config.get("EVAL_HEIGHT", 480)),
        }
        if camera is not None:
            kwargs["camera"] = camera
        frames = env.render(rollout, **kwargs)
        path = output_dir / f"{env_name}_skill_{skill_i:02d}.mp4"
        media.write_video(str(path), frames, fps=1.0 / (float(env.dt) * action_repeat))
        print(
            f"Recorded skill {skill_i:02d}: return={env_return:.2f}, "
            f"dx={final_x - initial_x:+.3f}, path={path}"
        )
        paths.append(path)

    return paths
