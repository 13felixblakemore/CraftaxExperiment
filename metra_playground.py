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
    """Returns a deterministic collection of skills for evaluation."""
    z_dim = int(config["Z_DIM"])

    if config.get("DISCRETE", False):
        return jnp.eye(z_dim, dtype=jnp.float32)

    if z_dim == 1:
        return jnp.array(
            [
                [-1.0],
                [1.0],
            ],
            dtype=jnp.float32,
        )

    if z_dim == 2:
        num_skills = int(config.get("NUM_EVAL_SKILLS", 8))

        angles = jnp.linspace(
            0.0,
            2.0 * jnp.pi,
            num_skills,
            endpoint=False,
        )

        return jnp.stack(
            [
                jnp.cos(angles),
                jnp.sin(angles),
            ],
            axis=-1,
        )

    num_skills = int(config.get("NUM_EVAL_SKILLS", 8))
    key = jax.random.PRNGKey(
        int(config.get("EVAL_SEED", 0))
    )

    z = jax.random.normal(
        key,
        shape=(num_skills, z_dim),
    )

    if config.get("UNIT_Z", True):
        z = z / (
            jnp.linalg.norm(z, axis=-1, keepdims=True)
            + 1e-8
        )

    return z.astype(jnp.float32)


def _make_raw_eval_env(config):
    """Creates the raw unbatched environment used for evaluation."""
    env_name = config.get("ENV_NAME", "WalkerWalk")
    env_config = registry.get_default_config(env_name)

    env_config_overrides = dict(
        config.get("ENV_CONFIG_OVERRIDES", {})
    )

    if "PLAYGROUND_IMPL" in config:
        env_config_overrides["impl"] = config["PLAYGROUND_IMPL"]

    eval_env = registry.load(
        env_name,
        config=env_config,
        config_overrides=env_config_overrides or None,
    )

    return env_name, eval_env


def _eval_reward_vector(
    z: jax.Array,
    config,
) -> jax.Array:
    """Returns the skill vector used by the METRA reward."""
    if not config.get("DISCRETE", False):
        return z

    num_skills = int(config["Z_DIM"])

    return (
        z - jnp.mean(z, axis=-1, keepdims=True)
    ) * num_skills / max(num_skills - 1, 1)


def _read_qpos(
    state,
    index: int | None,
) -> float:
    """Safely reads one qpos coordinate."""
    if index is None:
        return float("nan")

    qpos = np.asarray(
        jax.device_get(state.data.qpos)
    ).reshape(-1)

    if index < 0 or index >= qpos.size:
        return float("nan")

    return float(qpos[index])


def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(values.mean())


def _finite_std(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(values.std())


def _safe_correlation(
    x,
    y,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2:
        return float("nan")

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def _mean_pairwise_l2(
    vectors: np.ndarray,
) -> float:
    """Mean pairwise Euclidean distance between rows."""
    vectors = np.asarray(vectors, dtype=np.float64)

    if vectors.ndim != 2 or vectors.shape[0] < 2:
        return 0.0

    distances = []

    for first in range(vectors.shape[0]):
        for second in range(first + 1, vectors.shape[0]):
            distances.append(
                np.linalg.norm(
                    vectors[first] - vectors[second]
                )
            )

    return float(np.mean(distances))


def _json_serialisable(value):
    if isinstance(value, dict):
        return {
            key: _json_serialisable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_serialisable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.floating, np.integer)):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


def _write_eval_csv(
    path: Path,
    rows: list[dict],
    z_dim: int,
) -> None:
    """Writes rows while expanding z into z_0, z_1, ..."""
    if not rows:
        return

    expanded_rows = []

    for row in rows:
        expanded = {
            key: value
            for key, value in row.items()
            if key != "z"
        }

        z = np.asarray(row["z"]).reshape(-1)

        for dimension in range(z_dim):
            expanded[f"z_{dimension}"] = float(
                z[dimension]
            )

        expanded_rows.append(expanded)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(expanded_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(expanded_rows)


def evaluate_skill_diagnostics(
    config,
    actor_state: TrainState,
    phi_state: TrainState | None = None,
) -> dict:
    """Evaluates whether z produces reliably different behaviour.

    Every skill is evaluated from exactly the same collection of reset
    seeds. This prevents initial-state variation from being mistaken for
    skill-dependent behaviour.
    """
    env_name, eval_env = _make_raw_eval_env(config)

    actor = Actor(
        dim=config["LAYER_SIZE"],
        action_dim=eval_env.action_size,
        log_std_min=config.get("LOG_STD_MIN", -5.0),
        log_std_max=config.get("LOG_STD_MAX", 2.0),
    )

    encoder = None

    if phi_state is not None:
        encoder = Encoder(
            config["Z_DIM"],
            config["LAYER_SIZE"],
        )

    reset_fn = jax.jit(eval_env.reset)
    step_fn = jax.jit(eval_env.step)

    @jax.jit
    def policy_stats(
        actor_params,
        obs: jax.Array,
        z: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        mean, log_std = actor.apply(
            actor_params,
            obs,
            z,
        )

        deterministic_action = jnp.tanh(mean)
        policy_std = jnp.exp(log_std)

        return deterministic_action, policy_std

    if encoder is not None:
        @jax.jit
        def encode(
            phi_params,
            obs: jax.Array,
        ) -> jax.Array:
            return encoder.apply(phi_params, obs)
    else:
        encode = None

    eval_zs = _make_eval_latents(config)
    eval_zs_host = np.asarray(
        jax.device_get(eval_zs),
        dtype=np.float64,
    )

    num_skills = int(eval_zs.shape[0])

    num_rollouts = int(
        config.get("NUM_EVAL_ROLLOUTS", 5)
    )

    episode_length = int(
        config.get(
            "EVAL_EPISODE_LENGTH",
            config.get("EPISODE_LENGTH", 200),
        )
    )

    action_repeat = int(
        config.get("ACTION_REPEAT", 1)
    )

    direction_threshold = float(
        config.get("EVAL_DIRECTION_THRESHOLD", 0.25)
    )

    root_x_index = config.get(
        "ROOT_X_QPOS_INDEX",
        0,
    )
    root_z_index = config.get(
        "ROOT_Z_QPOS_INDEX",
        1,
    )
    root_pitch_index = config.get(
        "ROOT_PITCH_QPOS_INDEX",
        2,
    )

    max_probe_states = int(
        config.get("EVAL_ACTION_PROBE_STATES", 128)
    )
    probe_stride = max(
        1,
        int(config.get("EVAL_ACTION_PROBE_STRIDE", 10)),
    )
    probes_per_skill = max(
        1,
        max_probe_states // max(num_skills, 1),
    )

    base_key = jax.random.PRNGKey(
        int(
            config.get(
                "EVAL_SEED",
                config.get("SEED", 0),
            )
        )
    )

    rollout_rows: list[dict] = []
    skill_rows: list[dict] = []
    probe_observations: list[np.ndarray] = []

    for skill_index in range(num_skills):
        z = eval_zs[skill_index]
        z_host = eval_zs_host[skill_index]

        reward_vector = _eval_reward_vector(
            z,
            config,
        )

        this_skill_rollouts = []
        collected_skill_probes = 0

        for rollout_index in range(num_rollouts):
            # Do not include skill_index here. All skills receive
            # exactly the same reset keys.
            reset_key = jax.random.fold_in(
                base_key,
                rollout_index,
            )

            state = reset_fn(reset_key)

            initial_obs = get_metra_obs(state)

            initial_x = _read_qpos(
                state,
                root_x_index,
            )
            initial_root_z = _read_qpos(
                state,
                root_z_index,
            )
            initial_pitch = _read_qpos(
                state,
                root_pitch_index,
            )

            if encode is not None:
                initial_phi = encode(
                    phi_state.params,
                    initial_obs,
                )
                previous_phi = initial_phi
            else:
                initial_phi = None
                previous_phi = None

            environment_return = 0.0
            cumulative_phi_reward = 0.0

            action_abs_sum = 0.0
            action_saturation_sum = 0.0
            policy_std_sum = 0.0

            policy_steps = 0
            physics_steps = 0
            terminated = False

            for policy_step in range(episode_length):
                obs = get_metra_obs(state)

                if (
                    rollout_index == 0
                    and collected_skill_probes < probes_per_skill
                    and policy_step % probe_stride == 0
                ):
                    probe_observations.append(
                        np.asarray(
                            jax.device_get(obs),
                            dtype=np.float32,
                        )
                    )
                    collected_skill_probes += 1

                action, policy_std = policy_stats(
                    actor_state.params,
                    obs,
                    z,
                )

                action_host = np.asarray(
                    jax.device_get(action),
                    dtype=np.float64,
                )
                std_host = np.asarray(
                    jax.device_get(policy_std),
                    dtype=np.float64,
                )

                action_abs_sum += float(
                    np.mean(np.abs(action_host))
                )
                action_saturation_sum += float(
                    np.mean(np.abs(action_host) > 0.95)
                )
                policy_std_sum += float(
                    np.mean(std_host)
                )
                policy_steps += 1

                for _ in range(action_repeat):
                    state = step_fn(state, action)
                    physics_steps += 1

                    environment_return += float(
                        jax.device_get(state.reward)
                    )

                    if bool(jax.device_get(state.done)):
                        terminated = True
                        break

                if encode is not None:
                    next_obs = get_metra_obs(state)

                    next_phi = encode(
                        phi_state.params,
                        next_obs,
                    )

                    phi_diff = next_phi - previous_phi

                    cumulative_phi_reward += float(
                        jax.device_get(
                            jnp.sum(
                                phi_diff * reward_vector
                            )
                        )
                    )

                    previous_phi = next_phi

                if terminated:
                    break

            final_obs = get_metra_obs(state)

            final_x = _read_qpos(
                state,
                root_x_index,
            )
            final_root_z = _read_qpos(
                state,
                root_z_index,
            )
            final_pitch = _read_qpos(
                state,
                root_pitch_index,
            )

            delta_x = final_x - initial_x
            delta_root_z = final_root_z - initial_root_z
            delta_pitch = final_pitch - initial_pitch

            elapsed_seconds = (
                physics_steps * float(eval_env.dt)
            )

            mean_x_velocity = (
                delta_x / elapsed_seconds
                if elapsed_seconds > 0.0
                and np.isfinite(delta_x)
                else float("nan")
            )

            rollout_row = {
                "skill_index": skill_index,
                "rollout_index": rollout_index,
                "z": z_host,
                "initial_x": initial_x,
                "final_x": final_x,
                "delta_x": delta_x,
                "initial_root_z": initial_root_z,
                "final_root_z": final_root_z,
                "delta_root_z": delta_root_z,
                "initial_pitch": initial_pitch,
                "final_pitch": final_pitch,
                "delta_pitch": delta_pitch,
                "mean_x_velocity": mean_x_velocity,
                "env_return": environment_return,
                "policy_steps": policy_steps,
                "physics_steps": physics_steps,
                "terminated": float(terminated),
                "mean_action_abs": (
                    action_abs_sum / max(policy_steps, 1)
                ),
                "action_saturation_frac": (
                    action_saturation_sum
                    / max(policy_steps, 1)
                ),
                "mean_policy_std": (
                    policy_std_sum / max(policy_steps, 1)
                ),
            }

            if encode is not None:
                final_phi = encode(
                    phi_state.params,
                    final_obs,
                )

                endpoint_phi_delta = (
                    final_phi - initial_phi
                )

                endpoint_phi_reward = float(
                    jax.device_get(
                        jnp.sum(
                            endpoint_phi_delta
                            * reward_vector
                        )
                    )
                )

                endpoint_phi_delta_norm = float(
                    jax.device_get(
                        jnp.linalg.norm(
                            endpoint_phi_delta
                        )
                    )
                )

                reward_vector_norm = float(
                    jax.device_get(
                        jnp.linalg.norm(
                            reward_vector
                        )
                    )
                )

                endpoint_phi_cosine = (
                    endpoint_phi_reward
                    / (
                        endpoint_phi_delta_norm
                        * reward_vector_norm
                        + 1e-8
                    )
                )

                rollout_row.update(
                    {
                        "endpoint_phi_reward":
                            endpoint_phi_reward,
                        "cumulative_phi_reward":
                            cumulative_phi_reward,
                        "phi_telescoping_abs_error":
                            abs(
                                endpoint_phi_reward
                                - cumulative_phi_reward
                            ),
                        "endpoint_phi_delta_norm":
                            endpoint_phi_delta_norm,
                        "endpoint_phi_sq_dist":
                            float(
                                jax.device_get(
                                    jnp.mean(
                                        jnp.square(
                                            endpoint_phi_delta
                                        )
                                    )
                                )
                            ),
                        "endpoint_phi_z_cosine":
                            endpoint_phi_cosine,
                    }
                )

            rollout_rows.append(rollout_row)
            this_skill_rollouts.append(rollout_row)

        delta_x_values = np.asarray(
            [
                row["delta_x"]
                for row in this_skill_rollouts
            ],
            dtype=np.float64,
        )

        forward_fraction = float(
            np.mean(
                delta_x_values > direction_threshold
            )
        )
        backward_fraction = float(
            np.mean(
                delta_x_values < -direction_threshold
            )
        )
        stationary_fraction = float(
            np.mean(
                np.abs(delta_x_values)
                <= direction_threshold
            )
        )

        direction_consistency = max(
            forward_fraction,
            backward_fraction,
            stationary_fraction,
        )

        skill_row = {
            "skill_index": skill_index,
            "z": z_host,
            "mean_final_x": _finite_mean(
                [
                    row["final_x"]
                    for row in this_skill_rollouts
                ]
            ),
            "std_final_x": _finite_std(
                [
                    row["final_x"]
                    for row in this_skill_rollouts
                ]
            ),
            "mean_delta_x": _finite_mean(
                delta_x_values
            ),
            "std_delta_x": _finite_std(
                delta_x_values
            ),
            "mean_abs_delta_x": _finite_mean(
                np.abs(delta_x_values)
            ),
            "mean_x_velocity": _finite_mean(
                [
                    row["mean_x_velocity"]
                    for row in this_skill_rollouts
                ]
            ),
            "mean_final_root_z": _finite_mean(
                [
                    row["final_root_z"]
                    for row in this_skill_rollouts
                ]
            ),
            "std_final_root_z": _finite_std(
                [
                    row["final_root_z"]
                    for row in this_skill_rollouts
                ]
            ),
            "mean_final_pitch": _finite_mean(
                [
                    row["final_pitch"]
                    for row in this_skill_rollouts
                ]
            ),
            "std_final_pitch": _finite_std(
                [
                    row["final_pitch"]
                    for row in this_skill_rollouts
                ]
            ),
            "mean_env_return": _finite_mean(
                [
                    row["env_return"]
                    for row in this_skill_rollouts
                ]
            ),
            "std_env_return": _finite_std(
                [
                    row["env_return"]
                    for row in this_skill_rollouts
                ]
            ),
            "mean_policy_steps": _finite_mean(
                [
                    row["policy_steps"]
                    for row in this_skill_rollouts
                ]
            ),
            "termination_rate": _finite_mean(
                [
                    row["terminated"]
                    for row in this_skill_rollouts
                ]
            ),
            "forward_frac": forward_fraction,
            "backward_frac": backward_fraction,
            "stationary_frac": stationary_fraction,
            "direction_consistency":
                direction_consistency,
            "mean_action_abs": _finite_mean(
                [
                    row["mean_action_abs"]
                    for row in this_skill_rollouts
                ]
            ),
            "action_saturation_frac": _finite_mean(
                [
                    row["action_saturation_frac"]
                    for row in this_skill_rollouts
                ]
            ),
            "mean_policy_std": _finite_mean(
                [
                    row["mean_policy_std"]
                    for row in this_skill_rollouts
                ]
            ),
        }

        if encode is not None:
            skill_row.update(
                {
                    "mean_endpoint_phi_reward":
                        _finite_mean(
                            [
                                row["endpoint_phi_reward"]
                                for row
                                in this_skill_rollouts
                            ]
                        ),
                    "std_endpoint_phi_reward":
                        _finite_std(
                            [
                                row["endpoint_phi_reward"]
                                for row
                                in this_skill_rollouts
                            ]
                        ),
                    "mean_endpoint_phi_z_cosine":
                        _finite_mean(
                            [
                                row["endpoint_phi_z_cosine"]
                                for row
                                in this_skill_rollouts
                            ]
                        ),
                    "mean_endpoint_phi_delta_norm":
                        _finite_mean(
                            [
                                row["endpoint_phi_delta_norm"]
                                for row
                                in this_skill_rollouts
                            ]
                        ),
                    "mean_endpoint_phi_sq_dist":
                        _finite_mean(
                            [
                                row["endpoint_phi_sq_dist"]
                                for row
                                in this_skill_rollouts
                            ]
                        ),
                    "mean_phi_telescoping_abs_error":
                        _finite_mean(
                            [
                                row["phi_telescoping_abs_error"]
                                for row
                                in this_skill_rollouts
                            ]
                        ),
                }
            )

        skill_rows.append(skill_row)

        print(
            f"Skill {skill_index:02d} "
            f"z={np.array2string(z_host, precision=3)} | "
            f"final_x={skill_row['mean_final_x']:+.3f} "
            f"± {skill_row['std_final_x']:.3f} | "
            f"dx={skill_row['mean_delta_x']:+.3f} "
            f"± {skill_row['std_delta_x']:.3f} | "
            f"vx={skill_row['mean_x_velocity']:+.3f} | "
            f"return={skill_row['mean_env_return']:.2f} | "
            f"term={skill_row['termination_rate']:.2f} | "
            f"consistency="
            f"{skill_row['direction_consistency']:.2f}"
        )

    mean_delta_x_values = np.asarray(
        [
            row["mean_delta_x"]
            for row in skill_rows
        ],
        dtype=np.float64,
    )

    within_skill_std_values = np.asarray(
        [
            row["std_delta_x"]
            for row in skill_rows
        ],
        dtype=np.float64,
    )

    between_skill_std = _finite_std(
        mean_delta_x_values
    )
    mean_within_skill_std = _finite_mean(
        within_skill_std_values
    )

    displacement_separation_ratio = (
        between_skill_std
        / (mean_within_skill_std + 1e-8)
    )

    pairwise_displacement_differences = []

    for first in range(num_skills):
        for second in range(first + 1, num_skills):
            pairwise_displacement_differences.append(
                abs(
                    mean_delta_x_values[first]
                    - mean_delta_x_values[second]
                )
            )

    summary = {
        "environment": env_name,
        "num_skills": num_skills,
        "num_rollouts_per_skill": num_rollouts,
        "between_skill_delta_x_std":
            between_skill_std,
        "mean_within_skill_delta_x_std":
            mean_within_skill_std,
        "displacement_separation_ratio":
            displacement_separation_ratio,
        "mean_pairwise_delta_x_difference":
            _finite_mean(
                pairwise_displacement_differences
            ),
        "delta_x_range":
            float(
                np.nanmax(mean_delta_x_values)
                - np.nanmin(mean_delta_x_values)
            ),
        "num_forward_skills":
            int(
                np.sum(
                    mean_delta_x_values
                    > direction_threshold
                )
            ),
        "num_backward_skills":
            int(
                np.sum(
                    mean_delta_x_values
                    < -direction_threshold
                )
            ),
        "num_stationary_skills":
            int(
                np.sum(
                    np.abs(mean_delta_x_values)
                    <= direction_threshold
                )
            ),
        "mean_direction_consistency":
            _finite_mean(
                [
                    row["direction_consistency"]
                    for row in skill_rows
                ]
            ),
        "mean_skill_env_return":
            _finite_mean(
                [
                    row["mean_env_return"]
                    for row in skill_rows
                ]
            ),
        "mean_termination_rate":
            _finite_mean(
                [
                    row["termination_rate"]
                    for row in skill_rows
                ]
            ),
    }

    z_delta_x_correlations = [
        _safe_correlation(
            eval_zs_host[:, dimension],
            mean_delta_x_values,
        )
        for dimension in range(
            eval_zs_host.shape[1]
        )
    ]

    summary["z_dim_delta_x_correlations"] = (
        z_delta_x_correlations
    )

    finite_correlations = np.asarray(
        [
            value
            for value in z_delta_x_correlations
            if np.isfinite(value)
        ],
        dtype=np.float64,
    )

    summary["max_abs_z_delta_x_correlation"] = (
        float(
            np.max(
                np.abs(finite_correlations)
            )
        )
        if finite_correlations.size > 0
        else float("nan")
    )

    # Linear regression from z to mean horizontal displacement.
    if (
        num_skills > eval_zs_host.shape[1] + 1
        and np.std(mean_delta_x_values) > 1e-12
    ):
        design_matrix = np.concatenate(
            [
                eval_zs_host,
                np.ones(
                    (num_skills, 1),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )

        coefficients, *_ = np.linalg.lstsq(
            design_matrix,
            mean_delta_x_values,
            rcond=None,
        )

        predicted_delta_x = (
            design_matrix @ coefficients
        )

        residual_sum_squares = float(
            np.sum(
                np.square(
                    mean_delta_x_values
                    - predicted_delta_x
                )
            )
        )

        total_sum_squares = float(
            np.sum(
                np.square(
                    mean_delta_x_values
                    - mean_delta_x_values.mean()
                )
            )
        )

        summary["linear_z_to_delta_x_r2"] = (
            1.0
            - residual_sum_squares
            / (total_sum_squares + 1e-12)
        )
    else:
        summary["linear_z_to_delta_x_r2"] = (
            float("nan")
        )

    # Counterfactual test:
    # hold observation fixed and change only z.
    if probe_observations:
        probe_obs = jnp.asarray(
            np.stack(probe_observations),
            dtype=jnp.float32,
        )

        num_probe_states = int(
            probe_obs.shape[0]
        )

        observation_grid = jnp.broadcast_to(
            probe_obs[:, None, :],
            (
                num_probe_states,
                num_skills,
                probe_obs.shape[-1],
            ),
        )

        skill_grid = jnp.broadcast_to(
            eval_zs[None, :, :],
            (
                num_probe_states,
                num_skills,
                eval_zs.shape[-1],
            ),
        )

        action_grid, _ = policy_stats(
            actor_state.params,
            observation_grid.reshape(
                (-1, observation_grid.shape[-1])
            ),
            skill_grid.reshape(
                (-1, skill_grid.shape[-1])
            ),
        )

        action_grid = np.asarray(
            jax.device_get(
                action_grid.reshape(
                    (
                        num_probe_states,
                        num_skills,
                        eval_env.action_size,
                    )
                )
            ),
            dtype=np.float64,
        )

        summary["num_action_probe_states"] = (
            num_probe_states
        )

        summary[
            "counterfactual_action_std_across_skills"
        ] = float(
            action_grid.std(axis=1).mean()
        )

        summary[
            "counterfactual_action_range_across_skills"
        ] = float(
            (
                action_grid.max(axis=1)
                - action_grid.min(axis=1)
            ).mean()
        )

        summary[
            "counterfactual_action_pairwise_l2"
        ] = _finite_mean(
            [
                _mean_pairwise_l2(actions)
                for actions in action_grid
            ]
        )
    else:
        summary["num_action_probe_states"] = 0
        summary[
            "counterfactual_action_std_across_skills"
        ] = float("nan")
        summary[
            "counterfactual_action_range_across_skills"
        ] = float("nan")
        summary[
            "counterfactual_action_pairwise_l2"
        ] = float("nan")

    if encode is not None:
        summary.update(
            {
                "mean_endpoint_phi_reward":
                    _finite_mean(
                        [
                            row["mean_endpoint_phi_reward"]
                            for row in skill_rows
                        ]
                    ),
                "mean_endpoint_phi_z_cosine":
                    _finite_mean(
                        [
                            row["mean_endpoint_phi_z_cosine"]
                            for row in skill_rows
                        ]
                    ),
                "mean_endpoint_phi_delta_norm":
                    _finite_mean(
                        [
                            row[
                                "mean_endpoint_phi_delta_norm"
                            ]
                            for row in skill_rows
                        ]
                    ),
                "mean_phi_telescoping_abs_error":
                    _finite_mean(
                        [
                            row[
                                "mean_phi_telescoping_abs_error"
                            ]
                            for row in skill_rows
                        ]
                    ),
            }
        )

    output_dir = Path(
        config.get(
            "EVAL_DIAGNOSTICS_DIR",
            config.get(
                "EVAL_VIDEO_DIR",
                "eval_videos",
            ),
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rollout_csv_path = (
        output_dir / "eval_rollouts.csv"
    )
    skills_csv_path = (
        output_dir / "eval_skills.csv"
    )
    summary_json_path = (
        output_dir / "eval_summary.json"
    )

    _write_eval_csv(
        rollout_csv_path,
        rollout_rows,
        int(config["Z_DIM"]),
    )

    _write_eval_csv(
        skills_csv_path,
        skill_rows,
        int(config["Z_DIM"]),
    )

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            _json_serialisable(
                {
                    "summary": summary,
                    "per_skill": skill_rows,
                }
            ),
            file,
            indent=2,
        )

    print(
        "\nSkill-diversity summary"
        f"\n  between-skill std(dx): "
        f"{between_skill_std:.3f}"
        f"\n  mean within-skill std(dx): "
        f"{mean_within_skill_std:.3f}"
        f"\n  displacement separation ratio: "
        f"{displacement_separation_ratio:.3f}"
        f"\n  mean pairwise displacement difference: "
        f"{summary['mean_pairwise_delta_x_difference']:.3f}"
        f"\n  counterfactual action std: "
        f"{summary['counterfactual_action_std_across_skills']:.4f}"
        f"\n  counterfactual pairwise action L2: "
        f"{summary['counterfactual_action_pairwise_l2']:.4f}"
        f"\n  linear z -> dx R^2: "
        f"{summary['linear_z_to_delta_x_r2']:.3f}"
        f"\n  files: "
        f"{skills_csv_path}, "
        f"{rollout_csv_path}, "
        f"{summary_json_path}"
    )

    return {
        "summary": summary,
        "per_skill": skill_rows,
        "rollouts": rollout_rows,
        "paths": {
            "skills_csv": skills_csv_path,
            "rollouts_csv": rollout_csv_path,
            "summary_json": summary_json_path,
        },
    }


def record_eval_videos(
    config,
    actor_state: TrainState,
    phi_state: TrainState | None = None,
) -> list[Path]:
    """Records one deterministic video per skill.

    Multi-seed diagnostic evaluation runs first unless
    RUN_EVAL_DIAGNOSTICS is False.
    """
    if config.get("RUN_EVAL_DIAGNOSTICS", True):
        evaluate_skill_diagnostics(
            config,
            actor_state,
            phi_state,
        )

    env_name, eval_env = _make_raw_eval_env(config)

    actor = Actor(
        dim=config["LAYER_SIZE"],
        action_dim=eval_env.action_size,
        log_std_min=config.get("LOG_STD_MIN", -5.0),
        log_std_max=config.get("LOG_STD_MAX", 2.0),
    )

    encoder = None

    if phi_state is not None:
        encoder = Encoder(
            config["Z_DIM"],
            config["LAYER_SIZE"],
        )

    reset_fn = jax.jit(eval_env.reset)
    step_fn = jax.jit(eval_env.step)

    @jax.jit
    def deterministic_action(
        actor_params,
        obs: jax.Array,
        z: jax.Array,
    ) -> jax.Array:
        mean, _ = actor.apply(
            actor_params,
            obs,
            z,
        )

        return jnp.tanh(mean)

    if encoder is not None:
        @jax.jit
        def encode(
            phi_params,
            obs: jax.Array,
        ) -> jax.Array:
            return encoder.apply(phi_params, obs)
    else:
        encode = None

    eval_zs = _make_eval_latents(config)

    episode_length = int(
        config.get(
            "EVAL_EPISODE_LENGTH",
            config.get("EPISODE_LENGTH", 200),
        )
    )

    action_repeat = int(
        config.get("ACTION_REPEAT", 1)
    )

    root_x_index = config.get(
        "ROOT_X_QPOS_INDEX",
        0,
    )

    camera = config.get(
        "EVAL_CAMERA",
        (
            "side"
            if env_name in {
                "WalkerWalk",
                "WalkerRun",
                "WalkerStand",
            }
            else None
        ),
    )

    width = int(
        config.get("EVAL_WIDTH", 640)
    )
    height = int(
        config.get("EVAL_HEIGHT", 480)
    )

    output_dir = Path(
        config.get(
            "EVAL_VIDEO_DIR",
            "eval_videos",
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_key = jax.random.PRNGKey(
        int(
            config.get(
                "EVAL_SEED",
                config.get("SEED", 0),
            )
        )
    )

    shared_reset = bool(
        config.get(
            "EVAL_VIDEO_SHARED_RESET",
            True,
        )
    )

    video_reset_index = int(
        config.get(
            "EVAL_VIDEO_RESET_INDEX",
            0,
        )
    )

    output_paths = []

    for skill_index in range(eval_zs.shape[0]):
        z = eval_zs[skill_index]

        reward_vector = _eval_reward_vector(
            z,
            config,
        )

        reset_index = (
            video_reset_index
            if shared_reset
            else skill_index
        )

        reset_key = jax.random.fold_in(
            base_key,
            reset_index,
        )

        state = reset_fn(reset_key)

        rollout = [state]
        environment_return = 0.0

        initial_x = _read_qpos(
            state,
            root_x_index,
        )

        initial_obs = get_metra_obs(state)

        initial_phi = (
            encode(
                phi_state.params,
                initial_obs,
            )
            if encode is not None
            else None
        )

        for _ in range(episode_length):
            obs = get_metra_obs(state)

            action = deterministic_action(
                actor_state.params,
                obs,
                z,
            )

            terminated = False

            for _ in range(action_repeat):
                state = step_fn(state, action)

                environment_return += float(
                    jax.device_get(state.reward)
                )

                if bool(jax.device_get(state.done)):
                    terminated = True
                    break

            rollout.append(state)

            if terminated:
                break

        final_x = _read_qpos(
            state,
            root_x_index,
        )
        delta_x = final_x - initial_x

        endpoint_phi_reward = float("nan")
        endpoint_phi_cosine = float("nan")

        if encode is not None:
            final_obs = get_metra_obs(state)

            final_phi = encode(
                phi_state.params,
                final_obs,
            )

            endpoint_phi_delta = (
                final_phi - initial_phi
            )

            endpoint_phi_reward = float(
                jax.device_get(
                    jnp.sum(
                        endpoint_phi_delta
                        * reward_vector
                    )
                )
            )

            endpoint_phi_cosine = (
                endpoint_phi_reward
                / (
                    float(
                        jax.device_get(
                            jnp.linalg.norm(
                                endpoint_phi_delta
                            )
                        )
                    )
                    * float(
                        jax.device_get(
                            jnp.linalg.norm(
                                reward_vector
                            )
                        )
                    )
                    + 1e-8
                )
            )

        render_kwargs = {
            "width": width,
            "height": height,
        }

        if camera is not None:
            render_kwargs["camera"] = camera

        frames = eval_env.render(
            rollout,
            **render_kwargs,
        )

        video_path = output_dir / (
            f"{env_name}_skill_"
            f"{skill_index:02d}.mp4"
        )

        fps = 1.0 / (
            float(eval_env.dt)
            * action_repeat
        )

        media.write_video(
            str(video_path),
            frames,
            fps=fps,
        )

        z_host = np.asarray(
            jax.device_get(z)
        )

        phi_text = ""

        if encode is not None:
            phi_text = (
                f", endpoint_phi_reward="
                f"{endpoint_phi_reward:+.3f}, "
                f"phi_z_cos="
                f"{endpoint_phi_cosine:+.3f}"
            )

        print(
            f"Recorded skill {skill_index:02d}: "
            f"z={np.array2string(z_host, precision=3)}, "
            f"return={environment_return:.2f}, "
            f"final_x={final_x:+.3f}, "
            f"delta_x={delta_x:+.3f}, "
            f"steps={len(rollout) - 1}"
            f"{phi_text}, "
            f"path={video_path}"
        )

        output_paths.append(video_path)

    return output_paths