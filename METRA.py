from __future__ import annotations

from typing import NamedTuple

import flashbax as fbx
import gymnax
import jax
import jax.numpy as jnp
import optax
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn
from flax.training.train_state import TrainState
from mujoco_playground import wrapper
from mujoco_playground import registry

from logz.batch_logging import batch_log
from wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)


class Encoder(nn.Module):
    z_dim: int
    layer_size: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.layer_size)(state)
        x = nn.relu(x)
        x = nn.Dense(self.layer_size)(x)
        x = nn.relu(x)
        return nn.Dense(self.z_dim)(x)


class Actor(nn.Module):
    dim: int
    action_dim: int

    @nn.compact
    def __call__(self, obs, z):
        x = jnp.concatenate([obs, z], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class Critic(nn.Module):
    dim: int
    action_dim: int

    @nn.compact
    def __call__(self, obs, z):
        x = jnp.concatenate([obs, z], axis=-1)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    next_obs: jax.Array
    z: jax.Array
    done: jax.Array


def make_train(config):
    if config["ENV_NAME"] == "MountainCar-v0":
        env, env_params = gymnax.make("MountainCar-v0")
    else:
        env = make_craftax_env_from_name(
            config["ENV_NAME"],
            not config["USE_OPTIMISTIC_RESETS"],
        )
        env_params = env.default_params

    env_name = 'Go1JoystickFlatTerrain'
    env = registry.load(env_name)

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
        action_dim = env.action_space(env_params).n
        state_dim = env.observation_space(env_params).shape

        actor = Actor(config["LAYER_SIZE"], action_dim)
        q1 = Critic(config["LAYER_SIZE"], action_dim)
        q2 = Critic(config["LAYER_SIZE"], action_dim)
        phi = Encoder(config["Z_DIM"], config["LAYER_SIZE"])

        dummy_obs = jnp.zeros((1, *state_dim), dtype=jnp.float32)
        dummy_z = jnp.zeros((1, config["Z_DIM"]), dtype=jnp.float32)

        def sample_z(rng):
            z = jax.random.normal(rng, (config["NUM_ENVS"], config["Z_DIM"]))
            if config.get("UNIT_Z", True):
                z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
            return z

        rng, actor_key, q1_key, q2_key, phi_key = jax.random.split(rng, 5)

        actor_params = actor.init(actor_key, dummy_obs, dummy_z)
        q1_params = q1.init(q1_key, dummy_obs, dummy_z)
        q2_params = q2.init(q2_key, dummy_obs, dummy_z)
        phi_params = phi.init(phi_key, dummy_obs)

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=optax.adam(config["LR"]),
        )

        q1_state = TrainState.create(
            apply_fn=q1.apply,
            params=q1_params,
            tx=optax.adam(config["LR"]),
        )

        q2_state = TrainState.create(
            apply_fn=q2.apply,
            params=q2_params,
            tx=optax.adam(config["LR"]),
        )

        phi_state = TrainState.create(
            apply_fn=phi.apply,
            params=phi_params,
            tx=optax.adam(config["LR"]),
        )

        log_alpha_state = TrainState.create(
            apply_fn=lambda params: params["log_alpha"],
            params={
                "log_alpha": jnp.array(
                    jnp.log(config.get("ALPHA_INIT", 0.001)),
                    dtype=jnp.float32,
                )
            },
            tx=optax.adam(config["LR"]),
        )

        log_lambda_state = TrainState.create(
            apply_fn=lambda params: params["log_lambda"],
            params={
                "log_lambda": jnp.array(
                    jnp.log(config.get("LAMBDA_INIT", 30.0)),
                    dtype=jnp.float32,
                )
            },
            tx=optax.adam(config["LR"]),
        )

        target_q1_params = q1_params
        target_q2_params = q2_params

        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)

        rng, z_key = jax.random.split(rng)
        z = sample_z(z_key)

        buffer = fbx.make_item_buffer(
            config["BUFFER_CAPACITY"],
            config["WARMUP"],
            config["BATCH_SIZE"],
            add_sequences=False,
            add_batches=True,
        )

        single_obs = jax.tree.map(lambda x: x[0], obs)

        init_transition = Transition(
            obs=single_obs,
            action=jnp.zeros((), dtype=jnp.int32),
            next_obs=single_obs,
            z=jnp.zeros((config["Z_DIM"],), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.bool_),
        )

        buffer_state = buffer.init(init_transition)

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
                rng,
            ) = carry

            actor_state, q1_state, q2_state, phi_state = train_state

            def collect_rollout(carry, _):
                def step(carry, _):
                    obs, env_state, z, rng = carry

                    logits = actor_state.apply_fn(
                        actor_state.params,
                        obs,
                        z,
                    )

                    rng, action_key = jax.random.split(rng)
                    action = jax.random.categorical(action_key, logits, axis=-1)

                    rng, env_key = jax.random.split(rng)
                    next_obs, next_env_state, env_reward, done, info = env.step(
                        env_key,
                        env_state,
                        action,
                        env_params,
                    )

                    # Store the z that actually produced this transition.
                    transition = Transition(
                        obs=obs,
                        action=action,
                        next_obs=next_obs,
                        z=z,
                        done=done,
                    )

                    # Resample z only for envs whose episode ended.
                    rng, z_key = jax.random.split(rng)
                    new_z = sample_z(z_key)
                    z = jnp.where(done[:, None], new_z, z)

                    carry = next_obs, next_env_state, z, rng
                    return carry, (transition, info)

                obs, env_state, buffer_state, z, rng = carry

                state, (transitions, info) = jax.lax.scan(
                    step,
                    (obs, env_state, z, rng),
                    xs=None,
                    length=config["NUM_STEPS"],
                )

                obs, env_state, z, rng = state

                transitions = jax.tree.map(
                    lambda x: x.reshape((-1, *x.shape[2:])),
                    transitions,
                )

                buffer_state = buffer.add(buffer_state, transitions)

                return (obs, env_state, buffer_state, z, rng), info

            rollout_init = obs, env_state, buffer_state, z, rng

            rollout_carry, info = jax.lax.scan(
                collect_rollout,
                rollout_init,
                xs=None,
                length=config["NUM_TRAJECTORIES"],
            )

            obs, env_state, buffer_state, z, rng = rollout_carry

            returned = info["returned_episode"].astype(jnp.float32)
            num_episodes = returned.sum()

            episode_return_sum = (
                info["returned_episode_returns"] * returned
            ).sum()

            episode_length_sum = (
                info["returned_episode_lengths"] * returned
            ).sum()

            episode_return = jnp.where(
                num_episodes > 0,
                episode_return_sum / num_episodes,
                jnp.nan,
            )

            episode_length = jnp.where(
                num_episodes > 0,
                episode_length_sum / num_episodes,
                jnp.nan,
            )

            rollout_metric = {
                "episode_return": episode_return,
                "episode_length": episode_length,
                "num_episodes": num_episodes,
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

                actor_params = actor_state.params
                q1_params = q1_state.params
                q2_params = q2_state.params
                phi_params = phi_state.params

                log_lambda = log_lambda_state.params["log_lambda"]
                log_alpha = log_alpha_state.params["log_alpha"]

                rng, sample_key = jax.random.split(rng)
                transition = buffer.sample(buffer_state, sample_key).experience

                obs_b, action_b, next_obs_b, z_b, done_b = transition

                nonterminal = 1.0 - done_b.astype(jnp.float32)
                valid_denom = jnp.maximum(nonterminal.sum(), 1.0)

                def metra_components(phi_params):
                    phi_obs = phi.apply(phi_params, obs_b)
                    phi_next = phi.apply(phi_params, next_obs_b)
                    phi_diff = phi_next - phi_obs

                    raw_r = jnp.sum(phi_diff * z_b, axis=-1)
                    r = raw_r * nonterminal

                    sq_dist_unmasked = jnp.mean(jnp.square(phi_diff), axis=-1)
                    sq_dist = sq_dist_unmasked * nonterminal

                    cst_dist = jnp.ones_like(sq_dist_unmasked)
                    cst_penalty = cst_dist - sq_dist_unmasked
                    cst_penalty = jnp.minimum(
                        cst_penalty,
                        config["LAGRANGE_EPS"],
                    )
                    cst_penalty = cst_penalty * nonterminal

                    phi_delta_norm = jnp.linalg.norm(phi_diff, axis=-1)

                    return {
                        "phi_diff": phi_diff,
                        "raw_r": raw_r,
                        "r": r,
                        "sq_dist_unmasked": sq_dist_unmasked,
                        "sq_dist": sq_dist,
                        "cst_penalty": cst_penalty,
                        "phi_delta_norm": phi_delta_norm,
                    }

                def phi_loss_fn(phi_params):
                    comp = metra_components(phi_params)

                    lambda_ = jnp.exp(log_lambda)
                    objective = (
                        10 * comp["r"]
                        + jax.lax.stop_gradient(lambda_) * comp["cst_penalty"]
                    )

                    loss = -objective.sum() / valid_denom

                    aux = {
                        "metra_reward": comp["r"].sum() / valid_denom,
                        "raw_metra_reward": jnp.mean(comp["raw_r"]),
                        "abs_metra_reward": (
                            jnp.abs(comp["r"]).sum() / valid_denom
                        ),
                        "positive_metra_reward_frac": (
                            ((comp["r"] > 0).astype(jnp.float32) * nonterminal).sum()
                            / valid_denom
                        ),
                        "phi_sq_dist": comp["sq_dist"].sum() / valid_denom,
                        "phi_sq_dist_unmasked": jnp.mean(comp["sq_dist_unmasked"]),
                        "phi_delta_norm": (
                            (comp["phi_delta_norm"] * nonterminal).sum()
                            / valid_denom
                        ),
                        "cst_penalty": comp["cst_penalty"].sum() / valid_denom,
                    }

                    return loss, aux

                def lambda_loss_fn(log_lambda_params):
                    log_lambda = log_lambda_params["log_lambda"]

                    comp = metra_components(phi_state.params)
                    mean_cst_penalty = (
                        jax.lax.stop_gradient(comp["cst_penalty"]).sum()
                        / valid_denom
                    )

                    # Match PyTorch METRA: log_dual_lam * mean(cst_penalty.detach()).
                    loss = log_lambda * mean_cst_penalty

                    return loss

                def critic_loss_fn(q1_params, q2_params):
                    alpha = jnp.exp(log_alpha)
                    alpha_sg = jax.lax.stop_gradient(alpha)

                    q1_values = q1.apply(q1_params, obs_b, z_b)
                    q2_values = q2.apply(q2_params, obs_b, z_b)

                    batch_idx = jnp.arange(obs_b.shape[0])
                    q1_selected = q1_values[batch_idx, action_b]
                    q2_selected = q2_values[batch_idx, action_b]

                    next_action_logits = actor.apply(
                        actor_params,
                        next_obs_b,
                        z_b,
                    )

                    next_probs = jax.nn.softmax(next_action_logits, axis=-1)
                    next_log_probs = jax.nn.log_softmax(
                        next_action_logits,
                        axis=-1,
                    )

                    target_q1 = q1.apply(target_q1_params, next_obs_b, z_b)
                    target_q2 = q2.apply(target_q2_params, next_obs_b, z_b)
                    target_q = jnp.minimum(target_q1, target_q2)

                    next_v = jnp.sum(
                        next_probs * (target_q - alpha_sg * next_log_probs),
                        axis=-1,
                    )

                    comp = metra_components(phi_state.params)
                    intrinsic_r = jax.lax.stop_gradient(comp["r"])

                    target = intrinsic_r + config["GAMMA"] * nonterminal * next_v
                    target = jax.lax.stop_gradient(target)

                    q1_loss = 0.5 * jnp.mean(jnp.square(q1_selected - target))
                    q2_loss = 0.5 * jnp.mean(jnp.square(q2_selected - target))

                    aux = {
                        "q1_mean": jnp.mean(q1_selected),
                        "q2_mean": jnp.mean(q2_selected),
                        "target_q_mean": jnp.mean(target),
                        "q_abs_error": jnp.mean(jnp.abs(q1_selected - target)),
                    }

                    return q1_loss + q2_loss, aux

                def actor_loss_fn(actor_params):
                    logits = actor_state.apply_fn(actor_params, obs_b, z_b)
                    probs = jax.nn.softmax(logits, axis=-1)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)

                    entropy_per_state = -jnp.sum(probs * log_probs, axis=-1)
                    entropy = jnp.mean(entropy_per_state)

                    q1_values = q1_state.apply_fn(q1_state.params, obs_b, z_b)
                    q2_values = q2_state.apply_fn(q2_state.params, obs_b, z_b)
                    q_values = jax.lax.stop_gradient(
                        jnp.minimum(q1_values, q2_values)
                    )

                    expected_q = jnp.sum(probs * q_values, axis=-1)

                    loss = jnp.mean(
                        jnp.sum(
                            probs * (jnp.exp(log_alpha) * log_probs - q_values),
                            axis=-1,
                        )
                    )

                    aux = {
                        "entropy": entropy,
                        "expected_q": jnp.mean(expected_q),
                        "max_action_prob": jnp.mean(jnp.max(probs, axis=-1)),
                    }

                    return loss, aux

                def alpha_loss_fn(log_alpha_params):
                    log_alpha = log_alpha_params["log_alpha"]

                    logits = actor_state.apply_fn(actor_params, obs_b, z_b)
                    probs = jax.nn.softmax(logits, axis=-1)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)

                    entropy = -jnp.sum(probs * log_probs, axis=-1)

                    target_entropy = jnp.asarray(
                        config.get("TARGET_ENTROPY", 0.9),
                        dtype=jnp.float32,
                    )

                    loss = jnp.mean(
                        jnp.exp(log_alpha)
                        * jax.lax.stop_gradient(entropy - target_entropy)
                    )

                    return loss

                (phi_loss, phi_info), phi_grad = jax.value_and_grad(
                    phi_loss_fn,
                    has_aux=True,
                )(phi_params)

                phi_state = phi_state.apply_gradients(grads=phi_grad)

                lambda_loss, lambda_grad = jax.value_and_grad(lambda_loss_fn)(
                    log_lambda_state.params
                )

                log_lambda_state = log_lambda_state.apply_gradients(
                    grads=lambda_grad
                )

                (critic_loss, critic_info), critic_grad = jax.value_and_grad(
                    critic_loss_fn,
                    argnums=(0, 1),
                    has_aux=True,
                )(q1_params, q2_params)

                q1_grad, q2_grad = critic_grad
                q1_state = q1_state.apply_gradients(grads=q1_grad)
                q2_state = q2_state.apply_gradients(grads=q2_grad)

                (actor_loss, actor_info), actor_grad = jax.value_and_grad(
                    actor_loss_fn,
                    has_aux=True,
                )(actor_params)

                actor_state = actor_state.apply_gradients(grads=actor_grad)

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
                    "phi_sq_dist_unmasked": phi_info["phi_sq_dist_unmasked"],
                    "phi_delta_norm": phi_info["phi_delta_norm"],
                    "cst_penalty": phi_info["cst_penalty"],
                    "phi_loss": phi_loss,
                    "lambda_loss": lambda_loss,
                    "critic_loss": critic_loss,
                    "actor_loss": actor_loss,
                    "alpha_loss": alpha_loss,
                    "entropy": actor_info["entropy"],
                    "expected_q": actor_info["expected_q"],
                    "max_action_prob": actor_info["max_action_prob"],
                    "q1_mean": critic_info["q1_mean"],
                    "q2_mean": critic_info["q2_mean"],
                    "target_q_mean": critic_info["target_q_mean"],
                    "q_abs_error": critic_info["q_abs_error"],
                    "alpha": jnp.exp(log_alpha_state.params["log_alpha"]),
                    "lambda": jnp.exp(log_lambda_state.params["log_lambda"]),
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
                * config["NUM_ENVS"]
            )

            metric = {
                **rollout_metric,
                **update_metric,
            }

            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, global_step):
                    to_log = {}

                    for k, v in metric.items():
                        v = float(v)
                        if v == v:
                            to_log[k] = v

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
            rng,
        )

        iterations = config["TOTAL_TIMESTEPS"] // (
            config["NUM_TRAJECTORIES"]
            * config["NUM_STEPS"]
            * config["NUM_ENVS"]
        )

        carry, metric = jax.lax.scan(
            train_loop,
            init,
            xs=jnp.arange(iterations),
        )

        train_state, _, _, _, _, _, _, _, _, _ = carry
        return train_state

    return train