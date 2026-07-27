from __future__ import annotations
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from craftax.craftax_env import make_craftax_env_from_name
from wrappers import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper, OptimisticResetVecEnvWrapper
import flashbax as fbx
from typing import NamedTuple
import gymnax


class Encoder(nn.Module):
    z_dim: int
    layer_size: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.layer_size)(state)
        x = nn.relu(x)
        x = nn.Dense(self.layer_size)(x)
        x = nn.relu(x)

        x = nn.Dense(self.z_dim)(x)
        return x

class Actor(nn.Module):
    dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x, z):
        x = nn.Dense(self.dim)(jnp.concatenate([x, z], axis=-1))
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class Critic(nn.Module):
    dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x, z):
        x = nn.Dense(self.dim)(jnp.concatenate([x, z], axis=-1))
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
    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = env.default_params
    env, env_params = gymnax.make("MountainCar-v0")
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

        dummy_obs = jnp.zeros(
            (1, *state_dim),
            dtype=jnp.float32,
        )        

        def sample_z(rng):
            z = jax.random.normal(rng, (config["NUM_ENVS"], config["Z_DIM"]))
            if config.get("UNIT_Z", True):
                z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
            return z
        
        rng, _rng = jax.random.split(rng)
        dummy_z = jnp.zeros((1, config["Z_DIM"]), dtype=jnp.float32)

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
            tx=optax.adam(config["LR"])
        )

        log_alpha_state = TrainState.create(
            apply_fn=lambda params: params,
            params=jnp.array(jnp.log(config.get("ALPHA_INIT", 0.01))),
            tx=optax.adam(config["LR"]),
        )

        log_lambda_state = TrainState.create(
            apply_fn=lambda params: params,
            params=jnp.array(0.0),
            tx=optax.adam(config["LR"]),
        )

        target_q1_params = q1_params
        target_q2_params = q2_params

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)

        buffer = fbx.make_item_buffer(config["BUFFER_CAPACITY"], config["WARMUP"], config["BATCH_SIZE"], add_sequences=False, add_batches=True)

        single_obs = jax.tree.map(lambda x: x[0], obs)

        init_transition = Transition(
            obs=single_obs,
            action=jnp.zeros((), dtype=jnp.int32),
            next_obs=single_obs,
            z=jnp.zeros((config["Z_DIM"],), dtype=jnp.float32), 
            done=jnp.zeros((), dtype=jnp.bool_),
        )

        buffer_state = buffer.init(init_transition)

        def train_loop(carry, _):
            def collect_rollout(carry, _):

                def step(carry, _):
                    obs, env_state, z, rng = carry


                    logits = actor_state.apply_fn(
                        actor_state.params,
                        obs,
                        z,
                    )

                    rng, _rng = jax.random.split(rng)
                    action = jax.random.categorical(_rng, logits, axis=-1)

                    rng, _rng = jax.random.split(rng)
                    next_obs, next_env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

                    transition = Transition(obs, action, next_obs, z, done)

                    carry = next_obs, next_env_state, z, rng
                    return carry, transition

                obs, env_state, buffer_state, rng = carry

                rng, _rng = jax.random.split(rng)
                z = sample_z(rng)
                rng, _rng = jax.random.split(rng)
                init = obs, env_state, z, _rng
                state, transitions = jax.lax.scan(step, init, xs=None, length=config["NUM_STEPS"])

                obs, env_state, z, _rng = state

                transitions = jax.tree.map(
                    lambda x: x.reshape((-1, *x.shape[2:])),
                    transitions,
                )

                buffer_state = buffer.add(buffer_state, transitions)
                carry = obs, env_state, buffer_state, rng
                return carry, _

            train_state, target_q1_params, target_q2_params, log_lambda_state, log_alpha_state, obs, env_state, buffer_state, rng = carry
            actor_state, q1_state, q2_state, phi_state = train_state
            print("pre rollout")
            init = obs, env_state, buffer_state, rng
            carry, _ = jax.lax.scan(collect_rollout, init, xs=None, length=config["NUM_TRAJECTORIES"])
            
            print("past rollout")

            obs, env_state, buffer_state, rng = carry

            def update(carry, _):
                actor_state, q1_state, q2_state, target_q1_params, target_q2_params, log_lambda_state, log_alpha_state, phi_state, rng = carry
                actor_params = actor_state.params
                q1_params = q1_state.params
                q2_params = q2_state.params
                phi_params = phi_state.params

                log_lambda = log_lambda_state.params
                log_alpha = log_alpha_state.params
            
                rng, _rng = jax.random.split(rng)
                transition = buffer.sample(buffer_state, _rng).experience

                obs, action, next_obs, z, done = transition

                def phi_loss(phi_params):
                    phi_diff = phi.apply(phi_params, next_obs) - phi.apply(phi_params, obs)
                    abs_sq_diff = abs(phi_diff) ** 2
                    r = jnp.sum(phi_diff * z, axis=-1)

                    eps = config["LAGRANGE_EPS"]
                    lipschitz = config.get("LIPSCHITZ_CONSTRAINT", 1)
                    penalty = jnp.minimum(eps, (lipschitz - abs_sq_diff)).sum(axis=-1)
                    phi_loss = -r + jnp.exp(log_lambda) * penalty
                    return jnp.mean(phi_loss)

                def lambda_loss(log_lambda):
                    phi_diff = phi.apply(phi_params, next_obs) - phi.apply(phi_params, obs)
                    abs_sq_diff = abs(phi_diff) ** 2

                    eps = config["LAGRANGE_EPS"]
                    lipschitz = config.get("LIPSCHITZ_CONSTRAINT", 1)
                    penalty = jnp.minimum(eps, (lipschitz - abs_sq_diff)).sum(axis=-1)
                    lambda_loss = jnp.exp(log_lambda) * penalty
                    return jnp.mean(lambda_loss)

                def critic_loss(q1_params, q2_params):
                    alpha = jnp.exp(log_alpha)
                    alpha_sg = jax.lax.stop_gradient(alpha)

                    q1_values = q1.apply(q1_params, obs, z)
                    q2_values = q2.apply(q2_params, obs, z)

                    batch_idx = jnp.arange(obs.shape[0])

                    q1_selected = q1_values[batch_idx, action]
                    q2_selected = q2_values[batch_idx, action]

                    next_action_logits = actor.apply(
                        actor_params,
                        next_obs,
                        z, 
                    )
                    next_probs = jax.nn.softmax(next_action_logits, axis=-1)
                    next_log_probs = jax.nn.log_softmax(next_action_logits, axis=-1)

                    target_q1 = q1.apply(
                        target_q1_params,
                        next_obs,
                        z, 
                    )
                    target_q2 = q2.apply(
                        target_q2_params,
                        next_obs,
                        z, 
                    )
                    target_q = jnp.minimum(target_q1, target_q2)

                    next_v = jnp.sum(
                        next_probs * (target_q - alpha_sg * next_log_probs),
                        axis=-1,
                    )

                    phi_diff = phi.apply(phi_params, next_obs) - phi.apply(phi_params, obs)
                    r = jnp.sum(phi_diff * z, axis=-1)

                    target = r + config["GAMMA"] * (
                        1.0 - done
                    ) * next_v

                    target = jax.lax.stop_gradient(target)

                    q1_loss = 0.5 * jnp.mean((q1_selected - target) ** 2)
                    q2_loss = 0.5 * jnp.mean((q2_selected - target) ** 2)

                    return q1_loss + q2_loss

                def actor_loss(actor_params):
                    logits = actor_state.apply_fn(actor_params, obs, z)
                    probs = jax.nn.softmax(logits)
                    log_probs = jax.nn.log_softmax(logits)

                    q = q1_state.apply_fn(q1_state.params, obs, z)
                    q = jax.lax.stop_gradient(q)

                    loss = jnp.sum(probs * (jnp.exp(log_alpha) * log_probs - q), axis=-1).mean()
                    return loss

                def alpha_loss(log_alpha):
                    logits = actor_state.apply_fn(actor_params, obs, z)
                    probs = jax.nn.softmax(logits)
                    log_probs = jax.nn.log_softmax(logits)
                    entropy = -jnp.sum(probs * log_probs, axis=-1)

                    target_entropy = jnp.asarray(
                        config.get("TARGET_ENTROPY", 1.0),
                        dtype=jnp.float32,
                    )

                    alpha_loss = jnp.mean(
                        jnp.exp(log_alpha_state) * jax.lax.stop_gradient(entropy - target_entropy)
                    )

                    return alpha_loss

                phi_loss, grad = jax.value_and_grad(phi_loss)(phi_params)
                phi_state = phi_state.apply_gradients(grads=grad)

                lambda_loss, grad = jax.value_and_grad(lambda_loss)(log_lambda)
                log_lambda_state = log_lambda_state.apply_gradients(grads=grad)

                critic_loss, grad = jax.value_and_grad(critic_loss, argnums=(0,1))(q1_params, q2_params)
                grad1, grad2 = grad
                q1_state = q1_state.apply_gradients(grads=grad1)
                q2_state = q2_state.apply_gradients(grads=grad2)

                actor_loss, grad = jax.value_and_grad(actor_loss)(actor_params)
                actor_state = actor_state.apply_gradients(grads=grad)

                alpha_loss, grad = jax.value_and_grad(alpha_loss)(log_alpha)
                log_alpha_state = log_alpha_state.apply_gradients(grads=grad)

                carry = actor_state, q1_state, q2_state, target_q1_params, target_q2_params, log_lambda_state, log_alpha_state, phi_state, rng

                return carry, _

            init = actor_state, q1_state, q2_state, target_q1_params, target_q2_params, log_lambda_state, log_alpha_state, phi_state, rng
            carry, _ = jax.lax.scan(update, init, xs=None, length=config["NUM_UPDATE_STEPS"])
            return carry, _
        
        train_state = actor_state, q1_state, q2_state, phi_state

        init = train_state, target_q1_params, target_q2_params, log_lambda_state, log_alpha_state, obs, env_state, buffer_state, rng
        iterations = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] * config["NUM_TRAJECTORIES"]
        carry, _ = jax.lax.scan(train_loop, init, xs=None, length=iterations)
        train_state, _, _, _, _, _, _, _, _, _ = carry
        return train_state
    
    return train