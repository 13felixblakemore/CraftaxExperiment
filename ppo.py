# Used for reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
import sys
import time
from pickletools import uint8

import imageio
import jax
import numpy as np
import optax
import jax.numpy as jnp
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn

from test_environment import ChainEnv


class Actor(nn.Module):
    @nn.compact
    def __call__(self, x, env, env_params):
        x = nn.Dense(256,
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=nn.initializers.constant(0.0),
                    )(x)
        x = nn.tanh(x)

        x = nn.Dense(256,
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=nn.initializers.constant(0.0),
                    )(x)
        x = nn.tanh(x)

        x = nn.Dense(env.action_space(env_params).n,
                    kernel_init=nn.initializers.orthogonal(0.01),
                    bias_init=nn.initializers.constant(0.0),
                    )(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256,
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=nn.initializers.constant(0.0),
                     )(x)
        x = nn.tanh(x)

        x = nn.Dense(256,
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=nn.initializers.constant(0.0),
                     )(x)
        x = nn.tanh(x)

        x = nn.Dense(1,
                     kernel_init=nn.initializers.orthogonal(1.0),
                     bias_init=nn.initializers.constant(0.0)
                     )(x)
        return x



class PPO:
    def __init__(self, env, env_params, optimiser, gamma=0.99, lmbda=0.95, epsilon=0.2, obs_shape=(8268,)):
        self.step = 0
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.actor = Actor()
        self.critic = Critic()

        key = jax.random.PRNGKey(0)
        key, critic_key, actor_key = jax.random.split(key, 3)
        dummy = np.zeros(obs_shape)
        self.critic_params = self.critic.init(critic_key, dummy)
        self.actor_params = self.actor.init(actor_key, dummy, env, env_params)
        self.params = {
            "critic": self.critic_params,
            "actor": self.actor_params,
        }

        self.optimiser = optimiser
        self.opt_state = self.optimiser.init(self.params)

        self.train_step = make_training_step(self, env, self.optimiser)
        self.get_state_value = make_get_state_value_jit(self)
        self.choose_action = choose_action_jit(self, env)

def make_get_state_value_jit(agent):
    @jax.jit
    def get_state_value(critic_params, state_batch):
        return agent.critic.apply(critic_params, state_batch)
    return get_state_value

def choose_action_jit(agent, env):
    @jax.jit
    def choose_action(actor_params, state_batch, env_params, policy_keys):
        logits = agent.actor.apply(actor_params, state_batch, env, env_params)
        sample_action = jax.vmap(jax.random.categorical, in_axes=(0, 0))

        actions = sample_action(policy_keys, logits)
        get_action_log_probs = get_log_probs_jit(agent, env)
        log_probs, _ = get_action_log_probs(actor_params, state_batch, actions, env_params)
        return actions, log_probs
    return choose_action


def get_log_probs_jit(agent, env):
    @jax.jit
    def get_action_log_probs(actor_params, obs, actions, env_params):
        logits = agent.actor.apply(actor_params, obs, env, env_params)

        log_probs_all = jax.nn.log_softmax(logits, axis=-1)

        action_log_probs = jnp.take_along_axis(
            log_probs_all,
            actions.astype(jnp.int32)[:, None],
            axis=1,
        ).squeeze()

        return action_log_probs, log_probs_all
    return get_action_log_probs


def make_training_step(agent, env, optimiser):
    @jax.jit
    def train_step(
            params,
            opt_state,
            state_batch,
            action_batch,
            returns_batch,
            advantage_batch,
            env_params,
            old_log_probs,
    ):
        def loss_fn(params, state_batch, action_batch, returns_batch, advantage_batch, env_params, old_log_probs):
            logits = agent.actor.apply(params["actor"], state_batch, env, env_params)
            log_probs_all = jax.nn.log_softmax(logits, axis=-1)

            new_log_probs = jnp.take_along_axis(
                log_probs_all,
                action_batch.astype(jnp.int32)[:, None],
                axis=1,
            ).squeeze()
            probs_all = jnp.exp(log_probs_all)

            log_ratio = new_log_probs - old_log_probs
            ratio = jnp.exp(log_ratio)

            approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
            old_approx_kl = -jnp.mean(log_ratio)

            clipfrac = jnp.mean(
                (jnp.abs(ratio - 1.0) > agent.epsilon).astype(jnp.float32)
            )

            entropy = -jnp.mean(
                jnp.sum(probs_all * log_probs_all, axis=-1)
            )

            advantage_batch = (advantage_batch - jnp.mean(advantage_batch)) / (jnp.std(advantage_batch) + 1e-8)

            #jax.debug.print("Ratio: {}", ratio)
            basic_loss = ratio * advantage_batch

            clipped_loss = (
                    jnp.clip(ratio, 1 - agent.epsilon, 1 + agent.epsilon)
                    * advantage_batch
            )

            actor_loss = -jnp.mean(jnp.minimum(basic_loss, clipped_loss))

            new_values = agent.get_state_value(params["critic"], state_batch)
            critic_loss = jnp.mean((new_values - returns_batch) ** 2)

            vf_coef = 0.5
            ent_coef = 0.01

            loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy
            return loss, {
                            "actor_loss": actor_loss,
                            "critic_loss": critic_loss,
                            "entropy": entropy,
                            "approx_kl": approx_kl,
                            "old_approx_kl": old_approx_kl,
                            "clipfrac": clipfrac,
                        }

        (loss, info), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state_batch, action_batch, returns_batch, advantage_batch, env_params, old_log_probs
        )
        updates, new_opt_state = optimiser.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return train_step

def train(env, env_params, agent, schedule, optimiser, num_steps=5, total_timesteps=100, actors=2, num_minibatches=2, update_epochs=2, obs_shape=(8268,)):
    total_steps = 0
    batch_size = num_steps * actors # n parallel envs, with m step rollout each
    minibatch_size = batch_size // num_minibatches # split total rollout into minibatches
    num_iterations = total_timesteps // batch_size

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, actors)

    batched_reset = jax.vmap(env.reset, in_axes=(0, None))
    obs, env_states = batched_reset(keys, env_params)

    batched_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    state_seq = jnp.zeros((num_steps, actors, *obs_shape))
    action_seq = jnp.zeros((num_steps, actors))
    reward_seq = jnp.zeros((num_steps, actors))
    done_seq = jnp.zeros((num_steps, actors))
    value_seq = jnp.zeros((num_steps, actors))
    log_prob_seq = jnp.zeros((num_steps, actors))

    for iteration in range(num_iterations):
        print(f"Iteration: {iteration}/{num_iterations}")
        for step in range(num_steps):
            key, key_act, key_step = jax.random.split(key, 3)
            key_step = jax.random.split(key_step, actors)
            key_act = jax.random.split(key_act, actors)

            values = agent.get_state_value(agent.params["critic"], obs)
            values = values.squeeze()

            actions, log_probs = agent.choose_action(
                agent.params["actor"], obs, env_params, key_act
            )
            next_obs, next_env_states, rewards, dones, infos = batched_step(
                key_step, env_states, actions, env_params
            )

            state_seq = state_seq.at[step].set(obs)
            action_seq = action_seq.at[step].set(actions)
            reward_seq = reward_seq.at[step].set(rewards)
            value_seq = value_seq.at[step].set(values)
            done_seq = done_seq.at[step].set(dones)
            log_prob_seq = log_prob_seq.at[step].set(log_probs)

            total_steps += actors

            if total_steps % 100 == 0:
                print(f"Step {total_steps}")

            obs = next_obs
            env_states = next_env_states

        next_values = agent.get_state_value(agent.params["critic"], next_obs)
        advantages = jnp.zeros_like(reward_seq)
        last_lmbda = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones
                next_values = next_values
            else:
                next_non_terminal = 1.0 - done_seq[t + 1]
                next_values = value_seq[t + 1]
            next_values = next_values.squeeze()
            deltas = reward_seq[t] + agent.gamma * next_values * next_non_terminal - value_seq[t]
            last_lmbda = deltas + agent.gamma * agent.lmbda * next_non_terminal * last_lmbda
            advantages = advantages.at[t].set(last_lmbda)
        returns = advantages + value_seq

        key, key_act = jax.random.split(key, 2)
        keys_act = jax.random.split(key_act, minibatch_size)

        state_seq_flat = state_seq.reshape(-1, *obs_shape)
        action_seq_flat = action_seq.reshape(-1)
        value_seq_flat = value_seq.reshape(-1)
        reward_seq_flat = reward_seq.reshape(-1)
        done_seq_flat = done_seq.reshape(-1)
        log_prob_seq_flat = log_prob_seq.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)

        for epoch in range(update_epochs):
            key, key_idx = jax.random.split(key)
            b_idx = jax.random.permutation(key_idx, jnp.arange(batch_size))
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = b_idx[start:end]

                old_log_probs = log_prob_seq_flat[mb_idx]
                state_batch = state_seq_flat[mb_idx]
                action_batch = action_seq_flat[mb_idx]
                returns_batch = returns_flat[mb_idx]
                advantage_batch = advantages_flat[mb_idx]

                new_params, new_opt_state, loss = agent.train_step(agent.params, agent.opt_state, state_batch, action_batch, returns_batch, advantage_batch, env_params, old_log_probs)
                agent.params = new_params
                agent.opt_state = new_opt_state


def test(
    agent,
    env,
    env_params,
    num_episodes=10,
    actors=2,
    max_episode_steps=500,
    deterministic=False,
):
    key = jax.random.PRNGKey(1)

    batched_reset = jax.vmap(env.reset, in_axes=(0, None))
    batched_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    all_episode_steps = []
    all_episode_returns = []
    frames = []

    num_batches = int(np.ceil(num_episodes / actors))

    for batch in range(num_batches):
        key, key_reset = jax.random.split(key)
        reset_keys = jax.random.split(key_reset, actors)

        def get_first_env_state(env_states):
            return jax.tree_util.tree_map(lambda x: x[0], env_states)

        obs, env_states = batched_reset(reset_keys, env_params)

        frame = render_craftax_pixels(get_first_env_state(env_states), 64)
        frame = frame.astype(jnp.uint8)
        frames.append(np.asarray(jax.device_get(frame)))

        done_mask = jnp.zeros((actors,), dtype=bool)

        t = 0

        episode_steps = jnp.zeros((actors,), dtype=int)
        episode_returns = jnp.zeros((actors,), dtype=float)

        while (not bool(jax.device_get(jnp.all(done_mask)))) and t < max_episode_steps:
            key, key_act, key_step = jax.random.split(key, 3)

            step_keys = jax.random.split(key_step, actors)
            act_keys = jax.random.split(key_act, actors)

            if deterministic:
                logits = agent.actor.apply(
                    agent.params["actor"],
                    obs,
                    env,
                    env_params,
                )
                actions = jnp.argmax(logits, axis=-1)
            else:
                actions, _ = agent.choose_action(
                    agent.params["actor"],
                    obs,
                    env_params,
                    act_keys,
                )

            next_obs, next_env_states, rewards, dones, infos = batched_step(
                step_keys,
                env_states,
                actions,
                env_params,
            )

            active = ~done_mask

            episode_steps = episode_steps + active.astype(jnp.int32)
            episode_returns = episode_returns + jnp.where(active, rewards, 0.0)

            new_done_mask = done_mask | dones

            obs_mask = active.reshape((active.shape[0],) + (1,) * (obs.ndim - 1))
            obs = jnp.where(obs_mask, next_obs, obs)

            def mask_done(old, new):
                mask = active.reshape((active.shape[0],) + (1,) * (old.ndim - 1))
                return jnp.where(mask, new, old)

            env_states = jax.tree_util.tree_map(
                mask_done,
                env_states,
                next_env_states,
            )

            frame = render_craftax_pixels(get_first_env_state(env_states), 64)
            frame = frame.astype(jnp.uint8)
            frames.append(np.asarray(jax.device_get(frame)))

            done_mask = new_done_mask
            t += 1

        imageio.mimsave("episode_ppo.gif", frames, fps=5)

        all_episode_steps.extend(list(jax.device_get(episode_steps)))
        all_episode_returns.extend(list(jax.device_get(episode_returns)))

    all_episode_steps = all_episode_steps[:num_episodes]
    all_episode_returns = all_episode_returns[:num_episodes]

    print("Episode steps:", all_episode_steps)
    print("Episode returns:", all_episode_returns)
    print("Mean steps:", np.mean(all_episode_steps))
    print("Mean return:", np.mean(all_episode_returns))

    return all_episode_steps, all_episode_returns

env_name = "craft"
if env_name == "chain":
    length = 10
    env = ChainEnv(length=length, max_steps=1000)
    env_params = env.default_params
    shape = length
elif env_name == "craft":
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    shape = (8268,)

actors = 16
num_steps = 64
num_minibatches = 8
update_epochs = 4
total_timesteps = 500_000

lr = 2.5e-4
gamma = 0.99
lmbda = 0.8
epsilon = 0.2
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
batch_size = actors * num_steps
num_updates = total_timesteps // batch_size
num_optim_steps = num_updates * update_epochs * num_minibatches

schedule = optax.linear_schedule(
    init_value=2.5e-4,
    end_value=0.0,
    transition_steps=num_optim_steps,
)

optimiser = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(schedule, eps=1e-5),
)
agent = PPO(env, env_params, optimiser, obs_shape=shape)

ft = time.perf_counter()
train(env, env_params, agent, schedule, optimiser, update_epochs=update_epochs, actors=actors, total_timesteps=total_timesteps, num_steps=num_steps, num_minibatches=num_minibatches, obs_shape=shape)
nt = time.perf_counter()
latency= nt - ft
print("Latency: ", latency)
test(agent, env, env_params, actors=1, num_episodes=1, deterministic=False)