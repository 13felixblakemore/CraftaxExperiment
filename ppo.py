# Used for reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
import sys
import time
from pickletools import uint8

import imageio
import jax
import numpy as np
import optax
import jax.numpy as jnp
import wandb
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn

import configs
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
    def __init__(self, env, env_params, optimiser, obs_shape=(8268,)):
        self.step = 0
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
            epsilon,
            ent_coef,
            vf_coef,
    ):
        def loss_fn(params, state_batch, action_batch, returns_batch, advantage_batch, env_params, old_log_probs, epsilon, ent_coef, vf_coef):
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
                (jnp.abs(ratio - 1.0) > epsilon).astype(jnp.float32)
            )

            entropy = -jnp.mean(
                jnp.sum(probs_all * log_probs_all, axis=-1)
            )

            advantage_batch = (advantage_batch - jnp.mean(advantage_batch)) / (jnp.std(advantage_batch) + 1e-8)

            #jax.debug.print("Ratio: {}", ratio)
            basic_loss = ratio * advantage_batch

            clipped_loss = (
                    jnp.clip(ratio, 1 - epsilon, 1 + epsilon)
                    * advantage_batch
            )

            actor_loss = -jnp.mean(jnp.minimum(basic_loss, clipped_loss))

            new_values = agent.get_state_value(params["critic"], state_batch)
            critic_loss = jnp.mean((new_values - returns_batch) ** 2)

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
            params, state_batch, action_batch, returns_batch, advantage_batch, env_params, old_log_probs, epsilon, ent_coef, vf_coef,
        )
        updates, new_opt_state = optimiser.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, info
    return train_step

def train(env, env_params, agent, obs_shape=(8268,)):
    config = wandb.config
    num_steps = config["num_steps"]
    actors = config["actors"]
    num_minibatches = config["num_minibatches"]
    total_timesteps = config["total_timesteps"]
    update_epochs = config["update_epochs"]

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

        rollout_mean_reward = jnp.mean(reward_seq)
        rollout_sum_reward = jnp.sum(reward_seq)
        rollout_mean_done = jnp.mean(done_seq)

        next_values = agent.get_state_value(agent.params["critic"], next_obs)
        advantages = jnp.zeros_like(reward_seq)
        last_lmbda = 0
        lmbda = wandb.config["lambda"]
        gamma = wandb.config["gamma"]
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones
                next_values = next_values
            else:
                next_non_terminal = 1.0 - done_seq[t + 1]
                next_values = value_seq[t + 1]
            next_values = next_values.squeeze()
            deltas = reward_seq[t] + gamma * next_values * next_non_terminal - value_seq[t]
            last_lmbda = deltas + gamma * lmbda * next_non_terminal * last_lmbda
            advantages = advantages.at[t].set(last_lmbda)
        returns = advantages + value_seq

        value_seq_flat_for_ev = value_seq.reshape(-1)
        returns_flat_for_ev = returns.reshape(-1)

        returns_var = jnp.var(returns_flat_for_ev)
        explained_var = jnp.where(
            returns_var == 0.0,
            0.0,
            1.0 - jnp.var(returns_flat_for_ev - value_seq_flat_for_ev) / returns_var
        )

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

                ent_coef = config["ent_coef"]
                vf_coef = config["vf_coef"]
                epsilon = config["epsilon"]

                new_params, new_opt_state, loss, info = agent.train_step(agent.params, agent.opt_state, state_batch, action_batch, returns_batch, advantage_batch, env_params, old_log_probs, epsilon, ent_coef, vf_coef)
                agent.params = new_params
                agent.opt_state = new_opt_state

                wandb.log({
                    "loss/total": loss,
                    "loss/actor": info["actor_loss"],
                    "loss/critic": info["critic_loss"],
                    "policy/entropy": info["entropy"],
                    "policy/clipfrac": info["clipfrac"],
                    "policy/approx_kl": info["approx_kl"],
                    "charts/total_steps": total_steps,
                })

        log_data = {
            "charts/total_steps": total_steps,
            "charts/iteration": iteration,

            "rollout/mean_reward": float(jax.device_get(rollout_mean_reward)),
            "rollout/sum_reward": float(jax.device_get(rollout_sum_reward)),
            "rollout/mean_done": float(jax.device_get(rollout_mean_done)),
            "value/explained_variance": float(jax.device_get(explained_var)),
            "value/mean_return": float(jax.device_get(jnp.mean(returns))),
            "value/mean_value": float(jax.device_get(jnp.mean(value_seq))),
            "value/mean_advantage": float(jax.device_get(jnp.mean(advantages))),
        }

        wandb.log(log_data, step=total_steps)


def test(
    agent,
    env,
    env_params,
    num_episodes=10,
    actors=2,
    max_episode_steps=500,
    deterministic=False,
    record_vid=False,
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

        if record_vid:
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

            if record_vid:
                frame = render_craftax_pixels(get_first_env_state(env_states), 64)
                frame = frame.astype(jnp.uint8)
                frames.append(np.asarray(jax.device_get(frame)))

            done_mask = new_done_mask
            t += 1

        if record_vid:
            imageio.mimsave("episode_ppo.gif", frames, fps=5)

        all_episode_steps.extend(list(jax.device_get(episode_steps)))
        all_episode_returns.extend(list(jax.device_get(episode_returns)))

    all_episode_steps = all_episode_steps[:num_episodes]
    all_episode_returns = all_episode_returns[:num_episodes]

    wandb.log({
        "eval/episode_steps": all_episode_steps,
        "eval/episode_returns": all_episode_returns,
        "eval/mean_episode_return": np.mean(all_episode_returns),
    })
    print("Episode steps:", all_episode_steps)
    print("Episode returns:", all_episode_returns)
    print("Mean steps:", np.mean(all_episode_steps))
    print("Mean return:", np.mean(all_episode_returns))

    return all_episode_steps, all_episode_returns


if __name__ == "__main__":
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    shape = (8268,)

    wandb.init(
        project="craftax",
        name="ppo-1",
        config=configs.debug_config
    )

    config = wandb.config

    batch_size = config["actors"] * config["num_steps"]
    num_updates = config["total_timesteps"] // batch_size
    num_optim_steps = num_updates * config["update_epochs"] * config["num_minibatches"]

    schedule = optax.linear_schedule(
        init_value=config["learning_rate"],
        end_value=config["end_learning_rate"],
        transition_steps=num_optim_steps
    )

    optimiser = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(schedule, eps=config["adam_epsilon"]),
    )

    agent = PPO(env, env_params, optimiser, obs_shape=shape)

    ft = time.perf_counter()
    train(env, env_params, agent, obs_shape=shape)
    nt = time.perf_counter()
    latency= nt - ft
    print("Training time: ", latency)
    test(agent, env, env_params, actors=1, num_episodes=1, deterministic=False)

    # test is very inefficient. fix