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

        self.act_logs_values = make_act_logprob_value_jit(self, env)
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


def make_act_logprob_value_jit(agent, env):
    @jax.jit
    def act_logprob_value(params, obs, env_params, policy_keys):
        logits = agent.actor.apply(params["actor"], obs, env, env_params)
        values = agent.critic.apply(params["critic"], obs).reshape(-1)

        actions = jax.vmap(jax.random.categorical, in_axes=(0, 0))(
            policy_keys,
            logits,
        )

        log_probs_all = jax.nn.log_softmax(logits, axis=-1)
        log_probs = jnp.take_along_axis(
            log_probs_all,
            actions.astype(jnp.int32)[:, None],
            axis=1,
        ).reshape(-1)

        return actions, log_probs, values

    return act_logprob_value


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

            basic_loss = ratio * advantage_batch

            clipped_loss = (
                    jnp.clip(ratio, 1 - epsilon, 1 + epsilon)
                    * advantage_batch
            )

            actor_loss = -jnp.mean(jnp.minimum(basic_loss, clipped_loss))

            new_values = agent.get_state_value(params["critic"], state_batch).reshape(-1)
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



def make_update_rollout_jit(
        agent,
        batch_size,
        minibatch_size,
        num_minibatches,
        update_epochs,
):
    @jax.jit
    def update_rollout(
            params,
            opt_state,
            flat_batch,
            key,
            env_params,
            epsilon,
            ent_coef,
            vf_coef,
    ):
        def update_epoch(carry, _):
            params, opt_state, key = carry
            key, perm_key = jax.random.split(key)

            permutation = jax.random.permutation(
                perm_key,
                jnp.arange(batch_size),
            )

            minibatch_indices = permutation.reshape(
                (num_minibatches, minibatch_size)
            )

            def update_minibatch(carry, mb_idx):
                params, opt_state = carry

                state_batch = flat_batch["states"][mb_idx]
                action_batch = flat_batch["actions"][mb_idx]
                returns_batch = flat_batch["returns"][mb_idx]
                advantage_batch = flat_batch["advantages"][mb_idx]
                old_log_probs = flat_batch["log_probs"][mb_idx]

                new_params, new_opt_state, loss, info = agent.train_step(
                    params, opt_state, state_batch,
                    action_batch, returns_batch, advantage_batch,
                    env_params, old_log_probs, epsilon, ent_coef,
                    vf_coef)

                new_carry = (new_params, new_opt_state)

                metrics = {
                    "loss_total": loss,
                    "loss_actor": info["actor_loss"],
                    "loss_critic": info["critic_loss"],
                    "entropy": info["entropy"],
                    "approx_kl": info["approx_kl"],
                    "clipfrac": info["clipfrac"],
                }

                return new_carry, metrics

            (params, opt_state), minibatch_metrics = jax.lax.scan(
                update_minibatch,
                (params, opt_state),
                minibatch_indices,
            )

            return (params, opt_state, key), minibatch_metrics

        (params, opt_state, key), epoch_metrics = jax.lax.scan(
            update_epoch,
        (params, opt_state, key),
            xs=None,
            length=update_epochs,
        )

        mean_metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x),
            epoch_metrics,
        )

        return params, opt_state, key, mean_metrics
    return update_rollout

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

    @jax.jit
    def collect_rollout(params, obs, env_states, key, env_params):
        def rollout_step(carry, _):
            obs, env_states, key = carry

            key, key_act, key_step = jax.random.split(key, 3)
            act_keys = jax.random.split(key_act, actors)
            step_keys = jax.random.split(key_step, actors)

            actions, log_probs, values = agent.act_logs_values(
                params,
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

            transition = {
                "states": obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "values": values,
                "log_probs": log_probs,
            }

            new_carry = (next_obs, next_env_states, key)
            return new_carry, transition

        carry = (obs, env_states, key)
        carry, rollout = jax.lax.scan(
            rollout_step,
            carry,
            xs=None,
            length=num_steps,
        )

        next_obs, next_env_states, key = carry
        return next_obs, next_env_states, key, rollout

    update_rollout = make_update_rollout_jit(
        agent,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
    )

    for iteration in range(num_iterations):
        print(f"Iteration: {iteration}/{num_iterations}")

        obs, env_states, key, rollout = collect_rollout(
            agent.params,
            obs,
            env_states,
            key,
            env_params,
        )

        total_steps += batch_size

        state_seq = rollout["states"]
        action_seq = rollout["actions"]
        reward_seq = rollout["rewards"]
        done_seq = rollout["dones"]
        value_seq = rollout["values"]
        log_prob_seq = rollout["log_probs"]

        rollout_mean_reward = jnp.mean(reward_seq)
        rollout_sum_reward = jnp.sum(reward_seq)
        rollout_mean_done = jnp.mean(done_seq)

        next_values = agent.get_state_value(agent.params["critic"], obs).reshape(-1)

        advantages = jnp.zeros_like(reward_seq)
        last_lmbda = jnp.zeros((actors,))

        lmbda = config["lambda"]
        gamma = config["gamma"]

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = next_values
            else:
                next_value = value_seq[t + 1]

            next_non_terminal = 1.0 - done_seq[t]

            delta = reward_seq[t] + gamma * next_value * next_non_terminal - value_seq[t]
            last_lmbda = delta + gamma * lmbda * next_non_terminal * last_lmbda
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

        ent_coef = config["ent_coef"]
        vf_coef = config["vf_coef"]
        epsilon = config["epsilon"]

        flat_batch = {
            "states": state_seq_flat,
            "actions": action_seq_flat,
            "returns": returns_flat,
            "advantages": advantages_flat,
            "log_probs": log_prob_seq_flat,
        }

        params, opt_state, key, mean_metrics = update_rollout(
            agent.params,
            agent.opt_state,
            flat_batch,
            key,
            env_params,
            epsilon,
            ent_coef,
            vf_coef,)

        agent.params = params
        agent.opt_state = opt_state

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
    max_episode_steps=10000,
    deterministic=False,
    record_vid=False,
    seed=1,
    wandb_step=None,
):
    """
    Faster JAX evaluation.

    Same core functionality as your current test:
    - runs num_episodes evaluation episodes
    - uses `actors` parallel envs per batch
    - stops accumulating return/steps after each env is done
    - returns all_episode_steps, all_episode_returns
    - logs eval metrics to W&B

    Difference:
    - uses lax.scan instead of a Python while loop
    - runs for fixed max_episode_steps, masking finished envs
    - video recording is kept separate and optional
    """

    batched_reset = jax.vmap(env.reset, in_axes=(0, None))
    batched_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    @jax.jit
    def eval_batch(params, key):
        key, key_reset = jax.random.split(key)
        reset_keys = jax.random.split(key_reset, actors)

        obs, env_states = batched_reset(reset_keys, env_params)

        done_mask = jnp.zeros((actors,), dtype=bool)
        episode_steps = jnp.zeros((actors,), dtype=jnp.int32)
        episode_returns = jnp.zeros((actors,), dtype=jnp.float32)

        def scan_step(carry, _):
            obs, env_states, done_mask, episode_steps, episode_returns, key = carry

            key, key_act, key_step = jax.random.split(key, 3)
            act_keys = jax.random.split(key_act, actors)
            step_keys = jax.random.split(key_step, actors)

            logits = agent.actor.apply(
                params["actor"],
                obs,
                env,
                env_params,
            )

            if deterministic:
                actions = jnp.argmax(logits, axis=-1)
            else:
                actions = jax.vmap(jax.random.categorical, in_axes=(0, 0))(
                    act_keys,
                    logits,
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

            # Freeze finished envs so they do not keep changing after first done.
            obs_mask = active.reshape((actors,) + (1,) * (obs.ndim - 1))
            obs = jnp.where(obs_mask, next_obs, obs)

            def mask_done(old, new):
                mask = active.reshape((actors,) + (1,) * (old.ndim - 1))
                return jnp.where(mask, new, old)

            env_states = jax.tree_util.tree_map(
                mask_done,
                env_states,
                next_env_states,
            )

            new_carry = (
                obs,
                env_states,
                new_done_mask,
                episode_steps,
                episode_returns,
                key,
            )

            return new_carry, None

        carry = (
            obs,
            env_states,
            done_mask,
            episode_steps,
            episode_returns,
            key,
        )

        carry, _ = jax.lax.scan(
            scan_step,
            carry,
            xs=None,
            length=max_episode_steps,
        )

        _, _, done_mask, episode_steps, episode_returns, key = carry

        return key, episode_steps, episode_returns, done_mask

    key = jax.random.PRNGKey(seed)

    num_batches = int(np.ceil(num_episodes / actors))

    all_episode_steps = []
    all_episode_returns = []
    all_done_masks = []

    for _ in range(num_batches):
        key, episode_steps, episode_returns, done_mask = eval_batch(
            agent.params,
            key,
        )

        episode_steps = np.asarray(jax.device_get(episode_steps))
        episode_returns = np.asarray(jax.device_get(episode_returns))
        done_mask = np.asarray(jax.device_get(done_mask))

        all_episode_steps.extend(episode_steps.tolist())
        all_episode_returns.extend(episode_returns.tolist())
        all_done_masks.extend(done_mask.tolist())

    all_episode_steps = all_episode_steps[:num_episodes]
    all_episode_returns = all_episode_returns[:num_episodes]
    all_done_masks = all_done_masks[:num_episodes]

    mean_return = float(np.mean(all_episode_returns))
    mean_steps = float(np.mean(all_episode_steps))
    max_return = float(np.max(all_episode_returns))
    min_return = float(np.min(all_episode_returns))
    done_rate = float(np.mean(all_done_masks))

    log_data = {
        "eval/mean_episode_return": mean_return,
        "eval/max_episode_return": max_return,
        "eval/min_episode_return": min_return,
        "eval/mean_episode_steps": mean_steps,
        "eval/done_rate": done_rate,
    }

    if wandb.run is not None:
        if wandb_step is None:
            wandb.log(log_data)
        else:
            wandb.log(log_data, step=wandb_step)

    print("Episode steps:", all_episode_steps)
    print("Episode returns:", all_episode_returns)
    print("Mean steps:", mean_steps)
    print("Mean return:", mean_return)
    print("Done rate:", done_rate)

    if record_vid:
        record_eval_video(
            agent,
            env,
            env_params,
            max_episode_steps=max_episode_steps,
            deterministic=deterministic,
            seed=seed + 10_000,
            filename="episode_ppo.gif",
        )

        if wandb.run is not None:
            video_log = {
                "eval/video": wandb.Video(
                    "episode_ppo.gif",
                    fps=5,
                    format="gif",
                )
            }

            if wandb_step is None:
                wandb.log(video_log)
            else:
                wandb.log(video_log, step=wandb_step)

    return all_episode_steps, all_episode_returns

def record_eval_video(
    agent,
    env,
    env_params,
    max_episode_steps=10000,
    deterministic=False,
    seed=123,
    filename="episode_ppo.gif",
):
    """
    Records one evaluation episode as a GIF.

    This is intentionally not jitted because rendering and imageio.mimsave
    are Python/host-side operations.
    """

    key = jax.random.PRNGKey(seed)

    obs, env_state = env.reset(key, env_params)

    frames = []

    frame = render_craftax_pixels(env_state, 64)
    frame = frame.astype(jnp.uint8)
    frames.append(np.asarray(jax.device_get(frame)))

    done = False
    t = 0

    while (not done) and t < max_episode_steps:
        key, key_act, key_step = jax.random.split(key, 3)

        obs_batch = obs[None, ...]

        logits = agent.actor.apply(
            agent.params["actor"],
            obs_batch,
            env,
            env_params,
        )

        if deterministic:
            action = jnp.argmax(logits, axis=-1)[0]
        else:
            action = jax.random.categorical(key_act, logits[0])

        obs, env_state, reward, done, info = env.step(
            key_step,
            env_state,
            action,
            env_params,
        )

        frame = render_craftax_pixels(env_state, 64)
        frame = frame.astype(jnp.uint8)
        frames.append(np.asarray(jax.device_get(frame)))

        done = bool(jax.device_get(done))
        t += 1

    imageio.mimsave(filename, frames, fps=5)


if __name__ == "__main__":
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    shape = (8268,)

    wandb.init(
        project="craftax",
        name="ppo-1",
        config=configs.large_run
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
    test(agent, env, env_params, actors=16, num_episodes=16, deterministic=False, record_vid=True)

    # test is very inefficient. fix