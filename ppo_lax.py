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


def wandb_callback(metrics, step):
    def to_python(x):
        x = np.asarray(x)
        if x.shape == ():
            return x.item()
        return x

    metrics_host = jax.tree_util.tree_map(to_python, metrics)
    step_host = int(np.asarray(step).item())

    wandb.log(metrics_host, step=step_host)

def maybe_log(metrics, total_steps, update_step, log_every):
    should_log = (update_step % log_every) == 0

    def do_log(_):
        jax.debug.callback(
            wandb_callback,
            metrics,
            total_steps,
            ordered=True,
        )
        return None

    def dont_log(_):
        return None

    jax.lax.cond(
        should_log,
        do_log,
        dont_log,
        operand=None,
    )

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

def make_train_iteration(env, env_params, agent, config, obs_shape=(8268,)):

    num_steps = config["num_steps"]
    actors = config["actors"]
    num_minibatches = config["num_minibatches"]
    update_epochs = config["update_epochs"]
    batch_size = num_steps * actors
    minibatch_size = batch_size // num_minibatches

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

    def compute_gae(reward_seq, value_seq, done_seq, next_values, lmbda, gamma):
        def GAE(carry, transition):
            last_lmbda, next_value = carry
            reward, value, done = transition

            next_non_terminal = 1.0 - done.astype(jnp.float32)

            delta = reward + gamma * next_value * next_non_terminal - value
            last_lmbda = delta + gamma * lmbda * next_non_terminal * last_lmbda
            advantage = last_lmbda

            new_carry = (last_lmbda, value)

            return new_carry, advantage

        initial_carry = (jnp.zeros_like(next_values), next_values)

        _, advantages = jax.lax.scan(
            GAE,
            initial_carry,
            (reward_seq, value_seq, done_seq),
            reverse=True,
        )

        returns = advantages + value_seq

        return advantages, returns

    def update_rollout(params, opt_state, flat_batch, key
                       ,env_params,epsilon,ent_coef,vf_coef):
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

    def train_iteration(run_state):
        params, opt_state, obs, env_states, key, total_steps, update_steps = run_state

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

        next_values = agent.critic.apply(
            params["critic"],
            obs,
        ).reshape(-1)

        lmbda = config["lambda"]
        gamma = config["gamma"]
        advantages, returns = compute_gae(reward_seq, value_seq, done_seq, next_values, lmbda, gamma)

        flat_batch = {
            "states": state_seq.reshape(-1, *obs_shape),
            "actions": action_seq.reshape(-1),
            "returns": returns.reshape(-1),
            "advantages": advantages.reshape(-1),
            "log_probs": log_prob_seq.reshape(-1),
        }

        params, opt_state, key, mean_metrics = update_rollout(
            params,
            opt_state,
            flat_batch,
            key,
            env_params,
            config["epsilon"],
            config["ent_coef"],
            config["vf_coef"],
        )

        returns_flat = returns.reshape(-1)
        values_flat = value_seq.reshape(-1)

        returns_var = jnp.var(returns_flat)
        explained_var = jnp.where(
            returns_var == 0.0,
            0.0,
            1.0 - jnp.var(returns_flat - values_flat) / returns_var,
        )

        metrics = {
            "charts/total_steps": total_steps,
            "rollout/mean_reward": jnp.mean(reward_seq),
            "rollout/sum_reward": jnp.sum(reward_seq),
            "rollout/mean_done": jnp.mean(done_seq),
            "value/explained_variance": explained_var,
            "value/mean_return": jnp.mean(returns),
            "value/mean_value": jnp.mean(value_seq),
            "value/mean_advantage": jnp.mean(advantages),
            **mean_metrics,
        }

        maybe_log(
            metrics,
            total_steps,
            update_step,
            log_every=10,
        )

        update_steps += 1

        run_state = params, opt_state, obs, env_states, key, total_steps, update_steps

        return run_state, metrics
    return train_iteration

def make_train_all(train_iteration, num_iterations):
    @jax.jit
    def train_all(runner_state):
        def scan_step(runner_state, _):
            runner_state, metrics = train_iteration(runner_state)

            # Don't return metrics unless you want all metrics stored
            return runner_state, None

        runner_state, _ = jax.lax.scan(
            scan_step,
            runner_state,
            xs=None,
            length=num_iterations,
        )

        return runner_state

    return train_all

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

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, config["actors"])

    batched_reset = jax.vmap(env.reset, in_axes=(0, None))
    obs, env_states = batched_reset(keys, env_params)

    run_state = (
        agent.params,
        agent.opt_state,
        obs,
        env_states,
        key,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )

    batched_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))


    train_iteration = make_train_iteration(env, env_params, agent, config, obs_shape=shape)

    train_iterations = config["total_timesteps"] // (config["actors"] * config["num_steps"])

    train_all = make_train_all(train_iteration, train_iterations)
    run_state, metrics = train_all(run_state)

    params, opt_state, obs, env_states, key, total_steps, update_step = run_state
    agent.params = params
    agent.opt_state = opt_state