import json
import sys
from pathlib import Path
from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax.serialization import to_bytes
from flax.training.train_state import TrainState

import configs
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn

from configs import debug_config
from ppo_shared import LogWrapper

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray
    option: jnp.ndarray
    b: jnp.ndarray


class OptionCritic(nn.Module):
    num_options: int
    action_dim: int

    @nn.compact
    def __call__(self, s):
        # s is shape (b, obs_shape)
        # maybe dense instead of conv for symbolic

        s = nn.Dense(256)(s)
        s = nn.relu(s)
        s = nn.Dense(256)(s)
        s = nn.relu(s)

        q_w = nn.Dense(self.num_options)(s) # q_w shape: (n) -- choose policy with epsilon greedy
        b = nn.Dense(self.num_options, bias_init=nn.initializers.constant(-2.0))(s) # b shape: (n) -- terminate the active option i with probability b_i (sigmoid)
        actions = nn.Dense(self.num_options * self.action_dim)(s) # actions shape: (n * action_dim)
        actions = actions.reshape((s.shape[0], self.num_options, self.action_dim)) # actions shape: (n, action_dim)

        return q_w, b, actions


def make_train(config):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)

    num_options = 8
    batch_size = config["actors"] * config["num_steps"]
    num_updates = config["total_timesteps"] // batch_size
    config["minibatch_size"] = batch_size // config["num_minibatches"]
    num_optim_steps = num_updates * config["update_epochs"] * config["num_minibatches"]

    def train(rng):
        network = OptionCritic(num_options, env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)

        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)

        schedule = optax.linear_schedule(
            init_value=config["learning_rate"],
            end_value=config["end_learning_rate"],
            transition_steps=num_optim_steps
        )

        optimiser = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(schedule, eps=config["adam_epsilon"]),
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=optimiser,)

        rng, _rng = jax.random.split(rng)
        keys = jax.random.split(_rng, config["actors"])

        obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)

        # choose w according to Q_w(obs)
        q_w, b, action_logits = network.apply(train_state.params, obs)

        def epsilon_greedy_options(rng, q_w, epsilon):
            batch_size = q_w.shape[0]

            greedy_option = jnp.argmax(q_w, axis=-1)

            rng_random, rng_uniform = jax.random.split(rng)
            random_option = jax.random.randint(
                rng_random,
                shape=(batch_size,),
                minval=0,
                maxval=q_w.shape[-1],
            )

            choose_random = jax.random.uniform(rng_uniform, shape=(batch_size,)) < epsilon

            return jnp.where(choose_random, random_option, greedy_option)

        rng, _rng = jax.random.split(rng)
        option = epsilon_greedy_options(_rng, q_w, config["epsilon_option"])

        def log_callback(log_data, global_step):
            log_data = {k: float(v) for k, v in log_data.items()}
            wandb.log(log_data, step=int(global_step))

        def update_step(run_state, _):
            def rollout_step(carry, transition):
                train_state, obs, env_states, rng, option = carry

                # choose action for each env
                # calc log_probs of each action
                rng, _rng = jax.random.split(rng)
                values, b, action_logits = network.apply(train_state.params, obs)
                logits_o = action_logits[jnp.arange(config["actors"]), option, :]
                policy = distrax.Categorical(logits=logits_o)
                actions = policy.sample(seed=_rng)
                log_probs = policy.log_prob(actions)

                rng, _rng = jax.random.split(_rng)
                keys = jax.random.split(_rng, config["actors"])
                next_obs, next_env_states, rewards, dones, infos = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    keys,
                    env_states,
                    actions,
                    env_params,
                )

                q_w, b_next, _ = network.apply(train_state.params, next_obs)
                b_next = nn.sigmoid(b_next)

                b_next_o = b_next[jnp.arange(config["actors"]), option]

                rng, _rng = jax.random.split(_rng)
                terminate = dones | jax.random.bernoulli(_rng, b_next_o)

                rng, _rng = jax.random.split(_rng)
                new_option = epsilon_greedy_options(_rng, q_w, config["epsilon_option"])
                new_option = jnp.where((terminate == 1), new_option, option)

                transition = Transition(dones, actions, values, rewards, log_probs, obs, next_obs, infos, option, b)
                new_carry = train_state, next_obs, next_env_states, rng, new_option
                return new_carry, transition


            train_state, obs, env_states, rng, option, update_idx = run_state
            global_step = update_idx * config["actors"] * config["num_steps"]

            rollout_state = (train_state, obs, env_states, rng, option)

            rollout_state, rollout = jax.lax.scan(
                rollout_step,
                rollout_state,
                xs=None,
                length=config["num_steps"],
            )

            train_state, obs, env_states, rng, option = rollout_state

            def compute_gae(rollout, last_q, last_b, option):
                def gae_step(carry, transition):
                    last_gae, next_value, next_b = carry
                    reward, value, done, option, rollout_b_logits = transition
                    next_b = nn.sigmoid(next_b)

                    next_non_terminal = 1.0 - done.astype(jnp.float32)

                    bootstrap = ((1 - next_b[jnp.arange(config["actors"]), option]) * next_value[jnp.arange(config["actors"]), option] + next_b[jnp.arange(config["actors"]), option] * jnp.max(next_value, axis=-1))

                    delta = reward + config["gamma"] * bootstrap * next_non_terminal - value[jnp.arange(config["actors"]), option]

                    last_gae = (
                            delta
                            + config["gamma"]
                            * config["lambda"]
                            * next_non_terminal
                            * last_gae
                    )

                    return (last_gae, value, rollout_b_logits), last_gae

                initial_carry = (
                    jnp.zeros_like(option, dtype=jnp.float32),
                    last_q,
                    last_b
                )

                _, advantages = jax.lax.scan(
                    gae_step,
                    initial_carry,
                    (rollout.reward, rollout.value, rollout.done, rollout.option, rollout.b),
                    reverse=True,
                    unroll=16,
                )

                values = rollout.value[
                    jnp.arange(config["num_steps"])[:, None],
                    jnp.arange(config["actors"])[None, :],
                    rollout.option.astype(jnp.int32)
                ]

                returns = advantages + values

                return advantages, returns

            q_w_next, b_next, action_logits_next = network.apply(train_state.params, obs)

            advantages, returns = compute_gae(rollout, q_w_next, b_next, option)

            def update_epoch(update_state, _):
                def update_minibatch(train_state, batch_info):
                    rollout, advantages, returns = batch_info
                    def loss_fn(params, rollout, gae, returns):
                        q_w, b, action_logits = network.apply(params, rollout.obs)
                        q_w_next, b_next_logits, _ = network.apply(params, rollout.next_obs)

                        B = rollout.obs.shape[0]
                        batch_idx = jnp.arange(B)
                        options = rollout.option.astype(jnp.int32)

                        logits_o = action_logits[batch_idx, options, :]

                        policy = distrax.Categorical(logits=logits_o)

                        new_log_probs = policy.log_prob(rollout.action)
                        old_log_probs = rollout.log_prob
                        log_ratio = new_log_probs - old_log_probs
                        ratio = jnp.exp(log_ratio)

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss1 = ratio * gae
                        loss2 = (
                                jnp.clip(ratio, 1 - config["epsilon"], 1 + config["epsilon"])
                                * gae
                        )
                        actor_loss = -jnp.mean(jnp.minimum(loss1, loss2))

                        entropy = policy.entropy().mean()

                        B = rollout.obs.shape[0]
                        batch_idx = jnp.arange(B)
                        options = rollout.option.astype(jnp.int32)

                        values = q_w[batch_idx, options]
                        critic_loss = jnp.mean((values - returns) ** 2)

                        b_next = nn.sigmoid(b_next_logits)
                        beta_next_o = b_next[batch_idx, options]
                        q_next_o = q_w_next[batch_idx, options]

                        # Greedy baseline version
                        v_next = jnp.max(q_w_next, axis=-1)

                        termination_advantage = q_next_o - v_next
                        termination_advantage = jax.lax.stop_gradient(
                            termination_advantage + config.get("delib_cost", 0.01)
                        )

                        nonterminal = 1.0 - rollout.done.astype(jnp.float32)

                        termination_loss = jnp.mean(
                            nonterminal * beta_next_o * termination_advantage
                        )

                        total_loss = actor_loss + config["vf_coef"] * critic_loss - config["ent_coef"] * entropy + termination_loss

                        aux = {
                            "actor_loss": actor_loss,
                            "critic_loss": critic_loss,
                            "entropy": entropy,
                            "termination_loss": termination_loss,
                        }

                        return total_loss, aux

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (total_loss, losses), grads = grad_fn(
                        train_state.params, rollout, advantages, returns
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, rollout, advantages, returns, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["minibatch_size"] * config["num_minibatches"]
                assert (
                        batch_size == config["num_steps"] * config["actors"]
                ), "batch size must be equal to number of steps * number of envs"

                permutation = jax.random.permutation(_rng, batch_size)
                batch = (rollout, advantages, returns)

                # flatten rollout batch into actors * steps for each item (actions, states, rewards, etc)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )

                # shuffle the flattened batch with the permutation
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                # reshape into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["num_minibatches"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    update_minibatch, train_state, minibatches
                )
                update_state = (train_state, rollout, advantages, returns, rng)
                return update_state, total_loss

            update_state = train_state, rollout, advantages, returns, rng

            update_state, total_loss = jax.lax.scan(
                update_epoch,
                update_state,
                None,
                config["update_epochs"]
            )

            train_state = update_state[0]
            returned = rollout.info["returned_episode"]
            num_returned = returned.sum()

            episode_return = jnp.where(
                num_returned > 0,
                (rollout.info["returned_episode_returns"] * returned).sum() / num_returned,
                jnp.nan,
            )

            episode_length = jnp.where(
                num_returned > 0,
                (rollout.info["returned_episode_lengths"] * returned).sum() / num_returned,
                jnp.nan,
            )
            log_data = {
                "episode_return": episode_return,
                "episode_length": episode_length,
            }

            if config["log"]:
                jax.lax.cond(
                    update_idx % config["log_every"] == 0,
                    lambda _: jax.debug.callback(log_callback, log_data, global_step),
                    lambda _: None,
                    operand=None,
                )

            rng = update_state[-1]

            runner_state = (
                train_state,
                obs,
                env_states,
                rng,
                option,
                update_idx + 1,
            )

            return runner_state, None

        rng, _rng = jax.random.split(rng)
        run_state = (train_state, obs, env_states, _rng, option, jnp.array(0))

        steps = config["total_timesteps"] // (config["actors"] * config["num_steps"])

        run_state, _ = jax.lax.scan(
            update_step,
            run_state,
            None,
            length=steps,
        )

        return {"runner_state": run_state}
    return train

if __name__ == '__main__':
    wandb.init(project="craftax", name="option_critic_8", config=configs.large_run,)
    config = wandb.config

    train = jax.jit(make_train(config))

    rng = jax.random.PRNGKey(67)
    rng, _rng = jax.random.split(rng)
    run_state = train(_rng)
    final_train_state = run_state["runner_state"][0]
    params = final_train_state.params

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    params_path = save_dir / "final_params.msgpack"
    config_path = save_dir / "config.json"

    with open(params_path, "wb") as f:
        f.write(to_bytes(params))

    with open(config_path, "w") as f:
        json.dump(dict(wandb.config), f, indent=2)

    artifact = wandb.Artifact(
        name=f"{wandb.run.name}-final-model",
        type="model",
    )

    artifact.add_file(str(params_path))
    artifact.add_file(str(config_path))

    wandb.log_artifact(artifact)