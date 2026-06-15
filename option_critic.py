import distrax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import configs
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from flax import linen as nn

from ppo_shared import LogWrapper


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
        b = nn.Dense(self.num_options)(s) # b shape: (n) -- terminate the active option i with probability b_i (sigmoid)
        actions = nn.Dense(self.num_options * self.action_dim)(s) # actions shape: (n * action_dim)
        actions = actions.reshape((s.shape[0], self.num_options, self.action_dim)) # actions shape: (n, action_dim)

        return q_w, b, actions


def make_train(config):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)

    num_options = 4

    def train(rng):
        network = OptionCritic(num_options, env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)

        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)

        schedule = optax.linear_schedule(
            init_value=config["learning_rate"],
            end_value=config["end_learning_rate"],
            transition_steps=config["total_timesteps"]
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
        keys = jax.random.split(_rng, config["batch_size"])

        obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)

        # choose w according to Q_w(obs)
        rng, _rng = jax.random.split(rng)
        q_w, b, action_logits = network.apply(train_state.params, obs)
        random_value = jax.random.uniform(_rng)

        rng, _rng = jax.random.split(rng)
        option = jnp.where((random_value > config["epsilon"]), jnp.argmax(q_w, axis=-1), jax.random.randint(
                _rng,
                shape=(config["batch_size"],),
                minval=0,
                maxval=num_options,
            ))

        def log_callback(log_data, global_step):
            log_data = {k: float(v) for k, v in log_data.items()}
            wandb.log(log_data, step=int(global_step))

        # do not choose an option every step. only when it has terminated
        # convert this loop to lax.scan later: def rollout()
        def update_step(run_state, _):
            train_state, obs, env_states, option, rng, update_idx = run_state
            q_w, b, action_logits = network.apply(train_state.params, obs)

            rng, _rng = jax.random.split(rng)
            batch_idx = jnp.arange(config["batch_size"])

            logits_o = action_logits[batch_idx, option, :]
            policy = distrax.Categorical(logits=logits_o)
            action = policy.sample(seed=_rng)

            rng, _rng = jax.random.split(_rng)
            keys = jax.random.split(_rng, config["batch_size"])
            next_obs, next_env_states, rewards, dones, infos = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                keys,
                env_states,
                action,
                env_params,
            )
            def update(train_state, _):
                # this must take params to diff w.r.t.
                def loss_fn(params, obs, actions, next_obs, active_option):
                    q_w, b, action_logits = network.apply(params, obs)
                    q_w_next, b_next, action_logits_next = network.apply(params, next_obs)

                    batch_idx = jnp.arange(config["batch_size"])
                    b_next = nn.sigmoid(b_next)

                    target = rewards + config["gamma"] * (1 - dones) * ((1 - b_next[batch_idx, active_option]) * q_w_next[batch_idx, active_option] + b_next[batch_idx, active_option] * jnp.max(q_w_next, axis=-1))
                    target = jax.lax.stop_gradient(target)

                    critic_loss = 0.5 * (q_w[batch_idx, active_option] - target) ** 2

                    q_a = action_logits[batch_idx, active_option, :]
                    policy = distrax.Categorical(logits=q_a)

                    log_prob = policy.log_prob(actions)
                    entropy = policy.entropy()

                    td_error = target - q_w[batch_idx, active_option]
                    td_error = jax.lax.stop_gradient(td_error)

                    policy_loss = -log_prob * td_error

                    termination_advantage = q_w_next[batch_idx, active_option] - jnp.max(q_w_next, axis=-1)
                    termination_advantage = jax.lax.stop_gradient(termination_advantage)
                    termination_loss = b_next[batch_idx, active_option] * termination_advantage

                    total_loss = (critic_loss + policy_loss + termination_loss).mean()
                    return total_loss

                grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
                total_loss, grads = grad_fn(
                    train_state.params, obs, action, next_obs, option
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, total_loss = jax.lax.scan(
                update,
                train_state,
                None,
                length=config["update_epochs"],
            )

            q_w, b_next, _ = network.apply(train_state.params, next_obs)
            b_next = nn.sigmoid(b_next)

            b_next_o = b_next[batch_idx, option]
            rng, _rng = jax.random.split(_rng)
            terminate = dones | jax.random.bernoulli(_rng, b_next_o)

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

            rng, _rng = jax.random.split(_rng)
            new_option = epsilon_greedy_options(_rng, q_w, config["epsilon"])
            option = jnp.where((terminate==1), new_option, option)

            run_state = (train_state, next_obs, next_env_states, option, rng, (update_idx+1))

            mean_loss = total_loss.mean()

            returned = infos["returned_episode"]
            num_returned = returned.sum()

            returned_returns = jnp.where(
                returned,
                infos["returned_episode_returns"],
                0.0,
            )

            episode_return = jnp.where(
                num_returned > 0,
                returned_returns.sum() / num_returned,
                jnp.nan,
            )

            returned_lengths = jnp.where(
                returned,
                infos["returned_episode_lengths"],
                0.0,
            )

            episode_lengths = jnp.where(
                num_returned > 0,
                returned_lengths.sum() / num_returned,
                jnp.nan,
            )

            env_steps = (update_idx + 1) * config["batch_size"]

            log_data = {
                "loss/total_loss": mean_loss,
                "charts/episode_return": episode_return,
                "charts/episode_length": episode_lengths,
                "charts/num_returned_episodes": num_returned,
                "charts/mean_reward": rewards.mean(),
                "options/mean_option": option.mean(),
            }

            if config["log"] and config["log_every"] > 0:
                jax.lax.cond(
                    update_idx % config["log_every"] == 0,
                    lambda _: jax.debug.callback(log_callback, log_data, env_steps),
                    lambda _: None,
                    operand=None,
                )

            return run_state, _

        rng, _rng = jax.random.split(rng)
        run_state = (train_state, obs, env_states, option, _rng, jnp.array(0))

        steps = config["total_timesteps"] // config["batch_size"]

        run_state, _ = jax.lax.scan(
            update_step,
            run_state,
            None,
            length=steps,
        )

        return {"runner_state": run_state}
    return train
# need replay buffer for decorrelated samples.

if __name__ == '__main__':
    wandb.init(
        project="craftax",
        name="option_critic",
        config=configs.large_run,
    )

    config = wandb.config

    train = jax.jit(make_train(config))

    rng = jax.random.PRNGKey(67)
    rng, _rng = jax.random.split(rng)
    run_state = train(_rng)

    # need to finish logging