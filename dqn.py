# incomplete

import jax
import numpy as np
import jax.numpy as jnp
import optax
from flax import linen as nn


class ReplayBuffer:
    def __init__(self, capacity, batch_size, state_shape=(208,176,3)):
        self.capacity = capacity
        self.batch_size = batch_size
        self.write_index = 0
        self.size = 0
        self.state_shape = state_shape

        self.states = jnp.zeros((capacity, *self.state_shape), dtype=jnp.float32)
        self.actions = jnp.zeros(capacity, dtype=jnp.int32)
        self.rewards = jnp.zeros(capacity, dtype=jnp.float32)
        self.next_states = jnp.zeros((capacity, *state_shape), dtype=jnp.float32)
        self.dones = jnp.zeros(capacity, dtype=jnp.float32)

    def __len__(self):
        return self.size

    def sample_traj(self):
        indices = np.random.choice(self.size, self.batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def store_transition(self, state, action, reward, next_state, done):
        if self.write_index < self.capacity:
            self.states = self.states.at[self.write_index].set(state)
            self.actions = self.actions.at[self.write_index].set(action)
            self.rewards = self.rewards.at[self.write_index].set(reward)
            self.next_states = self.next_states.at[self.write_index].set(next_state)
            self.dones = self.dones.at[self.write_index].set(done)
            self.write_index += 1
        else:
            self.write_index = (self.write_index + 1) % self.capacity
            self.store_transition(state, action, reward, next_state, done)
        self.size = min(self.size + 1, self.capacity)
        if self.size == self.batch_size:
            print("Learning")

class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(8,), strides=4, padding='VALID')(x)
        x = nn.relu(x)

        x = nn.Conv(features=32, kernel_size=(4,), strides=2, padding='VALID')(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)


        num_actions = 43
        x = nn.Dense(num_actions)(x)
        return x


def make_greedy_action_fn(q_network):
    @jax.jit
    def greedy_action(params, state):
        state = state[None, ...]
        q_values = q_network.apply(params, state)
        return jnp.argmax(q_values[0]).astype(jnp.int32)

    return greedy_action

def make_training_step(q_network, optimiser):
    @jax.jit
    def train_step(
        params,
        target_params,
        opt_state,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
        gamma,
    ):
        def loss_fn(params):
            q = q_network.apply(params, state_batch)
            q_sa = jnp.take_along_axis(q, action_batch[:, None], axis=1).squeeze()

            q_target = q_network.apply(target_params, next_state_batch)

            td_errors = reward_batch + (gamma * (1 - done_batch) * jnp.max(q_target, axis=1)) - q_sa
            loss = jnp.mean(jnp.square(td_errors))
            return loss

        loss, grad = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimiser.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return train_step


class DQN:
    def __init__(self, q_network, optimiser, replay_buffer, gamma=0.99, epsilon = 0.8, decay = 0.05, decay_steps=5000):
        self.step = 0
        self.target_update_freq = 75

        self.q_network = q_network
        self.greedy_action_jit = make_greedy_action_fn(self.q_network)

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.decay_steps = decay_steps

        init_dummy = np.zeros((self.replay_buffer.batch_size, *self.replay_buffer.state_shape))
        key = jax.random.PRNGKey(0)
        q_key, t_key = jax.random.split(key, 2)
        self.params = self.q_network.init(q_key, init_dummy)
        self.t_params = self.q_network.init(t_key, init_dummy)

        self.optimiser = optimiser
        self.opt_state = self.optimiser.init(self.params)

        self.training_step = make_training_step(self.q_network, self.optimiser)

    def choose_action(self, state, env_params, env, policy_key):
        key, eps_key = jax.random.split(policy_key)
        random_value = jax.random.uniform(eps_key)

        if random_value < self.epsilon:
            action = env.action_space(env_params).sample(policy_key)
            action = jnp.asarray(action, dtype=jnp.int32)
        else:
            action = self.greedy_action_jit(self.params, state)
        return action

    def choose_action_basic(self, state, env, policy_key):
        key, eps_key, eps_key_2 = jax.random.split(policy_key, 3)
        random_value = jax.random.uniform(eps_key)

        if random_value < self.epsilon:
            rand = jax.random.uniform(eps_key_2)
            if rand < 0.5:
                action = 0
            else:
                action = 1
        else:
            action = self.greedy_action_jit(self.params, state)
        return action, key

    def learn(self):
        transitions = self.replay_buffer.sample_traj()
        self.params, self.opt_state, loss = self.training_step(self.params, self.t_params, self.opt_state, *transitions, self.gamma)

        self.step += 1

        self.epsilon = self.epsilon - ((self.epsilon - self.decay) / self.decay_steps)

        if self.step % self.target_update_freq == 0:
            self.t_params = self.params

        return loss