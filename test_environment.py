import jax
import numpy as np
import optax
from flax import linen as nn
import jax.numpy as jnp
from matplotlib import pyplot as plt

from dqn import DQN, ReplayBuffer

import jax
import jax.numpy as jnp


class Discrete:
    def __init__(self, n):
        self.n = n


class ChainEnv:
    def __init__(self, length=5, max_steps=500):
        self.length = length
        self.max_steps = max_steps
        self.num_actions = 2
        self.default_params = None

    def action_space(self, params=None):
        return Discrete(self.num_actions)

    def reset(self, key=None, params=None):
        state = {
            "pos": jnp.array(0, dtype=jnp.int32),
            "steps": jnp.array(0, dtype=jnp.int32),
        }

        obs = self._obs(state["pos"])
        return obs, state

    def step(self, key, state, action, params=None):
        pos = state["pos"]
        steps = state["steps"] + 1

        pos = jnp.where(
            action == 0,
            jnp.maximum(0, pos - 1),
            pos,
        )

        pos = jnp.where(
            action == 1,
            jnp.minimum(self.length - 1, pos + 1),
            pos,
        )

        reached_goal = pos == self.length - 1
        timed_out = steps >= self.max_steps
        done = reached_goal | timed_out

        reward = jnp.where(reached_goal, 15.0, -1.0)

        next_state = {
            "pos": pos,
            "steps": steps,
        }

        next_obs = self._obs(pos)
        info = {}

        return next_obs, next_state, reward, done, info

    def _obs(self, pos):
        return jax.nn.one_hot(pos, self.length, dtype=jnp.float32)


class MLPQNetwork(nn.Module):
    num_actions: int = 2

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x

def train(env, agent, rb, num_episodes=1):
    total_steps = 0
    key = jax.random.PRNGKey(0)
    ep_steps = []
    for episode in range(num_episodes):
        obs = env.reset()
        state_seq, reward_seq = [], []

        print("Episode:", episode)
        print("Epsilon: ", agent.epsilon)
        episode_steps = 0
        while True:
            state_seq.append(obs)

            key, key_act, key_step = jax.random.split(key, 3)

            action, new_key = agent.choose_action_basic(obs, env, key_act)
            next_obs, reward, done, info = env.step(action)
            #print(f"State: {obs}, Action: {action}, Next State: {next_obs}, Reward: {reward}, Done: {done}")
            reward_seq.append(reward)
            rb.store_transition(obs, action, reward, next_obs, done)
            if reward == 1:
                print(f"{sum(rb.rewards)}/{total_steps}")

            if len(rb) > rb.batch_size:
                agent.learn()

            total_steps += 1
            episode_steps += 1

            if total_steps % 1 == 0:
                pass
                #print("Step: ", total_steps)

            if done:
                break
            else:
                obs = next_obs
        print("Episode steps: ", episode_steps)

        threshold = 25
        if episode_steps > threshold:
            print("Spike detected")
        print("Q-values:", agent.q_network.apply(agent.params, obs[None, ...]))
        ep_steps.append(episode_steps)
        #print("Reward:", sum(reward_seq))
    print("Episode steps shape: ", len(ep_steps))
    plt.plot(range(num_episodes), ep_steps)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.show()

def test(env, agent, rb, num_episodes=1):
    key = jax.random.PRNGKey(0)
    agent.epsilon = 0

    ep_steps = []
    for i in range(num_episodes):
        print(i)
        episode_steps = 0
        obs = env.reset()
        state_seq, reward_seq = [], []
        while True:
            episode_steps += 1
            state_seq.append(obs)

            key, key_act, key_step = jax.random.split(key, 3)

            action, _ = agent.choose_action_basic(obs, env, key_act)
            next_obs, reward, done, info = env.step(action)
            # print(f"State: {obs}, Action: {action}, Next State: {next_obs}, Reward: {reward}, Done: {done}")
            if done:
                break
            else:
                obs = next_obs
        ep_steps.append(episode_steps)
    print(ep_steps)


if __name__ == "main":
    length = 15
    decay_steps = 10000
    env = ChainEnv(length=length)
    q_net = MLPQNetwork()
    rb = ReplayBuffer(10000, 128, (length,))
    optimiser = optax.adam(1e-5)
    agent = DQN(q_net, optimiser, rb, epsilon=0.9, decay=0.1, decay_steps=decay_steps)

    train(env, agent, rb, 300)
    test(env, agent, rb, 1)