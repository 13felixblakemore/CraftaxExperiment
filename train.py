import sys
import time

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels, render_craftax_symbolic
from gymnax.visualize import Visualizer
from dqn import DQN, QNetwork, ReplayBuffer

# Create environment
env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params

q_net = QNetwork()
schedule = optax.linear_schedule(
    init_value = 1e-3,
    end_value = 1e-5,
    transition_steps = 500,
)
optimiser = optax.adam(schedule)
replay_buffer = ReplayBuffer(500, 16)
agent = DQN(q_net, optimiser, replay_buffer, epsilon=0.9, decay=0.05, decay_steps=10000)

def train(num_episodes=10):
    total_steps = 0
    key = jax.random.PRNGKey(0)
    key, key_reset = jax.random.split(key)
    for episode in range(num_episodes):
        obs, env_state = env.reset(key_reset, env_params)
        state_seq, reward_seq = [], []

        print("Episode:", episode)

        while True:
            state_seq.append(env_state)

            key, key_act, key_step = jax.random.split(key, 3)

            frame = render_craftax_pixels(env_state, 16)
            action = agent.choose_action(frame, env_params, env, key_act)
            next_obs, next_env_state, reward, done, info = env.step(
                key_step, env_state, action, env_params
            )
            reward_seq.append(reward)

            next_frame = render_craftax_pixels(next_env_state, 16)

            if replay_buffer:
                replay_buffer.store_transition(frame, action, reward, next_frame, done)

                if len(replay_buffer) > replay_buffer.batch_size:
                    agent.learn()
            else:
                pass # insert ppo learning here

            total_steps += 1

            if total_steps % 15 == 0:
                print(f"Step {total_steps}: {schedule(agent.step)}")

            if done:
                break
            else:
              obs = next_obs
              env_state = next_env_state


def test(num_episodes=1):
    key = jax.random.PRNGKey(0)
    frames = []
    ep_steps = []
    for i in range(num_episodes):
        print(i)
        episode_steps = 0
        key, key_reset = jax.random.split(key, 2)
        obs, env_state = env.reset(key_reset, env_params)
        state_seq, reward_seq = [], []
        while True:
            episode_steps += 1
            state_seq.append(obs)

            key, key_act, key_step = jax.random.split(key, 3)

            frame = render_craftax_pixels(env_state, 16)
            action = agent.choose_action(frame, env_params, env, key_act)
            next_obs, next_env_state, reward, done, info = env.step(
                key_step, env_state, action, env_params
            )

            frame = frame.astype(jnp.uint8)
            frames.append(frame)

            if done:
                break
            else:
                obs = next_obs
                env_state = next_env_state
        ep_steps.append(episode_steps)
    print(ep_steps)
    imageio.mimsave("episode.gif", frames, fps=5)

t = time.perf_counter()
train(1)
nt = time.perf_counter()
latency= nt - t
print("Latency: ", latency)
#cum_rewards = jnp.cumsum(jnp.array(reward_seq))
#vis = Visualizer(env, env_params, state_seq, cum_rewards)
#vis.animate(f"anim.gif")