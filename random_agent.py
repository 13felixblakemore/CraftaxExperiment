import sys

import imageio
import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels, render_craftax_symbolic
from gymnax.visualize import Visualizer

state_seq, reward_seq = [], []
key = jax.random.PRNGKey(0)
key, key_reset = jax.random.split(key)

# Create environment
env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params

obs, env_state = env.reset(key_reset, env_params)

i = 0
frames = []
while i < 1000:
    frame = render_craftax_pixels(env_state, 64)
    frame = frame.astype(jnp.uint8)
    frames.append(frame)
    state_seq.append(env_state)
    key, key_act, key_step = jax.random.split(key, 3)
    action = env.action_space(env_params).sample(key_act)
    next_obs, next_env_state, reward, done, info = env.step(
        key_step, env_state, action, env_params
    )
    reward_seq.append(reward)
    i += 1
    if (i % 10) == 0:
        print(f"{i} - Action: + {action}")
    if done:
        break
    else:
      obs = next_obs
      env_state = next_env_state

cum_rewards = jnp.cumsum(jnp.array(reward_seq))
vis = Visualizer(env, env_params, state_seq, cum_rewards)
vis.animate(f"anim.gif")
imageio.mimsave("episode.gif", frames, fps=5)