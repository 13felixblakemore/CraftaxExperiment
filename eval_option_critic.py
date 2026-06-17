import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name
from flax import serialization

import wandb
from option_critic import OptionCritic

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import jax


def annotate_option(frame, option, reward=None, t=None):
    """
    frame: np array or jax array, shape [H, W, 3], uint8
    option: int or jax scalar
    """
    frame = np.asarray(jax.device_get(frame)).astype(np.uint8)

    option = int(jax.device_get(option))

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    text = f"Option: {option}"

    if reward is not None:
        reward = float(jax.device_get(reward))
        text += f" | r: {reward:.2f}"

    if t is not None:
        text += f" | t: {t}"

    # Box size
    margin = 4
    padding = 4

    # Default font
    font = ImageFont.load_default()

    # Text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    box = [
        margin,
        margin,
        margin + text_w + 2 * padding,
        margin + text_h + 2 * padding,
    ]

    # Draw black box + white text
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text(
        (margin + padding, margin + padding),
        text,
        fill=(255, 255, 255),
        font=font,
    )

    return np.asarray(img)

run = wandb.init(
    entity="13felixblakemore-uob",
    project="craftax",
    name="option-critic eval",
)

model_artifact = run.use_artifact(
    "13felixblakemore-uob/craftax/option_critic-final-model:latest",
    type="model",
)

ckpt_dir = model_artifact.download("models/oc_ckpt")
print("Downloaded to:", ckpt_dir)
print("ckpt_dir:", ckpt_dir)
print("Files:")
for root, dirs, files in os.walk(ckpt_dir):
    print("ROOT:", root)
    print("DIRS:", dirs)
    print("FILES:", files)


env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params

num_options = 4
action_dim = env.action_space(env_params).n

agent = OptionCritic(num_options, action_dim)

dummy_obs = jnp.zeros((1, *env.observation_space(env_params).shape))
params_template = agent.init(jax.random.PRNGKey(0), dummy_obs)

ckpt_path = os.path.join(ckpt_dir, "final_params.msgpack")

with open(ckpt_path, "rb") as f:
    params = serialization.from_bytes(params_template, f.read())

def get_outputs(obs):
    obs_batch = obs[None, ...]
    q_options, beta_logits, action_logits = agent.apply(params, obs_batch)

    q_options = q_options[0]
    beta_logits = beta_logits[0]
    action_logits = action_logits[0]

    action_logits = action_logits.reshape(num_options, action_dim)

    return q_options, beta_logits, action_logits

def record_eval_video(
    env,
    env_params,
    max_episode_steps=1000,
    deterministic=True,
    seed=64,
    filename="episode_oc.gif",
):
    key = jax.random.PRNGKey(seed)

    key, key_reset = jax.random.split(key)
    obs, env_state = env.reset(key_reset, env_params)

    frames = []

    frame = render_craftax_pixels(env_state, 64)
    frame = frame.astype(jnp.uint8)
    frames.append(np.asarray(jax.device_get(frame)))

    q_options, beta_logits, action_logits = get_outputs(obs)

    if deterministic:
        option = jnp.argmax(q_options)
    else:
        key, key_opt = jax.random.split(key)
        option = jax.random.categorical(key_opt, q_options)

    done = False
    t = 0
    total_reward = 0.0

    while (not done) and t < max_episode_steps:
        key, key_act, key_step, key_term, key_opt = jax.random.split(key, 5)

        active_option = option

        q_options, beta_logits, action_logits = get_outputs(obs)

        logits = action_logits[option]

        if deterministic:
            action = jnp.argmax(logits)
        else:
            action = jax.random.categorical(key_act, logits)

        obs, env_state, reward, done, info = env.step(
            key_step,
            env_state,
            action,
            env_params,
        )

        total_reward += float(jax.device_get(reward))

        # Termination decision is based on the new state
        q_next, beta_logits_next, _ = get_outputs(obs)

        beta_probs_next = jax.nn.sigmoid(beta_logits_next)
        beta_current = beta_probs_next[option]

        if deterministic:
            terminate = beta_current > 0.5
        else:
            terminate = jax.random.uniform(key_term) < beta_current

        if bool(jax.device_get(terminate)) or bool(jax.device_get(done)):
            if deterministic:
                option = jnp.argmax(q_next)
            else:
                option = jax.random.categorical(key_opt, q_next)

        frame = render_craftax_pixels(env_state, 64)
        frame = frame.astype(jnp.uint8)
        frame = annotate_option(frame, active_option, t=t)
        frames.append(frame)

        done = bool(jax.device_get(done))
        print(t)
        t += 1

    imageio.mimsave(filename, frames, fps=5)

    print(f"Saved {filename}")
    print(f"Episode length: {t}")
    print(f"Total reward: {total_reward}")


record_eval_video(env, env_params)
