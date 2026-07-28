import argparse
import os
import sys
import time
import gymnax

import jax
import numpy as np
from flax.training import orbax_utils
from orbax.checkpoint import CheckpointManagerOptions, CheckpointManager
from orbax.checkpoint._src.checkpointers.pytree_checkpointer import PyTreeCheckpointer

import METRA
import HAC
import SAC
import MOC
import dqn
import hac_jax
import metra_playground
import option_critic
import ppo_shared
import wandb


def run(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        run = wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ALGORITHM"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")


    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    if config["ALGORITHM"] == "PPO":
        make_train = ppo_shared.make_train
    elif config["ALGORITHM"] == "OPTION_CRITIC":
        make_train = option_critic.make_train
    elif config["ALGORITHM"] == "MOC":
        make_train = MOC.make_train
    elif config["ALGORITHM"] == "DQN":
        make_train = dqn.make_train
    elif config["ALGORITHM"] == "SAC":
        make_train = SAC.make_train
    elif config["ALGORITHM"] == "HAC":
        make_train = hac_jax.make_train
    elif config["ALGORITHM"] == "METRA":
        make_train = METRA.make_train
    elif config["ALGORITHM"] == "METRA_PLAYGROUND":
        make_train = metra_playground.make_train
    else:
        raise ValueError("Unsupported algorithm.")

    train_jit = jax.jit(make_train(config))
    #train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_jit(rngs[0])
    jax.block_until_ready(out)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))
    if config.get("EVAL", True):
        actor_state = out[0]

        video_paths = metra_playground.record_eval_videos(
            config,
            actor_state,
        )

        print("Evaluation recordings:")
        for video_path in video_paths:
            print(video_path)

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="WalkerWalk")
    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e6
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--update_epochs", type=int, default=50)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    # DQN
    # Flashbax would allow larger replay buffer and larger batch size
    # Prioritised replay
    # Double, dueling, distributional, noisy nets, categorical, rainbow
    # Could use n-step replay buffer
    parser.add_argument("--num_update_steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--buffer_capacity", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epsilon_start", type=float, default=0.9)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_steps", type=int, default=350_000)
    parser.add_argument("--tau", type=float, default=0.005)
    # SAC
    parser.add_argument("--target_entropy", type=float, default=-3.0)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--ent_temp", type=float, default=0.01)

    # Option Critic
    parser.add_argument("--num_options", type=int, default=4)
    parser.add_argument("--option_policy_eps", type=float, default=0.1)
    parser.add_argument("--delib_cost", type=float, default=0.05)
    # MOC
    parser.add_argument("--eta", type=float, default=0.9)
    # HAC
    parser.add_argument("--num_levels", type=int, default=2)
    parser.add_argument("--time_scale", type=int, default=10)
    parser.add_argument("--goal_indices", type=int, nargs="+", default=None)
    parser.add_argument("--end_goal", type=float, nargs="+", default=None)
    parser.add_argument("--goal_threshold", type=float, default=0.05)
    parser.add_argument("--subgoal_threshold", type=float, default=0.05)
    parser.add_argument("--subgoal_scale", type=float, default=1.0)
    parser.add_argument("--subgoal_noise", type=float, default=0.1)
    parser.add_argument("--atomic_random_prob", type=float, default=0.2)
    parser.add_argument("--subgoal_random_prob", type=float, default=0.2)
    parser.add_argument("--subgoal_test_prob", type=float, default=0.3)
    parser.add_argument(
        "--subgoal_testing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--subgoal_penalty", type=float, default=-10.0)
    parser.add_argument(
        "--hindsight_goal_replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # METRA
    parser.add_argument("--z_dim", type=int, default=2)
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--lagrange_eps", type=float, default=1e-3)
    parser.add_argument("--lipschitz_constraint", type=float, default=1.0)
    parser.add_argument("--discrete", type=bool, default=True)

    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=1024)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run(args)
    else:
        with jax.disable_jit():
            run(args)