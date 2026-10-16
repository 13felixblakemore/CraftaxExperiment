<img width="358" height="421" alt="image" src="https://github.com/user-attachments/assets/ae5492b1-1564-4d43-9c18-9e6a0afed147" />

# Hierarchical Reinforcement Learning in Craftax

Implementing and evaluating **reinforcement learning algorithms in JAX**, with a particular focus on hierarchical reinforcement learning (HRL).

## Overview

This project investigates whether hierarchical reinforcement learning can improve exploration and learning efficiency over conventional, flat RL algorithms.

Experiments are conducted in [Craftax](https://github.com/MichaelTMatthews/Craftax), a procedurally generated, Minecraft-inspired environment implemented in JAX for efficient parallelised training.

The project explores several approaches to learning temporally extended behaviours, comparing hierarchical methods against established RL baselines. This provided the base of my experiments in the repository jaxhrl.

## Implemented Algorithms

- **Proximal Policy Optimisation (PPO):** Implemented and reproduced as a baseline for comparison.
- **Option-Critic:** Learns temporally extended actions (options), including intra-option policies and termination functions.
- **METRA:** Unsupervised skill discovery through learned state representations and intrinsic motivation.
- **Hierarchical Actor-Critic (HAC):** Explores goal-conditioned hierarchical learning and hindsight experience replay.
- **Additional algorithms:** Experimental implementations of DQN, SAC, DDPG and modified Option-Critic variants.

## Implementation

- Implemented neural networks and training algorithms using **JAX, Flax and Optax**.
- Used `jax.jit`, `jax.vmap` and `jax.lax.scan` for efficient, parallelised training.
- Developed experimental configurations for comparing flat and hierarchical RL algorithms.
- Implemented evaluation, checkpointing and experiment logging functionality.

**Note:** The repository contains experimental code. Some imports in `train.py` reference modules not currently included in the repository and must be resolved before running this command.

## Technologies

**Python · JAX · Flax · Optax · Reinforcement Learning · Hierarchical RL**

## Author

[Felix Blakemore](https://github.com/13felixblakemore)
