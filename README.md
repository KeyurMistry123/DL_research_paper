# Deep Reinforcement Learning for Continuous Action Spaces

This repository contains the implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm using TensorFlow for continuous action spaces. The codebase is inspired by the research paper "Continuous control with deep reinforcement learning" by Lillicrap et al., and is designed for solving continuous control problems using Actor-Critic architecture.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Training Results](#training-results)
- [References](#references)


## Overview

Deep Deterministic Policy Gradient (DDPG) is a reinforcement learning algorithm that combines the advantages of both value-based methods (such as Q-learning) and policy-based methods. This project implements DDPG using TensorFlow to handle environments with continuous action spaces.

The algorithm leverages two neural networks: the Actor and the Critic. The Actor determines the actions to take in a given state, while the Critic evaluates the actions taken by the Actor. Target networks are used to stabilize training and avoid large updates to network weights.

This project is based on the paper "Continuous control with deep reinforcement learning" by Lillicrap et al., and is designed to solve continuous action problems such as the Pendulum environment in OpenAI's Gym.

## Features

- Deep Deterministic Policy Gradient (DDPG) with Actor-Critic architecture.
- Experience replay buffer and target networks for stable training.
- Continuous action space problem-solving.
- Custom TensorFlow implementation.

## Installation

To get started, clone the repository and install the required dependencies.

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

```bash
Episode 1: Reward = -15.0
Episode 2: Reward = -10.5
Episode 3: Reward = -8.2
...
```


## References
This implementation is based on the ideas and algorithms proposed in the following research paper:

Continuous control with deep reinforcement learning
Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra
Published in: International Conference on Learning Representations (ICLR), 2016
Link: https://arxiv.org/abs/1509.02971

