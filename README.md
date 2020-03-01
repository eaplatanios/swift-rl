**WARNING:** This is currently work-in-progress.

# Reinforcement Learning in Swift

This repository contains a reinforcement learning library
built using Swift for TensorFlow, that also encompasses the
functionality of OpenAI Gym. The following is a list of
currently supported features.

- All algorithms and interfaces are designed and 
  implemented with batching in mind to support efficient
  training of neural networks that often operate on batched
  inputs.
- [Environments](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Environments/Environment.swift):
  - [Cart-Pole (classic control example)](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Environments/ClassicControl/CartPole.swift)
  - [Atari Games (using the Arcade Learning Environment)](https://github.com/eaplatanios/swift-ale)
  - [Retro Games (atari, sega, etc., using Gym Retro)](https://github.com/eaplatanios/swift-retro)
- [Agents](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Agents/Agent.swift):
  - [Policy Gradient Algorithms](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Agents/PolicyGradientAgents.swift):
    - REINFORCE
    - Advantage Actor Critic (A2C)
    - Proximal Policy Optimization (PPO)
    - *UPCOMING: Deep Deterministic Policy Gradients (DDPG)*
    - *UPCOMING: Twin Delayed Deep Deterministic Policy Gradients (TD3)*
    - *UPCOMING: Soft Actor Critic (SAC)*
  - Q-Learning Algorithms:
    - [Deep Q-Networks (DQN)](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Agents/DeepQNetworks.swift)
    - *UPCOMING: Double Deep Q-Networks (DDQN)*
- [Advantage Estimation Methods](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Values.swift):
  - Empirical Advantage Estimation
  - Generalized Advantage Estimation (GAE)
- [Replay Buffers](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/ReplayBuffers.swift):
  - Uniform Replay Buffer
  - *UPCOMING: Prioritized Replay Buffer*
- [Visualization using OpenGL for all of the currently
  implemented environments.](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Utilities/Rendering.swift)

## Installation

### Prerequisites

#### GLFW

GLFW is used for rendering. You can install it using:

```bash
# For MacOS:
brew install --HEAD git glfw3

# For Linux:
sudo apt install libglfw3-dev libglfw3
```

**NOTE:** The Swift Package Manager uses `pkg-config` to 
locate the installed libraries and so you need to make sure
that `pkg-config` is configured correctly. That may require
you to set the `PKG_CONFIG_PATH` environment variable
correctly.

**NOTE:** If the rendered image does not update according 
to the specified frames per second value and you are using 
MacOS 10.14, you should update to 10.14.4 because there is 
a bug in previous releases of 10.14 which breaks VSync.

## Reinforcement Learning Library Design Notes

**WARNING:** The below is not relevant anymore. I have been
working on a new simpler and more powerful interface and
plan to update the examples shown in this file soon.

### Batching

Batching can occur at two levels:

  - __Environment:__
  - __Policy:__

For example, in the case of retro games, the environment 
can only operate on one action at a time (i.e., it is not 
batched). If we have a policy that is also not batched, 
then we the process of collecting trajectories for training 
looks as follows:

```
... → Policy → Environment → Policy → Environment → ...
```

In this diagram, the policy is invoked to produce the next 
action and then the environment is invoked to take a step 
using that action and return rewards, etc. If instead we 
are using a policy that can be batched (e.g., a 
convolutional neural network policy would be much more 
efficient if executed in a batched manner), then we can 
collect trajectories for training in the following manner:

```
               ↗ Environment ↘            ↗ Environment ↘
... ⇒ Policy ⇒ → Environment → ⇒ Policy ⇒ → Environment → ...
               ↘ Environment ↗            ↘ Environment ↗
```

where multiple copies of the environment are running 
separately, producing rewards that are then batched and fed 
all together to a single batched policy. This policy then 
produces a batch of actions that is split up and each action 
is in term fed to its corresponding environment. Similarly, 
we can have a batched environment being used together with 
an unbatched policy:

```
    ↗ Policy ↘                 ↗ Policy ↘
... → Policy → ⇒ Environment ⇒ → Policy → ⇒ Environment ⇒ ...
    ↘ Policy ↗                 ↘ Policy ↗
```

or, even better, a batched environment used together with a 
batched policy:

```
... ⇒ Policy ⇒ Environment ⇒ Policy ⇒ Environment ⇒ ...
```

**NOTE:** Note that a batched policy is always usable as a 
policy (the batch conversions are handled automatically), 
and the same is true for batched environments.
