# Reinforcement Learning in Swift

**WARNING:** This is currently work-in-progress and is 
changing rapidly. I expect it to be more stable within 
about a week or two.

This repository contains a Swift API for Gym Retro, but it
also contains a reinforcement learning library built using
Swift for TensorFlow, that also encompasses the
functionality of Gym. Currently supported features are:

- All algorithms and interfaces are designed and 
  implemented with batching in mind to support efficient
  training of neural networks that often operate on batched
  inputs.
- [Environments](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Environments/Environment.swift):
  - [Cart-Pole (classic control example)](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Environments/ClassicControl/CartPole.swift)
  - [Retro Games (atari/sega/... games)](https://github.com/eaplatanios/retro-swift/tree/master/Sources/Retro)
- [Agents](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Agents/Agent.swift):
  - Policy Gradient Algorithms:
    - [REINFORCE](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Agents/Reinforce.swift)
    - [Advantage Actor Critic (A2C)](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Agents/AdvantageActorCritic.swift)
    - *UPCOMING: Proximal Policy Optimization (PPO)*
    - *UPCOMING: Deep Deterministic Policy Gradients (DDPG)*
    - *UPCOMING: Twin Delayed Deep Deterministic Policy Gradients (TD3)*
    - *UPCOMING: Soft Actor Critic (SAC)*
  - Q-Learning Algorithms:
    - *UPCOMING: Deep Q-Networks (DQN)*
    - *UPCOMING: Double Deep Q-Networks (DDQN)*
- [Advantage Estimation Methods](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Values.swift):
  - Empirical Advantage Estimation
  - Generalized Advantage Estimation (GAE)
- [Replay Buffers](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/ReplayBuffers.swift):
  - Uniform Replay Buffer
  - *UPCOMING: Prioritized Replay Buffer*
- [Visualization using OpenGL for all of the currently
  implemented environments.](https://github.com/eaplatanios/retro-swift/blob/master/Sources/ReinforcementLearning/Environments/Rendering.swift)

## Installation

### Prerequisites

#### Retro

If `libretro.so` or `libretro.dylib` is not in your 
`LD_LIBRARY_PATH`, you need to provide the following 
extra flags when using `swift build` or any other SwiftPM 
command: `-Xlinker -L<path>`, where `<path>` represents the 
path to the dynamic library. For MacOS, you also need to 
use the following flags: `-Xlinker -rpath -Xlinker <path>`.
The simplest way to start is to execute the following 
commands from within your code directory:

```bash
git clone git@github.com:eaplatanios/retro.git
cd retro
git checkout c-api
cmake . -G 'Unix Makefiles' -DBUILD_PYTHON=OFF -DBUILD_C=ON
make -j4 retro-c

# The following is also necessary when you are on MacOS:
install_name_tool -id "$(pwd)/libretro.dylib" libretro.dylib
```

This will result in a `libretro.so` or `libretro.dylib` 
file in the `retro` subdirectory and in compiled core files 
for multiple gaming platforms in the `retro/cores`
subdirectory. Then you can set `<path>` to 
`<repository>/retro`, where `<repository>` is the path 
where you cloned the Swift Retro repository.

#### GLFW

**NOTE:** The GLFW flag is not currently working and so the 
GLFW library needs to be installed in order to use 
`retro-swift`.

In order to use the image renderer you need to first 
install GLFW. You can do so, as follows:

```bash
# For MacOS:
brew install --HEAD git glfw3

# For Linux:
sudo apt install libglfw3-dev libglfw3
```

Then, in order to use it you need to provide the following
extra flags when using `swift build` or any other SwiftPM 
command: `-Xcc -DGLFW -Xswiftc -DGLFW`. Furthermore, if 
`libglfw.so` or `libglfw.dylib` is not in your 
`LD_LIBRARY_PATH`, you also need to provide the following 
flags: `-Xlinker -lglfw -Xlinker -L<path>`, where `<path>` 
represents the path to the dynamic library. For example:

```bash
swift test \
  -Xcc -DGLFW -Xswiftc -DGLFW \
  -Xlinker -lglfw -Xlinker -L/usr/local/lib
```

**Note:** If the rendered image does not update according 
to the specified frames per second value and you are using 
MacOS 10.14, you should update to 10.14.4 because there is 
a bug in previous releases of 10.14 which breaks VSync.

### Example

This is how I can get things set up on my MacBook:

```bash
cd /Users/eaplatanios/Development/GitHub
git clone git@github.com:eaplatanios/retro-swift.git
cd retro-swift
git clone git@github.com:eaplatanios/retro.git
cd retro
git checkout c-api
cmake . -G 'Unix Makefiles' -DBUILD_PYTHON=OFF -DBUILD_C=ON
make -j4 retro-c
cd ..
swift test
```

## Example

**WARNING:** The below is not relevant anymore. I have been
working on a new simpler and more powerful interface and
plan to update the examples shown in this file soon.

The following code runs a random policy on the 
`Airstriker-Genesis` game for which a ROM is provided by 
Gym Retro.

```swift
let retroURL = URL(fileURLWithPath: "/Users/eaplatanios/Development/GitHub/retro-swift/retro")
let config = try! Emulator.Config(
  coreInformationLookupPath: retroURL.appendingPathComponent("cores"),
  coreLookupPathHint: retroURL.appendingPathComponent("retro/cores"),
  gameDataLookupPathHint: retroURL.appendingPathComponent("retro/data"))

// We only use the OpenGL-based renderer if the GLFW flag is enabled.
#if GLFW
var renderer = try! SingleImageRenderer(initialMaxWidth: 800)
#else
var renderer = ShapedArrayPrinter<UInt8>(maxEntries: 10)
#endif

let game = emulatorConfig.game(called: "Airstriker-Genesis")!
let emulator = try! Emulator(for: game, configuredAs: emulatorConfig)
var environment = try! Environment(using: emulator, actionsType: FilteredActions())
try! environment.render(using: &renderer)
for _ in 0..<1000000 {
  let action = environment.sampleAction()
  let result = environment.step(taking: action)
  try! environment.render(using: &renderer)
  if result.finished {
    environment.reset()
  }
}
```

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
