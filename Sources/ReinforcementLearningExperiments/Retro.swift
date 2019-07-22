// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import Foundation
import Logging
import ReinforcementLearning
import Retro

fileprivate struct RetroActor: Network {
  @noDerivative public var state: None = None()

  public var conv1: Conv2D<Float> = Conv2D<Float>(filterShape: (8, 8, 1, 16), strides: (4, 4))
  public var conv2: Conv2D<Float> = Conv2D<Float>(filterShape: (4, 4, 16, 32), strides: (2, 2))
  public var denseHidden: Dense<Float> = Dense<Float>(inputSize: 2592, outputSize: 256)
  public var denseAction: Dense<Float> = Dense<Float>(inputSize: 256, outputSize: 8) // TODO: Easy way to get the number of actions.

  public init() {}

  public init(copying other: RetroActor) {
    conv1 = other.conv1
    conv2 = other.conv2
    denseHidden = other.denseHidden
    denseAction = other.denseAction
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Categorical<Int32> {
    let outerDimCount = input.rank - 3
    let outerDims = [Int](input.shape.dimensions[0..<outerDimCount])
    let flattenedBatchInput = input.flattenedBatch(outerDimCount: outerDimCount)
    let conv1 = leakyRelu(self.conv1(flattenedBatchInput))
    let conv2 = leakyRelu(self.conv2(conv1)).reshaped(to: [-1, 2592])
    let hidden = leakyRelu(denseHidden(conv2))
    let actionLogits = denseAction(hidden)    
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return flattenedActionDistribution.unflattenedBatch(outerDims: outerDims)
  }
}

fileprivate struct RetroActorCritic: Network {
  @noDerivative public var state: None = None()

  public var conv1: Conv2D<Float> = Conv2D<Float>(filterShape: (8, 8, 1, 4), strides: (4, 4))
  public var conv2: Conv2D<Float> = Conv2D<Float>(filterShape: (4, 4, 4, 4), strides: (2, 2))
  public var denseHidden: Dense<Float> = Dense<Float>(inputSize: 324, outputSize: 32)
  public var denseAction: Dense<Float> = Dense<Float>(inputSize: 32, outputSize: 18) // TODO: Easy way to get the number of actions.
  public var denseValue: Dense<Float> = Dense<Float>(inputSize: 32, outputSize: 1)

  public init() {}

  public init(copying other: RetroActorCritic) {
    conv1 = other.conv1
    conv2 = other.conv2
    denseHidden = other.denseHidden
    denseAction = other.denseAction
    denseValue = other.denseValue
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> ActorCriticOutput<Categorical<Int32>> {
    let outerDimCount = input.rank - 3
    let outerDims = [Int](input.shape.dimensions[0..<outerDimCount])
    let flattenedBatchInput = input.flattenedBatch(outerDimCount: outerDimCount)
    let conv1 = leakyRelu(self.conv1(flattenedBatchInput))
    let conv2 = leakyRelu(self.conv2(conv1)).reshaped(to: [-1, 324])
    let hidden = leakyRelu(denseHidden(conv2))
    let actionLogits = denseAction(hidden)
    let flattenedValue = denseValue(hidden)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1))
  }
}

public func runRetro(
  using agentType: AgentType,
  batchSize: Int = 1,
  maxEpisodes: Int = 1,
  maxReplayedSequenceLength: Int = 1000,
  discountFactor: Float = 0.9,
  entropyRegularizationWeight: Float = 0.0
) {
  let emulatorConfig: RetroEmulator.Config = {
    let retroURL = URL(fileURLWithPath: "/Users/eaplatanios/Development/GitHub/retro-swift")
    return try! RetroEmulator.Config(
      coreInformationLookupPath: retroURL.appendingPathComponent("retro/cores"),
      coreLookupPathHint: retroURL.appendingPathComponent("retro/retro/cores"),
      gameDataLookupPathHint: retroURL.appendingPathComponent("retro/retro/data"),
      gameROMLookupPaths: [URL(fileURLWithPath: "temp")],
      gameRomsDownloadPath: URL(fileURLWithPath: "temp").appendingPathComponent("downloads"))
  }()
  // let game = emulatorConfig.game(called: "SpaceInvaders-Atari2600")!
  let game = emulatorConfig.game(called: "Breakout-Atari2600")!
  let emulator = try! RetroEmulator(for: game, configuredAs: emulatorConfig)

  let logger = Logger(label: "Breakout Experiment")

  // Environment:
  var environment = try! RetroEnvironment(using: emulator, actionsType: DiscreteActions())
  var renderer = TensorImageRenderer(initialMaxWidth: 800)

  // Metrics:
  var averageEpisodeReward = AverageEpisodeReward<
    Tensor<Float>,
    Tensor<Int32>,
    None
  >(batchSize: batchSize, bufferSize: 10)

  // Agent Type:
  switch agentType {
  case .reinforce:
    let network = RetroActor()
    var agent = ReinforceAgent(
      for: environment,
      network: network,
      optimizer: AMSGrad(for: network, learningRate: 1e-3),
      discountFactor: discountFactor,
      entropyRegularizationWeight: entropyRegularizationWeight)
    for step in 0..<10000 {
      let loss = agent.update(
        using: &environment,
        maxSteps: maxReplayedSequenceLength * batchSize,
        maxEpisodes: maxEpisodes,
        stepCallbacks: [{ (environment, trajectory) in
          averageEpisodeReward.update(using: trajectory)
          if step > 0 {
            try! renderer.render(Tensor<UInt8>(255 * trajectory.observation
              .reshaped(to: [84, 84, 1])
              .tiled(multiples: Tensor<Int32>([1, 1, 3]))))
          }
        }])
      if step % 1 == 0 {
        print("Step \(step) | Loss: \(loss) | Average Episode Reward: \(averageEpisodeReward.value())")
      }
    }
  case .advantageActorCritic:
    let network = RetroActorCritic()
    var agent = A2CAgent(
      for: environment,
      network: network,
      optimizer: AMSGrad(for: network, learningRate: 1e-3),
      advantageFunction: GeneralizedAdvantageEstimation(discountFactor: discountFactor),
      entropyRegularizationWeight: entropyRegularizationWeight)
    for step in 0..<10000 {
      let loss = agent.update(
        using: &environment,
        maxSteps: maxReplayedSequenceLength * batchSize,
        maxEpisodes: maxEpisodes,
        stepCallbacks: [{ (environment, trajectory) in
          averageEpisodeReward.update(using: trajectory)
          if step > 100 {
            try! renderer.render(Tensor<UInt8>(255 * trajectory.observation
              .reshaped(to: [84, 84, 1])
              .tiled(multiples: Tensor<Int32>([1, 1, 3]))))
          }
        }])
      if step % 1 == 0 {
        print("Step \(step) | Loss: \(loss) | Average Episode Reward: \(averageEpisodeReward.value())")
      }
    }
  case .ppo:
    let network = RetroActorCritic()
    var agent = PPOAgent(
      for: environment,
      network: network,
      optimizer: AMSGrad(for: network, learningRate: 1e-3),
      advantageFunction: GeneralizedAdvantageEstimation(discountFactor: discountFactor),
      clip: PPOClip(),
      entropyRegularization: PPOEntropyRegularization(weight: entropyRegularizationWeight))
    for step in 0..<10000 {
      let loss = agent.update(
        using: &environment,
        maxSteps: 4000,
        maxEpisodes: 1,
        stepCallbacks: [{ (environment, trajectory) in
          averageEpisodeReward.update(using: trajectory)
          if step > 1000 {
            try! renderer.render(Tensor<UInt8>(255 * trajectory.observation
              .reshaped(to: [84, 84, 1])
              .tiled(multiples: Tensor<Int32>([1, 1, 3]))))
          }
        }])
      if step % 1 == 0 {
        logger.info("Step \(step) | Loss: \(loss) | Average Episode Reward: \(averageEpisodeReward.value())")
      }
    }
  case _: fatalError("This agent type is not supported yet for this experiment.")
  }
}
