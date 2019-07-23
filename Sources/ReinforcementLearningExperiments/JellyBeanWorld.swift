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
import JellyBeanWorld
import Logging
import NELFramework
import ReinforcementLearning

fileprivate struct JellyBeanWorldActorCritic: Network {
  @noDerivative public var state: None = None()

  public var conv1: Conv2D<Float> = Conv2D<Float>(filterShape: (5, 5, 3, 2), strides: (3, 3))
  public var conv2: Conv2D<Float> = Conv2D<Float>(filterShape: (3, 3, 2, 2), strides: (2, 2))
  public var denseHidden: Dense<Float> = Dense<Float>(inputSize: 32, outputSize: 4)
  public var denseScent1: Dense<Float> = Dense<Float>(inputSize: 3, outputSize: 4)
  public var denseScent2: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 4)
  public var denseAction: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 3) // TODO: Easy way to get the number of actions.
  public var denseValue: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 1)

  @inlinable
  public init() {}

  @inlinable
  public init(copying other: JellyBeanWorldActorCritic) {
    conv1 = other.conv1
    conv2 = other.conv2
    denseHidden = other.denseHidden
    denseScent1 = other.denseScent1
    denseScent2 = other.denseScent2
    denseAction = other.denseAction
    denseValue = other.denseValue
  }

  @inlinable
  @differentiable
  public func callAsFunction(_ input: JellyBeanWorldObservation) -> ActorCriticOutput<Categorical<Int32>> {
    let outerDimCount = input.vision.rank - 3
    let outerDims = [Int](input.vision.shape.dimensions[0..<outerDimCount])
    let flattenedBatchInput = input.flattenedBatch(outerDimCount: outerDimCount)
    let vision = flattenedBatchInput.vision
    let scent = flattenedBatchInput.scent
    let conv1 = leakyRelu(self.conv1(vision))
    let conv2 = leakyRelu(self.conv2(conv1)).reshaped(to: [-1, 32])
    let visionHidden = leakyRelu(denseHidden(conv2))
    let scent1 = leakyRelu(denseScent1(scent))
    let scentHidden = leakyRelu(denseScent2(scent1))
    let hidden = visionHidden + scentHidden
    let actionLogits = denseAction(hidden)
    let flattenedValue = denseValue(hidden)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1))
  }
}

public func runJellyBeanWorld(
  using agentType: AgentType,
  batchSize: Int = 1,
  discountFactor: Float = 0.9,
  entropyRegularizationWeight: Float = 0.0
) {
  let banana = Item(
    name: "banana",
    scent: ShapedArray([0.0, 1.0, 0.0]),
    color: ShapedArray([0.0, 1.0, 0.0]),
    requiredItemCounts: [:],
    requiredItemCosts: [:],
    blocksMovement: false,
    energyFunctions: EnergyFunctions(
      intensityFn: constantIntensity(-5.3),
      interactionFns: [
        piecewiseBoxInteraction(itemId: 0,  10.0, 200.0,  0.0,  -6.0),
        piecewiseBoxInteraction(itemId: 1, 200.0,   0.0, -6.0,  -6.0),
        piecewiseBoxInteraction(itemId: 2,  10.0, 200.0, 2.0, -100.0)]))

  let onion = Item(
    name: "onion",
    scent: ShapedArray([1.0, 0.0, 0.0]),
    color: ShapedArray([1.0, 0.0, 0.0]),
    requiredItemCounts: [:],
    requiredItemCosts: [:],
    blocksMovement: false,
    energyFunctions: EnergyFunctions(
      intensityFn: constantIntensity(-5.0),
      interactionFns: [
        piecewiseBoxInteraction(itemId: 0, 200.0, 0.0,   -6.0,   -6.0),
        piecewiseBoxInteraction(itemId: 2, 200.0, 0.0, -100.0, -100.0)]))

  let jellyBean = Item(
    name: "jellyBean",
    scent: ShapedArray([0.0, 0.0, 1.0]),
    color: ShapedArray([0.0, 0.0, 1.0]),
    requiredItemCounts: [:],
    requiredItemCosts: [:],
    blocksMovement: false,
    energyFunctions: EnergyFunctions(
      intensityFn: constantIntensity(-5.3),
      interactionFns: [
        piecewiseBoxInteraction(itemId: 0,  10.0, 200.0,    2.0, -100.0),
        piecewiseBoxInteraction(itemId: 1, 200.0,   0.0, -100.0, -100.0),
        piecewiseBoxInteraction(itemId: 2,  10.0, 200.0,  0.0,   -6.0)]))

  let wall = Item(
    name: "wall",
    scent: ShapedArray([0.0, 0.0, 0.0]),
    color: ShapedArray([0.5, 0.5, 0.5]),
    requiredItemCounts: [3: 1], // Make walls impossible to collect.
    requiredItemCosts: [:],
    blocksMovement: true,
    energyFunctions: EnergyFunctions(
      intensityFn: constantIntensity(0.0),
      interactionFns: [
        crossInteraction(itemId: 3, 10.0, 15.0, 20.0, -200.0, -20.0, 1.0)]))
  
  let simulatorConfig = SimulatorConfig(
    randomSeed: 1234567890,
    maxStepsPerMove: 1,
    scentDimSize: 3,
    colorDimSize: 3,
    visionRange: 15,
    allowedMoves: [.up],
    allowedTurns: [.left, .right],
    patchSize: 32,
    gibbsIterations: 1000,
    items: [banana, onion, jellyBean, wall],
    agentColor: [0.0, 0.0, 1.0],
    moveConflictPolicy: .firstComeFirstServe,
    scentDecay: 0.4,
    scentDiffusion: 0.14,
    removedItemLifetime: 2000)

  let rewardFunction = CombinedReward(
    ItemCollectionReward(item: jellyBean, reward: 1.0),
    ItemCollectionReward(item: onion, reward: -1.0))

  let configurations = (0..<batchSize).map { _ in
    JellyBeanWorldConfiguration(simulatorConfig: simulatorConfig, rewardFunction: rewardFunction)
  }

  // Environment:
  let environment = JellyBeanWorldEnvironment(configurations: configurations)

  // Metrics:
  let totalCumulativeReward = TotalCumulativeReward(for: environment)

  let logger = Logger(label: "Jelly Bean World Experiment")

  // Agent Type:
  switch agentType {
  case .advantageActorCritic:
    let network = JellyBeanWorldActorCritic()
    var agent = A2CAgent(
      for: environment,
      network: network,
      optimizer: AMSGrad(for: network, learningRate: 1e-3),
      advantageFunction: GeneralizedAdvantageEstimation(discountFactor: discountFactor),
      entropyRegularizationWeight: entropyRegularizationWeight)
    for step in 0..<10000 {
      let loss = agent.update(
        using: environment,
        maxSteps: 1000,
        stepCallbacks: [{ (environment, trajectory) in
          totalCumulativeReward.update()
          // if step > 100 {
          //   try! renderer.render(Tensor<UInt8>(255 * trajectory.observation
          //     .reshaped(to: [84, 84, 1])
          //     .tiled(multiples: Tensor<Int32>([1, 1, 3]))))
          // }
        }])
      if step % 1 == 0 {
        print("Step \(step) | Loss: \(loss) | Reward Rate: \(totalCumulativeReward.value()[0] / (Float(step) * 1000))/s")
      }
    }
  case .ppo:
    let network = JellyBeanWorldActorCritic()
    var agent = PPOAgent(
      for: environment,
      network: network,
      optimizer: AMSGrad(for: network),
      learningRateSchedule: ConstantLearningRate(1e-3),
      advantageFunction: GeneralizedAdvantageEstimation(discountFactor: discountFactor),
      clip: PPOClip(),
      entropyRegularization: PPOEntropyRegularization(weight: entropyRegularizationWeight))
    for step in 0..<10000 {
      let loss = agent.update(
        using: environment,
        maxSteps: 100 * batchSize,
        stepCallbacks: [{ (environment, trajectory) in
          totalCumulativeReward.update()
          // if step > 140 {
          //   painter.draw()
          // }
        }])
      if step % 1 == 0 {
        logger.info("Step \(step) | Loss: \(loss) | Reward Rate: \(totalCumulativeReward.value()[0] / (Float(step) * 1000))/s")
      }
    }
  case _: fatalError("This agent type is not supported yet for this experiment.")
  }
}
