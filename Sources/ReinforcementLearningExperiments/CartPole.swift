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

import Logging
import ReinforcementLearning
import TensorFlow

fileprivate struct CartPoleActor: Layer {
  public var dense1: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 100)
  public var dense2: Dense<Float> = Dense<Float>(inputSize: 100, outputSize: 2)

  @differentiable
  public func callAsFunction(_ input: CartPoleEnvironment.Observation) -> Categorical<Int32> {
    let stackedInput = Tensor<Float>(
      stacking: [
        input.position, input.positionDerivative,
        input.angle, input.angleDerivative],
      alongAxis: input.position.rank)
    let outerDimCount = stackedInput.rank - 1
    let outerDims = [Int](stackedInput.shape.dimensions[0..<outerDimCount])
    let flattenedBatchStackedInput = stackedInput.flattenedBatch(outerDimCount: outerDimCount)
    let hidden = leakyRelu(dense1(flattenedBatchStackedInput))
    let actionLogits = dense2(hidden)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return flattenedActionDistribution.unflattenedBatch(outerDims: outerDims)
  }
}

fileprivate struct CartPoleActorCritic: Layer {
  public var dense1Action: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 100)
  public var dense2Action: Dense<Float> = Dense<Float>(inputSize: 100, outputSize: 2)
  public var dense1Value: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 100)
  public var dense2Value: Dense<Float> = Dense<Float>(inputSize: 100, outputSize: 1)

  @differentiable
  public func callAsFunction(
    _ input: CartPoleEnvironment.Observation
  ) -> ActorCriticOutput<Categorical<Int32>> {
    let stackedInput = Tensor<Float>(
      stacking: [
        input.position, input.positionDerivative,
        input.angle, input.angleDerivative],
      alongAxis: -1)
    let outerDimCount = stackedInput.rank - 1
    let outerDims = [Int](stackedInput.shape.dimensions[0..<outerDimCount])
    let flattenedBatchStackedInput = stackedInput.flattenedBatch(outerDimCount: outerDimCount)
    let actionLogits = dense2Action(leakyRelu(dense1Action(flattenedBatchStackedInput)))
    let flattenedValue = dense2Value(leakyRelu(dense1Value(flattenedBatchStackedInput)))
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1))
  }
}

fileprivate struct CartPoleQNetwork: Layer & Copyable {
  public var dense1: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 100)
  public var dense2: Dense<Float> = Dense<Float>(inputSize: 100, outputSize: 2)

  @differentiable
  public func callAsFunction(_ input: CartPoleEnvironment.Observation) -> Tensor<Float> {
    let stackedInput = Tensor<Float>(
      stacking: [
        input.position, input.positionDerivative,
        input.angle, input.angleDerivative],
      alongAxis: input.position.rank)
    let outerDimCount = stackedInput.rank - 1
    let outerDims = [Int](stackedInput.shape.dimensions[0..<outerDimCount])
    let flattenedBatchStackedInput = stackedInput.flattenedBatch(outerDimCount: outerDimCount)
    let hidden = leakyRelu(dense1(flattenedBatchStackedInput))
    let flattenedQValues = dense2(hidden)
    return flattenedQValues.unflattenedBatch(outerDims: outerDims)
  }

  public func copy() -> CartPoleQNetwork { self }
}

public func runCartPole(
  using agentType: AgentType,
  batchSize: Int = 32,
  maxEpisodes: Int = 32,
  maxReplayedSequenceLength: Int = 1000,
  discountFactor: Float = 0.9,
  entropyRegularizationWeight: Float = 0.01
) throws {
  let logger = Logger(label: "Cart-Pole Experiment")
  let baseEnvironment = CartPoleEnvironment(batchSize: batchSize)
  let averageEpisodeLength = AverageEpisodeLength(for: baseEnvironment, bufferSize: 10)
  let environment = EnvironmentCallbackWrapper(
    baseEnvironment,
    callbacks: averageEpisodeLength.updater())

  // Agent Type:
  switch agentType {
  case .reinforce:
    let network = CartPoleActor()
    var agent = ReinforceAgent(
      for: environment,
      network: network,
      optimizer: AMSGrad(for: network, learningRate: 1e-3),
      discountFactor: discountFactor,
      entropyRegularizationWeight: entropyRegularizationWeight)
    for step in 0..<10000 {
      let loss = try agent.update(
        using: environment,
        maxSteps: maxReplayedSequenceLength * batchSize,
        maxEpisodes: maxEpisodes,
        stepCallbacks: [{ (environment, trajectory) in
          if step > 100 { try! environment.render() }
        }])
      if step % 1 == 0 {
        logger.info("Step \(step) | Loss: \(loss) | Average Episode Length: \(averageEpisodeLength.value())")
      }
    }
  case .advantageActorCritic:
    let network = CartPoleActorCritic()
    var agent = A2CAgent(
      for: environment,
      network: network,
      optimizer: AMSGrad(for: network, learningRate: 1e-3),
      advantageFunction: GeneralizedAdvantageEstimation(discountFactor: discountFactor),
      entropyRegularizationWeight: entropyRegularizationWeight)
    for step in 0..<10000 {
      let loss = try agent.update(
        using: environment,
        maxSteps: maxReplayedSequenceLength * batchSize,
        maxEpisodes: maxEpisodes,
        stepCallbacks: [{ (environment, trajectory) in
          if step > 100 { try! environment.render() }
        }])
      if step % 1 == 0 {
        logger.info("Step \(step) | Loss: \(loss) | Average Episode Length: \(averageEpisodeLength.value())")
      }
    }
  case .ppo:
    let network = CartPoleActorCritic()
    var agent = PPOAgent(
      for: environment,
      network: network,
      optimizer: AMSGrad(for: network, learningRate: 1e-3),
      learningRateSchedule: LinearLearningRateDecay(slope: -1e-3 / 100.0, lowerBound: 1e-6),
      advantageFunction: GeneralizedAdvantageEstimation(
        discountFactor: 0.99,
        discountWeight: 0.95))
    for step in 0..<10000 {
      let loss = try agent.update(
        using: environment,
        maxSteps: maxReplayedSequenceLength * batchSize,
        maxEpisodes: maxEpisodes,
        stepCallbacks: [{ (environment, trajectory) in
          if step > 0 { try! environment.render() }
        }])
      if step % 1 == 0 {
        logger.info("Step \(step) | Loss: \(loss) | Average Episode Length: \(averageEpisodeLength.value())")
      }
    }
  case .dqn:
    let network = CartPoleQNetwork()
    var agent = DQNAgent(
      for: environment,
      qNetwork: network,
      optimizer: AMSGrad(for: network, learningRate: 1e-3),
      trainSequenceLength: 1,
      maxReplayedSequenceLength: maxReplayedSequenceLength,
      epsilonGreedy: 0.1,
      targetUpdateForgetFactor: 0.95,
      targetUpdatePeriod: 5,
      discountFactor: 0.99,
      trainStepsPerIteration: 1)
    for step in 0..<10000 {
      let loss = try agent.update(
        using: environment,
        maxSteps: 32 * 10,
        maxEpisodes: 32,
        stepCallbacks: [{ trajectory in
          if step > 100 { try! environment.render() }
        }])
      if step % 1 == 0 {
        logger.info("Step \(step) | Loss: \(loss) | Average Episode Length: \(averageEpisodeLength.value())")
      }
    }
  }
}
