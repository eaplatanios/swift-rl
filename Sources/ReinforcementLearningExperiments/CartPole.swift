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
  public func callAsFunction(
    _ input: AgentInput<CartPoleEnvironment.Observation, Empty>
  ) -> ActorOutput<Categorical<Int32>, Empty> {
    let input = input.observation
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
    return ActorOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      state: Empty())
  }
}

fileprivate struct CartPoleActorCritic: Layer {
  public var dense1Action: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 100)
  public var dense2Action: Dense<Float> = Dense<Float>(inputSize: 100, outputSize: 2)
  public var dense1Value: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 100)
  public var dense2Value: Dense<Float> = Dense<Float>(inputSize: 100, outputSize: 1)

  @differentiable
  public func callAsFunction(
    _ input: AgentInput<CartPoleEnvironment.Observation, Empty>
  ) -> ActorCriticOutput<Categorical<Int32>, Empty> {
    let input = input.observation
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
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1),
      state: Empty())
  }
}

fileprivate struct CartPoleQNetwork: Layer & Copyable {
  public var dense1: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 100)
  public var dense2: Dense<Float> = Dense<Float>(inputSize: 100, outputSize: 2)

  @differentiable
  public func callAsFunction(
    _ input: AgentInput<CartPoleEnvironment.Observation, Empty>
  ) -> QNetworkOutput<Empty> {
    let input = input.observation
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
    return QNetworkOutput(
      qValues: flattenedQValues.unflattenedBatch(outerDims: outerDims),
      state: Empty())
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
  var environment = CartPoleEnvironment(batchSize: batchSize)
  var averageEpisodeLength = AverageEpisodeLength<CartPoleEnvironment, Empty>(
    for: environment,
    bufferSize: 10)
  var agent: AnyAgent<CartPoleEnvironment, Empty> = {
    switch agentType {
    case .reinforce:
      let network = CartPoleActor()
      return AnyAgent(ReinforceAgent(
        for: environment,
        network: network,
        optimizer: AMSGrad(for: network, learningRate: 1e-3),
        discountFactor: discountFactor,
        entropyRegularizationWeight: entropyRegularizationWeight))
    case .advantageActorCritic:
      let network = CartPoleActorCritic()
      return AnyAgent(A2CAgent(
        for: environment,
        network: network,
        optimizer: AMSGrad(for: network, learningRate: 1e-3),
        advantageFunction: GeneralizedAdvantageEstimation(discountFactor: discountFactor),
        entropyRegularizationWeight: entropyRegularizationWeight))
    case .ppo:
      let network = CartPoleActorCritic()
      return AnyAgent(PPOAgent(
        for: environment,
        network: network,
        optimizer: AMSGrad(for: network),
        learningRate: LinearlyDecayedLearningRate(
          baseLearningRate: FixedLearningRate(1e-3),
          slope: -1e-3 / 100.0,
          lowerBound: 1e-6),
        advantageFunction: GeneralizedAdvantageEstimation(
          discountFactor: 0.99,
          discountWeight: 0.95)))
    case .dqn:
      let network = CartPoleQNetwork()
      return AnyAgent(DQNAgent(
        for: environment,
        qNetwork: network,
        optimizer: AMSGrad(for: network, learningRate: 1e-3),
        trainSequenceLength: 1,
        maxReplayedSequenceLength: maxReplayedSequenceLength,
        epsilonGreedy: 0.1,
        targetUpdateForgetFactor: 0.95,
        targetUpdatePeriod: 5,
        discountFactor: 0.99,
        trainStepsPerIteration: 1))
    }
  }()

  for step in 0..<10000 {
    let loss = try agent.update(
      using: &environment,
      maxSteps: maxReplayedSequenceLength * batchSize,
      maxEpisodes: maxEpisodes,
      callbacks: [{ (environment, trajectory) in
        averageEpisodeLength.update(using: trajectory)
        if step > 0 { environment.render() }
      }])
    if step % 1 == 0 {
      logger.info("Step \(step) | Loss: \(loss) | Average Episode Length: \(averageEpisodeLength.value())")
    }
  }
}
