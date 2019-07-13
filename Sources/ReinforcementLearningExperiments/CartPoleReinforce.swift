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

import ReinforcementLearning

fileprivate struct ActorNetwork: Network {
  @noDerivative public var state: None = None()

  // public var dense: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 2)
  public var dense1: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 100)
  public var dense2: Dense<Float> = Dense<Float>(inputSize: 100, outputSize: 2)
  // public var dense2Value: Dense<Float> = Dense<Float>(inputSize: 6, outputSize: 2)

  public func initialize(using input: CartPoleEnvironment.Observation) {}

  @differentiable
  public func callAsFunction(
    _ input: CartPoleEnvironment.Observation
  ) -> ReinforceNetworkOutput<Categorical<Int32>> {
    let stackedInput = Tensor<Float>(
      stacking: [
        input.position, input.positionDerivative,
        input.angle, input.angleDerivative],
      alongAxis: input.position.rank)
    let outerDimCount = stackedInput.rank - 1
    let flattenedBatchStackedInput = stackedInput.flattenedBatch(outerDimCount: outerDimCount)
    // let actionLogits = dense(flattenedBatchStackedInput)
    let hidden = leakyRelu(dense1(flattenedBatchStackedInput))
    let actionLogits = dense2(hidden)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    let actionDistribution = flattenedActionDistribution.unflattenedBatch(
      outerDims: [Int](stackedInput.shape.dimensions[0..<outerDimCount]))
    // let value = dense2Value(hidden)
    return ReinforceNetworkOutput(actionDistribution: actionDistribution)//, value: value)
  }

  public func copy() -> ActorNetwork {
    ActorNetwork()
  }
}

public func runCartPoleReinforce() {
  let batchSize = 1
  let maxEpisodes = 1

  var environment = CartPoleEnvironment(batchSize: batchSize)
  var renderer = CartPoleRenderer()

  let network = ActorNetwork()
  var agent = ReinforceAgent(
    for: environment,
    network: network,
    optimizer: AMSGrad(for: network, learningRate: 1e-3),
    maxReplayedSequenceLength: 2000,
    discountFactor: 0.9,
    // advantageFunction: EmpiricalAdvantageEstimation(),
    returnsNormalizer: { standardNormalize($0, alongAxes: 0, 1) },
    entropyRegularizationWeight: 0.0)

  // Metrics
  var averageEpisodeLength = AverageEpisodeLength(
    for: agent,
    batchSize: batchSize,
    bufferSize: 10)

  for step in 0..<10000 {
    let loss = agent.update(
      using: &environment,
      maxEpisodes: maxEpisodes,
      stepCallbacks: [{ trajectory in
        averageEpisodeLength.update(using: trajectory)
        if step > 1000 {
          try! renderer.render(trajectory.observation)
        }
      }])
    if step % 1 == 0 {
      print("Step \(step) | Loss: \(loss) | Average Episode Length: \(averageEpisodeLength.value())")
    }
  }
}