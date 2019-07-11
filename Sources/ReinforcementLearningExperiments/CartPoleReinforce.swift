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
  public var dense1: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 6)
  public var dense2: Dense<Float> = Dense<Float>(inputSize: 6, outputSize: 2)

  public func initialize(using input: CartPoleEnvironment.Observation) {}

  @differentiable
  public func callAsFunction(_ input: CartPoleEnvironment.Observation) -> Categorical<Int32> {
    let stackedInput = Tensor<Float>(
      stacking: [
        input.position, input.positionDerivative,
        input.angle, input.angleDerivative],
      alongAxis: input.position.rank)
    let outerDimCount = stackedInput.rank - 1
    let flattenedBatchStackedInput = stackedInput.flattenedBatch(outerDimCount: outerDimCount)
    // let logits = dense(flattenedBatchStackedInput)
    let logits = dense2(leakyRelu(dense1(flattenedBatchStackedInput)))
    let flattenedDistribution = Categorical<Int32>(logits: logits)
    return flattenedDistribution.unflattenedBatch(
      outerDims: [Int](stackedInput.shape.dimensions[0..<outerDimCount]))
  }

  public func copy() -> ActorNetwork {
    ActorNetwork()
  }
}

public func runCartPoleReinforce() {
  let batchSize = 32
  let maxSequenceLength = 2000

  var renderer = CartPoleRenderer()
  var environment = CartPoleEnvironment(batchSize: batchSize)

  let actorPolicy = ActorPolicy(
    for: environment,
    actorNetwork: ActorNetwork(),
    randomSeed: (1234, 5678))
  var agent = ReinforceAgent(
    actorPolicy: actorPolicy,
    optimizer: AMSGrad(for: actorPolicy),
    discountFactor: 0.9,
    returnsNormalizer: { standardNormalize($0, alongAxes: 0, 1) },
    entropyRegularizationWeight: 0.0)
  var replayBuffer = UniformReplayBuffer(
    for: environment,
    using: agent.policy,
    batchSize: batchSize,
    maxLength: maxSequenceLength)
  var driver = StepBasedDriver(
    for: environment,
    using: agent.policy,
    maxEpisodes: 128,
    batchSize: batchSize)

  agent.initialize()

  for step in 0..<10000 {
    driver.run(using: environment.reset(), updating: [{ trajectoryStep in
      replayBuffer.record(trajectoryStep)
      if step > 200 {
        try! environment.render(
          observation: trajectoryStep.currentStep.observation,
          using: &renderer)
      }
    }])
    let loss = agent.update(using: replayBuffer.recordedData())
    print("Step \(step) loss: \(loss)")
    replayBuffer.reset()
  }
}
