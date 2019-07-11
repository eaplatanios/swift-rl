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

import TensorFlow

public struct ActorPolicy<
  Environment: ReinforcementLearning.Environment,
  ActorNetwork: Network
>: ProbabilisticPolicy, Differentiable, KeyPathIterable
where
  ActorNetwork.Input == Environment.Observation,
  ActorNetwork.Output: DifferentiableDistribution
{
  public typealias ActionDistribution = ActorNetwork.Output
  public typealias Action = ActorNetwork.Output.Value
  public typealias Observation = Environment.Observation
  public typealias Reward = Environment.Reward
  public typealias State = ActorNetwork.State

  public private(set) var actorNetwork: ActorNetwork

  @noDerivative public let environment: Environment
  @noDerivative public let batched: Bool = true
  @noDerivative public let observationsNormalizer: Normalizer<Observation>
  @noDerivative public let randomSeed: TensorFlowSeed

  @noDerivative public var state: State {
    get { actorNetwork.state }
    set { actorNetwork.state = newValue }
  }

  public init(
    for environment: Environment,
    actorNetwork: ActorNetwork,
    observationsNormalizer: @escaping Normalizer<Observation> = { $0 },
    randomSeed: TensorFlowSeed = Context.local.randomSeed
  ) {
    self.environment = environment
    self.actorNetwork = actorNetwork
    self.observationsNormalizer = observationsNormalizer
    self.randomSeed = randomSeed
  }

  @inlinable
  public func initialize(using observation: Observation) {
    actorNetwork.initialize(using: observation)
  }

  @inlinable
  @differentiable(wrt: self)
  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    // actorNetwork(observationsNormalizer(step.observation))
    actorNetwork(step.observation)
  }

  public func copy() -> ActorPolicy {
    ActorPolicy(
      for: environment,
      actorNetwork: actorNetwork.copy(),
      observationsNormalizer: observationsNormalizer,
      randomSeed: randomSeed)
  }
}
