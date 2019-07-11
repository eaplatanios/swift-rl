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

/// Represents a policy that takes completely random actions.
public struct RandomPolicy<ManagedEnvironment: Environment>: ProbabilisticPolicy {
  public typealias Action = ManagedEnvironment.ActionSpace.Value
  public typealias ActionDistribution = ManagedEnvironment.ActionSpace.ValueDistribution
  public typealias Observation = ManagedEnvironment.ObservationSpace.Value
  public typealias Reward = ManagedEnvironment.Reward
  public typealias State = None

  public let batched: Bool = false

  public let environment: ManagedEnvironment
  public let randomSeed: TensorFlowSeed

  public var state: None = None()

  public init(
    for environment: ManagedEnvironment,
    randomSeed: TensorFlowSeed = Context.local.randomSeed
  ) {
    self.environment = environment
    self.randomSeed = randomSeed
  }

  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    environment.actionSpace.distribution
  }

  public func copy() -> RandomPolicy {
    RandomPolicy(for: environment, randomSeed: randomSeed)
  }
}
