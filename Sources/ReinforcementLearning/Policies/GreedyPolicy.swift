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

public struct GreedyPolicy<
  Scalar: TensorFlowScalar & Equatable,
  WrappedPolicy: ProbabilisticPolicy
>: ProbabilisticPolicy where WrappedPolicy.Action == Tensor<Scalar> {
  public typealias Action = WrappedPolicy.Action
  public typealias Observation = WrappedPolicy.Observation
  public typealias Reward = WrappedPolicy.Reward
  public typealias State = WrappedPolicy.State

  public private(set) var wrappedPolicy: WrappedPolicy

  public let randomSeed: TensorFlowSeed

  public var batched: Bool { wrappedPolicy.batched }

  public var state: State {
    get { wrappedPolicy.state }
    set { wrappedPolicy.state = newValue }
  }

  public init(wrapping wrappedPolicy: WrappedPolicy) {
    self.wrappedPolicy = wrappedPolicy
    self.randomSeed = wrappedPolicy.randomSeed
  }

  public func initialize(using observation: Observation) {
    wrappedPolicy.initialize(using: observation)
  }

  public func actionDistribution(for step: Step<Observation, Reward>) -> Deterministic<Scalar> {
    let distribution = wrappedPolicy.actionDistribution(for: step)
    let action = distribution.mode()
    return Deterministic(at: action)
  }

  public func copy() -> GreedyPolicy {
    var policy = GreedyPolicy(wrapping: wrappedPolicy.copy())
    policy.state = state
    return policy
  }
}
