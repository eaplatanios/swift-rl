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

public protocol Agent {
  associatedtype Action
  associatedtype Observation
  associatedtype Reward
  associatedtype State

  var batched: Bool { get }
  var state: State { get set }

  /// Initializes this agent.
  mutating func initialize()

  /// Trains this agent using the provided experience.
  /// - Returns: Loss function value.
  @discardableResult
  mutating func update(using trajectory: Trajectory<Action, Observation, Reward, State>) -> Float

  func action(for step: Step<Observation, Reward>) -> Action
}

extension Agent {
  public mutating func initialize() {}

  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Action, Observation, Reward, State>
  ) -> Float {
    0.0
  }
}

public protocol ProbabilisticAgent: Agent {
  associatedtype ActionDistribution: Distribution where ActionDistribution.Value == Action

  /// Generates the distribution over next actions given the current environment step.
  func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution
}

public extension ProbabilisticAgent {
  func action(for step: Step<Observation, Reward>) -> Action {
    actionDistribution(for: step).sample()
  }
}

public struct RandomAgent<Environment: ReinforcementLearning.Environment>: ProbabilisticAgent {
  public typealias Action = Environment.ActionSpace.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Observation = Environment.ObservationSpace.Value
  public typealias Reward = Environment.Reward
  public typealias State = None

  public let batched: Bool = false
  public let environment: Environment

  public var state: None = None()

  public init(for environment: Environment) {
    self.environment = environment
  }

  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    environment.actionSpace.distribution
  }
}
