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

public protocol Agent {
  associatedtype Environment: ReinforcementLearning.Environment

  var actionSpace: Environment.ActionSpace { get }

  mutating func action(for step: Step<Observation, Reward>) -> Action

  /// Updates this agent, effectively performing a single training step.
  ///
  /// - Parameter trajectory: Trajectory to use for the update.
  /// - Returns: Loss function value.
  @discardableResult
  mutating func update(using trajectory: Trajectory<Observation, Action, Reward>) -> Float

  @discardableResult
  mutating func update(
    using environment: inout Environment,
    maxSteps: Int,
    maxEpisodes: Int,
    callbacks: [StepCallback<Environment>]
  ) throws -> Float
}

extension Agent {
  public typealias Observation = Environment.Observation
  public typealias Action = Environment.Action
  public typealias Reward = Environment.Reward

  @inlinable
  public mutating func run(
    in environment: inout Environment,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment>] = []
  ) throws {
    var currentStep = environment.currentStep
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let action = self.action(for: currentStep)
      let nextStep = try environment.step(taking: action)
      var trajectory = Trajectory(
        stepKind: nextStep.kind,
        observation: currentStep.observation,
        action: action,
        reward: nextStep.reward)
      callbacks.forEach { $0(&environment, &trajectory) }
      numSteps += Int((1 - Tensor<Int32>(nextStep.kind.isLast())).sum().scalarized())
      numEpisodes += Int(Tensor<Int32>(nextStep.kind.isLast()).sum().scalarized())
      currentStep = nextStep
    }
  }
}

/// Trajectory generated by having an agent interact with an environment.
///
/// Trajectories consist of five main components, each of which can be a nested structure of
/// tensors with shapes whose first two dimensions are `[T, B]`, where `T` is the length of the
/// trajectory in terms of time steps and `B` is the batch size. The five components are:
///   - `stepKind`: Represents the kind of each time step (i.e., "first", "transition", or "last").
///     For example, if the agent takes an action in time step `t` that results in the current
///     episode ending, then `stepKind[t]` will be "last" and `stepKind[t + 1]` will be "first".
///   - `observation`: Observation that the agent receives from the environment in the beginning
///     of each time step.
///   - `action`: Action the agent took in each time step.
///   - `reward`: Reward that the agent received from the environment after each action. The reward
///     received after taking `action[t]` is `reward[t]`.
public struct Trajectory<Observation, Action, Reward>: KeyPathIterable {
  // These need to be mutable because we use `KeyPathIterable.recursivelyAllWritableKeyPaths` to
  // automatically derive conformance to `Replayable`.
  public var stepKind: StepKind
  public var observation: Observation
  public var action: Action
  public var reward: Reward

  @inlinable
  public init(stepKind: StepKind, observation: Observation, action: Action, reward: Reward) {
    self.stepKind = stepKind
    self.observation = observation
    self.action = action
    self.reward = reward
  }
}

public struct AnyAgent<Environment: ReinforcementLearning.Environment>: Agent {
  public typealias Observation = Environment.Observation
  public typealias Action = Environment.Action
  public typealias Reward = Environment.Reward

  @usableFromInline internal let _actionSpace: () -> Environment.ActionSpace

  @usableFromInline internal let _action: (Step<Observation, Reward>) -> Action

  @usableFromInline internal let _updateUsingTrajectory: (
    Trajectory<Observation, Action, Reward>
  ) -> Float

  @usableFromInline internal let _updateUsingEnvironment: (
    inout Environment,
    Int,
    Int,
    [StepCallback<Environment>]
  ) throws -> Float

  public var actionSpace: Environment.ActionSpace { _actionSpace() }

  @inlinable
  public init<A: Agent>(_ agent: A) where A.Environment == Environment {
    var agent = agent
    _actionSpace = { () in agent.actionSpace }
    _action = { agent.action(for: $0) }
    _updateUsingTrajectory = { agent.update(using: $0) }
    _updateUsingEnvironment = { try agent.update(
      using: &$0,
      maxSteps: $1,
      maxEpisodes: $2,
      callbacks: $3)
    }
  }

  @inlinable
  public mutating func action(for step: Step<Observation, Reward>) -> Action {
    _action(step)
  }

  @inlinable
  @discardableResult
  public mutating func update(using trajectory: Trajectory<Observation, Action, Reward>) -> Float {
    _updateUsingTrajectory(trajectory)
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment>] = []
  ) throws -> Float {
    try _updateUsingEnvironment(&environment, maxSteps, maxEpisodes, callbacks)
  }
}

// TODO: Support `boltzman(temperature:)`.
public enum ProbabilisticAgentMode {
  case random
  case greedy
  case epsilonGreedy(_ epsilon: Float)
  case probabilistic
}

public protocol ProbabilisticAgent: Agent {
  associatedtype ActionDistribution: Distribution where ActionDistribution.Value == Action

  /// Generates the distribution over next actions given the current environment step.
  mutating func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution
}

extension ProbabilisticAgent {
  @inlinable
  public mutating func action(for step: Step<Observation, Reward>) -> Action {
    action(for: step, mode: .greedy)
  }

  /// - Note: We cannot use a default argument value for `mode` here because of the `Agent`
  ///   protocol requirement for an `Agent.action(for:)` function.
  @inlinable
  public mutating func action(
    for step: Step<Observation, Reward>,
    mode: ProbabilisticAgentMode
  ) -> Action {
    switch mode {
    case .random:
      return actionSpace.sample()
    case .greedy:
      return actionDistribution(for: step).mode()
    case let .epsilonGreedy(epsilon) where Float.random(in: 0..<1) < epsilon:
      return actionSpace.sample()
    case .epsilonGreedy(_):
      return actionDistribution(for: step).mode()
    case .probabilistic:
      return actionDistribution(for: step).sample()
    }
  }

  @inlinable
  public mutating func run(
    in environment: inout Environment,
    mode: ProbabilisticAgentMode = .greedy,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment>] = []
  ) throws {
    var currentStep = environment.currentStep
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let action = self.action(for: currentStep, mode: mode)
      let nextStep = try environment.step(taking: action)
      var trajectory = Trajectory(
        stepKind: nextStep.kind,
        observation: currentStep.observation,
        action: action,
        reward: nextStep.reward)
      callbacks.forEach { $0(&environment, &trajectory) }
      numSteps += Int((1 - Tensor<Int32>(nextStep.kind.isLast())).sum().scalarized())
      numEpisodes += Int(Tensor<Int32>(nextStep.kind.isLast()).sum().scalarized())
      currentStep = nextStep
    }
  }
}

public struct AnyProbabilisticAgent<
  Environment: ReinforcementLearning.Environment,
  ActionDistribution: Distribution
>: ProbabilisticAgent where ActionDistribution.Value == Environment.Action {
  public typealias Observation = Environment.Observation
  public typealias Action = Environment.Action
  public typealias Reward = Environment.Reward

  @usableFromInline internal let _actionSpace: () -> Environment.ActionSpace

  @usableFromInline internal let _action: (Step<Observation, Reward>) -> Action

  @usableFromInline internal let _actionDistribution: (
    Step<Observation, Reward>
  ) -> ActionDistribution

  @usableFromInline internal let _updateUsingTrajectory: (
    Trajectory<Observation, Action, Reward>
  ) -> Float

  @usableFromInline internal let _updateUsingEnvironment: (
    inout Environment,
    Int,
    Int,
    [StepCallback<Environment>]
  ) throws -> Float

  public var actionSpace: Environment.ActionSpace { _actionSpace() }

  public init<A: ProbabilisticAgent>(
    _ agent: A
  ) where A.Environment == Environment, A.ActionDistribution == ActionDistribution {
    var agent = agent
    _actionSpace = { () in agent.actionSpace }
    _action = { agent.action(for: $0) }
    _actionDistribution = { agent.actionDistribution(for: $0) }
    _updateUsingTrajectory = { agent.update(using: $0) }
    _updateUsingEnvironment = { try agent.update(
      using: &$0,
      maxSteps: $1,
      maxEpisodes: $2,
      callbacks: $3)
    }
  }

  @inlinable
  public mutating func action(for step: Step<Observation, Reward>) -> Action {
    _action(step)
  }

  @inlinable
  public mutating func actionDistribution(
    for step: Step<Observation, Reward>
  ) -> ActionDistribution {
    _actionDistribution(step)
  }

  @inlinable
  @discardableResult
  public mutating func update(using trajectory: Trajectory<Observation, Action, Reward>) -> Float {
    _updateUsingTrajectory(trajectory)
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment>] = []
  ) throws -> Float {
    try _updateUsingEnvironment(&environment, maxSteps, maxEpisodes, callbacks)
  }
}

public struct RandomAgent<Environment: ReinforcementLearning.Environment>: ProbabilisticAgent {
  public typealias Observation = Environment.ObservationSpace.Value
  public typealias Action = Environment.ActionSpace.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Reward = Environment.Reward

  public let actionSpace: Environment.ActionSpace

  @inlinable
  public init(for environment: Environment) {
    actionSpace = environment.actionSpace
  }

  @inlinable
  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    actionSpace.distribution
  }

  @inlinable
  @discardableResult
  public mutating func update(using trajectory: Trajectory<Observation, Action, Reward>) -> Float {
    0.0
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    maxSteps: Int,
    maxEpisodes: Int,
    callbacks: [StepCallback<Environment>]
  ) -> Float {
    0.0
  }
}
