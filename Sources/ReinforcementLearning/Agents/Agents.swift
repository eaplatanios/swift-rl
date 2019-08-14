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

public typealias StepCallback<E: Environment, State> =
  (inout E, inout Trajectory<E.Observation, State, E.Action, E.Reward>) -> Void

public struct ActionStatePair<Action, State> {
  public var action: Action
  public var state: State

  @inlinable
  public init(action: Action,  state: State) {
    self.action = action
    self.state = state
  }
}

extension ActionStatePair: Differentiable where Action: Differentiable, State: Differentiable {}

public protocol Agent {
  associatedtype Environment: ReinforcementLearning.Environment
  associatedtype State

  var actionSpace: Environment.ActionSpace { get }

  // TODO: !!!! Need an `initialState` computed property.

  func action(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionStatePair<Action, State>

  /// Updates this agent, effectively performing a single training step.
  ///
  /// - Parameter trajectory: Trajectory to use for the update.
  /// - Returns: Loss function value.
  @discardableResult
  mutating func update(using trajectory: Trajectory<Observation, State, Action, Reward>) -> Float

  @discardableResult
  mutating func update(
    using environment: inout Environment,
    initialState: State,
    maxSteps: Int,
    maxEpisodes: Int,
    callbacks: [StepCallback<Environment, State>]
  ) throws -> Float
}

extension Agent where State == Empty {
  @inlinable
  public func action(for step: Step<Observation, Reward>) -> ActionStatePair<Action, State> {
    action(for: step, in: Empty())
  }

  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    maxSteps: Int,
    maxEpisodes: Int,
    callbacks: [StepCallback<Environment, State>]
  ) throws -> Float {
    try update(
      using: &environment,
      initialState: Empty(),
      maxSteps: maxSteps,
      maxEpisodes: maxEpisodes,
      callbacks: callbacks)
  }
}

// public protocol StatelessAgent: Agent where State == Empty {
//   func action(for step: Step<Observation, Reward>) -> Action
  
//   @discardableResult
//   mutating func update(
//     using environment: inout Environment,
//     maxSteps: Int,
//     maxEpisodes: Int,
//     callbacks: [StepCallback<Environment, State>]
//   ) throws -> Float
// }

// extension StatelessAgent {
//   @inlinable
//   public func action(
//     for step: Step<Observation, Reward>,
//     in state: State
//   ) -> ActionStatePair<Action, State> {
//     ActionStatePair(action: action(for: step), state: Empty())
//   }

//   @discardableResult
//   public mutating func update(
//     using environment: inout Environment,
//     initialState: State,
//     maxSteps: Int,
//     maxEpisodes: Int,
//     callbacks: [StepCallback<Environment, State>]
//   ) throws -> Float {
//     try update(
//       using: &environment,
//       maxSteps: maxSteps,
//       maxEpisodes: maxEpisodes,
//       callbacks: callbacks)
//   }
// }

extension Agent {
  public typealias Observation = Environment.Observation
  public typealias Action = Environment.Action
  public typealias Reward = Environment.Reward

  @inlinable
  public func run(
    in environment: inout Environment,
    initialState: State,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment, State>] = []
  ) throws {
    var currentStep = environment.currentStep
    var state = initialState
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let actionStatePair = self.action(for: currentStep, in: state)
      let nextStep = try environment.step(taking: actionStatePair.action)
      var trajectory = Trajectory(
        stepKind: nextStep.kind,
        observation: currentStep.observation,
        state: state,
        action: actionStatePair.action,
        reward: nextStep.reward)
      callbacks.forEach { $0(&environment, &trajectory) }
      numSteps += Int((1 - Tensor<Int32>(nextStep.kind.isLast())).sum().scalarized())
      numEpisodes += Int(Tensor<Int32>(nextStep.kind.isLast()).sum().scalarized())
      currentStep = nextStep
      state = actionStatePair.state
    }
  }
}

extension Agent where State == Empty {
  @inlinable
  public func run(
    in environment: inout Environment,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment, State>] = []
  ) throws {
    try run(
      in: &environment,
      initialState: Empty(),
      maxSteps: maxSteps,
      maxEpisodes: maxEpisodes,
      callbacks: callbacks)
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
///   - `state`: State of the agent at the beginning of each time step.
///   - `action`: Action the agent took in each time step.
///   - `reward`: Reward that the agent received from the environment after each action. The reward
///     received after taking `action[t]` is `reward[t]`.
public struct Trajectory<Observation, State, Action, Reward>: KeyPathIterable {
  // These need to be mutable because we use `KeyPathIterable.recursivelyAllWritableKeyPaths` to
  // automatically derive conformance to `Replayable`.
  public var stepKind: StepKind
  public var observation: Observation
  public var state: State
  public var action: Action
  public var reward: Reward

  @inlinable
  public init(
    stepKind: StepKind,
    observation: Observation,
    state: State,
    action: Action,
    reward: Reward
  ) {
    self.stepKind = stepKind
    self.observation = observation
    self.state = state
    self.action = action
    self.reward = reward
  }
}

public struct AnyAgent<Environment: ReinforcementLearning.Environment, State>: Agent {
  public typealias Observation = Environment.Observation
  public typealias Action = Environment.Action
  public typealias Reward = Environment.Reward

  @usableFromInline internal let _actionSpace: () -> Environment.ActionSpace

  @usableFromInline internal let _action: (
    Step<Observation, Reward>,
    State
  ) -> ActionStatePair<Action, State>

  @usableFromInline internal let _updateUsingTrajectory: (
    Trajectory<Observation, State, Action, Reward>
  ) -> Float

  @usableFromInline internal let _updateUsingEnvironment: (
    inout Environment,
    State,
    Int,
    Int,
    [StepCallback<Environment, State>]
  ) throws -> Float

  public var actionSpace: Environment.ActionSpace { _actionSpace() }

  @inlinable
  public init<A: Agent>(_ agent: A) where A.Environment == Environment, A.State == State {
    var agent = agent
    _actionSpace = { () in agent.actionSpace }
    _action = { agent.action(for: $0, in: $1) }
    _updateUsingTrajectory = { agent.update(using: $0) }
    _updateUsingEnvironment = { try agent.update(
      using: &$0,
      initialState: $1,
      maxSteps: $2,
      maxEpisodes: $3,
      callbacks: $4)
    }
  }

  @inlinable
  public func action(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionStatePair<Action, State> {
    _action(step, state)
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Observation, State, Action, Reward>
  ) -> Float {
    _updateUsingTrajectory(trajectory)
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    initialState: State,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment, State>] = []
  ) throws -> Float {
    try _updateUsingEnvironment(&environment, initialState, maxSteps, maxEpisodes, callbacks)
  }
}

// TODO: Support `boltzman(temperature:)`.
public enum ProbabilisticAgentMode {
  case random
  case greedy
  case epsilonGreedy(_ epsilon: Float)
  case probabilistic
}

public struct ActionDistributionStatePair<ActionDistribution: Distribution, State> {
  public let actionDistribution: ActionDistribution
  public let state: State

  @inlinable
  public init(actionDistribution: ActionDistribution,  state: State) {
    self.actionDistribution = actionDistribution
    self.state = state
  }
}

public protocol ProbabilisticAgent: Agent {
  associatedtype ActionDistribution: Distribution where ActionDistribution.Value == Action

  /// Generates the distribution over next actions given the current environment step.
  func actionDistribution(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionDistributionStatePair<ActionDistribution, State>
}

extension ProbabilisticAgent {
  @inlinable
  public func action(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionStatePair<Action, State> {
    action(for: step, in: state, mode: .greedy)
  }

  /// - Note: We cannot use a default argument value for `mode` here because of the `Agent`
  ///   protocol requirement for an `Agent.action(for:)` function.
  @inlinable
  public func action(
    for step: Step<Observation, Reward>,
    in state: State,
    mode: ProbabilisticAgentMode
  ) -> ActionStatePair<Action, State> {
    switch mode {
    case .random:
      return ActionStatePair(action: actionSpace.sample(), state: state)
    case .greedy:
      let pair = actionDistribution(for: step, in: state)
      return ActionStatePair(action: pair.actionDistribution.mode(), state: pair.state)
    case let .epsilonGreedy(epsilon) where Float.random(in: 0..<1) < epsilon:
      return ActionStatePair(action: actionSpace.sample(), state: state)
    case .epsilonGreedy(_):
      let pair = actionDistribution(for: step, in: state)
      return ActionStatePair(action: pair.actionDistribution.mode(), state: pair.state)
    case .probabilistic:
      let pair = actionDistribution(for: step, in: state)
      return ActionStatePair(action: pair.actionDistribution.sample(), state: pair.state)
    }
  }

  @inlinable
  public func run(
    in environment: inout Environment,
    initialState: State,
    mode: ProbabilisticAgentMode = .greedy,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment, State>] = []
  ) throws {
    var currentStep = environment.currentStep
    var state = initialState
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let actionStatePair = self.action(for: currentStep, in: state, mode: mode)
      let nextStep = try environment.step(taking: actionStatePair.action)
      var trajectory = Trajectory(
        stepKind: nextStep.kind,
        observation: currentStep.observation,
        state: state,
        action: actionStatePair.action,
        reward: nextStep.reward)
      callbacks.forEach { $0(&environment, &trajectory) }
      numSteps += Int((1 - Tensor<Int32>(nextStep.kind.isLast())).sum().scalarized())
      numEpisodes += Int(Tensor<Int32>(nextStep.kind.isLast()).sum().scalarized())
      currentStep = nextStep
      state = actionStatePair.state
    }
  }
}

extension ProbabilisticAgent where State == Empty {
  @inlinable
  public func action(for step: Step<Observation, Reward>) -> ActionStatePair<Action, State> {
    action(for: step, in: Empty(), mode: .greedy)
  }

  @inlinable
  public func actionDistribution(
    for step: Step<Observation, Reward>
  ) -> ActionDistributionStatePair<ActionDistribution, State> {
    actionDistribution(for: step, in: Empty())
  }

  @inlinable
  public func action(for step: Step<Observation, Reward>) -> Action {
    action(for: step, in: Empty(), mode: .greedy).action
  }

  @inlinable
  public func action(
    for step: Step<Observation, Reward>,
    mode: ProbabilisticAgentMode
  ) -> Action {
    action(for: step, in: Empty(), mode: mode).action
  }

  @inlinable
  public func run(
    in environment: inout Environment,
    mode: ProbabilisticAgentMode = .greedy,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment, State>] = []
  ) throws {
    try run(
      in: &environment,
      initialState: Empty(),
      mode: mode,
      maxSteps: maxSteps,
      maxEpisodes: maxEpisodes,
      callbacks: callbacks)
  }
}

// public protocol StatelessProbabilisticAgent: StatelessAgent & ProbabilisticAgent {
//   func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution
// }

// extension StatelessProbabilisticAgent {
//   @inlinable
//   public func actionDistribution(
//     for step: Step<Observation, Reward>,
//     in state: State
//   ) -> ActionDistributionStatePair<ActionDistribution, State> {
//     ActionDistributionStatePair(
//       actionDistribution: actionDistribution(for: step),
//       state: Empty())
//   }
// }

public struct AnyProbabilisticAgent<
  Environment: ReinforcementLearning.Environment,
  ActionDistribution: Distribution,
  State
>: ProbabilisticAgent where ActionDistribution.Value == Environment.Action {
  public typealias Observation = Environment.Observation
  public typealias Action = Environment.Action
  public typealias Reward = Environment.Reward

  @usableFromInline internal let _actionSpace: () -> Environment.ActionSpace

  @usableFromInline internal let _action: (
    Step<Observation, Reward>,
    State
  ) -> ActionStatePair<Action, State>

  @usableFromInline internal let _actionDistribution: (
    Step<Observation, Reward>,
    State
  ) -> ActionDistributionStatePair<ActionDistribution, State>

  @usableFromInline internal let _updateUsingTrajectory: (
    Trajectory<Observation, State, Action, Reward>
  ) -> Float

  @usableFromInline internal let _updateUsingEnvironment: (
    inout Environment,
    State,
    Int,
    Int,
    [StepCallback<Environment, State>]
  ) throws -> Float

  public var actionSpace: Environment.ActionSpace { _actionSpace() }

  public init<A: ProbabilisticAgent>(_ agent: A) where
    A.Environment == Environment,
    A.ActionDistribution == ActionDistribution,
    A.State == State
  {
    var agent = agent
    _actionSpace = { () in agent.actionSpace }
    _action = { agent.action(for: $0, in: $1) }
    _actionDistribution = { agent.actionDistribution(for: $0, in: $1) }
    _updateUsingTrajectory = { agent.update(using: $0) }
    _updateUsingEnvironment = { try agent.update(
      using: &$0,
      initialState: $1,
      maxSteps: $2,
      maxEpisodes: $3,
      callbacks: $4)
    }
  }

  @inlinable
  public func action(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionStatePair<Action, State> {
    _action(step, state)
  }

  @inlinable
  public func actionDistribution(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionDistributionStatePair<ActionDistribution, State> {
    _actionDistribution(step, state)
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Observation, State, Action, Reward>
   ) -> Float {
    _updateUsingTrajectory(trajectory)
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    initialState: State,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment, State>] = []
  ) throws -> Float {
    try _updateUsingEnvironment(&environment, initialState, maxSteps, maxEpisodes, callbacks)
  }
}

public struct RandomAgent<Environment: ReinforcementLearning.Environment>: ProbabilisticAgent {
  public typealias Observation = Environment.ObservationSpace.Value
  public typealias State = Empty
  public typealias Action = Environment.ActionSpace.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Reward = Environment.Reward

  public let actionSpace: Environment.ActionSpace

  @inlinable
  public init(for environment: Environment) {
    actionSpace = environment.actionSpace
  }

  @inlinable
  public func actionDistribution(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionDistributionStatePair<ActionDistribution, Empty> {
    ActionDistributionStatePair(
      actionDistribution: actionSpace.distribution,
      state: Empty())
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Observation, State, Action, Reward>
  ) -> Float {
    0.0
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    initialState: State,
    maxSteps: Int,
    maxEpisodes: Int,
    callbacks: [StepCallback<Environment, State>]
  ) -> Float {
    0.0
  }
}
