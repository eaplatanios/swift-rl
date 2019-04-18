import Gym
import TensorFlow

public protocol Policy {
  associatedtype Action
  associatedtype Observation
  associatedtype Reward
  associatedtype State

  var batched: Bool { get }

  func initialState(for observation: Observation) -> State

  func act(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<Action, State>
}

public extension Policy {
  @inlinable var isStateless: Bool { return false }
}

public extension Policy where State == None {
  @inlinable var isStateless: Bool { return true }

  @inlinable func initialState(for observation: Observation) -> State {
    return None()
  }
}

public extension Policy where State == None {
  @inlinable func act(
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<Action, State> {
    return act(in: None(), using: step)
  }
}

/// Contains the data emitted by a policy when taking an action.
public struct PolicyStep<ActionInformation, State> {
  public let actionInformation: ActionInformation
  public let state: State

  @differentiable(where ActionInformation: Differentiable, State: Differentiable)
  public init(actionInformation: ActionInformation, state: State) {
    self.actionInformation = actionInformation
    self.state = state
  }
}

public extension PolicyStep where State == None {
  @differentiable(where ActionInformation: Differentiable)
  init(actionInformation: ActionInformation) {
    self.actionInformation = actionInformation
    self.state = None()
  }
}

extension PolicyStep: Differentiable
where ActionInformation: Differentiable, State: Differentiable { }
