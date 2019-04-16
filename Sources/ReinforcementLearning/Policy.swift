import TensorFlow

public protocol Policy {
  associatedtype Action
  associatedtype Observation
  associatedtype Reward
  associatedtype State

  func initialState() -> State

  mutating func act(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<Action, State>
}

public extension Policy {
  @inlinable var isStateless: Bool { return false }
}

public extension Policy where State == None {
  @inlinable var isStateless: Bool { return true }

  @inlinable func initialState() -> State {
    return None()
  }
}

public extension Policy where State == None {
  @inlinable
  mutating func act(using step: EnvironmentStep<Observation, Reward>) -> PolicyStep<Action, State> {
    return act(in: None(), using: step)
  }
}

/// Contains the data emitted by a policy when taking an action.
public struct PolicyStep<ActionInformation, State> {
  public let actionInformation: ActionInformation
  public let state: State

  public init(actionInformation: ActionInformation, state: State) {
    self.actionInformation = actionInformation
    self.state = state
  }
}

public extension PolicyStep where State == None {
  init(actionInformation: ActionInformation) {
    self.actionInformation = actionInformation
    self.state = None()
  }
}

// extension PolicyStep: Differentiable where Action: Differentiable, State: Differentiable { }

//public protocol BatchedPolicy: Policy
//where
//  Action: Batchable,
//  Observation: Batchable,
//  Reward: Batchable,
//  State: Batchable
//{
//  func batchedInitialState(batchSize: Int) -> State.Batched
//
//  mutating func batchedAct(
//    in state: State.Batched,
//    using step: BatchedEnvironmentStep<Observation, Reward>
//  ) -> PolicyStep<Action.Batched, State.Batched>
//}
//
//public extension BatchedPolicy {
//  @inlinable
//  func initialState() -> State {
//    return State.unbatch(batchedInitialState(batchSize: 1))[0]
//  }
//
//  @inlinable
//  mutating func act(
//    in state: State,
//    using step: EnvironmentStep<Observation, Reward>
//  ) -> PolicyStep<Action, State> {
//    let batchedStep = batchedAct(in: State.batch([state]), using: step.batched())
//    return PolicyStep(
//      action: Action.unbatch(batchedStep.action)[0],
//      state: State.unbatch(batchedStep.state)[0])
//  }
//}
//
//public extension BatchedPolicy where State == None {
//  @inlinable
//  mutating func batchedAct(
//    using step: BatchedEnvironmentStep<Observation, Reward>
//  ) -> PolicyStep<Action.Batched, State.Batched> {
//    return batchedAct(in: None(), using: step)
//  }
//}
