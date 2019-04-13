
public protocol Policy {
  associatedtype Action
  associatedtype Observation
  associatedtype Reward
  associatedtype Discount
  associatedtype State

  func initialState() -> State

  mutating func act(
    in state: State,
    using step: EnvironmentStep<Observation, Reward, Discount>
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
  mutating func act(
    using step: EnvironmentStep<Observation, Reward, Discount>
  ) -> PolicyStep<Action, State> {
    return act(in: None(), using: step)
  }
}

public protocol BatchedPolicy: Policy {
  func initialState(batchSize: Int) -> State

  mutating func act(
    in state: State,
    using step: EnvironmentStep<Observation, Reward, Discount>,
    batchSize: Int
  ) -> PolicyStep<Action, State>
}

public extension BatchedPolicy {
  @inlinable
  func initialState() -> State {
    return initialState(batchSize: 1)
  }

  @inlinable
  mutating func act(
    in state: State,
    using step: EnvironmentStep<Observation, Reward, Discount>
  ) -> PolicyStep<Action, State> {
    return act(in: state, using: step, batchSize: 1)
  }
}

public extension BatchedPolicy where State == None {
  @inlinable
  mutating func act(
    using step: EnvironmentStep<Observation, Reward, Discount>,
    batchSize: Int
  ) -> PolicyStep<Action, State> {
    return act(in: None(), using: step, batchSize: batchSize)
  }

  @inlinable
  mutating func act(
    using step: EnvironmentStep<Observation, Reward, Discount>
  ) -> PolicyStep<Action, State> {
    return act(in: None(), using: step, batchSize: 1)
  }
}

public struct None {
  public init() { }
}

/// Contains the data emitted by a policy when taking an action.
public struct PolicyStep<Action, State> {
  public let action: Action
  public let state: State
  // TODO: Policy info.
  // TODO: Action probability.
}

public extension PolicyStep where State == None {
  init(action: Action) {
    self.action = action
    self.state = None()
  }
}
