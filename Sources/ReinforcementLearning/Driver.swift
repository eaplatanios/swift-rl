
/// A driver takes steps in an environment using the provided policy.
public protocol Driver {
  associatedtype ManagedEnvironment: Environment
  associatedtype ManagedPolicy: Policy
    where ManagedPolicy.Action == ManagedEnvironment.Action,
          ManagedPolicy.Observation == ManagedEnvironment.Observation,
          ManagedPolicy.Reward == ManagedEnvironment.Reward

  typealias Action = ManagedEnvironment.Action
  typealias Observation = ManagedEnvironment.Observation
  typealias Reward = ManagedEnvironment.Reward
  typealias State = ManagedPolicy.State

  typealias Listener = (TrajectoryStep<Action, Observation, Reward, State>) -> Void

  var environment: ManagedEnvironment { get }
  var policy: ManagedPolicy { get }

  /// Takes steps in the managed environment using the managed policy.
  @discardableResult
  mutating func run(
    startingIn state: State,
    using step: EnvironmentStep<Observation, Reward>,
    updating listeners: [Listener]
  ) -> (environmentStep: EnvironmentStep<Observation, Reward>, policyState: State)
}

public extension Driver where State == None {
  @inlinable @discardableResult
  mutating func run(
    using step: EnvironmentStep<Observation, Reward>,
    updating listeners: [Listener]
  ) -> EnvironmentStep<Observation, Reward> {
    return run(startingIn: None(), using: step, updating: listeners).environmentStep
  }
}

public struct TrajectoryStep<Action, Observation, Reward, State> {
  public let currentEnvironmentStep: EnvironmentStep<Observation, Reward>
  public let nextEnvironmentStep: EnvironmentStep<Observation, Reward>
  public let policyStep: PolicyStep<Action, State>

  @inlinable
  public func isFirst() -> Bool {
    return currentEnvironmentStep.kind == .first
  }

  @inlinable
  public func isTransition() -> Bool {
    return currentEnvironmentStep.kind == .transition && nextEnvironmentStep.kind == .transition
  }

  @inlinable
  public func isLast() -> Bool {
    return nextEnvironmentStep.kind == .last
  }

  @inlinable
  public func isBoundary() -> Bool {
    return currentEnvironmentStep.kind == .last
  }
}
