
/// A driver takes steps in an environment using the provided policy.
public protocol Driver {
  associatedtype ManagedEnvironment: Environment
  associatedtype ManagedPolicy: Policy
    where ManagedPolicy.Action == ManagedEnvironment.Action,
          ManagedPolicy.Observation == ManagedEnvironment.Observation,
          ManagedPolicy.Reward == ManagedEnvironment.Reward,
          ManagedPolicy.Discount == ManagedEnvironment.Discount

  var environment: ManagedEnvironment { get }
  var policy: ManagedPolicy { get }

  /// Takes steps in the managed environment using the managed policy.
  @discardableResult
  mutating func run(
    startingIn state: State,
    using step: EnvironmentStep<Observation, Reward, Discount>,
    updating listeners: [Listener]
  ) -> (environmentStep: EnvironmentStep<Observation, Reward, Discount>, policyState: State)
}

public extension Driver {
  typealias Action = ManagedEnvironment.Action
  typealias Observation = ManagedEnvironment.Observation
  typealias Reward = ManagedEnvironment.Reward
  typealias Discount = ManagedEnvironment.Discount
  typealias State = ManagedPolicy.State

  typealias Listener = (TrajectoryStep<Action, Observation, Reward, Discount, State>) -> Void
}

public extension Driver where State == None {
  @discardableResult
  mutating func run(
    using step: EnvironmentStep<Observation, Reward, Discount>,
    updating listeners: [Listener]
  ) -> EnvironmentStep<Observation, Reward, Discount> {
    return run(startingIn: None(), using: step, updating: listeners).environmentStep
  }
}

public struct TrajectoryStep<Action, Observation, Reward, Discount, State> {
  public let currentEnvironmentStep: EnvironmentStep<Observation, Reward, Discount>
  public let nextEnvironmentStep: EnvironmentStep<Observation, Reward, Discount>
  public let policyStep: PolicyStep<Action, State>

  @inlinable
  public func isFirst() -> Bool {
    return currentEnvironmentStep.isFirst()
  }

  @inlinable
  public func isTransition() -> Bool {
    return currentEnvironmentStep.isTransition() && nextEnvironmentStep.isTransition()
  }

  @inlinable
  public func isLast() -> Bool {
    return nextEnvironmentStep.isLast()
  }

  @inlinable
  public func isBoundary() -> Bool {
    return currentEnvironmentStep.isLast()
  }
}
