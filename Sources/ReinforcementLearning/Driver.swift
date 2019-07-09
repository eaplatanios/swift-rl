import TensorFlow

/// A driver takes steps in an environment using the provided policy.
public protocol Driver {
  associatedtype ManagedEnvironment: Environment
  associatedtype ManagedPolicy: Policy
    where ManagedPolicy.Action == ManagedEnvironment.ActionSpace.Value,
          ManagedPolicy.Observation == ManagedEnvironment.ObservationSpace.Value,
          ManagedPolicy.Reward == ManagedEnvironment.Reward

  typealias Action = ManagedPolicy.Action
  typealias Observation = ManagedPolicy.Observation
  typealias Reward = ManagedPolicy.Reward
  typealias State = ManagedPolicy.State

  typealias Listener = (TrajectoryStep<Action, Observation, Reward, State>) -> Void

  var environment: ManagedEnvironment { get }
  var policy: ManagedPolicy { get }

  /// Takes steps in the managed environment using the managed policy.
  @discardableResult
  mutating func run(
    startingIn state: State,
    using step: Step<Observation, Reward>,
    updating listeners: [Listener]
  ) -> Step<Observation, Reward>
}

public extension Driver where State == None {
  @inlinable @discardableResult
  mutating func run(
    using step: Step<Observation, Reward>,
    updating listeners: [Listener]
  ) -> Step<Observation, Reward> {
    run(startingIn: None(), using: step, updating: listeners)
  }
}

public typealias Trajectory<Action, Observation, Reward, State> = 
  [TrajectoryStep<Action, Observation, Reward, State>]

public struct TrajectoryStep<Action, Observation, Reward, State> {
  public let currentStep: Step<Observation, Reward>
  public let nextStep: Step<Observation, Reward>
  public let policyInformation: (action: Action, state: State)

  @inlinable
  public func isFirst() -> Tensor<Bool> {
    currentStep.kind.isFirst()
  }

  @inlinable
  public func isTransition() -> Tensor<Bool> {
    currentStep.kind.isTransition().elementsLogicalAnd(nextStep.kind.isTransition())
  }

  @inlinable
  public func isLast() -> Tensor<Bool> {
    nextStep.kind.isLast()
  }

  @inlinable
  public func isBoundary() -> Tensor<Bool> {
    currentStep.kind.isLast()
  }
}
