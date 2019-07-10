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

public struct StepBasedDriver<ManagedEnvironment: Environment, ManagedPolicy: Policy>
where
  ManagedEnvironment.ActionSpace.Value == ManagedPolicy.Action,
  ManagedEnvironment.ObservationSpace.Value == ManagedPolicy.Observation,
  ManagedEnvironment.Reward == ManagedPolicy.Reward,
  ManagedEnvironment.ActionSpace.Value: Stackable,
  ManagedEnvironment.ActionSpace.Value.Stacked == ManagedEnvironment.ActionSpace.Value,
  ManagedEnvironment.ObservationSpace.Value: Stackable,
  ManagedEnvironment.ObservationSpace.Value.Stacked == ManagedEnvironment.ObservationSpace.Value,
  ManagedEnvironment.Reward: Stackable,
  ManagedEnvironment.Reward.Stacked == ManagedEnvironment.Reward,
  ManagedPolicy.State: Stackable,
  ManagedPolicy.State.Stacked == ManagedPolicy.State
{
  public let maxSteps: Int
  public let maxEpisodes: Int
  public let batchSize: Int

  public let batchedEnvironment: Bool
  public let batchedPolicy: Bool

  public var environments: [ManagedEnvironment]
  public var policies: [ManagedPolicy]

  public init(
    for environment: ManagedEnvironment,
    using policy: ManagedPolicy,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    batchSize: Int = 1
  ) {
    precondition(maxSteps > 0 && maxEpisodes > 0, "'maxSteps' and 'maxEpisodes' must be > 0.")
    self.maxSteps = maxSteps
    self.maxEpisodes = maxEpisodes
    self.batchedEnvironment = environment.batched
    self.batchedPolicy = policy.batched
    self.environments = batchedEnvironment ? 
      [environment] : 
      (0..<batchSize).map { _  in environment.copy() }
    self.policies = batchedPolicy ?
      [policy] :
      (0..<batchSize).map { _ in policy.copy() }
    self.batchSize = batchSize
  }
}

extension StepBasedDriver: Driver {
  @discardableResult
  public mutating func run(
    startingIn state: State,
    using step: Step<Observation, Reward>,
    updating listeners: [Listener]
  ) -> Step<Observation, Reward> {
    if batchedPolicy {
      policies[0].state = State.stack([State](repeating: state, count: batchSize))
    } else {
      policies.indices.forEach { policies[$0].state = state }
    }
    var currentStep = Step<Observation, Reward>.stack(
      [Step<Observation, Reward>](repeating: step, count: batchSize))
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      var action: ManagedPolicy.Action
      var nextStep: Step<Observation, Reward>
      var state: ManagedPolicy.State
      switch (batchedPolicy, batchedEnvironment) {
      case (true, true):
        action = policies[0].action(for: currentStep)
        nextStep = environments[0].step(taking: action)
        state = policies[0].state
      case (true, false):
        action = policies[0].action(for: currentStep)
        let actions = action.unstacked()
        let nextSteps = environments.indices.map { environments[$0].step(taking: actions[$0]) }
        nextStep = Step<Observation, Reward>.stack(nextSteps)
        state = policies[0].state
      case (false, true):
        let currentSteps = currentStep.unstacked()
        let actions = policies.indices.map { policies[$0].action(for: currentSteps[$0]) }
        action = ManagedPolicy.Action.stack(actions)
        nextStep = environments[0].step(taking: action)
        state = ManagedPolicy.State.stack(policies.map { $0.state })
      case (false, false):
        let currentSteps = currentStep.unstacked()
        let actions = policies.indices.map { policies[$0].action(for: currentSteps[$0]) }
        action = ManagedPolicy.Action.stack(actions)
        let nextSteps = environments.indices.map { environments[$0].step(taking: actions[$0]) }
        nextStep = Step<Observation, Reward>.stack(nextSteps)
        state = ManagedPolicy.State.stack(policies.map { $0.state })
      }
      let trajectoryStep = TrajectoryStep(
        currentStep: currentStep,
        nextStep: nextStep,
        policyInformation: (action: action, state: state))
      listeners.forEach { $0(trajectoryStep) }
      numSteps += Int(Tensor<Int32>(trajectoryStep.isBoundary()).sum().scalar!)
      numEpisodes += Int(Tensor<Int32>(trajectoryStep.isLast()).sum().scalar!)
      currentStep = nextStep
    }
    return currentStep
  }
}
