import Gym

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

  public var environment: ManagedEnvironment
  public var policy: ManagedPolicy

  public init(
    for environment: ManagedEnvironment,
    using policy: ManagedPolicy,
    maxSteps: Int,
    batchSize: Int
  ) {
    precondition(environment.batched == false, "The managed environment must not be batched.")
    precondition(policy.batched == false, "The managed policy must not be batched.")
    precondition(maxSteps > 0, "'maxSteps' must be > 0.")
    self.maxSteps = maxSteps
    self.maxEpisodes = Int.max
    self.environment = environment
    self.policy = policy
    self.batchSize = batchSize
  }

  public init(
    for environment: ManagedEnvironment,
    using policy: ManagedPolicy,
    maxEpisodes: Int,
    batchSize: Int
  ) {
    precondition(maxEpisodes > 0, "'maxEpisodes' must be > 0.")
    self.maxSteps = Int.max
    self.maxEpisodes = maxEpisodes
    self.environment = environment
    self.policy = policy
    self.batchSize = batchSize
  }

  public init(
    for environment: ManagedEnvironment,
    using policy: ManagedPolicy,
    maxSteps: Int,
    maxEpisodes: Int,
    batchSize: Int
  ) {
    precondition(maxSteps > 0 && maxEpisodes > 0, "'maxSteps' and 'maxEpisodes' must be > 0.")
    self.maxSteps = maxSteps
    self.maxEpisodes = maxEpisodes
    self.environment = environment
    self.policy = policy
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
    policy.state = policy.batched ? State.stack([State](repeating: state, count: batchSize)) : state
    var currentStep = policy.batched ? Step<Observation, Reward>.stack(
      [Step<Observation, Reward>](repeating: step, count: batchSize)) : step
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let action = policy.action(for: currentStep)
      // TODO: Unstack if the policy is batched and the environment is not.
      let nextStep = environment.step(taking: action)
      let trajectoryStep = TrajectoryStep(
        currentStep: currentStep,
        nextStep: nextStep,
        policyInformation: (action: action, state: policy.state))

      for listener in listeners {
        listener(trajectoryStep)
      }

      if policy.batched {
        numSteps += Int(Tensor<Int32>(trajectoryStep.isBoundary()).sum().scalar!)
        numEpisodes += Int(Tensor<Int32>(trajectoryStep.isLast()).sum().scalar!)
      } else {
        numSteps += Int(Tensor<Int32>(trajectoryStep.isBoundary()).scalar!)
        numEpisodes += Int(Tensor<Int32>(trajectoryStep.isLast()).scalar!)
      }

      currentStep = nextStep
    }

    return currentStep
  }
}
