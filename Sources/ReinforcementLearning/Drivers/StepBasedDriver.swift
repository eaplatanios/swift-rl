
public struct StepBasedDriver<
  ManagedEnvironment: Environment,
  ManagedPolicy: Policy
>: Driver where ManagedEnvironment.Action == ManagedPolicy.Action,
                ManagedEnvironment.Observation == ManagedPolicy.Observation,
                ManagedEnvironment.Reward == ManagedPolicy.Reward,
                ManagedEnvironment.Discount == ManagedPolicy.Discount {
  public let maxSteps: UInt
  public let maxEpisodes: UInt

  public var environment: ManagedEnvironment
  public var policy: ManagedPolicy

  public init(for environment: ManagedEnvironment, using policy: ManagedPolicy, maxSteps: UInt) {
    precondition(maxSteps > 0, "The maximum number of steps must be greater than 0.")
    self.maxSteps = maxSteps
    self.maxEpisodes = UInt.max
    self.environment = environment
    self.policy = policy
  }

  public init(for environment: ManagedEnvironment, using policy: ManagedPolicy, maxEpisodes: UInt) {
    precondition(maxEpisodes > 0, "The maximum number of episodes must be greater than 0.")
    self.maxSteps = UInt.max
    self.maxEpisodes = maxEpisodes
    self.environment = environment
    self.policy = policy
  }

  public init(
    for environment: ManagedEnvironment,
    using policy: ManagedPolicy,
    maxSteps: UInt,
    maxEpisodes: UInt
  ) {
    precondition(maxSteps > 0 && maxEpisodes > 0, 
                 "The maximum number of steps and episodes must both be greater than 0.")
    self.maxSteps = maxSteps
    self.maxEpisodes = maxEpisodes
    self.environment = environment
    self.policy = policy
  }

  @discardableResult
  public mutating func run(
    startingIn state: State,
    using step: EnvironmentStep<Observation, Reward, Discount>,
    updating listeners: [Listener]
  ) -> (environmentStep: EnvironmentStep<Observation, Reward, Discount>, policyState: State) {
    var currentState = state
    var currentEnvironmentStep = step
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let policyStep = policy.act(in: currentState, using: currentEnvironmentStep)
      let nextEnvironmentStep = environment.step(taking: policyStep.action)
      let trajectoryStep = TrajectoryStep(
        currentEnvironmentStep: currentEnvironmentStep,
        nextEnvironmentStep: nextEnvironmentStep,
        policyStep: policyStep)
      for listener in listeners {
        listener(trajectoryStep)
      }

      numSteps += trajectoryStep.isBoundary() ? 0 : 1
      numEpisodes += trajectoryStep.isLast() ? 1 : 0

      currentState = policyStep.state
      currentEnvironmentStep = nextEnvironmentStep
    }
    return (environmentStep: currentEnvironmentStep, policyState: currentState)
  }
}
