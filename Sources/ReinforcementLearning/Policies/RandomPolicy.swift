
/// Represents a policy that takes completely random actions.
public struct RandomPolicy<ManagedEnvironment: Environment>: ProbabilisticPolicy {
  public typealias Action = ManagedEnvironment.ActionSpace.Value
  public typealias ActionDistribution = ManagedEnvironment.ActionSpace.ValueDistribution
  public typealias Observation = ManagedEnvironment.ObservationSpace.Value
  public typealias Reward = ManagedEnvironment.Reward
  public typealias State = None

  public let batched: Bool = false

  public let environment: ManagedEnvironment
  public let randomSeed: TensorFlowSeed

  public var state: None = None()

  public init(
    for environment: ManagedEnvironment,
    randomSeed: TensorFlowSeed = Context.local.randomSeed
  ) {
    self.environment = environment
    self.randomSeed = randomSeed
  }

  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    environment.actionSpace.distribution
  }

  public func copy() -> RandomPolicy {
    RandomPolicy(for: environment, randomSeed: randomSeed)
  }
}
