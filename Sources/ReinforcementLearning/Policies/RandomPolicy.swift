
/// Represents a policy that takes completely random actions.
public struct RandomPolicy<Observation, Reward, ActionSpace: Space>: ProbabilisticPolicy {
  public typealias Action = ActionSpace.Value
  public typealias State = None

  public let batched: Bool = false

  public let randomSeed: TensorFlowSeed
  public let actionSpace: ActionSpace

  public var state: None = None()

  public init<E: Environment>(
    for environment: E,
    randomSeed: TensorFlowSeed = Context.local.randomSeed
  ) where E.Reward == Reward,
          E.ActionSpace == ActionSpace,
          E.ObservationSpace.Value == Observation {
    self.actionSpace = environment.actionSpace
    self.randomSeed = randomSeed
  }

  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionSpace.ValueDistribution {
    actionSpace.distribution
  }
}
