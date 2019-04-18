import Gym

/// Represents a policy that takes completely random actions.
public struct RandomPolicy<Observation, Reward, ActionSpace: Space>: ProbabilisticPolicy {
  public typealias ActionDistribution = ActionSpace.ValueDistribution
  public typealias Action = ActionSpace.Value
  public typealias State = None

  public let batched: Bool = false

  public let randomSeed: UInt64?
  public let actionSpace: ActionSpace

  public init<E: Environment>(
    for environment: E,
    randomSeed: UInt64? = nil
  ) where E.Action == Action,
          E.Observation == Observation,
          E.Reward == Reward,
          E.ActionSpace == ActionSpace {
    self.actionSpace = environment.actionSpace
    self.randomSeed = hashSeed(createSeed(using: randomSeed))
  }

  public func actionDistribution(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<ActionDistribution, State> {
    return PolicyStep(actionInformation: actionSpace.distribution, state: None())
  }
}
