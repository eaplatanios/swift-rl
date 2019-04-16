import Gym

/// Represents a policy that takes completely random actions.
public struct RandomPolicy<Action, Observation, Reward, ActionSpace: Space>: ProbabilisticPolicy
  where ActionSpace.Value == Action {
  public typealias State = None

  public let randomSeed: UInt64?

  @usableFromInline internal let actionSpace: ActionSpace

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

  @inlinable
  public func actionDistribution(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<ActionSpace.ValueDistribution, State> {
    return PolicyStep(actionInformation: actionSpace.distribution, state: None())
  }
}
