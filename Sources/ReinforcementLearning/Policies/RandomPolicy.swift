import TensorFlow

/// Represents a policy that takes completely random actions.
public struct RandomPolicy<Action, Observation, Reward, ActionSpace: Space>: Policy
  where ActionSpace.Value == Action {
  public typealias State = None

  public let randomSeed: UInt64

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
  public mutating func act(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<Action, State> {
    let action = actionSpace.sample(seed: randomSeed)
    return PolicyStep(action: action, state: None())
  }
}
