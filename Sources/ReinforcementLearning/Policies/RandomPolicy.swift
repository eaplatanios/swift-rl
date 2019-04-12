import TensorFlow

/// Represents a policy that takes completely random actions.
public struct RandomPolicy<Action, Observation, Reward, Discount, ActionSpace: Space>: Policy 
  where ActionSpace.Value == Action {
  public typealias State = None

  public let randomSeed: UInt64

  @usableFromInline internal let actionSpace: ActionSpace
  @usableFromInline internal var rng: PhiloxRandomNumberGenerator

  public init<E: Environment>(
    for environment: E,
    randomSeed: UInt64? = nil
  ) where E.Action == Action,
          E.Observation == Observation,
          E.Reward == Reward,
          E.Discount == Discount,
          E.ActionSpace == ActionSpace {
    self.actionSpace = environment.actionSpace
    self.randomSeed = hashSeed(createSeed(using: randomSeed))
    self.rng = PhiloxRandomNumberGenerator(seed: self.randomSeed)
  }

  public mutating func act(
    in state: State,
    using step: EnvironmentStep<Observation, Reward, Discount>
  ) -> PolicyStep<Action, State> {
    let action = actionSpace.sample(generator: &rng)
    return PolicyStep(action: action, state: None())
  }
}
