import Gym

public protocol ProbabilisticPolicy: Policy {
  associatedtype ActionDistribution: Distribution where ActionDistribution.Value == Action

  var randomSeed: UInt64? { get }

  /// Generates the distribution over next actions given the current environment step.
  mutating func actionDistribution(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<ActionDistribution, State>
}

public extension ProbabilisticPolicy where State == None {
  /// Generates the distribution over next actions given the current environment step.
  mutating func actionDistribution(
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<ActionDistribution, State> {
    return actionDistribution(in: None(), using: step)
  }
}

public extension ProbabilisticPolicy {
  mutating func act(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<Action, State> {
    let step = actionDistribution(in: state, using: step)
    let sampledAction = step.actionInformation.sample(seed: randomSeed)
    return PolicyStep(actionInformation: sampledAction, state: step.state)
  }
}
