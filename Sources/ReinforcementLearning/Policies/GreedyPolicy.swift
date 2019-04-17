import Gym

public struct GreedyPolicy<
  Action,
  Observation,
  Reward,
  State,
  WrappedPolicy: ProbabilisticPolicy
>: ProbabilisticPolicy where
  WrappedPolicy.Action == Action,
  WrappedPolicy.Observation == Observation,
  WrappedPolicy.Reward == Reward,
  WrappedPolicy.State == State
{
  public private(set) var wrappedPolicy: WrappedPolicy
  public let randomSeed: UInt64?

  public init(wrappedPolicy: WrappedPolicy) {
    self.wrappedPolicy = wrappedPolicy
    self.randomSeed = wrappedPolicy.randomSeed
  }

  public func initialState() -> State {
    return wrappedPolicy.initialState()
  }

  public mutating func actionDistribution(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<Deterministic<Action>, State> {
    let step = wrappedPolicy.actionDistribution(in: state, using: step)
    let action = step.actionInformation.mode(seed: randomSeed)
    return PolicyStep(actionInformation: Deterministic(at: action), state: step.state)
  }
}
