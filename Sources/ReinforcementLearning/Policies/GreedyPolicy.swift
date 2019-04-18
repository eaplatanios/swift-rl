import Gym
import TensorFlow

public struct GreedyPolicy<
  Scalar: TensorFlowScalar & Equatable,
  WrappedPolicy: ProbabilisticPolicy
>: ProbabilisticPolicy where WrappedPolicy.Action == Tensor<Scalar> {
  public typealias Action = WrappedPolicy.Action
  public typealias Observation = WrappedPolicy.Observation
  public typealias Reward = WrappedPolicy.Reward
  public typealias State = WrappedPolicy.State

  public let wrappedPolicy: WrappedPolicy
  public let randomSeed: UInt64?

  public var batched: Bool {
    return wrappedPolicy.batched
  }

  public init(wrapping wrappedPolicy: WrappedPolicy) {
    self.wrappedPolicy = wrappedPolicy
    self.randomSeed = wrappedPolicy.randomSeed
  }

  public func initialState(for observation: Observation) -> State {
    return wrappedPolicy.initialState(for: observation)
  }

  public func actionDistribution(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<Deterministic<Scalar>, State> {
    let step = wrappedPolicy.actionDistribution(in: state, using: step)
    let action = step.actionInformation.mode(seed: randomSeed)
    return PolicyStep(actionInformation: Deterministic(at: action), state: step.state)
  }
}
