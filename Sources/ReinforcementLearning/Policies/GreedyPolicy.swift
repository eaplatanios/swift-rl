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

  public private(set) var wrappedPolicy: WrappedPolicy

  public let randomSeed: TensorFlowSeed

  public var batched: Bool { wrappedPolicy.batched }

  public var state: State {
    get { wrappedPolicy.state }
    set { wrappedPolicy.state = newValue }
  }

  public init(wrapping wrappedPolicy: WrappedPolicy) {
    self.wrappedPolicy = wrappedPolicy
    self.randomSeed = wrappedPolicy.randomSeed
  }

  public func initialize(using observation: Observation) {
    wrappedPolicy.initialize(using: observation)
  }

  public func actionDistribution(for step: Step<Observation, Reward>) -> Deterministic<Scalar> {
    let distribution = wrappedPolicy.actionDistribution(for: step)
    let action = distribution.mode(usingSeed: randomSeed)
    return Deterministic(at: action)
  }
}
