import Gym
import TensorFlow

public struct ActorPolicy<
  ActorNetwork: Network,
  Reward
>: ProbabilisticPolicy, Differentiable, KeyPathIterable
where ActorNetwork.Output: DifferentiableDistribution {
  public typealias ActionDistribution = ActorNetwork.Output
  public typealias Action = ActorNetwork.Output.Value
  public typealias Observation = ActorNetwork.Input
  public typealias State = ActorNetwork.State

  public private(set) var actorNetwork: ActorNetwork

  @noDerivative public let batched: Bool = true
  @noDerivative public let observationsNormalizer: Normalizer<Observation>
  @noDerivative public let randomSeed: TensorFlowSeed

  @noDerivative public var state: State {
    get { actorNetwork.state }
    set { actorNetwork.state = newValue }
  }

  public init(
    actorNetwork: ActorNetwork,
    observationsNormalizer: @escaping Normalizer<Observation> = { $0 },
    randomSeed: TensorFlowSeed = Context.local.randomSeed
  ) {
    self.actorNetwork = actorNetwork
    self.observationsNormalizer = observationsNormalizer
    self.randomSeed = randomSeed
  }

  @inlinable
  public func initialize(using observation: Observation) {
    actorNetwork.initialize(using: observation)
  }

  @inlinable
  @differentiable(wrt: self)
  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    let observation = observationsNormalizer(step.observation)
    return actorNetwork.applied(to: observation)
  }
}
