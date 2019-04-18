import Gym
import TensorFlow

public struct ActorPolicy<ActorNetwork: Network, Reward>: ProbabilisticPolicy, Differentiable
where ActorNetwork.Output: Distribution {
  public typealias ActionDistribution = ActorNetwork.Output
  public typealias Action = ActorNetwork.Output.Value
  public typealias Observation = ActorNetwork.Input
  public typealias State = ActorNetwork.State

  @noDerivative public let batched: Bool = true

  public let actorNetwork: ActorNetwork
  @noDerivative public let observationsNormalizer: Normalizer<Observation>
  @noDerivative public let randomSeed: UInt64?

  public init(
    actorNetwork: ActorNetwork,
    observationsNormalizer: @escaping Normalizer<Observation> = { $0 },
    randomSeed: UInt64? = nil
  ) {
    self.actorNetwork = actorNetwork
    self.observationsNormalizer = observationsNormalizer
    self.randomSeed = randomSeed
  }

  @inlinable
  public func initialState(for observation: Observation) -> State {
    return actorNetwork.initialState(for: observation)
  }

  @inlinable
  @differentiable(wrt: self)
  public func actionDistribution(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<ActionDistribution, State> {
    let observation = observationsNormalizer(step.observation)
    let result = actorNetwork.applied(to: observation, in: state)
    return PolicyStep(actionInformation: result.output, state: result.state)
  }
}
