import Gym
import TensorFlow

public typealias Normalizer<Value> = (Value) -> Value

public struct ActorPolicy<
  Action,
  ActionDistribution: Distribution,
  Observation,
  Reward,
  State,
  ActorNetwork: Network
>: ProbabilisticPolicy, Differentiable
where
  ActionDistribution.Value == Action,
  ActorNetwork.Input == Observation,
  ActorNetwork.Output == ActionDistribution,
  ActorNetwork.State == State
{
  @noDerivative public let batched: Bool = true

  public let actorNetwork: ActorNetwork
  @noDerivative public let observationNormalizer: Normalizer<Observation>
  @noDerivative public let randomSeed: UInt64?

  public init(
    actorNetwork: ActorNetwork,
    observationNormalizer: @escaping Normalizer<Observation> = { $0 },
    randomSeed: UInt64? = nil
  ) {
    self.actorNetwork = actorNetwork
    self.observationNormalizer = observationNormalizer
    self.randomSeed = randomSeed
  }

  @inlinable
  public func initialState() -> State {
    return actorNetwork.initialState()
  }

  @inlinable
  // TODO: @differentiable(wrt: self)
  public func actionDistribution(
    in state: State,
    using step: EnvironmentStep<Observation, Reward>
  ) -> PolicyStep<ActionDistribution, State> {
    let observation = observationNormalizer(step.observation)
    let result = actorNetwork.applied(to: observation, in: state)
    return PolicyStep(actionInformation: result.output, state: result.state)
  }
}
