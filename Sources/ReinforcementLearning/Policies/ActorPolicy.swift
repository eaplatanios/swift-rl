import TensorFlow

public struct ActorPolicy<
  Environment: ReinforcementLearning.Environment,
  ActorNetwork: Network
>: ProbabilisticPolicy, Differentiable, KeyPathIterable
where
  ActorNetwork.Input == Environment.Observation,
  ActorNetwork.Output: DifferentiableDistribution
{
  public typealias ActionDistribution = ActorNetwork.Output
  public typealias Action = ActorNetwork.Output.Value
  public typealias Observation = Environment.Observation
  public typealias Reward = Environment.Reward
  public typealias State = ActorNetwork.State

  public private(set) var actorNetwork: ActorNetwork

  @noDerivative public let environment: Environment
  @noDerivative public let batched: Bool = true
  @noDerivative public let observationsNormalizer: Normalizer<Observation>
  @noDerivative public let randomSeed: TensorFlowSeed

  @noDerivative public var state: State {
    get { actorNetwork.state }
    set { actorNetwork.state = newValue }
  }

  public init(
    for environment: Environment,
    actorNetwork: ActorNetwork,
    observationsNormalizer: @escaping Normalizer<Observation> = { $0 },
    randomSeed: TensorFlowSeed = Context.local.randomSeed
  ) {
    self.environment = environment
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
    // actorNetwork(observationsNormalizer(step.observation))
    actorNetwork(step.observation)
  }

  public func copy() -> ActorPolicy {
    ActorPolicy(
      for: environment,
      actorNetwork: actorNetwork.copy(),
      observationsNormalizer: observationsNormalizer,
      randomSeed: randomSeed)
  }
}
