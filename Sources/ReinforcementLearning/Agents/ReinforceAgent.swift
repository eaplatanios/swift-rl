import TensorFlow

public struct ReinforceAgent<
  Scalar: TensorFlowScalar & Equatable,
  ActorNetwork: Network,
  Optimizer: TensorFlow.Optimizer
>: Agent
where
  ActorNetwork.Input: Stackable,
  ActorNetwork.State: Stackable,
  ActorNetwork.Input.Stacked == ActorNetwork.Input,
  ActorNetwork.State.Stacked == ActorNetwork.State,
  ActorNetwork.Output: DifferentiableDistribution,
  ActorNetwork.Output.Value == Tensor<Scalar>,
  Optimizer.Model == ActorPolicy<ActorNetwork, Tensor<Float>>
{
  public typealias Action = ActorNetwork.Output.Value
  public typealias Observation = ActorNetwork.Input
  public typealias Reward = Tensor<Float>
  public typealias State = ActorNetwork.State

  public private(set) var policy: ActorPolicy<ActorNetwork, Reward>
  public private(set) var optimizer: Optimizer

  public let discountFactor: Float
  public let rewardsNormalizer: Normalizer<Tensor<Float>>
  public let entropyRegularizationWeight: Float

  public init(
    actorNetwork: ActorNetwork,
    optimizer: Optimizer,
    discountFactor: Float,
    observationsNormalizer: @escaping Normalizer<Observation> = { $0 },
    rewardsNormalizer: @escaping Normalizer<Tensor<Float>> = { $0 },
    entropyRegularizationWeight: Float = 0.0,
    randomSeed: TensorFlowSeed = Context.local.randomSeed
  ) {
    self.policy = ActorPolicy(
      actorNetwork: actorNetwork,
      observationsNormalizer: observationsNormalizer,
      randomSeed: randomSeed)
    self.optimizer = optimizer
    self.discountFactor = discountFactor
    self.rewardsNormalizer = rewardsNormalizer
    self.entropyRegularizationWeight = entropyRegularizationWeight
  }

  public func initialize() { }

  public mutating func update(
    using trajectory: Trajectory<Action, Observation, Reward, State>
  ) -> Float {
    let rewards = discount(
      discountFactor: discountFactor,
      stepKinds: trajectory.map{ $0.currentStep.kind.rawValue },
      rewards: trajectory.map{ Tensor<Float>($0.nextStep.reward) })

    // Tensor<Float> with shape [T, B]
    let normalizedRewards = rewardsNormalizer(Tensor<Float>(stacking: rewards))

    // Tensor<Scalar> with shape [T, B, ...]
    let stackedActions = Tensor<Scalar>.stack(trajectory.map{ $0.action })
    let stackedStep = Step<Observation, Reward>.stack(
      trajectory.map{ $0.currentStep }).copy(reward: normalizedRewards)

    policy.state = State.stack(trajectory.map{ $0.policyState })

    let (loss, gradient) = policy.valueWithGradient {
      [entropyRegularizationWeight] policy -> Tensor<Float> in
        ReinforceAgent<Scalar, ActorNetwork, Optimizer>.lossFn(
          policy: policy,
          step: stackedStep,
          action: stackedActions,
          entropyRegularizationWeight: entropyRegularizationWeight)
    }

    optimizer.update(&policy, along: gradient)

    return loss.scalar!
  }

  @differentiable(wrt: policy)
  internal static func lossFn(
    policy: ActorPolicy<ActorNetwork, Tensor<Float>>,
    step: Step<Observation, Reward>,
    action: Tensor<Scalar>,
    entropyRegularizationWeight: Float
  ) -> Tensor<Float> {
    let distribution = policy.actionDistribution(for: step)
    let logProbability = distribution.logProbability(of: action)
    let loss = logProbability * step.reward
    let maskedLoss = loss * Tensor<Float>(step.kind.isLast())
    let policyLoss = -maskedLoss.sum()
    let entropyLoss = -distribution.entropy().mean()
    return policyLoss + entropyRegularizationWeight * entropyLoss
  }
}
