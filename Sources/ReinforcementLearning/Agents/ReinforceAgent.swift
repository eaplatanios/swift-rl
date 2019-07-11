import TensorFlow

public struct ReinforceAgent<
  Scalar: TensorFlowScalar & Equatable,
  Environment: ReinforcementLearning.Environment,
  ActorNetwork: Network,
  Optimizer: TensorFlow.Optimizer
>: Agent
where
  Environment.Reward == Tensor<Float>,
  ActorNetwork.Input: Stackable,
  ActorNetwork.State: Stackable,
  ActorNetwork.Input.Stacked == ActorNetwork.Input,
  ActorNetwork.State.Stacked == ActorNetwork.State,
  ActorNetwork.Output: DifferentiableDistribution,
  ActorNetwork.Output.Value == Tensor<Scalar>,
  Optimizer.Model == ActorPolicy<Environment, ActorNetwork>
{
  public typealias Action = ActorNetwork.Output.Value
  public typealias Observation = ActorNetwork.Input
  public typealias Reward = Tensor<Float>
  public typealias State = ActorNetwork.State

  public private(set) var policy: ActorPolicy<Environment, ActorNetwork>
  public private(set) var optimizer: Optimizer

  public let discountFactor: Float
  public let rewardsNormalizer: Normalizer<Tensor<Float>>
  public let entropyRegularizationWeight: Float

  public init(
    actorPolicy: ActorPolicy<Environment, ActorNetwork>,
    optimizer: Optimizer,
    discountFactor: Float,
    rewardsNormalizer: @escaping Normalizer<Tensor<Float>> = { $0 },
    entropyRegularizationWeight: Float = 0.0
  ) {
    self.policy = actorPolicy
    self.optimizer = optimizer
    self.discountFactor = discountFactor
    self.rewardsNormalizer = rewardsNormalizer
    self.entropyRegularizationWeight = entropyRegularizationWeight
  }

  public func initialize() { }

  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Action, Observation, Reward, State>
  ) -> Float {
    let rewards = discount(
      discountFactor: discountFactor,
      stepKinds: trajectory.currentStep.kind.rawValue.unstacked(alongAxis: 1),
      rewards: trajectory.nextStep.reward.unstacked(alongAxis: 1))
    let normalizedRewards = rewardsNormalizer(Tensor<Float>(stacking: rewards)).transposed()
    policy.state = trajectory.policyState
    let (loss, gradient) = policy.valueWithGradient {
      [entropyRegularizationWeight] policy -> Tensor<Float> in
        ReinforceAgent<Scalar, Environment, ActorNetwork, Optimizer>.lossFn(
          policy: policy,
          step: trajectory.currentStep.copy(reward: normalizedRewards),
          action: trajectory.action,
          entropyRegularizationWeight: entropyRegularizationWeight)
    }
    optimizer.update(&policy, along: gradient)
    return loss.scalar!
  }

  @differentiable(wrt: policy)
  internal static func lossFn(
    policy: ActorPolicy<Environment, ActorNetwork>,
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
