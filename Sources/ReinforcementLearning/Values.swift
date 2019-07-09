import Gym
import TensorFlow

/// Computes discounted rewards.
///
/// Discounted rewards are defined as follows:
/// `Q_t = \sum_{t'=t}^T gamma^{t'-t} * r_{t'} + gamma^{T-t+1} * finalValue`,
/// where `r_t` represents the reward at time step `t` and `gamma` represents the discount factor.
/// For more details refer to "Reinforcement Learning: An Introduction" Second Edition by
/// Richard S. Sutton and Andrew G. Barto.
///
/// The discounted reward computation also takes into account the time steps when episodes end
/// (i.e., steps whose kind is `.last`) by making sure to reset the discounted reward being carried
/// backwards through time.
///
/// Typically, each reward tensor will have shape `[BatchSize]` (for batched rewards) or `[]` (for
/// "unbatched" rewards).
///
/// - Parameters:
///   - discountFactor: Reward discount factor (`gamma` in the above example).
///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
///   - rewards: Contains the rewards for each step.
///   - finalValue: Estimated value at the final step. This is used to bootstrap the reward-to-go
///     computation. Defaults to zeros.
///
/// - Returns: Array of discounted reward values over time.
@inlinable
public func discount<Scalar: TensorFlowNumeric>(
  discountFactor: Scalar,
  stepKinds: [Tensor<Int32>],
  rewards: [Tensor<Scalar>],
  finalValue: Tensor<Scalar>? = nil
) -> [Tensor<Scalar>] {
  precondition(stepKinds.count == rewards.count)

  if stepKinds.isEmpty {
    return [Tensor<Scalar>]()
  }

  let T = stepKinds.count
  let finalReward = finalValue ?? Tensor<Scalar>(zerosLike: rewards.last!)
  var discountedRewards = [Tensor<Scalar>]()
  discountedRewards.reserveCapacity(T)
  for t in (0..<T).reversed() {
    let futureReward = t + 1 < T ? discountedRewards[T - t - 2] : finalReward
    let discountedFutureReward = discountFactor * futureReward
    let discountedReward = rewards[t] + discountedFutureReward.replacing(
      with: Tensor<Scalar>(zerosLike: discountedFutureReward),
      where: stepKinds[t] .== StepKind.last.rawValue.scalar!)
    discountedRewards.append(discountedReward)
  }
  return discountedRewards.reversed()
}

public protocol AdvantageFunction {
  associatedtype Scalar: TensorFlowNumeric

  /// - Parameters:
  ///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
  ///   - rewards: Contains the rewards for each step.
  ///   - values: Contains the value estimates for each step.
  ///   - finalValue: Estimated value at the final step.
  func compute(
    stepKinds: [Tensor<Int32>],
    rewards: [Tensor<Scalar>],
    values: [Tensor<Scalar>],
    finalValue: Tensor<Scalar>
  ) -> [Tensor<Scalar>]
}

/// Performs empirical advantage estimation.
///
/// The empirical advantage estimate at step `t` is defined as:
/// `advantage[t] = reward[t] - value[t]`.
public struct EmpiricalAdvantageEstimation<Scalar: TensorFlowNumeric>: AdvantageFunction {
  /// - Parameters:
  ///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
  ///   - rewards: Contains the rewards for each step. These are typically already discounted
  ///     (i.e., the result of calling `discount(discountFactor:stepKinds:rewards:finalValue:`).
  ///   - values: Contains the value estimates for each step.
  ///   - finalValue: Estimated value at the final step.
  @inlinable
  public func compute(
    stepKinds: [Tensor<Int32>],
    rewards: [Tensor<Scalar>],
    values: [Tensor<Scalar>],
    finalValue: Tensor<Scalar>
  ) -> [Tensor<Scalar>] {
    precondition(stepKinds.count == rewards.count && stepKinds.count == values.count)
    return zip(rewards, values).map{ $0 - $1 }
  }
}

/// Performs generalized advantage estimation.
///
/// For more details refer to "High-Dimensional Continuous Control Using Generalized Advantage
/// Estimation" by John Schulman, Philipp Moritz et al. The full paper can be found at:
/// https://arxiv.org/abs/1506.02438.
public struct GeneralizedAdvantageEstimation<
  Scalar: TensorFlowNumeric & ExpressibleByFloatLiteral
>: AdvantageFunction {
  public let discountFactor: Scalar
  public let discountWeight: Scalar

  /// - Parameters:
  ///   - discountFactor: Reward discount factor value, which must be between `0.0` and `1.0`.
  ///   - discountWeight: A weight between `0.0` and `1.0` that is used for variance reduction in 
  ///     the temporal differences.
  public init(discountFactor: Scalar, discountWeight: Scalar = 1.0) {
    self.discountFactor = discountFactor
    self.discountWeight = discountWeight
  }

  /// - Parameters:
  ///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
  ///   - rewards: Contains the rewards for each step. These are typically already discounted
  ///     (i.e., the result of calling `discount(discountFactor:stepKinds:rewards:finalValue:)`).
  ///   - values: Contains the value estimates for each step.
  ///   - finalValue: Estimated value at the final step.
  @inlinable
  public func compute(
    stepKinds: [Tensor<Int32>],
    rewards: [Tensor<Scalar>],
    values: [Tensor<Scalar>],
    finalValue: Tensor<Scalar>
  ) -> [Tensor<Scalar>] {
    precondition(stepKinds.count == rewards.count && stepKinds.count == values.count)

    if stepKinds.isEmpty {
      return [Tensor<Scalar>]()
    }

    let T = stepKinds.count
    var advantages = [Tensor<Scalar>]()
    advantages.reserveCapacity(T)
    for t in (0..<T).reversed() {
      let futureAdvantage = t + 1 < T ?
        values[t + 1] + discountWeight * advantages[T - t - 2] :
        finalValue
      let discountedFutureAdvantage = discountFactor * futureAdvantage
      let advantage = rewards[t] - values[t] + discountedFutureAdvantage.replacing(
        with: Tensor<Scalar>(zerosLike: discountedFutureAdvantage),
        where: stepKinds[t] .== StepKind.last.rawValue.scalar!)
      advantages.append(advantage)
    }
    return advantages.reversed()
  }
}
