// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import TensorFlow

/// Computes discounted returns.
///
/// Discounted returns are defined as follows:
/// `Q_t = \sum_{t'=t}^T gamma^{t'-t} * r_{t'} + gamma^{T-t+1} * finalValue`,
/// where `r_t` represents the reward at time step `t` and `gamma` represents the discount factor.
/// For more details refer to "Reinforcement Learning: An Introduction" Second Edition by
/// Richard S. Sutton and Andrew G. Barto.
///
/// The discounted return computation also takes into account the time steps when episodes end
/// (i.e., steps whose kind is `.last`) by making sure to reset the discounted return being carried
/// backwards through time.
///
/// Typically, each reward tensor will have shape `[BatchSize]` (for batched rewards) or `[]` (for
/// unbatched rewards).
///
/// - Parameters:
///   - discountFactor: Reward discount factor (`gamma` in the above example).
///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
///   - rewards: Contains the rewards for each step.
///   - finalValue: Estimated value at the final step. This is used to bootstrap the reward-to-go
///     computation. Defaults to zeros.
///
/// - Returns: Array of discounted return values over time.
@inlinable
public func discountedReturns<Scalar: TensorFlowNumeric>(
  discountFactor: Scalar,
  stepKinds: StepKind,
  rewards: Tensor<Scalar>,
  finalValue: Tensor<Scalar>? = nil
) -> Tensor<Scalar> {
  let isLast = stepKinds.isLast()
  let T = stepKinds.rawValue.shape[0]
  let finalReward = finalValue ?? Tensor<Scalar>(zerosLike: rewards[0])
  var discountedReturns = [Tensor<Scalar>]()
  for t in 0..<T {
    let futureReturn = T - t < T ? discountedReturns[t - 1] : finalReward
    let discountedFutureReturn = discountFactor * futureReturn
    let discountedReturn = rewards[T - t - 1] + discountedFutureReturn.replacing(
      with: Tensor<Scalar>(zerosLike: discountedFutureReturn),
      where: isLast[T - t - 1])
    discountedReturns.append(discountedReturn)
  }
  return Tensor(stacking: discountedReturns.reversed())
}

/// Advantage estimation result, which contains two tensors:
///   - `advantages`: Estimated advantages that are typically used to train actor networks.
///   - `discountedReturns`: Discounted returns that are typically used to train value networks.
public struct AdvantageEstimate<Scalar: TensorFlowFloatingPoint> {
  public let advantages: Tensor<Scalar>
  public let discountedReturns: () -> Tensor<Scalar>

  @inlinable
  public init(advantages: Tensor<Scalar>, discountedReturns: @escaping () -> Tensor<Scalar>) {
    self.advantages = advantages
    self.discountedReturns = discountedReturns
  }
}

public protocol AdvantageFunction {
  /// - Parameters:
  ///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
  ///   - rewards: Contains the rewards obtained at each step.
  ///   - values: Contains the value estimates for each step.
  ///   - finalValue: Estimated value at the final step.
  func callAsFunction<Scalar: TensorFlowFloatingPoint>(
    stepKinds: StepKind,
    rewards: Tensor<Scalar>,
    values: Tensor<Scalar>,
    finalValue: Tensor<Scalar>
  ) -> AdvantageEstimate<Scalar>
}

/// Performs empirical advantage estimation.
///
/// The empirical advantage estimate at step `t` is defined as:
/// `advantage[t] = returns[t] - value[t]`, where the returns are computed using
/// `discountedReturns(discountFactor:stepKinds:rewards:finalValue:)`.
public struct EmpiricalAdvantageEstimation: AdvantageFunction {
  public let discountFactor: Float

  /// - Parameters:
  ///   - discountFactor: Reward discount factor value, which must be between `0.0` and `1.0`.
  @inlinable
  public init(discountFactor: Float) {
    self.discountFactor = discountFactor
  }

  /// - Parameters:
  ///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
  ///   - rewards: Contains the rewards obtained at each step.
  ///   - values: Contains the value estimates for each step.
  ///   - finalValue: Estimated value at the final step.
  @inlinable
  public func callAsFunction<Scalar: TensorFlowFloatingPoint>(
    stepKinds: StepKind,
    rewards: Tensor<Scalar>,
    values: Tensor<Scalar>,
    finalValue: Tensor<Scalar>
  ) -> AdvantageEstimate<Scalar> {
    let returns = discountedReturns(
      discountFactor: Scalar(discountFactor),
      stepKinds: stepKinds,
      rewards: rewards,
      finalValue: finalValue)
    return AdvantageEstimate(advantages: returns - values, discountedReturns: { () in returns })
  }
}

/// Performs generalized advantage estimation.
///
/// For more details refer to "High-Dimensional Continuous Control Using Generalized Advantage
/// Estimation" by John Schulman, Philipp Moritz et al. The full paper can be found at:
/// https://arxiv.org/abs/1506.02438.
public struct GeneralizedAdvantageEstimation: AdvantageFunction {
  public let discountFactor: Float
  public let discountWeight: Float

  /// - Parameters:
  ///   - discountFactor: Reward discount factor value, which must be between `0.0` and `1.0`.
  ///   - discountWeight: A weight between `0.0` and `1.0` that is used for variance reduction in 
  ///     the temporal differences.
  @inlinable
  public init(discountFactor: Float, discountWeight: Float = 1) {
    self.discountFactor = discountFactor
    self.discountWeight = discountWeight
  }

  /// - Parameters:
  ///   - stepKinds: Contains the step kinds (represented using their integer values) for each step.
  ///   - rewards: Contains the rewards obtained at each step.
  ///   - values: Contains the value estimates for each step.
  ///   - finalValue: Estimated value at the final step.
  @inlinable
  public func callAsFunction<Scalar: TensorFlowFloatingPoint>(
    stepKinds: StepKind,
    rewards: Tensor<Scalar>,
    values: Tensor<Scalar>,
    finalValue: Tensor<Scalar>
  ) -> AdvantageEstimate<Scalar> {
    let discountWeight = Scalar(self.discountWeight)
    let discountFactor = Scalar(self.discountFactor)
    let isNotLast = 1 - Tensor<Scalar>(stepKinds.isLast())
    let T = stepKinds.rawValue.shape[0]

    // Compute advantages in reverse order.
    let last = rewards[T - 1] + discountFactor * finalValue * isNotLast[T - 1] - values[T - 1]
    var advantages = [last]
    for t in 1..<T {
      let nextValue = values[T - t] * isNotLast[T - t - 1]
      let delta = rewards[T - t - 1] + discountFactor * nextValue - values[T - t - 1]
      let nextAdvantage = advantages[t - 1] * isNotLast[T - t - 1]
      advantages.append(delta + discountWeight * discountFactor * nextAdvantage)
    }

    return AdvantageEstimate(
      advantages: Tensor(advantages.reversed()),
      discountedReturns: { () in
        discountedReturns(
          discountFactor: Scalar(discountFactor),
          stepKinds: stepKinds,
          rewards: rewards,
          finalValue: finalValue)
      })
  }
}
