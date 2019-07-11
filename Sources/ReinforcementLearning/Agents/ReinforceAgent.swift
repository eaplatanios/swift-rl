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
