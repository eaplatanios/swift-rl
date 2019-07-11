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
  public let rewardsNormalizer: (Tensor<Float>) -> Tensor<Float>
  public let entropyRegularizationWeight: Float

  public init(
    actorPolicy: ActorPolicy<Environment, ActorNetwork>,
    optimizer: Optimizer,
    discountFactor: Float,
    rewardsNormalizer: @escaping (Tensor<Float>) -> Tensor<Float> = { $0 },
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
    let actionLogProbs = distribution.logProbability(of: action)

    // Policy gradient loss is defined as the sum, over time steps, of action log-probabilities
    // multiplied with the cumulative return from that time step onward.
    let actionLogProbWeightedReturns = actionLogProbs * step.reward
    
    // We mask out partial episodes at the end of each batch and also transitions between the end
    // state of previous episodes and the start state of the next episode.
    let isLast = Tensor<Float>(step.kind.isLast())
    let mask = Tensor<Float>(isLast.cumulativeSum(alongAxis: 1, reverse: true) .> 0) * (1 - isLast)
    let episodeCount = isLast.sum()

    // We compute the mean of the policy gradient loss over the number of episodes.
    let policyGradientLoss = -(actionLogProbWeightedReturns * mask).sum() / episodeCount
    let entropyLoss = entropyRegularizationWeight * -(distribution.entropy() * mask).mean()
    return policyGradientLoss + entropyLoss
  }
}
