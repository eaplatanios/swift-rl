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

// TODO: Add support for reward normalization.
// TODO: Add support for gradient clipping.

public struct ReinforceAgent<
  Environment: ReinforcementLearning.Environment,
  Network: ReinforcementLearning.Network,
  Optimizer: TensorFlow.Optimizer
>: ProbabilisticAgent
where
  Environment.Observation == Network.Input,
  Environment.ActionSpace.ValueDistribution: DifferentiableDistribution,
  Environment.Reward == Tensor<Float>,
  Network.Output == Environment.ActionSpace.ValueDistribution,
  Optimizer.Model == Network
{
  public typealias Observation = Network.Input
  public typealias Action = ActionDistribution.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Reward = Tensor<Float>
  public typealias State = Network.State

  public let actionSpace: Environment.ActionSpace
  public var network: Network
  public var optimizer: Optimizer

  public var state: State {
    get { network.state }
    set { network.state = newValue }
  }

  public let discountFactor: Float
  public let returnsNormalizer: (Tensor<Float>) -> Tensor<Float>
  public let entropyRegularizationWeight: Float

  public init(
    for environment: Environment,
    network: Network,
    optimizer: Optimizer,
    discountFactor: Float,
    returnsNormalizer: @escaping (Tensor<Float>) -> Tensor<Float> = {
      standardNormalize($0, alongAxes: 0, 1) },
    entropyRegularizationWeight: Float = 0.0
  ) {
    self.actionSpace = environment.actionSpace
    self.network = network
    self.optimizer = optimizer
    self.discountFactor = discountFactor
    self.returnsNormalizer = returnsNormalizer
    self.entropyRegularizationWeight = entropyRegularizationWeight
  }

  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    network(step.observation)
  }

  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Observation, Action, Reward, State>
  ) -> Float {
    let returns = discountedReturns(
      discountFactor: discountFactor,
      stepKinds: trajectory.stepKind,
      rewards: trajectory.reward)
    network.state = trajectory.state
    let (loss, gradient) = network.valueWithGradient { network -> Tensor<Float> in
      let actionDistribution = network(trajectory.observation)
      let normalizedReturns = returnsNormalizer(returns)
      let actionLogProbs = actionDistribution.logProbability(of: trajectory.action)

      // The policy gradient loss is defined as the sum, over time steps, of action
      // log-probabilities multiplied with the cumulative return from that time step onward.
      let actionLogProbWeightedReturns = actionLogProbs * normalizedReturns

      // REINFORCE requires completed episodes and thus we mask out incomplete ones.
      let mask = Tensor<Float>(trajectory.stepKind.completeEpisodeMask())
      let episodeCount = trajectory.stepKind.episodeCount()

      precondition(
        episodeCount.scalarized() > 0,
        "REINFORCE requires at least one completed episode.")
      
      // TODO: Mask out `isLast` steps?

      // We compute the mean of the policy gradient loss over the number of episodes.
      let policyGradientLoss = -(actionLogProbWeightedReturns * mask).sum() / episodeCount

      // If entropy regularization is being used for the action distribution, then we also
      // compute the entropy loss term.
      var entropyLoss = Tensor<Float>(0.0)
      if entropyRegularizationWeight > 0.0 {
        let entropy = actionDistribution.entropy()
        entropyLoss = entropyLoss - entropyRegularizationWeight * entropy.mean()
      }
      return policyGradientLoss + entropyLoss
    }
    optimizer.update(&network, along: gradient)
    return loss.scalarized()
  }

  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    stepCallbacks: [(Trajectory<Observation, Action, Reward, State>) -> Void]
  ) -> Float {
    var trajectories = [Trajectory<Observation, Action, Reward, State>]()
    var currentStep = environment.currentStep()
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let action = self.action(for: currentStep, mode: .probabilistic)
      let nextStep = environment.step(taking: action)
      let trajectory = Trajectory(
        stepKind: nextStep.kind,
        observation: currentStep.observation,
        action: action,
        reward: nextStep.reward,
        state: state)
      trajectories.append(trajectory)
      stepCallbacks.forEach { $0(trajectory) }
      numSteps += Int((1 - Tensor<Int32>(nextStep.kind.isLast())).sum().scalarized())
      numEpisodes += Int(Tensor<Int32>(nextStep.kind.isLast()).sum().scalarized())
      currentStep = nextStep
    }
    return update(using: Trajectory<Observation, Action, Reward, State>.stack(trajectories))
  }
}
