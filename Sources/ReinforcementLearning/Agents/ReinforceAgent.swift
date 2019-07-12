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

public struct ReinforceNetworkOutput<
  ActionDistribution: DifferentiableDistribution
>: Differentiable {
  public var actionDistribution: ActionDistribution
  @noDerivative public let value: Tensor<Float>? // TODO: !!! Allow differentiating this.

  @differentiable
  public init(actionDistribution: ActionDistribution, value: Tensor<Float>? = nil) {
    self.actionDistribution = actionDistribution
    self.value = value
  }
}

public struct ReinforceAgent<
  Environment: ReinforcementLearning.Environment,
  Network: ReinforcementLearning.Network,
  Optimizer: TensorFlow.Optimizer
>: ProbabilisticAgent
where
  Environment.Observation == Network.Input,
  Environment.Reward == Tensor<Float>,
  Network.Output == ReinforceNetworkOutput<Environment.ActionSpace.ValueDistribution>,
  Optimizer.Model == Network
{
  public typealias Observation = Network.Input
  public typealias Action = ActionDistribution.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Reward = Tensor<Float>
  public typealias State = Network.State

  public var network: Network
  public var optimizer: Optimizer

  public var state: State {
    get { network.state }
    set { network.state = newValue }
  }

  public let discountFactor: Float
  public let advantageFunction: AdvantageFunction
  public let returnsNormalizer: (Tensor<Float>) -> Tensor<Float>
  public let entropyRegularizationWeight: Float
  public let valueEstimationLossWeight: Float

  private var replayBuffer: UniformReplayBuffer<Trajectory<Observation, Action, Reward, State>>

  public init(
    for environment: Environment,
    network: Network,
    optimizer: Optimizer,
    maxReplayedSequenceLength: Int,
    discountFactor: Float,
    advantageFunction: AdvantageFunction = NoAdvantageFunction(),
    returnsNormalizer: @escaping (Tensor<Float>) -> Tensor<Float> = {
      standardNormalize($0, alongAxes: 0, 1) },
    entropyRegularizationWeight: Float = 0.0,
    valueEstimationLossWeight: Float = 0.0
  ) {
    self.network = network
    self.optimizer = optimizer
    self.discountFactor = discountFactor
    self.advantageFunction = advantageFunction
    self.returnsNormalizer = returnsNormalizer
    self.entropyRegularizationWeight = entropyRegularizationWeight
    self.valueEstimationLossWeight = valueEstimationLossWeight
    self.replayBuffer = UniformReplayBuffer<Trajectory<Observation, Action, Reward, State>>(
      batchSize: environment.batchSize,
      maxLength: maxReplayedSequenceLength)
  }

  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    network(step.observation).actionDistribution
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
        let networkOutput = network(trajectory.observation)
        var advantages = returns
        // if let value = networkOutput.value {
        //   let values = value.unstacked(alongAxis: 1)
        //   let values = networkOutput.value.unstacked(alongAxis: 1)
        //   advantages = advantageFunction(
        //     stepKinds: stepKinds,
        //     returns: returns,
        //     values: values,
        //     finalValue: values.last!) // TODO: I believe this is not correct!
        // }
        let normalizedReturns = returnsNormalizer(advantages)
        let distribution = networkOutput.actionDistribution
        let actionLogProbs = distribution.logProbability(of: trajectory.action)

        // Policy gradient loss is defined as the sum, over time steps, of action log-probabilities
        // multiplied with the cumulative return from that time step onward.
        let actionLogProbWeightedReturns = actionLogProbs * normalizedReturns

        // REINFORCE requires completed episodes and thus we mask out incomplete ones.
        let mask = Tensor<Float>(trajectory.stepKind.completeEpisodeMask())
        let episodeCount = trajectory.stepKind.episodeCount()

        precondition(
          episodeCount.scalarized() > 0,
          "REINFORCE requires at least one completed episode.")

        // We compute the mean of the policy gradient loss over the number of episodes.
        let policyGradientLoss = -(actionLogProbWeightedReturns * mask).sum() / episodeCount
        let entropyLoss = entropyRegularizationWeight * -(distribution.entropy() * mask).mean()
        return policyGradientLoss + entropyLoss
    }
    optimizer.update(&network, along: gradient)
    return loss.scalar!
  }

  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    stepCallbacks: [(Trajectory<Observation, Action, Reward, State>) -> Void]
  ) -> Float {
    var currentStep = environment.currentStep()
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let action = self.action(for: currentStep)
      let nextStep = environment.step(taking: action)
      let trajectory = Trajectory(
        stepKind: nextStep.kind,
        observation: currentStep.observation,
        action: action,
        reward: nextStep.reward,
        state: state)
      replayBuffer.record(trajectory)
      stepCallbacks.forEach { $0(trajectory) }
      numSteps += Int((1 - Tensor<Int32>(nextStep.kind.isLast())).sum().scalar!)
      numEpisodes += Int(Tensor<Int32>(nextStep.kind.isLast()).sum().scalar!)
      currentStep = nextStep
    }
    let batch = replayBuffer.recordedData()
    replayBuffer.reset()
    return update(using: batch)
  }
}
