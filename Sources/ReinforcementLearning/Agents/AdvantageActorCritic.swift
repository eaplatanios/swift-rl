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

public struct ActorCriticOutput<ActionDistribution: DifferentiableDistribution>: Differentiable {
  public var actionDistribution: ActionDistribution
  public var value: Tensor<Float>

  @differentiable
  public init(actionDistribution: ActionDistribution, value: Tensor<Float>) {
    self.actionDistribution = actionDistribution
    self.value = value
  }
}

public struct A2CAgent<
  Environment: ReinforcementLearning.Environment,
  Network: ReinforcementLearning.Network,
  Optimizer: TensorFlow.Optimizer
>: ProbabilisticAgent
where
  Environment.Observation == Network.Input,
  Environment.Reward == Tensor<Float>,
  Network.Output == ActorCriticOutput<Environment.ActionSpace.ValueDistribution>,
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

  public let advantageFunction: AdvantageFunction
  public let advantagesNormalizer: (Tensor<Float>) -> Tensor<Float>
  public let valueEstimationLossWeight: Float
  public let entropyRegularizationWeight: Float

  public init(
    for environment: Environment,
    network: Network,
    optimizer: Optimizer,
    advantageFunction: AdvantageFunction = GeneralizedAdvantageEstimation(discountFactor: 0.9),
    advantagesNormalizer: @escaping (Tensor<Float>) -> Tensor<Float> = {
      standardNormalize($0, alongAxes: 0, 1) },
    valueEstimationLossWeight: Float = 0.2,
    entropyRegularizationWeight: Float = 0.0
  ) {
    self.actionSpace = environment.actionSpace
    self.network = network
    self.optimizer = optimizer
    self.advantageFunction = advantageFunction
    self.advantagesNormalizer = advantagesNormalizer
    self.valueEstimationLossWeight = valueEstimationLossWeight
    self.entropyRegularizationWeight = entropyRegularizationWeight
  }

  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    network(step.observation).actionDistribution
  }

  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Observation, Action, Reward, State>
  ) -> Float {
    network.state = trajectory.state
    let (loss, gradient) = network.valueWithGradient { network -> Tensor<Float> in
      let networkOutput = network(trajectory.observation)

      // Split the trajectory such that the last step is only used to provide the final value
      // estimate used for advantage estimation.
      let sequenceLength = networkOutput.value.shape[0] - 1
      let stepKinds = StepKind(trajectory.stepKind.rawValue[0..<sequenceLength])
      let values = networkOutput.value[0..<sequenceLength]
      let finalValue = networkOutput.value[sequenceLength]

      // Estimate the advantages for the provided trajectory.
      let advantageEstimate = advantageFunction(
        stepKinds: stepKinds,
        rewards: trajectory.reward[0..<sequenceLength],
        values: withoutDerivative(at: values),
        finalValue: withoutDerivative(at: finalValue))
      let advantages = advantagesNormalizer(advantageEstimate.advantages)
      let returns = advantageEstimate.discountedReturns

      // Compute the action log probabilities.
      let actionDistribution = networkOutput.actionDistribution
      let actionLogProbs = actionDistribution.logProbability(
        of: trajectory.action
      )[0..<sequenceLength]

      // TODO: Mask out `isLast` steps?

      // The policy gradient loss is defined as the sum, over time steps, of action
      // log-probabilities multiplied with the normalized advantages.
      let actionLogProbWeightedReturns = actionLogProbs * advantages
      let policyGradientLoss = -actionLogProbWeightedReturns.mean()

      // The value estimation loss is defined as the mean squared error between the value
      // estimates and the discounted returns.
      let valueMSE = (values - returns).squared().mean()
      let valueEstimationLoss = valueEstimationLossWeight * valueMSE

      // If entropy regularization is being used for the action distribution, then we also
      // compute the entropy loss term.
      var entropyLoss = Tensor<Float>(0.0)
      if entropyRegularizationWeight > 0.0 {
        let entropy = actionDistribution.entropy()[0..<sequenceLength]
        entropyLoss = entropyLoss - entropyRegularizationWeight * entropy.mean()
      }
      return policyGradientLoss + valueEstimationLoss + entropyLoss
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
