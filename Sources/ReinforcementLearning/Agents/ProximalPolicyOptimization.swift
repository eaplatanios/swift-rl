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

// TODO: L2 regularization support for networks.
// TODO: Reward normalizer.
// TODO: Observation normalizer.
// TODO: Reward norm clipping.

public struct PPOAgent<
  Environment: ReinforcementLearning.Environment,
  Network: ReinforcementLearning.Network,
  Optimizer: TensorFlow.Optimizer
>: ProbabilisticAgent
where
  Environment.Observation == Network.Input,
  Environment.ActionSpace.ValueDistribution: DifferentiableKLDivergence,
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

  public let maxReplayedSequenceLength: Int
  public let importanceRatioClipping: Float?
  public let actionLogProbabilitiesClipping: Float?
  public let klCutoffFactor: Float?
  public let klCutoffCoefficient: Float
  public let initialAdaptiveKLBeta: Float?
  public let adaptiveKLTarget: Float
  public let adaptiveKLTolerance: Float
  public let advantageFunction: AdvantageFunction
  public let advantagesNormalizer: (Tensor<Float>) -> Tensor<Float>
  public let useTDLambdaReturn: Bool
  public let valueEstimationLossWeight: Float
  public let entropyRegularizationWeight: Float
  public let epochCount: Int

  private var replayBuffer: UniformReplayBuffer<Trajectory<Observation, Action, Reward, State>>?
  private var adaptiveKLBeta: Float?

  public init(
    for environment: Environment,
    network: Network,
    optimizer: Optimizer,
    maxReplayedSequenceLength: Int,
    importanceRatioClipping: Float? = nil,
    actionLogProbabilitiesClipping: Float? = nil,
    klCutoffFactor: Float? = 2.0,
    klCutoffCoefficient: Float = 1000.0,
    initialAdaptiveKLBeta: Float? = 1.0,
    adaptiveKLTarget: Float = 0.01,
    adaptiveKLTolerance: Float = 0.3,
    advantageFunction: AdvantageFunction = GeneralizedAdvantageEstimation(
      discountFactor: 0.99,
      discountWeight: 0.95),
    advantagesNormalizer: @escaping (Tensor<Float>) -> Tensor<Float> = {
      standardNormalize($0, alongAxes: 0, 1) },
    useTDLambdaReturn: Bool = false,
    valueEstimationLossWeight: Float = 0.2,
    entropyRegularizationWeight: Float = 0.0,
    epochCount: Int = 25
  ) {
    self.actionSpace = environment.actionSpace
    self.network = network
    self.optimizer = optimizer
    self.maxReplayedSequenceLength = maxReplayedSequenceLength
    self.importanceRatioClipping = importanceRatioClipping
    self.actionLogProbabilitiesClipping = actionLogProbabilitiesClipping
    self.klCutoffFactor = klCutoffFactor
    self.klCutoffCoefficient = klCutoffCoefficient
    self.initialAdaptiveKLBeta = initialAdaptiveKLBeta
    self.adaptiveKLTarget = adaptiveKLTarget
    self.adaptiveKLTolerance = adaptiveKLTolerance
    self.advantageFunction = advantageFunction
    self.advantagesNormalizer = advantagesNormalizer
    self.useTDLambdaReturn = useTDLambdaReturn
    self.valueEstimationLossWeight = valueEstimationLossWeight
    self.entropyRegularizationWeight = entropyRegularizationWeight
    self.epochCount = epochCount
    self.replayBuffer = nil
    self.adaptiveKLBeta = initialAdaptiveKLBeta
  }

  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    network(step.observation).actionDistribution
  }

  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Observation, Action, Reward, State>
  ) -> Float {
    network.state = trajectory.state
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
      values: values,
      finalValue: finalValue)
    let advantages = advantagesNormalizer(advantageEstimate.advantages)
    let usingGAE = advantageFunction is GeneralizedAdvantageEstimation
    let returns = useTDLambdaReturn && usingGAE ?
      advantages + withoutDerivative(at: values) :
      advantageEstimate.discountedReturns

    // Compute the action log probabilities.
    let actionDistribution = networkOutput.actionDistribution
    let actionLogProbs = actionDistribution.logProbability(
      of: trajectory.action
    )[0..<sequenceLength]
    
    var lastEpochLoss: Float = 0.0
    for _ in 0..<epochCount {
      // Restore the network state before computing the loss function.
      network.state = trajectory.state
      let (loss, gradient) = network.valueWithGradient { network -> Tensor<Float> in
        let newNetworkOutput = network(trajectory.observation)

        // Compute the new action log probabilities.
        let newActionDistribution = newNetworkOutput.actionDistribution
        var newActionLogProbs = newActionDistribution.logProbability(
          of: trajectory.action
        )[0..<sequenceLength]

        if let clipValue = actionLogProbabilitiesClipping {
          let value = Tensor<Float>(clipValue)
          newActionLogProbs = newActionLogProbs.clipping(min: -value, max: value)
        }

        // TODO: Mask out `isLast` steps?

        let importanceRatio = exp(newActionLogProbs - actionLogProbs)
        var policyGradientLoss = importanceRatio * advantages
        if let clipValue = importanceRatioClipping {
          let value = Tensor<Float>(clipValue)
          let importanceRatioClipped = importanceRatio.clipping(min: 1 - value, max: 1 + value)
          policyGradientLoss = min(policyGradientLoss, importanceRatioClipped * advantages)
        }
        policyGradientLoss = -policyGradientLoss.mean()

        // The value estimation loss is defined as the mean squared error between the value
        // estimates and the discounted returns.
        let values = newNetworkOutput.value[0..<sequenceLength]
        let valueMSE = (values - returns).squared().mean()
        let valueEstimationLoss = valueEstimationLossWeight * valueMSE

        // Compute the KL penalty loss.
        let klDivergence = actionDistribution.klDivergence(to: newActionDistribution)
        let klMean = klDivergence.mean()
        var klPenaltyLoss = Tensor<Float>(0.0)
        if let cutoffFactor = klCutoffFactor {
          let klCutoffLoss = max(klMean - cutoffFactor * adaptiveKLTarget, 0).squared()
          klPenaltyLoss = klPenaltyLoss + klCutoffCoefficient * klCutoffLoss
        }
        if let beta = adaptiveKLBeta {
          klPenaltyLoss = klPenaltyLoss + beta * klMean
        }

        // If entropy regularization is being used for the action distribution, then we also
        // compute the entropy loss term.
        var entropyLoss = Tensor<Float>(0.0)
        if entropyRegularizationWeight > 0.0 {
          let entropy = actionDistribution.entropy()[0..<sequenceLength]
          entropyLoss = entropyLoss - entropyRegularizationWeight * entropy.mean()
        }
        return policyGradientLoss + valueEstimationLoss + klPenaltyLoss + entropyLoss
      }
      optimizer.update(&network, along: gradient)
      lastEpochLoss = loss.scalarized()
    }

    // After the network is updated, we need to update the adaptive KL beta.
    if let beta = adaptiveKLBeta {
      let klDivergence = network(trajectory.observation).actionDistribution.klDivergence(
        to: actionDistribution)
      let klMean = klDivergence.mean().scalarized()
      if klMean < adaptiveKLTarget * (1 - adaptiveKLTolerance) {
        adaptiveKLBeta = max(beta / 1.5, 1e-16)
      } else if klMean > adaptiveKLTarget * (1 + adaptiveKLTolerance) {
        adaptiveKLBeta = max(beta * 1.5, 1e-16)
      }
    }
    
    // TODO: Update reward and observation normalizers.

    return lastEpochLoss
  }

  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    stepCallbacks: [(Trajectory<Observation, Action, Reward, State>) -> Void]
  ) -> Float {
    if replayBuffer == nil {
      replayBuffer = UniformReplayBuffer(
        batchSize: environment.batchSize,
        maxLength: maxReplayedSequenceLength)
    }
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
      replayBuffer!.record(trajectory)
      stepCallbacks.forEach { $0(trajectory) }
      numSteps += Int((1 - Tensor<Int32>(nextStep.kind.isLast())).sum().scalarized())
      numEpisodes += Int(Tensor<Int32>(nextStep.kind.isLast()).sum().scalarized())
      currentStep = nextStep
    }
    let batch = replayBuffer!.recordedData()
    replayBuffer!.reset()
    return update(using: batch)
  }
}
