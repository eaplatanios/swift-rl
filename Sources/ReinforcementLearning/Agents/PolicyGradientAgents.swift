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

// TODO: Add support for gradient clipping.
// TODO: L2 regularization support for networks.
// TODO: Reward normalizer.
// TODO: Reward norm clipping.

public protocol PolicyGradientAgent: ProbabilisticAgent {}

extension PolicyGradientAgent {
  @inlinable
  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    initialState: State,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment, State>] = []
  ) throws -> (loss: Float, state: State) {
    var trajectories = [Trajectory<Observation, State, Action, Reward>]()
    var currentStep = environment.currentStep
    var state = initialState
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let actionStatePair = self.action(for: currentStep, in: state, mode: .probabilistic)
      let nextStep = try environment.step(taking: actionStatePair.action)
      var trajectory = Trajectory(
        stepKind: nextStep.kind,
        observation: currentStep.observation,
        state: state,
        action: actionStatePair.action,
        reward: nextStep.reward)
      trajectories.append(trajectory)
      callbacks.forEach { $0(&environment, &trajectory) }
      numSteps += Int((1 - Tensor<Int32>(nextStep.kind.isLast())).sum().scalarized())
      numEpisodes += Int(Tensor<Int32>(nextStep.kind.isLast()).sum().scalarized())
      currentStep = nextStep
      state = actionStatePair.state
    }
    let loss = update(using: Trajectory<Observation, State, Action, Reward>.stack(trajectories))
    return (loss: loss, state: state)
  }
}

public struct AgentInput<Observation, State: Differentiable>: Differentiable {
  @noDerivative public let observation: Observation
  public var state: State

  @inlinable
  @differentiable
  public init(observation: Observation, state: State) {
    self.observation = observation
    self.state = state
  }
}

public struct ActorOutput<
  ActionDistribution: DifferentiableDistribution,
  State: Differentiable
>: Differentiable {
  public var actionDistribution: ActionDistribution
  public var state: State

  @inlinable
  @differentiable
  public init(actionDistribution: ActionDistribution,  state: State) {
    self.actionDistribution = actionDistribution
    self.state = state
  }
}

public struct ReinforceAgent<
  Environment: ReinforcementLearning.Environment,
  State: Differentiable,
  Network: Module,
  Optimizer: TensorFlow.Optimizer
>: PolicyGradientAgent
where
  Environment.ActionSpace.ValueDistribution: DifferentiableDistribution,
  Environment.Reward == Tensor<Float>,
  Network.Input == AgentInput<Environment.Observation, State>,
  Network.Output == ActorOutput<Environment.ActionSpace.ValueDistribution, State>,
  Optimizer.Model == Network
{
  public typealias Observation = Environment.Observation
  public typealias Action = ActionDistribution.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Reward = Tensor<Float>

  public let actionSpace: Environment.ActionSpace
  public var network: Network
  public var optimizer: Optimizer

  public let discountFactor: Float
  public let entropyRegularizationWeight: Float

  @usableFromInline internal var returnsNormalizer: TensorNormalizer<Float>?

  @inlinable
  public init(
    for environment: Environment,
    network: Network,
    optimizer: Optimizer,
    discountFactor: Float,
    normalizeReturns: Bool = true,
    entropyRegularizationWeight: Float = 0.0
  ) {
    self.actionSpace = environment.actionSpace
    self.network = network
    self.optimizer = optimizer
    self.discountFactor = discountFactor
    self.returnsNormalizer = normalizeReturns ?
      TensorNormalizer(streaming: true, alongAxes: 0, 1) :
      nil
    self.entropyRegularizationWeight = entropyRegularizationWeight
  }

  @inlinable
  public func actionDistribution(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionDistributionStatePair<ActionDistribution, State> {
    let networkOutput = network(AgentInput(observation: step.observation, state: state))
    return ActionDistributionStatePair(
      actionDistribution: networkOutput.actionDistribution,
      state: networkOutput.state)
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Observation, State, Action, Reward>
  ) -> Float {
    var returns = discountedReturns(
      discountFactor: discountFactor,
      stepKinds: trajectory.stepKind,
      rewards: trajectory.reward)
    let (loss, gradient) = network.valueWithGradient { network -> Tensor<Float> in
      let networkOutput = network(AgentInput(
        observation: trajectory.observation,
        state: trajectory.state))
      let actionDistribution = networkOutput.actionDistribution
      self.returnsNormalizer?.update(using: returns)
      if let normalizer = self.returnsNormalizer {
        returns = normalizer.normalize(returns)
      }
      let actionLogProbs = actionDistribution.logProbability(of: trajectory.action)

      // The policy gradient loss is defined as the sum, over time steps, of action
      // log-probabilities multiplied with the cumulative return from that time step onward.
      let actionLogProbWeightedReturns = actionLogProbs * returns

      // REINFORCE requires completed episodes and thus we mask out incomplete ones.
      let mask = Tensor<Float>(trajectory.stepKind.completeEpisodeMask())
      let episodeCount = trajectory.stepKind.episodeCount()

      precondition(
        episodeCount.scalarized() > 0,
        "REINFORCE requires at least one completed episode.")

      // We compute the mean of the policy gradient loss over the number of episodes.
      let policyGradientLoss = -(actionLogProbWeightedReturns * mask).sum() / episodeCount

      // If entropy regularization is being used for the action distribution, then we also
      // compute the entropy loss term.
      var entropyLoss = Tensor<Float>(0.0)
      if self.entropyRegularizationWeight > 0.0 {
        let entropy = actionDistribution.entropy()
        entropyLoss = entropyLoss - self.entropyRegularizationWeight * entropy.mean()
      }
      return policyGradientLoss + entropyLoss
    }
    optimizer.update(&network, along: gradient)
    return loss.scalarized()
  }
}

public struct ActorCriticOutput<
  ActionDistribution: DifferentiableDistribution,
  State: Differentiable
>: Differentiable {
  public var actionDistribution: ActionDistribution
  public var value: Tensor<Float>
  public var state: State

  @inlinable
  @differentiable
  public init(actionDistribution: ActionDistribution, value: Tensor<Float>, state: State) {
    self.actionDistribution = actionDistribution
    self.value = value
    self.state = state
  }
}

public struct A2CAgent<
  Environment: ReinforcementLearning.Environment,
  State: Differentiable,
  Network: Module,
  Optimizer: TensorFlow.Optimizer
>: PolicyGradientAgent
where
  Environment.Reward == Tensor<Float>,
  Network.Input == AgentInput<Environment.Observation, State>,
  Network.Output == ActorCriticOutput<Environment.ActionSpace.ValueDistribution, State>,
  Optimizer.Model == Network
{
  public typealias Observation = Environment.Observation
  public typealias Action = ActionDistribution.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Reward = Tensor<Float>

  public let actionSpace: Environment.ActionSpace
  public var network: Network
  public var optimizer: Optimizer

  public let advantageFunction: AdvantageFunction
  public let valueEstimationLossWeight: Float
  public let entropyRegularizationWeight: Float

  @usableFromInline internal var advantagesNormalizer: TensorNormalizer<Float>?

  @inlinable
  public init(
    for environment: Environment,
    network: Network,
    optimizer: Optimizer,
    advantageFunction: AdvantageFunction = GeneralizedAdvantageEstimation(discountFactor: 0.9),
    normalizeAdvantages: Bool = true,
    valueEstimationLossWeight: Float = 0.2,
    entropyRegularizationWeight: Float = 0.0
  ) {
    self.actionSpace = environment.actionSpace
    self.network = network
    self.optimizer = optimizer
    self.advantageFunction = advantageFunction
    self.advantagesNormalizer = normalizeAdvantages ?
      TensorNormalizer(streaming: true, alongAxes: 0, 1) :
      nil
    self.valueEstimationLossWeight = valueEstimationLossWeight
    self.entropyRegularizationWeight = entropyRegularizationWeight
  }

  @inlinable
  public func actionDistribution(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionDistributionStatePair<ActionDistribution, State> {
    let networkOutput = network(AgentInput(observation: step.observation, state: state))
    return ActionDistributionStatePair(
      actionDistribution: networkOutput.actionDistribution,
      state: networkOutput.state)
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Observation, State, Action, Reward>
  ) -> Float {
    let (loss, gradient) = network.valueWithGradient { network -> Tensor<Float> in
      let networkOutput = network(AgentInput(
        observation: trajectory.observation,
        state: trajectory.state))

      // Split the trajectory such that the last step is only used to provide the final value
      // estimate used for advantage estimation.
      let sequenceLength = networkOutput.value.shape[0] - 1
      let stepKinds = StepKind(trajectory.stepKind.rawValue[0..<sequenceLength])
      let values = networkOutput.value[0..<sequenceLength]
      let finalValue = networkOutput.value[sequenceLength]

      // Estimate the advantages for the provided trajectory.
      let advantageEstimate = self.advantageFunction(
        stepKinds: stepKinds,
        rewards: trajectory.reward[0..<sequenceLength],
        values: withoutDerivative(at: values),
        finalValue: withoutDerivative(at: finalValue))
      var advantages = advantageEstimate.advantages
      self.advantagesNormalizer?.update(using: advantages)
      if let normalizer = self.advantagesNormalizer {
        advantages = normalizer.normalize(advantages)
      }
      let returns = advantageEstimate.discountedReturns()

      // Compute the action log probabilities.
      let actionDistribution = networkOutput.actionDistribution
      let actionLogProbs = actionDistribution.logProbability(
        of: trajectory.action
      )[0..<sequenceLength]

      // The policy gradient loss is defined as the sum, over time steps, of action
      // log-probabilities multiplied with the normalized advantages.
      let actionLogProbWeightedReturns = actionLogProbs * advantages
      let policyGradientLoss = -actionLogProbWeightedReturns.mean()

      // The value estimation loss is defined as the mean squared error between the value
      // estimates and the discounted returns.
      let valueMSE = (values - returns).squared().mean()
      let valueEstimationLoss = self.valueEstimationLossWeight * valueMSE

      // If entropy regularization is being used for the action distribution, then we also
      // compute the entropy loss term.
      var entropyLoss = Tensor<Float>(0.0)
      if self.entropyRegularizationWeight > 0.0 {
        let entropy = actionDistribution.entropy()[0..<sequenceLength]
        entropyLoss = entropyLoss - self.entropyRegularizationWeight * entropy.mean()
      }
      return policyGradientLoss + valueEstimationLoss + entropyLoss
    }
    optimizer.update(&network, along: gradient)
    return loss.scalarized()
  }
}

// TODO: !! Allow `epsilon` to change while training.
public struct PPOClip {
  public let epsilon: Float

  @inlinable
  public init(epsilon: Float = 0.1) {
    self.epsilon = epsilon
  }
}

public struct PPOPenalty {
  public let klCutoffFactor: Float
  public let klCutoffCoefficient: Float
  public let adaptiveKLTarget: Float
  public let adaptiveKLToleranceFactor: Float
  public let adaptiveKLBetaScalingFactor: Float
  public var adaptiveKLBeta: Float?

  @inlinable
  public init(
    klCutoffFactor: Float = 0.2,
    klCutoffCoefficient: Float = 1000.0,
    adaptiveKLTarget: Float = 0.01,
    adaptiveKLToleranceFactor: Float = 1.5,
    adaptiveKLBetaScalingFactor: Float = 2.0,
    adaptiveKLBeta: Float? = 1.0
  ) {
    precondition(adaptiveKLBetaScalingFactor > 0, "The beta scaling factor must be positive.")
    self.klCutoffFactor = klCutoffFactor
    self.klCutoffCoefficient = klCutoffCoefficient
    self.adaptiveKLTarget = adaptiveKLTarget
    self.adaptiveKLToleranceFactor = adaptiveKLToleranceFactor
    self.adaptiveKLBetaScalingFactor = adaptiveKLBetaScalingFactor
    self.adaptiveKLBeta = adaptiveKLBeta
  }
}

// TODO: !! Allow `clipThreshold` to change while training.
public struct PPOValueEstimationLoss {
  public let weight: Float
  public let clipThreshold: Float?

  @inlinable
  public init(weight: Float = 0.5, clipThreshold: Float? = 0.1) {
    self.weight = weight
    self.clipThreshold = clipThreshold
  }
}

public struct PPOEntropyRegularization {
  public let weight: Float

  @inlinable
  public init(weight: Float) {
    self.weight = weight
  }
}

public struct PPOAgent<
  Environment: ReinforcementLearning.Environment,
  State: Differentiable,
  Network: Module,
  Optimizer: TensorFlow.Optimizer,
  LearningRate: ReinforcementLearning.LearningRate
>: PolicyGradientAgent
where
  Environment.ActionSpace.ValueDistribution: DifferentiableKLDivergence,
  Environment.Reward == Tensor<Float>,
  Network.Input == AgentInput<Environment.Observation, State>,
  Network.Output == ActorCriticOutput<Environment.ActionSpace.ValueDistribution, State>,
  Optimizer.Model == Network,
  LearningRate.Scalar == Optimizer.Scalar
{
  public typealias Observation = Environment.Observation
  public typealias Action = ActionDistribution.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Reward = Tensor<Float>

  public let actionSpace: Environment.ActionSpace
  public var network: Network
  public var optimizer: Optimizer
  public var trainingStep: UInt64 = 0

  public let learningRate: LearningRate
  public let maxGradientNorm: Float?
  public let advantageFunction: AdvantageFunction
  public let useTDLambdaReturn: Bool
  public let clip: PPOClip?
  public let penalty: PPOPenalty?
  public let valueEstimationLoss: PPOValueEstimationLoss
  public let entropyRegularization: PPOEntropyRegularization?
  public let iterationCountPerUpdate: Int

  @usableFromInline internal var advantagesNormalizer: TensorNormalizer<Float>?

  @inlinable
  public init(
    for environment: Environment,
    network: Network,
    optimizer: Optimizer,
    learningRate: LearningRate,
    maxGradientNorm: Float? = 0.5,
    advantageFunction: AdvantageFunction = GeneralizedAdvantageEstimation(
      discountFactor: 0.99,
      discountWeight: 0.95),
    advantagesNormalizer: TensorNormalizer<Float>? = TensorNormalizer<Float>(
      streaming: true,
      alongAxes: 0, 1),
    useTDLambdaReturn: Bool = true,
    clip: PPOClip? = PPOClip(),
    penalty: PPOPenalty? = PPOPenalty(),
    valueEstimationLoss: PPOValueEstimationLoss = PPOValueEstimationLoss(),
    entropyRegularization: PPOEntropyRegularization? = PPOEntropyRegularization(weight: 0.01),
    iterationCountPerUpdate: Int = 4
  ) {
    self.actionSpace = environment.actionSpace
    self.network = network
    self.optimizer = optimizer
    self.learningRate = learningRate
    self.maxGradientNorm = maxGradientNorm
    self.advantageFunction = advantageFunction
    self.advantagesNormalizer = advantagesNormalizer
    self.useTDLambdaReturn = useTDLambdaReturn
    self.clip = clip
    self.penalty = penalty
    self.valueEstimationLoss = valueEstimationLoss
    self.entropyRegularization = entropyRegularization
    self.iterationCountPerUpdate = iterationCountPerUpdate
  }

  @inlinable
  public func actionDistribution(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionDistributionStatePair<ActionDistribution, State> {
    let networkOutput = network(AgentInput(observation: step.observation, state: state))
    return ActionDistributionStatePair(
      actionDistribution: networkOutput.actionDistribution,
      state: networkOutput.state)
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Observation, State, Action, Reward>
  ) -> Float {
    optimizer.learningRate = learningRate(forStep: trainingStep)
    trainingStep += 1

    // Split the trajectory such that the last step is only used to provide the final value
    // estimate used for advantage estimation.
    let networkOutput = network(AgentInput(
      observation: trajectory.observation,
      state: trajectory.state))
    let sequenceLength = networkOutput.value.shape[0] - 1
    let stepKinds = StepKind(trajectory.stepKind.rawValue[0..<sequenceLength])
    let values = networkOutput.value[0..<sequenceLength]
    let finalValue = networkOutput.value[sequenceLength]

    // Estimate the advantages for the provided trajectory.
    let rewards = trajectory.reward[0..<sequenceLength]
    let advantageEstimate = advantageFunction(
      stepKinds: stepKinds,
      rewards: rewards,
      values: values,
      finalValue: finalValue)
    var advantages = advantageEstimate.advantages
    let usingGAE = advantageFunction is GeneralizedAdvantageEstimation
    let returns = useTDLambdaReturn && usingGAE ?
      advantages + values :
      advantageEstimate.discountedReturns()
    advantagesNormalizer?.update(using: advantages)
    if let normalizer = advantagesNormalizer {
      advantages = normalizer.normalize(advantages)
    }

    // Compute the action log probabilities.
    let actionDistribution = networkOutput.actionDistribution
    let actionLogProbs = actionDistribution.logProbability(
      of: trajectory.action
    )[0..<sequenceLength]
    
    var lastEpochLoss: Float = 0.0
    for _ in 0..<iterationCountPerUpdate {
      var (loss, gradient) = network.valueWithGradient { network -> Tensor<Float> in
        // TODO: Should we be updating the state here?
        let newNetworkOutput = network(AgentInput(
          observation: trajectory.observation,
          state: trajectory.state))

        // Compute the new action log probabilities.
        let newActionDistribution = newNetworkOutput.actionDistribution
        let newActionLogProbs = newActionDistribution.logProbability(
          of: trajectory.action
        )[0..<sequenceLength]

        let importanceRatio = exp(newActionLogProbs - actionLogProbs)
        var loss = importanceRatio * advantages

        // Importance ratio clipping loss term.
        if let c = self.clip {
          let ε = Tensor<Float>(c.epsilon)
          let importanceRatioClipped = importanceRatio.clipped(min: 1 - ε, max: 1 + ε)
          loss = -min(loss, importanceRatioClipped * advantages).mean()
        } else {
          loss = -loss.mean()
        }

        // KL penalty loss term.
        if let p = self.penalty {
          let klDivergence = actionDistribution.klDivergence(to: newActionDistribution)
          let klMean = klDivergence.mean()
          let klCutoffLoss = max(klMean - p.klCutoffFactor * p.adaptiveKLTarget, 0).squared()
          loss = loss + p.klCutoffCoefficient * klCutoffLoss
          if let beta = p.adaptiveKLBeta {
            loss = loss + beta * klMean
          }
        }

        // Entropy regularization loss term.
        if let e = self.entropyRegularization {
          let entropy = newActionDistribution.entropy()[0..<sequenceLength]
          loss = loss - e.weight * entropy.mean()
        }

        // Value estimation loss term.
        let newValues = newNetworkOutput.value[0..<sequenceLength]
        var valueLoss = (newValues - returns).squared()
        if let c = self.valueEstimationLoss.clipThreshold {
          let ε = Tensor<Float>(c)
          let clippedValues = values + (newValues - values).clipped(min: -ε, max: ε)
          let clippedValueLoss = (clippedValues - returns).squared()
          valueLoss = max(valueLoss, clippedValueLoss)
        }
        return loss + self.valueEstimationLoss.weight * valueLoss.mean() / 2
      }
      if let clipNorm = maxGradientNorm {
        gradient.clipByGlobalNorm(clipNorm: clipNorm)
      }
      optimizer.update(&network, along: gradient)
      lastEpochLoss = loss.scalarized()
    }

    // After the network is updated, we may need to update the adaptive KL beta.
    if var p = penalty, let beta = p.adaptiveKLBeta {
      let klDivergence = network(AgentInput(
        observation: trajectory.observation,
        state: trajectory.state)
      ).actionDistribution.klDivergence(to: actionDistribution)
      let klMean = klDivergence.mean().scalarized()
      if klMean < p.adaptiveKLTarget / p.adaptiveKLToleranceFactor {
        p.adaptiveKLBeta = max(beta / p.adaptiveKLBetaScalingFactor, 1e-16)
      } else if klMean > p.adaptiveKLTarget * p.adaptiveKLToleranceFactor {
        p.adaptiveKLBeta = beta * p.adaptiveKLBetaScalingFactor
      }
    }

    return lastEpochLoss
  }
}
