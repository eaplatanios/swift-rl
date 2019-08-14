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

// TODO: Fill the replay buffer with random data in the beginning of training.
// TODO: Reward scaling / reward shaping.
// TODO: Exploration schedules (i.e., how to vary ε while training).

public struct QNetworkOutput<State: Differentiable>: Differentiable {
  public var qValues: Tensor<Float>
  public var state: State

  @inlinable
  @differentiable
  public init(qValues: Tensor<Float>,  state: State) {
    self.qValues = qValues
    self.state = state
  }
}

// We let Q-networks output distributions over actions, making them able to handle both discrete
// and continuous action spaces.
public struct DQNAgent<
  Environment: ReinforcementLearning.Environment,
  State: Differentiable,
  Network: Module & Copyable,
  Optimizer: TensorFlow.Optimizer
>: ProbabilisticAgent
where
  Environment.ActionSpace.ValueDistribution == Categorical<Int32>,
  Environment.Reward == Tensor<Float>,
  Network.Input == AgentInput<Environment.Observation, State>,
  Network.Output == QNetworkOutput<State>,
  Optimizer.Model == Network
{
  public typealias Observation = Environment.Observation
  public typealias Action = Environment.ActionSpace.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Reward = Tensor<Float>

  public let actionSpace: Environment.ActionSpace
  public var qNetwork: Network
  public var targetQNetwork: Network
  public var optimizer: Optimizer

  public let trainSequenceLength: Int
  public let maxReplayedSequenceLength: Int
  public let epsilonGreedy: Float
  public let targetUpdateForgetFactor: Float
  public let targetUpdatePeriod: Int
  public let discountFactor: Float
  public let trainStepsPerIteration: Int

  @usableFromInline internal var replayBuffer: UniformReplayBuffer<
    Trajectory<Observation, State, Action, Reward>>?
  @usableFromInline internal var trainingStep: Int = 0

  @inlinable
  public init(
    for environment: Environment,
    qNetwork: Network,
    optimizer: Optimizer,
    trainSequenceLength: Int,
    maxReplayedSequenceLength: Int,
    epsilonGreedy: Float = 0.1,
    targetUpdateForgetFactor: Float = 1.0,
    targetUpdatePeriod: Int = 1,
    discountFactor: Float = 0.99,
    trainStepsPerIteration: Int = 1
  ) {
    precondition(
      trainSequenceLength > 0,
      "The provided training sequence length must be greater than 0.")
    precondition(
      trainSequenceLength < maxReplayedSequenceLength,
      "The provided training sequence length is larger than the maximum replayed sequence length.")
    precondition(
      targetUpdateForgetFactor > 0.0 && targetUpdateForgetFactor <= 1.0,
      "The target update forget factor must be in the interval (0, 1].")
    self.actionSpace = environment.actionSpace
    self.qNetwork = qNetwork
    self.targetQNetwork = qNetwork.copy()
    self.optimizer = optimizer
    self.trainSequenceLength = trainSequenceLength
    self.maxReplayedSequenceLength = maxReplayedSequenceLength
    self.epsilonGreedy = epsilonGreedy
    self.targetUpdateForgetFactor = targetUpdateForgetFactor
    self.targetUpdatePeriod = targetUpdatePeriod
    self.discountFactor = discountFactor
    self.trainStepsPerIteration = trainStepsPerIteration
    self.replayBuffer = nil
  }

  @inlinable
  public func actionDistribution(
    for step: Step<Observation, Reward>,
    in state: State
  ) -> ActionDistributionStatePair<ActionDistribution, State> {
    let qNetworkOutput = qNetwork(AgentInput(observation: step.observation, state: state))
    return ActionDistributionStatePair(
      actionDistribution: Categorical<Int32>(logits: qNetworkOutput.qValues),
      state: qNetworkOutput.state)
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using trajectory: Trajectory<Observation, State, Action, Reward>
   ) -> Float {
    let (loss, gradient) = qNetwork.valueWithGradient { qNetwork -> Tensor<Float> in
      let qNetworkOutput = qNetwork(AgentInput(
        observation: trajectory.observation,
        state: trajectory.state))
      let qValue = qNetworkOutput.qValues.batchGathering(
        atIndices: trajectory.action,
        alongAxis: 2,
        batchDimensionCount: 2)

      // Split the trajectory such that the last step is only used to compute the next Q value.
      let sequenceLength = qValue.shape[0] - 1
      let currentStepKind = StepKind(trajectory.stepKind.rawValue[0..<sequenceLength])
      let nextQValue = self.computeNextQValue(
        stepKind: trajectory.stepKind,
        observation: trajectory.observation,
        state: trajectory.state
      )[1...]
      let currentReward = trajectory.reward[0..<sequenceLength]
      let targetQValue = currentReward + self.discountFactor * nextQValue

      // Compute the temporal difference (TD) loss.
      let error = abs(qValue[0..<sequenceLength] - targetQValue)
      let delta = Tensor<Float>(1.0)
      let quadratic = min(error, delta)
      // The following expression is the same in value as `max(error - delta, 0)`, but
      // importantly the gradient for the expression when `error == delta` is `0` (for the form
      // using `max(_:_:)` it would be `1`). This is necessary to avoid doubling the gradient
      // because there is already a nonzero contribution to it from the quadratic term.
      var tdLoss = 0.5 * quadratic * quadratic + delta * (error - quadratic)
      
      // Mask the loss for all steps that mark the end of an episode.
      tdLoss = tdLoss * (1 - Tensor<Float>(currentStepKind.isLast()))

      // Finally, sum the loss over the time dimension and average across the batch dimension.
      // Note that we use an element-wise loss up to this point in order to ensure that each
      // element is always weighted by `1/B` where `B` is the batch size, even when some of the
      // steps have zero loss due to episode transitions. Weighting by `1/K` where `K` is the
      // actual number of non-zero loss weights (e.g., due to the mask) would artificially increase
      // the contribution of the non-masked loss elements. This would get increasingly worse as the
      // number of episode transitions increases.
      return tdLoss.sum(squeezingAxes: 0).mean()
    }
    optimizer.update(&qNetwork, along: gradient)
    updateTargetQNetwork()
    return loss.scalarized()
  }

  @inlinable
  @discardableResult
  public mutating func update(
    using environment: inout Environment,
    initialState: State,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    callbacks: [StepCallback<Environment, State>] = []
  ) throws -> (loss: Float, state: State) {
    if replayBuffer == nil {
      replayBuffer = UniformReplayBuffer(
        batchSize: environment.batchSize,
        maxLength: maxReplayedSequenceLength)
    }
    var currentStep = environment.currentStep
    var state = initialState
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      let actionStatePair = action(
        for: currentStep,
        in: state,
        mode: .epsilonGreedy(epsilonGreedy))
      let nextStep = try environment.step(taking: actionStatePair.action)
      var trajectory = Trajectory(
        stepKind: nextStep.kind,
        observation: currentStep.observation,
        state: state,
        action: actionStatePair.action,
        reward: nextStep.reward)
      replayBuffer!.record(trajectory)
      callbacks.forEach { $0(&environment, &trajectory) }
      numSteps += Int((1 - Tensor<Int32>(nextStep.kind.isLast())).sum().scalarized())
      numEpisodes += Int(Tensor<Int32>(nextStep.kind.isLast()).sum().scalarized())
      currentStep = nextStep
      state = actionStatePair.state
    }
    var loss: Float = 0.0
    for _ in 0..<trainStepsPerIteration {
      let batch = replayBuffer!.sampleBatch(
        batchSize: environment.batchSize,
        stepCount: trainSequenceLength + 1)
      loss = update(using: batch.batch)
    }
    return (loss: loss, state: state)
  }

  /// Updates the target Q-network using the current Q-network. The update is only performed every
  /// `targetUpdatePeriod` training steps, irrespective of how often this function is being called.
  ///
  /// For each parameter, `pTarget`, in the target Q-network the following update is performed:
  /// `pTarget = targetUpdateForgetFactor * pTarget + (1 - targetUpdateForgetFactor) * p`,
  /// where `p` is the corresponding parameter in the current Q-network.
  @inlinable
  internal mutating func updateTargetQNetwork() {
    if trainingStep % targetUpdatePeriod == 0 && targetUpdateForgetFactor < 1.0 {
      targetQNetwork.update(using: qNetwork, forgetFactor: targetUpdateForgetFactor)
    }
  }

  @inlinable
  internal func computeNextQValue(
    stepKind: StepKind,
    observation: Observation,
    state: State
  ) -> Tensor<Float> {
    targetQNetwork(AgentInput(
      observation: observation,
      state: state
    )).qValues.max(squeezingAxes: -1)
  }
}
