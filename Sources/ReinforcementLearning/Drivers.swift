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

/// A driver takes steps in an environment using the provided policy.
public protocol Driver {
  associatedtype Agent: ReinforcementLearning.Agent

  typealias Action = Agent.Action
  typealias Observation = Agent.Observation
  typealias Reward = Agent.Reward
  typealias State = Agent.State

  typealias Listener = (Trajectory<Action, Observation, Reward, State>) -> Void

  /// Takes steps in the managed environment using the managed policy.
  @discardableResult
  mutating func run(
    startingIn state: State,
    using step: Step<Observation, Reward>,
    updating listeners: [Listener]
  ) -> Step<Observation, Reward>
}

public extension Driver where State == None {
  @discardableResult
  @inlinable
  mutating func run(
    using step: Step<Observation, Reward>,
    updating listeners: [Listener]
  ) -> Step<Observation, Reward> {
    run(startingIn: None(), using: step, updating: listeners)
  }
}

public struct Trajectory<Action, Observation, Reward, State>: KeyPathIterable {
  public var currentStep: Step<Observation, Reward>
  public var nextStep: Step<Observation, Reward>
  public var action: Action
  public var agentState: State

  @inlinable
  public func isFirst() -> Tensor<Bool> {
    currentStep.kind.isFirst()
  }

  @inlinable
  public func isTransition() -> Tensor<Bool> {
    currentStep.kind.isTransition().elementsLogicalAnd(nextStep.kind.isTransition())
  }

  @inlinable
  public func isLast() -> Tensor<Bool> {
    nextStep.kind.isLast()
  }

  @inlinable
  public func isBoundary() -> Tensor<Bool> {
    currentStep.kind.isLast()
  }
}

public struct StepBasedDriver<
  Environment: ReinforcementLearning.Environment,
  Agent: ReinforcementLearning.Agent
> where
  Environment.ActionSpace.Value == Agent.Action,
  Environment.ObservationSpace.Value == Agent.Observation,
  Environment.Reward == Agent.Reward,
  Agent.Action: Stackable,
  Agent.Observation: Stackable,
  Agent.Reward: Stackable,
  Agent.State: Stackable,
  Agent.Action.Stacked == Agent.Action,
  Agent.Observation.Stacked == Agent.Observation,
  Agent.Reward.Stacked == Agent.Reward,
  Agent.State.Stacked == Agent.State
{
  public let maxSteps: Int
  public let maxEpisodes: Int
  public let batchSize: Int

  public let batchedEnvironment: Bool
  public let batchedAgent: Bool

  public var environments: [Environment]
  public var agents: [Agent]

  public init(
    for environment: Environment,
    using agent: Agent,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    batchSize: Int = 1
  ) {
    precondition(maxSteps > 0 && maxEpisodes > 0, "'maxSteps' and 'maxEpisodes' must be > 0.")
    self.maxSteps = maxSteps
    self.maxEpisodes = maxEpisodes
    self.batchedEnvironment = environment.batched
    self.batchedAgent = agent.batched
    self.environments = batchedEnvironment ? 
      [environment] : 
      (0..<batchSize).map { _  in environment.copy() }
    self.agents = batchedAgent ? [agent] : [Agent](repeating: agent, count: batchSize)
    self.batchSize = batchSize
  }
}

extension StepBasedDriver: Driver {
  @discardableResult
  public mutating func run(
    startingIn state: State,
    using step: Step<Observation, Reward>,
    updating listeners: [Listener]
  ) -> Step<Observation, Reward> {
    if batchedAgent {
      agents[0].state = State.stack([State](repeating: state, count: batchSize))
    } else {
      agents.indices.forEach { agents[$0].state = state }
    }
    var currentStep = Step<Observation, Reward>.stack(
      [Step<Observation, Reward>](repeating: step, count: batchSize))
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      var action: Agent.Action
      var nextStep: Step<Observation, Reward>
      var state: Agent.State
      switch (batchedAgent, batchedEnvironment) {
      case (true, true):
        action = agents[0].action(for: currentStep)
        nextStep = environments[0].step(taking: action)
        state = agents[0].state
      case (true, false):
        action = agents[0].action(for: currentStep)
        let actions = action.unstacked()
        let nextSteps = environments.indices.map { environments[$0].step(taking: actions[$0]) }
        nextStep = Step<Observation, Reward>.stack(nextSteps)
        state = agents[0].state
      case (false, true):
        let currentSteps = currentStep.unstacked()
        let actions = agents.indices.map { agents[$0].action(for: currentSteps[$0]) }
        action = Agent.Action.stack(actions)
        nextStep = environments[0].step(taking: action)
        state = Agent.State.stack(agents.map { $0.state })
      case (false, false):
        let currentSteps = currentStep.unstacked()
        let actions = agents.indices.map { agents[$0].action(for: currentSteps[$0]) }
        action = Agent.Action.stack(actions)
        let nextSteps = environments.indices.map { environments[$0].step(taking: actions[$0]) }
        nextStep = Step<Observation, Reward>.stack(nextSteps)
        state = Agent.State.stack(agents.map { $0.state })
      }
      let trajectory = Trajectory(
        currentStep: currentStep,
        nextStep: nextStep,
        action: action,
        agentState: state)
      listeners.forEach { $0(trajectory) }
      numSteps += Int(Tensor<Int32>(trajectory.isBoundary().elementsLogicalNot()).sum().scalar!)
      numEpisodes += Int(Tensor<Int32>(trajectory.isBoundary()).sum().scalar!)
      currentStep = nextStep
    }
    return currentStep
  }
}
