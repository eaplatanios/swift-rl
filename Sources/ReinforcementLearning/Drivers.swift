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
  associatedtype ManagedEnvironment: Environment
  associatedtype ManagedPolicy: Policy
    where ManagedPolicy.Action == ManagedEnvironment.ActionSpace.Value,
          ManagedPolicy.Observation == ManagedEnvironment.ObservationSpace.Value,
          ManagedPolicy.Reward == ManagedEnvironment.Reward

  typealias Action = ManagedPolicy.Action
  typealias Observation = ManagedPolicy.Observation
  typealias Reward = ManagedPolicy.Reward
  typealias State = ManagedPolicy.State

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
  public var policyState: State

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

public struct StepBasedDriver<ManagedEnvironment: Environment, ManagedPolicy: Policy>
where
  ManagedEnvironment.ActionSpace.Value == ManagedPolicy.Action,
  ManagedEnvironment.ObservationSpace.Value == ManagedPolicy.Observation,
  ManagedEnvironment.Reward == ManagedPolicy.Reward,
  ManagedEnvironment.ActionSpace.Value: Stackable,
  ManagedEnvironment.ActionSpace.Value.Stacked == ManagedEnvironment.ActionSpace.Value,
  ManagedEnvironment.ObservationSpace.Value: Stackable,
  ManagedEnvironment.ObservationSpace.Value.Stacked == ManagedEnvironment.ObservationSpace.Value,
  ManagedEnvironment.Reward: Stackable,
  ManagedEnvironment.Reward.Stacked == ManagedEnvironment.Reward,
  ManagedPolicy.State: Stackable,
  ManagedPolicy.State.Stacked == ManagedPolicy.State
{
  public let maxSteps: Int
  public let maxEpisodes: Int
  public let batchSize: Int

  public let batchedEnvironment: Bool
  public let batchedPolicy: Bool

  public var environments: [ManagedEnvironment]
  public var policies: [ManagedPolicy]

  public init(
    for environment: ManagedEnvironment,
    using policy: ManagedPolicy,
    maxSteps: Int = Int.max,
    maxEpisodes: Int = Int.max,
    batchSize: Int = 1
  ) {
    precondition(maxSteps > 0 && maxEpisodes > 0, "'maxSteps' and 'maxEpisodes' must be > 0.")
    self.maxSteps = maxSteps
    self.maxEpisodes = maxEpisodes
    self.batchedEnvironment = environment.batched
    self.batchedPolicy = policy.batched
    self.environments = batchedEnvironment ? 
      [environment] : 
      (0..<batchSize).map { _  in environment.copy() }
    self.policies = batchedPolicy ?
      [policy] :
      (0..<batchSize).map { _ in policy.copy() }
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
    if batchedPolicy {
      policies[0].state = State.stack([State](repeating: state, count: batchSize))
    } else {
      policies.indices.forEach { policies[$0].state = state }
    }
    var currentStep = Step<Observation, Reward>.stack(
      [Step<Observation, Reward>](repeating: step, count: batchSize))
    var numSteps = 0
    var numEpisodes = 0
    while numSteps < maxSteps && numEpisodes < maxEpisodes {
      var action: ManagedPolicy.Action
      var nextStep: Step<Observation, Reward>
      var state: ManagedPolicy.State
      switch (batchedPolicy, batchedEnvironment) {
      case (true, true):
        action = policies[0].action(for: currentStep)
        nextStep = environments[0].step(taking: action)
        state = policies[0].state
      case (true, false):
        action = policies[0].action(for: currentStep)
        let actions = action.unstacked()
        let nextSteps = environments.indices.map { environments[$0].step(taking: actions[$0]) }
        nextStep = Step<Observation, Reward>.stack(nextSteps)
        state = policies[0].state
      case (false, true):
        let currentSteps = currentStep.unstacked()
        let actions = policies.indices.map { policies[$0].action(for: currentSteps[$0]) }
        action = ManagedPolicy.Action.stack(actions)
        nextStep = environments[0].step(taking: action)
        state = ManagedPolicy.State.stack(policies.map { $0.state })
      case (false, false):
        let currentSteps = currentStep.unstacked()
        let actions = policies.indices.map { policies[$0].action(for: currentSteps[$0]) }
        action = ManagedPolicy.Action.stack(actions)
        let nextSteps = environments.indices.map { environments[$0].step(taking: actions[$0]) }
        nextStep = Step<Observation, Reward>.stack(nextSteps)
        state = ManagedPolicy.State.stack(policies.map { $0.state })
      }
      let trajectory = Trajectory(
        currentStep: currentStep,
        nextStep: nextStep,
        action: action,
        policyState: state)
      listeners.forEach { $0(trajectory) }
      numSteps += Int(Tensor<Int32>(trajectory.isBoundary()).sum().scalar!)
      numEpisodes += Int(Tensor<Int32>(trajectory.isLast()).sum().scalar!)
      currentStep = nextStep
    }
    return currentStep
  }
}
