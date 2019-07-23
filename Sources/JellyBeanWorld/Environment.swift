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

import Foundation
import NELFramework
import ReinforcementLearning
import TensorFlow

@usableFromInline internal typealias Agent = NELFramework.Agent

public protocol JellyBeanWorldRewardFunction {
  func callAsFunction(previousItems: [Item: UInt32]?, currentItems: [Item: UInt32]?) -> Float
}

public struct JellyBeanWorldConfiguration<RewardFunction: JellyBeanWorldRewardFunction> {
  public let simulatorConfig: SimulatorConfig
  public let rewardFunction: RewardFunction

  @inlinable
  public init(simulatorConfig: SimulatorConfig, rewardFunction: RewardFunction) {
    self.simulatorConfig = simulatorConfig
    self.rewardFunction = rewardFunction
  }
}

public struct JellyBeanWorldObservation: Differentiable, KeyPathIterable {
  public var vision: Tensor<Float>
  public var scent: Tensor<Float>
  @noDerivative public var moved: Tensor<Bool>

  @inlinable
  public init(vision: Tensor<Float>, scent: Tensor<Float>, moved: Tensor<Bool>) {
    self.vision = vision
    self.scent = scent
    self.moved = moved
  }
}

public final class JellyBeanWorldEnvironment<
  RewardFunction: JellyBeanWorldRewardFunction
>: Environment {
  public let batchSize: Int
  public let configurations: [JellyBeanWorldConfiguration<RewardFunction>]
  public let actionSpace: Discrete
  public let observationSpace: ObservationSpace

  @usableFromInline internal var states: [State]
  @usableFromInline internal var step: Step<JellyBeanWorldObservation, Tensor<Float>>

  @inlinable public var currentStep: Step<JellyBeanWorldObservation, Tensor<Float>> { step }

  @inlinable
  public init(configurations: [JellyBeanWorldConfiguration<RewardFunction>]) {
    let batchSize = configurations.count
    self.batchSize = batchSize
    self.configurations = configurations
    self.actionSpace = Discrete(withSize: 3, batchSize: batchSize)
    self.observationSpace = ObservationSpace(batchSize: batchSize)
    self.states = configurations.map { configuration -> State in
      let simulator = Simulator(using: configuration.simulatorConfig)
      let agentDelegate = AgentDelegate()
      let agent = Agent(in: simulator, with: agentDelegate)
      return State(simulator: simulator, agent: agent, agentDelegate: agentDelegate)
    }
    let observation = JellyBeanWorldObservation.stack(states.map { state in
      JellyBeanWorldObservation(
        vision: Tensor<Float>(state.agent.vision!),
        scent: Tensor<Float>(state.agent.scent!),
        moved: Tensor<Bool>(repeating: false, shape: [batchSize]))
    })
    self.step =  Step(
      kind: StepKind.first(batchSize: batchSize),
      observation: observation,
      reward: Tensor<Float>(zeros: [batchSize]))
  }

  /// Updates the environment according to the provided action.
  @inlinable
  @discardableResult
  public func step(
    taking action: Tensor<Int32>
  ) -> Step<JellyBeanWorldObservation, Tensor<Float>> {
    let actions = action.unstacked()
    step = Step<JellyBeanWorldObservation, Tensor<Float>>.stack(states.indices.map { i in
      let previousPosition = states[i].agent.position
      let previousItems = states[i].agent.items
      states[i].agentDelegate.nextAction = Int(actions[i].scalarized())
      states[i].agent.act()
      let observation = JellyBeanWorldObservation(
        vision: Tensor<Float>(states[i].agent.vision!),
        scent: Tensor<Float>(states[i].agent.scent!),
        moved: Tensor<Bool>(states[i].agent.position != previousPosition))
      let reward = Tensor<Float>(configurations[i].rewardFunction(
        previousItems: previousItems,
        currentItems: states[i].agent.items))
      return Step(kind: StepKind.transition(), observation: observation, reward: reward)
    })
    return step
  }

  /// Resets the environment.
  @inlinable
  @discardableResult
  public func reset() -> Step<JellyBeanWorldObservation, Tensor<Float>> {
    states = configurations.map { configuration -> State in
      let simulator = Simulator(using: configuration.simulatorConfig)
      let agentDelegate = AgentDelegate()
      let agent = Agent(in: simulator, with: agentDelegate)
      return State(simulator: simulator, agent: agent, agentDelegate: agentDelegate)
    }
    let observation = JellyBeanWorldObservation.stack(states.map { state in
      JellyBeanWorldObservation(
        vision: Tensor<Float>(state.agent.vision!),
        scent: Tensor<Float>(state.agent.scent!),
        moved: Tensor<Bool>(repeating: false, shape: [batchSize]))
    })
    step =  Step(
      kind: StepKind.first(batchSize: batchSize),
      observation: observation,
      reward: Tensor<Float>(zeros: [batchSize]))
    return step
  }

  /// Returns a copy of this environment that is reset before being returned.
  @inlinable
  public func copy() -> JellyBeanWorldEnvironment<RewardFunction> {
    JellyBeanWorldEnvironment<RewardFunction>(configurations: configurations)
  }
}

extension JellyBeanWorldEnvironment {
  @usableFromInline internal struct State {
    @usableFromInline internal let simulator: Simulator
    @usableFromInline internal let agent: Agent
    @usableFromInline internal var agentDelegate: AgentDelegate

    @inlinable
    internal init(simulator: Simulator, agent: Agent, agentDelegate: AgentDelegate) {
      self.simulator = simulator
      self.agent = agent
      self.agentDelegate = agentDelegate
    }
  }

  @usableFromInline internal class AgentDelegate: NELFramework.AgentDelegate {
    @usableFromInline internal var nextAction: Int? = nil

    @inlinable
    internal init() {}

    @inlinable
    internal func act(_ agent: Agent) {
      switch nextAction {
        case 0: agent.move(towards: .up, by: 1)
        case 1: agent.turn(towards: .left)
        case 2: agent.turn(towards: .right)
        case _: ()
      }
    }

    @inlinable
    internal func save(_ agent: Agent, to file: URL) throws {}


    @inlinable
    internal func load(_ agent: Agent, from file: URL) throws {}
  }
}

extension JellyBeanWorldEnvironment {
  public struct ObservationSpace: Space {
    public let distribution: ValueDistribution

    @inlinable
    public init(batchSize: Int) {
      self.distribution = ValueDistribution()
    }

    @inlinable
    public var description: String {
      "JellyBeanWorldObservation"
    }

    @inlinable
    public func contains(_ value: JellyBeanWorldObservation) -> Bool {
      true // TODO: Check for the range of values.
    }

    public struct ValueDistribution: DifferentiableDistribution, KeyPathIterable {
      @usableFromInline internal var visionDistribution: Uniform<Float> = Uniform<Float>(
        lowerBound: Tensor<Float>(0),
        upperBound: Tensor<Float>(1))
      // TODO: Should we limit the range of the following values?
      @usableFromInline internal var scentDistribution: Uniform<Float> = Uniform<Float>(
        lowerBound: Tensor<Float>(-Float.greatestFiniteMagnitude),
        upperBound: Tensor<Float>(Float.greatestFiniteMagnitude))
      @usableFromInline internal var movedDistribution: Bernoulli<Int32> = Bernoulli<Int32>(
        probabilities: Tensor<Float>(0.5))
      
      @inlinable
      public init() {}

      @inlinable
      @differentiable(wrt: self)
      public func logProbability(of value: JellyBeanWorldObservation) -> Tensor<Float> {
        visionDistribution.logProbability(of: value.vision) +
          scentDistribution.logProbability(of: value.scent) +
          movedDistribution.logProbability(of: Tensor<Int32>(value.moved))
      }

      @inlinable
      @differentiable(wrt: self)
      public func entropy() -> Tensor<Float> {
        visionDistribution.entropy() + scentDistribution.entropy() + movedDistribution.entropy()
      }

      @inlinable
      public func mode() -> JellyBeanWorldObservation {
        JellyBeanWorldObservation(
          vision: visionDistribution.mode(),
          scent: scentDistribution.mode(),
          moved: movedDistribution.mode() .> 0)
      }

      @inlinable
      public func sample() -> JellyBeanWorldObservation {
        JellyBeanWorldObservation(
          vision: visionDistribution.sample(),
          scent: scentDistribution.sample(),
          moved: movedDistribution.sample() .> 0)
      }
    }
  }
}
