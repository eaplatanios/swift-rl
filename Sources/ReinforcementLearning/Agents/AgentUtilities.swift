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

public struct AgentInput<Observation, State: Differentiable>: Differentiable, KeyPathIterable {
  @noDerivative public let observation: Observation
  public var state: State

  @inlinable
  @differentiable(wrt: state)
  public init(observation: Observation, state: State) {
    self.observation = observation
    self.state = state
  }
}

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

public struct ActorOutput<
  ActionDistribution: DifferentiableDistribution & KeyPathIterable,
  State: Differentiable & KeyPathIterable
>: Differentiable, KeyPathIterable {
  public var actionDistribution: ActionDistribution
  public var state: State

  @inlinable
  @differentiable
  public init(actionDistribution: ActionDistribution,  state: State) {
    self.actionDistribution = actionDistribution
    self.state = state
  }
}

public struct ActorCriticOutput<
  ActionDistribution: DifferentiableDistribution & KeyPathIterable,
  State: Differentiable & KeyPathIterable
>: Differentiable, KeyPathIterable {
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

public struct StatelessActorCriticOutput<
  ActionDistribution: DifferentiableDistribution & KeyPathIterable
>: Differentiable, KeyPathIterable {
  public var actionDistribution: ActionDistribution
  public var value: Tensor<Float>

  @inlinable
  @differentiable
  public init(actionDistribution: ActionDistribution, value: Tensor<Float>) {
    self.actionDistribution = actionDistribution
    self.value = value
  }
}

public struct StatelessQNetwork<
  Environment: ReinforcementLearning.Environment,
  Network: Module & Copyable
>: Module & Copyable where
  Network.Input == Environment.Observation,
  Network.Output == Tensor<Float>
{
  public var statelessNetwork: Network

  @inlinable
  public init(_ statelessNetwork: Network) {
    self.statelessNetwork = statelessNetwork
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: AgentInput<Environment.Observation, Empty>
  ) -> QNetworkOutput<Empty> {
    QNetworkOutput(qValues: statelessNetwork(input.observation), state: Empty())
  }

  @inlinable
  public func copy() -> StatelessQNetwork<Environment, Network> {
    StatelessQNetwork(statelessNetwork.copy())
  }
}

public struct StatelessActorNetwork<
  Environment: ReinforcementLearning.Environment,
  Network: Module
>: Module where
  Environment.ActionSpace.ValueDistribution: DifferentiableDistribution,
  Network.Input == Environment.Observation,
  Network.Output == Environment.ActionSpace.ValueDistribution,
  Network.Output: KeyPathIterable
{
  public var statelessNetwork: Network

  @inlinable
  public init(_ statelessNetwork: Network) {
    self.statelessNetwork = statelessNetwork
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: AgentInput<Environment.Observation, Empty>
  ) -> ActorOutput<Environment.ActionSpace.ValueDistribution, Empty> {
    ActorOutput(actionDistribution: statelessNetwork(input.observation), state: Empty())
  }
}

public struct StatelessActorCriticNetwork<
  Environment: ReinforcementLearning.Environment,
  Network: Module
>: Module where
  Environment.ActionSpace.ValueDistribution: DifferentiableDistribution,
  Network.Input == Environment.Observation,
  Network.Output == StatelessActorCriticOutput<Environment.ActionSpace.ValueDistribution>
{
  public var statelessNetwork: Network

  @inlinable
  public init(_ statelessNetwork: Network) {
    self.statelessNetwork = statelessNetwork
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: AgentInput<Environment.Observation, Empty>
  ) -> ActorCriticOutput<Environment.ActionSpace.ValueDistribution, Empty> {
    let networkOutput = statelessNetwork(input.observation)
    return ActorCriticOutput(
      actionDistribution: networkOutput.actionDistribution,
      value: networkOutput.value,
      state: Empty())
  }
}
