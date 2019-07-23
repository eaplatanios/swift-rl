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

public protocol Metric {
  associatedtype Observation
  associatedtype Action
  associatedtype Reward
  associatedtype Value

  mutating func update(
    stepKind: StepKind,
    observation: Observation,
    action: Action,
    reward: Reward)

  mutating func reset()

  func value() -> Value
}

public struct AverageEpisodeLength<Observation, Action, Reward>: Metric {
  public let batchSize: Int

  private var deque: Deque<Float>
  private var episodeSteps: Tensor<Int32>

  public init(batchSize: Int, bufferSize: Int) {
    self.batchSize = batchSize
    self.deque = Deque(size: bufferSize)
    self.episodeSteps = Tensor<Int32>(repeating: 0, shape: [batchSize])
  }

  public mutating func update(
    stepKind: StepKind,
    observation: Observation,
    action: Action,
    reward: Reward
  ) {
    let isLast = stepKind.isLast()
    let isNotLast = 1 - Tensor<Int32>(isLast)
    episodeSteps += isNotLast
    for length in episodeSteps.gathering(where: isLast).scalars {
      deque.push(Float(length))
    }
    episodeSteps *= isNotLast
  }

  public mutating func reset() {
    deque.reset()
    episodeSteps = Tensor<Int32>(repeating: 0, shape: [batchSize])
  }

  public func value() -> Float {
    deque.mean()
  }
}

public struct AverageEpisodeReward<Observation, Action, Reward: TensorFlowFloatingPoint>: Metric {
  public let batchSize: Int

  private var deque: Deque<Reward>
  private var episodeRewards: Tensor<Reward>

  public init(batchSize: Int, bufferSize: Int) {
    self.batchSize = batchSize
    self.deque = Deque(size: bufferSize)
    self.episodeRewards = Tensor<Reward>(repeating: 0, shape: [batchSize])
  }

  public mutating func update(
    stepKind: StepKind,
    observation: Observation,
    action: Action,
    reward: Tensor<Reward>
  ) {
    let isLast = stepKind.isLast()
    episodeRewards += reward
    for reward in episodeRewards.gathering(where: isLast).scalars {
      deque.push(reward)
    }
    episodeRewards *= (1 - Tensor<Reward>(isLast))
  }

  public mutating func reset() {
    deque.reset()
    episodeRewards = Tensor<Reward>(repeating: 0, shape: [batchSize])
  }

  public func value() -> Reward {
    deque.mean()
  }
}

// TODO: Make generic over the `Reward` type.
public struct TotalCumulativeReward<Observation, Action, Reward: TensorFlowFloatingPoint>: Metric {
  public let batchSize: Int

  private var rewards: Tensor<Reward>

  public init(batchSize: Int) {
    self.batchSize = batchSize
    self.rewards = Tensor<Reward>(repeating: 0, shape: [batchSize])
  }

  public mutating func update(
    stepKind: StepKind,
    observation: Observation,
    action: Action,
    reward: Tensor<Reward>
  ) {
    rewards += reward
  }

  public mutating func reset() {
    rewards = Tensor<Reward>(repeating: 0, shape: [batchSize])
  }

  public func value() -> [Reward] {
    rewards.scalars
  }
}
