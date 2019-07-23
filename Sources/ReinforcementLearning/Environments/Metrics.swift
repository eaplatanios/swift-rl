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

  @usableFromInline internal var deque: Deque<Float>
  @usableFromInline internal var episodeSteps: Tensor<Int32>

  @inlinable
  public init(batchSize: Int, bufferSize: Int) {
    self.batchSize = batchSize
    self.deque = Deque(size: bufferSize)
    self.episodeSteps = Tensor<Int32>(repeating: 0, shape: [batchSize])
  }

  @inlinable
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

  @inlinable
  public mutating func reset() {
    deque.reset()
    episodeSteps = Tensor<Int32>(repeating: 0, shape: [batchSize])
  }

  @inlinable
  public func value() -> Float {
    deque.mean()
  }
}

public struct AverageEpisodeReward<Observation, Action, Reward: TensorFlowFloatingPoint>: Metric {
  public let batchSize: Int

  @usableFromInline internal var deque: Deque<Reward>
  @usableFromInline internal var episodeRewards: Tensor<Reward>

  @inlinable
  public init(batchSize: Int, bufferSize: Int) {
    self.batchSize = batchSize
    self.deque = Deque(size: bufferSize)
    self.episodeRewards = Tensor<Reward>(repeating: 0, shape: [batchSize])
  }

  @inlinable
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

  @inlinable
  public mutating func reset() {
    deque.reset()
    episodeRewards = Tensor<Reward>(repeating: 0, shape: [batchSize])
  }

  @inlinable
  public func value() -> Reward {
    deque.mean()
  }
}

// TODO: Make generic over the `Reward` type.
public struct TotalCumulativeReward<Observation, Action, Reward: TensorFlowFloatingPoint>: Metric {
  public let batchSize: Int

  @usableFromInline internal var rewards: Tensor<Reward>

  @inlinable
  public init(batchSize: Int) {
    self.batchSize = batchSize
    self.rewards = Tensor<Reward>(repeating: 0, shape: [batchSize])
  }

  @inlinable
  public mutating func update(
    stepKind: StepKind,
    observation: Observation,
    action: Action,
    reward: Tensor<Reward>
  ) {
    rewards += reward
  }

  @inlinable
  public mutating func reset() {
    rewards = Tensor<Reward>(repeating: 0, shape: [batchSize])
  }

  @inlinable
  public func value() -> [Reward] {
    rewards.scalars
  }
}

@usableFromInline
internal struct Deque<Scalar: FloatingPoint> {
  @usableFromInline internal let size: Int
  @usableFromInline internal var buffer: [Scalar]
  @usableFromInline internal var index: Int
  @usableFromInline internal var full: Bool

  @inlinable
  init(size: Int) {
    self.size = size
    self.buffer = [Scalar](repeating: 0, count: size)
    self.index = 0
    self.full = false
  }

  @inlinable
  mutating func push(_ value: Scalar) {
    buffer[index] = value
    index += 1
    full = full || index == buffer.count
    index = index % buffer.count
  }

  @inlinable
  mutating func reset() {
    index = 0
    full = false
  }

  @inlinable
  func mean() -> Scalar {
    let sum = full ? buffer.reduce(0, +) : buffer[0..<index].reduce(0, +)
    let count = full ? buffer.count : index
    return sum / Scalar(count)
  }
}
