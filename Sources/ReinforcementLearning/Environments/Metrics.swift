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

public protocol Metric: AnyObject {
  associatedtype Environment: ReinforcementLearning.Environment
  associatedtype Value

  var environment: Environment { get }

  func update()
  func reset()
  func value() -> Value
}

extension Metric {
  @inlinable public func updater() -> MetricUpdater<Self> {
    MetricUpdater(for: self)
  }
}

public final class MetricUpdater<Metric: ReinforcementLearning.Metric>: EnvironmentCallback {
  public let metric: Metric
  @inlinable public init(for metric: Metric) { self.metric = metric }
  @inlinable public func updateOnStep() { metric.update() }
  @inlinable public func updateOnReset() {}
}

public class AverageEpisodeLength<Environment: ReinforcementLearning.Environment>: Metric {
  public let environment: Environment

  @usableFromInline internal var deque: Deque<Float>
  @usableFromInline internal var episodeSteps: Tensor<Int32>

  @inlinable
  public init(for environment: Environment, bufferSize: Int) {
    self.environment = environment
    self.deque = Deque(size: bufferSize)
    self.episodeSteps = Tensor<Int32>(repeating: 0, shape: [environment.batchSize])
  }

  @inlinable
  public func update() {
    let step = environment.currentStep
    let isLast = step.kind.isLast()
    let isNotLast = 1 - Tensor<Int32>(isLast)
    episodeSteps += isNotLast
    for length in episodeSteps.gathering(where: isLast).scalars {
      deque.push(Float(length))
    }
    episodeSteps *= isNotLast
  }

  @inlinable
  public func reset() {
    deque.reset()
    episodeSteps = Tensor<Int32>(zerosLike: episodeSteps)
  }

  @inlinable
  public func value() -> Float {
    deque.mean()
  }
}

public class AverageEpisodeReward<Environment: ReinforcementLearning.Environment>: Metric
where Environment.Reward == Tensor<Float> {
  public let environment: Environment

  @usableFromInline internal var deque: Deque<Float>
  @usableFromInline internal var episodeRewards: Tensor<Float>

  @inlinable
  public init(for environment: Environment, bufferSize: Int) {
    self.environment = environment
    self.deque = Deque(size: bufferSize)
    self.episodeRewards = Tensor<Float>(repeating: 0, shape: [environment.batchSize])
  }

  @inlinable
  public func update() {
    let step = environment.currentStep
    let isLast = step.kind.isLast()
    episodeRewards += step.reward
    for reward in episodeRewards.gathering(where: isLast).scalars {
      deque.push(reward)
    }
    episodeRewards *= (1 - Tensor<Float>(isLast))
  }

  @inlinable
  public func reset() {
    deque.reset()
    episodeRewards = Tensor<Float>(zerosLike: episodeRewards)
  }

  @inlinable
  public func value() -> Float {
    deque.mean()
  }
}

public class TotalCumulativeReward<Environment: ReinforcementLearning.Environment>: Metric
where Environment.Reward == Tensor<Float> {
  public let environment: Environment

  @usableFromInline internal var rewards: Tensor<Float>

  @inlinable
  public init(for environment: Environment) {
    self.environment = environment
    self.rewards = Tensor<Float>(repeating: 0, shape: [environment.batchSize])
  }

  @inlinable
  public func update() {
    rewards += environment.currentStep.reward
  }

  @inlinable
  public func reset() {
    rewards = Tensor<Float>(zerosLike: rewards)
  }

  @inlinable
  public func value() -> [Float] {
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
