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
  associatedtype State
  associatedtype Value

  mutating func update(using trajectory: Trajectory<Observation, Action, Reward, State>)
  mutating func reset()
  func value() -> Value
}

// TODO: Improve this temporary and bad implementation. :)
public struct AverageEpisodeLength<Observation, Action, Reward, State>: Metric {
  public let batchSize: Int
  public let bufferSize: Int

  private var buffer: [Float]
  private var index: Int
  private var full: Bool
  private var episodeSteps: Tensor<Int32>

  public init(batchSize: Int, bufferSize: Int) {
    self.batchSize = batchSize
    self.bufferSize = bufferSize
    self.buffer = [Float](repeating: 0, count: bufferSize)
    self.index = 0
    self.full = false
    self.episodeSteps = Tensor<Int32>(repeating: 0, shape: [batchSize])
  }

  public mutating func update(using trajectory: Trajectory<Observation, Action, Reward, State>) {
    let isLast = Tensor<Int32>(trajectory.stepKind.isLast())
    let isNotLast = 1 - isLast
    episodeSteps += isNotLast
    let count = isLast.sum().scalarized()
    let totalLength = (episodeSteps * isLast).sum().scalarized()
    let averageLength = Float(totalLength) / Float(count)
    for _ in 0..<count {
      buffer[index] = averageLength
      index += 1
      full = full || index == buffer.count
      index = index % buffer.count
    }
    episodeSteps *= isNotLast
  }

  public mutating func reset() {
    index = 0
    full = false
    episodeSteps = Tensor<Int32>(repeating: 0, shape: [batchSize])
  }

  public func value() -> Float {
    if full {
      return buffer.reduce(0, +) / Float(buffer.count)
    }
    return buffer[0..<index].reduce(0, +) / Float(index)
  }
}
