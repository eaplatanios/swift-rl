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
import TensorFlow

public protocol ReplayBuffer {
  associatedtype Data: KeyPathIterable

  /// Total capacity of this replay buffer (i.e., total number of trajectory steps that can be
  /// recorded to it).
  var capacity: Int { get }
  
  /// Records the provided data batch to this replay buffer.
  mutating func record(_ batch: Data)

  /// Returns all of the data recorded in this replay buffer.
  ///
  /// - Returns: Recorded data in the form of a tensor group where each tensor has shape
  ///   `[batchSize, maxSequenceLength, ...]`.
  func recordedData() -> Data

  /// Returns a batch sampled uniformly at random from the recorded data.
  ///
  /// - Parameters:
  ///   - batchSize: Batch size.
  ///   - stepCount: Number of time steps to include for each batch element. If
  ///     `stepCount == nil`, the returned batch consists of tensor groups where each
  ///     tensor has shape `[batchSize, ...]`. Otherwise, each such tensor has shape
  ///     `[batchSize, stepCount, ...]`.
  /// - Returns: Batch sampled uniformly at random from the recorded data.
  func sampleBatch(batchSize: Int, stepCount: Int?) -> ReplayBufferBatch<Data>

  /// Resets the contents of this buffer.
  mutating func reset()
}

public struct ReplayBufferBatch<Data: KeyPathIterable> {
  public let batch: Data
  public let ids: Tensor<Int64>
  public let probabilities: Tensor<Float>
}

public struct UniformReplayBuffer<Data: KeyPathIterable>: ReplayBuffer {
  public let batchSize: Int
  public let maxLength: Int
  public let capacity: Int

  internal var batchOffsets: Tensor<Int64> {
    Tensor<Int64>(rangeFrom: 0, to: Int64(batchSize), stride: 1) * Int64(maxLength)
  }

  internal let lastIDCounterDispatchQueue = DispatchQueue(label: "UniformReplayBuffer")
  internal var lastID: Int64 = -1

  internal var idsStorage: Tensor<Int64>? = nil
  internal var dataStorage: Data? = nil

  public init<Agent: ReinforcementLearning.Agent>(
    for agent: Agent,
    batchSize: Int,
    maxLength: Int
  ) where Data == Trajectory<Agent.Action, Agent.Observation, Agent.Reward, Agent.State> {
    self.batchSize = batchSize
    self.maxLength = maxLength
    self.capacity = batchSize * maxLength
  }

  /// Records the provided data batch to this replay buffer.
  public mutating func record(_ batch: Data) {
    if idsStorage == nil {
      idsStorage = Tensor<Int64>(emptyLike: Tensor<Int64>([1]), withCapacity: capacity)
      dataStorage = Data(emptyLike: batch, withCapacity: capacity)
    }
    let id: Int64 = lastIDCounterDispatchQueue.sync {
      lastID += 1
      return lastID
    }
    let indices = (batchOffsets + id % Int64(maxLength)).expandingShape(at: 1)
    idsStorage!.update(
      atIndices: indices,
      using: Tensor<Int64>([Int64](repeating: id, count: batchSize)))
    dataStorage!.update(atIndices: indices, using: batch)
  }

  /// Returns all of the data recorded in this replay buffer.
  ///
  /// - Returns: Recorded data in the form of a tensor group where each tensor has shape
  ///   `[maxSequenceLength, batchSize, ...]`.
  public func recordedData() -> Data {
    // Repeat `ids` over `batchSize` resulting in a tensor with shape
    // `[idsRange.1 - idsRange.0, batchSize, ...]`.
    let idsRange = validIDsRange()
    let ids = Tensor<Int64>(
      rangeFrom: idsRange.0,
      to: idsRange.1,
      stride: 1
    ).expandingShape(at: 1).tiled(multiples: Tensor<Int32>([1, Int32(batchSize)]))

    // Create the `batchOffsets` with shape `[1, batchSize]`, and then add them to `ids` to obtain
    // the row indices in the storage tensors.
    let batchOffsets = Tensor<Int64>(
      rangeFrom: Tensor<Int64>(0),
      to: Tensor<Int64>(Int64(batchSize)),
      stride: Tensor<Int64>(1)
    ).expandingShape(at: 0) * Int64(maxLength)
    let indices = ids % Int64(maxLength) + batchOffsets

    return dataStorage!.gathering(atIndices: indices)
  }

  /// Returns a batch sampled uniformly at random from the recorded data.
  ///
  /// - Parameters:
  ///   - batchSize: Batch size.
  ///   - stepCount: Number of time steps to include for each batch element. If
  ///     `stepCount == nil`, the returned batch consists of tensor groups where each
  ///     tensor has shape `[batchSize, ...]`. Otherwise, each such tensor has shape
  ///     `[stepCount, batchSize, ...]`.
  /// - Returns: Batch sampled uniformly at random from the recorded data.
  public func sampleBatch(batchSize: Int = 1, stepCount: Int? = nil) -> ReplayBufferBatch<Data> {
    let idsRange = validIDsRange(stepCount: Int64(stepCount ?? 1))
    let idsCount = idsRange.1 - idsRange.0
    precondition(idsCount > 0 && idsStorage != nil && dataStorage != nil, "Empty buffer.")

    // Sample random IDs across multiple random batches.
    let batchOffsets = Tensor<Int64>(
      randomUniform: [batchSize],
      lowerBound: Tensor<Int64>(0),
      upperBound: Tensor<Int64>(Int64(self.batchSize))
    ) * Int64(maxLength)
    var indices = Tensor<Int64>(
      randomUniform: [batchSize],
      lowerBound: Tensor<Int64>(idsRange.0),
      upperBound: Tensor<Int64>(idsRange.1)
    ) + batchOffsets

    if let stepCount = stepCount {
      indices = indices.expandingShape(at: -1)
        .tiled(multiples: Tensor<Int32>([Int32(stepCount), 1]))
      let stepRange = Tensor<Int64>(rangeFrom: 0, to: Int64(stepCount), stride: 1)
        .reshaped(to: [stepCount, 1])
        .tiled(multiples: Tensor<Int32>([1, Int32(batchSize)]))
      indices = (stepRange + indices) % Int64(capacity)
    } else {
      indices = indices % Int64(capacity)
    }

    let ids = idsStorage!.gathering(atIndices: indices)
    let batch = dataStorage!.gathering(atIndices: indices)
    let probabilities = Tensor<Float>(
      repeating: 1 / Float(idsCount * Int64(self.batchSize)),
      shape: [batchSize])

    return ReplayBufferBatch(batch: batch, ids: ids, probabilities: probabilities)
  }

  /// Resets the contents of this buffer.
  public mutating func reset() {
    lastIDCounterDispatchQueue.sync {
      lastID = -1
      idsStorage = nil
      dataStorage = nil
    }
  }

  /// Returns the range of valid IDs.
  ///
  /// - Parameter stepCount: Optional way to specify how many IDs need to be valid.
  /// - Returns: Tuple representing the range, where the first element is the inclusive lower bound
  ///   and the second element is the exclusive upper bound.
  /// - Note: When `stepCount` is provided, the upper bound of the range can be increased by up to
  ///   `stepCount`.
  internal func validIDsRange(stepCount: Int64 = 1) -> (Int64, Int64) {
    let lastID = lastIDCounterDispatchQueue.sync { self.lastID }
    let minIDNotFull = Int64(0)
    let maxIDNotFull = max(lastID + 1 - stepCount + 1, 0)
    let minIDFull = lastID + 1 - Int64(maxLength)
    let maxIDFull = lastID + 1 - stepCount + 1
    return lastID < maxLength ? (minIDNotFull, maxIDNotFull) : (minIDFull, maxIDFull)
  }
}
