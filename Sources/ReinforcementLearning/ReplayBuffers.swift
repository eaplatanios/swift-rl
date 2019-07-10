import Foundation
import TensorFlow

// TODO: Should be derived automatically.
public protocol Replayable: TensorGroup {
  init(emptyLike example: Self, withCapacity capacity: Int)

  mutating func update(atIndices indices: Tensor<Int64>, using values: Self)
  func gathering(atIndices indices: Tensor<Int64>) -> Self
}

extension Tensor: Replayable where Scalar: Numeric {
  public init(emptyLike example: Tensor, withCapacity capacity: Int) {
    self.init(zeros: TensorShape([capacity] + example.shape.dimensions[1...]))
  }

  public mutating func update(atIndices indices: Tensor<Int64>, using values: Tensor) {
    self = Raw.tensorScatterUpdate(self, indices: indices, updates: values)
  }

  public func gathering(atIndices indices: Tensor<Int64>) -> Tensor {
    gathering(atIndices: indices, alongAxis: 0)
  }
}

public protocol ReplayBuffer {
  associatedtype Data: Replayable

  /// Total capacity of this replay buffer (i.e., total number of trajectory steps that can be
  /// recorded to it).
  var capacity: Int { get }
  
  /// Records the provided data batch to this replay buffer.
  mutating func record(_ batch: Data)

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

public struct ReplayBufferBatch<Data: Replayable> {
  public let batch: Data
  public let ids: Tensor<Int64>
  public let probabilities: Tensor<Float>
}

public struct UniformReplayBuffer<Data: Replayable>: ReplayBuffer {
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

  public init(batchSize: Int, maxLength: Int) {
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
    let rows = batchOffsets + id % Int64(maxLength)
    idsStorage!.update(atIndices: rows, using: Tensor<Int64>(id))
    dataStorage!.update(atIndices: rows, using: batch)
  }

  /// Returns a batch sampled uniformly at random from the recorded data.
  ///
  /// - Parameters:
  ///   - batchSize: Batch size.
  ///   - stepCount: Number of time steps to include for each batch element. If
  ///     `stepCount == nil`, the returned batch consists of tensor groups where each
  ///     tensor has shape `[batchSize, ...]`. Otherwise, each such tensor has shape
  ///     `[batchSize, stepCount, ...]`.
  /// - Returns: Batch sampled uniformly at random from the recorded data.
  public func sampleBatch(batchSize: Int = 1, stepCount: Int? = nil) -> ReplayBufferBatch<Data> {
    let idsRange = validIDsRange(stepCount: Int64(stepCount ?? 1))
    let validIDCount = idsRange.1 - idsRange.0
    precondition(
      validIDCount > 0 && idsStorage != nil && dataStorage != nil,
      "The buffer is empty. Make sure to add data before sampling batches from the buffer.")

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
        .tiled(multiples: Tensor<Int32>([1, Int32(stepCount)]))
      let stepRange = Tensor<Int64>(rangeFrom: 0, to: Int64(stepCount), stride: 1)
        .reshaped(to: [1, stepCount])
        .tiled(multiples: Tensor<Int32>([Int32(batchSize), 1]))
      indices = (stepRange + indices) % Int64(capacity)
    } else {
      indices = indices % Int64(capacity)
    }

    let ids = idsStorage!.gathering(atIndices: indices)
    let batch = dataStorage!.gathering(atIndices: indices)
    let probabilities = Tensor<Float>(
      repeating: 1 / Float(validIDCount * Int64(self.batchSize)),
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
    return lastID < maxLength ? (minIDNotFull, minIDFull) : (maxIDNotFull, maxIDFull)
  }
}
