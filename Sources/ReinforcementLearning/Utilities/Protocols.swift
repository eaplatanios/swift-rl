import TensorFlow

public protocol Replayable {
  init(emptyLike example: Self, withCapacity capacity: Int)

  mutating func update(atIndices indices: Tensor<Int64>, using values: Self)
  func gathering(atIndices indices: Tensor<Int64>) -> Self
}

public protocol Batchable {
  func flattenedBatch(outerDimCount: Int) -> Self
  func unflattenedBatch(outerDims: [Int]) -> Self
}

public protocol DifferentiableBatchable: Batchable, Differentiable {
  @differentiable(wrt: self)
  func flattenedBatch(outerDimCount: Int) -> Self

  @differentiable(wrt: self)
  func unflattenedBatch(outerDims: [Int]) -> Self
}

extension Tensor: Replayable where Scalar: Numeric {
  public init(emptyLike example: Tensor, withCapacity capacity: Int) {
    if example.rank <= 1 {
      self.init(zeros: [capacity])
    } else {
      self.init(zeros: TensorShape([capacity] + example.shape.dimensions[1...]))
    }
  }

  public mutating func update(atIndices indices: Tensor<Int64>, using values: Tensor) {
    self = Raw.tensorScatterUpdate(self, indices: indices, updates: values)
  }

  public func gathering(atIndices indices: Tensor<Int64>) -> Tensor {
    gathering(atIndices: indices, alongAxis: 0)
  }
}

extension Tensor: Batchable {
  public func flattenedBatch(outerDimCount: Int) -> Tensor {
    if outerDimCount == 1 {
      return self
    }
    var newShape = [-1]
    for i in outerDimCount..<rank {
      newShape.append(shape[i])
    }
    return reshaped(to: TensorShape(newShape))
  }

  public func unflattenedBatch(outerDims: [Int]) -> Tensor {
    if rank > 1 {
      return reshaped(to: TensorShape(outerDims + shape.dimensions[1...]))
    }
    return reshaped(to: TensorShape(outerDims))
  }
}

extension Tensor: DifferentiableBatchable where Scalar: TensorFlowFloatingPoint {
  @differentiable(wrt: self)
  public func flattenedBatch(outerDimCount: Int) -> Tensor {
    if outerDimCount == 1 {
      return self
    }
    var newShape = [-1]
    for i in outerDimCount..<rank {
      newShape.append(shape[i])
    }
    return reshaped(to: TensorShape(newShape))
  }

  @differentiable(wrt: self)
  public func unflattenedBatch(outerDims: [Int]) -> Tensor {
    if rank > 1 {
      return reshaped(to: TensorShape(outerDims + shape.dimensions[1...]))
    }
    return reshaped(to: TensorShape(outerDims))
  }
}

extension KeyPathIterable {
  public init(emptyLike example: Self, withCapacity capacity: Int) {
    self = example
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Int32>.self) {
      self[keyPath: kp] = Tensor(emptyLike: example[keyPath: kp], withCapacity: capacity)
    }
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Int64>.self) {
      self[keyPath: kp] = Tensor(emptyLike: example[keyPath: kp], withCapacity: capacity)
    }
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      self[keyPath: kp] = Tensor(emptyLike: example[keyPath: kp], withCapacity: capacity)
    }
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      self[keyPath: kp] = Tensor(emptyLike: example[keyPath: kp], withCapacity: capacity)
    }
  }

  public mutating func update(atIndices indices: Tensor<Int64>, using values: Self) {
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Int32>.self) {
      self[keyPath: kp].update(atIndices: indices, using: values[keyPath: kp])
    }
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Int64>.self) {
      self[keyPath: kp].update(atIndices: indices, using: values[keyPath: kp])
    }
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      self[keyPath: kp].update(atIndices: indices, using: values[keyPath: kp])
    }
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      self[keyPath: kp].update(atIndices: indices, using: values[keyPath: kp])
    }
  }

  public func gathering(atIndices indices: Tensor<Int64>) -> Self {
    var result = self
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int32>.self) {
      result[keyPath: kp] = result[keyPath: kp].gathering(atIndices: indices)
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int64>.self) {
      result[keyPath: kp] = result[keyPath: kp].gathering(atIndices: indices)
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      result[keyPath: kp] = result[keyPath: kp].gathering(atIndices: indices)
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      result[keyPath: kp] = result[keyPath: kp].gathering(atIndices: indices)
    }
    return result
  }

  public func flattenedBatch(outerDimCount: Int) -> Self {
    var result = self
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      result[keyPath: kp] = result[keyPath: kp].flattenedBatch(outerDimCount: outerDimCount)
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      result[keyPath: kp] = result[keyPath: kp].flattenedBatch(outerDimCount: outerDimCount)
    }
    return result
  }

  public func unflattenedBatch(outerDims: [Int]) -> Self {
    var result = self
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      result[keyPath: kp] = result[keyPath: kp].unflattenedBatch(outerDims: outerDims)
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      result[keyPath: kp] = result[keyPath: kp].unflattenedBatch(outerDims: outerDims)
    }
    return result
  }
}

extension KeyPathIterable where Self: Differentiable, Self.TangentVector: KeyPathIterable {
  @differentiable(wrt: self, vjp: _vjpFlattenedBatch)
  public func flattenedBatch(outerDimCount: Int) -> Self {
    var result = self
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      result[keyPath: kp] = result[keyPath: kp].flattenedBatch(outerDimCount: outerDimCount)
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      result[keyPath: kp] = result[keyPath: kp].flattenedBatch(outerDimCount: outerDimCount)
    }
    return result
  }

  @differentiable(wrt: self, vjp: _vjpUnflattenedBatch)
  public func unflattenedBatch(outerDims: [Int]) -> Self {
    var result = self
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      result[keyPath: kp] = result[keyPath: kp].unflattenedBatch(outerDims: outerDims)
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      result[keyPath: kp] = result[keyPath: kp].unflattenedBatch(outerDims: outerDims)
    }
    return result
  }
}

internal extension KeyPathIterable
where Self: Differentiable, Self.TangentVector: KeyPathIterable {
  @usableFromInline
  func _vjpFlattenedBatch(outerDimCount: Int) -> (Self, (TangentVector) -> TangentVector) {
    // TODO: This is very hacky.
    var outerDims = [Int]()
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      outerDims = [Int](self[keyPath: kp].shape.dimensions[0..<outerDimCount])
      break
    }
    let result = flattenedBatch(outerDimCount: outerDimCount)
    return (result, { seed in
      var result = seed
      for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
        result[keyPath: kp] = seed[keyPath: kp].unflattenedBatch(outerDims: outerDims)
      }
      for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
        result[keyPath: kp] = seed[keyPath: kp].unflattenedBatch(outerDims: outerDims)
      }
      return result
    })
  }

  @usableFromInline
  func _vjpUnflattenedBatch(outerDims: [Int]) -> (Self, (TangentVector) -> TangentVector) {
    let result = unflattenedBatch(outerDims: outerDims)
    return (result, { seed in
      var result = seed
      for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
        result[keyPath: kp] = seed[keyPath: kp].flattenedBatch(outerDimCount: outerDims.count)
      }
      for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
        result[keyPath: kp] = seed[keyPath: kp].flattenedBatch(outerDimCount: outerDims.count)
      }
      return result
    })
  }
}
