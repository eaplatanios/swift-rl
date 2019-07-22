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

public protocol Stackable {
  static func stack(_ values: [Self]) -> Self
  func unstacked() -> [Self]
}

public protocol DifferentiableStackable: Stackable, Differentiable {
  @differentiable
  static func stack(_ values: [Self]) -> Self

  @differentiable
  func unstacked() -> [Self]
}

extension Tensor: Stackable {
  public static func stack(_ values: [Tensor]) -> Tensor {
    Tensor(stacking: values, alongAxis: 0)
  }

  public func unstacked() -> [Tensor] {
    unstacked(alongAxis: 0)
  }
}

extension Tensor: DifferentiableStackable where Scalar: TensorFlowFloatingPoint {
  @differentiable
  public static func stack(_ values: [Tensor]) -> Tensor {
    Tensor(stacking: values, alongAxis: 0)
  }

  @differentiable
  public func unstacked() -> [Tensor] {
    unstacked(alongAxis: 0)
  }
}

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
    // TODO: Remove this hack once the S4TF auto-diff memory leak is fixed.
    let newShape = Swift.withoutDerivative(at: self.shape) { shape -> [Int] in 
      var newShape = [-1]
      for i in outerDimCount..<shape.count {
        newShape.append(shape[i])
      }
      return newShape
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

extension Tensor where Scalar: TensorFlowFloatingPoint {
  public mutating func update(using other: Tensor, forgetFactor: Float) {
    let forgetFactor = Scalar(forgetFactor)
    self = forgetFactor * self + (1 - forgetFactor) * other
  }
}

extension KeyPathIterable {
  public static func stack(_ values: [Self]) -> Self {
    var result = values[0]
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<UInt8>.self) {
      result[keyPath: kp] = Tensor.stack(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int32>.self) {
      result[keyPath: kp] = Tensor.stack(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int64>.self) {
      result[keyPath: kp] = Tensor.stack(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      result[keyPath: kp] = Tensor.stack(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      result[keyPath: kp] = Tensor.stack(values.map { $0[keyPath: kp] })
    }
    return result
  }

  public func unstacked() -> [Self] {
    var result = [Self]()
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<UInt8>.self) {
      let unstacked = self[keyPath: kp].unstacked()
      if result.isEmpty {
        result = [Self](repeating: self, count: unstacked.count)
      }
      for i in result.indices {
        result[i][keyPath: kp] = unstacked[i]
      }
    }
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Int32>.self) {
      let unstacked = self[keyPath: kp].unstacked()
      if result.isEmpty {
        result = [Self](repeating: self, count: unstacked.count)
      }
      for i in result.indices {
        result[i][keyPath: kp] = unstacked[i]
      }
    }
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Int64>.self) {
      let unstacked = self[keyPath: kp].unstacked()
      if result.isEmpty {
        result = [Self](repeating: self, count: unstacked.count)
      }
      for i in result.indices {
        result[i][keyPath: kp] = unstacked[i]
      }
    }
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      let unstacked = self[keyPath: kp].unstacked()
      if result.isEmpty {
        result = [Self](repeating: self, count: unstacked.count)
      }
      for i in result.indices {
        result[i][keyPath: kp] = unstacked[i]
      }
    }
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      let unstacked = self[keyPath: kp].unstacked()
      if result.isEmpty {
        result = [Self](repeating: self, count: unstacked.count)
      }
      for i in result.indices {
        result[i][keyPath: kp] = unstacked[i]
      }
    }
    return result
  }

  public init(emptyLike example: Self, withCapacity capacity: Int) {
    self = example
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<UInt8>.self) {
      self[keyPath: kp] = Tensor(emptyLike: example[keyPath: kp], withCapacity: capacity)
    }
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
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<UInt8>.self) {
      self[keyPath: kp].update(atIndices: indices, using: values[keyPath: kp])
    }
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
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<UInt8>.self) {
      result[keyPath: kp] = result[keyPath: kp].gathering(atIndices: indices)
    }
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

  public mutating func update(using other: Self, forgetFactor: Float) {
    var result = self
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      result[keyPath: kp].update(using: other[keyPath: kp], forgetFactor: forgetFactor)
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      result[keyPath: kp].update(using: other[keyPath: kp], forgetFactor: forgetFactor)
    }
  }
}

extension KeyPathIterable where Self: Differentiable, Self.TangentVector: KeyPathIterable {
  // TODO: Differentiable `stack` and `unstacked`.

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
