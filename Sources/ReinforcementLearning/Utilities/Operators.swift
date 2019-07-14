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

public extension Tensor where Scalar: Numeric {
  /// Returns the cumulative sum of this tensor along the specified axis. By default, this
  /// function performs an inclusive cumulative sum which means that the first element of the
  /// input is identical to the first element of the output:
  /// ```
  /// Tensor<Float>([a, b, c]).cumulativeSum() = Tensor<Float>([a, a + b, a + b + c])
  /// ```
  /// By setting the `exclusive` argument to `true`, an exclusive cumulative sum is performed
  /// instead:
  /// ```
  /// Tensor<Float>([a, b, c]).cumulativeSum(exclusive: true) = Tensor<Float>([0, a, a + b])
  /// ```
  /// By setting the `reverse` argument to `true`, the cumulative sum is performed in the
  /// opposite direction:
  /// ```
  /// Tensor<Float>([a, b, c]).cumulativeSum(reverse: true) = 
  ///   Tensor<Float>([a + b + c, a + b, a])
  /// ```
  /// This is more efficient than separately reversing the resulting tensor.
  ///
  /// - Parameters:
  ///   - axis: Axis along which to perform the cumulative sum operation.
  ///   - exclusive: Indicates whether to perform an exclusive cumulative sum.
  ///   - reverse: Indicates whether to perform the cumulative sum in reversed order.
  /// - Returns: Result of the cumulative sum operation.
  /// - Precondition: `axis` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func cumulativeSum(
    alongAxis axis: Int,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> Tensor {
    cumulativeSum(
      alongAxis: Tensor<Int32>(Int32(axis)),
      exclusive: exclusive,
      reverse: reverse)
  }

  /// Returns the cumulative sum of this tensor along the specified axis. By default, this
  /// function performs an inclusive cumulative sum which means that the first element of the
  /// input is identical to the first element of the output:
  /// ```
  /// Tensor<Float>([a, b, c]).cumulativeSum() = Tensor<Float>([a, a + b, a + b + c])
  /// ```
  /// By setting the `exclusive` argument to `true`, an exclusive cumulative sum is performed
  /// instead:
  /// ```
  /// Tensor<Float>([a, b, c]).cumulativeSum(exclusive: true) = Tensor<Float>([0, a, a + b])
  /// ```
  /// By setting the `reverse` argument to `true`, the cumulative sum is performed in the
  /// opposite direction:
  /// ```
  /// Tensor<Float>([a, b, c]).cumulativeSum(reverse: true) = 
  ///   Tensor<Float>([a + b + c, a + b, a])
  /// ```
  /// This is more efficient than separately reversing the resulting tensor.
  ///
  /// - Parameters:
  ///   - axis: Axis along which to perform the cumulative sum operation.
  ///   - exclusive: Indicates whether to perform an exclusive cumulative sum.
  ///   - reverse: Indicates whether to perform the cumulative sum in reversed order.
  /// - Returns: Result of the cumulative sum operation.
  /// - Precondition: `axis.rank` must be `0`.
  /// - Precondition: `axis` must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(wrt: self, vjp: _vjpCumulativeSum where Scalar: TensorFlowFloatingPoint)
  func cumulativeSum(
    alongAxis axis: Tensor<Int32>,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> Tensor {
    Raw.cumsum(self, axis: axis, exclusive: exclusive, reverse: reverse)
  }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  func _vjpCumulativeSum(
    alongAxis axis: Tensor<Int32>,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> (Tensor, (Tensor) -> Tensor) {
    (cumulativeSum(alongAxis: axis, exclusive: exclusive, reverse: reverse), { v in
      v.cumulativeSum(alongAxis: axis, exclusive: exclusive, reverse: !reverse)
    })
  }
}

public extension Tensor where Scalar: Numeric {
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func cumulativeProduct(
    alongAxis axis: Int,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> Tensor {
    cumulativeProduct(
      alongAxis: Tensor<Int32>(Int32(axis)),
      exclusive: exclusive,
      reverse: reverse)
  }

  @inlinable
  @differentiable(wrt: self, vjp: _vjpCumulativeProduct where Scalar: TensorFlowFloatingPoint)
  func cumulativeProduct(
    alongAxis axis: Tensor<Int32>,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> Tensor {
    Raw.cumprod(self, axis: axis, exclusive: exclusive, reverse: reverse)
  }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  func _vjpCumulativeProduct(
    alongAxis axis: Tensor<Int32>,
    exclusive: Bool = false,
    reverse: Bool = false
  ) -> (Tensor, (Tensor) -> Tensor) {
    let result = cumulativeProduct(alongAxis: axis, exclusive: exclusive, reverse: reverse)
    return (result, { v in
      (result * v).cumulativeSum(alongAxis: axis, exclusive: exclusive, reverse: !reverse) / self
    })
  }
}

/// Returns the squared difference between `x` and `y`.
/// - Returns: `(x - y) ^ 2`.
@inlinable
@differentiable(vjp: _vjpSquaredDifference where T: TensorFlowFloatingPoint)
public func squaredDifference<T: TensorFlowNumeric>(_ x: Tensor<T>, _ y: Tensor<T>) -> Tensor<T> {
  Raw.squaredDifference(x, y)
}

@inlinable
internal func _vjpSquaredDifference<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
  (squaredDifference(x, y), { seed in
    let lhsGrad = 2 * seed * (x - y)
    let rhsGrad = -lhsGrad
    let (lhsShape, rhsShape) = (x.shapeTensor, y.shapeTensor)
    let (lhsAxes, rhsAxes) = Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
    return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
            rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
  })
}

/// Pair of first and second moments (i.e., mean and variance).
/// - Note: This is needed because tuple types are not differentiable.
public struct Moments<Scalar: TensorFlowFloatingPoint>: Differentiable {
  public var mean: Tensor<Scalar>
  public var variance: Tensor<Scalar>

  @differentiable
  public init(mean: Tensor<Scalar>, variance: Tensor<Scalar>) {
    self.mean = mean
    self.variance = variance
  }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Returns the mean and variance of this tensor along the specified axes. The reduced dimensions
  /// are removed.
  @inlinable
  @differentiable(wrt: self)
  func moments(squeezingAxes axes: [Int]) -> Moments<Scalar> {
    let mean = self.mean(alongAxes: axes)
    let variance = squaredDifference(self, mean).mean(alongAxes: axes)
    return Moments(
      mean: mean.squeezingShape(at: axes),
      variance: variance.squeezingShape(at: axes))
  }

  /// Returns the mean and variance of this tensor along the specified axes. The reduced dimensions
  /// are removed.
  @inlinable
  @differentiable(wrt: self)
  func moments(squeezingAxes axes: Int...) -> Moments<Scalar> {
    moments(squeezingAxes: axes)
  }

  /// Returns the mean and variance of this tensor's elements.
  @inlinable
  @differentiable(wrt: self)
  func moments() -> Moments<Scalar> {
    moments(squeezingAxes: Array(0..<shape.rank))
  }

  /// Returns the mean and variance of this tensor along the specified axes. The reduced dimensions
  /// are retained with value `1`.
  @inlinable
  @differentiable(wrt: self)
  func moments(alongAxes axes: [Int]) -> Moments<Scalar> {
    let mean = self.mean(alongAxes: axes)
    let variance = squaredDifference(self, mean).mean(alongAxes: axes)
    return Moments(mean: mean, variance: variance)
  }

  /// Returns the mean and variance of this tensor along the specified axes. The reduced dimensions
  /// are retained with value `1`.
  @inlinable
  @differentiable(wrt: self)
  func moments(alongAxes axes: Int...) -> Moments<Scalar> {
    moments(alongAxes: axes)
  }
}

public extension Tensor {
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func batchGatheringV2<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>,
    alongAxis axis: Int = 1,
    batchDimensionCount: Int = 1
  ) -> Tensor {
    // TODO: precondition(batchDimensionCount >= 0,
    //                    "'batchDimensionCount' must be non-negative.")
    // TODO: precondition(batchDimensionCount < indices.rank,
    //                    "'batchDimensionCount' must be less than 'indices.rank'.")
    // TODO: precondition(batchDimensionCount < rank, 
    //                    "'batchDimensionCount' must be less than the tensor's rank.")

    // Handle the axis argument by transposing the axis dimension so that it is the first
    // non-batch dimension, recursively calling `batchGathering` with `axis = 0`, and then
    // transposing the result to put the pre-axis dimensions before the indices dimensions.
    if axis != batchDimensionCount {
      // Adjust axis to be positive.
      let posAxis = axis < 0 ? axis + rank : axis

      // TODO: precondition(posAxis >= 0 && posAxis < rank, "'axis' is out of range.")
      // TODO: precondition(batchDimensionCount <= posAxis,
      //                    "'batchDimensionCount' must be less than or equal to 'axis'.")

      // Move self[axis] up to self[batchDimensionCount].
      let permutation = Tensor<Int32>(concatenating: [
        Tensor<Int32>(rangeFrom: 0, to: Int32(batchDimensionCount), stride: 1),
        Tensor<Int32>(Int32(axis)).rankLifted(),
        Tensor<Int32>(rangeFrom: Int32(batchDimensionCount), to: Int32(posAxis), stride: 1),
        Tensor<Int32>(rangeFrom: Int32(axis) + 1, to: Int32(rank), stride: 1)])
      let tensor = transposed(withPermutations: permutation)
      let result = tensor.batchGathering(
        atIndices: indices,
        alongAxis: batchDimensionCount,
        batchDimensionCount: batchDimensionCount)

      // Move the result dimensions corresponding to self[batchDimensionCount..<axis] to
      // just before the dimensions corresponding to indices[batchDimensionCount...].
      let start = indices.rank + posAxis - batchDimensionCount
      let resultPermutation = Tensor<Int32>(concatenating: [
        Tensor<Int32>(rangeFrom: 0, to: Int32(batchDimensionCount), stride: 1),
        Tensor<Int32>(rangeFrom: Int32(indices.rank), to: Int32(start), stride: 1),
        Tensor<Int32>(
          rangeFrom: Int32(batchDimensionCount),
          to: Int32(indices.rank),
          stride: 1),
        Tensor<Int32>(rangeFrom: Int32(start), to: Int32(result.rank), stride: 1)])
      return result.transposed(withPermutations: resultPermutation)
    }

    let batchIndices: Tensor<Index> = withoutDerivative(at: {
      var batchIndices = indices
      var accumulated = Tensor<Index>(ones: [])
      for d in (1...batchDimensionCount).reversed() {
        accumulated *= Tensor<Index>(self.shapeTensor[d])
        let dValue = self.shapeTensor[d - 1]
        let dIndices = Tensor<Index>(
          rangeFrom: Tensor<Index>(zeros: []),
          to: Tensor<Index>(dValue),
          stride: Tensor<Index>(ones: [])
        ) * accumulated
        let dShape = Tensor<Int32>(concatenating: [
          Tensor<Int32>([Int32](repeating: 1, count: d - 1)),
          dValue.rankLifted(),
          Tensor<Int32>([Int32](repeating: 1, count: indices.rank - d))])
        batchIndices += dIndices.reshaped(toShape: dShape)
      }
      return batchIndices
    }())

    let flatIndices = batchIndices.flattened()
    let outerShape = shapeTensor[(batchDimensionCount + 1)...]
    let innerShape = shapeTensor[..<(batchDimensionCount + 1)].product(squeezingAxes: [0])
    let flatTensor = reshaped(toShape: innerShape.rankLifted().concatenated(with: outerShape))
    let flatResult = flatTensor.gathering(atIndices: flatIndices)
    return flatResult.reshaped(toShape: indices.shapeTensor.concatenated(with: outerShape))
  }
}

/// Returns the softmax cross entropy (categorical cross entropy) between logits and labels.
///
/// - Parameters:
///   - logits: One-hot encoded outputs from a neural network.
///   - labels: Indices (zero-indexed) of the correct outputs.
@inlinable
@differentiable(wrt: logits, vjp: _vjpSoftmaxCrossEntropy)
internal func softmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    labels: Tensor<Int32>
) -> Tensor<Scalar> {
    Raw.sparseSoftmaxCrossEntropyWithLogits(features: logits, labels: labels).loss
}

@inlinable
internal func _vjpSoftmaxCrossEntropy<Scalar: TensorFlowFloatingPoint>(
    logits: Tensor<Scalar>,
    labels: Tensor<Int32>
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> Tensor<Scalar>) {
    let (loss, grad) = Raw.sparseSoftmaxCrossEntropyWithLogits(features: logits, labels: labels)
    return (loss, { $0.expandingShape(at: -1) * grad })
}

/// Returns the element-wise maximum of two tensors.
/// - Note: `max` supports broadcasting.
@inlinable
@differentiable(vjp: _vjpMax(_:_:) where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  Raw.maximum(lhs, rhs)
}

@inlinable
internal func _vjpMax<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
  let value = max(x, y)
  return (value, { v in _vjpMinMaxHelper(x, y, originalValue: value, seed: v) })
}

/// Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: rhs where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  max(Tensor(lhs), rhs)
}

/// Returns the element-wise maximum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: lhs where T: TensorFlowFloatingPoint)
public func max<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable {
  max(lhs, Tensor(rhs))
}

/// Returns the element-wise minimum of two tensors.
/// - Note: `min` supports broadcasting.
@inlinable
@differentiable(vjp: _vjpMin(_:_:) where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  Raw.minimum(lhs, rhs)
}

@inlinable
internal func _vjpMin<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>, Tensor<T>)) {
  let value = min(x, y)
  return (value, { v in _vjpMinMaxHelper(x, y, originalValue: value, seed: v) })
}

/// Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: rhs where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> where T: Numeric & Comparable {
  min(Tensor(lhs), rhs)
}

/// Returns the element-wise minimum of the scalar and the tensor, broadcasting the scalar.
@inlinable
@differentiable(wrt: lhs where T: TensorFlowFloatingPoint)
public func min<T>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> where T: Numeric & Comparable {
  min(lhs, Tensor(rhs))
}

@inlinable
internal func _vjpMinMaxHelper<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>,
  _ y: Tensor<T>,
  originalValue: Tensor<T>,
  seed: Tensor<T>
) -> (Tensor<T>, Tensor<T>) {
  let denominator = 1 + Tensor<T>(x .== y)
  let lhsGrad = seed * Tensor<T>(x .== originalValue) / denominator
  let rhsGrad = seed * Tensor<T>(y .== originalValue) / denominator
  let (lhsShape, rhsShape) = (x.shapeTensor, y.shapeTensor)
  let (lhsAxes, rhsAxes) = Raw.broadcastGradientArgs(s0: lhsShape, s1: rhsShape)
  return (lhsGrad.sum(squeezingAxes: lhsAxes).reshaped(toShape: lhsShape),
          rhsGrad.sum(squeezingAxes: rhsAxes).reshaped(toShape: rhsShape))
}

public extension Tensor where Scalar: TensorFlowNumeric {
  @inlinable
  @differentiable(vjp: _vjpClipping where Scalar: TensorFlowFloatingPoint)
  func clipping(min: Tensor, max: Tensor) -> Tensor {
    Raw.clipByValue(t: self, clipValueMin: min, clipValueMax: max)
  }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  func _vjpClipping(min: Tensor, max: Tensor) -> (Tensor, (Tensor) -> (Tensor, Tensor, Tensor)) {
    (clipping(min: min, max: max), { v in
      let selfShape = self.shapeTensor
      let minShape = min.shapeTensor
      let maxShape = max.shapeTensor
      let zeros = Tensor(zerosLike: v)
      let minMask = self .< min
      let maxMask = self .> max
      let selfGradient = v.replacing(with: zeros, where: minMask.elementsLogicalOr(maxMask))
      let minGradient = zeros.replacing(with: v, where: minMask)
      let maxGradient = zeros.replacing(with: v, where: maxMask)
      let (selfAxes, minAxes) = Raw.broadcastGradientArgs(s0: selfShape, s1: minShape)
      let (_, maxAxes) = Raw.broadcastGradientArgs(s0: selfShape, s1: maxShape)
      return (selfGradient.sum(squeezingAxes: selfAxes).reshaped(toShape: selfShape),
              minGradient.sum(squeezingAxes: minAxes).reshaped(toShape: minShape),
              maxGradient.sum(squeezingAxes: maxAxes).reshaped(toShape: maxShape))
    })
  }
}
