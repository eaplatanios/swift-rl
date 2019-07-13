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
