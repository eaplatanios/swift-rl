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
    (cumulativeSum(alongAxis: axis, exclusive: exclusive, reverse: reverse), { seed in
      seed.cumulativeSum(alongAxis: axis, exclusive: exclusive, reverse: !reverse)
    })
  }
}
