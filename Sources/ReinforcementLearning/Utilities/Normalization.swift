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

public protocol Normalizer {
  associatedtype Value

  func normalize(_ value: Value) -> Value
  mutating func update(using value: Value)
  mutating func reset()
}

public protocol TensorNormalizer: Normalizer where Value == Tensor<Scalar> {
  associatedtype Scalar: TensorFlowFloatingPoint
}

public struct BatchTensorNormalizer<Scalar: TensorFlowFloatingPoint>: TensorNormalizer {
  public let axes: Tensor<Int32>

  public init(alongAxes axes: Tensor<Int32>) {
    self.axes = axes
  }

  public init(alongAxes axes: [Int]) {
    self.init(alongAxes: Tensor<Int32>(axes.map(Int32.init)))
  }

  public init(alongAxes axes: Int...) {
    self.init(alongAxes: Tensor<Int32>(axes.map(Int32.init)))
  }

  public func normalize(_ value: Tensor<Scalar>) -> Tensor<Scalar> {
    let moments = value.moments(alongAxes: axes)
    return (value - moments.mean) / (sqrt(moments.variance) + Scalar(Float.ulpOfOne))
  }

  public mutating func update(using value: Tensor<Scalar>) {}
  public mutating func reset() {}
}

public struct StreamingTensorNormalizer<Scalar: TensorFlowFloatingPoint>: TensorNormalizer {
  public let axes: Tensor<Int32>

  private var count: Tensor<Scalar>
  private var valueSum: Tensor<Scalar>
  private var valueSquaredSum: Tensor<Scalar>

  public init(alongAxes axes: Tensor<Int32>) {
    self.axes = axes
    self.count = Tensor<Scalar>(Scalar(Float.ulpOfOne))
    self.valueSum = Tensor<Scalar>(zeros: [])
    self.valueSquaredSum = Tensor<Scalar>(zeros: [])
  }

  public init(alongAxes axes: [Int]) {
    self.init(alongAxes: Tensor<Int32>(axes.map(Int32.init)))
  }

  public init(alongAxes axes: Int...) {
    self.init(alongAxes: Tensor<Int32>(axes.map(Int32.init)))
  }

  public func normalize(_ value: Tensor<Scalar>) -> Tensor<Scalar> {
    let moments = momentsEstimate()
    return (value - moments.mean) / (sqrt(moments.variance) + Scalar(Float.ulpOfOne))
  }

  public func momentsEstimate() -> Moments<Scalar> {
    Moments(
      mean: valueSum / count,
      variance: (valueSquaredSum - valueSum.squared() / count) / count)
  }

  public mutating func update(using value: Tensor<Scalar>) {
    count += Tensor<Scalar>(value.shapeTensor.gathering(atIndices: axes).product())
    valueSum += value.sum(alongAxes: axes)
    valueSquaredSum += value.squared().sum(alongAxes: axes)
  }

  public mutating func reset() {
    count = Tensor<Scalar>(Scalar(Float.ulpOfOne))
    valueSum = Tensor<Scalar>(zeros: [])
    valueSquaredSum = Tensor<Scalar>(zeros: [])
  }
}
