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

public struct TensorNormalizer<Scalar: TensorFlowFloatingPoint>: Normalizer {
  public let axes: Tensor<Int32>
  public let streaming: Bool

  private var count: Tensor<Scalar>?
  private var valueSum: Tensor<Scalar>?
  private var valueSquaredSum: Tensor<Scalar>?

  public init(streaming: Bool, alongAxes axes: Tensor<Int32>) {
    self.axes = axes
    self.streaming = streaming
    self.count = streaming ? Tensor<Scalar>(Scalar(Float.ulpOfOne)) : nil
    self.valueSum = streaming ? Tensor<Scalar>(zeros: []) : nil
    self.valueSquaredSum = streaming ? Tensor<Scalar>(zeros: []) : nil
  }

  public init(streaming: Bool, alongAxes axes: [Int]) {
    self.init(streaming: streaming, alongAxes: Tensor<Int32>(axes.map(Int32.init)))
  }

  public init(streaming: Bool, alongAxes axes: Int...) {
    self.init(streaming: streaming, alongAxes: Tensor<Int32>(axes.map(Int32.init)))
  }

  public func normalize(_ value: Tensor<Scalar>) -> Tensor<Scalar> {
    if streaming {
      let mean = valueSum! / count!
      let variance = (valueSquaredSum! - valueSum!.squared() / count!) / count!
      return (value - mean) / (sqrt(variance) + Scalar(Float.ulpOfOne))
    }
    let moments = value.moments(alongAxes: axes)
    return (value - moments.mean) / (sqrt(moments.variance) + Scalar(Float.ulpOfOne))
  }

  public mutating func update(using value: Tensor<Scalar>) {
    if streaming {
      count = count! + Tensor<Scalar>(value.shapeTensor.gathering(atIndices: axes).product())
      valueSum = valueSum! + value.sum(alongAxes: axes)
      valueSquaredSum = valueSquaredSum! + value.squared().sum(alongAxes: axes)
    }
  }

  public mutating func reset() {
    if streaming {
      count = Tensor<Scalar>(Scalar(Float.ulpOfOne))
      valueSum = Tensor<Scalar>(zeros: [])
      valueSquaredSum = Tensor<Scalar>(zeros: [])
    }
  }
}
