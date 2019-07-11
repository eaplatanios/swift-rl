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

public struct Uniform<
  Scalar: TensorFlowFloatingPoint
>: DifferentiableDistribution, KeyPathIterable {
  @noDerivative public let shape: Tensor<Int32>
  public var lowerBound: Tensor<Scalar>
  public var upperBound: Tensor<Scalar>

  @inlinable
  @differentiable(wrt: (lowerBound, upperBound))
  public init(
    shape: Tensor<Int32>,
    lowerBound: Tensor<Scalar> = Tensor<Scalar>(zeros: []),
    upperBound: Tensor<Scalar> = Tensor<Scalar>(ones: [])
  ) {
    self.shape = shape
    self.lowerBound = lowerBound
    self.upperBound = upperBound
  }

  @inlinable
  @differentiable(wrt: self)
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    log(1.0) - log(Tensor<Float>(upperBound - lowerBound))
  }

  @inlinable
  @differentiable(wrt: self)
  public func entropy() -> Tensor<Float> {
    log(Tensor<Float>(upperBound - lowerBound))
  }

  @inlinable
  public func mode() -> Tensor<Scalar> {
    sample()
  }

  @inlinable
  public func sample() -> Tensor<Scalar> {
    // TODO: Make `Tensor.init(randomUniform:...)` accept `Tensor<Int32>` for the shape.
    let seed = Context.local.randomSeed
    let sample: Tensor<Scalar> = Raw.randomUniform(
      shape: shape, seed: Int64(seed.graph), seed2: Int64(seed.op))
    return sample * (upperBound - lowerBound) + lowerBound
  }
}

extension Uniform {
  @inlinable
  @differentiable(wrt: (lowerBound, upperBound))
  public init(lowerBound: Tensor<Scalar>, upperBound: Tensor<Scalar>) {
    self.shape = withoutDerivative(at: lowerBound.shapeTensor)
    self.lowerBound = lowerBound
    self.upperBound = upperBound
  }
}
