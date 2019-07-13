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

public struct Bernoulli<Scalar: TensorFlowInteger>: DifferentiableDistribution, KeyPathIterable {
  /// Unnormalized log-probabilities of this bernoulli distribution.
  public var logits: Tensor<Float>

  @inlinable
  @differentiable(wrt: logits)
  public init(logits: Tensor<Float>) {
    self.logits = logits
  }

  @inlinable
  @differentiable(wrt: logProbabilities)
  public init(logProbabilities: Tensor<Float>) {
    self.logits = logProbabilities
  }

  @inlinable
  @differentiable(wrt: probabilities)
  public init(probabilities: Tensor<Float>) {
    self.logits = log(probabilities)
  }

  @inlinable
  @differentiable(wrt: self)
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    max(logits, Tensor<Float>(0.0)) - logits * Tensor<Float>(value) + softplus(-abs(logits))
  }

  @inlinable
  @differentiable(wrt: self)
  public func entropy() -> Tensor<Float> {
    max(logits, Tensor<Float>(0.0)) - logits * sigmoid(logits) + softplus(-abs(logits))
  }

  @inlinable
  public func mode() -> Tensor<Scalar> {
    Tensor<Scalar>(logSigmoid(logits) .> log(0.5))
  }

  @inlinable
  public func sample() -> Tensor<Scalar> {
    let seed = Context.local.randomSeed
    let logProbabilities = logSigmoid(logits)
    let uniform: Tensor<Float> = Raw.randomUniform(
      shape: logProbabilities.shapeTensor,
      seed: Int64(seed.graph),
      seed2: Int64(seed.op))
    return Tensor<Scalar>(logProbabilities .< log(uniform))
  }
}