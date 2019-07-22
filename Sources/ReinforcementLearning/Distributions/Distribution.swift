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

public protocol Distribution {
  associatedtype Value

  func logProbability(of value: Value) -> Tensor<Float>
  func entropy() -> Tensor<Float>

  /// Returns the mode of this distribution. If the distribution has multiple modes, then one of
  /// them is sampled randomly (and uniformly) and returned.
  func mode() -> Value

  /// Returns a random sample drawn from this distribution.
  func sample() -> Value
}

public extension Distribution {
  func probability(of value: Value) -> Tensor<Float> {
    exp(logProbability(of: value))
  }
}

public protocol DifferentiableDistribution: Distribution, Differentiable {
  @differentiable(wrt: self)
  func logProbability(of value: Value) -> Tensor<Float>

  @differentiable(wrt: self)
  func entropy() -> Tensor<Float>
}

extension DifferentiableDistribution {
  @inlinable
  @differentiable(wrt: self)
  public func probability(of value: Value) -> Tensor<Float> {
    exp(logProbability(of: value))
  }
}

// TODO: It would be great to support KL divergence between different distributions, but that
// would require multiple conformances to the same protocol with different `TargetDistribution`
// types, which is not currently supported in Swift. :( This is also a place where a feature
// similar to Scala implicits would be great.
public protocol KLDivergence where Self: Distribution {
  func klDivergence(to target: Self) -> Tensor<Float>
}

public protocol DifferentiableKLDivergence: KLDivergence where Self: DifferentiableDistribution {
  @differentiable
  func klDivergence(to target: Self) -> Tensor<Float>
}
