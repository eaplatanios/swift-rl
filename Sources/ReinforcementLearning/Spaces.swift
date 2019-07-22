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

public protocol Space: CustomStringConvertible {
  associatedtype Value
  associatedtype ValueDistribution: Distribution where ValueDistribution.Value == Value

  var distribution: ValueDistribution { get }

  /// Returns a boolean specifying if `value` is a valid member of this space.
  func contains(_ value: Value) -> Bool
}

public extension Space {
  /// Sample a random element from this space.
  @inlinable
  func sample() -> Value {
    distribution.sample()
  }
}

public struct Discrete: Space {
  public let size: Int
  public let distribution: Categorical<Int32>

  @inlinable
  public init(withSize size: Int, batchSize: Int) {
    self.size = size
    self.distribution = Categorical<Int32>(logits: Tensor<Float>(ones: [batchSize, size]))
  }

  @inlinable
  public var description: String {
    "Discrete(\(size))"
  }

  @inlinable
  public func contains(_ value: Tensor<Int32>) -> Bool {
    value.rank < 2 && value >= 0 && value < Int32(size)
  }
}

public struct MultiBinary: Space {
  public let size: Int
  public let shape: TensorShape
  public let distribution: Bernoulli<Int32>

  @inlinable
  public init(withSize size: Int, batchSize: Int) {
    self.size = size
    self.shape = [size]
    self.distribution = Bernoulli<Int32>(logits: Tensor<Float>(ones: [batchSize, size]))
  }

  @inlinable
  public var description: String {
    "MultiBinary(\(size))"
  }

  @inlinable
  public func contains(_ value: Tensor<Int32>) -> Bool {
    value.shape == shape && value.scalars.allSatisfy{$0 == 0 || $0 == 1}
  }
}

public struct MultiDiscrete: Space {
  public let sizes: [Int]
  public let shape: TensorShape
  public let distribution: ValueDistribution

  @inlinable
  public init(withSizes sizes: [Int], batchSize: Int) {
    self.sizes = sizes
    self.shape = [sizes.count]
    self.distribution = ValueDistribution(sizes: sizes, batchSize: batchSize)
  }

  @inlinable
  public var description: String {
    "MultiDiscrete(\(sizes.map{String($0)}.joined(separator: ", ")))"
  }

  @inlinable
  public func contains(_ value: Tensor<Int32>) -> Bool {
    let scalars = value.scalars
    return scalars.allSatisfy { $0 >= 0 } && zip(scalars, sizes).allSatisfy { $0 < $1 }
  }

  public struct ValueDistribution: DifferentiableDistribution {
    @noDerivative @usableFromInline internal let sizes: [Int]
    @usableFromInline internal var distributions: [Categorical<Int32>]

    @inlinable
    public init(sizes: [Int], batchSize: Int) {
      self.sizes = sizes
      self.distributions = sizes.map {
        Categorical<Int32>(logits: Tensor<Float>(ones: [batchSize, $0]))
      }
    }

    @inlinable
    @differentiable(wrt: self)
    public func logProbability(of value: Tensor<Int32>) -> Tensor<Float> {
      let values = value.unstacked()
      var logProbability = Tensor<Float>(0.0)
      for i in 0..<sizes.count {
        logProbability = logProbability + distributions[i].logProbability(of: values[i])
      }
      return logProbability
    }

    @inlinable
    @differentiable(wrt: self)
    public func entropy() -> Tensor<Float> {
      var entropy = Tensor<Float>(0.0)
      for i in 0..<sizes.count {
        entropy = entropy + distributions[i].entropy()
      }
      return entropy
    }

    @inlinable
    public func mode() -> Tensor<Int32> {
      Tensor<Int32>(concatenating: distributions.map { $0.mode() })
    }

    @inlinable
    public func sample() -> Tensor<Int32> {
      Tensor<Int32>(concatenating: distributions.map { $0.sample() })
    }
  }
}

public struct DiscreteBox<Scalar: TensorFlowInteger>: Space {
  public let shape: TensorShape
  public let lowerBound: Tensor<Scalar>
  public let upperBound: Tensor<Scalar>

  public let distribution: ValueDistribution

  @inlinable
  public init(shape: TensorShape, lowerBound: Scalar, upperBound: Scalar) {
    self.shape = shape
    self.lowerBound = Tensor<Scalar>(lowerBound)
    self.upperBound = Tensor<Scalar>(upperBound)
    self.distribution = ValueDistribution(
      distribution: Uniform<Float>(
        shape: Tensor<Int32>(shape.dimensions.map(Int32.init)),
        lowerBound: Tensor<Float>(self.lowerBound),
        upperBound: Tensor<Float>(self.upperBound)))
  }

  @inlinable
  public init(lowerBound: Tensor<Scalar>, upperBound: Tensor<Scalar>) {
    precondition(lowerBound.shape == upperBound.shape,
      "'lowerBound' and 'upperBound' must have the same shape.")
    self.shape = lowerBound.shape
    self.lowerBound = lowerBound
    self.upperBound = upperBound
    self.distribution = ValueDistribution(
      distribution: Uniform<Float>(
        shape: Tensor<Int32>(shape.dimensions.map(Int32.init)),
        lowerBound: Tensor<Float>(self.lowerBound),
        upperBound: Tensor<Float>(self.upperBound)))
  }

  @inlinable
  public var description: String {
    "DiscreteBox(\(shape.dimensions.map{String($0)}.joined(separator: ", ")))"
  }

  @inlinable
  public func contains(_ value: Tensor<Scalar>) -> Bool {
    let scalars = value.scalars
    return value.shape == shape &&
      zip(scalars, lowerBound.scalars).allSatisfy{$0 >= $1} &&
      zip(scalars, upperBound.scalars).allSatisfy{$0 <= $1}
  }

  public struct ValueDistribution: DifferentiableDistribution {
    @usableFromInline internal var distribution: Uniform<Float>

    @inlinable
    public init(distribution: Uniform<Float>) {
      self.distribution = distribution
    }

    @inlinable
    @differentiable(wrt: self)
    public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
      distribution.logProbability(of: Tensor<Float>(value))
    }

    @inlinable
    @differentiable(wrt: self)
    public func entropy() -> Tensor<Float> {
      distribution.entropy()
    }

    @inlinable
    public func mode() -> Tensor<Scalar> {
      Tensor<Scalar>(distribution.mode())
    }

    @inlinable
    public func sample() -> Tensor<Scalar> {
      Tensor<Scalar>(distribution.sample())
    }
  }
}

public struct Box<Scalar: TensorFlowFloatingPoint>: Space {
  public let shape: TensorShape
  public let lowerBound: Tensor<Scalar>
  public let upperBound: Tensor<Scalar>

  public let distribution: Uniform<Scalar>

  @inlinable
  public init(shape: TensorShape, lowerBound: Scalar, upperBound: Scalar) {
    self.shape = shape
    self.lowerBound = Tensor<Scalar>(lowerBound)
    self.upperBound = Tensor<Scalar>(upperBound)
    self.distribution = Uniform<Scalar>(
      shape: Tensor<Int32>(shape.dimensions.map(Int32.init)),
      lowerBound: self.lowerBound,
      upperBound: self.upperBound)
  }

  @inlinable
  public init(lowerBound: Tensor<Scalar>, upperBound: Tensor<Scalar>) {
    precondition(lowerBound.shape == upperBound.shape,
                 "'lowerBound' and 'upperBound' must have the same shape.")
    self.shape = lowerBound.shape
    self.lowerBound = lowerBound
    self.upperBound = upperBound
    self.distribution = Uniform<Scalar>(
      shape: Tensor<Int32>(shape.dimensions.map(Int32.init)),
      lowerBound: self.lowerBound,
      upperBound: self.upperBound)
  }

  @inlinable
  public var description: String {
    "Box(\(shape.dimensions.map{String($0)}.joined(separator: ", ")))"
  }

  @inlinable
  public func contains(_ value: Tensor<Scalar>) -> Bool {
    let scalars = value.scalars
    return value.shape == shape &&
      zip(scalars, lowerBound.scalars).allSatisfy{$0 >= $1} &&
      zip(scalars, upperBound.scalars).allSatisfy{$0 <= $1}
  }
}
