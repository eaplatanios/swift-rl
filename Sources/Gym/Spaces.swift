import TensorFlow

public protocol Space: CustomStringConvertible {
  associatedtype Value
  associatedtype ValueDistribution: Distribution where ValueDistribution.Value == Value

  var shape: TensorShape { get }
  var distribution: ValueDistribution { get }

  /// Returns a boolean specifying if `value` is a valid member of this space.
  func contains(_ value: Value) -> Bool
}

public extension Space {
  /// Sample a random element from this space.
  func sample(seed: UInt64?) -> Value {
    return distribution.sample(seed: seed)
  }
}

public struct Discrete: Space {
  public let size: Int
  public let shape: TensorShape = []

  public let distribution: Categorical<Int32>

  public init(withSize size: Int) {
    self.size = size
    self.distribution = Categorical<Int32>(logits: Tensor<Float>(ones: [1, size]))
  }

  public var description: String {
    return "Discrete(\(size))"
  }

  public func contains(_ value: Tensor<Int32>) -> Bool {
    let scalar = value.scalar ?? -1
    return scalar >= 0 && scalar < size
  }
}

public struct MultiBinary: Space {
  public let size: Int
  public let shape: TensorShape
  public let distribution: ValueDistribution

  public init(withSize size: Int) {
    self.size = size
    self.shape = [size]
    self.distribution = ValueDistribution(size: size)
  }

  public var description: String {
    return "MultiBinary(\(size))"
  }

  public func contains(_ value: Tensor<Int32>) -> Bool {
    return value.shape == shape && value.scalars.allSatisfy{$0 == 0 || $0 == 1}
  }

  public struct ValueDistribution: Distribution {
    private let size: Int
    private let distribution: Categorical<Int32>

    public init(size: Int) {
      self.size = size
      self.distribution = Categorical<Int32>(logits: Tensor<Float>(ones: [1, 2]))
    }

    public func mode(seed: UInt64? = nil) -> Tensor<Int32> {
      let modes = (0..<size).map{ _ in distribution.mode(seed: seed) }
      return Tensor<Int32>(concatenating: modes)
    }

    public func sample(seed: UInt64? = nil) -> Tensor<Int32> {
      let samples = (0..<size).map{ _ in distribution.sample(seed: seed) }
      return Tensor<Int32>(concatenating: samples)
    }
  }
}

public struct MultiDiscrete: Space {
  public typealias Scalar = Int32

  public let sizes: [Int]
  public let shape: TensorShape
  public let distribution: ValueDistribution

  public init(withSizes sizes: [Int]) {
    self.sizes = sizes
    self.shape = [sizes.count]
    self.distribution = ValueDistribution(sizes: sizes)
  }

  public var description: String {
    return "MultiDiscrete(\(sizes.map{String($0)}.joined(separator: ", ")))"
  }

  public func contains(_ value: Tensor<Int32>) -> Bool {
    let scalars = value.scalars
    return scalars.allSatisfy{$0 >= 0} && zip(scalars, sizes).allSatisfy{$0 < $1}
  }

  public struct ValueDistribution: Distribution {
    private let sizes: [Int]
    private let distributions: [Categorical<Int32>]

    public init(sizes: [Int]) {
      self.sizes = sizes
      self.distributions = sizes.map {
        Categorical<Int32>(logits: Tensor<Float>(ones: [1, $0]))
      }
    }

    public func mode(seed: UInt64? = nil) -> Tensor<Int32> {
      let modes = distributions.map { $0.mode(seed: seed) }
      return Tensor<Int32>(concatenating: modes)
    }

    public func sample(seed: UInt64? = nil) -> Tensor<Int32> {
      let samples = distributions.map { $0.sample(seed: seed) }
      return Tensor<Int32>(concatenating: samples)
    }
  }
}

public struct DiscreteBox<Scalar: TensorFlowInteger>: Space {
  public let shape: TensorShape
  public let lowerBound: Tensor<Scalar>
  public let upperBound: Tensor<Scalar>

  public let distribution: ValueDistribution

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

  public var description: String {
    return "DiscreteBox(\(shape.dimensions.map{String($0)}.joined(separator: ", ")))"
  }

  public func contains(_ value: Tensor<Scalar>) -> Bool {
    let scalars = value.scalars
    return value.shape == shape &&
      zip(scalars, lowerBound.scalars).allSatisfy{$0 >= $1} &&
      zip(scalars, upperBound.scalars).allSatisfy{$0 <= $1}
  }

  public struct ValueDistribution: Distribution {
    private let distribution: Uniform<Float>

    public init(distribution: Uniform<Float>) {
      self.distribution = distribution
    }

    public func mode(seed: UInt64? = nil) -> Tensor<Scalar> {
      return Tensor<Scalar>(distribution.mode(seed: seed))
    }

    public func sample(seed: UInt64? = nil) -> Tensor<Scalar> {
      return Tensor<Scalar>(distribution.sample(seed: seed))
    }
  }
}

public struct Box<Scalar: TensorFlowFloatingPoint>: Space {
  public let shape: TensorShape
  public let lowerBound: Tensor<Scalar>
  public let upperBound: Tensor<Scalar>

  public let distribution: Uniform<Scalar>

  public init(shape: TensorShape, lowerBound: Scalar, upperBound: Scalar) {
    self.shape = shape
    self.lowerBound = Tensor<Scalar>(lowerBound)
    self.upperBound = Tensor<Scalar>(upperBound)
    self.distribution = Uniform<Scalar>(
      shape: Tensor<Int32>(shape.dimensions.map(Int32.init)),
      lowerBound: self.lowerBound,
      upperBound: self.upperBound)
  }

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

  public var description: String {
    return "Box(\(shape.dimensions.map{String($0)}.joined(separator: ", ")))"
  }

  public func contains(_ value: Tensor<Scalar>) -> Bool {
    let scalars = value.scalars
    return value.shape == shape &&
      zip(scalars, lowerBound.scalars).allSatisfy{$0 >= $1} &&
      zip(scalars, upperBound.scalars).allSatisfy{$0 <= $1}
  }
}
