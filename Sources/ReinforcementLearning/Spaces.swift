import TensorFlow

public protocol Space: CustomStringConvertible {
  associatedtype Value

  var shape: TensorShape { get }

  /// Uniformly randomly sample a random element of this space.
  func sample<G: RandomNumberGenerator>(generator: inout G) -> Value

  /// Returns a boolean specifying if `value` is a valid member of this space.
  func contains(_ value: Value) -> Bool
}

public struct Discrete: Space {
  public typealias ValueDistribution = Categorical

  public let size: Int32
  public let shape: TensorShape = []

  public init(withSize size: Int32) {
    self.size = size
  }

  public var description: String {
    return "Discrete(\(size))"
  }

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> Tensor<Int32> {
    return Tensor<Int32>(Int32.random(in: 0..<size, using: &generator))
  }

  public func contains(_ value: Tensor<Int32>) -> Bool {
    let scalar = value.scalar ?? -1
    return scalar >= 0 && scalar < size
  }
}

public struct MultiBinary: Space {
  public let size: Int
  public let shape: TensorShape

  public init(withSize size: Int) {
    self.size = size
    self.shape = [Int32(size)]
  }

  public var description: String {
    return "MultiBinary(\(size))"
  }

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> Tensor<Int32> {
    var scalars: [Int32] = []
    for _ in 0..<size {
      scalars.append(Int32.random(in: 0...1, using: &generator))
    }
    return Tensor<Int32>(shape: [Int32(size)], scalars: scalars)
  }

  public func contains(_ value: Tensor<Int32>) -> Bool {
    return value.shape == shape && value.scalars.allSatisfy{$0 == 0 || $0 == 1}
  }
}

public struct MultiDiscrete: Space {
  public typealias Scalar = Int32

  public let sizes: [Int32]
  public let shape: TensorShape

  public init(withSizes sizes: [Int32]) {
    self.sizes = sizes
    self.shape = [Int32(sizes.count)]
  }

  public var description: String {
    return "MultiDiscrete(\(sizes.map{String($0)}.joined(separator: ", ")))"
  }

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> Tensor<Int32> {
    var scalars: [Int32] = []
    for size in sizes {
      scalars.append(Int32.random(in: 0..<size, using: &generator))
    }
    return Tensor<Int32>(shape: shape, scalars: scalars)
  }

  public func contains(_ value: Tensor<Int32>) -> Bool {
    let scalars = value.scalars
    return scalars.allSatisfy{$0 >= 0} && zip(scalars, sizes).allSatisfy{$0 < $1}
  }
}

// TODO: How can we make this also support Scalar: BinaryFloatingPoint where 
// Scalar.RawSignificand: FixedWidthInteger?
public struct Box<Scalar: TensorFlowScalar & FixedWidthInteger>: Space {
  public let low: Tensor<Scalar>
  public let high: Tensor<Scalar>
  public let shape: TensorShape

  public init(low: Scalar, high: Scalar, shape: TensorShape) {
    let count = Int(shape.dimensions.reduce(1, *))
    self.low = Tensor<Scalar>(shape: shape, scalars: [Scalar](repeating: low, count: count))
    self.high = Tensor<Scalar>(shape: shape, scalars: [Scalar](repeating: high, count: count))
    self.shape = shape
  }

  public init(low: Tensor<Scalar>, high: Tensor<Scalar>) {
    precondition(low.shape == high.shape, "'low' and 'high' must have the same shape.")
    self.low = low
    self.high = high
    self.shape = low.shape
  }

  public var description: String {
    return "Box(\(shape.dimensions.map{String($0)}.joined(separator: ", ")))"
  }

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> Tensor<Scalar> {
    var scalars: [Scalar] = []
    for (low, high) in zip(self.low.scalars, self.high.scalars) {
      scalars.append(Scalar.random(in: low...high, using: &generator))
    }
    return Tensor<Scalar>(shape: shape, scalars: scalars)
  }

  public func contains(_ value: Tensor<Scalar>) -> Bool {
    let scalars = value.scalars
    return value.shape == shape &&
      zip(scalars, low.scalars).allSatisfy{$0 >= $1} &&
      zip(scalars, high.scalars).allSatisfy{$0 <= $1}
  }
}
