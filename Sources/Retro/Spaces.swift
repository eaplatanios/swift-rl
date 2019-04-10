import TensorFlow

public protocol Space: Hashable, Codable, CustomStringConvertible {
  associatedtype Scalar: TensorFlowScalar

  var shape: [Int] { get }

  /// Uniformly randomly sample a random element of this space.
  func sample<G: RandomNumberGenerator>(generator: inout G) -> ShapedArray<Scalar>

  /// Returns a boolean specifying if `value` is a valid member of this space.
  func contains(_ value: ShapedArray<Scalar>) -> Bool
}

public struct Discrete: Space {
  public typealias Scalar = Int32
  
  public let size: Int32
  public let shape: [Int] = []

  public init(withSize size: Int32) {
    self.size = size
  }

  public var description: String {
    return "Discrete(\(size))"
  }

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> ShapedArray<Int32> {
    return ShapedArray(Int32.random(in: 0..<size, using: &generator))
  }

  public func contains(_ value: ShapedArray<Int32>) -> Bool {
    let scalar = value.scalar ?? -1
    return scalar >= 0 && scalar < size
  }
}

public struct MultiBinary: Space {
  public typealias Scalar = Int32
  
  public let size: Int
  public let shape: [Int]

  public init(withSize size: Int) {
    self.size = size
    self.shape = [size]
  }

  public var description: String {
    return "MultiBinary(\(size))"
  }

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> ShapedArray<Int32> {
    var scalars: [Scalar] = []
    for _ in 0..<size {
      scalars.append(Int32.random(in: 0...1, using: &generator))
    }
    return ShapedArray(shape: [size], scalars: scalars)
  }

  public func contains(_ value: ShapedArray<Int32>) -> Bool {
    return value.shape == shape && value.scalars.allSatisfy{$0 == 0 || $0 == 1}
  }
}

public struct MultiDiscrete: Space {
  public typealias Scalar = Int32

  public let sizes: [Int32]
  public let shape: [Int]

  public init(withSizes sizes: [Int32]) {
    self.sizes = sizes
    self.shape = [sizes.count]
  }

  public var description: String {
    return "MultiDiscrete(\(sizes.map{String($0)}.joined(separator: ", ")))"
  }

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> ShapedArray<Int32> {
    var scalars: [Scalar] = []
    for size in sizes {
      scalars.append(Int32.random(in: 0..<size, using: &generator))
    }
    return ShapedArray(shape: shape, scalars: scalars)
  }

  public func contains(_ value: ShapedArray<Int32>) -> Bool {
    return value.scalars.allSatisfy{$0 >= 0} && zip(value.scalars, sizes).allSatisfy{$0 < $1}
  }
}

// TODO: How can we make this also support Scalar: BinaryFloatingPoint where 
// Scalar.RawSignificand: FixedWidthInteger?
public struct Box<Scalar: TensorFlowScalar & FixedWidthInteger & Codable>: Space {
  public let low: ShapedArray<Scalar>
  public let high: ShapedArray<Scalar>
  public let shape: [Int]

  public init(low: Scalar, high: Scalar, shape: [Int]) {
    let count = shape.reduce(1, *)
    self.low = ShapedArray(shape: shape, scalars: [Scalar](repeating: low, count: count))
    self.high = ShapedArray(shape: shape, scalars: [Scalar](repeating: high, count: count))
    self.shape = shape
  }

  public init(low: ShapedArray<Scalar>, high: ShapedArray<Scalar>) {
    precondition(low.shape == high.shape, "'low' and 'high' must have the same shape.")
    self.low = low
    self.high = high
    self.shape = low.shape
  }

  public var description: String {
    return "Box(\(shape.map{String($0)}.joined(separator: ", ")))"
  }

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> ShapedArray<Scalar> {
    var scalars: [Scalar] = []
    for (low, high) in zip(self.low.scalars, self.high.scalars) {
      scalars.append(Scalar.random(in: low...high, using: &generator))
    }
    return ShapedArray(shape: shape, scalars: scalars)
  }

  public func contains(_ value: ShapedArray<Scalar>) -> Bool {
    return value.shape == shape &&
      zip(value.scalars, low.scalars).allSatisfy{$0 >= $1} &&
      zip(value.scalars, high.scalars).allSatisfy{$0 <= $1}
  }
}
