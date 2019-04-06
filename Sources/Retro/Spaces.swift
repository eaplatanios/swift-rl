
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

import TensorFlow

public protocol Space {
  public associatedtype Scalar: TensorFlowScalar

  public var shape: TensorShape { get }

  /// Uniformly randomly sample a random element of this space.
  public func sample() -> Tensor<Scalar>

  /// Returns a boolean specifying if `value` is a valid member of this space.
  public func contains(_ value: Tensor<Scalar>) -> Bool
}

public extension Space {
  public private(set) var randomNumberGenerator = PhiloxRandomNumberGenerator(
    uint64Seed: UInt64(time(nil)))

  /// Seeds the pseudorandom number generator of this space.
  public func seed(_ seed: [UInt8]) {
    randomNumberGenerator = PhiloxRandomNumberGenerator(seed)
  }

  /// Seeds the pseudorandom number generator of this space.
  public func seed<T: BinaryInteger>(_ seed: T) {
    randomNumberGenerator = PhiloxRandomNumberGenerator(seed)
  }
}

public struct Discrete: Space<Int32> {
  public let size: Int32

  public let shape: TensorShape = TensorShape([])

  public func sample() -> Tensor<Int32> {
    // TODO
    return Tensor<Int32>(0)
  }

  public func contains(_ value: Tensor<Int32>) -> Bool {
    let scalar = value.scalar
    return scalar >= 0 && scalar < size
  }
}

public struct MultiBinary: Space<Int32> {
  public let size: Int32

  // TODO
}

public struct MultiDiscrete: Space<Int32> {
  public let sizes: [Int32]

  // TODO
}

public struct Box<Scalar: TensorFlowScalar>: Space {
  public let low: Scalar
  public let high: Scalar
  public let shape: TensorShape
}
