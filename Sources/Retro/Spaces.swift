
#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

import TensorFlow

public protocol Space {
  associatedtype Scalar: TensorFlowScalar

  var shape: [Int] { get }

  /// Uniformly randomly sample a random element of this space.
  func sample<G: RandomNumberGenerator>(generator: inout G) -> Tensor<Scalar>

  /// Returns a boolean specifying if `value` is a valid member of this space.
  func contains(_ value: Tensor<Scalar>) -> Bool
}

public struct Discrete: Space {
  public typealias Scalar = Int32
  
  public let size: Int32
  public let shape: [Int] = []

  public init(withSize size: Int32) {
    self.size = size
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
  public typealias Scalar = Int32
  
  public let size: Int32
  public let shape: [Int]

  public init(withSize size: Int32) {
    self.size = size
    self.shape = [Int(size)]
  }

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> Tensor<Int32> {
    // TODO
    fatalError("Not implemented.")
  }

  public func contains(_ value: Tensor<Int32>) -> Bool {
    // TODO
    fatalError("Not implemented.")
  }
}

public struct MultiDiscrete: Space {
  public typealias Scalar = Int32

  public let sizes: [Int32]
  public let shape: [Int]

  public init(withSizes sizes: [Int32]) {
    self.sizes = sizes
    self.shape = sizes.map { Int($0) }
  }

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> Tensor<Int32> {
    // TODO
    fatalError("Not implemented.")
  }

  public func contains(_ value: Tensor<Int32>) -> Bool {
    // TODO
    fatalError("Not implemented.")
  }
}

public struct Box<Scalar: TensorFlowScalar>: Space {
  public let low: Scalar
  public let high: Scalar
  public let shape: [Int]

  public func sample<G: RandomNumberGenerator>(generator: inout G) -> Tensor<Scalar> {
    // TODO
    fatalError("Not implemented.")
  }

  public func contains(_ value: Tensor<Scalar>) -> Bool {
    // TODO
    fatalError("Not implemented.")
  }
}
