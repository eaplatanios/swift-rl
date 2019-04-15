import TensorFlow

public protocol Distribution {
  associatedtype Value

  func sample(seed: UInt64?) -> Value
}
