import TensorFlow

public protocol Distribution {
  associatedtype Value

  func mode(seed: UInt64?) -> Value
  func sample(seed: UInt64?) -> Value
}
