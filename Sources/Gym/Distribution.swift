import TensorFlow

public protocol Distribution {
  associatedtype Value

  /// Returns the mode of this distribution. If the distribution has multiple modes, then one of
  /// them is sampled randomly (and uniformly) and returned.
  func mode(seed: UInt64?) -> Value

  /// Returns a random sample drawn from this distribution.
  func sample(seed: UInt64?) -> Value
}
