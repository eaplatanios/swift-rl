import TensorFlow

public protocol Distribution {
  associatedtype Value

  func probability(of value: Value) -> Tensor<Float>
  func logProbability(of value: Value) -> Tensor<Float>

  func entropy() -> Tensor<Float>

  /// Returns the mode of this distribution. If the distribution has multiple modes, then one of
  /// them is sampled randomly (and uniformly) and returned.
  func mode(seed: UInt64?) -> Value

  /// Returns a random sample drawn from this distribution.
  func sample(seed: UInt64?) -> Value
}

public protocol DifferentiableDistribution: Distribution, Differentiable {
  @differentiable(wrt: self)
  override func probability(of value: Value) -> Tensor<Float>

  @differentiable(wrt: self)
  override func logProbability(of value: Value) -> Tensor<Float>

  @differentiable(wrt: self)
  override func entropy() -> Tensor<Float>
}
