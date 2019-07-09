import TensorFlow

public protocol Distribution {
  associatedtype Value

  func logProbability(of value: Value) -> Tensor<Float>
  func entropy() -> Tensor<Float>

  /// Returns the mode of this distribution. If the distribution has multiple modes, then one of
  /// them is sampled randomly (and uniformly) and returned.
  func mode(usingSeed seed: TensorFlowSeed) -> Value

  /// Returns a random sample drawn from this distribution.
  func sample(usingSeed seed: TensorFlowSeed) -> Value
}

public extension Distribution {
  func probability(of value: Value) -> Tensor<Float> {
    exp(logProbability(of: value))
  }
}

public protocol DifferentiableDistribution: Distribution, Differentiable {
  @differentiable(wrt: self)
  func logProbability(of value: Value) -> Tensor<Float>

  @differentiable(wrt: self)
  func entropy() -> Tensor<Float>
}

public extension DifferentiableDistribution {
  @differentiable(wrt: self)
  func probability(of value: Value) -> Tensor<Float> {
    exp(logProbability(of: value))
  }
}
