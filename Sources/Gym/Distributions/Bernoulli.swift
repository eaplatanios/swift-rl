import TensorFlow

public struct Bernoulli<Scalar: TensorFlowInteger>: DifferentiableDistribution {
  /// Unnormalized log-probabilities of this bernoulli distribution.
  public let logits: Tensor<Float>

  @inlinable
  // TODO: @differentiable(wrt: logits)
  public init(logits: Tensor<Float>) {
    self.logits = logits
  }

  @inlinable
  // TODO: @differentiable(wrt: logProbabilities)
  public init(logProbabilities: Tensor<Float>) {
    self.logits = logProbabilities
  }

  @inlinable
  // TODO: @differentiable(wrt: probabilities)
  public init(probabilities: Tensor<Float>) {
    self.logits = log(probabilities)
  }

  @inlinable
  @differentiable(wrt: self)
  public func probability(of value: Tensor<Scalar>) -> Tensor<Float> {
    return exp(logProbability(of: value))
  }

  @inlinable
  @differentiable(wrt: self)
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    return max(logits, Tensor<Float>(0.0)) - logits * Tensor<Float>(value) + softplus(-abs(logits))
  }

  @inlinable
  @differentiable(wrt: self)
  public func entropy() -> Tensor<Float> {
    return max(logits, Tensor<Float>(0.0)) - logits * sigmoid(logits) + softplus(-abs(logits))
  }

  @inlinable
  public func mode(seed: UInt64?) -> Tensor<Scalar> {
    return Tensor<Scalar>(logSigmoid(logits) .> log(0.5))
  }

  @inlinable
  public func sample(seed: UInt64? = nil) -> Tensor<Scalar> {
    let logProbabilities = logSigmoid(logits)
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    let uniform: Tensor<Float> = Raw.randomUniform(
      shape: logProbabilities.shapeTensor,
      seed: tfSeed.graph,
      seed2: tfSeed.op)
    return Tensor<Scalar>(logProbabilities .< log(uniform))
  }
}
