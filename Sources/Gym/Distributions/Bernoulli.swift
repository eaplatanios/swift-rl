import TensorFlow

public struct Bernoulli<Scalar: TensorFlowInteger>: Distribution, Differentiable {
  /// Unnormalized log-probabilities of this bernoulli distribution.
  public let logProbabilities: Tensor<Float>

  @inlinable
  // TODO: @differentiable
  public init(logProbabilities: Tensor<Float>) {
    self.logProbabilities = logProbabilities
  }

  @inlinable
  // TODO: @differentiable
  public init(probabilities: Tensor<Float>) {
    self.logProbabilities = log(probabilities)
  }

  @inlinable
  // TODO: @differentiable
  public init(logits: Tensor<Float>) {
    self.logProbabilities = logSigmoid(logits)
  }

  @inlinable
  @differentiable(wrt: self)
  public func probability(of value: Tensor<Scalar>) -> Tensor<Float> {
    return exp(logProbability(of: value))
  }

  @inlinable
  @differentiable(wrt: self)
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    let zeros = Tensor<Float>(value)
    let ones = 1.0 - zeros
    return zeros * log(1 - exp(logProbabilities)) + ones * logProbabilities
  }

  @inlinable
  public func mode(seed: UInt64?) -> Tensor<Scalar> {
    return Tensor<Scalar>(logProbabilities .> log(0.5))
  }

  @inlinable
  public func sample(seed: UInt64? = nil) -> Tensor<Scalar> {
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    let uniform: Tensor<Float> = Raw.randomUniform(
      shape: logProbabilities.shapeTensor,
      seed: tfSeed.graph,
      seed2: tfSeed.op)
    return Tensor<Scalar>(logProbabilities .< log(uniform))
  }
}
