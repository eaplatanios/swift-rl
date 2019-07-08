import TensorFlow

public struct Categorical<Scalar: TensorFlowIndex>: DifferentiableDistribution {
  /// Log-probabilities of this categorical distribution.
  public var logProbabilities: Tensor<Float>

  @inlinable
  @differentiable(wrt: logProbabilities)
  public init(logProbabilities: Tensor<Float>) {
    self.logProbabilities = logProbabilities
  }

  @inlinable
  @differentiable(wrt: probabilities)
  public init(probabilities: Tensor<Float>) {
    self.logProbabilities = log(probabilities)
  }

  @inlinable
  @differentiable(wrt: logits)
  public init(logits: Tensor<Float>) {
    self.logProbabilities = logSoftmax(logits)
  }

  @inlinable
  @differentiable(wrt: self)
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    softmaxCrossEntropy(logits: logProbabilities, labels: Tensor<Int32>(value))
  }

  @inlinable
  @differentiable(wrt: self)
  public func entropy() -> Tensor<Float> {
    -(logProbabilities * exp(logProbabilities)).sum(squeezingAxes: -1)
  }

  @inlinable
  public func mode(seed: UInt64?) -> Tensor<Scalar> {
    Tensor<Scalar>(logProbabilities.argmax(squeezingAxis: 1))
  }

  @inlinable
  public func sample(seed: UInt64? = nil) -> Tensor<Scalar> {
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    let multinomial: Tensor<Scalar> = Raw.multinomial(
      logits: logProbabilities,
      numSamples: Tensor<Int32>(1),
      seed: tfSeed.graph,
      seed2: tfSeed.op)
    return multinomial.gathering(atIndices: Tensor<Int32>(0), alongAxis: 1)
  }
}
