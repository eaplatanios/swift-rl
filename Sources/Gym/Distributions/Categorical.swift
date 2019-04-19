import TensorFlow

public struct Categorical<Scalar: TensorFlowInteger>: DifferentiableDistribution {
  /// Log-probabilities of this categorical distribution.
  public let logProbabilities: Tensor<Float>

  @inlinable
  // TODO: @differentiable(wrt: logProbabilities)
  public init(logProbabilities: Tensor<Float>) {
    self.logProbabilities = logProbabilities
  }

  @inlinable
  // TODO: @differentiable(wrt: probabilities)
  public init(probabilities: Tensor<Float>) {
    self.logProbabilities = log(probabilities)
  }

  @inlinable
  // TODO: @differentiable(wrt: logits)
  public init(logits: Tensor<Float>) {
    self.logProbabilities = logSoftmax(logits)
  }

  @inlinable
  @differentiable(wrt: self)
  public func probability(of value: Tensor<Scalar>) -> Tensor<Float> {
    return exp(logProbability(of: value))
  }

  @inlinable
  @differentiable(wrt: self, vjp: _vjpLogProbability)
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    return Raw.sparseSoftmaxCrossEntropyWithLogits(
      features: logProbabilities, labels: Tensor<Int32>(value)).loss
  }

  @inlinable
  @differentiable(wrt: self)
  public func entropy() -> Tensor<Float> {
    return -(logProbabilities * exp(logProbabilities)).sum(squeezingAxes: -1)
  }

  @inlinable
  public func mode(seed: UInt64?) -> Tensor<Scalar> {
    return Tensor<Scalar>(logProbabilities.argmax(squeezingAxis: 1))
  }

  @inlinable
  public func sample(seed: UInt64? = nil) -> Tensor<Scalar> {
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    return Raw.multinomial(
      logits: logProbabilities,
      numSamples: Tensor<Int32>(1),
      seed: tfSeed.graph,
      seed2: tfSeed.op).gathering(atIndices: Tensor<Int32>(0), alongAxis: 1)
  }
}

internal extension Categorical {
  @usableFromInline
  func _vjpLogProbability(
      of value: Tensor<Scalar>
  ) -> (Tensor<Float>, (Tensor<Float>) -> Categorical.CotangentVector) {
      let (crossEntropy, gradient) = Raw.sparseSoftmaxCrossEntropyWithLogits(
        features: logProbabilities, labels: Tensor<Int32>(value))
      return (crossEntropy, { v in
        Categorical.CotangentVector(logProbabilities: v * gradient)
      })
  }
}
