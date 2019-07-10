import TensorFlow

public struct Categorical<Scalar: TensorFlowIndex>: DifferentiableDistribution, TensorGroup {
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
  public func mode(
    usingSeed seed: TensorFlowSeed = Context.local.randomSeed
  ) -> Tensor<Scalar> {
    Tensor<Scalar>(logProbabilities.argmax(squeezingAxis: 1))
  }

  @inlinable
  public func sample(
    usingSeed seed: TensorFlowSeed = Context.local.randomSeed
  ) -> Tensor<Scalar> {
    let multinomial: Tensor<Scalar> = Raw.multinomial(
      logits: logProbabilities,
      numSamples: Tensor<Int32>(1),
      seed: Int64(seed.graph),
      seed2: Int64(seed.op))
    // TODO: Proper handling of batched samples.
    return multinomial
      .gathering(atIndices: Tensor<Int32>(0), alongAxis: 1)
      .gathering(atIndices: Tensor<Int32>(0), alongAxis: 0)
  }
}

// TODO: Should be derived automatically.
extension Categorical: Replayable {
  public init(emptyLike example: Categorical, withCapacity capacity: Int) {
    self.init(logProbabilities: Tensor<Float>(
      emptyLike: example.logProbabilities,
      withCapacity: capacity))
  }

  public mutating func update(atIndices indices: Tensor<Int64>, using values: Categorical) {
    logProbabilities.update(atIndices: indices, using: values.logProbabilities)
  }

  public func gathering(atIndices indices: Tensor<Int64>) -> Categorical {
    Categorical(logProbabilities: logProbabilities.gathering(atIndices: indices))
  }
}
