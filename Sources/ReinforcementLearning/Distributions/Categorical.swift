import TensorFlow

public struct Categorical<Scalar: TensorFlowIndex>: DifferentiableDistribution, KeyPathIterable {
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
    let outerDimCount = logProbabilities.rank - 1
    return softmaxCrossEntropy(
      logits: logProbabilities.flattenedBatch(outerDimCount: outerDimCount),
      labels: Tensor<Int32>(value).flattenedBatch(outerDimCount: outerDimCount))
  }

  @inlinable
  @differentiable(wrt: self)
  public func entropy() -> Tensor<Float> {
    -(logProbabilities * exp(logProbabilities)).sum(squeezingAxes: -1)
  }

  @inlinable
  public func mode(usingSeed seed: TensorFlowSeed = Context.local.randomSeed) -> Tensor<Scalar> {
    Tensor<Scalar>(logProbabilities.argmax(squeezingAxis: 1))
  }

  @inlinable
  public func sample(
    usingSeed seed: TensorFlowSeed = Context.local.randomSeed
  ) -> Tensor<Scalar> {
    let outerDimCount = self.logProbabilities.rank - 1
    let logProbabilities = self.logProbabilities.flattenedBatch(outerDimCount: outerDimCount)
    let multinomial: Tensor<Scalar> = Raw.multinomial(
      logits: logProbabilities,
      numSamples: Tensor<Int32>(1),
      seed: Int64(seed.graph),
      seed2: Int64(seed.op))
    let flattenedSamples = multinomial.gathering(atIndices: Tensor<Int32>(0), alongAxis: 1)
    return flattenedSamples.unflattenedBatch(
      outerDims: [Int](self.logProbabilities.shape.dimensions[0..<outerDimCount]))
  }
}

/// Returns the log-softmax of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpLogSoftmax(_:))
public func logSoftmax<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  Raw.logSoftmax(logits: x)
}

@inlinable
func _vjpLogSoftmax<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
  ) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
  let value = logSoftmax(x)
  return (value, { v in
    v - v.sum(alongAxes: -1) * exp(value)
  })
}
