import TensorFlow

public struct Categorical<Scalar: TensorFlowInteger>: Distribution {
  /// Log-probabilities of this categorical distribution.
  public let logProbabilities: Tensor<Float>
  public let domain: Tensor<Scalar>
  // @noDerivative public let domain: Tensor<Scalar>

  @inlinable
  // TODO: @differentiable(wrt: logProbabilities)
  public init(logProbabilities: Tensor<Float>, domain: Tensor<Scalar>) {
    self.logProbabilities = logProbabilities
    self.domain = domain
  }

  @inlinable
  // TODO: @differentiable(wrt: probabilities)
  public init(probabilities: Tensor<Float>, domain: Tensor<Scalar>) {
    self.logProbabilities = log(probabilities)
    self.domain = domain
  }

  @inlinable
  // TODO: @differentiable(wrt: logits)
  public init(logits: Tensor<Float>, domain: Tensor<Scalar>) {
    self.logProbabilities = logSoftmax(logits)
    self.domain = domain
  }

  @inlinable
  // TODO: @differentiable(wrt: self)
  public func probability(of value: Tensor<Scalar>) -> Tensor<Float> {
    return exp(logProbability(of: value))
  }

  @inlinable
  // TODO: @differentiable(wrt: self)
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    return logProbabilities.batchGathering(
      atIndices: Tensor<Int32>(value),
      alongAxis: 1,
      numBatchDims: 1)
  }

  @inlinable
  public func mode(seed: UInt64?) -> Tensor<Scalar> {
    let indices = logProbabilities.argmax(squeezingAxis: 1)
    return domain.gathering(atIndices: indices, alongAxis: 1)
  }

  @inlinable
  public func sample(seed: UInt64? = nil) -> Tensor<Scalar> {
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    let indices: Tensor<Int32> = Raw.multinomial(
      logits: logProbabilities,
      numSamples: Tensor<Int32>(1),
      seed: tfSeed.graph,
      seed2: tfSeed.op).gathering(atIndices: Tensor<Int32>(0), alongAxis: 1)
    return domain.gathering(atIndices: indices, alongAxis: 1)
  }
}

public extension Categorical where Scalar == Int32 {
  @inlinable
  // TODO: @differentiable(wrt: logProbabilities)
  init(logProbabilities: Tensor<Float>) {
    let range = Tensor<Int32>(
      rangeFrom: 0, to: Int32(logProbabilities.shape[1]), stride: 1).rankLifted()
    let domain = Tensor<Int32>(
      tiling: range, multiples: Tensor<Int32>([Int32(logProbabilities.shape[0]), 1]))
    self = Categorical<Scalar>(logProbabilities: logProbabilities, domain: domain)
  }

  @inlinable
  // TODO: @differentiable(wrt: probabilities)
  init(probabilities: Tensor<Float>) {
    let range = Tensor<Int32>(
      rangeFrom: 0, to: Int32(probabilities.shape[1]), stride: 1).rankLifted()
    let domain = Tensor<Int32>(
      tiling: range, multiples: Tensor<Int32>([Int32(probabilities.shape[0]), 1]))
    self = Categorical<Scalar>(probabilities: probabilities, domain: domain)
  }

  @inlinable
  // TODO: @differentiable(wrt: logits)
  init(logits: Tensor<Float>) {
    let range = Tensor<Int32>(rangeFrom: 0, to: Int32(logits.shape[1]), stride: 1).rankLifted()
    let domain = Tensor<Int32>(tiling: range, multiples: Tensor<Int32>([Int32(logits.shape[0]), 1]))
    self = Categorical<Scalar>(logits: logits, domain: domain)
  }
}
