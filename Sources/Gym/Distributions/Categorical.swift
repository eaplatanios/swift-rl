import TensorFlow

public struct Categorical<ValueDataType: TensorFlowInteger>: Distribution, Differentiable {
  /// Log-probabilities of this categorical distribution.
  public let logProbabilities: Tensor<Float>
  @noDerivative public let domain: Tensor<ValueDataType>

  @differentiable(wrt: logProbabilities)
  public init(logProbabilities: Tensor<Float>, domain: Tensor<ValueDataType>) {
    self.logProbabilities = logProbabilities
    self.domain = domain
  }

  @differentiable(wrt: probabilities)
  public init(probabilities: Tensor<Float>, domain: Tensor<ValueDataType>) {
    self.logProbabilities = log(probabilities)
    self.domain = domain
  }

  @differentiable(wrt: logits)
  public init(logits: Tensor<Float>, domain: Tensor<ValueDataType>) {
    self.logProbabilities = logSoftmax(logits)
    self.domain = domain
  }

  // TODO: @differentiable(wrt: self)
  public func probability(of value: Tensor<ValueDataType>) -> Tensor<Float> {
    return exp(logProbability(of: value))
  }

  // TODO: @differentiable(wrt: self)
  public func logProbability(of value: Tensor<ValueDataType>) -> Tensor<Float> {
    return logProbabilities.batchGathering(
      atIndices: Tensor<Int32>(value),
      alongAxis: 1,
      numBatchDims: 1)
  }

  public func mode(seed: UInt64?) -> Tensor<ValueDataType> {
    let indices = logProbabilities.argmax(squeezingAxis: 1)
    return domain.gathering(atIndices: indices, alongAxis: 1)
  }

  public func sample(seed: UInt64? = nil) -> Tensor<ValueDataType> {
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    let indices: Tensor<Int32> = Raw.multinomial(
      logits: logProbabilities,
      numSamples: Tensor<Int32>(1),
      seed: tfSeed.graph,
      seed2: tfSeed.op).gathering(atIndices: Tensor<Int32>(0), alongAxis: 1)
    return domain.gathering(atIndices: indices, alongAxis: 1)
  }
}

public extension Categorical where ValueDataType == Int32 {
  // @differentiable(wrt: logits)
  init(logits: Tensor<Float>) {
    let range = Tensor<Int32>(rangeFrom: 0, to: Int32(logits.shape[1]), stride: 1).rankLifted()
    let domain = Tensor<Int32>(tiling: range, multiples: Tensor<Int32>([Int32(logits.shape[0]), 1]))
    self = Categorical<ValueDataType>(logits: logits, domain: domain)
  }

  // @differentiable(wrt: probabilities)
  init(probabilities: Tensor<Float>) {
    let range = Tensor<Int32>(
      rangeFrom: 0, to: Int32(probabilities.shape[1]), stride: 1).rankLifted()
    let domain = Tensor<Int32>(
      tiling: range, multiples: Tensor<Int32>([Int32(probabilities.shape[0]), 1]))
    self = Categorical<ValueDataType>(probabilities: probabilities, domain: domain)
  }
}
