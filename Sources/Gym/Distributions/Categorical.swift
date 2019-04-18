import TensorFlow

public struct Categorical<ValueDataType: TensorFlowInteger>: Distribution, Differentiable {
  /// Unnormalized log-probabilities of this categorical distribution.
  public let logits: Tensor<Float>
  @noDerivative public let domain: Tensor<ValueDataType>

  @differentiable(wrt: logits)
  public init(logits: Tensor<Float>, domain: Tensor<ValueDataType>) {
    // TODO: Find a way to use this with `@differentiable`.
    // precondition(logits.rank == 1, "'logits' must be a vector.")
    self.logits = logits
    self.domain = domain
  }

  @differentiable(wrt: probabilities)
  public init(probabilities: Tensor<Float>, domain: Tensor<ValueDataType>) {
    self = Categorical(logits: log(probabilities), domain: domain)
  }

  public func mode(seed: UInt64?) -> Tensor<ValueDataType> {
    let indices = logits.argmax(squeezingAxis: 1)
    return domain.gathering(atIndices: indices, alongAxis: 1)
  }

  public func sample(seed: UInt64? = nil) -> Tensor<ValueDataType> {
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    let indices: Tensor<Int32> = Raw.multinomial(
      logits: logits,
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
