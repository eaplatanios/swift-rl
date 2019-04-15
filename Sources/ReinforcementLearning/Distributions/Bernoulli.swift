import TensorFlow

public struct Bernoulli<ValueDataType: TensorFlowInteger>: Distribution, Differentiable {
  /// Unnormalized log-probabilities of this bernoulli distribution.
  public let probabilities: Tensor<Float>

  @differentiable(wrt: probabilities)
  public init(probabilities: Tensor<Float>) {
    // TODO: Find a way to use this with `@differentiable`.
    // precondition(probabilities.rank == 1, "'probabilities' must be a vector.")
    self.probabilities = probabilities
  }

  @differentiable(wrt: logits)
  public init(logits: Tensor<Float>) {
    self.probabilities = sigmoid(logits)
  }

  public func mode(seed: UInt64?) -> Tensor<ValueDataType> {
    return Tensor<ValueDataType>(probabilities .> 0.5)
  }

  public func sample(seed: UInt64? = nil) -> Tensor<ValueDataType> {
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    let uniform: Tensor<Float> = Raw.randomUniform(
      shape: probabilities.shapeTensor,
      seed: tfSeed.graph,
      seed2: tfSeed.op)
    return Tensor<ValueDataType>(probabilities .< uniform)
  }
}
