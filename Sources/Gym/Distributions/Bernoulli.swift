import TensorFlow

public struct Bernoulli<ValueDataType: TensorFlowInteger>: Distribution, Differentiable {
  /// Unnormalized log-probabilities of this bernoulli distribution.
  public let logProbabilities: Tensor<Float>

  @differentiable
  public init(logProbabilities: Tensor<Float>) {
    self.logProbabilities = logProbabilities
  }

  @differentiable
  public init(probabilities: Tensor<Float>) {
    self.logProbabilities = log(probabilities)
  }

  @differentiable
  public init(logits: Tensor<Float>) {
    self.logProbabilities = logSigmoid(logits)
  }

  @differentiable(wrt: self)
  public func probability(of value: Tensor<ValueDataType>) -> Tensor<Float> {
    return exp(logProbability(of: value))
  }

  @differentiable(wrt: self)
  public func logProbability(of value: Tensor<ValueDataType>) -> Tensor<Float> {
    let zeros = Tensor<Float>(value)
    let ones = 1.0 - zeros
    return zeros * log(1 - exp(logProbabilities)) + ones * logProbabilities
  }

  public func mode(seed: UInt64?) -> Tensor<ValueDataType> {
    return Tensor<ValueDataType>(logProbabilities .> log(0.5))
  }

  public func sample(seed: UInt64? = nil) -> Tensor<ValueDataType> {
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    let uniform: Tensor<Float> = Raw.randomUniform(
      shape: logProbabilities.shapeTensor,
      seed: tfSeed.graph,
      seed2: tfSeed.op)
    return Tensor<ValueDataType>(logProbabilities .< log(uniform))
  }
}
