import TensorFlow

public struct Bernoulli<Scalar: TensorFlowInteger>: DifferentiableDistribution, TensorGroup {
  /// Unnormalized log-probabilities of this bernoulli distribution.
  public var logits: Tensor<Float>

  @inlinable
  @differentiable(wrt: logits)
  public init(logits: Tensor<Float>) {
    self.logits = logits
  }

  @inlinable
  @differentiable(wrt: logProbabilities)
  public init(logProbabilities: Tensor<Float>) {
    self.logits = logProbabilities
  }

  @inlinable
  @differentiable(wrt: probabilities)
  public init(probabilities: Tensor<Float>) {
    self.logits = log(probabilities)
  }

  @inlinable
  @differentiable(wrt: self)
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    max(logits, Tensor<Float>(0.0)) - logits * Tensor<Float>(value) + softplus(-abs(logits))
  }

  @inlinable
  @differentiable(wrt: self)
  public func entropy() -> Tensor<Float> {
    max(logits, Tensor<Float>(0.0)) - logits * sigmoid(logits) + softplus(-abs(logits))
  }

  @inlinable
  public func mode(
    usingSeed seed: TensorFlowSeed = Context.local.randomSeed
  ) -> Tensor<Scalar> {
    Tensor<Scalar>(logSigmoid(logits) .> log(0.5))
  }

  @inlinable
  public func sample(
    usingSeed seed: TensorFlowSeed = Context.local.randomSeed
  ) -> Tensor<Scalar> {
    let logProbabilities = logSigmoid(logits)
    let uniform: Tensor<Float> = Raw.randomUniform(
      shape: logProbabilities.shapeTensor,
      seed: Int64(seed.graph),
      seed2: Int64(seed.op))
    return Tensor<Scalar>(logProbabilities .< log(uniform))
  }
}

// TODO: Should be derived automatically.
extension Bernoulli: Replayable {
  public init(emptyLike example: Bernoulli, withCapacity capacity: Int) {
    self.init(logits: Tensor<Float>(emptyLike: example.logits, withCapacity: capacity))
  }

  public mutating func update(atIndices indices: Tensor<Int64>, using values: Bernoulli) {
    logits.update(atIndices: indices, using: values.logits)
  }

  public func gathering(atIndices indices: Tensor<Int64>) -> Bernoulli {
    Bernoulli(logits: logits.gathering(atIndices: indices))
  }
}
