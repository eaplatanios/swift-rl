import TensorFlow

public struct Uniform<Scalar: TensorFlowFloatingPoint>: DifferentiableDistribution {
  @noDerivative public let shape: Tensor<Int32>
  public let lowerBound: Tensor<Scalar>
  public let upperBound: Tensor<Scalar>

  @inlinable
  // TODO: @differentiable(wrt: (lowerBound, upperBound))
  public init(
    shape: Tensor<Int32>,
    lowerBound: Tensor<Scalar> = Tensor<Scalar>(zeros: []),
    upperBound: Tensor<Scalar> = Tensor<Scalar>(ones: [])
  ) {
    self.shape = shape
    self.lowerBound = lowerBound
    self.upperBound = upperBound
  }

  @inlinable
  @differentiable(wrt: self)
  public func probability(of value: Tensor<Scalar>) -> Tensor<Float> {
    return 1.0 / Tensor<Float>(upperBound - lowerBound)
  }

  @inlinable
  @differentiable(wrt: self)
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    return log(probability(of: value))
  }

  @inlinable
  @differentiable(wrt: self)
  public func entropy() -> Tensor<Float> {
    return log(Tensor<Float>(upperBound - lowerBound))
  }

  @inlinable
  public func mode(seed: UInt64?) -> Tensor<Scalar> {
    return sample(seed: seed)
  }

  @inlinable
  public func sample(seed: UInt64? = nil) -> Tensor<Scalar> {
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    let sample: Tensor<Scalar> = Raw.randomUniform(
      shape: shape, seed: tfSeed.graph, seed2: tfSeed.op)
    return sample * (upperBound - lowerBound) + lowerBound
  }
}

public extension Uniform {
  @inlinable
  // TODO: @differentiable(wrt: (lowerBound, upperBound))
  init(lowerBound: Tensor<Scalar>, upperBound: Tensor<Scalar>) {
    self.shape = lowerBound.withoutDerivative().shapeTensor
    self.lowerBound = lowerBound
    self.upperBound = upperBound
  }
}
