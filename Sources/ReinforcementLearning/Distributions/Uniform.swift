import TensorFlow

public struct Uniform<
  Scalar: TensorFlowFloatingPoint
>: DifferentiableDistribution, KeyPathIterable {
  @noDerivative public let shape: Tensor<Int32>
  public var lowerBound: Tensor<Scalar>
  public var upperBound: Tensor<Scalar>

  @inlinable
  @differentiable(wrt: (lowerBound, upperBound))
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
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    log(1.0) - log(Tensor<Float>(upperBound - lowerBound))
  }

  @inlinable
  @differentiable(wrt: self)
  public func entropy() -> Tensor<Float> {
    log(Tensor<Float>(upperBound - lowerBound))
  }

  @inlinable
  public func mode(
    usingSeed seed: TensorFlowSeed = Context.local.randomSeed
  ) -> Tensor<Scalar> {
    sample(usingSeed: seed)
  }

  @inlinable
  public func sample(
    usingSeed seed: TensorFlowSeed = Context.local.randomSeed
  ) -> Tensor<Scalar> {
    let sample: Tensor<Scalar> = Raw.randomUniform(
      shape: shape, seed: Int64(seed.graph), seed2: Int64(seed.op))
    return sample * (upperBound - lowerBound) + lowerBound
  }
}

extension Uniform {
  @inlinable
  @differentiable(wrt: (lowerBound, upperBound))
  public init(lowerBound: Tensor<Scalar>, upperBound: Tensor<Scalar>) {
    self.shape = withoutDerivative(at: lowerBound.shapeTensor)
    self.lowerBound = lowerBound
    self.upperBound = upperBound
  }
}
