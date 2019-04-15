import TensorFlow

public struct Uniform<ValueDataType: TensorFlowFloatingPoint>: Distribution, Differentiable {
  @noDerivative public let shape: Tensor<Int32>
  public let lowerBound: Tensor<ValueDataType>
  public let upperBound: Tensor<ValueDataType>

  // TODO: [AUTODIFF].
  // @differentiable(wrt: (lowerBound, upperBound))
  public init(
    shape: Tensor<Int32>,
    lowerBound: Tensor<ValueDataType> = Tensor<ValueDataType>(zeros: []),
    upperBound: Tensor<ValueDataType> = Tensor<ValueDataType>(ones: [])
  ) {
    self.shape = shape
    self.lowerBound = lowerBound
    self.upperBound = upperBound
  }

  public func mode(seed: UInt64?) -> Tensor<ValueDataType> {
    return sample(seed: seed)
  }

  public func sample(seed: UInt64? = nil) -> Tensor<ValueDataType> {
    let tfSeed = seed?.tensorFlowSeed() ?? TensorFlowSeed(graph: 0, op: 0)
    let sample: Tensor<ValueDataType> = Raw.randomUniform(
      shape: shape, seed: tfSeed.graph, seed2: tfSeed.op)
    return sample * (upperBound - lowerBound) + lowerBound
  }
}

public extension Uniform {
  // TODO: [AUTODIFF] .shapeTensor is not differentiable.
  // @differentiable(wrt: (lowerBound, upperBound))
  init(lowerBound: Tensor<ValueDataType>, upperBound: Tensor<ValueDataType>) {
    self.shape = lowerBound.shapeTensor
    self.lowerBound = lowerBound
    self.upperBound = upperBound
  }
}
