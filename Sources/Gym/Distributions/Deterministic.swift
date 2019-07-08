import TensorFlow

public struct Deterministic<Scalar: TensorFlowScalar & Equatable>: Distribution {
  public let value: Tensor<Scalar>

  @inlinable
  public init(at value: Tensor<Scalar>) {
    self.value = value
  }

  @inlinable
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    // TODO: What about NaNs?
    return log(Tensor<Float>(value .== self.value))
  }

  @inlinable
  public func entropy() -> Tensor<Float> {
    return Tensor<Float>(zeros: value.shape)
  }

  @inlinable
  public func mode(seed: UInt64?) -> Tensor<Scalar> {
    return value
  }

  @inlinable
  public func sample(seed: UInt64? = nil) -> Tensor<Scalar> {
    return value
  }
}
