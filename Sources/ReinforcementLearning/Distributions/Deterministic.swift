import TensorFlow

public struct Deterministic<ValueDataType: TensorFlowScalar>: Distribution {
  public let value: Tensor<ValueDataType>

  public init(at value: Tensor<ValueDataType>) {
    self.value = value
  }

  public func sample(seed: UInt64? = nil) -> Tensor<ValueDataType> {
    return value
  }
}

extension Deterministic: Differentiable where ValueDataType: TensorFlowFloatingPoint {
  @differentiable
  public init(at value: Tensor<ValueDataType>) {
    self.value = value
  }
}
