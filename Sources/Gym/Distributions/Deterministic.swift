import TensorFlow

public struct Deterministic<ValueDataType: TensorFlowScalar & Equatable>: Distribution {
  public let value: Tensor<ValueDataType>

  public init(at value: Value) {
    self.value = value
  }

  public func probability(of value: Tensor<ValueDataType>) -> Tensor<Float> {
    return Tensor<Float>(value .== self.value)
  }

  public func logProbability(of value: Tensor<ValueDataType>) -> Tensor<Float> {
    return log(probability(of: value))
  }

  public func mode(seed: UInt64?) -> Tensor<ValueDataType> {
    return value
  }

  public func sample(seed: UInt64? = nil) -> Tensor<ValueDataType> {
    return value
  }
}
