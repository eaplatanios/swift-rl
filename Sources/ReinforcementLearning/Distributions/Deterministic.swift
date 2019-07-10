import TensorFlow

public struct Deterministic<Scalar: TensorFlowScalar & Equatable>: Distribution, TensorGroup {
  // TODO: Make `internal(set)` once `@usableFromInline` is supported.
  public var value: Tensor<Scalar>

  @inlinable
  public init(at value: Tensor<Scalar>) {
    self.value = value
  }

  @inlinable
  public func logProbability(of value: Tensor<Scalar>) -> Tensor<Float> {
    // TODO: What about NaNs?
    log(Tensor<Float>(value .== self.value))
  }

  @inlinable
  public func entropy() -> Tensor<Float> {
    Tensor<Float>(zeros: value.shape)
  }

  @inlinable
  public func mode(usingSeed seed: TensorFlowSeed = Context.local.randomSeed) -> Tensor<Scalar> {
    value
  }

  @inlinable
  public func sample(usingSeed seed: TensorFlowSeed = Context.local.randomSeed) -> Tensor<Scalar> {
    value
  }
}

// TODO: Should be derived automatically.
extension Deterministic: Replayable where Scalar: Numeric {
  public init(emptyLike example: Deterministic, withCapacity capacity: Int) {
    self.init(at: Tensor<Scalar>(emptyLike: example.value, withCapacity: capacity))
  }

  public mutating func update(atIndices indices: Tensor<Int64>, using values: Deterministic) {
    value.update(atIndices: indices, using: values.value)
  }

  public func gathering(atIndices indices: Tensor<Int64>) -> Deterministic {
    Deterministic(at: value.gathering(atIndices: indices))
  }
}
