import TensorFlow

public struct Deterministic<Scalar: TensorFlowScalar & Equatable>: Distribution, KeyPathIterable {
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
