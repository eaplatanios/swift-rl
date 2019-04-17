import TensorFlow

public struct Deterministic<Value>: Distribution {
  public let value: Value

  public init(at value: Value) {
    self.value = value
  }

  public func mode(seed: UInt64?) -> Value {
    return value
  }

  public func sample(seed: UInt64? = nil) -> Value {
    return value
  }
}

// extension Deterministic: Differentiable where Value: Differentiable {
//   @differentiable
//   public init(at value: Value) {
//     self.value = value
//   }

//   @differentiable(wrt: self)
//   public func mode(seed: UInt64?) -> Value {
//     return value
//   }

//   @differentiable(wrt: self)
//   public func sample(seed: UInt64? = nil) -> Value {
//     return value
//   }
// }
