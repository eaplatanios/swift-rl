import TensorFlow

public protocol Network: Layer {
  associatedtype State
  var state: State { get set }
  func initialize(using input: Input)
  func copy() -> Self
}
