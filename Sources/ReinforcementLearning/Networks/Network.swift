import TensorFlow

public protocol Network: Layer {
  /// State type of the network.
  associatedtype State: Differentiable

  var state: State { get set }

  /// Initializes this network.
  func initialize(using input: Input)
}

public extension Network {
  func initialize(using input: Input) { }
}

// extension Layer: Network {
//   public typealias State = None
// }
