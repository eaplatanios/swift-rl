import Gym
import TensorFlow

public protocol Network: Differentiable, KeyPathIterable
where AllDifferentiableVariables: KeyPathIterable {
  /// Input type of the network.
  associatedtype Input

  /// State type of the network.
  associatedtype State: Differentiable

  /// Output type of the network.
  associatedtype Output: Differentiable

  var state: State { get set }

  /// Initializes this network.
  func initialize(using input: Input)

  /// Returns the output obtained from applying the network to the given input.
  ///
  /// - Parameters:
  ///   - input: Input to the network.
  ///   - state: State of the network.
  /// - Returns: Output of the network along with the updated state.
  @differentiable(wrt: self)
  func applied(to input: Input) -> Output
}

public extension Network {
  func initialize(using input: Input) { }
}

// extension Layer: Network {
//   public typealias State = None

//   @differentiable(wrt: self)
//   public func applied(to input: Input, in state: State) -> NetworkResult<Output, State> {
//     return NetworkResult(output: applied(to: input), state: state)
//   }
// }
