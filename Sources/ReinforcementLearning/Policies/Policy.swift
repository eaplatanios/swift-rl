import TensorFlow

public protocol Policy {
  associatedtype Action
  associatedtype Observation
  associatedtype Reward
  associatedtype State

  var batched: Bool { get }
  var state: State { get set }

  func initialize(using observation: Observation)
  func action(for step: Step<Observation, Reward>) -> Action
  func copy() -> Self
}

public extension Policy where State == None {
  func initialize(using observation: Observation) { }
}
