import Foundation

public protocol Environment {
  associatedtype Action
  associatedtype Observation
  associatedtype Reward
  associatedtype Discount
  associatedtype ActionSpace: Space where ActionSpace.Value == Action
  associatedtype ObservationSpace: Space where ObservationSpace.Value == Observation

  var actionSpace: ActionSpace { get }
  var observationSpace: ObservationSpace { get }

  /// Updates the environment according to the provided action.
  @discardableResult
  mutating func step(taking action: Action) -> EnvironmentStep<Observation, Reward, Discount>

  /// Resets the environment.
  @discardableResult
  mutating func reset() -> EnvironmentStep<Observation, Reward, Discount>
}

public extension Environment {
  /// Renders the last step observation using the provided renderer.
  @inlinable
  func render<R: Renderer>(
    observation: Observation,
    using renderer: inout R
  ) throws where R.Data == Observation {
    try renderer.render(observation)
  }
}

// public protocol BatchedEnvironment: Environment {
//   var batchSize: Int? { get }
// }

/// Contains the data emitted by an environment at a single step of interaction.
public struct EnvironmentStep<Observation, Reward, Discount> {
  public let kind: Kind
  public let observation: Observation
  public let reward: Reward
  public let discount: Discount?

  public init(
    kind: Kind,
    observation: Observation,
    reward: Reward,
    discount: Discount? = nil
  ) {
    self.kind = kind
    self.observation = observation
    self.reward = reward
    self.discount = discount
  }

  @inlinable
  public func isFirst() -> Bool {
    return kind == .first
  }

  @inlinable
  public func isTransition() -> Bool {
    return kind == .transition
  }

  @inlinable
  public func isLast() -> Bool {
    return kind == .last
  }

  @inlinable
  public func copy(
    kind: Kind? = nil,
    observation: Observation? = nil,
    reward: Reward? = nil,
    discount: Discount? = nil
  ) -> EnvironmentStep<Observation, Reward, Discount> {
    return EnvironmentStep(
      kind: kind ?? self.kind,
      observation: observation ?? self.observation,
      reward: reward ?? self.reward,
      discount: discount ?? self.discount)
  }
}

public extension EnvironmentStep {
  /// Represents the type of a step.
  enum Kind: Int {
    /// Denotes the first step in a sequence.
    case first = 0
    /// Denotes an transition step in a sequence (i.e., not first or last).
    case transition = 1
    /// Denotes the last step in a sequence.
    case last = 2
  }
}
