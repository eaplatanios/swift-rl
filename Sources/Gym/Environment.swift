import Foundation
import TensorFlow

public protocol Environment {
  associatedtype Action
  associatedtype Observation
  associatedtype Reward
  associatedtype ActionSpace: Space where ActionSpace.Value == Action
  associatedtype ObservationSpace: Space where ObservationSpace.Value == Observation

  typealias Step = EnvironmentStep<Observation, Reward>

  var batched: Bool { get }
  var actionSpace: ActionSpace { get }
  var observationSpace: ObservationSpace { get }

  /// Updates the environment according to the provided action.
  @discardableResult
  mutating func step(taking action: Action) -> Step

  /// Resets the environment.
  @discardableResult
  mutating func reset() -> Step
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

/// Contains the data emitted by an environment at a single step of interaction.
public struct EnvironmentStep<Observation, Reward> {
  public let kind: EnvironmentStepKind
  public let observation: Observation
  public let reward: Reward

  public init(kind: EnvironmentStepKind, observation: Observation, reward: Reward) {
    self.kind = kind
    self.observation = observation
    self.reward = reward
  }

  @inlinable
  public func copy(
    kind: EnvironmentStepKind? = nil,
    observation: Observation? = nil,
    reward: Reward? = nil
  ) -> EnvironmentStep<Observation, Reward> {
    return EnvironmentStep(
      kind: kind ?? self.kind,
      observation: observation ?? self.observation,
      reward: reward ?? self.reward)
  }
}

extension EnvironmentStep: Stackable where Observation: Stackable, Reward: Stackable {
  public static func stack(
    _ values: [EnvironmentStep]
  ) -> EnvironmentStep<Observation.Stacked, Reward.Stacked> {
    return EnvironmentStep<Observation.Stacked, Reward.Stacked>(
      kind: EnvironmentStepKind.stack(values.map { $0.kind }),
      observation: Observation.stack(values.map { $0.observation }),
      reward: Reward.stack(values.map { $0.reward }))
  }

  public static func unstack(
    _ value: EnvironmentStep<Observation.Stacked, Reward.Stacked>
  ) -> [EnvironmentStep] {
    let kinds = EnvironmentStepKind.unstack(value.kind)
    let observations = Observation.unstack(value.observation)
    let rewards = Reward.unstack(value.reward)
    return zip(kinds, observations, rewards).map {
      EnvironmentStep(kind: $0, observation: $1, reward: $2)
    }
  }
}

/// Represents the type of a step.
public struct EnvironmentStepKind {
  public let rawValue: Tensor<Int32>

  public init(_ rawValue: Tensor<Int32>) {
    self.rawValue = rawValue
  }

  @inlinable
  public func isFirst() -> Tensor<Bool> {
    return rawValue .== 0
  }

  @inlinable
  public func isTransition() -> Tensor<Bool> {
    return rawValue .== 1
  }

  @inlinable
  public func isLast() -> Tensor<Bool> {
    return rawValue .== 2
  }
}

public extension EnvironmentStepKind {
  /// Denotes the first step in a sequence.
  static let first = EnvironmentStepKind(Tensor<Int32>(0))

  /// Denotes an transition step in a sequence (i.e., not first or last).
  static let transition = EnvironmentStepKind(Tensor<Int32>(1))

  /// Denotes the last step in a sequence.
  static let last = EnvironmentStepKind(Tensor<Int32>(2))
}

extension EnvironmentStepKind: Stackable {
  public static func stack(_ values: [EnvironmentStepKind]) -> EnvironmentStepKind {
    return EnvironmentStepKind(Tensor<Int32>.stack(values.map{ $0.rawValue }))
  }

  public static func unstack(_ value: EnvironmentStepKind) -> [EnvironmentStepKind] {
    return Tensor<Int32>.unstack(value.rawValue).map(EnvironmentStepKind.init)
  }
}
