import Foundation

public protocol Environment {
  associatedtype Action
  associatedtype Observation
  associatedtype Reward
  associatedtype ActionSpace: Space where ActionSpace.Value == Action
  associatedtype ObservationSpace: Space where ObservationSpace.Value == Observation

  typealias Step = EnvironmentStep<Observation, Reward>

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

/// Represents the type of a step.
public enum EnvironmentStepKind: Int {
  /// Denotes the first step in a sequence.
  case first = 0
  /// Denotes an transition step in a sequence (i.e., not first or last).
  case transition = 1
  /// Denotes the last step in a sequence.
  case last = 2
}

public protocol BatchedEnvironment: Environment
where Action: Batchable, Observation: Batchable, Reward: Batchable {
  typealias BatchedStep = BatchedEnvironmentStep<Observation, Reward>

  /// Updates the environment according to the provided action.
  @discardableResult
  mutating func batchedStep(taking action: Action.Batched) -> BatchedStep

  /// Resets the environment.
  @discardableResult
  mutating func batchedReset(batchSize: Int) -> BatchedStep
}

public extension BatchedEnvironment {
  /// Updates the environment according to the provided action.
  @discardableResult
  mutating func step(taking action: Action) -> Step {
    let batched = batchedStep(taking: Action.batch([action]))
    return EnvironmentStep(
      kind: batched.kind[0],
      observation: Observation.unbatch(batched.observation)[0],
      reward: Reward.unbatch(batched.reward)[0])
  }

  /// Resets the environment.
  @discardableResult
  mutating func reset() -> Step {
    let batched = batchedReset(batchSize: 1)
    return EnvironmentStep(
      kind: batched.kind[0],
      observation: Observation.unbatch(batched.observation)[0],
      reward: Reward.unbatch(batched.reward)[0])
  }
}

/// Contains the data emitted by a batched environment at a single step of interaction.
public struct BatchedEnvironmentStep<Observation: Batchable, Reward: Batchable> {
  public let kind: [EnvironmentStepKind]
  public let observation: Observation.Batched
  public let reward: Reward.Batched

  public init(
    kind: [EnvironmentStepKind],
    observation: Observation.Batched,
    reward: Reward.Batched
  ) {
    self.kind = kind
    self.observation = observation
    self.reward = reward
  }

  @inlinable
  public func copy(
    kind: [EnvironmentStepKind]? = nil,
    observation: Observation.Batched? = nil,
    reward: Reward.Batched? = nil
  ) -> BatchedEnvironmentStep<Observation, Reward> {
    return BatchedEnvironmentStep(
      kind: kind ?? self.kind,
      observation: observation ?? self.observation,
      reward: reward ?? self.reward)
  }
}

public extension EnvironmentStep where Observation: Batchable, Reward: Batchable {
  func batched() -> BatchedEnvironmentStep<Observation, Reward> {
    return BatchedEnvironmentStep(
      kind: [kind],
      observation: Observation.batch([observation]),
      reward: Reward.batch([reward]))
  }
}
