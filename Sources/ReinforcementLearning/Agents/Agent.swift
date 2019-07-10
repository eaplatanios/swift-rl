
public protocol Agent {
  associatedtype Policy: ProbabilisticPolicy

  /// Policy used to act by this agent.
  var policy: Policy { get }

  /// Initializes this agent.
  mutating func initialize()

  /// Trains this agent using the provided experience.
  /// - Returns: Loss function value.
  mutating func update(using trajectory: Trajectory<Action, Observation, Reward, State>) -> Float
}

public extension Agent {
  typealias Action = Policy.Action
  typealias Observation = Policy.Observation
  typealias Reward = Policy.Reward
  typealias State = Policy.State
}
