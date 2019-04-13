
public protocol Wrapper: Environment {
  associatedtype WrappedEnvironment: Environment
  var wrappedEnvironment: WrappedEnvironment { get set }
}

public extension Wrapper where WrappedEnvironment.ActionSpace == ActionSpace {
  var actionSpace: ActionSpace {
    get { return wrappedEnvironment.actionSpace }
  }
}

public extension Wrapper where WrappedEnvironment.ObservationSpace == ObservationSpace {
  var observationSpace: ObservationSpace {
    get { return wrappedEnvironment.observationSpace }
  }
}

public extension Wrapper
where
  WrappedEnvironment.Action == Action,
  WrappedEnvironment.Observation == Observation,
  WrappedEnvironment.Reward == Reward
{
  @inlinable
  mutating func step(taking action: Action) -> EnvironmentStep<Observation, Reward> {
    return wrappedEnvironment.step(taking: action)
  }

  @inlinable
  mutating func reset() -> EnvironmentStep<Observation, Reward> {
    return wrappedEnvironment.reset()
  }
}

/// Ends episodes after a specified number of steps.
public struct TimeLimit<WrappedEnvironment: Environment>: Wrapper {
  public typealias Action = WrappedEnvironment.Action
  public typealias Observation = WrappedEnvironment.Observation
  public typealias Reward = WrappedEnvironment.Reward
  public typealias ActionSpace = WrappedEnvironment.ActionSpace
  public typealias ObservationSpace = WrappedEnvironment.ObservationSpace

  public var wrappedEnvironment: WrappedEnvironment
  public let limit: Int

  @usableFromInline internal var numSteps: Int = 0
  @usableFromInline internal var resetRequired: Bool = false

  public init(wrapping environment: WrappedEnvironment, withLimit limit: Int) {
    self.wrappedEnvironment = environment
    self.limit = limit
  }

  @inlinable
  public mutating func step(taking action: Action) -> EnvironmentStep<Observation, Reward> {
    if resetRequired {
      return reset()
    }
    
    var result = wrappedEnvironment.step(taking: action)
    numSteps += 1

    if numSteps >= limit {
      result = result.copy(kind: .last)
    }

    if result.kind == .last {
      numSteps = 0
      resetRequired = true
    }

    return result
  }

  @inlinable
  public mutating func reset() -> EnvironmentStep<Observation, Reward> {
    numSteps = 0
    resetRequired = false
    return wrappedEnvironment.reset()
  }
}

/// Repeats actions multiple times while acummulating the collected reward.
public struct ActionRepeat<WrappedEnvironment: Environment>: Wrapper 
  where WrappedEnvironment.Reward: AdditiveArithmetic {
  public typealias Action = WrappedEnvironment.Action
  public typealias Observation = WrappedEnvironment.Observation
  public typealias Reward = WrappedEnvironment.Reward
  public typealias ActionSpace = WrappedEnvironment.ActionSpace
  public typealias ObservationSpace = WrappedEnvironment.ObservationSpace

  public var wrappedEnvironment: WrappedEnvironment
  public let numRepeats: Int

  public init(wrapping environment: WrappedEnvironment, repeating numRepeats: Int) {
    precondition(numRepeats > 1, "'numRepeats' should be greater than 1.")
    self.wrappedEnvironment = environment
    self.numRepeats = numRepeats
  }

  @inlinable
  public mutating func step(taking action: Action) -> EnvironmentStep<Observation, Reward> {
    var result = wrappedEnvironment.step(taking: action)
    var reward = result.reward
    for _ in 1..<numRepeats {
      result = wrappedEnvironment.step(taking: action)
      reward += result.reward
      if result.kind == .last {
        break
      }
    }
    return result.copy(reward: reward)
  }
}

/// Collects statistics as the environment is being used.
public struct RunStatistics<WrappedEnvironment: Environment>: Wrapper {
  public typealias Action = WrappedEnvironment.Action
  public typealias Observation = WrappedEnvironment.Observation
  public typealias Reward = WrappedEnvironment.Reward
  public typealias ActionSpace = WrappedEnvironment.ActionSpace
  public typealias ObservationSpace = WrappedEnvironment.ObservationSpace

  public var wrappedEnvironment: WrappedEnvironment

  /// Number of `.first` steps.
  public private(set) var numResets: Int = 0

  /// Number of `.last` steps. Note that this will not count for episodes that are not terminated 
  /// with a `.last` step.
  public private(set) var numEpisodes: Int = 0

  /// Number of steps in the current episode.
  public private(set) var numEpisodeSteps: Int = 0

  /// Total number of steps, ignoring `.first` steps.
  public private(set) var numTotalSteps: Int = 0

  public init(wrapping environment: WrappedEnvironment) {
    self.wrappedEnvironment = environment
  }

  @inlinable
  public mutating func step(taking action: Action) -> EnvironmentStep<Observation, Reward> {
    let result = wrappedEnvironment.step(taking: action)
    
    if result.kind == .first {
      numResets += 1
      numEpisodeSteps = 0
    } else {
      numEpisodeSteps += 1
      numTotalSteps += 1
    }

    if result.kind == .last {
      numEpisodes += 1
    }

    return result
  }

  @inlinable
  public mutating func reset() -> EnvironmentStep<Observation, Reward> {
    numResets += 1
    numEpisodeSteps = 0
    return wrappedEnvironment.reset()
  }
}
