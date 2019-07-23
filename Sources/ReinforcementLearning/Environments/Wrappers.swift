// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

public protocol Wrapper: Environment {
  associatedtype WrappedEnvironment: Environment
  var wrappedEnvironment: WrappedEnvironment { get set }
}

public extension Wrapper where WrappedEnvironment.ActionSpace == ActionSpace {
  var actionSpace: ActionSpace {
    get { wrappedEnvironment.actionSpace }
  }
}

public extension Wrapper where WrappedEnvironment.ObservationSpace == ObservationSpace {
  var observationSpace: ObservationSpace {
    get { wrappedEnvironment.observationSpace }
  }
}

public extension Wrapper
where
  WrappedEnvironment.Action == Action,
  WrappedEnvironment.Observation == Observation,
  WrappedEnvironment.Reward == Reward
{
  @inlinable
  func step(taking action: Action) -> Step<Observation, Reward> {
    wrappedEnvironment.step(taking: action)
  }

  @inlinable
  func reset() -> Step<Observation, Reward> {
    wrappedEnvironment.reset()
  }
}

// TODO: !!! Support the following wrappers for batched environments.
// /// Ends episodes after a specified number of steps.
// public struct TimeLimit<WrappedEnvironment: Environment>: Wrapper {
//   public typealias Action = WrappedEnvironment.Action
//   public typealias Observation = WrappedEnvironment.Observation
//   public typealias Reward = WrappedEnvironment.Reward
//   public typealias ActionSpace = WrappedEnvironment.ActionSpace
//   public typealias ObservationSpace = WrappedEnvironment.ObservationSpace

//   public let batched: Bool = false

//   public var wrappedEnvironment: WrappedEnvironment
//   public let limit: Int

//   @usableFromInline internal var numSteps: Int = 0
//   @usableFromInline internal var resetRequired: Bool = false

//   public init(wrapping environment: WrappedEnvironment, withLimit limit: Int) {
//     precondition(environment.batched == false, "The wrapped environment must not be batched.")
//     self.wrappedEnvironment = environment
//     self.limit = limit
//   }

//   @inlinable
//   public mutating func step(taking action: Action) -> Step<Observation, Reward> {
//     if resetRequired {
//       return reset()
//     }

//     var result = wrappedEnvironment.step(taking: action)
//     numSteps += 1

//     if numSteps >= limit {
//       result = result.copy(kind: .last)
//     }

//     if result.kind.isLast().scalarized() {
//       numSteps = 0
//       resetRequired = true
//     }

//     return result
//   }

//   @inlinable
//   public mutating func reset() -> Step<Observation, Reward> {
//     numSteps = 0
//     resetRequired = false
//     return wrappedEnvironment.reset()
//   }

//   @inlinable
//   public func copy() -> TimeLimit<WrappedEnvironment> {
//     TimeLimit(wrapping: wrappedEnvironment.copy(), withLimit: limit)
//   }
// }

// /// Repeats actions multiple times while acummulating the collected reward.
// public struct ActionRepeat<WrappedEnvironment: Environment>: Wrapper 
//   where WrappedEnvironment.Reward: AdditiveArithmetic {
//   public typealias Action = WrappedEnvironment.Action
//   public typealias Observation = WrappedEnvironment.Observation
//   public typealias Reward = WrappedEnvironment.Reward
//   public typealias ActionSpace = WrappedEnvironment.ActionSpace
//   public typealias ObservationSpace = WrappedEnvironment.ObservationSpace

//   public let batched: Bool = false

//   public var wrappedEnvironment: WrappedEnvironment
//   public let numRepeats: Int

//   public init(wrapping environment: WrappedEnvironment, repeating numRepeats: Int) {
//     precondition(environment.batched == false, "The wrapped environment must not be batched.")
//     precondition(numRepeats > 1, "'numRepeats' should be greater than 1.")
//     self.wrappedEnvironment = environment
//     self.numRepeats = numRepeats
//   }

//   @inlinable
//   public mutating func step(taking action: Action) -> Step<Observation, Reward> {
//     var result = wrappedEnvironment.step(taking: action)
//     var reward = result.reward
//     for _ in 1..<numRepeats {
//       result = wrappedEnvironment.step(taking: action)
//       reward += result.reward
//       if result.kind.isLast().scalarized() {
//         break
//       }
//     }
//     return result.copy(reward: reward)
//   }

//   @inlinable
//   public func copy() -> ActionRepeat<WrappedEnvironment> {
//     ActionRepeat(wrapping: wrappedEnvironment.copy(), repeating: numRepeats)
//   }
// }

// /// Collects statistics as the environment is being used.
// public struct RunStatistics<WrappedEnvironment: Environment>: Wrapper {
//   public typealias Action = WrappedEnvironment.Action
//   public typealias Observation = WrappedEnvironment.Observation
//   public typealias Reward = WrappedEnvironment.Reward
//   public typealias ActionSpace = WrappedEnvironment.ActionSpace
//   public typealias ObservationSpace = WrappedEnvironment.ObservationSpace

//   public let batched: Bool = false

//   public var wrappedEnvironment: WrappedEnvironment

//   // TODO: Add `private(set)` to the following properties.

//   /// Number of `.first` steps.
//   public var numResets: Int = 0

//   /// Number of `.last` steps. Note that this will not count for episodes that are not terminated 
//   /// with a `.last` step.
//   public var numEpisodes: Int = 0

//   /// Number of steps in the current episode.
//   public var numEpisodeSteps: Int = 0

//   /// Total number of steps, ignoring `.first` steps.
//   public var numTotalSteps: Int = 0

//   public init(wrapping environment: WrappedEnvironment) {
//     precondition(environment.batched == false, "The wrapped environment must not be batched.")
//     self.wrappedEnvironment = environment
//   }

//   @inlinable
//   public mutating func step(taking action: Action) -> Step<Observation, Reward> {
//     let result = wrappedEnvironment.step(taking: action)

//     if result.kind.isFirst().scalarized() {
//       numResets += 1
//       numEpisodeSteps = 0
//     } else {
//       numEpisodeSteps += 1
//       numTotalSteps += 1
//     }

//     if result.kind.isLast().scalarized() {
//       numEpisodes += 1
//     }

//     return result
//   }

//   @inlinable
//   public mutating func reset() -> Step<Observation, Reward> {
//     numResets += 1
//     numEpisodeSteps = 0
//     return wrappedEnvironment.reset()
//   }

//   @inlinable
//   public func copy() -> RunStatistics<WrappedEnvironment> {
//     RunStatistics(wrapping: wrappedEnvironment.copy())
//   }
// }
