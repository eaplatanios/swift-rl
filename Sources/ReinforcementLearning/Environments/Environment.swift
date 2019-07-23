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

import Foundation
import TensorFlow

public protocol Environment: AnyObject {
  associatedtype ObservationSpace: Space
  associatedtype ActionSpace: Space
  associatedtype Reward

  var batchSize: Int { get }
  var observationSpace: ObservationSpace { get }
  var actionSpace: ActionSpace { get }

  /// Result of the last step taken in this environment (i.e., its current state).
  var currentStep: Step<Observation, Reward> { get }

  /// Updates the environment according to the provided action.
  @discardableResult
  func step(taking action: Action) -> Step<Observation, Reward>

  /// Resets the environment.
  @discardableResult
  func reset() -> Step<Observation, Reward>

  /// Returns a copy of this environment that is reset before being returned.
  func copy() -> Self
}

public extension Environment {
  typealias Observation = ObservationSpace.Value
  typealias Action = ActionSpace.Value
}

public protocol RenderableEnvironment: Environment {
  func render() throws
}

/// Contains the data emitted by an environment at a single step of interaction.
public struct Step<Observation, Reward>: KeyPathIterable {
  // TODO: Make `internal(set)` once `@usableFromInline` is supported.
  public var kind: StepKind
  public var observation: Observation
  public var reward: Reward

  @inlinable
  public init(kind: StepKind, observation: Observation, reward: Reward) {
    self.kind = kind
    self.observation = observation
    self.reward = reward
  }

  @inlinable
  public func copy(
    kind: StepKind? = nil,
    observation: Observation? = nil,
    reward: Reward? = nil
  ) -> Step<Observation, Reward> {
    Step(
      kind: kind ?? self.kind,
      observation: observation ?? self.observation,
      reward: reward ?? self.reward)
  }
}

/// Represents the type of a step.
public struct StepKind: KeyPathIterable {
  // TODO: Make `internal(set)` once `@usableFromInline` is supported.
  public var rawValue: Tensor<Int32>

  @inlinable
  public init(_ rawValue: Tensor<Int32>) {
    self.rawValue = rawValue
  }
}

extension StepKind {
  /// Denotes the first step in a sequence.
  @inlinable
  public static func first() -> StepKind {
    StepKind(Tensor<Int32>(0))
  }

  /// Denotes an transition step in a sequence (i.e., not first or last).
  @inlinable
  public static func transition() -> StepKind {
    StepKind(Tensor<Int32>(1))
  }

  /// Denotes the last step in a sequence.
  @inlinable
  public static func last(withReset: Bool = true) -> StepKind {
    StepKind(Tensor<Int32>(withReset ? 3 : 2))
  }

  /// Returns a batched `StepKind` filled with "first" step kind values.
  @inlinable
  public static func first(batchSize: Int) -> StepKind {
    StepKind(first().rawValue.expandingShape(at: 0)
      .tiled(multiples: Tensor<Int32>([Int32(batchSize)])))
  }

  /// Returns a batched `StepKind` filled with "transition" step kind values.
  @inlinable
  public static func transition(batchSize: Int) -> StepKind {
    StepKind(transition().rawValue.expandingShape(at: 0)
      .tiled(multiples: Tensor<Int32>([Int32(batchSize)])))
  }

  /// Returns a batched `StepKind` filled with "last" step kind values.
  @inlinable
  public static func last(batchSize: Int, withReset: Bool = true) -> StepKind {
    StepKind(last(withReset: withReset).rawValue.expandingShape(at: 0)
      .tiled(multiples: Tensor<Int32>([Int32(batchSize)])))
  }

  @inlinable
  public func isFirst() -> Tensor<Bool> {
    rawValue .== 0
  }

  @inlinable
  public func isTransition() -> Tensor<Bool> {
    rawValue .== 1
  }

  @inlinable
  public func isLast(withReset: Bool = false) -> Tensor<Bool> {
    withReset ? rawValue .== 3 : (rawValue .== 2).elementsLogicalOr(rawValue .== 3)
  }

  /// Returns a tensor containing the number of completed episodes contained in the trajectory
  /// that this step kind corresponds to.
  @inlinable
  public func episodeCount(withReset: Bool = false) -> Tensor<Float> {
    Tensor<Float>(isLast(withReset: withReset)).sum()
  }

  /// Returns a boolean tensor whose `false`-valued elements correspond to steps of episodes that
  /// did not complete by the end of the trajectory that this step kind corresponds to.
  @inlinable
  public func completeEpisodeMask(withReset: Bool = false) -> Tensor<Bool> {
    Tensor<Float>(isLast(withReset: withReset)).cumulativeSum(alongAxis: 0, reverse: true) .> 0
  }
}
