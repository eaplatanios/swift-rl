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

public protocol Environment {
  associatedtype ActionSpace: Space
  associatedtype ObservationSpace: Space
  associatedtype Reward

  var batchSize: Int { get }
  var actionSpace: ActionSpace { get }
  var observationSpace: ObservationSpace { get }

  /// Returns the result of the last step taken in this environment (i.e., its current state).
  func currentStep() -> Step<Observation, Reward>

  /// Updates the environment according to the provided action.
  @discardableResult
  mutating func step(taking action: Action) -> Step<Observation, Reward>

  /// Resets the environment.
  @discardableResult
  mutating func reset() -> Step<Observation, Reward>

  /// Returns a copy of this environment that is reset before being returned.
  func copy() -> Self
}

public extension Environment {
  typealias Action = ActionSpace.Value
  typealias Observation = ObservationSpace.Value

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
public struct Step<Observation, Reward>: KeyPathIterable {
  // TODO: Make `internal(set)` once `@usableFromInline` is supported.
  public var kind: StepKind
  public var observation: Observation
  public var reward: Reward

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

  public init(_ rawValue: Tensor<Int32>) {
    self.rawValue = rawValue
  }
}

extension StepKind {
  /// Denotes the first step in a sequence.
  public static let first = StepKind(Tensor<Int32>(0))

  /// Denotes an transition step in a sequence (i.e., not first or last).
  public static let transition = StepKind(Tensor<Int32>(1))

  /// Denotes the last step in a sequence.
  public static let last = StepKind(Tensor<Int32>(2))

  /// Returns a batched `StepKind` filled with "first" step kind values.
  public static func first(batchSize: Int) -> StepKind {
    StepKind(first.rawValue.expandingShape(at: 0)
      .tiled(multiples: Tensor<Int32>([Int32(batchSize)])))
  }

  /// Returns a batched `StepKind` filled with "transition" step kind values.
  public static func transition(batchSize: Int) -> StepKind {
    StepKind(transition.rawValue.expandingShape(at: 0)
      .tiled(multiples: Tensor<Int32>([Int32(batchSize)])))
  }

  /// Returns a batched `StepKind` filled with "last" step kind values.
  public static func last(batchSize: Int) -> StepKind {
    StepKind(last.rawValue.expandingShape(at: 0)
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
  public func isLast() -> Tensor<Bool> {
    rawValue .== 2
  }

  /// Returns a tensor containing the number of completed episodes contained in the trajectory
  /// that this step kind corresponds to.
  public func episodeCount() -> Tensor<Float> {
    Tensor<Float>(isLast()).sum()
  }

  /// Returns a boolean tensor whose `false`-valued elements correspond to steps of episodes that
  /// did not complete by the end of the trajectory that this step kind corresponds to.
  @inlinable
  public func completeEpisodeMask() -> Tensor<Bool> {
    Tensor<Float>(self.isLast()).cumulativeSum(alongAxis: 0, reverse: true) .> 0
  }
}
