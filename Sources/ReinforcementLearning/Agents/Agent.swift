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

public protocol Agent {
  associatedtype Policy: ProbabilisticPolicy

  /// Policy used to act by this agent.
  var policy: Policy { get }

  /// Initializes this agent.
  mutating func initialize()

  /// Trains this agent using the provided experience.
  /// - Returns: Loss function value.
  @discardableResult
  mutating func update(using trajectory: Trajectory<Action, Observation, Reward, State>) -> Float
}

public extension Agent {
  typealias Action = Policy.Action
  typealias Observation = Policy.Observation
  typealias Reward = Policy.Reward
  typealias State = Policy.State
}
