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

public protocol ProbabilisticPolicy: Policy {
  associatedtype ActionDistribution: Distribution where ActionDistribution.Value == Action

  var randomSeed: TensorFlowSeed { get }

  /// Generates the distribution over next actions given the current environment step.
  func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution
}

public extension ProbabilisticPolicy {
  func action(for step: Step<Observation, Reward>) -> Action {
    let distribution = actionDistribution(for: step)
    return distribution.sample(usingSeed: randomSeed)
  }
}
