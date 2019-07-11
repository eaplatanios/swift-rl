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

import TensorFlow

public protocol Stackable {
  associatedtype Stacked

  static func stack(_ values: [Self]) -> Stacked
  func unstacked() -> [Self]
}

extension Tensor: Stackable {
  public typealias Stacked = Tensor

  public static func stack(_ values: [Tensor]) -> Tensor {
    Tensor(stacking: values, alongAxis: 0)
  }

  public func unstacked() -> [Tensor] {
    unstacked(alongAxis: 0)
  }
}

// TODO: Array
// TODO: ShapedArray
