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

import CRetro
import Foundation

public class Movie {
  private var handle: UnsafeMutablePointer<CMovie>?

  public init(at url: URL, recording: Bool, numPlayers: UInt32) {
    self.handle = movieCreate(url.path, recording, numPlayers)
  }

  deinit {
    movieDelete(handle)
  }

  public var state: Data {
    get {
      let cBytesPointer = movieGetState(handle)!
      defer { cBytesPointer.deallocate() }
      let cBytes = cBytesPointer.pointee
      return Data(bytes: cBytes.bytes, count: cBytes.numBytes)
    }
    set (newValue) {
      let bytes = [UInt8](newValue)
      bytes.withUnsafeBufferPointer {
        var cBytesStruct = CBytes(bytes: $0.baseAddress, numBytes: bytes.count)
        movieSetState(handle, &cBytesStruct)
      }
    }
  }

  public func configure<ActionsType: Retro.ActionsType>(
    for environment: RetroEnvironment<ActionsType>,
    batchIndex: Int
  ) {
    movieConfigure(
      handle,
      environment.game(batchIndex: batchIndex).name,
      environment.emulators[batchIndex].handle)
  }

  public func game() -> String {
    let cString = movieGetGameName(handle)!
    defer { cString.deallocate() }
    return String(cString: cString)
  }

  @discardableResult
  public func step() -> Bool {
    movieStep(handle)
  }

  public func close() {
    movieClose(handle)
  }

  public func numPlayers() -> UInt32 {
    moviePlayers(handle)
  }

  public subscript(key: Int, forPlayer player: UInt32) -> Bool {
    get { movieGetKey(handle, Int32(key), player) }
    set { movieSetKey(handle, Int32(key), newValue, player) }
  }
}
