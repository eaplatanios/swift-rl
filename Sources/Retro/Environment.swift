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
import Gzip
import ReinforcementLearning
import TensorFlow

public final class RetroEnvironment<ActionsType: Retro.ActionsType>: Environment {
  public let batchSize: Int
  public let emulators: [RetroEmulator]
  public let actionsType: ActionsType
  public let actionSpace: ActionsType.Space
  public let observationsType: ObservationsType
  public let observationSpace: Box<Float> // TODO: Use UInt8 for large replay buffers.
  public let startingStates: [StartingState]
  public let randomSeed: TensorFlowSeed

  @usableFromInline internal let startingStateData: [String?]
  @usableFromInline internal var movies: [Movie?]
  @usableFromInline internal var movieIDs: [Int]
  @usableFromInline internal var movieURLs: [URL?]
  @usableFromInline internal var needsReset: [Bool]
  @usableFromInline internal var step: Step<Tensor<Float>, Tensor<Float>>? = nil
  @usableFromInline internal var renderer: ImageRenderer? = nil

  @inlinable
  public var currentStep: Step<Tensor<Float>, Tensor<Float>> {
    if step == nil { step = reset() }
    return step!
  }

  @inlinable
  public convenience init(
    using emulator: RetroEmulator,
    actionsType: ActionsType,
    observationsType: ObservationsType = .screen(height: 84, width: 84, grayscale: true),
    startingState: StartingState? = nil,
    movieURL: URL? = nil,
    renderer: ImageRenderer? = nil,
    randomSeed: TensorFlowSeed = Context.local.randomSeed
  ) throws {
    try self.init(
      using: [emulator],
      actionsType: actionsType,
      observationsType: observationsType,
      startingStates: startingState == nil ? nil : [startingState!],
      movieURLs: [movieURL],
      renderer: renderer,
      randomSeed: randomSeed)
  }

  @inlinable
  public init(
    using emulators: [RetroEmulator],
    actionsType: ActionsType,
    observationsType: ObservationsType = .screen(height: 84, width: 84, grayscale: true),
    startingStates: [StartingState]? = nil,
    movieURLs: [URL?]? = nil,
    renderer: ImageRenderer? = nil,
    randomSeed: TensorFlowSeed = Context.local.randomSeed
  ) throws {
    precondition(emulators.count > 0, "At least one emulator must be provided.")
    self.batchSize = emulators.count
    self.emulators = emulators
    self.actionsType = actionsType
    self.actionSpace = actionsType.space(for: emulators[0], batchSize: batchSize)
    self.observationsType = observationsType
    switch observationsType {
    case let .screen(height, width, grayscale):
      self.observationSpace = Box<Float>(
        shape: TensorShape([height, width, grayscale ? 1 : 3]),
        lowerBound: 0,
        upperBound: 1)
    case .memory:
      self.observationSpace = Box<Float>(
        shape: emulators[0].memory()!.shape,
        lowerBound: 0,
        upperBound: 1)
    }
    self.renderer = renderer
    self.randomSeed = randomSeed
    self.movies = [Movie?](repeating: nil, count: batchSize)
    self.movieIDs = [Int](repeating: 0, count: batchSize)
    if let urls = movieURLs {
      self.movieURLs = urls
    } else {
      self.movieURLs = [URL?](repeating: nil, count: batchSize)
    }

    // Determine the starting state for each emulator.
    if let states = startingStates {
      self.startingStates = states
    } else {
      self.startingStates = [StartingState](repeating: .provided, count: batchSize)
    }
    var startingStateData = [String?]()
    for i in 0..<batchSize {
      var stateData: String? = nil
      switch self.startingStates[i] {
      case .none:
        stateData = nil
      case .provided:
        let gameMetadataJson = try? String(contentsOf: emulators[i].game.metadataFile!)
        let gameMetadata = try? RetroGame.Metadata(fromJson: gameMetadataJson!)
        if let metadata = gameMetadata {
          let defaultState = metadata.defaultState
          let defaultPlayerState = metadata.defaultPlayerState
          if defaultPlayerState != nil && emulators[i].numPlayers <= defaultPlayerState!.count {
            stateData = defaultPlayerState![Int(emulators[i].numPlayers) - 1]
          } else if defaultState != nil {
            stateData = defaultState!
          } else {
            stateData = nil
          }
        } else {
          stateData = nil
        }
      case .custom(let state):
        stateData = state
      }
      startingStateData.append(stateData)

      if let state = stateData {
        let statePath = emulators[i].game.dataDir.appendingPathComponent("\(state).state")
        try emulators[i].loadStartingState(from: statePath)
      }
    }
    self.startingStateData = startingStateData
    self.needsReset = [Bool](repeating: true, count: batchSize)
  }

  @inlinable
  internal func currentObservation(batchIndex: Int) -> Tensor<Float> {
    switch observationsType {
    case let .screen(height, width, true):
      let emulatorScreen = Tensor<Float>(emulators[batchIndex].screen()!) / 255.0
      return resize(
        images: convertRGBToGrayscale(emulatorScreen),
        to: Tensor<Int32>([Int32(height), Int32(width)]),
        method: .area)
    case let .screen(height, width, false):
      let emulatorScreen = Tensor<Float>(emulators[batchIndex].screen()!) / 255.0
      return resize(
        images: emulatorScreen,
        to: Tensor<Int32>([Int32(height), Int32(width)]),
        method: .area)
    case .memory:
      return Tensor<Float>(emulators[batchIndex].memory()!) / 255.0
    }
  }

  @inlinable
  @discardableResult
  public func step(taking action: ActionsType.Space.Value) -> Step<Tensor<Float>, Tensor<Float>> {
    let actions = action.unstacked()
    step = Step<Tensor<Float>, Tensor<Float>>.stack((0..<batchSize).map {
      step(taking: actions[$0], batchIndex: $0)
    })
    return step!
  }

  @inlinable
  @discardableResult
  public func step(
    taking action: ActionsType.Space.Value,
    batchIndex: Int
  ) -> Step<Tensor<Float>, Tensor<Float>> {
    if needsReset[batchIndex] { reset(batchIndex: batchIndex) }
    for p in 0..<numPlayers(batchIndex: batchIndex) {
      let numButtons = emulators[batchIndex].buttons().count
      let encodedAction = actionsType.encodeAction(action, for: p, in: emulators[batchIndex])
      var buttonMask = [UInt8](repeating: 0, count: numButtons)
      for i in 0..<numButtons {
        buttonMask[i] = UInt8((encodedAction >> i) & 1)
        movies[batchIndex]?[i, forPlayer: p] = buttonMask[i] > 0
      }
      emulators[batchIndex].setButtonMask(for: p, to: buttonMask)
    }

    movies[batchIndex]?.step()
    emulators[batchIndex].step()

    if emulators[batchIndex].finished() {
      needsReset[batchIndex] = true
    }

    let observation = currentObservation(batchIndex: batchIndex)
    let finished = emulators[batchIndex].finished()

    // TODO: What about the 'info' dict?
    let numPlayers = self.numPlayers(batchIndex: batchIndex)
    return Step(
      kind: finished ? .last() : .transition(),
      observation: observation,
      reward: Tensor<Float>((0..<numPlayers).map { emulators[batchIndex].reward(for: $0) }))
  }

  @inlinable
  @discardableResult
  public func reset() -> Step<Tensor<Float>, Tensor<Float>> {
    step = Step<Tensor<Float>, Tensor<Float>>.stack((0..<batchSize).map { reset(batchIndex: $0) })
    return step!
  }

  @inlinable
  @discardableResult
  public func reset(batchIndex: Int) -> Step<Tensor<Float>, Tensor<Float>> {
    emulators[batchIndex].reset()

    // Reset the recording.
    if let url = movieURLs[batchIndex] {
      let game = emulators[batchIndex].game
      let state = String(startingStateData[batchIndex]?.split(separator: ".")[0] ?? "none")
      let movieFilename = "\(game)-\(state)-\(String(format: "%06d", movieIDs[batchIndex])).bk2"
      startRecording(at: url.appendingPathComponent(movieFilename), batchIndex: batchIndex)
      movieIDs[batchIndex] += 1
    }

    movies[batchIndex]?.step()

    let observation = currentObservation(batchIndex: batchIndex)
    let numPlayers = emulators[batchIndex].numPlayers
    let reward = Tensor<Float>((0..<numPlayers).map { emulators[batchIndex].reward(for: $0) })
    needsReset[batchIndex] = false
    return Step(kind: .first(), observation: observation, reward: reward)
  }

  @inlinable
  public func copy() -> RetroEnvironment<ActionsType> {
    try! RetroEnvironment(
      using: emulators.map { try $0.copy() },
      actionsType: actionsType,
      observationsType: observationsType,
      startingStates: startingStates,
      movieURLs: movieURLs,
      renderer: renderer,
      randomSeed: randomSeed)
  }

  @inlinable
  public func render() throws {
    if renderer == nil { renderer = ImageRenderer() }
    let observation = currentStep.observation
    switch observationsType {
    case let .screen(height, width, true):
      try renderer!.render(
        Tensor<UInt8>(255 * observation
          .reshaped(to: [height, width, 1])
          .tiled(multiples: Tensor<Int32>([1, 1, 3]))).array)
    case let .screen(height, width, false):
      try renderer!.render(
        Tensor<UInt8>(255 * observation.reshaped(to: [height, width, 3])).array)
    case .memory:
      let size = observation.shape.contiguousSize
      try renderer!.render(
        Tensor<UInt8>(255 * observation
          .reshaped(to: [size / 2, size / 2, 1])
          .tiled(multiples: Tensor<Int32>([1, 1, 3]))).array)
    }
  }

  @inlinable
  public func startRecordings(at urls: [URL]) {
    for batchIndex in 0..<batchSize {
      startRecording(at: urls[batchIndex], batchIndex: batchIndex)
    }
  }

  @inlinable
  public func startRecording(at url: URL, batchIndex: Int) {
    let numPlayers = emulators[batchIndex].numPlayers
    movies[batchIndex] = Movie(at: url, recording: true, numPlayers: numPlayers)
    movies[batchIndex]!.configure(for: self, batchIndex: batchIndex)
    if let state = emulators[batchIndex].startingStateData {
      movies[batchIndex]!.state = state
    }
  }

  @inlinable
  public func enableRecordings(at urls: [URL]) {
    movieURLs = urls
  }

  @inlinable
  public func enableRecording(at url: URL, batchIndex: Int) {
    movieURLs[batchIndex] = url
  }

  @inlinable
  public func disableRecordings() {
    for batchIndex in 0..<batchSize {
      disableRecording(batchIndex: batchIndex)
    }
  }

  @inlinable
  public func disableRecording(batchIndex: Int) {
    movieIDs[batchIndex] = 0
    movieURLs[batchIndex] = nil
    movies[batchIndex]?.close()
    movies[batchIndex] = nil
  }

  @inlinable
  public func games() -> [RetroGame] {
    emulators.map { $0.game }
  }

  @inlinable
  public func game(batchIndex: Int) -> RetroGame {
    emulators[batchIndex].game
  }

  @inlinable
  public func numPlayers() -> [UInt32] {
    emulators.map { $0.numPlayers }
  }

  @inlinable
  public func numPlayers(batchIndex: Int) -> UInt32 {
    emulators[batchIndex].numPlayers
  }
}

public extension RetroEnvironment {
  /// Represents the initial state of the emulator.
  enum StartingState {
    /// Start the game at the power on screen of the emulator.
    case none
    
    /// Start the game at the default save state from `metadata.json`.
    case provided

    /// Start the game from the save state file specified.
    /// The provided string is the name of the `.state` file to use.
    case custom(String)
  }
}

/// Represents different settings for the observation space of the environment.
public enum ObservationsType {
  /// Use RGB image observations.
  case screen(height: Int, width: Int, grayscale: Bool)

  /// Use RAM observations where you can see the memory of the game instead of the screen.
  case memory
}

/// Represents different types of action space for the environment.
public protocol ActionsType {
  associatedtype Scalar: TensorFlowScalar
  associatedtype Space: ReinforcementLearning.Space where Space.Value == Tensor<Scalar>

  func space(for emulator: RetroEmulator, batchSize: Int) -> Space

  func encodeAction(
    _ action: Tensor<Scalar>,
    for player: UInt32,
    in emulator: RetroEmulator
  ) -> UInt16
}

/// Multi-binary action space with no filtered actions.
public struct FullActions: ActionsType {
  public typealias Scalar = Int32
  public typealias Space = MultiBinary

  public init() {}

  public func space(for emulator: RetroEmulator, batchSize: Int) -> MultiBinary {
    MultiBinary(
      withSize: emulator.buttons().count * Int(emulator.numPlayers),
      batchSize: batchSize)
  }

  public func encodeAction(
    _ action: Tensor<Int32>,
    for player: UInt32,
    in emulator: RetroEmulator
  ) -> UInt16 {
    let startIndex = emulator.buttons().count * Int(player)
    let endIndex = emulator.buttons().count * Int(player + 1)
    let playerAction = action[startIndex..<endIndex].scalars
    var encodedAction = UInt16(0)
    for i in 0..<playerAction.count {
      encodedAction |= UInt16(playerAction[i]) << i
    }
    return encodedAction
  }
}

/// Multi-binary action space with invalid or not allowed actions filtered out.
public struct FilteredActions: ActionsType {
  public typealias Scalar = Int32
  public typealias Space = MultiBinary

  public init() {}

  public func space(for emulator: RetroEmulator, batchSize: Int) -> MultiBinary {
    MultiBinary(
      withSize: emulator.buttons().count * Int(emulator.numPlayers),
      batchSize: batchSize)
  }

  public func encodeAction(
    _ action: Tensor<Int32>,
    for player: UInt32,
    in emulator: RetroEmulator
  ) -> UInt16 {
    let startIndex = emulator.buttons().count * Int(player)
    let endIndex = emulator.buttons().count * Int(player + 1)
    let playerAction = action[startIndex..<endIndex].scalars
    var encodedAction = UInt16(0)
    for i in 0..<playerAction.count {
      encodedAction |= UInt16(playerAction[i]) << i
    }
    return gameDataFilterAction(emulator.gameData.handle, encodedAction)
  }
}

/// Discrete action space for filtered actions.
public struct DiscreteActions: ActionsType {
  public typealias Scalar = Int32
  public typealias Space = Discrete

  public init() {}

  public func space(for emulator: RetroEmulator, batchSize: Int) -> Discrete {
    let numCombos = emulator.buttonCombos().map { Int32($0.count) } .reduce(1, *)
    return Discrete(
      withSize: Int(pow(Float(numCombos), Float(emulator.numPlayers))),
      batchSize: batchSize)
  }

  public func encodeAction(
    _ action: Tensor<Int32>,
    for player: UInt32,
    in emulator: RetroEmulator
  ) -> UInt16 {
    var playerAction = UInt16(action.scalarized())
    var encodedAction = UInt16(0)
    var current = 0
    for combo in emulator.buttonCombos() {
      current = Int(playerAction) % combo.count
      playerAction /= UInt16(combo.count)
      encodedAction |= UInt16(combo[current])
    }
    return encodedAction
  }
}

/// Multi-discrete action space for filtered actions.
public struct MultiDiscreteActions: ActionsType {
  public typealias Scalar = Int32
  public typealias Space = MultiDiscrete

  public init() {}

  public func space(for emulator: RetroEmulator, batchSize: Int) -> MultiDiscrete {
    MultiDiscrete(
      withSizes: emulator.buttonCombos().map { $0.count * Int(emulator.numPlayers) },
      batchSize: batchSize)
  }

  public func encodeAction(
    _ action: Tensor<Int32>,
    for player: UInt32,
    in emulator: RetroEmulator
  ) -> UInt16 {
    let startIndex = emulator.buttons().count * Int(player)
    let endIndex = emulator.buttons().count * Int(player + 1)
    let playerAction = action[startIndex..<endIndex].scalars
    var encodedAction = UInt16(0)
    for i in 0..<playerAction.count {
      let combo = emulator.buttonCombos()[i]
      encodedAction |= UInt16(combo[Int(playerAction[i])])
    }
    return encodedAction
  }
}
