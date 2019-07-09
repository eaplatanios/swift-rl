import CRetro
import Foundation
import Gzip
import ReinforcementLearning
import TensorFlow

public struct RetroEnvironment<ActionsType: Retro.ActionsType>: Environment {
  public let batched: Bool = false

  public let emulator: RetroEmulator
  public let actionsType: ActionsType
  public let actionSpace: ActionsType.Space
  public let observationsType: ObservationsType
  public let observationSpace: DiscreteBox<UInt8>
  public let startingState: StartingState
  public let randomSeed: TensorFlowSeed

  private let startingStateData: String?

  public private(set) var movie: Movie?
  public private(set) var movieID: Int
  public private(set) var movieURL: URL?

  private var needsReset: Bool = true

  public init(
    using emulator: RetroEmulator,
    actionsType: ActionsType,
    observationsType: ObservationsType = .screen,
    startingState: StartingState = .provided,
    movieURL: URL? = nil,
    randomSeed: TensorFlowSeed = Context.local.randomSeed
  ) throws {
    self.emulator = emulator
    self.actionsType = actionsType
    self.actionSpace = actionsType.space(for: emulator)
    self.observationsType = observationsType
    switch observationsType {
    case .screen: self.observationSpace = DiscreteBox(
      shape: emulator.screen()!.shape, lowerBound: 0, upperBound: 255)
    case .memory: self.observationSpace = DiscreteBox(
      shape: emulator.memory()!.shape, lowerBound: 0, upperBound: 255)
    }
    self.randomSeed = randomSeed
    self.movie = nil
    self.movieID = 0
    self.movieURL = movieURL

    self.startingState = startingState
    switch startingState {
    case .none:
      self.startingStateData = nil
    case .provided:
      let gameMetadataJson = try? String(contentsOf: self.emulator.game.metadataFile!)
      let gameMetadata = try? RetroGame.Metadata(fromJson: gameMetadataJson!)
      if let metadata = gameMetadata {
        let defaultState = metadata.defaultState
        let defaultPlayerState = metadata.defaultPlayerState
        if defaultPlayerState != nil && emulator.numPlayers <= defaultPlayerState!.count {
          self.startingStateData = defaultPlayerState![Int(emulator.numPlayers) - 1]
        } else if defaultState != nil {
          self.startingStateData = defaultState!
        } else {
          self.startingStateData = nil
        }
      } else {
        self.startingStateData = nil
      }
    case .custom(let state):
      self.startingStateData = state
    }

    if let state = self.startingStateData {
      try self.emulator.loadStartingState(
        from: game().dataDir.appendingPathComponent("\(state).state"))
    }
  }

  @discardableResult
  public mutating func step(
    taking action: ActionsType.Space.Value
  ) -> Step<Tensor<UInt8>, Tensor<Float>> {
    if needsReset {
      return reset()
    }

    for p in 0..<numPlayers() {
      let numButtons = emulator.buttons().count
      let encodedAction = actionsType.encodeAction(action, for: p, in: emulator)
      var buttonMask = [UInt8](repeating: 0, count: numButtons)
      for i in 0..<numButtons {
        buttonMask[i] = UInt8((encodedAction >> i) & 1)
        movie?[i, forPlayer: p] = buttonMask[i] > 0
      }
      emulator.setButtonMask(for: p, to: buttonMask)
    }

    movie?.step()
    emulator.step()

    let observation: Tensor<UInt8>? = {
      switch observationsType {
      case .screen: return emulator.screen()
      case .memory: return emulator.memory()
      }
    }()
    let finished = emulator.finished()

    if finished {
      needsReset = true
    }

    // TODO: What about the 'info' dict?
    return Step(
      kind: finished ? .last : .transition,
      observation: observation!,
      reward: Tensor<Float>((0..<numPlayers()).map { emulator.reward(for: $0) }))
  }

  @discardableResult
  public mutating func reset() -> Step<Tensor<UInt8>, Tensor<Float>> {
    emulator.reset()

    // Reset the recording.
    if let url = movieURL {
      let state = String(startingStateData?.split(separator: ".")[0] ?? "none")
      let movieFilename = "\(game())-\(state)-\(String(format: "%06d", movieID)).bk2"
      startRecording(at: url.appendingPathComponent(movieFilename))
      movieID += 1
    }

    movie?.step()

    let observation: Tensor<UInt8>? = {
      switch observationsType {
      case .screen: return emulator.screen()
      case .memory: return emulator.memory()
      }
    }()
    let reward = Tensor<Float>((0..<numPlayers()).map { emulator.reward(for: $0) })

    needsReset = false
    return Step(kind: .first, observation: observation!, reward: reward)
  }

  @inlinable
  public func copy() throws -> RetroEnvironment<ActionsType> {
    try RetroEnvironment(
      using: emulator.copy(),
      actionsType: actionsType,
      observationsType: observationsType,
      startingState: startingState,
      movieURL: movieURL,
      randomSeed: randomSeed)
  }

  public mutating func startRecording(at url: URL) {
    movie = Movie(at: url, recording: true, numPlayers: emulator.numPlayers)
    movie!.configure(for: self)
    if let state = emulator.startingStateData {
      movie!.state = state
    }
  }

  public mutating func enableRecording(at url: URL) {
    movieURL = url
  }

  public mutating func disableRecording() {
    movieID = 0
    movieURL = nil
    if let m = movie {
      m.close()
      movie = nil
    }
  }

  @inlinable
  public func game() -> RetroGame {
    return emulator.game
  }

  @inlinable
  public func numPlayers() -> UInt32 {
    return emulator.numPlayers
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

  struct StepResult {
    let observation: Tensor<UInt8>?
    let reward: Tensor<Float>
    let finished: Bool
  }
}

/// Represents different settings for the observation space of the environment.
public enum ObservationsType: Int {
  /// Use RGB image observations.
  case screen

  /// Use RAM observations where you can see the memory of the game instead of the screen.
  case memory
}

/// Represents different types of action space for the environment.
public protocol ActionsType {
  associatedtype Scalar: TensorFlowScalar
  associatedtype Space: ReinforcementLearning.Space where Space.Value == Tensor<Scalar>

  func space(for emulator: RetroEmulator) -> Space

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

  public func space(for emulator: RetroEmulator) -> MultiBinary {
    return MultiBinary(withSize: emulator.buttons().count * Int(emulator.numPlayers))
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

  public func space(for emulator: RetroEmulator) -> MultiBinary {
    return MultiBinary(withSize: emulator.buttons().count * Int(emulator.numPlayers))
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

  public func space(for emulator: RetroEmulator) -> Discrete {
    let numCombos = emulator.buttonCombos().map { Int32($0.count) } .reduce(1, *)
    return Discrete(withSize: Int(pow(Float(numCombos), Float(emulator.numPlayers))))
  }

  public func encodeAction(
    _ action: Tensor<Int32>,
    for player: UInt32,
    in emulator: RetroEmulator
  ) -> UInt16 {
    var playerAction = UInt16(action.scalar!)
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

  public func space(for emulator: RetroEmulator) -> MultiDiscrete {
    return MultiDiscrete(withSizes: emulator.buttonCombos().map {
      $0.count * Int(emulator.numPlayers)
    })
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
