import CRetro
import Foundation
import Gzip
import TensorFlow

/// Represents the initial state of the emulator.
public enum State {
  /// Start the game at the power on screen of the emulator.
  case none
  
  /// Start the game at the default save state from `metadata.json`.
  case provided

  /// Start the game from the save state file specified.
  /// The provided string is the name of the `.state` file to use.
  case custom(String)
}

public struct EnvironmentStepResult {
  public let observation: ShapedArray<UInt8>
  public let reward: [Float]
  public let finished: Bool
}

public class Environment<A: RetroActions> {
  public let config: EmulatorConfig<A>
  public let game: String
  public let numPlayers: UInt32
  public let core: String
  public let buttons: [String?]
  public let actionSpace: A.ActionSpace
  public let observationSpace: Box<UInt8>

  public private(set) var rng: SeedableRandomNumberGenerator
  public private(set) var gameData: GameData
  public private(set) var gameDataFile: URL?
  public private(set) var gameScenarioFile: URL?
  public private(set) var movie: Movie?
  public private(set) var movieID: Int
  public private(set) var movieURL: URL?

  public private(set) var memory: ShapedArray<UInt8>? = nil
  public private(set) var screen: ShapedArray<UInt8>? = nil

  internal var emulatorHandle: UnsafeMutablePointer<CEmulator>?

  private var state: String? = nil
  private var initialState: Data? = nil

  public init(
    withConfig config: EmulatorConfig<A>,
    for game: String,
    in state: State = .provided,
    using integration: GameIntegration = .stable,
    numPlayers: UInt32 = 1,
    gameDataFile: String = "data.json",
    gameScenarioFile: String = "scenario.json",
    randomSeed: UInt64? = nil
   ) throws {
    let gameROMFile = try config.gameROMFile(for: game, using: integration)
    let gameMetadataFile = config.gameFile("metadata.json", for: game, using: integration)
    let seed = hashSeed(createSeed(using: randomSeed))

    self.config = config
    self.game = game
    self.numPlayers = numPlayers
    self.core = try getCore(forROM: gameROMFile)
    self.rng = PhiloxRandomNumberGenerator(seed: seed)
    self.gameData = try GameData(withConfig: config, for: game, using: integration)
    self.gameDataFile = config.gameFile(gameDataFile, for: game, using: integration)
    self.gameScenarioFile = config.gameFile(gameScenarioFile, for: game, using: integration)
    self.movie = nil
    self.movieID = 0
    self.movieURL = config.movieURL

    switch state {
    case .none:
      self.state = nil
    case .provided:
      let gameMetadataJson = try? String(contentsOf: gameMetadataFile!)
      let gameMetadata = try? GameMetadata(fromJson: gameMetadataJson!)
      if let metadata = gameMetadata {
        if metadata.defaultPlayerState != nil && numPlayers <= metadata.defaultPlayerState!.count {
          self.state = metadata.defaultPlayerState![Int(numPlayers) - 1]
        } else if metadata.defaultState != nil {
          self.state = metadata.defaultState!
        } else {
          self.state = nil
        }
      } else {
        self.state = nil
      }
    case .custom(let state):
      self.state = state
    }

    let coreInformation = try getInformation(forCore: self.core)
    if !self.gameData.load(dataFile: self.gameDataFile, scenarioFile: self.gameScenarioFile) {
      throw RetroError.GameDataFailure(
        message: "Failed to load game data from '\(gameDataFile)'" + 
                 "or game scenario from '\(gameScenarioFile)'.")
    }

    self.emulatorHandle = emulatorCreate(gameROMFile.path)
    emulatorConfigureData(self.emulatorHandle, self.gameData.handle)
    emulatorStep(self.emulatorHandle)

    self.buttons = coreInformation.buttons
    self.actionSpace = config.actionSpaceType.space(
      for: self.gameData, buttons: self.buttons, numPlayers: numPlayers)

    // Configure the observation space.
    switch config.observationSpaceType {
    case .screen:
      self.screen = Environment.getScreen(
        gameData: self.gameData,
        emulatorHandle: self.emulatorHandle,
        forPlayer: 0)
      self.observationSpace = Box(low: 0, high: 255, shape: screen!.shape)
    case .memory:
      self.memory = Environment.getMemory(gameData: self.gameData)
      self.observationSpace = Box(low: 0, high: 255, shape: memory!.shape)
    }

    if let state = self.state {
      try loadState(named: state, using: integration)
    }
  }

  deinit {
    emulatorDelete(emulatorHandle)
  }

  public func seed(using seed: UInt64? = nil) -> UInt64 {
    let strongSeed = hashSeed(createSeed(using: seed))
    self.rng = PhiloxRandomNumberGenerator(seed: strongSeed)
    return strongSeed
  }

  public func step(taking action: ShapedArray<A.ActionSpace.Scalar>) -> EnvironmentStepResult {
    for p in 0..<numPlayers {
      let encodedAction = config.actionSpaceType.encodeAction(
        action, for: gameData, buttons: buttons, player: p)
      var playerAction = [UInt8](repeating: 0, count: buttons.count)
      for i in 0..<buttons.count {
        playerAction[i] = UInt8((encodedAction >> i) & 1)
        movie?[i, forPlayer: p] = playerAction[i] > 0
      }
      playerAction.withUnsafeBufferPointer {
        emulatorSetButtonMask(emulatorHandle, $0.baseAddress, playerAction.count, p)
      }
    }

    movie?.step()
    emulatorStep(emulatorHandle)
    gameData.updateMemory()

    // TODO: What about the 'info' dict?
    return EnvironmentStepResult(
      observation: updateCachedObservations(),
      reward: (0..<numPlayers).map { gameDataCurrentReward(gameData.handle, $0) },
      finished: gameDataIsDone(gameData.handle))
  }

  public func render<R: Renderer>(using renderer: R) where R.Data == ShapedArray<UInt8> {
    let screen = self.screen ?? updateCachedScreen()
    renderer.render(screen)
  }

  public func reset() {
    // Reset the initial state.
    if let state = initialState {
      let bytes = [UInt8](state)
      bytes.withUnsafeBufferPointer {
        var cBytesStruct = CBytes(bytes: $0.baseAddress, numBytes: bytes.count)
        emulatorSetState(emulatorHandle, &cBytesStruct)
      }
    }

    // Reset the button masks.
    for p in 0..<numPlayers {
      [UInt8](repeating: 0, count: buttons.count).withUnsafeBufferPointer {
        emulatorSetButtonMask(emulatorHandle, $0.baseAddress, buttons.count, p)
      }
    }

    emulatorStep(emulatorHandle)
    gameData.reset()
    gameData.updateMemory()
    updateCachedObservations()

    // Reset the recording.
    if let url = movieURL {
      let stateName = String(state!.split(separator: ".")[0])
      let movieFilename = "\(game)-\(stateName)-\(String(format: "%06d", movieID)).bk2"
      startRecording(at: url.appendingPathComponent(movieFilename))
      movieID += 1
    }

    movie?.step()
  }

  public func startRecording(at url: URL) {
    movie = Movie(at: url, recording: true, numPlayers: numPlayers)
    movie!.configure(for: self)
    if let state = initialState {
      movie!.state = state
    }
  }

  public func enableRecording(at url: URL? = nil) {
    movieURL = url ?? config.movieURL
  }

  public func disableRecording() {
    movieID = 0
    movieURL = nil
    if let m = movie {
      m.close()
      movie = nil
    }
  }

  public func loadState(named state: String, using integration: GameIntegration = .stable) throws {
    let file = state.hasSuffix(".state") ? state : "\(state).state"
    let fileURL = config.gameFile(file, for: game, using: integration)
    let data = try Data(contentsOf: fileURL!)
    self.initialState = try data.gunzipped()
    self.state = state
  }

  @discardableResult
  public func updateCachedScreen() -> ShapedArray<UInt8> {
    self.screen = Environment.getScreen(
      gameData: self.gameData,
      emulatorHandle: self.emulatorHandle,
      forPlayer: 0)
    return self.screen!
  }

  @discardableResult
  public func updateCachedMemory() -> ShapedArray<UInt8> {
    self.memory = Environment.getMemory(gameData: self.gameData)
    return self.memory!
  }

  @discardableResult
  private func updateCachedObservations() -> ShapedArray<UInt8> {
    switch config.observationSpaceType {
    case .screen: return updateCachedScreen()
    case .memory: return updateCachedMemory()
    }
  }

  private static func getScreen(
    gameData: GameData,
    emulatorHandle: UnsafeMutablePointer<CEmulator>?, 
    forPlayer player: UInt32 = 0
  ) -> ShapedArray<UInt8> {
    let cScreen = emulatorGetScreen(emulatorHandle)!.pointee
    let shape = [cScreen.height, cScreen.width, cScreen.channels]
    let values = Array(UnsafeBufferPointer(start: cScreen.values, count: shape.reduce(1, *)))
    let screen = ShapedArray(shape: shape, scalars: values)
    let cropInformation = gameDataCropInfo(gameData.handle, player)!.pointee
    let x = cropInformation.x
    let y = cropInformation.y

    var width = cropInformation.width
    var height = cropInformation.height
    if width == 0 || x + width > cScreen.width {
      width = cScreen.width
    } else {
      width += x
    }
    if height == 0 || y + height > cScreen.height {
      height = cScreen.height
    } else {
      height += y
    }

    if x == 0 && y == 0 && width == cScreen.width && height == cScreen.height {
      return screen
    } else {
      // TODO !!!: return screen[y..<height, x..<width]
      fatalError("Not implemented.")
    }
  }

  private static func getMemory(gameData: GameData) -> ShapedArray<UInt8> {
    let handle = gameDataMemory(gameData.handle)
    let cBlocks = memoryViewBlocks(handle)!.pointee
    let blocks = Array(UnsafeBufferPointer(start: cBlocks.blocks, count: cBlocks.numBlocks))
    var memoryBytes = [UInt8]()
    var numBytesPerBlock = 0
    blocks.sorted(by: { $0.address > $1.address }).forEach {
      memoryBytes += Array(UnsafeBufferPointer(start: $0.bytes, count: $0.numBytes))
      numBytesPerBlock = $0.numBytes
    }
    return ShapedArray(shape: [blocks.count, numBytesPerBlock], scalars: memoryBytes)
  }
}
