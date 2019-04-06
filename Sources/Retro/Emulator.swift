import CRetro
import Gzip
import Foundation

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

public class Environment {
  public let config: EmulatorConfig
  public let game: String
  public let numPlayers: Int
  public let core: String
  public let buttons: [String?]
  public let buttonCombos: [[Int]]
  public let actionSpace: Space
  public let observationSpace: Space

  public private(set) var gameData: GameData
  public private(set) var gameDataFile: URL?
  public private(set) var gameScenarioFile: URL?

  private var handle: UnsafeMutablePointer<CEmulator>?

  private var state: String? = nil
  private var initialState: Data? = nil

  public init(
    withConfig config: EmulatorConfig,
    for game: String, 
    in state: State = .provided,
    using integration: GameIntegration = .stable,
    numPlayers: Int = 1,
    gameDataFile: String = "data.json",
    gameScenarioFile: String = "scenario.json"
  ) {
    let gameROMFile = config.gameROMFile(for: game, using: integration)
    let gameMetadataFile = config.gameFile("metadata.json", for: game, using: integration)
    
    self.config = config
    self.game = game
    self.numPlayers = numPlayers
    self.core = core(forROM: gameROMFile)
    self.gameData = GameData(withConfig: config, for: game, using: integration)
    self.gameDataFile = config.gameFile(gameDataFile, for: game, using: integration)
    self.gameScenarioFile = config.gameFile(gameScenarioFile, for: game, using: integration)
    
    switch state {
    case .none:
      self.state = nil
    case .provided:
      guard let metadataFile = gameMetadataFile else { self.state = nil }
      guard let metadataJson = try String(contentsOf: metadataFile) else { self.state = nil }
      guard let metadata = try GameMetadata(fromJson: metadataJson) else { self.state = nil }
      if metadata.defaultPlayerState != nil && numPlayers <= metadata.defaultPlayerState!.count {
        self.state = metadata.defaultPlayerState![numPlayers - 1]
      } else if metadata.defaultState != nil {
        self.state = metadata.defaultState!
      } else {
        self.state = nil
      }
    case .custom(let state):
      self.state = state
    }

    if let state = self.state {
      loadState(named: state, using: integration)
    }

    self.handle = emulatorCreate(gameROMFile.path)
    emulatorConfigureData(self.handle, self.gameData.handle)
    emulatorStep(self.handle)

    let coreInformation = information(forCore: self.core)
    if !self.gameData.load(dataFile: self.gameDataFile, scenarioFile: self.gameScenarioFile) {
      throw GameDataFailure(
        "Failed to load game data from '\(dataFile.path)'"
        "or game scenario from '\(scenarioFile.path)'.")
    }

    self.buttons = coreInformation.buttons
    self.buttonCombos = self.gameData.validActions

    switch config.actionSpaceType {
    case .discrete:
      let numCombos = self.buttonCombos.map { $0.count } .reduce(1, *)
      self.actionSpace = Discrete(size: Int32(pow(combos, numPlayers)))
    case .multiDiscrete:
      self.actionSpace = MultiDiscrete(sizes: self.buttonCombos.map { Int32($0.count * numPlayers) })
    default:
      self.actionSpace = MultiBinary(size: self.buttons.count * numPlayers)
    }




  }

  deinit {
    emulatorDelete(self.handle)
  }

  public func loadState(named state: String, using integration: GameIntegration = .stable) throws {
    let file = state.hasSuffix(".state") ? state : "\(state).state"
    let fileURL = config.gameFile(file, for: game, using: integration)
    let data = Data(contentsOf: fileURL)
    self.initialState = try data.gunzipped()
    self.state = state
  }
}
