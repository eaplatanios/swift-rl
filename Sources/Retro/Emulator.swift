import CRetro
import Foundation
import Gzip
import TensorFlow

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
internal let libExtension = "dylib"
#elseif os(Linux)
internal let libExtension = "so"
#else
internal let libExtension = "dll"
#endif

public internal(set) var supportedCores = [String: Core]()
public internal(set) var supportedExtensions = [String: Core]()

public struct Core {
  public let name: String
  public let information: Information

  public struct Information: Codable {
    public let library: String
    public let extensions: [String]
    public let memorySize: Int?
    public let keyBinds: [String?]
    public let buttons: [String?]
    public let actions: [[[String]]]

    public lazy var libraryPath = URL(fileURLWithPath: "\(library)_libretro.\(libExtension)")

    enum CodingKeys: String, CodingKey {
      case library = "lib"
      case extensions = "ext"
      case memorySize = "rambase"
      case keyBinds = "keybinds"
      case buttons = "buttons"
      case actions = "actions"
    }
  }
}

public class Emulator {
  @usableFromInline internal var handle: UnsafeMutablePointer<CEmulator>?

  public let config: Config
  public let core: Core
  public let game: Game
  public let numPlayers: UInt32
  public let scenario: URL

  @usableFromInline internal var gameData: Game.Data
  @usableFromInline internal var startingStateData: Data? = nil
  @usableFromInline internal var cachedScreenUpdated: Bool = false
  @usableFromInline internal var cachedScreen: ShapedArray<UInt8>? = nil
  @usableFromInline internal var cachedMemoryUpdated: Bool = false
  @usableFromInline internal var cachedMemory: ShapedArray<UInt8>? = nil

  public init(
    for game: Game,
    configuredAs config: Config,
    playing scenario: URL? = nil,
    numPlayers: UInt32 = 1
  ) throws {
    if game.rom == nil {
      throw RetroError.ROMFileNotFound(game: game.name)
    }

    guard let core = config.core(forROM: game.rom!) else {
      throw RetroError.UnsupportedCore(message: "For ROM: \(game.rom!).")
    }

    self.handle = emulatorCreate(game.rom!.path)
    self.config = config
    self.core = core
    self.game = game
    self.scenario = scenario ?? game.scenarios.first(where: {
      $0.lastPathComponent == "scenario.json"
    })!
    self.numPlayers = numPlayers

    self.gameData = try Game.Data(dataFile: game.dataFile, scenarioFile: self.scenario)
    emulatorConfigureData(self.handle, self.gameData.handle)
    emulatorStep(self.handle)
  }

  deinit {
    emulatorDelete(handle)
  }
  
  public func loadStartingState(from url: URL) throws {
    let compressedData = try Data(contentsOf: url)
    startingStateData = try compressedData.gunzipped()
  }

  @inlinable
  public func buttons() -> [String?] {
    return core.information.buttons
  }

  @inlinable
  public func buttonCombos() -> [[Int32]] {
    return gameData.buttonCombos
  }

  @inlinable
  public func setButtonMask(for player: UInt32, to mask: [UInt8]) {
    mask.withUnsafeBufferPointer {
      emulatorSetButtonMask(handle, $0.baseAddress, mask.count, player)
    }
  }

  @inlinable
  public func step() {
    emulatorStep(handle)
    gameData.updateMemory()
    cachedScreenUpdated = false
    cachedMemoryUpdated = false
  }

  @inlinable
  public func reset() {
    // Reset the initial state.
    if let data = startingStateData {
      let bytes = [UInt8](data)
      bytes.withUnsafeBufferPointer {
        var cBytesStruct = CBytes(bytes: $0.baseAddress, numBytes: bytes.count)
        emulatorSetState(handle, &cBytesStruct)
      }
    }

    // Reset the button masks.
    for p in 0..<numPlayers {
      [UInt8](repeating: 0, count: buttons().count).withUnsafeBufferPointer {
        emulatorSetButtonMask(handle, $0.baseAddress, buttons().count, p)
      }
    }

    emulatorStep(handle)
    gameData.reset()
    gameData.updateMemory()
    cachedScreenUpdated = false
    cachedMemoryUpdated = false
  }

  @inlinable
  public func screen() -> ShapedArray<UInt8>? {
    return cachedScreenUpdated ? cachedScreen : updateCachedScreen()
  }

  @inlinable
  public func memory() -> ShapedArray<UInt8>? {
    return cachedMemoryUpdated ? cachedMemory : updateCachedMemory()
  }

  @inlinable
  public func reward(for player: UInt32) -> Float {
    return gameDataCurrentReward(gameData.handle, player)
  }

  @inlinable
  public func finished() -> Bool {
    return gameDataIsDone(gameData.handle)
  }
  
  @usableFromInline @discardableResult
  internal func updateCachedScreen() -> ShapedArray<UInt8> {
    let cScreen = emulatorGetScreen(handle)!.pointee
    let shape = [cScreen.height, cScreen.width, cScreen.channels]
    let values = Array(UnsafeBufferPointer(start: cScreen.values, count: shape.reduce(1, *)))
    let screen = ShapedArray(shape: shape, scalars: values)
    let cropInformation = gameDataCropInfo(gameData.handle, 0)!.pointee
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
      self.cachedScreen = screen
    } else {
      // TODO !!!: return screen[y..<height, x..<width]
      fatalError("Not implemented.")
    }
    self.cachedScreenUpdated = true
    return self.cachedScreen!
  }

  @usableFromInline @discardableResult
  internal func updateCachedMemory() -> ShapedArray<UInt8> {
    let memoryHandle = gameDataMemory(gameData.handle)
    let cBlocks = memoryViewBlocks(memoryHandle)!.pointee
    let blocks = Array(UnsafeBufferPointer(start: cBlocks.blocks, count: cBlocks.numBlocks))
    var memoryBytes = [UInt8]()
    var numBytesPerBlock = 0
    blocks.sorted(by: { $0.address > $1.address }).forEach {
      memoryBytes += Array(UnsafeBufferPointer(start: $0.bytes, count: $0.numBytes))
      numBytesPerBlock = $0.numBytes
    }
    self.cachedMemory = ShapedArray(shape: [blocks.count, numBytesPerBlock], scalars: memoryBytes)
    self.cachedMemoryUpdated = true
    return self.cachedMemory!
  }
}

public extension Emulator {
  struct Config {
    public let coreInformationLookupPath: URL
    public let coreLookupPath: URL
    public let gameDataLookupPath: URL?

    public init(
      coreInformationLookupPath: URL,
      coreLookupPathHint: URL,
      gameDataLookupPathHint: URL? = nil
    ) throws {
      // Load information about the supported cores.
      let files = try FileManager.default.contentsOfDirectory(
      at: coreInformationLookupPath,
      includingPropertiesForKeys: [.nameKey])
      for file in files {
        if file.pathExtension == "json" {
          let json = try String(contentsOf: file, encoding: .utf8)
          retroLoadCoreInfo(json)
          let coresInformation = try [String: Core.Information](fromJson: json)
          var cores = [String: Core]()
          for (name, information) in coresInformation {
            cores[name] = Core(name: name, information: information)
          }
          supportedCores.merge(cores, uniquingKeysWith: { return $1 })
          for (_, core) in cores {
            for ext in core.information.extensions {
              supportedExtensions[ext] = core
            }
          }
        }
      }

      self.coreInformationLookupPath = coreInformationLookupPath
      let cCoreLookupPath = retroCorePath(coreLookupPathHint.path)!
      self.coreLookupPath = URL(fileURLWithPath: String(cString: cCoreLookupPath))
      cCoreLookupPath.deallocate()

      if let hint = gameDataLookupPathHint {
        let cGameDataLookupPath = retroDataPath(hint.path)!
        self.gameDataLookupPath = URL(fileURLWithPath: String(cString: cGameDataLookupPath))
        cGameDataLookupPath.deallocate()
      } else {
        gameDataLookupPath = nil
      }
    }
    
    @inlinable
    public func core(called core: String) -> Core? {
      return supportedCores[core]
    }

    @inlinable
    public func core(forROM rom: URL) -> Core? {
      return supportedExtensions[rom.pathExtension]
    }

    public func game(
      called gameName: String,
      using integration: Game.Integration = .stable
    ) -> Game? {
      if let lookupPath = gameDataLookupPath {
        for path in integration.paths {
          let paths = FileManager.default.enumerator(
            at: lookupPath.appendingPathComponent(path), includingPropertiesForKeys: nil)
          while let gameDataPath = paths?.nextObject() as? URL {
            let name = gameDataPath.lastPathComponent
            if name != gameName { continue }
            if let game = Game(called: name, withDataIn: gameDataPath) {
              return game
            }
          }
        }
      }
      return nil
    }

    public func games(using integration: Game.Integration = .stable) -> [Game] {
      if let lookupPath = gameDataLookupPath {
        var games: Set<Game> = []
        for path in integration.paths {
          let paths = FileManager.default.enumerator(
            at: lookupPath.appendingPathComponent(path), includingPropertiesForKeys: nil)
          while let gameDataPath = paths?.nextObject() as? URL {
            let gameName = gameDataPath.lastPathComponent
            if let game = Game(called: gameName, withDataIn: gameDataPath) {
              games.insert(game)
            }
          }
        }
        return Array(games).sorted(by: { $0.name > $1.name })
      } else {
        return [Game]()
      }
    }
  }
}
