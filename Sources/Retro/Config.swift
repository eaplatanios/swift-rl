import CRetro
import Foundation

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
internal let libExtension = "dylib"
#elseif os(Linux)
internal let libExtension = "so"
#else
internal let libExtension = "dll"
#endif

public private(set) var supportedCores = [String: CoreInformation]()
public private(set) var supportedExtensions = [String: String]()

internal func initializeRetroCoreInformation(withConfig config: EmulatorConfig) throws {
  let files = try FileManager.default.contentsOfDirectory(at: config.coresInformationPath, includingPropertiesForKeys: [.nameKey])
  for file in files {
    if file.pathExtension == "json" {
      let json = try String(contentsOf: file, encoding: .utf8)
      retroLoadCoreInfo(json)
      let cores = try [String: CoreInformation].from(json: json)
      supportedCores.merge(cores, uniquingKeysWith: { return $1 })
      for (core, coreInformation) in cores {
        for ext in coreInformation.extensions {
          supportedExtensions[".\(ext)"] = core
        }
      }
    }
  }
}

public struct CoreInformation: Codable {
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

public struct EmulatorConfig: Codable {
  let coresInformationPath: URL
  let coresPath: URL
  let gameDataPath: URL

  public init(coresInformationPath: URL, coresPath: URL, gameDataPath: URL) {
    self.coresInformationPath = coresInformationPath
    self.coresPath = coresPath
    self.gameDataPath = gameDataPath
    retroCorePath(coresPath.path)
    retroDataPath(gameDataPath.path)
  }
}

public extension EmulatorConfig {
  func coreInformation(for core: String) -> CoreInformation? {
    return supportedCores[core]
  }

  func coreLibraryPath(for core: String) -> URL? {
    return supportedCores[core]?.libraryPath
  }

  func core(forROM rom: URL) -> String? {
    return supportedExtensions[rom.pathExtension]
  }

  /// Returns a URL for the provided file and game.
  func gameFile(
    _ file: String, 
    for game: String, 
    with integration: GameIntegration = .stable
  ) -> URL? {
    for path in integration.paths {
      let possibleFile = gameDataPath
        .appendingPathComponent(path)
        .appendingPathComponent(game)
        .appendingPathComponent(file)
      if FileManager.default.fileExists(atPath: possibleFile.path) {
        return possibleFile
      }
    }
    return nil
  }

  func gameROMFile(
    for game: String, 
    with integration: GameIntegration = .stable
  ) throws -> URL {
    for ext in supportedExtensions.keys {
      let possibleFile = gameFile("rom\(ext)", for: game, with: integration)
      if let file = possibleFile {
        return file
      }
    }
    throw RetroError.ROMFileNotFound(game: game)
  }

  func games(for integration: GameIntegration = .stable) -> [String] {
    var possibleGames: Set<String> = []
    for path in integration.paths {
      let pathFiles = FileManager.default.enumerator(
        atPath: gameDataPath.appendingPathComponent(path).path)
      while let game = pathFiles?.nextObject() as? String {
        if let _ = gameFile("rom.sha", for: game, with: integration) {
          possibleGames.insert(game)
        }
      }
    }
    return Array(possibleGames).sorted()
  }
}
