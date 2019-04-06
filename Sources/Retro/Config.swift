import AnyCodable
import CRetro
import Foundation

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
internal let libExtension = "dylib"
#elseif os(Linux)
internal let libExtension = "so"
#else
internal let libExtension = "dll"
#endif

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

/// Represents different settings for the observation space of the environment.
public enum ObservationSpaceType: Int {
  /// Use RGB image observations.
  case screen

  /// Use RAM observations where you can see the memory of the game instead of the screen.
  case memory
}

public struct EmulatorConfig<A: RetroActions> {
  let coresInformationPath: URL
  let coresPath: URL
  let gameDataPath: URL
  let actionSpaceType: A
  let observationSpaceType: ObservationSpaceType
  let movieURL: URL?

  public init(
    coresInformationPath: URL,
    coresPath: URL,
    gameDataPath: URL,
    actionSpaceType: A,
    observationSpaceType: ObservationSpaceType = .screen,
    movieURL: URL? = nil
  ) {
    self.coresInformationPath = coresInformationPath
    self.coresPath = coresPath
    self.gameDataPath = gameDataPath
    self.actionSpaceType = actionSpaceType
    self.observationSpaceType = observationSpaceType
    self.movieURL = movieURL
    retroCorePath(coresPath.path)
    retroDataPath(gameDataPath.path)
  }
}

public extension EmulatorConfig where A == FilteredRetroActions {
  init(
    coresInformationPath: URL,
    coresPath: URL,
    gameDataPath: URL,
    observationSpaceType: ObservationSpaceType = .screen,
    movieURL: URL? = nil
  ) {
    self.init(
      coresInformationPath: coresInformationPath,
      coresPath: coresPath,
      gameDataPath: gameDataPath,
      actionSpaceType: FilteredRetroActions(),
      observationSpaceType: observationSpaceType,
      movieURL: movieURL)
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
    using integration: GameIntegration = .stable
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
    using integration: GameIntegration = .stable
  ) throws -> URL {
    for ext in supportedExtensions.keys {
      let possibleFile = gameFile("rom\(ext)", for: game, using: integration)
      if let file = possibleFile {
        return file
      }
    }
    throw RetroError.ROMFileNotFound(game: game)
  }

  func games(using integration: GameIntegration = .stable) -> [String] {
    var games: Set<String> = []
    for path in integration.paths {
      let pathFiles = FileManager.default.enumerator(
        atPath: gameDataPath.appendingPathComponent(path).path)
      while let game = pathFiles?.nextObject() as? String {
        if let _ = gameFile("rom.sha", for: game, using: integration) {
          games.insert(game)
        }
      }
    }
    return Array(games).sorted()
  }

  func states(for game: String, using integration: GameIntegration = .stable) -> [String] {
    var states: Set<String> = []
    for path in integration.paths {
      let gamePath = gameDataPath
        .appendingPathComponent(path)
        .appendingPathComponent(game)
      let pathFiles = FileManager.default.enumerator(atPath: gamePath.path)
      while let file = pathFiles?.nextObject() as? String {
        if file.hasSuffix(".state") && !file.hasPrefix("_") {
          let endIndex = file.index(file.endIndex, offsetBy: -6)
          states.insert(String(file.prefix(upTo: endIndex)))
        }
      }
    }
    return Array(states).sorted()
  }

  func scenarios(for game: String, using integration: GameIntegration = .stable) -> [String] {
    var scenarios: Set<String> = []
    for path in integration.paths {
      let gamePath = gameDataPath
        .appendingPathComponent(path)
        .appendingPathComponent(game)
      let pathFiles = FileManager.default.enumerator(atPath: gamePath.path)
      while let file = pathFiles?.nextObject() as? String {
        if file.hasSuffix(".json") {
          let fileURL = gamePath.appendingPathComponent(file)
          guard let json = try? String(contentsOf: fileURL) else { continue }
          guard let decoded = try? [String: AnyCodable](fromJson: json) else { continue }
          if decoded.keys.contains("reward") || 
              decoded.keys.contains("rewards") || 
              decoded.keys.contains("done") {
            let endIndex = file.index(file.endIndex, offsetBy: -5)
            scenarios.insert(String(file.prefix(upTo: endIndex)))
          }
        }
      }
    }
    return Array(scenarios).sorted()
  }
}
