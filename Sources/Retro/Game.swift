import AnyCodable
import CRetro
import Foundation

public struct Game: Hashable {
  public let name: String
  public let dataDir: URL
  public let rom: URL?
  public let romHash: String
  public let dataFile: URL?
  public let metadataFile: URL?
  public let states: [URL]
  public let scenarios: [URL]
}

public extension Game {
  init?(called name: String, withDataIn dataDir: URL) {
    self.name = name
    self.dataDir = dataDir
    self.rom = Game.findRom(for: name, withDataIn: dataDir)

    // Go through the available game data files.
    var romHash: String? = nil
    var dataFile: URL? = nil
    var metadataFile: URL? = nil
    var states: Set<URL> = []
    var scenarios: Set<URL> = []
    let dataFiles = FileManager.default.enumerator(at: dataDir, includingPropertiesForKeys: nil)
    while let file = dataFiles?.nextObject() as? URL {
      if file.lastPathComponent == "data.json" {
        dataFile = file
      } else if file.lastPathComponent == "metadata.json" {
        metadataFile = file
      } else if file.lastPathComponent == "rom.sha" {
        romHash = try? String(contentsOf: file)
      } else if file.pathExtension == "state" && !file.lastPathComponent.hasPrefix("_") {
        states.insert(file)
      } else if file.pathExtension == "json" {
        guard let json = try? String(contentsOf: file) else { continue }
        guard let decoded = try? [String: AnyCodable](fromJson: json) else { continue }
        if decoded.keys.contains("reward") || 
            decoded.keys.contains("rewards") || 
            decoded.keys.contains("done") {
          scenarios.insert(file)
        }
      }
    }

    if let hash = romHash {
      self.romHash = hash
    } else {
      return nil
    }

    self.dataFile = dataFile
    self.metadataFile = metadataFile
    self.states = Array(states).sorted(by: { $0.path > $1.path })
    self.scenarios = Array(scenarios).sorted(by: { $0.path > $1.path })
  }
}

public extension Game {
  struct Integration {
    public let paths: [String]
    public let name: String

    public init(paths: [String], name: String = "custom") {
      self.paths = paths
      self.name = name
    }
    
    public static let stable = Integration(paths: ["stable"], name: "stable")
    public static let experimental = Integration(paths: ["experimental"], name: "experimental")
    public static let contrib = Integration(paths: ["contrib"], name: "contrib")
    public static let all = (stable + experimental + contrib).named(as: "all")
    
    public func named(as name: String) -> Integration {
      return Integration(paths: paths, name: name)
    }

    public static func +(left: Integration, right: Integration) -> Integration {
      return Integration(paths: left.paths + right.paths, name: "\(left.name) | \(right.name)")
    }
  }
}

fileprivate extension Game {
  static func findRom(
    for game: String,
    withDataIn dataDir: URL
  ) -> URL? {
    // TODO: Support a game ROM registry.
    for ext in supportedExtensions.keys {
      let possibleFile = dataDir.appendingPathComponent("rom.\(ext)")
      if FileManager.default.fileExists(atPath: possibleFile.path) {
        return possibleFile
      }
    }
    return nil
  }
}

internal extension Game {
  @usableFromInline
  class Data {
    @usableFromInline var handle: UnsafeMutablePointer<CGameData>?

    @usableFromInline lazy var buttonCombos: [[Int32]] = {
      let cValidActions = gameDataValidActions(handle)!.pointee
      return (0 ..< cValidActions.numActionsOuter).map { combo in
        let numColumns = cValidActions.numActionsInner.advanced(by: combo).pointee
        return Array(UnsafeBufferPointer(
          start: cValidActions.actions.advanced(by: combo).pointee,
          count: numColumns))
      }
    }()

    init(dataFile: URL?, scenarioFile: URL?) throws {
      self.handle = gameDataCreate()
      if !load(dataFile: dataFile, scenarioFile: scenarioFile) {
        throw RetroError.GameDataFailure(
          message: "Failed to load game data from '\(dataFile?.path ?? "NOT_PROVIDED")'" + 
                   "or game scenario from '\(scenarioFile?.path ?? "NOT_PROVIDED")'.")
      }
    }

    deinit {
      gameDataDelete(handle)
    }

    @inlinable
    func load(dataFile: URL?, scenarioFile: URL?) -> Bool {
      switch (dataFile?.path, scenarioFile?.path) {
      case let (data?, scenario?): return gameDataLoad(self.handle, data, scenario)
      case let (data?, nil): return gameDataLoad(self.handle, data, nil)
      case let (nil, scenario?): return gameDataLoad(self.handle, nil, scenario)
      case (nil, nil): return gameDataLoad(self.handle, nil, nil)
      }
    }

    @inlinable
    subscript(_ name: String) -> Bool {
      get { return gameDataLookupBoolValue(handle, name) }
      set(newValue) { gameDataSetBoolValue(handle, name, newValue) }
    }

    @inlinable
    subscript(_ name: String) -> Int64 {
      get { return gameDataLookupIntValue(handle, name) }
      set(newValue) { gameDataSetIntValue(handle, name, newValue) }
    }

    @inlinable
    subscript(_ name: String) -> Double {
      get { return gameDataLookupDoubleValue(handle, name) }
      set(newValue) { gameDataSetDoubleValue(handle, name, newValue) }
    }

    @inlinable
    func updateMemory() {
      gameDataUpdateRam(handle)
    }

    @inlinable
    func reset() {
      gameDataReset(handle)
    }
  }
}

internal extension Game {
  @usableFromInline
  struct Metadata: Codable {
    let defaultState: String?
    let defaultPlayerState: [String]?

    enum CodingKeys: String, CodingKey {
      case defaultState = "default_state"
      case defaultPlayerState = "default_player_state"
    }
  }
}
