import AnyCodable
import CRetro
import CryptoSwift
import Foundation

public struct Game: Hashable {
  public let name: String
  public let dataDir: URL
  public let romHashes: [String]
  public let romLookupPaths: [URL]
  public let dataFile: URL?
  public let metadataFile: URL?
  public let states: [URL]
  public let scenarios: [URL]

  public lazy var rom: URL? = {
    return try? Game.findRom(
      romHashes: romHashes, dataDir: dataDir, romLookupPaths: romLookupPaths)
  }()
}

public extension Game {
  init?(called name: String, withDataIn dataDir: URL, romLookupPaths: [URL] = []) {
    self.name = name
    self.dataDir = dataDir

    // Go through the available game data files.
    var romHashes: [String] = []
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
        if let hashes = try? String(contentsOf: file) {
          romHashes = hashes
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: "\n")
            .map{$0.lowercased()}
        }
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

    if romHashes.isEmpty {
      return nil
    }

    self.romHashes = romHashes
    self.romLookupPaths = romLookupPaths
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
    romHashes: [String],
    dataDir: URL, 
    romLookupPaths: [URL] = []
  ) throws -> URL? {
    // Check if the game data directory contains the ROM.
    print("Attempting to obtain the game ROM from the game data directory.")
    for ext in supportedExtensions.keys {
      let possibleFile = dataDir.appendingPathComponent("rom.\(ext)")
      if FileManager.default.fileExists(atPath: possibleFile.path) {
        print("Found the game ROM from the game data directory.")
        return possibleFile
      }
    }

    // Check if the ROM can be found in the provided lookup directories.
    print("Attempting to obtain the game ROM from the provided lookup paths.")
    for dir in romLookupPaths {
      let files = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)
      while let file = files?.nextObject() as? URL {
        var bytes = [UInt8]()
        if let data = NSData(contentsOfFile: file.path) {
          var buffer = [UInt8](repeating: 0, count: data.length)
          data.getBytes(&buffer, length: data.length)
          bytes = buffer
        }

        // Obtain the ROM body and its SHA1 hash.
        var hash: String
        let ext = file.pathExtension.lowercased()
        if ext == "smd" {
          // Read the Super Magic Drive header.
          let body = bytes[512...]
          if body.count > 129 &&
             body[128] == UInt8(ascii: "E") && 
             body[129] == UInt8(ascii: "A") {
            var converted = [UInt8]()
            converted.reserveCapacity(body.count)
            for i in 0..<(body.count / 16384) {
              let block = body[(i * 16384)..<((i + 1) * 16384)]
              for j in 0..<8192 {
                converted.append(block[j + 8192])
                converted.append(block[j])
              }
            }
            bytes = converted
          }
          hash = bytes.sha1().map{String(format: "%02X", $0)}.joined()
        } else if ext == "nes" {
          hash = [UInt8](bytes[16...]).sha1().map{String(format: "%02X", $0)}.joined()
        } else {
          if bytes.count > 32 * 1024 * 1024 {
            throw RetroError.GameROMTooBig(
              "The ROM at '\(file)' is too big. Maximum supported size is 32MB.")
          }
          hash = bytes.sha1().map{String(format: "%02X", $0)}.joined()
        }

        if romHashes.contains(hash.lowercased()) {
          let romURL = dataDir.appendingPathComponent("rom.\(ext)")
          try Foundation.Data(bytes).write(to: romURL)
          print("Found the game ROM in the provided lookup paths.")
          return romURL
        }
      }
    }

    // TODO: Support automatically downloading ROMs.
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
          "Failed to load game data from '\(dataFile?.path ?? "NOT_PROVIDED")'" + 
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
