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
import ReinforcementLearning
import ZIPFoundation

public struct RetroGame: Hashable {
  public let name: String
  public let dataDir: URL
  public let romHashes: [String]
  public let romLookupPaths: [URL]
  public let romsDownloadPath: URL?
  public let dataFile: URL?
  public let metadataFile: URL?
  public let states: [URL]
  public let scenarios: [URL]

  public lazy var core: RetroCore? = {
    for (coreName, core) in supportedCores {
      if name.hasSuffix("-\(coreName)") {
        return core
      }
    }
    return nil
  }()

  public lazy var rom: URL? = {
    let romFinder = RomFinder()
    return try? romFinder.findRom(
      core: core,
      romHashes: romHashes,
      gameDataDir: dataDir,
      lookupPaths: romLookupPaths,
      downloadPath: romsDownloadPath)
  }()
}

public extension RetroGame {
  init?(
    called name: String,
    withDataIn dataDir: URL,
    romLookupPaths: [URL] = [],
    romsDownloadPath: URL? = nil
  ) {
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
        guard let keys = try? (JSONSerialization.jsonObject(
            with: json.data(using: .utf8)!,
            options: .mutableContainers) as? [String: Any])?.keys else { continue }
        if keys.contains("reward") || keys.contains("rewards") || keys.contains("done") {
          scenarios.insert(file)
        }
      }
    }

    if romHashes.isEmpty {
      return nil
    }

    self.romHashes = romHashes
    self.romLookupPaths = romLookupPaths
    self.romsDownloadPath = romsDownloadPath
    self.dataFile = dataFile
    self.metadataFile = metadataFile
    self.states = Array(states).sorted(by: { $0.path > $1.path })
    self.scenarios = Array(scenarios).sorted(by: { $0.path > $1.path })
  }
}

public extension RetroGame {
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
      Integration(paths: paths, name: name)
    }

    public static func +(left: Integration, right: Integration) -> Integration {
      Integration(paths: left.paths + right.paths, name: "\(left.name) | \(right.name)")
    }
  }
}

internal extension RetroGame {
  @usableFromInline
  class Data {
    @usableFromInline var handle: UnsafeMutablePointer<CGameData>?

    @usableFromInline lazy var buttonCombos: [[Int32]] = {
      let cValidActionsHandle = gameDataValidActions(handle)!
      defer { gameDataValidActionsDelete(cValidActionsHandle) }
      let cValidActions = cValidActionsHandle.pointee
      var counter = 0
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
      get { gameDataLookupBoolValue(handle, name) }
      set(newValue) { gameDataSetBoolValue(handle, name, newValue) }
    }

    @inlinable
    subscript(_ name: String) -> Int64 {
      get { gameDataLookupIntValue(handle, name) }
      set(newValue) { gameDataSetIntValue(handle, name, newValue) }
    }

    @inlinable
    subscript(_ name: String) -> Double {
      get { gameDataLookupDoubleValue(handle, name) }
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

internal extension RetroGame {
  @usableFromInline
  struct Metadata: Codable {
    @usableFromInline let defaultState: String?
    @usableFromInline let defaultPlayerState: [String]?

    enum CodingKeys: String, CodingKey {
      case defaultState = "default_state"
      case defaultPlayerState = "default_player_state"
    }
  }
}

fileprivate extension RetroGame {
  struct RomFinder {
    static let atari2600RomsURLs: [String: URL] = [
      "ROMS.zip": URL(
        string: "https://dl.dropboxusercontent.com/s/uyb07bkcfzv60hz/ROMS.zip?dl=0")!,
      "HC ROMS.zip": URL(
        string: "https://dl.dropboxusercontent.com/s/qjzb553lxw5av8b/HC%20ROMS.zip?dl=0")!]

    func findRom(
      core: RetroCore?,
      romHashes: [String],
      gameDataDir: URL,
      lookupPaths: [URL] = [],
      downloadPath: URL? = nil
    ) throws -> URL? {
      // Check if the game data directory contains the ROM.
      if let rom = try findRom(for: romHashes, inGameDataDir: gameDataDir) {
        return rom
      }

      // Check if the ROM can be found in the provided lookup directories.
      if let rom = try findRom(
        for: romHashes,
        core: core,
        inLookupPaths: lookupPaths,
        gameDataDir: gameDataDir
      ) {
        return rom
      }

      // Check if the ROM can be downloaded automatically.
      if let path = downloadPath {
        print("Attempting to download the game ROM.")
        let romDownloadPath = path.appendingPathComponent("roms")
        for (name, url) in core?.romURLs() ?? [:] {
          let file = romDownloadPath.appendingPathComponent(name)

          // Download the data, if necessary.
          try maybeDownload(from: url, to: file)

          // Extract the data, if necessary.
          let fileExtension = file.pathExtension
          if fileExtension == "zip" {
            let extractedPath = file.deletingPathExtension()
            if !FileManager.default.fileExists(atPath: extractedPath.path) {
              try FileManager.default.unzipItem(at: file, to: extractedPath)
            }
          }
        }

        if let rom = try findRom(
          for: romHashes,
          core: core,
          inLookupPaths: [romDownloadPath],
          gameDataDir: gameDataDir
        ) {
          let romFile = gameDataDir.appendingPathComponent("rom.\(core!.information.extensions[0])")
          try FileManager.default.copyItem(at: rom, to: romFile)
          print("Downloaded the game ROM successfully.")
          return romFile
        }
      }

      return nil
    }

    func findRom(for romHashes: [String], inGameDataDir dataDir: URL) throws -> URL? {
      print("Attempting to obtain the game ROM from the game data directory.")
      for ext in supportedExtensions.keys {
        let possibleFile = dataDir.appendingPathComponent("rom.\(ext)")
        if FileManager.default.fileExists(atPath: possibleFile.path) {
          let (_, hash) = try computeROMHash(for: possibleFile)
          if romHashes.contains(hash.lowercased()) {
            print("Found the game ROM in the game data directory.")
            return possibleFile
          }
        }
      }
      return nil
    }

    func findRom(
      for romHashes: [String],
      core: RetroCore?,
      inLookupPaths lookupPaths: [URL],
      gameDataDir dataDir: URL
    ) throws -> URL? {
      print("Attempting to obtain the game ROM from the provided lookup paths.")
      for dir in lookupPaths {
        let files = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil)
        while let file = files?.nextObject() as? URL {
          let (bytes, hash) = try computeROMHash(for: file)
          if romHashes.contains(hash.lowercased()) {
            let ext = core?.information.extensions[0] ?? file.pathExtension.lowercased()
            let romURL = dataDir.appendingPathComponent("rom.\(ext)")
            try Foundation.Data(bytes).write(to: romURL)
            print("Found the game ROM in the provided lookup paths.")
            return romURL
          }
        }
      }
      return nil
    }

    func computeROMHash(for file: URL) throws -> (bytes: [UInt8], hash: String) {
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
        let bytesHash = bytes.sha1()
        hash = (0..<20).map{String(format: "%02X", bytesHash[$0])}.joined()
      } else if ext == "nes" {
        let bytesHash = [UInt8](bytes[16...]).sha1()
        hash = (0..<20).map{String(format: "%02X", bytesHash[$0])}.joined()
      } else {
        if bytes.count > 32 * 1024 * 1024 {
          throw RetroError.GameROMTooBig(
            "The ROM at '\(file)' is too big. Maximum supported size is 32MB.")
        }
        let bytesHash = bytes.sha1()
        hash = (0..<20).map{String(format: "%02X", bytesHash[$0])}.joined()
      }
      return (bytes: bytes, hash: hash)
    }
  }
}

fileprivate extension RetroCore {
  func romURLs() -> [String: URL] {
    name == "Atari2600" ? RetroGame.RomFinder.atari2600RomsURLs : [:]
  }
}
