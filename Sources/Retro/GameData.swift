import CRetro
import Foundation

public class GameData {
  internal var handle: UnsafeMutablePointer<CGameData>?

  public lazy var buttonCombos: [[Int]] = gameDataValidActions(handle)

  public init(
    withConfig config: EmulatorConfig,
    for game: String,
    using integration: GameIntegration = .stable,
    dataFile: URL? = nil,
    scenarioFile: URL? = nil
  ) {
    self.handle = gameDataCreate()
    let data = dataFile ?? config.gameFile(
      "data.json", for: game, with: integration)
    let scenario = scenarioFile ?? config.gameFile(
      "scenario.json", for: game, with: integration)
    load(dataFile: data, scenarioFile: scenario)
  }

  public init(
    withConfig config: EmulatorConfig,
    loadingFrom dataFile: URL,
    using integration: GameIntegration = .stable,
    scenarioFile: URL? = nil
  ) {
    self.handle = gameDataCreate()
    load(dataFile: dataFile, scenarioFile: scenarioFile)
  }

  deinit {
    gameDataDelete(handle)
  }

  public func load(dataFile: URL?, scenarioFile: URL?) -> Bool {
    switch (dataFile?.path, scenarioFile?.path) {
    case let (data?, scenario?): return gameDataLoad(self.handle, data, scenario)
    case let (data?, nil): return gameDataLoad(self.handle, data, nil)
    case let (nil, scenario?): return gameDataLoad(self.handle, nil, scenario)
    case (nil, nil): return gameDataLoad(self.handle, nil, nil)
    }
  }

  public subscript(_ name: String) -> Bool {
    get { return gameDataLookupBoolValue(handle, name) }
    set(newValue) { gameDataSetBoolValue(handle, name, newValue) }
  }

  public subscript(_ name: String) -> Int64 {
    get { return gameDataLookupIntValue(handle, name) }
    set(newValue) { gameDataSetIntValue(handle, name, newValue) }
  }

  public subscript(_ name: String) -> Double {
    get { return gameDataLookupDoubleValue(handle, name) }
    set(newValue) { gameDataSetDoubleValue(handle, name, newValue) }
  }

  
}

public struct GameMetadata: Codable {
  public let defaultState: String?
  public let defaultPlayerState: [String]?
}

// public class SearchListHandle {
//   private let gameData: GameData

//   public init(gameData: GameData) {
//     self.gameData = gameData
//   }

//   public subscript(_ name: String) -> SearchHandle {
//     return SearchHandle(gameData: gameData, name: name)
//   }

//   public func remove(_ name: String) {
//     gameDataRemoveSearch(gameData.handle, name)
//   }

//   public func searches() -> [String] {
//     let cNames = gameDataListSearchNames(gameData.handle)!.pointee
//     let numNames = cNames.numNames
//     var namesPointer = cNames.names!
//     var names = [String]()
//     for _ in 0..<numNames {
//       names.append(String(cString: namesPointer.pointee!))
//       namesPointer = namesPointer.advanced(by: 1)
//     }
//     return names
//   }
// }

// public class SearchHandle {
//   private var handle: UnsafeMutablePointer<CSearch>?

//   private let gameData: GameData
//   private let name: String

//   public init(gameData: GameData, name: String) {
//     self.handle = nil
//     self.gameData = gameData
//     self.name = name
//   }

//   deinit {
//     if let h = handle {
//       searchDelete(h)
//     }
//   }

//   public func search(for value: Int64) {
//     gameDataSearch(gameData.handle, name, value)
//   }

//   public func delta(op: String, reference: Int64) {
//     gameDataDeltaSearch(gameData.handle, name, op, reference)
//   }
// }
