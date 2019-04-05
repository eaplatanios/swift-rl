import CRetro
import Foundation

public class GameData {
  private var handle: UnsafeMutablePointer<CGameData>?

  public init(
    using config: EmulatorConfig,
    for game: String,
    with integration: GameIntegration = .stable,
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
    using config: EmulatorConfig,
    loadingFrom dataFile: URL,
    with integration: GameIntegration = .stable,
    scenarioFile: URL? = nil
  ) {
    self.handle = gameDataCreate()
    load(dataFile: dataFile, scenarioFile: scenarioFile)
  }

  deinit {
    gameDataDelete(handle)
  }

  public func load(dataFile: URL?, scenarioFile: URL?) {
    switch (dataFile?.path, scenarioFile?.path) {
    case let (data?, scenario?): gameDataLoad(self.handle, data, scenario)
    case let (data?, nil): gameDataLoad(self.handle, data, nil)
    case let (nil, scenario?): gameDataLoad(self.handle, nil, scenario)
    case (nil, nil): gameDataLoad(self.handle, nil, nil)
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
