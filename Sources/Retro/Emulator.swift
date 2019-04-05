import CRetro
import Foundation

public func initializeRetro(withConfigFromFile configFile: String) throws {
  let configFileURL = URL(fileURLWithPath: configFile)
  let configString = try String(contentsOf: configFileURL, encoding: .utf8)
  let config = try EmulatorConfig.from(json: configString)
  try initializeRetro(withConfig: config)
}

public func initializeRetro(withConfig config: EmulatorConfig) throws {
  try initializeRetroCoreInformation(withConfig: config)
}
