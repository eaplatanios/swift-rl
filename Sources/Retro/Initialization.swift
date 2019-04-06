import CRetro
import Foundation

public internal(set) var supportedCores = [String: CoreInformation]()
public internal(set) var supportedExtensions = [String: String]()

public func initializeRetro(withConfigFromFile configFile: String) throws {
  let configFileURL = URL(fileURLWithPath: configFile)
  let configString = try String(contentsOf: configFileURL, encoding: .utf8)
  let config = try EmulatorConfig(fromJson: configString)
  try initializeRetro(withConfig: config)
}

public func initializeRetro(withConfig config: EmulatorConfig) throws {
  let files = try FileManager.default.contentsOfDirectory(
    at: config.coresInformationPath,
    includingPropertiesForKeys: [.nameKey])
  for file in files {
    if file.pathExtension == "json" {
      let json = try String(contentsOf: file, encoding: .utf8)
      retroLoadCoreInfo(json)
      let cores = try [String: CoreInformation](fromJson: json)
      supportedCores.merge(cores, uniquingKeysWith: { return $1 })
      for (core, coreInformation) in cores {
        for ext in coreInformation.extensions {
          supportedExtensions[".\(ext)"] = core
        }
      }
    }
  }
}

public func core(forROM romURL: URL) throws -> String {
  let ext = romURL.pathExtension
  if let core = supportedExtensions[ext] {
    return core
  } else {
    throw UnsupportedROMFile(romURL)
  }
}

public func information(forCore core: String) throws -> CoreInformation {
  if let coreInformation = supportedCores[core] {
    return coreInformation
  } else {
    throw UnsupportedCore(core)
  }
}
