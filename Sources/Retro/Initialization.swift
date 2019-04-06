import CRetro
import Foundation

public internal(set) var supportedCores = [String: CoreInformation]()
public internal(set) var supportedExtensions = [String: String]()

// public func initializeRetro(withConfigFromFile configFile: String) throws {
//   let configFileURL = URL(fileURLWithPath: configFile)
//   let configString = try String(contentsOf: configFileURL, encoding: .utf8)
//   let config = try EmulatorConfig(fromJson: configString)
//   try initializeRetro(withConfig: config)
// }

public func initializeRetro<A: RetroActions>(withConfig config: EmulatorConfig<A>) throws {
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

public func getCore(forROM romURL: URL) throws -> String {
  let ext = romURL.pathExtension
  if let core = supportedExtensions[ext] {
    return core
  } else {
    throw RetroError.UnsupportedROMFile(romURL: romURL)
  }
}

public func getInformation(forCore core: String) throws -> CoreInformation {
  if let coreInformation = supportedCores[core] {
    return coreInformation
  } else {
    throw RetroError.UnsupportedCore(core: core)
  }
}
