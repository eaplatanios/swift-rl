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

public struct EmulatorConfig: Codable {
  let coresPath: URL
  let coresInformationPath: URL

  public func coreInformation(for core: String) -> CoreInformation? {
    return supportedCores[core]
  }

  public func coreLibraryPath(for core: String) -> URL? {
    return supportedCores[core]?.libraryPath
  }

  public func core(forROM rom: URL) -> String? {
    return supportedExtensions[rom.pathExtension]
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
