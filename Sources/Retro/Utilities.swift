import Foundation

public enum RetroError: Error {
  case ROMFileNotFound(game: String)
  case UnsupportedROMFile(romURL: URL)
  case UnsupportedCore(core: String)
  case GameDataFailure(message: String)
}

public extension Encodable {
  func json(pretty: Bool = true) throws -> String {
    let encoder = JSONEncoder()
    if pretty {
      encoder.outputFormatting = .prettyPrinted
    }
    let data = try encoder.encode(self)
    return String(data: data, encoding: .utf8)!
  }
}

public extension Decodable {
  init(fromJson json: String) throws {
    let jsonDecoder = JSONDecoder()
    self = try jsonDecoder.decode(Self.self, from: json.data(using: .utf8)!)
  }
}
