import Foundation

public enum RetroError: Error {
  case ROMFileNotFound(game: String)
}

public extension Encodable {
  func toJson(pretty: Bool = true) throws -> String {
    let encoder = JSONEncoder()
    if pretty {
      encoder.outputFormatting = .prettyPrinted
    }
    let data = try encoder.encode(self)
    return String(data: data, encoding: .utf8)!
  }
}

public extension Decodable {
  static func from(json: String) throws -> Self {
    let jsonDecoder = JSONDecoder()
    return try jsonDecoder.decode(Self.self, from: json.data(using: .utf8)!)
  }
}
