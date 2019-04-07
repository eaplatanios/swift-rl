import Foundation
import TensorFlow

public enum RetroError: Error {
  case ROMFileNotFound(game: String)
  case UnsupportedROMFile(romURL: URL)
  case UnsupportedCore(core: String)
  case GameDataFailure(message: String)
  case GLFWError(message: String)
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

public extension ShapedArray where Scalar: LosslessStringConvertible {
  func summarize(
    maxEntries: Int = 6,
    flattened: Bool = false,
    includeInfo: Bool = true
  ) -> String {
    func summarize(array: ShapedArraySlice<Scalar>, maxEntries: Int) -> String {
      switch array.rank {
        case 0:
          return String(describing: array.scalar!)
        case let i where i == 1 && array.scalarCount <= Swift.max(maxEntries, 6):
          return "[\(array.scalars.map { String($0) } .joined(separator: ", "))]"
        case 1:
          let start = array[0..<(maxEntries / 2)].scalars.map { String($0) }
          let end = array[(array.shape[0] - maxEntries / 2)..<array.shape[0]].scalars.map {
            String($0)
          }
          let slice = start + ["..."] + end
          return "[\(slice.joined(separator: ", "))]"
        default:
          let innerSummary: [String] = {
            func summarizeSlice(at index: Int) -> String {
              return summarize(array: array[index], maxEntries: maxEntries)
            }

            if array.shape[0] <= Swift.max(maxEntries, 6) {
              return (0..<array.shape[0]).map { summarizeSlice(at: $0) }
            } else {
              let start = (0..<(maxEntries / 2)).map { summarizeSlice(at: $0) }
              let end = ((array.shape[0] - maxEntries / 2)..<array.shape[0]).map { 
                summarizeSlice(at: $0)
              }
              return start + ["..."] + end
            }
          }()
          let padding = String(repeating: " ", count: self.rank - array.rank + 1)
          let extraLine = !flattened && array.rank >= 3 ? "\n" : ""
          let separator = (!flattened ? ",\n" : ", ") + extraLine + padding
          return "[\(innerSummary.joined(separator: separator))]"
      }
    }

    let slice = ShapedArraySlice(shape: shape, scalars: scalars)

    if includeInfo {
      let shapeString = "[" + shape.map { String($0) } .joined(separator: ", ") + "]"
      let info = "ShapedArray[\(Scalar.self), \(shapeString)]"
      let separator = flattened ? ": " : "\n"
      return info + separator + summarize(array: slice, maxEntries: maxEntries)
    }

    return summarize(array: slice, maxEntries: maxEntries)
  }
}
