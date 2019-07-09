import Foundation
import TensorFlow

public typealias TensorFlowSeed = (graph: Int32, op: Int32)

public struct None: Differentiable {
  public init() { }
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

struct Zip3Generator<
  A: IteratorProtocol,
  B: IteratorProtocol,
  C: IteratorProtocol
>: IteratorProtocol {
  private var first: A
  private var second: B
  private var third: C

  private var index = 0

  init(_ first: A, _ second: B, _ third: C) {
    self.first = first
    self.second = second
    self.third = third
  }

  mutating func next() -> (A.Element, B.Element, C.Element)? {
    if let first = first.next(), let second = second.next(), let third = third.next() {
      return (first, second, third)
    }
    return nil
  }
}

func zip<A: Sequence, B: Sequence, C: Sequence>(
  _ first: A, _ second: B, _ third: C
) -> IteratorSequence<Zip3Generator<A.Iterator, B.Iterator, C.Iterator>> {
    return IteratorSequence(Zip3Generator(
      first.makeIterator(), second.makeIterator(), third.makeIterator()))
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

/// Downloads the file at `url` to `path`, if `path` does not exist.
///
/// - Parameters:
///     - from: URL to download data from.
///     - to: Destination file path.
///
/// - Returns: Boolean value indicating whether a download was
///     performed (as opposed to not needed).
public func maybeDownload(from url: URL, to destination: URL) throws {
  print(destination.path)
  if !FileManager.default.fileExists(atPath: destination.path) {
    // Create any potentially missing directories.
    try FileManager.default.createDirectory(
      atPath: destination.deletingLastPathComponent().path,
      withIntermediateDirectories: true)

    // Create the URL session that will be used to download the dataset.
    let semaphore = DispatchSemaphore(value: 0)
    let delegate = DataDownloadDelegate(destinationFileUrl: destination, semaphore: semaphore)
    let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

    // Download the data to a temporary file and then copy that file to
    // the destination path.
    print("Downloading \(url).")
    let task = session.downloadTask(with: url)
    task.resume()

    // Wait for the download to finish.
    semaphore.wait()
  }
}

internal class DataDownloadDelegate: NSObject, URLSessionDownloadDelegate {
  let destinationFileUrl: URL
  let semaphore: DispatchSemaphore
  let numBytesFrequency: Int64

  internal var logCount: Int64 = 0

  init(
    destinationFileUrl: URL,
    semaphore: DispatchSemaphore,
    numBytesFrequency: Int64 = 1024 * 1024
  ) {
    self.destinationFileUrl = destinationFileUrl
    self.semaphore = semaphore
    self.numBytesFrequency = numBytesFrequency
  }

  internal func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64
  ) -> Void {
    if (totalBytesWritten / numBytesFrequency > logCount) {
      let mBytesWritten = String(format: "%.2f", Float(totalBytesWritten) / (1024 * 1024))
      if totalBytesExpectedToWrite > 0 {
        let mBytesExpectedToWrite = String(
          format: "%.2f", Float(totalBytesExpectedToWrite) / (1024 * 1024))
        print("Downloaded \(mBytesWritten) MBs out of \(mBytesExpectedToWrite).")
      } else {
        print("Downloaded \(mBytesWritten) MBs.")
      }
      logCount += 1
    }
  }

  internal func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) -> Void {
    logCount = 0
    do {
      try FileManager.default.moveItem(at: location, to: destinationFileUrl)
    } catch (let writeError) {
      print("Error writing file \(location.path) : \(writeError)")
    }
    print("The file was downloaded successfully to \(location.path).")
    semaphore.signal()
  }
}
