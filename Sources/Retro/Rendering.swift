import TensorFlow

public protocol Renderer {
  associatedtype Data

  func render(_ data: Data)
}

public struct ShapedArrayPrinter<Scalar: LosslessStringConvertible>: Renderer {
  public typealias Data = ShapedArray<Scalar>

  public let maxEntries: Int
  public let flattened: Bool
  public let includeInfo: Bool

  public init(
    maxEntries: Int = 6,
    flattened: Bool = false,
    includeInfo: Bool = true
  ) {
    self.maxEntries = maxEntries
    self.flattened = flattened
    self.includeInfo = includeInfo
  }

  public func render(_ data: ShapedArray<Scalar>) {
    print(data.summarize(
      maxEntries: maxEntries,
      flattened: flattened,
      includeInfo: includeInfo))
  }
}
