import TensorFlow

public protocol Stackable {
  associatedtype Stacked

  static func stack(_ values: [Self]) -> Stacked
  func unstacked() -> [Self]
}

extension Tensor: Stackable {
  public typealias Stacked = Tensor

  public static func stack(_ values: [Tensor]) -> Tensor {
    Tensor(stacking: values, alongAxis: 0)
  }

  public func unstacked() -> [Tensor] {
    unstacked(alongAxis: 0)
  }
}

// TODO: Array
// TODO: ShapedArray
