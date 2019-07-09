import TensorFlow

public protocol Stackable {
  associatedtype Stacked

  static func stack(_ values: [Self]) -> Stacked
  func unstacked() -> [Self]
}

extension None: Stackable {
  public typealias Stacked = None

  public static func stack(_ values: [None]) -> None {
    None()
  }

  public func unstacked() -> [None] {
    [None]()
  }
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
