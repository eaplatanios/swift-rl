import TensorFlow

public protocol Stackable {
  associatedtype Stacked

  static func stack(_ values: [Self]) -> Stacked
  static func unstack(_ value: Stacked) -> [Self]
}

extension None: Stackable {
  public typealias Stacked = None

  public static func stack(_ values: [None]) -> None {
    return None()
  }

  public static func unstack(_ values: None) -> [None] {
    return [None]()
  }
}

extension Tensor: Stackable {
  public typealias Stacked = Tensor

  public static func stack(_ values: [Tensor]) -> Tensor {
    return Tensor(stacking: values, alongAxis: 0)
  }

  public static func unstack(_ values: Tensor) -> [Tensor] {
    return values.unstack()
  }
}

// TODO: Array
// TODO: ShapedArray
