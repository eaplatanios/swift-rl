
public protocol Batchable {
  associatedtype Batched

  static func batch(_ values: [Self]) -> Batched
  static func unbatch(_ values: Batched) -> [Self]
}

extension None: Batchable {
  public typealias Batched = None

  public static func batch(_ values: [None]) -> None {
    return None()
  }

  public static func unbatch(_ values: None) -> [None] {
    return [None]()
  }
}

extension Int: Batchable {
  public typealias Batched = [Int]

  public static func batch(_ values: [Int]) -> [Int] {
    return values
  }

  public static func unbatch(_ values: [Int]) -> [Int] {
    return values
  }
}

// TODO: Array
// TODO: ShapedArray
// TODO: Tensor
