
public protocol Batchable {
  init(batched values: [Self])
  func unbatch() -> [Self]
}

// TODO: ShapedArray
// TODO: Tensor
