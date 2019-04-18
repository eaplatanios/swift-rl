// import TensorFlow

// public protocol Batchable {
//   associatedtype Batched

//   static func batch(_ values: [Self]) -> Batched
//   static func unbatch(_ values: Batched) -> [Self]
// }

// extension None: Batchable {
//   public typealias Batched = None

//   public static func batch(_ values: [None]) -> None {
//     return None()
//   }

//   public static func unbatch(_ values: None) -> [None] {
//     return [None]()
//   }
// }

// extension Int: Batchable {
//   public typealias Batched = [Int]

//   public static func batch(_ values: [Int]) -> [Int] {
//     return values
//   }

//   public static func unbatch(_ values: [Int]) -> [Int] {
//     return values
//   }
// }

// extension Tensor: Batchable {
//   public typealias Batched = Tensor

//   public static func batch(_ values: [Tensor]) -> Tensor {
//     // TODO: return Tensor(stacking: values, alongAxis: 0)
//     fatalError("Not implemented yet.")
//   }

//   public static func unbatch(_ values: Tensor) -> [Tensor] {
//     // TODO: return values.unstacked()?
//     fatalError("Not implemented yet.")
//   }
// }

// // TODO: Array
// // TODO: ShapedArray
