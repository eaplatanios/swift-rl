import TensorFlow

/// Computes the sigmoid of the specified tensor element-wise.
@inlinable
@differentiable(vjp: _vjpSigmoid)
public func sigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  return Raw.sigmoid(x)
}

@inlinable
func _vjpSigmoid<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
  return (sigmoid(x), { v in Raw.sigmoidGrad(x, dy: v) })
}

public extension Tensor {
  /// Returns a stacked tensor, constructed by stacking the provided tensors along
  /// the specified axis.
  /// - Precondition: The tensors must have the same dimensions,.
  /// - Precondition: The axis must be in the range `-rank..<rank`.
  @inlinable
  init(stacking tensors: [Tensor<Scalar>], alongAxis axis: Int32 = 0) {
    self = Raw.pack(tensors, axis: Int64(axis))
  }

  /// Returns a tiled tensor, constructed by tiling the provided tensor.
  ///
  /// This constructor creates a new tensor by replicating `tensor` `multiples` times. The
  /// constructed tensor's `i`'th dimension has `tensor.shape[i] * multiples[i]` elements, and the
  /// values of `tensor` are replicated `multiples[i]` times along the `i`'th dimension. For
  /// example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.
  ///
  /// - Precondition: The shape of `multiples` must be `[tensor.rank]`.
  @inlinable
  init(tiling tensor: Tensor<Scalar>, multiples: Tensor<Int32>) {
    self = Raw.tile(tensor, multiples: multiples)
  }

  @inlinable
  // @differentiable(vjp: _vjpConcatenated where Scalar : TensorFlowFloatingPoint)
  func concatenated(with tensors: [Tensor], alongAxis axis: Int32 = 0) -> Tensor {
    return Raw.concatV2([self] + tensors, axis: Tensor<Int32>(axis))
  }

  @inlinable
  // @differentiable(vjp: _vjpGathering where Scalar: TensorFlowFloatingPoint)
  func gathering(
    atIndices indices: Tensor<Int32>,
    alongAxis axis: Int32 = 0
  ) -> Tensor {
    return Raw.gatherV2(params: self, indices: indices, axis: Tensor<Int32>(axis))
  }

  @inlinable
  func batchGathering(
    atIndices indices: Tensor<Int32>,
    alongAxis axis: Int,
    numBatchDims batchDims: Int
  ) -> Tensor {
    precondition(batchDims >= 0 && batchDims < indices.rank,
      "'numBatchDims' must be non-negative and less than 'indices.rank'.")
    precondition(batchDims < rank, "'numBatchDims' must be less than the tensor's rank.")

    // Handle the axis argument by transposing the axis dimension so that it is the first
    // non-batch dimension, recursively calling `batchGathering` with `axis = 0`, and then
    // transposing the result to put the pre-axis dimensions before the indices dimensions.
    if axis != batchDims {
      // Adjust axis to be positive.
      let posAxis = axis < 0 ? axis + rank : axis

      precondition(posAxis >= 0 && posAxis < rank, "'axis' is out of range.")
      precondition(batchDims <= posAxis, "'batchDims' must be less than or equal to 'axis'.")

      // Move self[axis] up to self[batchDims].
      let permutation = Tensor<Int32>(concatenating: [
        Tensor<Int32>(0 ..< Int32(batchDims)),
        Tensor<Int32>(Int32(axis)).rankLifted(),
        Tensor<Int32>(rangeFrom: Int32(batchDims), to: Int32(posAxis), stride: 1),
        Tensor<Int32>(rangeFrom: Int32(axis) + 1, to: Int32(rank), stride: 1)])
      let tensor = transposed(withPermutations: permutation)
      let result = tensor.batchGathering(
        atIndices: indices, alongAxis: batchDims, numBatchDims: batchDims)

      // Move the result dimensions corresponding to self[batchDims ..< axis] to just before
      // the dimensions corresponding to indices[batchDims ...].
      let start = indices.rank + posAxis - batchDims
      let resultPermutation = Tensor<Int32>(concatenating: [
        Tensor<Int32>(0 ..< Int32(batchDims)),
        Tensor<Int32>(rangeFrom: Int32(indices.rank), to: Int32(start), stride: 1),
        Tensor<Int32>(Int32(batchDims) ..< Int32(indices.rank)),
        Tensor<Int32>(rangeFrom: Int32(start), to: Int32(result.rank), stride: 1)])
      return result.transposed(withPermutations: resultPermutation)
    }

    let castedShape = Tensor<Int32>(shapeTensor)
    var batchIndices = indices
    var accumulated = Tensor<Int32>(ones: [])
    for d in (1...batchDims).reversed() {
      accumulated *= castedShape[d]
      let dValue = castedShape[d - 1]
      let dIndices = Tensor<Int32>(
        rangeFrom: Tensor<Int32>(zeros: []),
        to: dValue,
        stride: Tensor<Int32>(ones: [])
      ) * accumulated
      let dShape = Tensor<Int32>(concatenating: [
        Tensor<Int32>([Int32](repeating: 1, count: Int(d - 1))),
        Tensor<Int32>([dValue]),
        Tensor<Int32>([Int32](repeating: 1, count: Int(indices.rank - 1)))])
      batchIndices += dIndices.reshaped(toShape: dShape)
    }

    let flatIndices = batchIndices.flattened()
    let outerShape = shapeTensor[Int(batchDims + 1)...]
    let innerShape = shapeTensor[..<Int(batchDims + 1)].product(squeezingAxes: [0])
    let flatTensor = reshaped(toShape: innerShape.rankLifted().concatenated(with: outerShape))
    let flatResult = flatTensor.gathering(atIndices: flatIndices)
    return flatResult.reshaped(toShape: indices.shapeTensor.concatenated(with: outerShape))
  }
}

public extension Tensor where Scalar : Numeric {
  /// Creates a tensor with all scalars set to zero that has the same shape and type as the provided
  /// tensor.
  ///
  /// - Parameter other: Tensor whose shape and data type to use.
  @inlinable
  init(zerosLike other: Tensor) {
    self = Raw.zerosLike(other)
  }

  /// Creates a tensor with all scalars set to one that has the same shape and type as the provided
  /// tensor.
  ///
  /// - Parameter other: Tensor whose shape and data type to use.
  @inlinable
  init(onesLike other: Tensor) {
    self = Raw.onesLike(other)
  }

  /// Creates a 1-D tensor representing a sequence from a starting value to, but not including, an
  /// end value, stepping by the specified amount.
  ///
  /// - Parameters:
  ///   - start: The starting value to use for the sequence. If the sequence contains any values,
  ///     the first one is `start`.
  ///   - end: An end value to limit the sequence. `end` is never an element of the resulting
  ///     sequence.
  ///   - stride: The amount to step by with each iteration. `stride` must be positive.
  @inlinable
  init(rangeFrom start: Tensor<Scalar>, to end: Tensor<Scalar>, stride: Tensor<Scalar>) {
    self = Raw.range(start: start, limit: end, delta: stride)
  }
}
