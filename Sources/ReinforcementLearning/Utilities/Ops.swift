import TensorFlow

/// Computes the sigmoid of the specified tensor element-wise.
@inlinable @inline(__always)
@differentiable(vjp: _vjpSigmoid)
public func sigmoid<T: TensorFlowFloatingPoint>(_ x: Tensor<T>) -> Tensor<T> {
  return Raw.sigmoid(x)
}

@inlinable @inline(__always)
func _vjpSigmoid<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> (Tensor<T>, (Tensor<T>) -> Tensor<T>) {
  return (sigmoid(x), { v in Raw.sigmoidGrad(x, dy: v) })
}
