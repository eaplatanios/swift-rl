import TensorFlow

public protocol Network: Layer {
  associatedtype State
  var state: State { get set }
  func initialize(using input: Input)
  func copy() -> Self
}

public struct LayerNetwork<WrappedLayer: Layer>: Network {
  public typealias State = None

  @noDerivative public var state: State = None()

  public var layer: WrappedLayer

  public init(_ layer: WrappedLayer) {
    self.layer = layer
  }

  public func initialize(using input: WrappedLayer.Input) {}

  @differentiable
  public func callAsFunction(_ input: WrappedLayer.Input) -> WrappedLayer.Output {
    layer(input)
  }

  public func copy() -> LayerNetwork<WrappedLayer> {
    LayerNetwork(layer)
  }
}
