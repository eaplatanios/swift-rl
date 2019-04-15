import TensorFlow

extension Array: Distribution where Element: Distribution {
  public func sample(seed: UInt64? = nil) -> [Element.Value] {
    return map{$0.sample(seed: seed)}
  }
}
