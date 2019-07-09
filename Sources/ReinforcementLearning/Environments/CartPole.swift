import TensorFlow

public struct CartPoleEnvironment: Environment {
  private let gravity: Float = 9.8
  private let cartMass: Float = 1.0
  private let poleMass: Float = 0.1
  private let length: Float = 0.5
  private let forceMagnitude: Float = 10.0
  private let secondCountBetweenUpdates: Float = 0.02
  private let angleThreshold: Float = 12 * 2 * Float.pi / 360
  private let positionThreshold: Float = 2.4

  private var totalMass: Float { cartMass + poleMass }
  private var poleMassLength: Float { poleMass * length }

  public let batched: Bool = false
  public let actionSpace: Discrete = Discrete(withSize: 2)
  public var observationSpace: Box<Float> {
    let boundary = Tensor<Float>([
      positionThreshold * 2,
      Float.greatestFiniteMagnitude,
      angleThreshold * 2,
      Float.greatestFiniteMagnitude])
    return Box(lowerBound: -boundary, upperBound: boundary)
  }

  private var position: Tensor<Float> = CartPoleEnvironment.randomTensor()
  private var positionDerivative: Tensor<Float> = CartPoleEnvironment.randomTensor()
  private var angle: Tensor<Float> = CartPoleEnvironment.randomTensor()
  private var angleDerivative: Tensor<Float> = CartPoleEnvironment.randomTensor()
  private var needsReset: Tensor<Bool> = Tensor<Bool>([false])

  /// Updates the environment according to the provided action.
  @discardableResult
  public mutating func step(taking action: Tensor<Int32>) -> Step<Tensor<Float>, Tensor<Float>> {
    precondition(actionSpace.contains(action), "Invalid action provided.")
    let force = Tensor<Float>(2 * action - 1) * forceMagnitude
    let angleCosine = cos(angle)
    let angleSine = sin(angle)
    let temp = force + poleMassLength * angleDerivative * angleDerivative * angleSine
    let angleAccNominator = gravity * angleSine - temp * angleCosine / totalMass
    let angleAccDenominator = 4/3 - poleMass * angleCosine * angleCosine / totalMass
    let angleAcc = angleAccNominator / (length * angleAccDenominator)
    let positionAcc = (temp - poleMassLength * angleAcc * angleCosine) / totalMass
    position += secondCountBetweenUpdates * positionDerivative
    positionDerivative += secondCountBetweenUpdates * positionAcc
    angle += secondCountBetweenUpdates * angleDerivative
    angleDerivative += secondCountBetweenUpdates * angleAcc

    // Take into account the finished simulations in the batch.
    position = position.replacing(
      with: CartPoleEnvironment.randomTensor(), where: needsReset)
    positionDerivative = positionDerivative.replacing(
      with: CartPoleEnvironment.randomTensor(), where: needsReset)
    angle = angle.replacing(
      with: CartPoleEnvironment.randomTensor(), where: needsReset)
    angleDerivative = angleDerivative.replacing(
      with: CartPoleEnvironment.randomTensor(), where: needsReset)
    let newNeedsReset = needsReset.elementsLogicalNot().elementsLogicalOr(
      (position .< -positionThreshold)
        .elementsLogicalOr(position .> positionThreshold)
        .elementsLogicalOr(angle .< -angleThreshold)
        .elementsLogicalOr(angle .> angleThreshold))
    let kind = StepKind((Tensor<Int32>(newNeedsReset) + 1)
      .replacing(with: Tensor<Int32>(zeros: []), where: needsReset))
    let observation = Tensor<Float>(
      concatenating: [position, positionDerivative, angle, angleDerivative],
      alongAxis: -1)
    let reward = Tensor<Float>(ones: action.shape)
    needsReset = newNeedsReset
    return Step(kind: kind, observation: observation, reward: reward)
  }

  /// Resets the environment.
  @discardableResult
  public mutating func reset() -> Step<Tensor<Float>, Tensor<Float>> {
    position = CartPoleEnvironment.randomTensor()
    positionDerivative = CartPoleEnvironment.randomTensor()
    angle = CartPoleEnvironment.randomTensor()
    angleDerivative = CartPoleEnvironment.randomTensor()
    needsReset = Tensor<Bool>([false])
    let observation = Tensor<Float>(
      concatenating: [position, positionDerivative, angle, angleDerivative],
      alongAxis: -1)
    return Step(kind: .first, observation: observation, reward: Tensor<Float>(zeros: []))
  }

  /// Returns a copy of this environment that is reset before being returned.
  public func copy() throws -> CartPoleEnvironment {
    CartPoleEnvironment()
  }
}

extension CartPoleEnvironment {
  private static func randomTensor() -> Tensor<Float> {
    Tensor<Float>(
      randomUniform: [],
      lowerBound: Tensor<Float>(-0.05),
      upperBound: Tensor<Float>(0.05))
  }
}
