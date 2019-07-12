// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import TensorFlow

fileprivate let gravity: Float = 9.8
fileprivate let cartMass: Float = 1.0
fileprivate let poleMass: Float = 0.1
fileprivate let length: Float = 0.5
fileprivate let forceMagnitude: Float = 10.0
fileprivate let secondCountBetweenUpdates: Float = 0.02
fileprivate let angleThreshold: Float = 12 * 2 * Float.pi / 360
fileprivate let positionThreshold: Float = 2.4
fileprivate let totalMass: Float = cartMass + poleMass
fileprivate let poleMassLength: Float = poleMass * length

public struct CartPoleEnvironment: Environment {
  public let batchSize: Int
  public let actionSpace: Discrete = Discrete(withSize: 2)
  public var observationSpace: ObservationSpace = ObservationSpace()

  private var step: Step<Observation, Tensor<Float>>
  private var needsReset: Tensor<Bool>

  public init(batchSize: Int) {
    self.batchSize = batchSize
    self.step = Step(
      kind: StepKind.first(batchSize: batchSize),
      observation: Observation(
        position: CartPoleEnvironment.randomTensor(withShape: [batchSize]),
        positionDerivative: CartPoleEnvironment.randomTensor(withShape: [batchSize]),
        angle: CartPoleEnvironment.randomTensor(withShape: [batchSize]),
        angleDerivative: CartPoleEnvironment.randomTensor(withShape: [batchSize])),
      reward: Tensor<Float>(ones: [batchSize]))
    self.needsReset = Tensor<Bool>(repeating: false, shape: [batchSize])
  }

  public func currentStep() -> Step<Observation, Reward> {
    step
  }

  /// Updates the environment according to the provided action.
  @discardableResult
  public mutating func step(taking action: Tensor<Int32>) -> Step<Observation, Tensor<Float>> {
    precondition(actionSpace.contains(action), "Invalid action provided.")
    var position = step.observation.position
    var positionDerivative = step.observation.positionDerivative
    var angle = step.observation.angle
    var angleDerivative = step.observation.angleDerivative

    // Calculate the updates to the pole position, angle, and their derivatives.
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
    step.observation.position = position.replacing(
      with: CartPoleEnvironment.randomTensor(withShape: position.shape),
      where: needsReset)
    step.observation.positionDerivative = positionDerivative.replacing(
      with: CartPoleEnvironment.randomTensor(withShape: positionDerivative.shape),
      where: needsReset)
    step.observation.angle = angle.replacing(
      with: CartPoleEnvironment.randomTensor(withShape: angle.shape),
      where: needsReset)
    step.observation.angleDerivative = angleDerivative.replacing(
      with: CartPoleEnvironment.randomTensor(withShape: angleDerivative.shape),
      where: needsReset)
    let newNeedsReset = (step.observation.position .< -positionThreshold)
      .elementsLogicalOr(step.observation.position .> positionThreshold)
      .elementsLogicalOr(step.observation.angle .< -angleThreshold)
      .elementsLogicalOr(step.observation.angle .> angleThreshold)
    step.kind.rawValue = (Tensor<Int32>(newNeedsReset) + 1)
      .replacing(with: Tensor<Int32>(zeros: newNeedsReset.shape), where: needsReset)
    // Rewards need not be updated because they are always equal to one.
    needsReset = newNeedsReset
    return step
  }

  /// Resets the environment.
  @discardableResult
  public mutating func reset() -> Step<Observation, Tensor<Float>> {
    step.kind = StepKind.first(batchSize: batchSize)
    step.observation.position = CartPoleEnvironment.randomTensor(withShape: [batchSize])
    step.observation.positionDerivative = CartPoleEnvironment.randomTensor(withShape: [batchSize])
    step.observation.angle = CartPoleEnvironment.randomTensor(withShape: [batchSize])
    step.observation.angleDerivative = CartPoleEnvironment.randomTensor(withShape: [batchSize])
    needsReset = Tensor<Bool>(repeating: false, shape: [batchSize])
    return step
  }

  /// Returns a copy of this environment that is reset before being returned.
  public func copy() -> CartPoleEnvironment {
    CartPoleEnvironment(batchSize: batchSize)
  }
}

extension CartPoleEnvironment {
  private static func randomTensor(withShape shape: TensorShape = []) -> Tensor<Float> {
    Tensor<Float>(
      randomUniform: shape,
      lowerBound: Tensor<Float>(-0.05),
      upperBound: Tensor<Float>(0.05))
  }

  public struct Observation: Differentiable, KeyPathIterable {
    public var position: Tensor<Float>
    public var positionDerivative: Tensor<Float>
    public var angle: Tensor<Float>
    public var angleDerivative: Tensor<Float> 
  }

  public struct ObservationSpace: Space {
    public typealias Scalar = Int32

    public let shape: TensorShape
    public let distribution: ValueDistribution

    public init() {
      self.shape = [4]
      self.distribution = ValueDistribution()
    }

    public var description: String {
      "CartPoleObservation"
    }

    public func contains(_ value: Observation) -> Bool {
      true
    }

    public struct ValueDistribution: DifferentiableDistribution, KeyPathIterable {
      private var positionDistribution: Uniform<Float> = Uniform<Float>(
        lowerBound: Tensor<Float>(0),
        upperBound: Tensor<Float>(positionThreshold * 2))
      private var positionDerivativeDistribution: Uniform<Float> = Uniform<Float>(
        lowerBound: Tensor<Float>(0),
        upperBound: Tensor<Float>(Float.greatestFiniteMagnitude))
      private var angleDistribution: Uniform<Float> = Uniform<Float>(
        lowerBound: Tensor<Float>(0),
        upperBound: Tensor<Float>(angleThreshold * 2))
      private var angleDerivativeDistribution: Uniform<Float> = Uniform<Float>(
        lowerBound: Tensor<Float>(0),
        upperBound: Tensor<Float>(Float.greatestFiniteMagnitude))

      @differentiable(wrt: self)
      public func logProbability(of value: Observation) -> Tensor<Float> {
        positionDistribution.logProbability(of: value.position) +
          positionDerivativeDistribution.logProbability(of: value.positionDerivative) +
          angleDistribution.logProbability(of: value.angle) +
          angleDerivativeDistribution.logProbability(of: value.angleDerivative)
      }

      @differentiable(wrt: self)
      public func entropy() -> Tensor<Float> {
        positionDistribution.entropy() +
          positionDerivativeDistribution.entropy() +
          angleDistribution.entropy() +
          angleDerivativeDistribution.entropy()
      }

      public func mode() -> Observation {
        Observation(
          position: positionDistribution.mode(),
          positionDerivative: positionDerivativeDistribution.mode(),
          angle: angleDistribution.mode(),
          angleDerivative: angleDerivativeDistribution.mode())
      }

      public func sample() -> Observation {
        Observation(
          position: positionDistribution.sample(),
          positionDerivative: positionDerivativeDistribution.sample(),
          angle: angleDistribution.sample(),
          angleDerivative: angleDerivativeDistribution.sample())
      }
    }
  }
}

#if GLFW

public struct CartPoleRenderer: Renderer, GLFWScene {
  public let windowWidth: Int
  public let windowHeight: Int
  public let worldWidth: Float
  public let scale: Float
  public let cartTop: Float
  public let poleWidth: Float
  public let poleLength: Float
  public let cartWidth: Float
  public let cartHeight: Float

  @usableFromInline internal var window: GLFWWindow
  @usableFromInline internal var cart: GLFWGeometry
  @usableFromInline internal var pole: GLFWGeometry
  @usableFromInline internal var axle: GLFWGeometry
  @usableFromInline internal var track: GLFWGeometry
  @usableFromInline internal var cartTransform: GLFWTransform
  @usableFromInline internal var poleTransform: GLFWTransform

  public init(
    windowWidth: Int = 600,
    windowHeight: Int = 400,
    positionThreshold: Float = 2.4,
    cartTop: Float = 100.0,
    poleWidth: Float = 10.0,
    cartWidth: Float = 50.0,
    cartHeight: Float = 30.0
  ) {
    self.windowWidth = windowWidth
    self.windowHeight = windowHeight
    self.worldWidth = positionThreshold * 2
    self.scale = Float(windowWidth) / worldWidth
    self.cartTop = cartTop
    self.poleWidth = poleWidth
    self.poleLength = scale
    self.cartWidth = cartWidth
    self.cartHeight = cartHeight

    // Create the GLFW window along with all the shapes.
    self.window = try! GLFWWindow(
        name: "CartPole Environment",
        width: windowWidth,
        height: windowHeight,
        framesPerSecond: 60)
    let (cl, cr, ct, cb) = (
      -cartWidth / 2, cartWidth / 2,
      cartHeight / 2, -cartHeight / 2)
    self.cart = GLFWPolygon(vertices: [(cl, cb), (cl, ct), (cr, ct), (cr, cb)])
    self.cartTransform = GLFWTransform()
    self.cart.attributes.append(cartTransform)
    let (pl, pr, pt, pb) = (
      -poleWidth / 2, poleWidth / 2,
      poleLength - poleWidth / 2, -poleWidth / 2)
    self.pole = GLFWPolygon(vertices: [(pl, pb), (pl, pt), (pr, pt), (pr, pb)])
    self.pole.attributes.append(GLFWColor(red: 0.8, green: 0.6, blue: 0.4))
    self.poleTransform = GLFWTransform(translation: (0.0, cartHeight / 4))
    self.pole.attributes.append(poleTransform)
    self.pole.attributes.append(cartTransform)
    let axleVertices = (0..<30).map { i -> (Float, Float) in
      let angle = 2 * Float.pi * Float(i) / Float(30)
      return (cos(angle) * poleWidth / 2, sin(angle) * poleWidth / 2)
    }
    self.axle = GLFWPolygon(vertices: axleVertices)
    self.axle.attributes.append(poleTransform)
    self.axle.attributes.append(cartTransform)
    self.axle.attributes.append(GLFWColor(red: 0.5, green: 0.5, blue: 0.8))
    self.track = GLFWLine(start: (0.0, cartTop), end: (Float(windowWidth), cartTop))
    self.track.attributes.append(GLFWColor(red: 0, green: 0, blue: 0))
  }

  public func draw() {
    cart.renderWithAttributes()
    pole.renderWithAttributes()
    axle.renderWithAttributes()
    track.renderWithAttributes()
  }

  @inlinable
  public mutating func render(_ data: CartPoleEnvironment.Observation) throws {
    // TODO: Support batched environments.
    let position = data.position[0].scalarized()
    let angle = data.angle[0].scalarized()
    cartTransform.translation = (position * scale + Float(windowWidth) / 2, cartTop)
    poleTransform.rotation = -angle
    window.render(scene: self)
  }
}

#endif
