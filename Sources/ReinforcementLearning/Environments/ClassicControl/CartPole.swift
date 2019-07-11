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
  public let batched: Bool
  public let actionSpace: Discrete = Discrete(withSize: 2)
  public var observationSpace: ObservationSpace = ObservationSpace()

  private var position: Tensor<Float>
  private var positionDerivative: Tensor<Float>
  private var angle: Tensor<Float>
  private var angleDerivative: Tensor<Float>
  private var needsReset: Tensor<Bool>

  public init(batchSize: Int) {
    self.batchSize = batchSize
    self.batched = batchSize > 1
    self.position = CartPoleEnvironment.randomTensor(withShape: [batchSize])
    self.positionDerivative = CartPoleEnvironment.randomTensor(withShape: [batchSize])
    self.angle = CartPoleEnvironment.randomTensor(withShape: [batchSize])
    self.angleDerivative = CartPoleEnvironment.randomTensor(withShape: [batchSize])
    self.needsReset = Tensor<Bool>(repeating: false, shape: [batchSize])
  }

  /// Updates the environment according to the provided action.
  @discardableResult
  public mutating func step(taking action: Tensor<Int32>) -> Step<Observation, Tensor<Float>> {
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
      with: CartPoleEnvironment.randomTensor(withShape: position.shape),
      where: needsReset)
    positionDerivative = positionDerivative.replacing(
      with: CartPoleEnvironment.randomTensor(withShape: positionDerivative.shape),
      where: needsReset)
    angle = angle.replacing(
      with: CartPoleEnvironment.randomTensor(withShape: angle.shape),
      where: needsReset)
    angleDerivative = angleDerivative.replacing(
      with: CartPoleEnvironment.randomTensor(withShape: angleDerivative.shape),
      where: needsReset)
    let newNeedsReset = (position .< -positionThreshold)
      .elementsLogicalOr(position .> positionThreshold)
      .elementsLogicalOr(angle .< -angleThreshold)
      .elementsLogicalOr(angle .> angleThreshold)
    let kind = StepKind((Tensor<Int32>(newNeedsReset) + 1)
      .replacing(with: Tensor<Int32>(zeros: newNeedsReset.shape), where: needsReset))
    let observation = Observation(
      position: position,
      positionDerivative: positionDerivative,
      angle: angle,
      angleDerivative: angleDerivative)
    let reward = Tensor<Float>(ones: action.shape)
    needsReset = newNeedsReset
    return Step(kind: kind, observation: observation, reward: reward)
  }

  /// Resets the environment.
  @discardableResult
  public mutating func reset() -> Step<Observation, Tensor<Float>> {
    position = CartPoleEnvironment.randomTensor()
    positionDerivative = CartPoleEnvironment.randomTensor()
    angle = CartPoleEnvironment.randomTensor()
    angleDerivative = CartPoleEnvironment.randomTensor()
    needsReset = Tensor<Bool>(repeating: false, shape: [batchSize])
    let observation = Observation(
      position: position,
      positionDerivative: positionDerivative,
      angle: angle,
      angleDerivative: angleDerivative)
    return Step(kind: .first, observation: observation, reward: Tensor<Float>(zeros: []))
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

      public func mode(
        usingSeed seed: TensorFlowSeed = Context.local.randomSeed
      ) -> Observation {
        Observation(
          position: positionDistribution.mode(usingSeed: seed),
          positionDerivative: positionDerivativeDistribution.mode(usingSeed: seed),
          angle: angleDistribution.mode(usingSeed: seed),
          angleDerivative: angleDerivativeDistribution.mode(usingSeed: seed))
      }

      public func sample(
        usingSeed seed: TensorFlowSeed = Context.local.randomSeed
      ) -> Observation {
        Observation(
          position: positionDistribution.sample(usingSeed: seed),
          positionDerivative: positionDerivativeDistribution.sample(usingSeed: seed),
          angle: angleDistribution.sample(usingSeed: seed),
          angleDerivative: angleDerivativeDistribution.sample(usingSeed: seed))
      }
    }
  }
}

extension CartPoleEnvironment.Observation: Stackable {
  public typealias Observation = CartPoleEnvironment.Observation

  public static func stack(_ values: [Observation]) -> Observation {
    Observation(
      position: Tensor(stacking: values.map { $0.position }, alongAxis: 0),
      positionDerivative: Tensor(stacking: values.map { $0.positionDerivative }, alongAxis: 0),
      angle: Tensor(stacking: values.map { $0.angle }, alongAxis: 0),
      angleDerivative: Tensor(stacking: values.map { $0.angleDerivative }, alongAxis: 0))
  }

  public func unstacked() -> [Observation] {
    let positions = position.unstacked(alongAxis: 0)
    let positionDerivatives = positionDerivative.unstacked(alongAxis: 0)
    let angles = angle.unstacked(alongAxis: 0)
    let angleDerivatives = angleDerivative.unstacked(alongAxis: 0)
    // TODO: Make this more efficient with a zip operation.
    var observations = [Observation]()
    for i in positions.indices {
      observations.append(Observation(
        position: positions[i],
        positionDerivative: positionDerivatives[i],
        angle: angles[i],
        angleDerivative: angleDerivatives[i]))
    }
    return observations
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
