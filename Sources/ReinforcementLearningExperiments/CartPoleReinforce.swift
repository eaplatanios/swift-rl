import ReinforcementLearning

fileprivate struct ActorNetwork: Network {
  @noDerivative public var state: None = None()

  public var dense: Dense<Float> = Dense<Float>(inputSize: 4, outputSize: 2)

  public func initialize(using input: CartPoleEnvironment.Observation) {}

  @differentiable
  public func callAsFunction(_ input: CartPoleEnvironment.Observation) -> Categorical<Int32> {
    let stackedInput = Tensor<Float>(
      stacking: [
        input.position, input.positionDerivative,
        input.angle, input.angleDerivative],
      alongAxis: input.position.rank)
    let outerDimCount = stackedInput.rank - 1
    let flattenedBatchStackedInput = stackedInput.flattenedBatch(outerDimCount: outerDimCount)
    let logits = dense(flattenedBatchStackedInput)
    let flattenedDistribution = Categorical<Int32>(logits: logits)
    return flattenedDistribution.unflattenedBatch(
      outerDims: [Int](stackedInput.shape.dimensions[0..<outerDimCount]))
  }

  public func copy() -> ActorNetwork {
    ActorNetwork()
  }
}

public func runCartPoleReinforce() {
  let batchSize = 32
  let maxSequenceLength = 2000

  var renderer = CartPoleRenderer()
  var environment = CartPoleEnvironment(batchSize: batchSize)

  let actorPolicy = ActorPolicy(
    for: environment,
    actorNetwork: ActorNetwork(),
    observationsNormalizer: { $0 },
    randomSeed: (1234, 5678))
  var agent = ReinforceAgent(
    actorPolicy: actorPolicy,
    optimizer: AMSGrad(for: actorPolicy),
    discountFactor: 0.9,
    rewardsNormalizer: { $0 },
    entropyRegularizationWeight: 0.0)
  var replayBuffer = UniformReplayBuffer(
    for: environment,
    using: agent.policy,
    batchSize: batchSize,
    maxLength: maxSequenceLength)
  var driver = StepBasedDriver(
    for: environment,
    using: agent.policy,
    maxEpisodes: 100,
    batchSize: 32)

  agent.initialize()

  for step in 0..<1000 {
    driver.run(using: environment.reset(), updating: [{ trajectoryStep in
      replayBuffer.record(trajectoryStep)
      try! environment.render(
        observation: trajectoryStep.currentStep.observation,
        using: &renderer)
    }])
    let loss = agent.update(using: replayBuffer.recordedData())
    print("Step \(step) loss: \(loss)")
    replayBuffer.reset()
  }
}
