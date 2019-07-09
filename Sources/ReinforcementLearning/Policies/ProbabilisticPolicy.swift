
public protocol ProbabilisticPolicy: Policy {
  associatedtype ActionDistribution: Distribution where ActionDistribution.Value == Action

  var randomSeed: TensorFlowSeed { get }

  /// Generates the distribution over next actions given the current environment step.
  func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution
}

public extension ProbabilisticPolicy {
  func action(for step: Step<Observation, Reward>) -> Action {
    let distribution = actionDistribution(for: step)
    return distribution.sample(usingSeed: randomSeed)
  }
}
