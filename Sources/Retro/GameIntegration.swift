
public struct GameIntegration {
  public let paths: [String]
  public let name: String

  public init(paths: [String], name: String = "custom") {
    self.paths = paths
    self.name = name
  }
}

public extension GameIntegration {
  static let stable = GameIntegration(paths: ["stable"], name: "stable")
  static let experimental = GameIntegration(paths: ["experimental"], name: "experimental")
  static let contrib = GameIntegration(paths: ["contrib"], name: "contrib")
  static let all = (stable + experimental + contrib).named(as: "all")
}

public extension GameIntegration {
  func named(as name: String) -> GameIntegration {
    return GameIntegration(paths: paths, name: name)
  }

  static func +(left: GameIntegration, right: GameIntegration) -> GameIntegration {
    return GameIntegration(paths: left.paths + right.paths, name: "\(left.name) | \(right.name)")
  }
}
