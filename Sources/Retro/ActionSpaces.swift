import Foundation

/// Represents different types of action space for the environment.
public protocol RetroActions {
  associatedtype ActionSpace: Space

  func space(
    for gameData: GameData,
    buttons: [String?],
    numPlayers: UInt32
  ) -> ActionSpace
}

/// Multi-binary action space with no filtered actions.
public struct FullRetroActions: RetroActions {
  public typealias ActionSpace = MultiBinary

  public func space(
    for gameData: GameData,
    buttons: [String?],
    numPlayers: UInt32
  ) -> MultiBinary {
    return MultiBinary(withSize: Int32(buttons.count) * Int32(numPlayers))
  }
}

/// Multi-binary action space with invalid or not allowed actions filtered out.
public struct FilteredRetroActions: RetroActions {
  public typealias ActionSpace = MultiBinary

  public func space(
    for gameData: GameData,
    buttons: [String?],
    numPlayers: UInt32
  ) -> MultiBinary {
    return MultiBinary(withSize: Int32(buttons.count) * Int32(numPlayers))
  }
}

/// Discrete action space for filtered actions.
public struct DiscreteRetroActions: RetroActions {
  public typealias ActionSpace = Discrete

  public func space(
    for gameData: GameData,
    buttons: [String?],
    numPlayers: UInt32
  ) -> Discrete {
    let numCombos = gameData.buttonCombos.map { Int32($0.count) } .reduce(1, *)
    return Discrete(withSize: Int32(pow(Float(numCombos), Float(numPlayers))))
  }
}

/// Multi-discete action space for filtered actions.
public struct MultiDiscreteRetroActions: RetroActions {
  public typealias ActionSpace = MultiDiscrete

  public func space(
    for gameData: GameData,
    buttons: [String?],
    numPlayers: UInt32
  ) -> MultiDiscrete {
    return MultiDiscrete(withSizes: gameData.buttonCombos.map {
      Int32($0.count) * Int32(numPlayers)
    })
  }
}
