import CRetro
import Foundation
import TensorFlow

/// Represents different types of action space for the environment.
public protocol RetroActions {
  associatedtype ActionSpace: Space

  func space(
    for gameData: GameData,
    buttons: [String?],
    numPlayers: UInt32
  ) -> ActionSpace

  func encodeAction(
    _ action: ShapedArray<ActionSpace.Scalar>,
    for gameData: GameData,
    buttons: [String?],
    player: UInt32
  ) -> UInt16
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

  public func encodeAction(
    _ action: ShapedArray<Int32>,
    for gameData: GameData,
    buttons: [String?],
    player: UInt32
  ) -> UInt16 {
    let startIndex = buttons.count * Int(player)
    let endIndex = buttons.count * Int(player + 1)
    let playerAction = action[startIndex..<endIndex].scalars
    var encodedAction = UInt16(0)
    for i in 0..<playerAction.count {
      encodedAction |= UInt16(playerAction[i]) << i
    }
    return encodedAction
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

  public func encodeAction(
    _ action: ShapedArray<Int32>,
    for gameData: GameData,
    buttons: [String?],
    player: UInt32
  ) -> UInt16 {
    let startIndex = buttons.count * Int(player)
    let endIndex = buttons.count * Int(player + 1)
    let playerAction = action[startIndex..<endIndex].scalars
    var encodedAction = UInt16(0)
    for i in 0..<playerAction.count {
      encodedAction |= UInt16(playerAction[i]) << i
    }
    return gameDataFilterAction(gameData.handle, encodedAction)
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

  public func encodeAction(
    _ action: ShapedArray<Int32>,
    for gameData: GameData,
    buttons: [String?],
    player: UInt32
  ) -> UInt16 {
    var playerAction = UInt16(action.scalar!)
    var encodedAction = UInt16(0)
    var current = 0
    for combo in gameData.buttonCombos {
      current = Int(playerAction) % combo.count
      playerAction /= UInt16(combo.count)
      encodedAction |= UInt16(combo[current])
    }
    return encodedAction
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

  public func encodeAction(
    _ action: ShapedArray<Int32>,
    for gameData: GameData,
    buttons: [String?],
    player: UInt32
  ) -> UInt16 {
    let startIndex = buttons.count * Int(player)
    let endIndex = buttons.count * Int(player + 1)
    let playerAction = action[startIndex..<endIndex].scalars
    var encodedAction = UInt16(0)
    for i in 0..<playerAction.count {
      let combo = gameData.buttonCombos[i]
      encodedAction |= UInt16(combo[Int(playerAction[i])])
    }
    return encodedAction
  }
}
