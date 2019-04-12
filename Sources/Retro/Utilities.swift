import Foundation

public enum RetroError: Error {
  case UnsupportedROMFile(URL)
  case UnsupportedCore(String)
  case GameDataFailure(String)
  case GameROMNotFound(String)
  case GameROMTooBig(String)
  case GLFWError(String)
}
