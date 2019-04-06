import CRetro
import Foundation

public class Movie {
  private var handle: UnsafeMutablePointer<CMovie>?

  public init(at url: URL, recording: Bool, numPlayers: UInt32) {
    self.handle = movieCreate(url.path, recording, numPlayers)
  }

  deinit {
    movieDelete(handle)
  }

  public var state: Data {
    get {
      let cBytes = movieGetState(handle)!.pointee
      return Data(bytes: cBytes.bytes, count: cBytes.numBytes)
    }
    set (newValue) {
      let bytes = [UInt8](newValue)      
      var cBytes = UnsafeMutablePointer<UInt8>.allocate(capacity: bytes.count)
      // TODO !!!: Need to provide a deallocator maybe?
      defer { cBytes.deallocate() }
      cBytes.initialize(from: bytes, count: bytes.count)
      var cBytesStruct = CBytes(bytes: cBytes, numBytes: bytes.count)
      movieSetState(handle, &cBytesStruct)
    }
  }

  public func configure<A: RetroActions>(for environment: Environment<A>) {
    movieConfigure(handle, environment.game, environment.emulatorHandle)
  }

  public func game() -> String {
    return String(cString: movieGetGameName(handle))
  }

  public func step() -> Bool {
    return movieStep(handle)
  }

  public func close() {
    movieClose(handle)
  }

  public func numPlayers() -> UInt32 {
    return moviePlayers(handle)
  }

  public func get(key: Int32, forPlayer player: UInt32) -> Bool {
    return movieGetKey(handle, key, player)
  }

  public func set(key: Int32, forPlayer player: UInt32, to value: Bool) {
    movieSetKey(handle, key, value, player)
  }
}
