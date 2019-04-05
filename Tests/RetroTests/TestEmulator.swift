import XCTest
@testable import CRetro
@testable import Retro

class EmulatorTests: XCTestCase {
  let emulatorConfig: EmulatorConfig = {
    let retroURL = URL(fileURLWithPath: "/Users/eaplatanios/Development/GitHub/retro-swift/retro")
    return EmulatorConfig(
      coresInformationPath: retroURL.appendingPathComponent("cores"),
      coresPath: retroURL.appendingPathComponent("retro/cores"),
      gameDataPath: retroURL.appendingPathComponent("retro/data"))
  }()

  override func setUp() {
    try! initializeRetro(withConfig: emulatorConfig)
  }

  func testSupportedCores() {
    XCTAssert(supportedCores.keys.contains("Atari2600"))
    XCTAssert(supportedCores["Atari2600"]!.library == "stella")
    XCTAssert(supportedCores["Atari2600"]!.extensions == ["a26"])
    XCTAssert(supportedCores["Atari2600"]!.memorySize == 128)
    XCTAssert(supportedCores["Atari2600"]!.keyBinds == [
      "Z", nil, "TAB", "ENTER", "UP", "DOWN", "LEFT", "RIGHT"])
    XCTAssert(supportedCores["Atari2600"]!.buttons == [
      "BUTTON", nil, "SELECT", "RESET", "UP", "DOWN", "LEFT", "RIGHT"])
    XCTAssert(supportedCores["Atari2600"]!.actions == [
      [[], ["UP"], ["DOWN"]],
      [[], ["LEFT"], ["RIGHT"]],
      [[], ["BUTTON"]]])
  }

  func testSupportedExtensions() {
    XCTAssert(supportedExtensions.keys.contains(".a26"))
    XCTAssert(supportedExtensions[".a26"]! == "Atari2600")
  }

  func testGames() {
    print(emulatorConfig.states(for: "Pong-Atari2600"))
    XCTAssert(emulatorConfig.games().contains("Pong-Atari2600"))
  }

	// func testEmulatorScreenRate() {
  //   let romPath = "/Users/eaplatanios/Development/GitHub/retro-swift/retro/tests/roms/Dekadence-Dekadrive.md"
  //   let emulator = emulatorCreate(romPath)
  //   let screenRate = emulatorGetScreenRate(emulator)
  //   XCTAssertEqual(screenRate, 0.0)
  // }
}

#if os(Linux)
extension EmulatorTests {
  static var allTests : [(String, (EmulatorTests) -> () throws -> Void)] {
    return [
      ("testEmulatorScreenRate", testEmulatorScreenRate)
    ]
  }
}
#endif
