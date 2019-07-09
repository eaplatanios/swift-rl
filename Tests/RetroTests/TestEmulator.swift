import XCTest
@testable import CRetro
@testable import ReinforcementLearning
@testable import Retro

class EmulatorTests: XCTestCase {
  let emulatorConfig: RetroEmulator.Config = {
    let retroURL = URL(fileURLWithPath: "/Users/eaplatanios/Development/GitHub/retro-swift")
    return try! RetroEmulator.Config(
      coreInformationLookupPath: retroURL.appendingPathComponent("retro/cores"),
      coreLookupPathHint: retroURL.appendingPathComponent("retro/retro/cores"),
      gameDataLookupPathHint: retroURL.appendingPathComponent("retro/retro/data"),
      gameROMLookupPaths: [URL(fileURLWithPath: "temp")],
      gameRomsDownloadPath: URL(fileURLWithPath: "temp").appendingPathComponent("downloads"))
  }()

  func testSupportedCores() {
    XCTAssert(supportedCores.keys.contains("Atari2600"))
    XCTAssert(supportedCores["Atari2600"]!.information.library == "stella")
    XCTAssert(supportedCores["Atari2600"]!.information.extensions == ["a26"])
    XCTAssert(supportedCores["Atari2600"]!.information.memorySize == 128)
    XCTAssert(supportedCores["Atari2600"]!.information.keyBinds == [
      "Z", nil, "TAB", "ENTER", "UP", "DOWN", "LEFT", "RIGHT"])
    XCTAssert(supportedCores["Atari2600"]!.information.buttons == [
      "BUTTON", nil, "SELECT", "RESET", "UP", "DOWN", "LEFT", "RIGHT"])
    XCTAssert(supportedCores["Atari2600"]!.information.actions == [
      [[], ["UP"], ["DOWN"]],
      [[], ["LEFT"], ["RIGHT"]],
      [[], ["BUTTON"]]])
  }

  func testSupportedExtensions() {
    XCTAssert(supportedExtensions.keys.contains(".a26"))
    XCTAssert(supportedExtensions[".a26"]!.name == "Atari2600")
  }

  func testGames() {
    XCTAssert(emulatorConfig.games().contains(where: { $0.name == "Pong-Atari2600" }))

    #if GLFW
    var renderer = try! TensorImageRenderer(initialMaxWidth: 800)
    #else
    var renderer = TensorPrinter<UInt8>(maxEntries: 10)
    #endif

    // let game = emulatorConfig.game(called: "Airstriker-Genesis")!
    let game = emulatorConfig.game(called: "SpaceInvaders-Atari2600")!
    let emulator = try! RetroEmulator(for: game, configuredAs: emulatorConfig)
    var environment = try! RetroEnvironment(using: emulator, actionsType: FilteredActions())
    let policy = RandomPolicy(for: environment)
    var driver = StepBasedDriver(for: environment, using: policy, maxSteps: 1000000, batchSize: 1)
    driver.run(using: environment.reset(), updating: [{
      try! environment.render(
        observation: $0.currentStep.observation,
        using: &renderer)
    }])
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
      ("testSupportedCores", testSupportedCores),
      ("testSupportedExtensions", testSupportedExtensions),
      ("testGames", testGames)
    ]
  }
}
#endif
