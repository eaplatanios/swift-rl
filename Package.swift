// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
import Foundation
import PackageDescription

let package = Package(
    name: "ReinforcementLearning",
    platforms: [.macOS(.v10_12)],
    products: [
        .library(
            name: "ReinforcementLearning",
            targets: ["ReinforcementLearning"]),
        .executable(
            name: "ReinforcementLearningExperiments",
            targets: ["ReinforcementLearningExperiments"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0")
    ],
    targets: [
        .systemLibrary(
            name: "GLFW",
            path: "Sources/GLFW",
            pkgConfig: "glfw3",
            providers: [
                .brew(["--HEAD git glfw3"]),
                .apt(["libglfw3", "libglfw3-dev"])
            ]),
        .target(
            name: "ReinforcementLearning",
            dependencies: ["GLFW"],
            path: "Sources/ReinforcementLearning"),
        .target(
            name: "ReinforcementLearningExperiments",
            dependencies: ["Logging", "ReinforcementLearning"])
    ]
)
