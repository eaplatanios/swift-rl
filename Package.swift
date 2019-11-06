// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
import Foundation
import PackageDescription

let package = Package(
    name: "ReinforcementLearning",
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
            name: "CGLFW",
            path: "Sources/CGLFW",
            pkgConfig: "glfw3",
            providers: [
                .brew(["--HEAD git glfw3"]),
                .apt(["libglfw3", "libglfw3-dev"])
            ]),
        .systemLibrary(
            name: "CVulkan",
            path: "Sources/CVulkan",
            pkgConfig: "vulkan"),
        .target(
            name: "ReinforcementLearning",
            dependencies: ["CGLFW", "CVulkan"],
            path: "Sources/ReinforcementLearning"),
        .target(
            name: "ReinforcementLearningExperiments",
            dependencies: ["Logging", "ReinforcementLearning"])
    ]
)
