// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
import Foundation
import PackageDescription

let package = Package(
    name: "Retro",
    platforms: [.macOS(.v10_12)],
    products: [
        .library(
            name: "ReinforcementLearning",
            targets: ["ReinforcementLearning"]),
        .executable(
            name: "ReinforcementLearningExperiments",
            targets: ["ReinforcementLearningExperiments"]),
        .library(
            name: "JellyBeanWorld",
            targets: ["JellyBeanWorld"]),
        .library(
            name: "Retro",
            targets: ["Retro"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
        .package(url: "https://github.com/eaplatanios/nel_framework.git", .branch("master")),
        .package(url: "https://github.com/1024jp/GzipSwift.git", from: "4.1.0"),
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", .branch("master"))
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
            dependencies: ["JellyBeanWorld", "Logging", "ReinforcementLearning", "Retro"]),
        .target(
            name: "JellyBeanWorld",
            dependencies: ["NELFramework", "ReinforcementLearning"],
            path: "Sources/JellyBeanWorld"),
        .target(
            name: "CRetro",
            path: ".",
            sources: ["Sources/CRetro"],
            publicHeadersPath: "Sources/CRetro/include",
            linkerSettings: [
                .linkedLibrary("retro"),
                .unsafeFlags(["-L./retro"])]),
        .target(
            name: "Retro",
            dependencies: ["CRetro", "ReinforcementLearning", "Gzip", "ZIPFoundation"],
            path: "Sources/Retro"),
        .testTarget(
            name: "RetroTests",
            dependencies: ["Retro"])
    ]
)
