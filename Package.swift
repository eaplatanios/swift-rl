// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
import Foundation
import PackageDescription

let package = Package(
    name: "Retro",
    platforms: [.macOS(.v10_12)],
    products: [
        .library(
            name: "Gym",
            targets: ["Gym"]),
        .library(
            name: "ReinforcementLearning",
            targets: ["ReinforcementLearning"]),
        .library(
            name: "Retro",
            targets: ["Retro"])
    ],
    dependencies: [
        .package(url: "https://github.com/Flight-School/AnyCodable.git", from: "0.1.0"),
        .package(url: "https://github.com/krzyzanowskim/CryptoSwift.git", from: "1.0.0"),
        .package(url: "https://github.com/1024jp/GzipSwift.git", from: "4.1.0"),
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", .branch("master"))
    ],
    targets: [
        .target(
            name: "GLFW",
            path: ".",
            sources: ["Sources/GLFW"],
            publicHeadersPath: "Sources/GLFW/include",
            linkerSettings: [
                .linkedLibrary("glfw"),
                .unsafeFlags(["-L/usr/local/lib"])]),
        .target(
            name: "Gym",
            dependencies: ["CryptoSwift", "GLFW"],
            path: "Sources/Gym",
            swiftSettings: [.define("GLFW")]),
        .target(
            name: "ReinforcementLearning",
            dependencies: ["Gym"],
            path: "Sources/ReinforcementLearning"),
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
            dependencies: [
                "CRetro", "ReinforcementLearning",
                "AnyCodable", "CryptoSwift", "Gzip", "ZIPFoundation"],
            path: "Sources/Retro"),
        .testTarget(
            name: "RetroTests",
            dependencies: ["Retro"],
            swiftSettings: [.define("GLFW")])
    ]
)
