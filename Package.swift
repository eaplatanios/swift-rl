// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
import Foundation
import PackageDescription

let package = Package(
    name: "Retro",
    products: [
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
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", .upToNextMajor(from: "0.9.8"))
    ],
    targets: [
        .target(
            name: "ReinforcementLearning",
            dependencies: ["CryptoSwift"],
            path: "Sources/ReinforcementLearning"),
        .target(
            name: "CRetro",
            path: ".", 
            sources: ["Sources/CRetro"],
            publicHeadersPath: "Sources/CRetro/include",
            linkerSettings: [.linkedLibrary("retro")]),
        .target(
            name: "Retro",
            dependencies: [
                "CRetro", "ReinforcementLearning",
                "AnyCodable", "CryptoSwift", "Gzip", "ZIPFoundation"],
            path: "Sources/Retro"),
        .testTarget(name: "RetroTests", dependencies: ["Retro"])
    ]
)
