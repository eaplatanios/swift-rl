// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
import Foundation
import PackageDescription

let package = Package(
    name: "Retro",
    products: [
        .library(
            name: "Retro",
            targets: ["Retro"]),
    ],
    dependencies: [
        .package(url: "https://github.com/Flight-School/AnyCodable.git", from: "0.1.0"),
        .package(url: "https://github.com/1024jp/GzipSwift.git", from: "4.1.0")
    ],
    targets: [
        .target(
            name: "CRetro",
            path: ".", 
            sources: ["Sources/CRetro"],
            publicHeadersPath: "Sources/CRetro/include",
            linkerSettings: [
                .linkedLibrary("retro"),
                .linkedLibrary("retro-base"),
                .unsafeFlags(["-L", "retro"])
            ]),
        .testTarget(name: "RetroTests", dependencies: ["Retro"]),
        .target(
            name: "Retro",
            dependencies: ["CRetro", "AnyCodable", "Gzip"],
            path: "Sources/Retro")
    ]
)