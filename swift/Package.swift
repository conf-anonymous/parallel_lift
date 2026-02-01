// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "ParallelLift",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "parallel-lift", targets: ["ParallelLift"])
    ],
    dependencies: [
        .package(url: "https://github.com/attaswift/BigInt.git", from: "5.3.0")
    ],
    targets: [
        .executableTarget(
            name: "ParallelLift",
            dependencies: ["BigInt"],
            resources: [
                .process("Metal")
            ]
        )
    ]
)
