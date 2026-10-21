// swift-tools-version:6.0
import PackageDescription

// ONNX Runtime is not a Swift package: scripts/get-ort downloads Microsoft's release into .ort/,
// and scripts/swift passes its lib folder to the linker. libjpeg comes from the system (libjpeg-turbo8-dev).
let package = Package(
    name: "Lineage",
    products: [.executable(name: "lineage", targets: ["lineage"])],
    targets: [
        .systemLibrary(name: "CORT", path: "Sources/CORT"),
        .systemLibrary(name: "CJPEG", path: "Sources/CJPEG"),
        // Plain Swift with no Foundation: the parts that also compile to WebAssembly for the page.
        .target(name: "LineageCore"),
        .target(name: "LineageTrain", dependencies: ["LineageCore", "CORT", "CJPEG"]),
        .executableTarget(name: "lineage", dependencies: ["LineageTrain"],
                          swiftSettings: [.unsafeFlags(["-Ounchecked"])]),
        .testTarget(name: "LineageTests", dependencies: ["LineageCore", "LineageTrain"]),
    ],
    swiftLanguageModes: [.v5]
)
