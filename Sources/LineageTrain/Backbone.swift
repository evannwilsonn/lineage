import CORT
import Foundation

public struct ORTError: Error, CustomStringConvertible {
    public let description: String
}

/// The pretrained MobileNetV4 backbone, run through ONNX Runtime's C API.
/// Input: N x 3 x side x side standardised floats. Output: N x 1280 embeddings.
public final class Backbone {
    public static let dims = 1280
    public static let side = 256

    private let api: OrtApi
    private var env: OpaquePointer?
    private var options: OpaquePointer?
    private var session: OpaquePointer?
    private var memory: OpaquePointer?

    public init(modelPath: String, threads: Int = 2) throws {
        guard let base = OrtGetApiBase(), let apiPtr = base.pointee.GetApi(UInt32(ORT_API_VERSION)) else {
            throw ORTError(description: "ONNX Runtime does not offer API version \(ORT_API_VERSION)")
        }
        api = apiPtr.pointee
        try check(api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "lineage", &env))
        try check(api.CreateSessionOptions(&options))
        try check(api.SetIntraOpNumThreads(options, Int32(threads)))
        try check(api.SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL))
        try check(api.CreateSession(env, modelPath, options, &session))
        try check(api.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory))
    }

    deinit {
        if let memory { api.ReleaseMemoryInfo(memory) }
        if let session { api.ReleaseSession(session) }
        if let options { api.ReleaseSessionOptions(options) }
        if let env { api.ReleaseEnv(env) }
    }

    private func check(_ status: OpaquePointer?) throws {
        guard let status else { return }
        let message = String(cString: api.GetErrorMessage(status)!)
        api.ReleaseStatus(status)
        throw ORTError(description: message)
    }

    /// Embeds `count` prepared images stored back to back in `input`.
    public func embed(_ input: [Float], count: Int) throws -> [Float] {
        precondition(input.count == count * 3 * Backbone.side * Backbone.side)
        var shape: [Int64] = [Int64(count), 3, Int64(Backbone.side), Int64(Backbone.side)]
        var data = input
        var inValue: OpaquePointer?
        var outValue: OpaquePointer?
        defer {
            if let inValue { api.ReleaseValue(inValue) }
            if let outValue { api.ReleaseValue(outValue) }
        }
        try data.withUnsafeMutableBytes { bytes in
            try check(api.CreateTensorWithDataAsOrtValue(memory, bytes.baseAddress, bytes.count, &shape, 4,
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inValue))
            let inName = strdup("pixel_values"), outName = strdup("features")
            defer { free(inName); free(outName) }
            var inNames: [UnsafePointer<CChar>?] = [UnsafePointer(inName)]
            var outNames: [UnsafePointer<CChar>?] = [UnsafePointer(outName)]
            var inputs: [OpaquePointer?] = [inValue]
            try check(api.Run(session, nil, &inNames, &inputs, 1, &outNames, 1, &outValue))
        }
        var raw: UnsafeMutableRawPointer?
        try check(api.GetTensorMutableData(outValue, &raw))
        let n = count * Backbone.dims
        return Array(UnsafeBufferPointer(start: raw!.bindMemory(to: Float.self, capacity: n), count: n))
    }
}
