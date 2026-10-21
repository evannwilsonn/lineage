import Foundation
import LineageCore

/// One way of turning a photo into backbone input.
public struct Variant: Hashable {
    public let framing: Framing
    public let flip: Bool
    public init(_ framing: Framing, flip: Bool) { self.framing = framing; self.flip = flip }
    public var name: String { (framing == .squash ? "squash" : "crop") + (flip ? "-flip" : "") }
    public init?(name: String) {
        guard let v = Variant.all.first(where: { $0.name == name }) else { return nil }
        self = v
    }
    public static let all = [Variant(.squash, flip: false), Variant(.centerCrop, flip: false),
                             Variant(.squash, flip: true), Variant(.centerCrop, flip: true)]
}

/// Prepares one decoded photo for the backbone with the shared Core code.
public func prepareImage(_ img: RGBImage, _ v: Variant) -> [Float] {
    let side = Backbone.side
    var scratch = [Float](repeating: 0, count: prepareScratchCount(height: img.height, side: side))
    var out = [Float](repeating: 0, count: 3 * side * side)
    img.pixels.withUnsafeBufferPointer { px in
        scratch.withUnsafeMutableBufferPointer { s in
            out.withUnsafeMutableBufferPointer { o in
                prepare(px.baseAddress!, width: img.width, height: img.height, framing: v.framing, side: side,
                        flip: v.flip, scratch: s.baseAddress!, out: o.baseAddress!)
            }
        }
    }
    return out
}

/// Runs every photo through the backbone once per variant and writes data/embeddings/<variant>.f32.
/// About 20 minutes on two cores for all four variants.
public func embedAll(root: URL, splits: DogSplits, variants: [Variant], limit: Int? = nil, batch: Int = 32) throws {
    let backbone = try Backbone(modelPath: root.appendingPathComponent("docs/backbone.onnx").path)
    let dir = root.appendingPathComponent("data/embeddings")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    var paths = splits.all
    if let limit { paths = Array(paths.prefix(limit)) }
    var handles = [Variant: FileHandle]()
    for v in variants {
        let url = dir.appendingPathComponent(limit == nil ? "\(v.name).f32" : "\(v.name).sample.f32")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handles[v] = try FileHandle(forWritingTo: url)
    }
    let start = Date()
    var done = 0
    for chunk in stride(from: 0, to: paths.count, by: batch) {
        let part = Array(paths[chunk..<min(chunk + batch, paths.count)])
        // decode and prepare on both cores, then hand the batch to ONNX Runtime
        var prepared = [[[Float]]](repeating: [], count: part.count)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: part.count) { i in
            let url = root.appendingPathComponent("data/images").appendingPathComponent(part[i])
            let img = try! decodeJPEG(try! Data(contentsOf: url))
            let inputs = variants.map { prepareImage(img, $0) }
            lock.lock(); prepared[i] = inputs; lock.unlock()
        }
        for (k, v) in variants.enumerated() {
            let input = prepared.flatMap { $0[k] }
            let emb = try backbone.embed(input, count: part.count)
            emb.withUnsafeBytes { handles[v]!.write(Data($0)) }
        }
        done += part.count
        if done % 1024 < batch || done == paths.count {
            let rate = Double(done) / Date().timeIntervalSince(start)
            print(String(format: "  %5d / %d photos, %.1f per second", done, paths.count, rate))
        }
    }
    for h in handles.values { try h.close() }
}
