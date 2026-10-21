import Foundation
import LineageCore

/// Breed probabilities for one decoded photo: the shared preparation, the backbone, the trained head.
public func classify(_ img: RGBImage, backbone: Backbone, head: TrainedHead, meta: HeadFile) throws -> (embedding: [Float], probs: [Float]) {
    var first: [Float]? = nil
    var acc = [Float](repeating: 0, count: meta.classes)
    for name in meta.views {
        let e = try backbone.embed(prepareImage(img, Variant(name: name)!), count: 1)
        if first == nil { first = e }
        for (i, p) in predictProbs(head, X: e, rows: 1).enumerated() { acc[i] += p }
    }
    return (first!, acc.map { $0 / Float(meta.views.count) })
}

public func topK(_ probs: [Float], _ k: Int) -> [(Int, Float)] {
    Array(probs.enumerated().sorted { $0.element > $1.element }.prefix(k)).map { ($0.offset, $0.element) }
}

public func predict(root: URL, files: [String]) throws {
    let (head, meta) = try loadHead(root: root)
    let backbone = try Backbone(modelPath: root.appendingPathComponent("docs/backbone.onnx").path)
    for f in files {
        let img = try decodeJPEG(try Data(contentsOf: URL(fileURLWithPath: f)))
        let (_, p) = try classify(img, backbone: backbone, head: head, meta: meta)
        print(f)
        for (k, v) in topK(p, 5) { print(String(format: "  %5.1f%%  %@", v * 100, meta.breeds[k])) }
    }
}
