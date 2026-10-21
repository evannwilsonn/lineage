import Foundation
import LineageCore

// MARK: - shrinking a photo like the data preparation does (shorter side to 272)

/// Separable triangle-filter resize of an RGB image to w x h (rounded back to 8 bits).
public func resizeRGB(_ img: RGBImage, width ow: Int, height oh: Int) -> RGBImage {
    func pass(_ src: [Float], _ sw: Int, _ sh: Int, horizontal: Bool, to n: Int) -> [Float] {
        let len = horizontal ? sw : sh, other = horizontal ? sh : sw
        let scale = Float(len) / Float(n), f = max(scale, 1)
        var out = [Float](repeating: 0, count: (horizontal ? n * sh : sw * n) * 3)
        for o in 0..<n {
            let c = (Float(o) + 0.5) * scale
            let lo = max(floorInt(c - f), 0), hi = min(ceilInt(c + f), len)
            var ws = [Float](), sum: Float = 0
            for i in lo..<hi { let t = abs((Float(i) + 0.5 - c) / f); let w = t < 1 ? 1 - t : 0; ws.append(w); sum += w }
            for k in 0..<other {
                var acc: (Float, Float, Float) = (0, 0, 0)
                for (j, i) in (lo..<hi).enumerated() {
                    let s = (horizontal ? k * sw + i : i * sw + k) * 3
                    acc.0 += ws[j] * src[s]; acc.1 += ws[j] * src[s + 1]; acc.2 += ws[j] * src[s + 2]
                }
                let d = (horizontal ? k * n + o : o * sw + k) * 3
                out[d] = acc.0 / sum; out[d + 1] = acc.1 / sum; out[d + 2] = acc.2 / sum
            }
        }
        return out
    }
    let src = img.pixels.map(Float.init)
    let h = pass(src, img.width, img.height, horizontal: true, to: ow)
    let v = pass(h, ow, img.height, horizontal: false, to: oh)
    return RGBImage(width: ow, height: oh, pixels: v.map { UInt8(max(0, min(255, $0.rounded()))) })
}

public func shrinkToShortSide(_ img: RGBImage, _ short: Int = 272) -> RGBImage {
    let s = Double(short) / Double(min(img.width, img.height))
    guard s < 1 else { return img }
    return resizeRGB(img, width: max(1, Int((Double(img.width) * s).rounded())), height: max(1, Int((Double(img.height) * s).rounded())))
}

// MARK: - reference answers for the page and for the WebAssembly check

public struct Sample: Codable {
    public var file: String
    public var label: String
    public var title: String
    public var creator: String
    public var source: String
    public var license: String
    public var width: Int
    public var height: Int
}

public struct Award: Codable { public var breed: String; public var p: Float }

public struct SampleAnswer: Codable {
    public var sample: Sample
    public var top5: [Award]
    public var probs: [Float]      // what the page shows (averaged with the mirror image if the head uses that)
    public var headProbs: [Float]  // the head on `embedding` alone
    public var embedding: [Float]
}

public struct PrepareCase: Codable {
    public var width: Int
    public var height: Int
    public var framing: Int32
    public var flip: Bool
    public var first: [Float]
    public var sum: Double
    public var sumSquares: Double
}

public struct ParityFile: Codable {
    public var samples: [SampleAnswer]
    public var prepare: [PrepareCase]
}

/// A deterministic RGBA test card for checking the WebAssembly preparation against native Swift.
public func testCard(width w: Int, height h: Int) -> [UInt8] {
    var px = [UInt8](repeating: 255, count: w * h * 4)
    for y in 0..<h { for x in 0..<w {
        let i = (y * w + x) * 4
        px[i] = UInt8((x * 7 + y * 3) % 256); px[i + 1] = UInt8((x * y) % 251); px[i + 2] = UInt8((x ^ y) & 255)
    } }
    return px
}

public func parity(root: URL) throws {
    let (head, meta) = try loadHead(root: root)
    let backbone = try Backbone(modelPath: root.appendingPathComponent("docs/backbone.onnx").path)
    let samples = try readJSON([Sample].self, from: root.appendingPathComponent("docs/samples/samples.json"))
    var answers = [SampleAnswer]()
    for s in samples {
        let img = shrinkToShortSide(try decodeJPEG(try Data(contentsOf: root.appendingPathComponent("docs/\(s.file)"))))
        let (e, p) = try classify(img, backbone: backbone, head: head, meta: meta)
        let top = topK(p, 5).map { Award(breed: meta.breeds[$0.0], p: $0.1) }
        print(s.label, "->", top.map { "\($0.breed) \(String(format: "%.3f", $0.p))" }.joined(separator: ", "))
        answers.append(SampleAnswer(sample: s, top5: top, probs: p, headProbs: predictProbs(head, X: e, rows: 1), embedding: e))
    }
    var cases = [PrepareCase]()
    for (w, h, fr, flip) in [(300, 220, Framing.squash, false), (300, 220, .centerCrop, true), (180, 333, .centerCrop, false)] {
        let rgba = testCard(width: w, height: h)
        var rgb = [UInt8](); for i in 0..<(w * h) { rgb += rgba[(i * 4)..<(i * 4 + 3)] }
        let out = prepareImage(RGBImage(width: w, height: h, pixels: rgb), Variant(fr, flip: flip))
        cases.append(PrepareCase(width: w, height: h, framing: fr.rawValue, flip: flip, first: Array(out.prefix(16)),
                                 sum: out.reduce(0.0) { $0 + Double($1) }, sumSquares: out.reduce(0.0) { $0 + Double($1 * $1) }))
    }
    try writeJSON(ParityFile(samples: answers, prepare: cases), to: root.appendingPathComponent("results/parity.json"))
}

// MARK: - docs/index.html

func pct(_ x: Double, _ digits: Int = 1) -> String { String(format: "%.\(digits)f%%", x * 100) }

public func buildPage(root: URL) throws {
    let holdout = try readJSON(HoldoutReport.self, from: root.appendingPathComponent("results/holdout.json"))
    let par = try readJSON(ParityFile.self, from: root.appendingPathComponent("results/parity.json"))
    let (_, meta) = try loadHead(root: root)
    struct PageSample: Codable { var sample: Sample; var top5: [Award] }
    struct PageData: Codable {
        var head: HeadFile
        var breeds: [BreedResult]
        var lookalikes: [ConfusedPair]
        var samples: [PageSample]
        var final: Score
    }
    let data = PageData(head: meta, breeds: holdout.breeds, lookalikes: Array(holdout.lookalikes.prefix(8)),
                        samples: par.samples.map { PageSample(sample: $0.sample, top5: $0.top5) }, final: holdout.final)
    let enc = JSONEncoder(); enc.outputFormatting = [.sortedKeys]
    let json = String(data: try enc.encode(data), encoding: .utf8)!.replacingOccurrences(of: "</", with: "<\\/")

    func base(_ prefix: String) -> NamedScore { holdout.baselines.first { $0.name.hasPrefix(prefix) }! }
    let imagenet = base("ImageNet"), knn = base("k nearest"), centroid = base("Nearest")
    let ranked = holdout.breeds.sorted { $0.top1 > $1.top1 }
    var tokens: [String: String] = [
        "top1": pct(holdout.final.top1), "top5": pct(holdout.final.top5),
        "top1_short": String(format: "%.1f", holdout.final.top1 * 100), "top5_short": String(format: "%.1f", holdout.final.top5 * 100),
        "test_n": holdout.testPhotos.formatted(), "train_n": "12,000",
        "course": pct(holdout.course[0].score.top1), "imagenet": pct(imagenet.score.top1), "knn": pct(knn.score.top1),
        "knn_name": knn.name, "majority": pct(base("Guess").score.top1), "centroid": pct(centroid.score.top1),
        "head_kind": meta.hidden == 0 ? "softmax regression" : "a \(meta.hidden)-unit hidden layer",
        "best_breed": ranked.first!.breed, "worst_breed": ranked.last!.breed, "worst_acc": pct(ranked.last!.top1, 0),
        "perfect": String(holdout.breeds.filter { $0.top1 == 1 }.count),
        "data": json,
    ]
    let names = ["squash": "the whole photo squeezed to 256 by 256", "crop": "the centre 256 by 256",
                 "squash-flip": "its mirror image, squeezed", "crop-flip": "the mirrored centre"]
    tokens["framing"] = meta.views.map { names[$0]! }.joined(separator: " and ")
    var html = try String(contentsOf: root.appendingPathComponent("web/template.html"), encoding: .utf8)
    for (k, v) in tokens { html = html.replacingOccurrences(of: "{{\(k)}}", with: v) }
    if let r = html.range(of: "{{") { throw ORTError(description: "unfilled token near \(html[r.lowerBound...].prefix(30))") }
    try html.write(to: root.appendingPathComponent("docs/index.html"), atomically: true, encoding: .utf8)
    let fm = FileManager.default
    for f in ["app.js", "rosette.js"] {
        let dst = root.appendingPathComponent("docs/\(f)")
        try? fm.removeItem(at: dst)
        try fm.copyItem(at: root.appendingPathComponent("web/\(f)"), to: dst)
    }
    print("wrote docs/index.html")
}
