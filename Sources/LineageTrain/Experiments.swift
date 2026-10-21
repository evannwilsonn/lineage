import Foundation
import LineageCore

// The experiment runs behind every number in the README. `tune` only ever looks at the 12,000 training
// photos (fit on 10,200, judge on 1,800); `holdout` refits the chosen head on all 12,000 and opens the
// 8,580 test photos once.
//
// A "view" is one way of showing a photo to the backbone: squashed or centre-cropped, mirrored or not.
// A head can be trained on several views of each training photo (more rows, same labels), and at test
// time its probabilities can be averaged over several views (one backbone pass per view).

public let valPerClass = 15
public let splitSeed: UInt64 = 42

func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    enc.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try enc.encode(value).write(to: url)
}

public func readJSON<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
    try JSONDecoder().decode(type, from: Data(contentsOf: url))
}

/// The four embedding tables, loaded on demand.
public final class Embeddings {
    let root: URL
    var cache = [Variant: EmbeddingTable]()
    public init(root: URL) { self.root = root }
    public func table(_ v: Variant) throws -> EmbeddingTable {
        if let t = cache[v] { return t }
        let t = try EmbeddingTable(contentsOf: root.appendingPathComponent("data/embeddings/\(v.name).f32"))
        precondition(t.rows == 20_580, "\(v.name).f32 has \(t.rows) rows; run `lineage embed`")
        cache[v] = t
        return t
    }
}

public struct NamedScore: Codable {
    public var name: String
    public var score: Score
    public var note: String?
}

public struct TuningRun: Codable {
    public var stage: String
    public var trainViews: [String]
    public var testViews: [String]
    public var config: HeadConfig
    public var epochs: Int
    public var seconds: Double
    public var val: Score
    public var log: [EpochLog]?
}

public struct Choice: Codable {
    public var trainViews: [String]
    public var testViews: [String]
    public var config: HeadConfig
    public var epochs: Int
}

public struct TuningReport: Codable {
    public var fitPhotos: Int
    public var valPhotos: Int
    public var valPerClass: Int
    public var seed: UInt64
    public var baselines: [NamedScore]
    public var runs: [TuningRun]
    public var chosen: Choice
}

/// ImageNet's own classifier rows for the 120 breeds (written by scripts/make_backbone.py).
func imagenetDogHead(root: URL) throws -> (W: [Float], b: [Float]) {
    let t = try EmbeddingTable(contentsOf: root.appendingPathComponent("model/imagenet_dogs.bin"), dims: 1)
    let d = Backbone.dims
    return (Array(t.data[0..<(120 * d)]), Array(t.data[(120 * d)...]))
}

/// Trains on `trainViews` of the rows `trainIdx` and returns the head.
func fitHead(_ cfg: HeadConfig, emb: Embeddings, trainViews: [Variant], trainIdx: [Int], y: [Int],
             valIdx: [Int]? = nil, valY: [Int]? = nil, valView: Variant? = nil, fixedEpochs: Int? = nil) throws -> TrainedHead {
    var X = [Float](), Y = [Int]()
    for v in trainViews { X += try emb.table(v).gather(trainIdx); Y += y }
    let VX = try valIdx.map { try emb.table(valView!).gather($0) }
    return trainHead(cfg, X: X, y: Y, valX: VX, valY: valY, fixedEpochs: fixedEpochs)
}

/// Probabilities averaged over the test views.
func viewProbs(_ head: TrainedHead, emb: Embeddings, views: [Variant], rows idx: [Int]) throws -> [Float] {
    var acc: [Float]? = nil
    for v in views {
        let p = predictProbs(head, X: try emb.table(v).gather(idx), rows: idx.count)
        acc = acc.map { zip($0, p).map(+) } ?? p
    }
    let k = Float(views.count)
    return acc!.map { $0 / k }
}

public func tune(root: URL, splits: DogSplits) throws {
    let emb = Embeddings(root: root)
    let yAll = splits.trainLabels
    let (fit, val) = stratifiedSplit(labels: yAll, perClass: valPerClass, seed: splitSeed)
    try writeJSON(["fit": fit, "val": val], to: root.appendingPathComponent("results/split.json"))
    let fitY = fit.map { yAll[$0] }, valY = val.map { yAll[$0] }
    print("tuning: \(fit.count) fit photos, \(val.count) validation photos")

    var runs = [TuningRun]()
    var heads = [Int: TrainedHead]()
    @discardableResult
    func run(_ stage: String, train: [Variant], test: [Variant], _ cfg: HeadConfig) throws -> Int {
        let t0 = Date()
        // early stopping always watches the first test view's validation loss
        let head = try fitHead(cfg, emb: emb, trainViews: train, trainIdx: fit, y: fitY, valIdx: val, valY: valY, valView: test[0])
        let probs = try viewProbs(head, emb: emb, views: test, rows: val)
        let r = TuningRun(stage: stage, trainViews: train.map(\.name), testViews: test.map(\.name), config: cfg,
                          epochs: head.epochs, seconds: Date().timeIntervalSince(t0), val: score(probs, labels: valY), log: head.log)
        let views = "train \(r.trainViews.joined(separator: "+")) test \(r.testViews.joined(separator: "+"))"
        print("  " + [stage.padding(toLength: 8, withPad: " ", startingAt: 0), cfg.description.padding(toLength: 46, withPad: " ", startingAt: 0),
                      views.padding(toLength: 52, withPad: " ", startingAt: 0),
                      String(format: "epochs %3d  val top-1 %.4f  top-5 %.4f  loss %.4f  (%.0fs)", r.epochs, r.val.top1, r.val.top5, r.val.logLoss!, r.seconds)].joined(separator: " "))
        runs.append(r)
        heads[runs.count - 1] = head
        return runs.count - 1
    }
    func better(_ a: Int, than b: Int) -> Bool {
        let x = runs[a].val, y = runs[b].val
        return x.top1 != y.top1 ? x.top1 > y.top1 : x.logLoss! < y.logLoss!
    }

    // baselines
    var baselines = [NamedScore]()
    baselines.append(NamedScore(name: "Guess the most common breed", score: score(oneHot(valY.map { _ in fitY.first! }, classes: 120), labels: valY, withLoss: false)))
    let (iw, ib) = try imagenetDogHead(root: root)
    for v in [Variant(.squash, flip: false), Variant(.centerCrop, flip: false)] {
        let t = try emb.table(v)
        let F = t.gather(fit), V = t.gather(val)
        baselines.append(NamedScore(name: "ImageNet classifier's own dog classes, no training (\(v.name))",
                                    score: score(linearProbs(W: iw, b: ib, X: V, rows: val.count), labels: valY)))
        baselines.append(NamedScore(name: "Nearest breed average, cosine (\(v.name))",
                                    score: score(nearestCentroid(trainX: F, trainY: fitY, X: V, rows: val.count), labels: valY, withLoss: false)))
        let knn = kNearest(trainX: F, trainY: fitY, X: V, rows: val.count, ks: [1, 5, 10, 20])
        for k in [1, 5, 10, 20] {
            baselines.append(NamedScore(name: "k nearest photos, k = \(k), cosine (\(v.name))", score: score(knn[k]!, labels: valY, withLoss: false)))
        }
    }
    for b in baselines { print(String(format: "  baseline  top-1 %.4f  top-5 %.4f  ", b.score.top1, b.score.top5) + b.name) }

    // 1. framing x embedding scaling x learning rate, softmax regression, one view
    var best = -1
    for fr in [Framing.squash, .centerCrop] {
        for norm in Norm.allCases {
            for lr: Float in [1e-4, 3e-4, 1e-3] {
                var cfg = HeadConfig(); cfg.norm = norm; cfg.learningRate = lr
                let v = Variant(fr, flip: false)
                let i = try run("scaling", train: [v], test: [v], cfg)
                if best < 0 || better(i, than: best) { best = i }
            }
        }
    }
    let view = Variant(name: runs[best].trainViews[0])!
    // 2. weight decay
    for wd: Float in [0, 1e-3, 1e-2] {
        var cfg = runs[best].config; cfg.weightDecay = wd
        let i = try run("decay", train: [view], test: [view], cfg)
        if better(i, than: best) { best = i }
    }
    // 3. more views of each training photo
    let other = Variant(view.framing == .squash ? .centerCrop : .squash, flip: false)
    let mirror = Variant(view.framing, flip: true)
    for train in [[view, mirror], [view, mirror, other, Variant(other.framing, flip: true)]] {
        let i = try run("views", train: train, test: [view], runs[best].config)
        if better(i, than: best) { best = i }
    }
    let trainViews = runs[best].trainViews.compactMap(Variant.init(name:))
    // 4. a hidden layer, with dropout on the embedding as in the course
    for (hidden, drop) in [(512, Float(0.2)), (512, Float(0.5)), (1024, Float(0.5))] {
        var cfg = runs[best].config; cfg.hidden = hidden; cfg.dropout = drop
        let i = try run("hidden", train: trainViews, test: [view], cfg)
        if better(i, than: best) { best = i }
    }
    // 5. averaging over views at test time, for the chosen head (no retraining: same weights)
    let chosenHead = heads[best]!
    var testChoice = (views: [view], score: runs[best].val)
    for test in [[view, mirror], [view, other], [view, mirror, other, Variant(other.framing, flip: true)]] {
        let s = score(try viewProbs(chosenHead, emb: emb, views: test, rows: val), labels: valY)
        print(String(format: "  test views %@: val top-1 %.4f  top-5 %.4f  loss %.4f", test.map(\.name).joined(separator: "+"), s.top1, s.top5, s.logLoss!))
        runs.append(TuningRun(stage: "test-views", trainViews: runs[best].trainViews, testViews: test.map(\.name), config: runs[best].config,
                              epochs: runs[best].epochs, seconds: 0, val: s, log: nil))
        // each extra view is another backbone pass in the page, so it has to earn half a point per pass
        if s.top1 >= testChoice.score.top1 + 0.005 * Double(test.count - testChoice.views.count) { testChoice = (test, s) }
    }
    let chosen = Choice(trainViews: runs[best].trainViews, testViews: testChoice.views.map(\.name), config: runs[best].config, epochs: runs[best].epochs)
    print("chosen: \(chosen.config), train views \(chosen.trainViews), test views \(chosen.testViews), \(chosen.epochs) epochs; val top-1 \(testChoice.score.top1)")
    for i in runs.indices where i != best { runs[i].log = nil }
    try writeJSON(TuningReport(fitPhotos: fit.count, valPhotos: val.count, valPerClass: valPerClass, seed: splitSeed,
                               baselines: baselines, runs: runs, chosen: chosen),
                  to: root.appendingPathComponent("results/tuning.json"))
}

// MARK: - held-out test

public struct HoldoutReport: Codable {
    public var trainPhotos: Int
    public var trainRows: Int
    public var testPhotos: Int
    public var chosen: Choice
    public var final: Score
    public var singleView: Score
    public var imagenetOnTrain: Score
    public var baselines: [NamedScore]
    public var course: [NamedScore]
    public var breeds: [BreedResult]
    public var lookalikes: [ConfusedPair]
    public var trainingSeconds: Double
}

public func holdout(root: URL, splits: DogSplits) throws {
    let tuning = try readJSON(TuningReport.self, from: root.appendingPathComponent("results/tuning.json"))
    let ch = tuning.chosen
    let trainViews = ch.trainViews.compactMap(Variant.init(name:)), testViews = ch.testViews.compactMap(Variant.init(name:))
    let emb = Embeddings(root: root)
    let trainIdx = Array(0..<12_000), testIdx = Array(12_000..<20_580)
    let y = splits.trainLabels, yt = splits.testLabels
    print("refitting \(ch.config) on \(trainViews.count) x 12,000 rows for \(ch.epochs) epochs")
    let t0 = Date()
    let head = try fitHead(ch.config, emb: emb, trainViews: trainViews, trainIdx: trainIdx, y: y, fixedEpochs: ch.epochs)
    let seconds = Date().timeIntervalSince(t0)
    try saveHead(head, choice: ch, root: root, folders: splits.folders)

    // the one look at the test photos
    let probs = try viewProbs(head, emb: emb, views: testViews, rows: testIdx)
    let final = score(probs, labels: yt)
    let single = score(try viewProbs(head, emb: emb, views: [testViews[0]], rows: testIdx), labels: yt)
    print(String(format: "TEST  top-1 %.4f  top-5 %.4f  loss %.4f   (first view alone: top-1 %.4f)", final.top1, final.top5, final.logLoss!, single.top1))

    let base = try emb.table(testViews[0])
    let trainX = base.gather(trainIdx), TX = base.gather(testIdx)
    var baselines = [NamedScore]()
    var counts = [Int](repeating: 0, count: 120); for t in y { counts[t] += 1 }
    baselines.append(NamedScore(name: "Guess the most common breed", score: score(oneHot(yt.map { _ in counts.firstIndex(of: counts.max()!)! }, classes: 120), labels: yt, withLoss: false)))
    let (iw, ib) = try imagenetDogHead(root: root)
    baselines.append(NamedScore(name: "ImageNet classifier's own dog classes, no training", score: score(linearProbs(W: iw, b: ib, X: TX, rows: yt.count), labels: yt)))
    let imagenetOnTrain = score(linearProbs(W: iw, b: ib, X: trainX, rows: y.count), labels: y)
    baselines.append(NamedScore(name: "Nearest breed average, cosine", score: score(nearestCentroid(trainX: trainX, trainY: y, X: TX, rows: yt.count), labels: yt, withLoss: false)))
    let view0 = testViews[0].name
    let bestK = tuning.baselines.filter { $0.name.hasPrefix("k nearest") && $0.name.hasSuffix("(\(view0))") }.max { $0.score.top1 < $1.score.top1 }!
    let k = Int(bestK.name.split(separator: "=")[1].split(separator: ",")[0].trimmingCharacters(in: .whitespaces))!
    baselines.append(NamedScore(name: "k nearest photos, k = \(k), cosine", score: score(kNearest(trainX: trainX, trainY: y, X: TX, rows: yt.count, ks: [k])[k]!, labels: yt, withLoss: false)))
    for b in baselines { print(String(format: "  baseline  top-1 %.4f  top-5 %.4f  ", b.score.top1, b.score.top5) + b.name) }

    let course = [
        NamedScore(name: "Course notebook model_1: EfficientNetV2-B0 + Dense(120), all 12,000 photos, 5 epochs",
                   score: Score(top1: 0.8767, top5: 0, logLoss: 0.3753, n: 8580), note: "its own report; the test photos were also its validation data"),
        NamedScore(name: "Course notebook model_0: the same on 10% of the training photos",
                   score: Score(top1: 0.8107, top5: 0, logLoss: 0.8792, n: 8580), note: "its own report"),
    ]
    try writeJSON(HoldoutReport(trainPhotos: 12_000, trainRows: 12_000 * trainViews.count, testPhotos: yt.count, chosen: ch,
                                final: final, singleView: single, imagenetOnTrain: imagenetOnTrain,
                                baselines: baselines, course: course,
                                breeds: perBreed(probs, labels: yt, folders: splits.folders),
                                lookalikes: confusedPairs(probs, labels: yt, folders: splits.folders, top: 20),
                                trainingSeconds: seconds),
                  to: root.appendingPathComponent("results/holdout.json"))
}

// MARK: - the files the page loads

public struct HeadFile: Codable {
    public var inputs: Int
    public var hidden: Int
    public var classes: Int
    public var l2: Bool
    public var views: [String]   // test-time views, averaged
    public var side: Int
    public var breeds: [String]
    public var folders: [String]
    public var floats: Int
}

/// docs/head.bin (little-endian float32, the Core parameter layout) and docs/head.json.
public func saveHead(_ head: TrainedHead, choice: Choice, root: URL, folders: [String]) throws {
    let docs = root.appendingPathComponent("docs")
    try head.params.withUnsafeBytes { try Data($0).write(to: docs.appendingPathComponent("head.bin")) }
    try writeJSON(HeadFile(inputs: head.shape.inputs, hidden: head.shape.hidden, classes: head.shape.classes,
                           l2: head.shape.l2, views: choice.testViews, side: Backbone.side,
                           breeds: folders.map(breedName), folders: folders, floats: head.params.count),
                  to: docs.appendingPathComponent("head.json"))
}

public func loadHead(root: URL) throws -> (TrainedHead, HeadFile) {
    let docs = root.appendingPathComponent("docs")
    let meta = try readJSON(HeadFile.self, from: docs.appendingPathComponent("head.json"))
    let t = try EmbeddingTable(contentsOf: docs.appendingPathComponent("head.bin"), dims: 1)
    let shape = HeadShape(inputs: meta.inputs, hidden: meta.hidden, classes: meta.classes, l2: meta.l2)
    precondition(t.data.count == shape.paramCount)
    return (TrainedHead(shape: shape, params: t.data, epochs: 0, log: []), meta)
}
