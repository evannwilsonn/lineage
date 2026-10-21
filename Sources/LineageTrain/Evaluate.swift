import Foundation
import LineageCore

/// Scores for a set of predictions (probabilities, rows x classes).
public struct Score: Codable {
    public var top1: Double
    public var top5: Double
    public var logLoss: Double?
    public var n: Int
}

public func score(_ probs: [Float], labels y: [Int], classes c: Int = 120, withLoss: Bool = true) -> Score {
    var t1 = 0, t5 = 0
    var nll = 0.0
    for i in 0..<y.count {
        let row = probs[(i * c)..<((i + 1) * c)]
        let p = row[row.startIndex + y[i]]
        // ties count against the right answer, so one-hot guesses don't get a free top-5
        var higher = 0
        for (k, q) in row.enumerated() where q > p || (q == p && k != y[i]) { higher += 1 }
        if higher == 0 { t1 += 1 }
        if higher < 5 { t5 += 1 }
        nll -= log(Double(max(p, 1e-12)))
    }
    let n = Double(y.count)
    return Score(top1: Double(t1) / n, top5: Double(t5) / n, logLoss: withLoss ? nll / n : nil, n: y.count)
}

public func argmaxRows(_ probs: [Float], classes c: Int = 120) -> [Int] {
    (0..<(probs.count / c)).map { i in
        var best = 0
        for k in 1..<c where probs[i * c + k] > probs[i * c + best] { best = k }
        return best
    }
}

/// Averages two sets of probabilities (e.g. a photo and its mirror image).
public func average(_ a: [Float], _ b: [Float]) -> [Float] { zip(a, b).map { ($0 + $1) / 2 } }

// MARK: - baselines that need no training of a head

/// Scores from a fixed linear layer (logits = W x + b), e.g. the ImageNet classifier's own dog rows.
public func linearProbs(W: [Float], b: [Float], X: [Float], rows: Int, dims d: Int = Backbone.dims) -> [Float] {
    let c = b.count
    var out = [Float](repeating: 0, count: rows * c)
    W.withUnsafeBufferPointer { wb in
        X.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                let o = ob.baseAddress!
                DispatchQueue.concurrentPerform(iterations: 2) { p in
                    for i in (p * rows / 2)..<((p + 1) * rows / 2) {
                        for k in 0..<c { o[i * c + k] = dot(wb.baseAddress! + k * d, xb.baseAddress! + i * d, d) + b[k] }
                        softmaxInPlace(o + i * c, c)
                    }
                }
            }
        }
    }
    return out
}

func unitRows(_ X: [Float], rows: Int, dims d: Int) -> [Float] {
    var out = X
    for i in 0..<rows {
        var n2: Float = 0
        for j in 0..<d { n2 += X[i * d + j] * X[i * d + j] }
        let s = n2 > 0 ? 1 / n2.squareRoot() : 0
        for j in 0..<d { out[i * d + j] *= s }
    }
    return out
}

/// Nearest class mean by cosine similarity. Returns one-hot "probabilities" (accuracy only).
public func nearestCentroid(trainX: [Float], trainY: [Int], X: [Float], rows: Int, classes c: Int = 120) -> [Float] {
    let d = Backbone.dims
    let U = unitRows(trainX, rows: trainY.count, dims: d)
    var means = [Float](repeating: 0, count: c * d)
    for (i, y) in trainY.enumerated() { for j in 0..<d { means[y * d + j] += U[i * d + j] } }
    let M = unitRows(means, rows: c, dims: d)
    let probs = linearProbs(W: M.map { $0 * 1000 }, b: [Float](repeating: 0, count: c), X: unitRows(X, rows: rows, dims: d), rows: rows)
    return oneHot(argmaxRows(probs), classes: c)
}

func oneHot(_ labels: [Int], classes c: Int) -> [Float] {
    var out = [Float](repeating: 0, count: labels.count * c)
    for (i, y) in labels.enumerated() { out[i * c + y] = 1 }
    return out
}

/// k nearest training photos by cosine similarity, each voting with its similarity.
/// Returns the vote shares, for several k at once.
public func kNearest(trainX: [Float], trainY: [Int], X: [Float], rows: Int, ks: [Int], classes c: Int = 120) -> [Int: [Float]] {
    let d = Backbone.dims, nT = trainY.count
    let U = unitRows(trainX, rows: nT, dims: d), Q = unitRows(X, rows: rows, dims: d)
    let kmax = ks.max()!
    var result = [Int: [Float]]()
    for k in ks { result[k] = [Float](repeating: 0, count: rows * c) }
    var neighbours = [[(Float, Int)]](repeating: [], count: rows)
    U.withUnsafeBufferPointer { ub in
        Q.withUnsafeBufferPointer { qb in
            neighbours.withUnsafeMutableBufferPointer { nb in
                let nbase = nb.baseAddress!
                DispatchQueue.concurrentPerform(iterations: 2) { p in
                    var sims = [Float](repeating: 0, count: nT)
                    for i in (p * rows / 2)..<((p + 1) * rows / 2) {
                        for t in 0..<nT { sims[t] = dot(ub.baseAddress! + t * d, qb.baseAddress! + i * d, d) }
                        let top = sims.enumerated().sorted { $0.element > $1.element }.prefix(kmax)
                        nbase[i] = top.map { ($0.element, trainY[$0.offset]) }
                    }
                }
            }
        }
    }
    for k in ks {
        for i in 0..<rows {
            var total: Float = 0
            for (s, y) in neighbours[i].prefix(k) { result[k]![i * c + y] += max(s, 0); total += max(s, 0) }
            if total > 0 { for j in 0..<c { result[k]![i * c + j] /= total } }
        }
    }
    return result
}

// MARK: - the held-out report

public struct BreedResult: Codable {
    public var breed: String
    public var folder: String
    public var n: Int
    public var top1: Double
    public var top5: Double
    public var mostConfusedWith: String?
    public var mostConfusedCount: Int
}

public struct ConfusedPair: Codable {
    public var a: String
    public var b: String
    public var aAsB: Int   // photos of a called b
    public var bAsA: Int
    public var nA: Int
    public var nB: Int
    public var total: Int { aAsB + bAsA }
}

public func perBreed(_ probs: [Float], labels y: [Int], folders: [String]) -> [BreedResult] {
    let c = folders.count
    let pred = argmaxRows(probs, classes: c)
    var out = [BreedResult]()
    for k in 0..<c {
        let idx = y.indices.filter { y[$0] == k }
        let s = score(idx.flatMap { Array(probs[($0 * c)..<(($0 + 1) * c)]) }, labels: idx.map { _ in k }, classes: c)
        var wrong = [Int: Int]()
        for i in idx where pred[i] != k { wrong[pred[i], default: 0] += 1 }
        let top = wrong.max { $0.value == $1.value ? $0.key > $1.key : $0.value < $1.value }
        out.append(BreedResult(breed: breedName(folders[k]), folder: folders[k], n: idx.count, top1: s.top1, top5: s.top5,
                               mostConfusedWith: top.map { breedName(folders[$0.key]) }, mostConfusedCount: top?.value ?? 0))
    }
    return out
}

/// Breed pairs the model mixes up most, counting both directions.
public func confusedPairs(_ probs: [Float], labels y: [Int], folders: [String], top: Int) -> [ConfusedPair] {
    let c = folders.count
    let pred = argmaxRows(probs, classes: c)
    var counts = [Int](repeating: 0, count: c * c)
    for (i, t) in y.enumerated() where pred[i] != t { counts[t * c + pred[i]] += 1 }
    var nPer = [Int](repeating: 0, count: c)
    for t in y { nPer[t] += 1 }
    var pairs = [ConfusedPair]()
    for a in 0..<c { for b in (a + 1)..<c where counts[a * c + b] + counts[b * c + a] > 0 {
        pairs.append(ConfusedPair(a: breedName(folders[a]), b: breedName(folders[b]), aAsB: counts[a * c + b],
                                  bAsA: counts[b * c + a], nA: nPer[a], nB: nPer[b]))
    } }
    return Array(pairs.sorted { $0.total == $1.total ? $0.a < $1.a : $0.total > $1.total }.prefix(top))
}
