import Foundation
import LineageCore

/// How embeddings are scaled before the head sees them.
public enum Norm: String, Codable, CaseIterable {
    case none, standardize, l2standardize
}

/// Everything that decides how a head is trained.
public struct HeadConfig: Codable, Hashable, CustomStringConvertible {
    public var hidden = 0                 // 0: softmax regression; > 0: one ReLU hidden layer
    public var learningRate: Float = 1e-3 // Adam step size
    public var weightDecay: Float = 1e-4  // L2 penalty on weights (not biases): loss += wd/2 * |W|^2
    public var dropout: Float = 0         // dropout on the embedding while training
    public var batch = 128
    public var maxEpochs = 200
    public var patience = 8               // epochs without a better validation loss before stopping
    public var norm: Norm = .standardize
    public var seed: UInt64 = 7
    public init() {}
    public var description: String {
        (hidden == 0 ? "linear" : "mlp\(hidden)") + " lr \(learningRate) wd \(weightDecay)"
            + (dropout > 0 ? " drop \(dropout)" : "") + " \(norm.rawValue) batch \(batch)"
    }
}

public struct EpochLog: Codable {
    public var epoch: Int
    public var trainLoss: Double
    public var valLoss: Double?
    public var valAccuracy: Double?
}

public struct TrainedHead {
    public var shape: HeadShape
    public var params: [Float]   // the full Core layout: norm section + layers
    public var epochs: Int       // epochs of the kept weights
    public var log: [EpochLog]
}

// MARK: - vector helpers (SIMD, for training speed; Core uses plain loops)

@inline(__always)
func dot(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int) -> Float {
    var acc0 = SIMD8<Float>(), acc1 = SIMD8<Float>()
    var i = 0
    let ra = UnsafeRawPointer(a), rb = UnsafeRawPointer(b)
    while i + 16 <= n {
        acc0 += ra.loadUnaligned(fromByteOffset: i * 4, as: SIMD8<Float>.self)
            * rb.loadUnaligned(fromByteOffset: i * 4, as: SIMD8<Float>.self)
        acc1 += ra.loadUnaligned(fromByteOffset: i * 4 + 32, as: SIMD8<Float>.self)
            * rb.loadUnaligned(fromByteOffset: i * 4 + 32, as: SIMD8<Float>.self)
        i += 16
    }
    var s = (acc0 + acc1).sum()
    while i < n { s += a[i] * b[i]; i += 1 }
    return s
}

@inline(__always)
func axpy(_ alpha: Float, _ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<Float>, _ n: Int) {
    var i = 0
    while i < n { y[i] += alpha * x[i]; i += 1 }
}

// MARK: - the model as a function of its layer parameters

/// Layer parameters only (no norm section), plus which of them are weights (decayed) vs biases.
struct Layers {
    let shape: HeadShape
    var w: [Float]
    var decay: [Float] // 1 for weights, 0 for biases

    init(shape: HeadShape, rng: inout SplitMix64) {
        self.shape = shape
        let d = shape.inputs, h = shape.hidden, c = shape.classes
        w = [Float](repeating: 0, count: shape.layerCount)
        decay = [Float](repeating: 0, count: shape.layerCount)
        if h == 0 {
            for i in 0..<(c * d) { w[i] = 0.01 * rng.normal(); decay[i] = 1 }
        } else {
            let s1 = (2 / Float(d)).squareRoot(), s2 = (1 / Float(h)).squareRoot()
            for i in 0..<(h * d) { w[i] = s1 * rng.normal(); decay[i] = 1 }
            let w2 = h * d + h
            for i in w2..<(w2 + c * h) { w[i] = s2 * rng.normal(); decay[i] = 1 }
        }
    }
}

/// Mean cross-entropy over `rows` of Z (already normalised), its gradient into `grad`, and the number
/// of correct top-1 predictions. Dropout masks come from `rng` when `dropout > 0`.
/// Work is split over the cores; each part has its own gradient buffer, summed in a fixed order.
func lossAndGradient(_ layers: Layers, Z: UnsafePointer<Float>, y: [Int], rows: ArraySlice<Int>,
                     dropout: Float, seed: UInt64, grad: inout [Float], wantGrad: Bool = true) -> (Double, Int) {
    let shape = layers.shape
    let d = shape.inputs, h = shape.hidden, c = shape.classes
    let parts = rows.count >= 64 ? 2 : 1
    let rowArray = Array(rows)
    let L = shape.layerCount
    let partLoss = UnsafeMutablePointer<Double>.allocate(capacity: parts)
    let partCorrect = UnsafeMutablePointer<Int>.allocate(capacity: parts)
    let partGrad = UnsafeMutablePointer<Float>.allocate(capacity: wantGrad ? parts * L : 1)
    if wantGrad { partGrad.initialize(repeating: 0, count: parts * L) }
    defer { partLoss.deallocate(); partCorrect.deallocate(); partGrad.deallocate() }
    let scale = 1 / Float(rows.count)
    layers.w.withUnsafeBufferPointer { wb in
        let W = wb.baseAddress!
        do {
            DispatchQueue.concurrentPerform(iterations: parts) { p in
                var rng = SplitMix64(seed: seed &+ UInt64(p) &* 0x9E37)
                let lo = p * rowArray.count / parts, hi = (p + 1) * rowArray.count / parts
                var x = [Float](repeating: 0, count: d)
                var a = [Float](repeating: 0, count: max(h, 1))
                var pre = [Float](repeating: 0, count: max(h, 1))
                var z = [Float](repeating: 0, count: c)
                var gh = [Float](repeating: 0, count: max(h, 1))
                var loss = 0.0, correct = 0
                for r in lo..<hi {
                    let i = rowArray[r]
                    let src = Z + i * d
                    if dropout > 0 {
                        let keep = 1 / (1 - dropout)
                        for j in 0..<d { x[j] = rng.uniform() < dropout ? 0 : src[j] * keep }
                    } else {
                        for j in 0..<d { x[j] = src[j] }
                    }
                    x.withUnsafeBufferPointer { xb in
                        let xp = xb.baseAddress!
                        if h == 0 {
                            let b = W + c * d
                            for k in 0..<c { z[k] = dot(W + k * d, xp, d) + b[k] }
                        } else {
                            let b1 = W + h * d, w2 = b1 + h, b2 = w2 + c * h
                            for j in 0..<h { let v = dot(W + j * d, xp, d) + b1[j]; pre[j] = v; a[j] = v > 0 ? v : 0 }
                            a.withUnsafeBufferPointer { ab in
                                for k in 0..<c { z[k] = dot(w2 + k * h, ab.baseAddress!, h) + b2[k] }
                            }
                        }
                    }
                    // softmax, loss, accuracy
                    var m = z[0], arg = 0
                    for k in 1..<c where z[k] > m { m = z[k]; arg = k }
                    var s: Float = 0
                    for k in 0..<c { z[k] = expf(z[k] - m); s += z[k] }
                    for k in 0..<c { z[k] /= s }
                    loss -= log(Double(max(z[y[i]], 1e-30)))
                    if arg == y[i] { correct += 1 }
                    guard wantGrad else { continue }
                    // dLoss/dlogits = p - onehot, averaged over the batch
                    z[y[i]] -= 1
                    for k in 0..<c { z[k] *= scale }
                    let G = partGrad + p * L
                    x.withUnsafeBufferPointer { xb in
                        let xp = xb.baseAddress!
                        if h == 0 {
                            for k in 0..<c { axpy(z[k], xp, G + k * d, d); G[c * d + k] += z[k] }
                        } else {
                            let w2 = W + h * d + h
                            let gb1 = G + h * d, gw2 = gb1 + h, gb2 = gw2 + c * h
                            for j in 0..<h { gh[j] = 0 }
                            a.withUnsafeBufferPointer { ab in
                                gh.withUnsafeMutableBufferPointer { ghb in
                                    for k in 0..<c {
                                        axpy(z[k], ab.baseAddress!, gw2 + k * h, h)
                                        gb2[k] += z[k]
                                        axpy(z[k], w2 + k * h, ghb.baseAddress!, h)
                                    }
                                }
                            }
                            for j in 0..<h where pre[j] > 0 {
                                axpy(gh[j], xp, G + j * d, d)
                                gb1[j] += gh[j]
                            }
                        }
                    }
                }
                partLoss[p] = loss
                partCorrect[p] = correct
            }
        }
    }
    if wantGrad {
        for i in 0..<L { grad[i] = 0 }
        for p in 0..<parts { for i in 0..<L { grad[i] += partGrad[p * L + i] } }
    }
    var loss = 0.0, correct = 0
    for p in 0..<parts { loss += partLoss[p]; correct += partCorrect[p] }
    return (loss / Double(rows.count), correct)
}

/// The full objective used by training: mean cross-entropy plus the L2 penalty.
func objective(_ layers: Layers, Z: UnsafePointer<Float>, y: [Int], rows: ArraySlice<Int>, weightDecay: Float,
               grad: inout [Float]) -> Double {
    let (ce, _) = lossAndGradient(layers, Z: Z, y: y, rows: rows, dropout: 0, seed: 0, grad: &grad)
    var penalty = 0.0
    for i in 0..<layers.w.count where layers.decay[i] > 0 {
        penalty += Double(layers.w[i] * layers.w[i])
        grad[i] += weightDecay * layers.w[i]
    }
    return ce + Double(weightDecay) / 2 * penalty
}

// MARK: - normalisation

/// Mean and 1/std per dimension, measured on the training rows only.
public func normStats(_ X: [Float], rows: Int, dims d: Int, norm: Norm) -> [Float] {
    var out = [Float](repeating: 0, count: 2 * d)
    for j in 0..<d { out[d + j] = 1 }
    guard norm != .none else { return out }
    var mean = [Double](repeating: 0, count: d), sq = [Double](repeating: 0, count: d)
    for i in 0..<rows {
        var s: Float = 1
        if norm == .l2standardize {
            var n2: Float = 0
            for j in 0..<d { n2 += X[i * d + j] * X[i * d + j] }
            s = n2 > 0 ? 1 / n2.squareRoot() : 1
        }
        for j in 0..<d { let v = Double(X[i * d + j] * s); mean[j] += v; sq[j] += v * v }
    }
    for j in 0..<d {
        let m = mean[j] / Double(rows)
        let v = max(sq[j] / Double(rows) - m * m, 0)
        out[j] = Float(m)
        out[d + j] = Float(1 / (v.squareRoot() + 1e-3))
    }
    return out
}

/// Applies the Core normalisation to every row.
public func normalizeRows(_ X: [Float], rows: Int, shape: HeadShape, normParams: [Float]) -> [Float] {
    let d = shape.inputs
    var out = [Float](repeating: 0, count: rows * d)
    X.withUnsafeBufferPointer { xb in
        normParams.withUnsafeBufferPointer { pb in
            out.withUnsafeMutableBufferPointer { ob in
                for i in 0..<rows {
                    normalizeEmbedding(xb.baseAddress! + i * d, shape: shape, params: pb.baseAddress!,
                                       out: ob.baseAddress! + i * d)
                }
            }
        }
    }
    return out
}

// MARK: - training loop

/// Trains a head with mini-batch Adam. With validation rows it stops early on validation loss and keeps
/// the best epoch's weights; without, it trains for exactly `fixedEpochs`.
public func trainHead(_ cfg: HeadConfig, X: [Float], y: [Int], valX: [Float]? = nil, valY: [Int]? = nil,
                      fixedEpochs: Int? = nil, classes: Int = 120, verbose: Bool = false) -> TrainedHead {
    let d = Backbone.dims
    let n = y.count
    let shape = HeadShape(inputs: d, hidden: cfg.hidden, classes: classes, l2: cfg.norm == .l2standardize)
    let normP = normStats(X, rows: n, dims: d, norm: cfg.norm)
    let Z = normalizeRows(X, rows: n, shape: shape, normParams: normP)
    let ZV = valX.map { normalizeRows($0, rows: valY!.count, shape: shape, normParams: normP) }
    var rng = SplitMix64(seed: cfg.seed)
    var layers = Layers(shape: shape, rng: &rng)
    var m = [Float](repeating: 0, count: shape.layerCount), v = m, grad = m
    let b1: Float = 0.9, b2: Float = 0.999, eps: Float = 1e-8
    var step = 0
    var best = (loss: Double.infinity, w: layers.w, epoch: 0)
    var log = [EpochLog]()
    var order = Array(0..<n)
    let epochs = fixedEpochs ?? cfg.maxEpochs
    for epoch in 1...epochs {
        rng.shuffle(&order)
        var total = 0.0
        Z.withUnsafeBufferPointer { zb in
            for start in stride(from: 0, to: n, by: cfg.batch) {
                let rows = order[start..<min(start + cfg.batch, n)]
                let (l, _) = lossAndGradient(layers, Z: zb.baseAddress!, y: y, rows: rows, dropout: cfg.dropout,
                                             seed: rng.next(), grad: &grad)
                total += l * Double(rows.count)
                step += 1
                let c1 = 1 / (1 - powf(b1, Float(step))), c2 = 1 / (1 - powf(b2, Float(step)))
                for i in 0..<grad.count {
                    let g = grad[i] + cfg.weightDecay * layers.decay[i] * layers.w[i]
                    m[i] = b1 * m[i] + (1 - b1) * g
                    v[i] = b2 * v[i] + (1 - b2) * g * g
                    layers.w[i] -= cfg.learningRate * (m[i] * c1) / ((v[i] * c2).squareRoot() + eps)
                }
            }
        }
        var entry = EpochLog(epoch: epoch, trainLoss: total / Double(n))
        if let ZV, let valY {
            var none = [Float]()
            let (vl, vc) = ZV.withUnsafeBufferPointer {
                lossAndGradient(layers, Z: $0.baseAddress!, y: valY, rows: Array(0..<valY.count)[...],
                                dropout: 0, seed: 0, grad: &none, wantGrad: false)
            }
            entry.valLoss = vl
            entry.valAccuracy = Double(vc) / Double(valY.count)
            if vl < best.loss { best = (vl, layers.w, epoch) }
            if verbose { print(String(format: "    epoch %3d  train %.4f  val %.4f  acc %.4f", epoch, entry.trainLoss, vl, entry.valAccuracy!)) }
            log.append(entry)
            if epoch - best.epoch >= cfg.patience { break }
        } else {
            log.append(entry)
            best = (0, layers.w, epoch)
        }
    }
    return TrainedHead(shape: shape, params: normP + best.w, epochs: best.epoch, log: log)
}

/// Class probabilities for every row, through the Core code the page also runs.
public func predictProbs(_ head: TrainedHead, X: [Float], rows: Int) -> [Float] {
    let d = head.shape.inputs, c = head.shape.classes
    var out = [Float](repeating: 0, count: rows * c)
    let parts = 2
    out.withUnsafeMutableBufferPointer { ob in
        let obase = ob.baseAddress!
        X.withUnsafeBufferPointer { xb in
            head.params.withUnsafeBufferPointer { pb in
                DispatchQueue.concurrentPerform(iterations: parts) { p in
                    var scratch = [Float](repeating: 0, count: head.shape.scratchCount)
                    scratch.withUnsafeMutableBufferPointer { sb in
                        for i in (p * rows / parts)..<((p + 1) * rows / parts) {
                            headForward(xb.baseAddress! + i * d, shape: head.shape, params: pb.baseAddress!,
                                        scratch: sb.baseAddress!, probs: obase + i * c)
                        }
                    }
                }
            }
        }
    }
    return out
}
