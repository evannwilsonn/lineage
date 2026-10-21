// The classifier head that sits on top of the backbone's 1280-number embedding.
// Pointer-only code with no allocation, so it compiles unchanged for the page (Embedded Swift -> Wasm).
//
// Parameter layout, one flat Float array:
//   mean[d], invStd[d]                         embedding standardisation (0 and 1 if unused)
//   linear (hidden == 0):  W[classes*d], b[classes]
//   MLP    (hidden > 0):   W1[hidden*d], b1[hidden], W2[classes*hidden], b2[classes]
// `l2` says whether the embedding is scaled to unit length before standardising.

public struct HeadShape: Equatable {
    public var inputs: Int
    public var hidden: Int
    public var classes: Int
    public var l2: Bool
    public init(inputs: Int, hidden: Int, classes: Int, l2: Bool) {
        self.inputs = inputs; self.hidden = hidden; self.classes = classes; self.l2 = l2
    }
    public var normCount: Int { 2 * inputs }
    public var layerCount: Int {
        hidden == 0 ? classes * inputs + classes
                    : hidden * inputs + hidden + classes * hidden + classes
    }
    public var paramCount: Int { normCount + layerCount }
    /// Floats of scratch space `forward` needs.
    public var scratchCount: Int { inputs + hidden }
}

@inline(__always)
func dotScalar(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int) -> Float {
    // four accumulators: fewer dependent adds, same result on every platform
    var s0: Float = 0, s1: Float = 0, s2: Float = 0, s3: Float = 0
    var i = 0
    while i + 4 <= n {
        s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3]
        i += 4
    }
    while i < n { s0 += a[i] * b[i]; i += 1 }
    return (s0 + s1) + (s2 + s3)
}

/// Standardises one embedding into `out` (d floats).
public func normalizeEmbedding(_ x: UnsafePointer<Float>, shape: HeadShape, params: UnsafePointer<Float>,
                               out: UnsafeMutablePointer<Float>) {
    let d = shape.inputs
    var scale: Float = 1
    if shape.l2 {
        let n2 = dotScalar(x, x, d)
        scale = n2 > 0 ? 1 / n2.squareRoot() : 1
    }
    let mean = params, invStd = params + d
    var i = 0
    while i < d { out[i] = (x[i] * scale - mean[i]) * invStd[i]; i += 1 }
}

/// Class probabilities for one embedding. `scratch` holds shape.scratchCount floats,
/// `probs` receives shape.classes floats.
public func headForward(_ x: UnsafePointer<Float>, shape: HeadShape, params: UnsafePointer<Float>,
                        scratch: UnsafeMutablePointer<Float>, probs: UnsafeMutablePointer<Float>) {
    let d = shape.inputs, h = shape.hidden, c = shape.classes
    let z = scratch
    normalizeEmbedding(x, shape: shape, params: params, out: z)
    let layers = params + shape.normCount
    if h == 0 {
        let w = layers, b = layers + c * d
        var k = 0
        while k < c { probs[k] = dotScalar(w + k * d, z, d) + b[k]; k += 1 }
    } else {
        let w1 = layers, b1 = w1 + h * d, w2 = b1 + h, b2 = w2 + c * h
        let a = scratch + d
        var j = 0
        while j < h {
            let v = dotScalar(w1 + j * d, z, d) + b1[j]
            a[j] = v > 0 ? v : 0
            j += 1
        }
        var k = 0
        while k < c { probs[k] = dotScalar(w2 + k * h, a, h) + b2[k]; k += 1 }
    }
    softmaxInPlace(probs, c)
}
