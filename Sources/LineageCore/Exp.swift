// exp() without Foundation or libc, so the same code runs natively and in WebAssembly
// (Embedded Swift has no math library). exp(x) = 2^k * e^r with r = x - k*ln2, |r| <= ln2/2,
// and e^r from its Taylor series to r^7 (truncation error below 1e-8 relative).

// floor and ceil by hand: Embedded Swift on wasm32 has no libm to call for Float.rounded().
@inline(__always)
public func floorInt(_ x: Float) -> Int { let i = Int(x); return Float(i) > x ? i - 1 : i }
@inline(__always)
public func ceilInt(_ x: Float) -> Int { let i = Int(x); return Float(i) < x ? i + 1 : i }

@inline(__always)
public func lnExp(_ x: Float) -> Float {
    let c = x < -87 ? -87 : (x > 88 ? 88 : x)
    let k = Float(floorInt(c * 1.442695041 + 0.5))
    // ln2 split in two parts so k*ln2 is subtracted without losing precision
    let r = c - k * 0.693145752 - k * 1.428606765e-06
    var p: Float = 1.0 / 5040
    p = p * r + 1.0 / 720
    p = p * r + 1.0 / 120
    p = p * r + 1.0 / 24
    p = p * r + 1.0 / 6
    p = p * r + 0.5
    p = p * r + 1
    p = p * r + 1
    let twoK = Float(bitPattern: UInt32(Int32(k) + 127) << 23)
    return p * twoK
}

/// Softmax in place over n values, subtracting the largest first so nothing overflows.
public func softmaxInPlace(_ z: UnsafeMutablePointer<Float>, _ n: Int) {
    var m = z[0]
    var i = 1
    while i < n { if z[i] > m { m = z[i] }; i += 1 }
    var s: Float = 0
    i = 0
    while i < n { let e = lnExp(z[i] - m); z[i] = e; s += e; i += 1 }
    let inv = 1 / s
    i = 0
    while i < n { z[i] *= inv; i += 1 }
}
