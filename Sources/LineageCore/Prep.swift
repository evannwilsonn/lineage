// Turning a photo into the backbone's input: pick a region, resample it to side x side with a
// triangle (bilinear) filter widened when shrinking, as Pillow's BILINEAR does, then scale to 0..1 and
// standardise with ImageNet's mean and std into channel-first floats.
// Pointer-only and allocation-free, so the page runs this exact code in WebAssembly.

public enum Framing: Int32 {
    /// the whole photo squeezed to a square (what the course's image_dataset_from_directory does)
    case squash = 0
    /// resize the shorter side to 269, take the centre 256 (the backbone's own evaluation recipe)
    case centerCrop = 1
}

public let imagenetMean: (Float, Float, Float) = (0.485, 0.456, 0.406)
public let imagenetStd: (Float, Float, Float) = (0.229, 0.224, 0.225)

/// The source rectangle (x, y, width, height) in pixels that a framing samples from.
public func framingRegion(_ framing: Framing, width w: Int, height h: Int, side: Int) -> (Float, Float, Float, Float) {
    switch framing {
    case .squash:
        return (0, 0, Float(w), Float(h))
    case .centerCrop:
        let short = Float(w < h ? w : h)
        let s = short * Float(side) / Float(side * 100 / 95)  // 256 / 269 of the shorter side
        return ((Float(w) - s) / 2, (Float(h) - s) / 2, s, s)
    }
}

@inline(__always)
func triangle(_ t: Float) -> Float {
    let a = t < 0 ? -t : t
    return a < 1 ? 1 - a : 0
}

/// Floats of scratch space `prepare` needs for a w x h source.
public func prepareScratchCount(height h: Int, side: Int) -> Int { side * h * 3 }

/// Resamples the region of an RGB (interleaved, 8-bit) w x h image to side x side and writes
/// standardised channel-first floats (3 * side * side) to `out`. `flip` mirrors left to right.
public func prepare(_ rgb: UnsafePointer<UInt8>, width w: Int, height h: Int, framing: Framing, side: Int,
                    flip: Bool, scratch: UnsafeMutablePointer<Float>, out: UnsafeMutablePointer<Float>) {
    let (rx, ry, rw, rh) = framingRegion(framing, width: w, height: h, side: side)
    // horizontal pass: every source row -> side columns, into scratch[row][col][ch]
    let sx = rw / Float(side)
    let fx = sx > 1 ? sx : 1
    var ox = 0
    while ox < side {
        let center = rx + (Float(ox) + 0.5) * sx
        var lo = floorInt(center - fx); if lo < 0 { lo = 0 }
        var hi = ceilInt(center + fx); if hi > w { hi = w }
        var wsum: Float = 0
        var i = lo
        while i < hi { wsum += triangle((Float(i) + 0.5 - center) / fx); i += 1 }
        let inv: Float = wsum > 0 ? 1 / wsum : 0
        let col = flip ? side - 1 - ox : ox
        var y = 0
        while y < h {
            var r: Float = 0, g: Float = 0, b: Float = 0
            let row = rgb + y * w * 3
            i = lo
            while i < hi {
                let wt = triangle((Float(i) + 0.5 - center) / fx)
                if wt != 0 {
                    let p = row + i * 3
                    r += wt * Float(p[0]); g += wt * Float(p[1]); b += wt * Float(p[2])
                }
                i += 1
            }
            let s = scratch + (y * side + col) * 3
            s[0] = r * inv; s[1] = g * inv; s[2] = b * inv
            y += 1
        }
        ox += 1
    }
    // vertical pass: side rows, then scale and standardise
    let sy = rh / Float(side)
    let fy = sy > 1 ? sy : 1
    let plane = side * side
    let m0 = imagenetMean.0 * 255, m1 = imagenetMean.1 * 255, m2 = imagenetMean.2 * 255
    let k0 = 1 / (imagenetStd.0 * 255), k1 = 1 / (imagenetStd.1 * 255), k2 = 1 / (imagenetStd.2 * 255)
    var oy = 0
    while oy < side {
        let center = ry + (Float(oy) + 0.5) * sy
        var lo = floorInt(center - fy); if lo < 0 { lo = 0 }
        var hi = ceilInt(center + fy); if hi > h { hi = h }
        var wsum: Float = 0
        var j = lo
        while j < hi { wsum += triangle((Float(j) + 0.5 - center) / fy); j += 1 }
        let inv: Float = wsum > 0 ? 1 / wsum : 0
        var x = 0
        while x < side {
            var r: Float = 0, g: Float = 0, b: Float = 0
            j = lo
            while j < hi {
                let wt = triangle((Float(j) + 0.5 - center) / fy)
                if wt != 0 {
                    let s = scratch + (j * side + x) * 3
                    r += wt * s[0]; g += wt * s[1]; b += wt * s[2]
                }
                j += 1
            }
            let o = oy * side + x
            out[o] = (r * inv - m0) * k0
            out[plane + o] = (g * inv - m1) * k1
            out[2 * plane + o] = (b * inv - m2) * k2
            x += 1
        }
        oy += 1
    }
}
