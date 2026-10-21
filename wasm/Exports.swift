// The page's entry points into LineageCore, compiled with Embedded Swift to a small WebAssembly module
// (scripts/build-wasm). JavaScript owns the memory layout: it writes the photo, the head's weights and
// the backbone's embedding into linear memory and passes pointers.

@_cdecl("lineage_prepare_scratch")
public func lineagePrepareScratch(_ height: Int32, _ side: Int32) -> Int32 {
    Int32(prepareScratchCount(height: Int(height), side: Int(side)))
}

/// RGBA from a canvas (4 bytes per pixel) is packed to RGB in place first, then prepared.
@_cdecl("lineage_prepare_rgba")
public func lineagePrepareRGBA(_ rgba: UnsafeMutablePointer<UInt8>, _ width: Int32, _ height: Int32, _ framing: Int32,
                               _ side: Int32, _ flip: Int32, _ scratch: UnsafeMutablePointer<Float>,
                               _ out: UnsafeMutablePointer<Float>) {
    let n = Int(width) * Int(height)
    var i = 0
    while i < n {
        rgba[i * 3] = rgba[i * 4]; rgba[i * 3 + 1] = rgba[i * 4 + 1]; rgba[i * 3 + 2] = rgba[i * 4 + 2]
        i += 1
    }
    prepare(UnsafePointer(rgba), width: Int(width), height: Int(height), framing: Framing(rawValue: framing) ?? .squash,
            side: Int(side), flip: flip != 0, scratch: scratch, out: out)
}

@_cdecl("lineage_head_floats")
public func lineageHeadFloats(_ inputs: Int32, _ hidden: Int32, _ classes: Int32) -> Int32 {
    Int32(HeadShape(inputs: Int(inputs), hidden: Int(hidden), classes: Int(classes), l2: false).paramCount)
}

@_cdecl("lineage_head")
public func lineageHead(_ x: UnsafePointer<Float>, _ inputs: Int32, _ hidden: Int32, _ classes: Int32, _ l2: Int32,
                        _ params: UnsafePointer<Float>, _ scratch: UnsafeMutablePointer<Float>,
                        _ probs: UnsafeMutablePointer<Float>) {
    let shape = HeadShape(inputs: Int(inputs), hidden: Int(hidden), classes: Int(classes), l2: l2 != 0)
    headForward(x, shape: shape, params: params, scratch: scratch, probs: probs)
}
