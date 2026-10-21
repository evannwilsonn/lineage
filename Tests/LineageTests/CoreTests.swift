import Foundation
import XCTest
@testable import LineageCore

final class CoreTests: XCTestCase {
    func testExpMatchesFoundation() {
        var worst: Float = 0
        var x: Float = -80
        while x <= 80 {
            let rel = abs(lnExp(x) - expf(x)) / expf(x)
            worst = max(worst, rel)
            x += 0.0137
        }
        XCTAssertLessThan(worst, 3e-7)
        XCTAssertEqual(lnExp(0), 1)
        XCTAssertGreaterThan(lnExp(-200), 0)       // clamped, never zero or NaN
        XCTAssertFalse(lnExp(200).isInfinite)
    }

    func testSoftmaxIsStableAndSumsToOne() {
        var z: [Float] = [1000, 1001, 999, -5000]
        z.withUnsafeMutableBufferPointer { softmaxInPlace($0.baseAddress!, 4) }
        XCTAssertEqual(z.reduce(0, +), 1, accuracy: 1e-6)
        XCTAssertEqual(z[1], 1 / (1 + expf(-1) + expf(-2)), accuracy: 1e-6)
        XCTAssertLessThan(z[3], 1e-30)
    }

    /// A 3-input, 2-class linear head against the formula worked by hand.
    func testLinearHeadForward() {
        let shape = HeadShape(inputs: 3, hidden: 0, classes: 2, l2: false)
        // mean, invStd, W (2x3), b
        let params: [Float] = [1, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 0, 0.5, -0.5]
        let x: [Float] = [3, 4, 5]          // normalised: (3-1)*2, 4, 5 = 4, 4, 5
        var scratch = [Float](repeating: 0, count: shape.scratchCount)
        var probs = [Float](repeating: 0, count: 2)
        headForward(x, shape: shape, params: params, scratch: &scratch, probs: &probs)
        let z0: Float = 4 + 0.5, z1: Float = 4 - 0.5
        XCTAssertEqual(probs[0], 1 / (1 + expf(z1 - z0)), accuracy: 1e-6)
        XCTAssertEqual(params.count, shape.paramCount)
    }

    func testMLPHeadForwardAndL2() {
        let shape = HeadShape(inputs: 2, hidden: 2, classes: 2, l2: true)
        // mean 0, invStd 1; W1 = I, b1 = [0, -1]; W2 = [[1, 0], [0, 1]], b2 = 0
        let params: [Float] = [0, 0, 1, 1, 1, 0, 0, 1, 0, -1, 1, 0, 0, 1, 0, 0]
        XCTAssertEqual(params.count, shape.paramCount)
        let x: [Float] = [30, 40]          // unit length: 0.6, 0.8; hidden: relu(0.6), relu(-0.2) = 0.6, 0
        var scratch = [Float](repeating: 0, count: shape.scratchCount)
        var probs = [Float](repeating: 0, count: 2)
        headForward(x, shape: shape, params: params, scratch: &scratch, probs: &probs)
        XCTAssertEqual(probs[0], 1 / (1 + expf(-0.6)), accuracy: 1e-6)
    }

    // MARK: preparation

    func image(_ w: Int, _ h: Int, _ f: (Int, Int) -> (UInt8, UInt8, UInt8)) -> [UInt8] {
        var px = [UInt8](); px.reserveCapacity(w * h * 3)
        for y in 0..<h { for x in 0..<w { let (r, g, b) = f(x, y); px += [r, g, b] } }
        return px
    }

    func run(_ px: [UInt8], _ w: Int, _ h: Int, _ fr: Framing, side: Int, flip: Bool = false) -> [Float] {
        var scratch = [Float](repeating: 0, count: prepareScratchCount(height: h, side: side))
        var out = [Float](repeating: 0, count: 3 * side * side)
        prepare(px, width: w, height: h, framing: fr, side: side, flip: flip, scratch: &scratch, out: &out)
        return out
    }

    func testConstantImageGivesConstantStandardisedValues() {
        let px = image(37, 23) { _, _ in (200, 100, 50) }
        let out = run(px, 37, 23, .squash, side: 8)
        let plane = 64
        for i in 0..<plane {
            XCTAssertEqual(out[i], (200 / 255 - 0.485) / 0.229, accuracy: 1e-4)
            XCTAssertEqual(out[plane + i], (100 / 255 - 0.456) / 0.224, accuracy: 1e-4)
            XCTAssertEqual(out[2 * plane + i], (50 / 255 - 0.406) / 0.225, accuracy: 1e-4)
        }
    }

    func testSameSizeSquashIsIdentityAndFlipMirrors() {
        let px = image(8, 8) { x, y in (UInt8(x * 30), UInt8(y * 30), 7) }
        let out = run(px, 8, 8, .squash, side: 8)
        let flipped = run(px, 8, 8, .squash, side: 8, flip: true)
        for y in 0..<8 { for x in 0..<8 {
            XCTAssertEqual(out[y * 8 + x], (Float(x * 30) / 255 - 0.485) / 0.229, accuracy: 1e-4)
            XCTAssertEqual(out[64 + y * 8 + x], (Float(y * 30) / 255 - 0.456) / 0.224, accuracy: 1e-4)
            XCTAssertEqual(flipped[y * 8 + x], out[y * 8 + (7 - x)], accuracy: 1e-6)
        } }
    }

    func testHalvingAveragesPairs() {
        // shrinking 2x with the widened triangle averages neighbouring pixels (weights 1/4, 3/4, 3/4, 1/4)
        let px = image(4, 1) { x, _ in (UInt8([0, 100, 200, 40][x]), 0, 0) }
        var scratch = [Float](repeating: 0, count: prepareScratchCount(height: 1, side: 2))
        var out = [Float](repeating: 0, count: 3 * 2 * 2)
        // a 4x1 strip squashed to 2x2: the vertical pass just repeats the single row
        prepare(px, width: 4, height: 1, framing: .squash, side: 2, flip: false, scratch: &scratch, out: &out)
        let left = (0 * 0.75 + 100 * 0.75 + 200 * 0.25) / 1.75     // pixel 0..2 around centre 1.0
        XCTAssertEqual(out[0] * 0.229 + 0.485, Float(left) / 255, accuracy: 1e-4)
    }

    func testCentreCropRegion() {
        let (x, y, w, h) = framingRegion(.centerCrop, width: 400, height: 300, side: 256)
        XCTAssertEqual(w, h)
        XCTAssertEqual(w, 300 * 256 / 269, accuracy: 1e-3)
        XCTAssertEqual(x, (400 - w) / 2, accuracy: 1e-3)
        XCTAssertEqual(y, (300 - h) / 2, accuracy: 1e-3)
        XCTAssertTrue(framingRegion(.squash, width: 400, height: 300, side: 256) == (0, 0, 400, 300))
    }
}
