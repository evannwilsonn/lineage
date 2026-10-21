import Foundation
import XCTest
@testable import LineageCore
@testable import LineageTrain

final class TrainTests: XCTestCase {
    /// Analytic gradients against central differences, for both head shapes, with the L2 penalty.
    func checkGradient(hidden: Int) {
        let d = 6, c = 4, n = 9
        let shape = HeadShape(inputs: d, hidden: hidden, classes: c, l2: false)
        var rng = SplitMix64(seed: 3)
        var layers = Layers(shape: shape, rng: &rng)
        for i in layers.w.indices { layers.w[i] += 0.3 * rng.normal() }   // biases too, away from zero
        let Z = (0..<(n * d)).map { _ in rng.normal() }
        let y = (0..<n).map { $0 % c }
        let wd: Float = 0.05
        var grad = [Float](repeating: 0, count: shape.layerCount)
        var scratch = grad
        _ = Z.withUnsafeBufferPointer { objective(layers, Z: $0.baseAddress!, y: y, rows: Array(0..<n)[...], weightDecay: wd, grad: &grad) }
        var worst: Float = 0
        for i in layers.w.indices {
            let h: Float = 1e-2
            var plus = layers, minus = layers
            plus.w[i] += h; minus.w[i] -= h
            let fp = Z.withUnsafeBufferPointer { objective(plus, Z: $0.baseAddress!, y: y, rows: Array(0..<n)[...], weightDecay: wd, grad: &scratch) }
            let fm = Z.withUnsafeBufferPointer { objective(minus, Z: $0.baseAddress!, y: y, rows: Array(0..<n)[...], weightDecay: wd, grad: &scratch) }
            let numeric = Float((fp - fm) / Double(2 * h))
            worst = max(worst, abs(numeric - grad[i]) / max(1e-2, abs(numeric) + abs(grad[i])))
        }
        XCTAssertLessThan(worst, 2e-2, "hidden \(hidden)")
    }

    func testLinearGradient() { checkGradient(hidden: 0) }
    func testMLPGradient() { checkGradient(hidden: 5) }

    /// Three well-separated clusters in 1280 dimensions: both heads should get every validation point.
    func testHeadsLearnSeparableClusters() {
        let d = Backbone.dims, c = 3
        var rng = SplitMix64(seed: 11)
        let centres = (0..<c).map { _ in (0..<d).map { _ in rng.normal() } }
        func sample(_ k: Int) -> [Float] { centres[k].map { $0 + 0.8 * rng.normal() } }
        var X = [Float](), y = [Int](), VX = [Float](), vy = [Int]()
        for i in 0..<90 { X += sample(i % c); y.append(i % c) }
        for i in 0..<30 { VX += sample(i % c); vy.append(i % c) }
        for hidden in [0, 16] {
            var cfg = HeadConfig(); cfg.hidden = hidden; cfg.batch = 16; cfg.maxEpochs = 30; cfg.patience = 5
            let head = trainHead(cfg, X: X, y: y, valX: VX, valY: vy, classes: c)
            let s = score(predictProbs(head, X: VX, rows: vy.count), labels: vy, classes: c)
            XCTAssertEqual(s.top1, 1, "hidden \(hidden)")
            XCTAssertEqual(head.params.count, head.shape.paramCount)
        }
    }

    func testTrainingRepeatsExactly() {
        let d = Backbone.dims
        var rng = SplitMix64(seed: 5)
        let X = (0..<(40 * d)).map { _ in rng.normal() }, y = (0..<40).map { $0 % 4 }
        var cfg = HeadConfig(); cfg.batch = 8
        let a = trainHead(cfg, X: X, y: y, fixedEpochs: 3, classes: 4)
        let b = trainHead(cfg, X: X, y: y, fixedEpochs: 3, classes: 4)
        XCTAssertEqual(a.params, b.params)
    }

    func testStratifiedSplit() {
        let labels = (0..<1200).map { $0 / 10 }  // 120 classes x 10
        let (fit, val) = stratifiedSplit(labels: labels, perClass: 3, seed: 42)
        XCTAssertEqual(val.count, 360)
        XCTAssertEqual(fit.count, 840)
        XCTAssertTrue(Set(fit).isDisjoint(with: val))
        XCTAssertEqual(Set(val.map { labels[$0] }).count, 120)
        XCTAssertEqual(stratifiedSplit(labels: labels, perClass: 3, seed: 42).val, val)
    }

    func testScoreTopKAndPairs() {
        // 2 photos, 6 classes: first right at rank 1, second right at rank 3
        let p: [Float] = [0.5, 0.2, 0.1, 0.1, 0.05, 0.05,
                          0.4, 0.3, 0.2, 0.05, 0.03, 0.02]
        let s = score(p, labels: [0, 2], classes: 6)
        XCTAssertEqual(s.top1, 0.5)
        XCTAssertEqual(s.top5, 1)
        let folders = (0..<6).map { "n0-breed_\($0)" }
        let pairs = confusedPairs(p, labels: [0, 2], folders: folders, top: 5)
        XCTAssertEqual(pairs.count, 1)
        XCTAssertEqual(pairs[0].a, "Breed 0"); XCTAssertEqual(pairs[0].b, "Breed 2"); XCTAssertEqual(pairs[0].bAsA, 1)
        XCTAssertEqual(breedName("n02085620-Chihuahua"), "Chihuahua")
        XCTAssertEqual(breedName("n02113186-Cardigan"), "Cardigan Welsh corgi")
        XCTAssertEqual(breedName("n02105162-malinois"), "Malinois")
    }
}
