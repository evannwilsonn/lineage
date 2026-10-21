import Foundation
import XCTest
@testable import LineageCore
@testable import LineageTrain

/// The full native path on a shipped sample photo: libjpeg decode, shrink, Swift preparation, the backbone
/// through ONNX Runtime's C API, and the trained head, against the answers saved in results/parity.json.
final class BackboneTests: XCTestCase {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    func testSamplePhotoMatchesSavedAnswers() throws {
        let parityURL = root.appendingPathComponent("results/parity.json")
        guard FileManager.default.fileExists(atPath: parityURL.path) else { throw XCTSkip("no results/parity.json") }
        let parity = try readJSON(ParityFile.self, from: parityURL)
        let (head, meta) = try loadHead(root: root)
        let backbone = try Backbone(modelPath: root.appendingPathComponent("docs/backbone.onnx").path, threads: 1)
        for ref in parity.samples.prefix(2) {
            let jpeg = try Data(contentsOf: root.appendingPathComponent("docs/\(ref.sample.file)"))
            let img = shrinkToShortSide(try decodeJPEG(jpeg))
            XCTAssertEqual(min(img.width, img.height), 272)
            let (e, p) = try classify(img, backbone: backbone, head: head, meta: meta)
            let worstE = zip(e, ref.embedding).map { abs($0 - $1) }.max()!
            let worstP = zip(p, ref.probs).map { abs($0 - $1) }.max()!
            XCTAssertLessThan(worstE, 1e-3, ref.sample.label)
            XCTAssertLessThan(worstP, 1e-4, ref.sample.label)
            XCTAssertEqual(topK(p, 1)[0].0, meta.breeds.firstIndex(of: ref.top5[0].breed))
        }
    }

    func testJPEGDecodeSize() throws {
        let jpeg = try Data(contentsOf: root.appendingPathComponent("docs/samples/beagle.jpg"))
        let img = try decodeJPEG(jpeg)
        XCTAssertEqual(img.pixels.count, img.width * img.height * 3)
        XCTAssertGreaterThan(img.width, 500)
        XCTAssertThrowsError(try decodeJPEG(Data([1, 2, 3, 4, 5])))
    }
}
