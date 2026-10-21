import Foundation
import LineageCore

/// The official Stanford Dogs split: 12,000 training photos (100 per breed) and 8,580 test photos.
public struct DogSplits {
    public let train: [String]   // "n02085620-Chihuahua/n02085620_5927.jpg"
    public let test: [String]
    public let folders: [String] // the 120 breed folders, sorted (= ImageNet synset order)

    public init(root: URL) throws {
        func read(_ name: String) throws -> [String] {
            try String(contentsOf: root.appendingPathComponent("data/splits/\(name).txt"), encoding: .utf8)
                .split(separator: "\n").map(String.init).filter { !$0.isEmpty }
        }
        train = try read("train")
        test = try read("test")
        folders = Array(Set(train.map { String($0.split(separator: "/")[0]) })).sorted()
        precondition(folders.count == 120 && train.count == 12_000 && test.count == 8_580)
    }

    public var all: [String] { train + test }
    public func label(_ path: String) -> Int {
        folders.firstIndex(of: String(path.split(separator: "/")[0]))!
    }
    public var trainLabels: [Int] { train.map(label) }
    public var testLabels: [Int] { test.map(label) }
}

/// Display names for the Stanford Dogs labels that are terse or old-fashioned ("Pembroke", "Lhasa", "chow").
/// Same 120 classes, clearer names; everything else keeps its Stanford Dogs name.
let displayNames = [
    "Pembroke": "Pembroke Welsh corgi", "Cardigan": "Cardigan Welsh corgi", "Lhasa": "Lhasa apso", "chow": "Chow chow",
    "basset": "Basset hound", "Leonberg": "Leonberger", "Boston bull": "Boston terrier", "Maltese dog": "Maltese",
    "Japanese spaniel": "Japanese chin", "Shih-Tzu": "Shih Tzu", "EntleBucher": "Entlebucher", "Scotch terrier": "Scottish terrier",
    "cairn": "Cairn terrier", "Airedale": "Airedale terrier", "Dandie Dinmont": "Dandie Dinmont terrier",
    "clumber": "Clumber spaniel", "English springer": "English springer spaniel", "kelpie": "Australian kelpie",
    "redbone": "Redbone coonhound", "bluetick": "Bluetick coonhound", "Brabancon griffon": "Brabançon griffon",
    "malamute": "Alaskan malamute", "Staffordshire bullterrier": "Staffordshire bull terrier",
]

/// "n02085620-Chihuahua" -> "Chihuahua", "n02088094-Afghan_hound" -> "Afghan hound", "n02113023-Pembroke" -> "Pembroke Welsh corgi"
public func breedName(_ folder: String) -> String {
    let raw = (folder.split(separator: "-", maxSplits: 1).last.map(String.init) ?? folder).replacingOccurrences(of: "_", with: " ")
    let name = displayNames[raw] ?? raw
    return name.prefix(1).uppercased() + name.dropFirst()
}

/// Carves a validation set out of the training photos: `perClass` of each breed, picked with a fixed seed.
public func stratifiedSplit(labels: [Int], perClass: Int, seed: UInt64) -> (fit: [Int], val: [Int]) {
    var rng = SplitMix64(seed: seed)
    var byClass = [Int: [Int]]()
    for (i, y) in labels.enumerated() { byClass[y, default: []].append(i) }
    var fit = [Int](), val = [Int]()
    for y in byClass.keys.sorted() {
        var idx = byClass[y]!
        rng.shuffle(&idx)
        val += idx.prefix(perClass)
        fit += idx.dropFirst(perClass)
    }
    return (fit.sorted(), val.sorted())
}

/// Embeddings for every photo (train list then test list), N x 1280 floats, little-endian on disk.
public struct EmbeddingTable {
    public let rows: Int
    public let dims: Int
    public var data: [Float]

    public init(rows: Int, dims: Int, data: [Float]) {
        self.rows = rows; self.dims = dims; self.data = data
    }

    public init(contentsOf url: URL, dims: Int = Backbone.dims) throws {
        let bytes = try Data(contentsOf: url)
        let n = bytes.count / 4
        var data = [Float](repeating: 0, count: n)
        _ = data.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
        self.init(rows: n / dims, dims: dims, data: data)
    }

    public func gather(_ index: [Int]) -> [Float] {
        var out = [Float](); out.reserveCapacity(index.count * dims)
        for i in index { out += data[(i * dims)..<((i + 1) * dims)] }
        return out
    }
}

/// A small, fast, seedable generator (SplitMix64), so every run repeats exactly.
public struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    public mutating func shuffle<T>(_ a: inout [T]) {
        guard a.count > 1 else { return }
        for i in stride(from: a.count - 1, to: 0, by: -1) {
            let j = Int(next() % UInt64(i + 1))
            a.swapAt(i, j)
        }
    }
    public mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }
    /// Standard normal (Box-Muller).
    public mutating func normal() -> Float {
        let u1 = max(uniform(), 1e-7), u2 = uniform()
        return (-2 * logf(u1)).squareRoot() * cosf(2 * .pi * u2)
    }
}
