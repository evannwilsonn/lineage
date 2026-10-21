import Foundation
import LineageTrain

// lineage embed     every photo through the backbone (four framings), data/embeddings/*.f32
// lineage tune      baselines and head choices on the 1,800 validation photos, results/tuning.json
// lineage holdout   refit the chosen head, score the 8,580 test photos once, results/holdout.json, docs/head.*
// lineage predict   breed guesses for JPEG files
// lineage parity    reference answers for the page's checks, results/parity.json
// lineage page      web/template.html + results -> docs/index.html

setvbuf(stdout, nil, _IOLBF, 0)  // progress lines show up promptly when piped
let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let args = Array(CommandLine.arguments.dropFirst())
do {
    switch args.first {
    case "embed":
        try embedAll(root: root, splits: try DogSplits(root: root), variants: Variant.all, limit: args.dropFirst().first.flatMap(Int.init))
    case "tune":
        try tune(root: root, splits: try DogSplits(root: root))
    case "holdout":
        try holdout(root: root, splits: try DogSplits(root: root))
    case "predict":
        try predict(root: root, files: Array(args.dropFirst()))
    case "parity":
        try parity(root: root)
    case "page":
        try buildPage(root: root)
    default:
        print("usage: lineage embed | tune | holdout | predict <jpg>... | parity | page")
        exit(2)
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
