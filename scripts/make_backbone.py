#!/usr/bin/env python3
"""Turns the downloaded MobileNetV4 (ImageNet classifier, ONNX) into the backbone Lineage uses:

1. Cuts the graph before the 1000-way ImageNet classifier, so the output is the 1280-number
   embedding (after conv_head + ReLU, flattened). Swift trains a new 120-way head on it.
2. Stores the weights as float16 with a Cast back to float32 in front of each use. The arithmetic
   stays float32; only the stored weights are rounded. This halves the file (about 17 MB) so the
   page can download it, and Swift and the browser run the exact same file.
3. Saves the ImageNet classifier's own rows for the 120 Stanford Dogs breeds (all 120 are
   ImageNet-1k classes) to model/imagenet_dogs.bin, for the "no training at all" baseline.

Needs the `onnx` Python package (graph surgery only; nothing is executed here).

usage: python3 scripts/make_backbone.py model/mnv4.onnx model/backbone.onnx model/config.json data/splits/train.txt
"""
import json
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

src, dst, config, split = sys.argv[1:5]
m = onnx.load(src)
g = m.graph

# 3. the ImageNet head's rows for the 120 breeds, in the order of the sorted folder names (synset ids)
labels = {int(k): v for k, v in json.load(open(config))["id2label"].items()}
folders = sorted({line.split("/")[0] for line in open(split) if line.strip()})
rows = []
for f in folders:
    name = f.split("-", 1)[1].replace("_", " ").lower()
    hits = [i for i, v in labels.items() if 151 <= i <= 275 and name in [p.strip().lower() for p in v.split(",")]]
    assert len(hits) == 1, (f, hits)
    rows.append(hits[0])
assert len(rows) == 120 and rows == sorted(rows)
weights = {t.name: numpy_helper.to_array(t) for t in g.initializer}
W, b = weights["classifier.weight"][rows], weights["classifier.bias"][rows]
np.concatenate([W.ravel(), b]).astype("<f4").tofile(dst.replace("backbone.onnx", "imagenet_dogs.bin"))
json.dump({"imagenetIndex": rows, "folders": folders}, open(dst.replace("backbone.onnx", "imagenet_dogs.json"), "w"))
FEATURES = "/flatten/Flatten_output_0"

# 1. drop the classifier
keep = []
for n in g.node:
    if n.op_type == "Gemm" and "classifier.weight" in n.input:
        continue
    keep.append(n)
del g.node[:]
g.node.extend(keep)
used = {i for n in g.node for i in n.input}
inits = [t for t in g.initializer if t.name in used]
del g.initializer[:]
del g.output[:]
g.output.append(helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch_size", 1280]))
g.node.append(helper.make_node("Identity", [FEATURES], ["features"], name="features_out"))

# 2. float16 storage, float32 compute
casts = []
for t in inits:
    a = numpy_helper.to_array(t)
    if a.dtype == np.float32 and a.size >= 16:
        h = numpy_helper.from_array(a.astype(np.float16), t.name + "_f16")
        g.initializer.append(h)
        casts.append(helper.make_node("Cast", [h.name], [t.name], to=TensorProto.FLOAT, name=t.name + "_cast"))
    else:
        g.initializer.append(t)
nodes = casts + list(g.node)
del g.node[:]
g.node.extend(nodes)
del g.value_info[:]
m.producer_name = "lineage/make_backbone.py"
onnx.checker.check_model(m)
onnx.save(m, dst)
print("wrote", dst, "with", len(casts), "float16 weights")
