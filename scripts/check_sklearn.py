#!/usr/bin/env python3
"""The library check: scikit-learn's LogisticRegression on the same embeddings, the same validation
split and the same L2 strength as the best softmax-regression head trained in Swift.

Swift minimises  mean cross-entropy + wd/2 * |W|^2  with Adam and early stopping. scikit-learn minimises
sum cross-entropy + 1/(2C) * |W|^2  exactly (L-BFGS), so C = 1 / (wd * n). With wd = 0 it gets a very
large C instead. Writes results/sklearn.json.

usage: python3 scripts/check_sklearn.py   (after `lineage embed` and `lineage tune`)
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

root = Path(__file__).resolve().parent.parent
tuning = json.loads((root / "results/tuning.json").read_text())
split = json.loads((root / "results/split.json").read_text())
linear = [r for r in tuning["runs"] if r["config"]["hidden"] == 0 and len(r["trainViews"]) == 1
          and r["testViews"] == r["trainViews"] and r["stage"] != "test-views"]
best = max(linear, key=lambda r: (r["val"]["top1"], -r["val"]["logLoss"]))
cfg, framing = best["config"], best["trainViews"][0]
assert cfg["norm"] in ("standardize", "none"), "this check reproduces plain standardisation only"

E = np.fromfile(root / f"data/embeddings/{framing}.f32", "<f4").reshape(-1, 1280)
lines = [l for l in (root / "data/splits/train.txt").read_text().split("\n") if l]
folders = sorted({l.split("/")[0] for l in lines})
y = np.array([folders.index(l.split("/")[0]) for l in lines])
fit, val = np.array(split["fit"]), np.array(split["val"])
X, V = E[fit].astype(np.float64), E[val].astype(np.float64)
if cfg["norm"] == "standardize":
    mu, sd = X.mean(0), X.std(0)
    X, V = (X - mu) / (sd + 1e-3), (V - mu) / (sd + 1e-3)
wd = cfg["weightDecay"]
C = 1 / (wd * len(fit)) if wd > 0 else 1e6
t0 = time.time()
clf = LogisticRegression(C=C, max_iter=2000, tol=1e-6)
clf.fit(X, y[fit])
P = clf.predict_proba(V)
top1 = float((P.argmax(1) == y[val]).mean())
top5 = float(np.mean([y[val][i] in np.argsort(-P[i])[:5] for i in range(len(val))]))
out = {
    "framing": framing,
    "weightDecay": wd,
    "C": C,
    "sklearn": {"top1": top1, "top5": top5, "logLoss": float(log_loss(y[val], P, labels=range(120))),
                "iterations": int(clf.n_iter_[0]), "seconds": time.time() - t0},
    "swift": best["val"] | {"epochs": best["epochs"], "seconds": best["seconds"]},
}
(root / "results/sklearn.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
