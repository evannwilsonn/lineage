#!/usr/bin/env python3
"""Opens the demo in Chromium at 1280 and 375 px, in light and dark, judges a sample photo live and checks:
the backbone and the Swift WebAssembly load, the live top breed matches native Swift's answer for that photo
(results/parity.json), the page's embedding is close to Swift's, the WebAssembly head reproduces Swift's
probabilities exactly on Swift's embedding, nothing scrolls sideways and the console stays clean.
Screenshots go to shots/ (not committed).

usage: python3 -m http.server 8833 -d docs &   python3 scripts/check-page.py [url] [ort-web-dist-dir]
The optional ort-web-dist-dir serves ONNX Runtime Web from a local copy of the npm package's dist/
instead of the CDN (same files, for machines without CDN access).
"""
import json
import math
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parent.parent
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8833/"
ort_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
out = root / "shots"
out.mkdir(exist_ok=True)
parity = json.loads((root / "results/parity.json").read_text())
SAMPLE = 4  # the pug
ref = parity["samples"][SAMPLE]
failures = []


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def route_ort(route):
    name = route.request.url.split("/dist/")[-1].split("?")[0]
    path = ort_dir / name
    if not path.exists():
        return route.abort()
    ctype = "application/wasm" if name.endswith(".wasm") else "text/javascript"
    route.fulfill(status=200, body=path.read_bytes(), headers={"content-type": ctype, "access-control-allow-origin": "*"})


with sync_playwright() as p:
    browser = p.chromium.launch()
    for scheme in ["light", "dark"]:
        for width in [1280, 375]:
            ctx = browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme, device_scale_factor=1)
            page = ctx.new_page()
            if ort_dir:
                page.route("https://cdn.jsdelivr.net/npm/onnxruntime-web@*/dist/*", route_ort)
            errors = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(url)
            page.wait_for_selector(".rosette svg")
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(300)
            at_rest = page.locator(".rosette").count()
            page.screenshot(path=str(out / f"{scheme}-{width}-rest.png"), full_page=True)
            page.locator(".sample").nth(SAMPLE).click()
            page.wait_for_function("document.getElementById('status').dataset.state === 'ready'", timeout=180000)
            page.wait_for_timeout(1200)  # let the rosettes finish pinning
            live = page.evaluate("window.lineage.last")
            head_diff = page.evaluate(
                "([e, p]) => { const q = window.lineage.judge.head(e); return Math.max(...p.map((v, i) => Math.abs(v - q[i]))); }",
                [ref["embedding"], ref["headProbs"]])
            cos = cosine(live["embedding"], ref["embedding"])
            sideways = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            status = page.text_content("#status-text")
            page.screenshot(path=str(out / f"{scheme}-{width}.png"), full_page=True)
            page.locator(".show").screenshot(path=str(out / f"{scheme}-{width}-ring.png"))
            # the theme toggle flips the scheme
            page.click("#theme")
            flipped = page.evaluate("document.documentElement.dataset.theme")
            top = live["top"][0]
            print(f"{scheme:5} {width:4}px  rest rosettes {at_rest}  live top-1 {top['breed']} {top['p']:.3f} "
                  f"(swift {ref['top5'][0]['breed']} {ref['top5'][0]['p']:.3f})  embedding cosine {cos:.4f}  "
                  f"head diff {head_diff:.1e}  sideways {sideways}px  toggle -> {flipped}  errors {len(errors)}")
            print("            ", status)
            if errors: failures.append(f"{scheme} {width}: console errors {errors}")
            if sideways > 0: failures.append(f"{scheme} {width}: scrolls sideways by {sideways}px")
            if at_rest != 3: failures.append(f"{scheme} {width}: {at_rest} rosettes at rest")
            if top["breed"] != ref["top5"][0]["breed"]: failures.append(f"{scheme} {width}: live top-1 {top['breed']} != Swift's")
            if cos < 0.97: failures.append(f"{scheme} {width}: embedding cosine {cos:.4f}")
            if head_diff > 1e-5: failures.append(f"{scheme} {width}: WebAssembly head differs by {head_diff}")
            if flipped == scheme: failures.append(f"{scheme} {width}: theme toggle did nothing")
            ctx.close()
    browser.close()
if failures:
    print("FAILED")
    for f in failures:
        print(" ", f)
    sys.exit(1)
print("all checks passed; screenshots in", out)
