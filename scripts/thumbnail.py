#!/usr/bin/env python3
"""Renders .github/thumbnail/thumbnail.html to .github/thumbnail.png (1200x627 at 2x) and a 1200x627 JPEG
copy to docs/thumbnail.jpg for og:image. The numbers come from results/holdout.json.
Needs Python Playwright and Pillow."""
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parent.parent
h = json.loads((root / "results/holdout.json").read_text())
tokens = {
    "top1": f'{h["final"]["top1"] * 100:.1f}%',
    "top5": f'{h["final"]["top5"] * 100:.1f}%',
    "test_n": f'{h["testPhotos"]:,}',
}
src = root / ".github/thumbnail/thumbnail.html"
html = src.read_text()
for k, v in tokens.items():
    html = html.replace("{{" + k + "}}", v)
assert "{{" not in html
tmp = src.with_name(".render.html")
tmp.write_text(html)
try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 627}, device_scale_factor=2)
        page.goto(tmp.as_uri())
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(400)
        page.screenshot(path=str(root / ".github/thumbnail.png"))
        browser.close()
finally:
    tmp.unlink()
png = root / ".github/thumbnail.png"
img = Image.open(png).convert("RGB")
img.save(png, optimize=True)
img.resize((1200, 627), Image.LANCZOS).save(root / "docs/thumbnail.jpg", quality=86, optimize=True, progressive=True)
print("wrote .github/thumbnail.png and docs/thumbnail.jpg", tokens)
