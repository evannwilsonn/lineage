#!/usr/bin/env python3
"""Reads the Stanford Dogs images.tar as a stream (from stdin) and writes each photo, shrunk so its
shorter side is at most 272 px, to data/images/<synset-breed>/<file>.jpg. The tar is never stored.

This is the one data-preparation step that isn't Swift: Pillow decodes and shrinks the 20,580 JPEGs
(about 750 MB) to about a fifth of their size, so the Swift side has a small, uniform set to read.
Everything after it (decoding the shrunk JPEGs, resizing to 256, normalising, the network, the head)
happens in Swift.

usage: curl -sL http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar | python3 -I scripts/prepare_images.py data/images
"""
import io
import sys
import tarfile
from pathlib import Path

from PIL import Image

SHORT = 272  # >= 269, the shorter edge the backbone's own preprocessing asks for before its 256 crop

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
n = 0
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tar:
    for member in tar:
        if not member.isfile() or not member.name.lower().endswith(".jpg"):
            continue
        parts = Path(member.name).parts  # Images/n02085620-Chihuahua/n02085620_10074.jpg
        if len(parts) != 3:
            continue
        data = tar.extractfile(member).read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        s = SHORT / min(w, h)
        if s < 1:
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC, reducing_gap=3.0)
        d = out / parts[1]
        d.mkdir(exist_ok=True)
        img.save(d / parts[2], quality=90, optimize=True)
        n += 1
        if n % 2000 == 0:
            print(n, "images", file=sys.stderr, flush=True)
print(n, "images written to", out, file=sys.stderr)
