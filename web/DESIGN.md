# Design: Best in Show

The page is a dog show. Your photo is judged in the ring, and the model hands out rosettes. Everything else on
the page is the show's printed matter: the results board, the judge's notebook, the catalogue's breed index and
the rules.

## The concept, piece by piece

- **The ring.** An oval of show-ring felt with a white chalk ring line. The line is an SVG ellipse with an
  irregular dash pattern, roughened by a small `feTurbulence` displacement, so it looks laid down by hand
  rather than drawn by a compass. The photo sits in the middle as a mounted print (white border, deeper at
  the bottom like a Polaroid mount, tilted 1.2 degrees). The whole ring is the drop target: when a file is
  dragged over it, the chalk line thickens and turns solid and the hint reads "Release to enter the ring".
- **The armband.** A white exhibitor's armband pinned over the ring's edge, "RING 7" above a big condensed
  number and "EXHIBIT" below, with a safety pin drawn in CSS. The number is a hash of the photo, so the same
  photo always gets the same number. It is decorative and marked `aria-hidden`.
- **Rosettes (the signature).** Drawn in SVG by `web/rosette.js`, which the page and the thumbnail share.
  Each one has two rings of pleats (46 and 38 pleats, offset half a pleat). Every pleat is a lit half and a
  shaded half meeting at a fold, and the outer edge dips between pleats, which is what makes it read as
  gathered ribbon rather than a starburst. A soft white-to-black sheen gradient lies over the pleats, then a
  dark band with a cream edge, then the printed centre button: a radial-gradient card with "1st"/"2nd"/"3rd"
  in Gloock and curved small caps ("LINEAGE · RING 7", "OF 120 BREEDS") on text paths. Two satin streamers
  hang behind, swung 11 degrees apart, swallow-tailed, with a dashed stitch line inset along their edges.
  The left streamer is lettered with the breed (Gloock, sized to fit); the right with "PROBABILITY" and the
  number in Big Shoulders. A small brass pin head sits at the top. Each rosette has its own small rotation
  (-4, 3 and -2 degrees, and a slight drop) so the three look pinned by hand, and a short "pin" animation
  (drop and settle, staggered) when new awards come in, turned off under `prefers-reduced-motion`.
- **Show colours.** US placings: blue 1st, red 2nd, yellow 3rd. 4th and 5th get flat white "Highly
  commended" ribbons: a strip with swallow-tail notches at both ends (`clip-path`) and dashed stitching.
- **Lookalikes.** The judge's notebook: a ruled page with a margin line and punched holes, tilted slightly,
  listing the breed pairs most often confused on the test photos with tally marks and the counts in words.
- **Breed index.** The show catalogue's index of all 120 breeds, alphabetical (AKC-style groups would need a
  mapping several of these breeds don't have: dingo, dhole, African hunting dog, Eskimo dog). Dotted leaders
  run to each breed's test accuracy, and a small ribbon-tail marker in a show colour bands it (blue 95%+,
  red 85-95%, yellow 70-85%, white below). It can be searched and sorted.
- **Rules of the show.** The pipeline as four numbered rules, in plain words.

## Palette

Rosette colours are the only bright notes. Everything else is bottle-green ink or felt.

| Token | Light: the daytime catalogue | Dark: the evening ring |
|---|---|---|
| `--paper` | `#f2f3ec` cool ivory stock (greyer and greener than the usual cream) | `#0f2a20` deep bottle-green felt |
| `--ink` | `#1f4d3a` bottle-green ink (8.6:1 on paper) | `#edf0e6` ivory (13:1) |
| `--ink-2` | `#446555` (5.9:1) | `#b9cbbf` (9:1) |
| `--felt` | `#2c6346` ring felt | `#1d4a35`, with a warm spotlight (`--spot`) falling on the ring |
| blue / red / gold | `#1e4fa0` / `#b0232e` / `#e0b021` | `#2b62c6` / `#cc3340` / `#efc23a`, richer, with a faint warm glow |

The page background carries a faint noise tile: paper tooth in light, felt nap in dark. In dark mode the
ring gets a radial spotlight and the rosettes a soft glow, as if under the evening lights. The focus ring is
royal blue in light and gold in dark, 3 px, offset.

Avoided on purpose (other projects use them): navy and brass, cream paper with red, cobalt and yellow on
near-black, cinema red, cream newsprint, paper and deep teal, riso blue and pink, black and taxi yellow, mint
ECG paper, garnet and gold.

## Type

- **Gloock** (SIL OFL): the high-contrast display serif, for the title, headings, breed names on streamers,
  ribbons, the notebook and the index.
- **Instrument Sans** (SIL OFL): the sturdy humanist sans for everything you read or press: body, buttons,
  small caps labels.
- **Big Shoulders** (SIL OFL): condensed, for armband numbers, probabilities and the results board's figures,
  with tabular numerals.

All three are self-hosted from `docs/fonts/` with their licence files.

## At rest, and states

The page is complete before anything is downloaded: the first sample photo (a CC0 beagle) is in the ring
with the awards the Swift build gave it, and a line under the controls says so. Choosing a sample or your own
photo loads the judge (ONNX Runtime Web from a pinned jsDelivr URL, the 17 MB backbone with a progress count,
the 5 KB Swift WebAssembly and the head's weights) and replaces the awards with the live result. The status
line has a coloured pip: grey idle, gold loading, green ready, red error. Low-confidence results get a plain
"split decision" verdict rather than pretending.

## Layout and access

- 1280 px: the ring (7 columns) beside the awards (5). Below 980 px they stack; below 640 px the ring becomes
  nearly round so the photo stays large, the armband shrinks, and the index goes to one column. No sideways
  scroll at 375 px.
- Rosettes are `role="img"` with a full label ("1st place: Beagle, 92.1%"); ribbons have labels too.
  The armband and chalk line are hidden from assistive tech. The status line is a polite live region.
- The theme toggle is a real button with `aria-pressed`; the choice is remembered in `localStorage`
  (wrapped in try/catch) and otherwise follows `prefers-color-scheme`.

## Thumbnail

`.github/thumbnail/thumbnail.html` is the same piece: the evening ring's felt with a chalk line, the title in
Gloock, an armband reading "120 BREEDS", and the three rosettes fanned across the right, drawn by the same
`rosette.js`. The blue rosette's streamer carries the held-out top-1, the red the top-5, both filled in from
`results/holdout.json` by `scripts/thumbnail.py`; the yellow one says "Trained in Swift".
