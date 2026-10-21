// Runs the page's WebAssembly (Swift compiled with Embedded Swift) in Node and compares it with the
// native Swift answers in results/parity.json: the head on the sample photos' embeddings, and the
// photo preparation on synthetic test cards.
// usage: node scripts/check-wasm.mjs
import { readFileSync } from "node:fs";

const root = new URL("..", import.meta.url).pathname;
const parity = JSON.parse(readFileSync(root + "results/parity.json", "utf8"));
const head = JSON.parse(readFileSync(root + "docs/head.json", "utf8"));
const hb = readFileSync(root + "docs/head.bin");
const params = new Float32Array(hb.buffer, hb.byteOffset, hb.byteLength / 4);
const { instance } = await WebAssembly.instantiate(readFileSync(root + "docs/lineage.wasm"), {});
const w = instance.exports;
const align = (n) => (n + 15) & ~15;
let p = align(w.__heap_base.value);
const alloc = (bytes) => { const at = p; p = align(p + bytes); const need = p - w.memory.buffer.byteLength; if (need > 0) w.memory.grow(Math.ceil(need / 65536)); return at; };

const pParams = alloc(params.length * 4), pEmb = alloc(head.inputs * 4), pScratch = alloc((head.inputs + head.hidden) * 4), pProbs = alloc(head.classes * 4);
new Float32Array(w.memory.buffer, pParams, params.length).set(params);
if (w.lineage_head_floats(head.inputs, head.hidden, head.classes) !== params.length) throw new Error("head.bin size does not match");
let worstHead = 0;
for (const s of parity.samples) {
  new Float32Array(w.memory.buffer, pEmb, head.inputs).set(s.embedding);
  w.lineage_head(pEmb, head.inputs, head.hidden, head.classes, head.l2 ? 1 : 0, pParams, pScratch, pProbs);
  let probs = Float32Array.from(new Float32Array(w.memory.buffer, pProbs, head.classes));
  for (let i = 0; i < head.classes; i++) worstHead = Math.max(worstHead, Math.abs(probs[i] - s.headProbs[i]));
  const top = [...probs].map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])[0];
  console.log(`${s.sample.label.padEnd(18)} wasm top-1 ${head.breeds[top[1]]} ${(top[0] * 100).toFixed(2)}%   swift ${s.top5[0].breed} ${(s.top5[0].p * 100).toFixed(2)}%`);
}

let worstPrep = 0;
for (const c of parity.prepare) {
  const n = c.width * c.height;
  const rgba = new Uint8Array(n * 4);
  for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
    const i = (y * c.width + x) * 4;
    rgba[i] = (x * 7 + y * 3) % 256; rgba[i + 1] = (x * y) % 251; rgba[i + 2] = (x ^ y) & 255; rgba[i + 3] = 255;
  }
  const side = head.side;
  const pOut = alloc(3 * side * side * 4), pS = alloc(w.lineage_prepare_scratch(c.height, side) * 4), pR = alloc(rgba.length);
  new Uint8Array(w.memory.buffer, pR, rgba.length).set(rgba);
  w.lineage_prepare_rgba(pR, c.width, c.height, c.framing, side, c.flip ? 1 : 0, pS, pOut);
  const out = new Float32Array(w.memory.buffer, pOut, 3 * side * side);
  let sum = 0, sq = 0;
  for (const v of out) { sum += v; sq += v * v; }
  for (let i = 0; i < 16; i++) worstPrep = Math.max(worstPrep, Math.abs(out[i] - c.first[i]));
  const rel = Math.abs(sum - c.sum) / Math.max(1, Math.abs(c.sum)) + Math.abs(sq - c.sumSquares) / c.sumSquares;
  worstPrep = Math.max(worstPrep, rel);
  console.log(`prepare ${c.width}x${c.height} framing ${c.framing}${c.flip ? " flipped" : ""}: sum ${sum.toFixed(3)} vs ${c.sum.toFixed(3)}`);
}
console.log(`largest difference: head ${worstHead.toExponential(2)}, preparation ${worstPrep.toExponential(2)}`);
if (worstHead > 1e-5 || worstPrep > 1e-5) { console.error("FAILED: WebAssembly and native Swift disagree"); process.exit(1); }
console.log("WebAssembly matches native Swift");
