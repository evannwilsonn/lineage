// Lineage page script: draws the ring, the awards, the notebook and the index, and runs the judge.
// The judging pipeline: canvas shrink -> Swift (WebAssembly) framing + standardising -> MobileNetV4 in
// ONNX Runtime Web -> Swift (WebAssembly) head -> rosettes. This file only moves bytes and draws.
"use strict";

const DATA = JSON.parse(document.getElementById("data").textContent);
const HEAD = DATA.head;
const ORT_VERSION = "1.22.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const SHORT_SIDE = 272; // the training photos were shrunk to this before Swift framed them
const $ = (id) => document.getElementById(id);
const fmtPct = (p) => (p * 100).toFixed(p >= 0.995 ? 0 : 1) + "%";
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

// ---------- theme ----------
const root = document.documentElement;
function currentTheme() {
  return root.dataset.theme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
function syncThemeButton() {
  const dark = currentTheme() === "dark";
  $("theme").setAttribute("aria-pressed", String(dark));
  $("theme-label").textContent = dark ? "Day show" : "Evening ring";
  $("theme").setAttribute("aria-label", dark ? "Switch to the daytime catalogue (light theme)" : "Switch to the evening ring (dark theme)");
}
$("theme").addEventListener("click", () => {
  root.dataset.theme = currentTheme() === "dark" ? "light" : "dark";
  try { localStorage.setItem("lineage-theme", root.dataset.theme); } catch (e) {}
  syncThemeButton();
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncThemeButton);
syncThemeButton();

// ---------- awards ----------
function hashNumber(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return String(1000 + ((h >>> 0) % 9000));
}

function showAwards(top, how) {
  const ros = $("rosettes");
  ros.innerHTML = "";
  top.slice(0, 3).forEach((a, i) => {
    const fig = document.createElement("figure");
    fig.className = `rosette rosette--${i + 1}`;
    fig.innerHTML = window.rosetteSVG({ place: i + 1, breed: a.breed, prob: a.p, uid: `r${i}${Date.now() % 1e6}` });
    if (!reduceMotion.matches) fig.classList.add("pinned");
    ros.appendChild(fig);
  });
  const com = $("commended");
  com.innerHTML = "";
  top.slice(3, 5).forEach((a, i) => {
    const li = document.createElement("li");
    li.className = "ribbon";
    li.innerHTML = `<span class="rk">${i === 0 ? "4th" : "5th"} · Highly commended</span><span class="rb"></span><span class="rp">${fmtPct(a.p)}</span>`;
    li.querySelector(".rb").textContent = a.breed;
    li.setAttribute("aria-label", `${i === 0 ? "4th" : "5th"}, highly commended: ${a.breed}, ${fmtPct(a.p)}`);
    com.appendChild(li);
  });
  $("judged").textContent = how;
  const p = top[0].p, gap = top[0].p - top[1].p;
  let v;
  if (p >= 0.8) v = `A clear winner: the ${top[0].breed} takes ${fmtPct(p)} of the judge's belief.`;
  else if (p >= 0.45) v = `The judge leans ${top[0].breed}, but the ${top[1].breed} is in the running (${fmtPct(top[1].p)}).`;
  else v = `A split decision: no breed gets half the judge's belief. That happens with mixed breeds, puppies, odd angles, and anything that isn't one of the 120 breeds.`;
  if (gap < 0.05 && p < 0.8) v += " First and second are nearly tied.";
  $("verdict").textContent = v;
}

// ---------- the ring ----------
let currentUrl = null;
function putInRing(src, alt, credit, key) {
  const img = $("photo");
  img.src = src;
  img.alt = alt;
  $("photo-credit").textContent = credit || "";
  $("armband-no").textContent = hashNumber(key || src);
}

function setStatus(state, text) {
  $("status").dataset.state = state;
  $("status-text").textContent = text;
}

// ---------- samples ----------
const sampleList = $("samples");
DATA.samples.forEach((s, i) => {
  const li = document.createElement("li");
  const b = document.createElement("button");
  b.type = "button";
  b.className = "sample";
  b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
  b.innerHTML = `<img src="${s.sample.file}" alt="" width="34" height="34" loading="lazy"><span></span>`;
  b.querySelector("span").textContent = s.sample.label;
  b.addEventListener("click", () => judgeSample(i));
  li.appendChild(b);
  sampleList.appendChild(li);
});
$("sample-credits").textContent = DATA.samples.map((s) => `${s.sample.label.toLowerCase()} by ${s.sample.creator}`).join(", ");

function pressSample(i) {
  sampleList.querySelectorAll(".sample").forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
}

// at rest: the first sample, as judged by the Swift build
{
  const s = DATA.samples[0];
  putInRing(s.sample.file, `Sample exhibit: ${s.sample.label.toLowerCase()} (CC0 photo by ${s.sample.creator})`, `${s.sample.label} · CC0, ${s.sample.creator}`, s.sample.file);
  showAwards(s.top5, "Pre-judged by the Swift build. Pick a photo to judge it live in your browser.");
  $("rosettes").querySelectorAll(".rosette").forEach((r) => r.classList.remove("pinned"));
}

// ---------- the judge: ONNX Runtime Web + Swift WebAssembly ----------
let judgePromise = null;
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src; s.crossOrigin = "anonymous"; s.onload = resolve; s.onerror = () => reject(new Error("could not load " + src));
    document.head.appendChild(s);
  });
}

async function fetchWithProgress(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  // content-length is the size on the wire; a gzipped response (GitHub Pages compresses .onnx)
  // decodes to more bytes than that, so collect chunks rather than preallocating.
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    const pct = Math.min(99, Math.round((got / total) * 100));
    setStatus("loading", `${label}: ${pct}% of ${(total / 1048576).toFixed(0)} MB`);
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out;
}

function loadJudge() {
  if (judgePromise) return judgePromise;
  judgePromise = (async () => {
    setStatus("loading", "Calling the judge: loading ONNX Runtime Web");
    const [wasmBytes, headBytes] = await Promise.all([
      fetch("lineage.wasm").then((r) => r.arrayBuffer()),
      fetch("head.bin").then((r) => r.arrayBuffer()),
      loadScript(ORT_BASE + "ort.wasm.min.js"),
    ]);
    const swift = (await WebAssembly.instantiate(wasmBytes, {})).instance.exports;
    const model = await fetchWithProgress("backbone.onnx", "Fetching the backbone");
    window.ort.env.wasm.wasmPaths = ORT_BASE;
    window.ort.env.wasm.numThreads = 1;
    setStatus("loading", "Starting the backbone");
    const session = await window.ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
    return makeJudge(swift, new Float32Array(headBytes), session);
  })();
  judgePromise.catch((e) => {
    judgePromise = null;
    setStatus("error", `The judge could not start (${e.message}). The awards shown are the Swift build's.`);
  });
  return judgePromise;
}

// Linear-memory layout, all float32 except the photo: [head params][embedding][head scratch][probs][prep out][prep scratch][rgba]
function makeJudge(swift, headParams, session) {
  const mem = swift.memory;
  const align = (n) => (n + 15) & ~15;
  const base = align(swift.__heap_base.value);
  const nParams = swift.lineage_head_floats(HEAD.inputs, HEAD.hidden, HEAD.classes);
  if (nParams !== headParams.length) throw new Error(`head.bin has ${headParams.length} floats, expected ${nParams}`);
  const pParams = base;
  const pEmb = align(pParams + nParams * 4);
  const pScratch = align(pEmb + HEAD.inputs * 4);
  const pProbs = align(pScratch + (HEAD.inputs + HEAD.hidden) * 4);
  const pOut = align(pProbs + HEAD.classes * 4);
  const side = HEAD.side;
  const pPrep = align(pOut + 3 * side * side * 4);
  function ensure(bytes) {
    const need = bytes - mem.buffer.byteLength;
    if (need > 0) mem.grow(Math.ceil(need / 65536));
  }
  ensure(pPrep);
  new Float32Array(mem.buffer, pParams, nParams).set(headParams);

  function head(embedding) {
    new Float32Array(mem.buffer, pEmb, HEAD.inputs).set(embedding);
    swift.lineage_head(pEmb, HEAD.inputs, HEAD.hidden, HEAD.classes, HEAD.l2 ? 1 : 0, pParams, pScratch, pProbs);
    return Float32Array.from(new Float32Array(mem.buffer, pProbs, HEAD.classes));
  }

  function prepare(rgba, w, h, framing, flip) {
    const nScratch = swift.lineage_prepare_scratch(h, side);
    const pRGBA = align(pPrep + nScratch * 4);
    ensure(pRGBA + rgba.length);
    new Uint8Array(mem.buffer, pRGBA, rgba.length).set(rgba);
    swift.lineage_prepare_rgba(pRGBA, w, h, framing, side, flip ? 1 : 0, pPrep, pOut);
    return Float32Array.from(new Float32Array(mem.buffer, pOut, 3 * side * side));
  }

  async function embed(input) {
    const t = new window.ort.Tensor("float32", input, [1, 3, side, side]);
    const out = await session.run({ pixel_values: t });
    return out.features.data;
  }

  // average the head's probabilities over the test views (one backbone pass each)
  const VIEWS = { squash: [0, false], crop: [1, false], "squash-flip": [0, true], "crop-flip": [1, true] };
  async function judge(rgba, w, h) {
    let first = null, probs = null;
    for (const name of HEAD.views) {
      const [framing, flip] = VIEWS[name];
      const e = await embed(prepare(rgba, w, h, framing, flip));
      const p = head(e);
      if (!first) first = e;
      probs = probs ? probs.map((v, i) => v + p[i]) : p;
    }
    return { embedding: first, probs: probs.map((v) => v / HEAD.views.length) };
  }
  return { judge, head, prepare };
}

// The photo, shrunk so its shorter side is 272 px (as the training photos were), as RGBA bytes.
async function pixelsOf(blobOrUrl) {
  let bitmap;
  if (typeof blobOrUrl === "string") {
    const blob = await (await fetch(blobOrUrl)).blob();
    bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  } else {
    bitmap = await createImageBitmap(blobOrUrl, { imageOrientation: "from-image" });
  }
  const s = Math.min(1, SHORT_SIDE / Math.min(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * s)), h = Math.max(1, Math.round(bitmap.height * s));
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close && bitmap.close();
  return { rgba: ctx.getImageData(0, 0, w, h).data, w, h };
}

function topFive(probs) {
  return Array.from(probs, (p, i) => ({ breed: HEAD.breeds[i], p, i })).sort((a, b) => b.p - a.p).slice(0, 5);
}

let runId = 0;
async function judgeSource(source, how) {
  const id = ++runId;
  $("ring").classList.add("busy");
  try {
    const judge = await loadJudge();
    if (id !== runId) return;
    setStatus("loading", "Judging");
    const t0 = performance.now();
    const { rgba, w, h } = await pixelsOf(source);
    const r = await judge.judge(rgba, w, h);
    if (id !== runId) return;
    const top = topFive(r.probs);
    window.lineage = { last: { embedding: Array.from(r.embedding), probs: Array.from(r.probs), top }, judge };
    showAwards(top, how);
    setStatus("ready", `Judged live in your browser in ${Math.round(performance.now() - t0)} ms.`);
  } catch (e) {
    if (judgePromise) setStatus("error", `Could not judge that photo (${e.message}).`);
  } finally {
    if (id === runId) $("ring").classList.remove("busy");
  }
}

function judgeSample(i) {
  const s = DATA.samples[i];
  pressSample(i);
  putInRing(s.sample.file, `Sample exhibit: ${s.sample.label.toLowerCase()} (CC0 photo by ${s.sample.creator})`, `${s.sample.label} · CC0, ${s.sample.creator}`, s.sample.file);
  judgeSource(s.sample.file, `Judged live in your browser. The sample is labelled ${s.sample.label.toLowerCase()}.`);
}

function judgeFile(file) {
  if (!file || !file.type.startsWith("image/")) { setStatus("error", "That isn't an image file."); return; }
  pressSample(-1);
  if (currentUrl) URL.revokeObjectURL(currentUrl);
  currentUrl = URL.createObjectURL(file);
  putInRing(currentUrl, "Your exhibit", "Your exhibit · stays on this device", file.name + file.size);
  judgeSource(file, "Judged live in your browser.");
}

$("file").addEventListener("change", (e) => judgeFile(e.target.files[0]));
const ring = $("ring");
["dragenter", "dragover"].forEach((t) => ring.addEventListener(t, (e) => { e.preventDefault(); ring.classList.add("drag"); $("drop-hint").textContent = "Release to enter the ring"; }));
["dragleave", "drop"].forEach((t) => ring.addEventListener(t, () => { ring.classList.remove("drag"); $("drop-hint").textContent = "Drop a dog photo in the ring"; }));
ring.addEventListener("drop", (e) => { e.preventDefault(); judgeFile(e.dataTransfer.files[0]); });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => { if (!ring.contains(e.target)) { e.preventDefault(); judgeFile(e.dataTransfer.files[0]); } });
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((it) => it.type.startsWith("image/"));
  if (item) judgeFile(item.getAsFile());
});

// ---------- lookalikes ----------
DATA.lookalikes.forEach((p) => {
  const li = document.createElement("li");
  li.className = "pair";
  li.innerHTML = `<div class="pair-names"><span class="a"></span><span class="vs">and</span><span class="b"></span></div>
    <p class="pair-tally"><span class="t"></span></p><span class="count" aria-hidden="true">${p.aAsB + p.bAsA}</span>`;
  li.querySelector(".a").textContent = p.a;
  li.querySelector(".b").textContent = p.b;
  const parts = [];
  if (p.aAsB) parts.push(`${p.aAsB} of ${p.nA} ${p.a} photos called ${p.b}`);
  if (p.bAsA) parts.push(`${p.bAsA} of ${p.nB} the other way`);
  li.querySelector(".t").textContent = parts.join("; ") + ".";
  $("pairs").appendChild(li);
});

// ---------- breed index ----------
function marker(acc) {
  return acc >= 0.95 ? ["b", "blue"] : acc >= 0.85 ? ["r", "red"] : acc >= 0.7 ? ["y", "yellow"] : ["w", "white"];
}
let sortMode = "az";
function drawIndex() {
  const q = $("find").value.trim().toLowerCase();
  let rows = DATA.breeds.filter((b) => b.breed.toLowerCase().includes(q));
  const name = (b) => b.breed.toLowerCase();
  if (sortMode === "az") rows.sort((a, b) => name(a).localeCompare(name(b)));
  if (sortMode === "best") rows.sort((a, b) => b.top1 - a.top1 || name(a).localeCompare(name(b)));
  if (sortMode === "worst") rows.sort((a, b) => a.top1 - b.top1 || name(a).localeCompare(name(b)));
  const ul = $("breeds");
  ul.innerHTML = "";
  if (!rows.length) { ul.innerHTML = `<li class="empty">No breed in the show matches that.</li>`; return; }
  for (const b of rows) {
    const [m, word] = marker(b.top1);
    const li = document.createElement("li");
    li.innerHTML = `<span class="mk ${m}" role="img" aria-label="${word} ribbon"></span><div><span class="bn"></span><small class="conf"></small></div><span class="lead"></span><span class="ba">${fmtPct(b.top1)}</span>`;
    li.querySelector(".bn").innerHTML = `<span></span><small>${b.n} photos</small>`;
    li.querySelector(".bn span").textContent = b.breed;
    li.querySelector(".conf").textContent = b.mostConfusedWith && b.mostConfusedCount ? `mistaken for ${b.mostConfusedWith} ×${b.mostConfusedCount}` : "";
    ul.appendChild(li);
  }
}
$("find").addEventListener("input", drawIndex);
document.querySelectorAll(".sort button").forEach((btn) => btn.addEventListener("click", () => {
  sortMode = btn.dataset.sort;
  document.querySelectorAll(".sort button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  drawIndex();
}));
drawIndex();
