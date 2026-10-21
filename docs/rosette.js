// Show rosettes drawn as SVG: two layers of pleated ribbon, a printed centre button, and two
// swallow-tailed streamers lettered with the breed and the probability. Colours come from CSS custom
// properties on the .rosette--1/2/3 classes, so both themes (and the thumbnail) restyle them.
// Shared by the page and .github/thumbnail/thumbnail.html.

(function (global) {
  const CX = 100, CY = 100;

  function polar(r, a) {
    return [(CX + r * Math.cos(a)).toFixed(2), (CY + r * Math.sin(a)).toFixed(2)];
  }

  // One ring of pleats: each pleat is a lit half and a shaded half meeting at a fold,
  // with the outer edge dipping between pleats so it reads as gathered ribbon.
  function pleats(n, rOut, rIn, offset, lit, shade) {
    let s = "";
    const step = (Math.PI * 2) / n;
    for (let i = 0; i < n; i++) {
      const a0 = offset + i * step, a1 = a0 + step / 2, a2 = a0 + step;
      const p0 = polar(rOut, a0), p1 = polar(rIn, a1), p2 = polar(rOut, a2);
      const c = polar(rIn * 0.35, a1);
      s += `<path d="M${c} L${p0} L${p1} Z" fill="${lit}"/>`;
      s += `<path d="M${c} L${p1} L${p2} Z" fill="${shade}"/>`;
    }
    return s;
  }

  function esc(t) {
    return String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function tail(x, angle, uid) {
    // a satin streamer hanging from under the rosette, forked at the end
    const w = 27, top = 112, bottom = 372, fork = 22;
    return `<g transform="rotate(${angle} ${CX} ${CY})">
      <path d="M${x - w},${top} H${x + w} V${bottom} L${x},${bottom - fork} L${x - w},${bottom} Z" fill="url(#satin-${uid})"/>
      <path d="M${x - w + 4},${top + 4} H${x + w - 4} V${bottom - 6} L${x},${bottom - fork - 4.5} L${x - w + 4},${bottom - 6} Z"
        fill="none" stroke="var(--stitch)" stroke-width="0.9" stroke-dasharray="2.6 2.2" opacity="0.75"/>
    </g>`;
  }

  function fitSize(text, length, max) {
    // Gloock averages about 0.5 em per character
    return Math.max(9.5, Math.min(max, length / (text.length * 0.5)));
  }

  /**
   * @param {{place:number, breed:string, prob:number, uid:string, caption?:string, value?:string, valueLabel?:string}} o
   * @returns {string} an <svg> element
   */
  function rosetteSVG(o) {
    const uid = o.uid || Math.random().toString(36).slice(2, 8);
    const ord = ["1st", "2nd", "3rd"][o.place - 1] || `${o.place}th`;
    const pct = o.value || (o.prob * 100).toFixed(o.prob >= 0.995 ? 0 : 1) + "%";
    const nameSize = fitSize(o.breed, 150, 15.5);
    const lt = -11, rt = 11; // tail angles (left tail swings left)
    return `<svg class="rosette-svg" viewBox="0 0 200 380" role="img"
      aria-label="${esc(ord)} place: ${esc(o.breed)}, ${esc(pct)}">
    <defs>
      <linearGradient id="satin-${uid}" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0" stop-color="var(--dark)"/>
        <stop offset="0.32" stop-color="var(--light)"/>
        <stop offset="0.55" stop-color="var(--base)"/>
        <stop offset="1" stop-color="var(--dark)"/>
      </linearGradient>
      <radialGradient id="button-${uid}" cx="0.38" cy="0.32" r="0.85">
        <stop offset="0" stop-color="var(--button-hi)"/>
        <stop offset="0.7" stop-color="var(--button)"/>
        <stop offset="1" stop-color="var(--button-lo)"/>
      </radialGradient>
      <linearGradient id="sheen-${uid}" x1="0" y1="0" x2="0.7" y2="1">
        <stop offset="0" stop-color="#fff" stop-opacity="0.34"/>
        <stop offset="0.45" stop-color="#fff" stop-opacity="0"/>
        <stop offset="1" stop-color="#000" stop-opacity="0.16"/>
      </linearGradient>
      <path id="arc-top-${uid}" d="M${CX - 37},${CY} A37,37 0 0 1 ${CX + 37},${CY}"/>
      <path id="arc-bot-${uid}" d="M${CX - 41},${CY} A41,41 0 0 0 ${CX + 41},${CY}"/>
    </defs>
    ${tail(CX - 20, -lt, uid)}
    ${tail(CX + 20, -rt, uid)}
    <g transform="rotate(${-lt} ${CX} ${CY})" class="tail-text">
      <text x="${CX - 20}" y="202" transform="rotate(90 ${CX - 20} 202)" font-size="${nameSize.toFixed(1)}"
        class="rosette-breed" dominant-baseline="central">${esc(o.breed)}</text>
    </g>
    <g transform="rotate(${-rt} ${CX} ${CY})" class="tail-text">
      <text x="${CX + 20}" y="204" transform="rotate(90 ${CX + 20} 204)" class="rosette-small" font-size="7.4"
        dominant-baseline="central">${esc(o.valueLabel || "PROBABILITY")}</text>
      <text x="${CX + 20}" y="334" class="rosette-prob" text-anchor="middle" font-size="${pct.length > 5 ? 17 : 19}">${esc(pct)}</text>
    </g>
    <g class="ruffle">
      ${pleats(46, 94, 86, 0, "var(--light)", "var(--dark)")}
      ${pleats(38, 77, 71, Math.PI / 38, "var(--base)", "var(--dark)")}
      <circle cx="${CX}" cy="${CY}" r="94" fill="url(#sheen-${uid})"/>
      <circle cx="${CX}" cy="${CY}" r="60" fill="var(--dark)"/>
      <circle cx="${CX}" cy="${CY}" r="57.5" fill="none" stroke="var(--edge)" stroke-width="2.4"/>
    </g>
    <circle cx="${CX}" cy="${CY}" r="52" fill="url(#button-${uid})"/>
    <circle cx="${CX}" cy="${CY}" r="47" fill="none" stroke="var(--base)" stroke-width="1.3"/>
    <circle cx="${CX}" cy="${CY}" r="29" fill="none" stroke="var(--base)" stroke-width="0.6" opacity="0.6"/>
    <text class="rosette-arc" font-size="7.2"><textPath href="#arc-top-${uid}" startOffset="50%" text-anchor="middle">${esc(o.caption || "LINEAGE · RING 7")}</textPath></text>
    <text class="rosette-arc" font-size="7.2"><textPath href="#arc-bot-${uid}" startOffset="50%" text-anchor="middle">OF 120 BREEDS</textPath></text>
    <text x="${CX}" y="${CY + 1}" class="rosette-place" text-anchor="middle" dominant-baseline="central" font-size="30">${ord}</text>
    <circle cx="${CX}" cy="${CY - 89}" r="3.6" class="pin"/>
    <circle cx="${CX - 1}" cy="${CY - 90}" r="1.2" fill="#fff" opacity="0.8"/>
  </svg>`;
  }

  global.rosetteSVG = rosetteSVG;
})(typeof window !== "undefined" ? window : globalThis);
