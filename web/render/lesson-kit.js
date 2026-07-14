/* lesson-kit.js — shared building blocks for lessons, so each one
   stops re-declaring the same SVG helpers. These are globals; older
   lessons that still define their own local copies simply shadow
   these (harmless) until they're migrated. All on the vermilion
   "instrument plate" palette. */

const KIT = {
  accent: "#d6401f",
  ink: "#e8e4da",
  faint: "#7d8595",
  dim: "#55524a",
  gridline: "#2c2c22",
  dead: "#191913", // a zeroed (ReLU-killed) cell
};

function section(n, title) {
  return `<div class="lz-section"><span class="lz-secnum">${n}</span>${escapeHtml(title)}</div>`;
}
function note(html) { return `<div class="lz-note">${html}</div>`; }
function txt(x, y, str, cls, anchor = "middle") {
  return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${escapeHtml(str)}</text>`;
}
function arrow(x0, y0, x1, y1, label) {
  let s = `<line x1="${x0}" y1="${y0}" x2="${x1 - 7}" y2="${y1}" class="lz-arrow"/>`;
  s += `<path d="M ${x1 - 7} ${y1 - 4} L ${x1} ${y1} L ${x1 - 7} ${y1 + 4} Z" class="lz-arrowhead"/>`;
  if (label) s += `<text x="${(x0 + x1) / 2}" y="${y0 - 8}" class="lz-arrowlabel" text-anchor="middle">${escapeHtml(label)}</text>`;
  return s;
}
function svg(w, h, inner, cap) {
  return `<div class="lz-diagram">${cap ? `<div class="lz-diagram-cap">${escapeHtml(cap)}</div>` : ""}` +
    `<div class="lz-svgwrap"><svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid meet">${inner}</svg></div></div>`;
}

/* token selector row. Caller wires clicks: querySelectorAll('.lz-chip'). */
function chipRow(tokens, chars, active, label) {
  let s = `<div class="lz-chips"><span class="lz-chips-label">${escapeHtml(label || "click a position to inspect it")}</span><div class="lz-chip-row">`;
  tokens.forEach((t, i) => {
    s += `<button class="lz-chip${i === active ? " on" : ""}" data-i="${i}" title="position ${i}">` +
      `<span class="lz-chip-g">${escapeHtml(glyph(chars[i]))}</span><span class="lz-chip-id">pos ${i}</span></button>`;
  });
  return s + "</div></div>";
}

/* one 1×N vector as a coloured row. opts: {label, sub, max, cw, dead, tip, cap} */
function vecDiagram(vec, opts = {}) {
  const N = vec.length;
  const cw = opts.cw || (N > 120 ? 4 : 11);
  const rowH = 26, padL = 108, padT = 16;
  const gridW = N * cw;
  const w = padL + gridW + 30, h = padT + rowH + 22;
  const mx = opts.max != null ? opts.max : maxAbs([vec]);
  let s = "";
  if (opts.label) s += txt(padL - 12, padT + rowH / 2 + 3, opts.label, "lz-slab on", "end");
  if (opts.sub) s += txt(padL - 12, padT + rowH / 2 + 16, opts.sub, "lz-subslab", "end");
  for (let c = 0; c < N; c++) {
    const v = vec[c];
    const fill = opts.dead && v === 0 ? KIT.dead : colorDiverging(v, mx);
    const tip = opts.tip ? opts.tip(c, v) : `dim ${c} = ${v.toFixed(4)}`;
    s += `<rect x="${padL + c * cw}" y="${padT}" width="${cw - (cw > 6 ? 0.5 : 0.3)}" height="${rowH}" fill="${fill}" data-tip="${tip}"/>`;
  }
  s += `<rect x="${padL}" y="${padT}" width="${gridW}" height="${rowH}" class="lz-outframe"/>`;
  return svg(w, h, s, opts.cap || "");
}

/* stacked rows that add: rows=[{vec,label,sub,max,op,out}] (same length). */
function sumDiagram(rows, cap) {
  const N = rows[0].vec.length;
  const cw = N > 120 ? 4 : 11, padL = 108, rowH = 26, gapR = 30, padT = 16;
  const gridW = N * cw;
  const w = padL + gridW + 30, h = padT + rows.length * rowH + (rows.length - 1) * gapR + 20;
  let s = "";
  rows.forEach((r, ri) => {
    const mx = r.max != null ? r.max : maxAbs([r.vec]);
    const y = padT + ri * (rowH + gapR);
    s += txt(padL - 12, y + rowH / 2 + 3, r.label, r.out ? "lz-slab on" : "lz-slab", "end");
    if (r.sub) s += txt(padL - 12, y + rowH / 2 + 16, r.sub, "lz-subslab", "end");
    for (let c = 0; c < N; c++) {
      const tip = r.tip ? r.tip(c) : `dim ${c}: ${r.label} = ${r.vec[c].toFixed(4)}`;
      s += `<rect x="${padL + c * cw}" y="${y}" width="${cw - (cw > 6 ? 0.5 : 0.3)}" height="${rowH}" fill="${colorDiverging(r.vec[c], mx)}" data-tip="${tip}"/>`;
    }
    if (r.out) s += `<rect x="${padL}" y="${y}" width="${gridW}" height="${rowH}" class="lz-outframe"/>`;
    if (r.op && ri < rows.length - 1) s += txt(padL + gridW + 16, y + rowH + gapR / 2 + 4, r.op, "lz-op");
  });
  return svg(w, h, s, cap);
}

/* bar chart over a vocabulary. opts: {labels, highlight, signed, cap} */
function barsDiagram(values, opts = {}) {
  const N = values.length, labels = opts.labels || values.map((_, i) => i);
  const bw = 15, g = 3, padL = 34, padT = 16, chartH = 150;
  const w = padL + N * (bw + g) + 12, h = padT + chartH + 34;
  const max = Math.max(...values), min = Math.min(...values);
  const span = opts.signed ? Math.max(Math.abs(max), Math.abs(min)) || 1 : max || 1;
  const base = opts.signed ? padT + chartH / 2 : padT + chartH;
  let s = "";
  s += `<line x1="${padL}" y1="${base}" x2="${padL + N * (bw + g)}" y2="${base}" class="lz-arrow"/>`;
  for (let i = 0; i < N; i++) {
    const v = values[i];
    const bh = (Math.abs(v) / span) * (opts.signed ? chartH / 2 : chartH);
    const x = padL + i * (bw + g);
    const y = opts.signed ? (v >= 0 ? base - bh : base) : base - bh;
    const hot = i === opts.highlight;
    s += `<rect x="${x}" y="${y}" width="${bw}" height="${bh}" fill="${hot ? KIT.accent : "#4a4a3e"}" data-tip="${escapeHtml(String(labels[i]))} = ${v.toFixed(4)}"/>`;
    s += `<text x="${x + bw / 2}" y="${padT + chartH + 12}" class="${hot ? "lz-tick on" : "lz-tick"}" text-anchor="middle">${escapeHtml(String(labels[i]))}</text>`;
  }
  return svg(w, h, s, opts.cap || "");
}
