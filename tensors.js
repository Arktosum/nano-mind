/* tensors.js — draws the output tensor of a Step into the dark
   instrument panel. Canvas heatmaps for matrices/attention, DOM
   bars for logits/probs, badges for tokens. Everything is
   hoverable down to the exact float. */

const PANEL_BG = "#0d0d0b";
const POLE_POS = [214, 64, 31]; // vermilion (positive)
const POLE_NEG = [47, 111, 176]; // blue (negative)
const POLE_SEQ = [214, 64, 31]; // attention 0..1

let _tip;
function tip() {
  if (!_tip) {
    _tip = document.createElement("div");
    _tip.className = "tensor-tip";
    document.body.appendChild(_tip);
  }
  return _tip;
}
function showTip(html, x, y) {
  const t = tip();
  t.innerHTML = html;
  t.style.display = "block";
  t.style.left = Math.min(x + 14, window.innerWidth - t.offsetWidth - 8) + "px";
  t.style.top = Math.max(y - 8, 8) + "px";
}
function hideTip() {
  if (_tip) _tip.style.display = "none";
}

function lerp(a, b, t) {
  return Math.round(a + (b - a) * t);
}
function rgb(c) {
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
// signed value → color on the dark panel (blue ← 0 → vermilion)
function colorDiverging(v, maxAbs) {
  const t = maxAbs === 0 ? 0 : Math.max(-1, Math.min(1, v / maxAbs));
  const pole = t >= 0 ? POLE_POS : POLE_NEG;
  const m = Math.abs(t);
  const base = [20, 20, 15];
  return rgb([lerp(base[0], pole[0], m), lerp(base[1], pole[1], m), lerp(base[2], pole[2], m)]);
}
// [0,1] value → sequential intensity
function colorSeq(v) {
  const base = [20, 20, 15];
  return rgb([
    lerp(base[0], POLE_SEQ[0], v),
    lerp(base[1], POLE_SEQ[1], v),
    lerp(base[2], POLE_SEQ[2], v),
  ]);
}

function maxAbs(m) {
  let mx = 0;
  for (const r of m) for (const v of r) if (Math.abs(v) > mx) mx = Math.abs(v);
  return mx;
}

/* ── dispatcher ──────────────────────────────────────────── */
function renderInstrument(host, viz) {
  host.innerHTML = "";
  hideTip();
  switch (viz.type) {
    case "tokens": return renderTokens(host, viz);
    case "matrix": return renderMatrix(host, viz);
    case "attention": return renderAttention(host, viz);
    case "bars": return renderBars(host, viz);
  }
}

function renderTokens(host, { tokens, chars }) {
  const row = document.createElement("div");
  row.className = "token-row";
  tokens.forEach((t, i) => {
    const el = document.createElement("div");
    el.className = "tok";
    el.innerHTML = `<span class="tok-glyph">${escapeHtml(glyph(chars[i]))}</span><span class="tok-id">${t}</span>`;
    row.appendChild(el);
  });
  host.appendChild(row);
}

/* generic canvas heatmap for matrices & attention */
function heatmap(host, data, opts) {
  const rows = data.length;
  const cols = data[0].length;
  const rowLabels = opts.rowLabels || [];
  const colLabels = opts.colLabels || null;
  const square = opts.square;
  const dpr = window.devicePixelRatio || 1;

  const wrap = document.createElement("div");
  wrap.className = "hm-wrap";

  const caption = document.createElement("div");
  caption.className = "hm-caption";
  caption.innerHTML =
    `<span>${rows} &times; ${cols}</span>` +
    (opts.colLabel ? `<span class="hm-col">${opts.colLabel}</span>` : "");
  wrap.appendChild(caption);

  const scroll = document.createElement("div");
  scroll.className = "hm-scroll";
  const canvas = document.createElement("canvas");
  scroll.appendChild(canvas);
  wrap.appendChild(scroll);
  host.appendChild(wrap);

  // pick cell size: attention stays square & roomy; wide matrices get thin cells
  const gutterL = 30; // row labels
  const gutterT = colLabels ? 16 : 4;
  const avail = Math.max(host.clientWidth - 48, 240);
  let cell = square ? Math.min(26, Math.floor((avail - gutterL) / cols)) : Math.floor((avail - gutterL) / cols);
  cell = Math.max(cell, square ? 12 : 5);
  const rowH = square ? cell : Math.min(cell + 6, 22);

  const w = gutterL + cols * cell;
  const h = gutterT + rows * rowH;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "middle";

  const mx = opts.seq ? 1 : maxAbs(data);

  for (let r = 0; r < rows; r++) {
    // row label
    ctx.fillStyle = "#8f897a";
    ctx.textAlign = "right";
    ctx.fillText(rowLabels[r] ?? r, gutterL - 6, gutterT + r * rowH + rowH / 2);
    for (let c = 0; c < cols; c++) {
      const v = data[r][c];
      let fill;
      if (opts.seq) fill = square && c > r ? "#0b0b09" : colorSeq(v);
      else fill = colorDiverging(v, mx);
      ctx.fillStyle = fill;
      ctx.fillRect(gutterL + c * cell, gutterT + r * rowH, cell - (cell > 8 ? 1 : 0.4), rowH - (rowH > 8 ? 1 : 0.4));
    }
  }
  // sparse column ticks for wide non-square matrices
  if (!square && cols > 8) {
    ctx.fillStyle = "#6f6c60";
    ctx.textAlign = "center";
    const stepc = Math.ceil(cols / 8);
    for (let c = 0; c < cols; c += stepc) {
      ctx.fillText(colLabels ? colLabels[c] : c, gutterL + c * cell + cell / 2, gutterT - 8);
    }
  } else if (square && colLabels) {
    ctx.fillStyle = "#6f6c60";
    ctx.textAlign = "center";
    for (let c = 0; c < cols; c++) ctx.fillText(colLabels[c], gutterL + c * cell + cell / 2, 8);
  }

  // group separators (concat heads)
  if (opts.groups) {
    ctx.strokeStyle = "rgba(214,64,31,0.5)";
    ctx.lineWidth = 1;
    const per = cols / opts.groups;
    for (let gI = 1; gI < opts.groups; gI++) {
      const gx = gutterL + gI * per * cell;
      ctx.beginPath();
      ctx.moveTo(gx, gutterT);
      ctx.lineTo(gx, gutterT + rows * rowH);
      ctx.stroke();
    }
  }

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left - gutterL;
    const cy = e.clientY - rect.top - gutterT;
    const c = Math.floor(cx / cell);
    const r = Math.floor(cy / rowH);
    if (r < 0 || r >= rows || c < 0 || c >= cols) return hideTip();
    const rl = rowLabels[r] ?? r;
    const cl = colLabels ? colLabels[c] : c;
    showTip(
      `<b>[${escapeHtml(String(rl))}, ${escapeHtml(String(cl))}]</b><span>${data[r][c].toFixed(4)}</span>`,
      e.clientX,
      e.clientY
    );
  });
  canvas.addEventListener("mouseleave", hideTip);
}

function renderMatrix(host, viz) {
  heatmap(host, viz.data, {
    rowLabels: viz.rowLabels,
    colLabels: viz.colLabels,
    colLabel: viz.colLabel,
    square: viz.square,
    groups: viz.groups,
  });
}

function renderAttention(host, viz) {
  heatmap(host, viz.data, {
    rowLabels: viz.rowLabels,
    colLabels: viz.colLabels,
    square: true,
    seq: true,
  });
}

/* bar chart for logits / probabilities */
function renderBars(host, { values, labels, signed, highlight }) {
  const wrap = document.createElement("div");
  wrap.className = "bars-wrap";
  const max = Math.max(...values);
  const min = Math.min(...values);
  const span = signed ? Math.max(Math.abs(max), Math.abs(min)) || 1 : max || 1;

  values.forEach((v, i) => {
    const col = document.createElement("div");
    col.className = "bar-col" + (i === highlight ? " hot" : "");
    const track = document.createElement("div");
    track.className = "bar-track";
    const bar = document.createElement("div");
    bar.className = "bar" + (signed && v < 0 ? " neg" : "");
    const pct = (Math.abs(v) / span) * 100;
    if (signed) {
      bar.style.height = pct / 2 + "%";
      bar.style.alignSelf = v < 0 ? "flex-start" : "flex-end";
    } else {
      bar.style.height = pct + "%";
    }
    track.appendChild(bar);
    const lab = document.createElement("div");
    lab.className = "bar-label";
    lab.textContent = labels[i];
    col.appendChild(track);
    col.appendChild(lab);
    col.title = `${labels[i]} = ${v.toFixed(4)}`;
    col.addEventListener("mousemove", (e) =>
      showTip(`<b>${escapeHtml(labels[i])}</b><span>${v.toFixed(4)}</span>`, e.clientX, e.clientY)
    );
    col.addEventListener("mouseleave", hideTip);
    wrap.appendChild(col);
  });
  host.appendChild(wrap);
}

/* residual-stream spine: one mini-column per token, intensity = L2 norm */
function renderSpine(strip, stream) {
  strip.innerHTML = "";
  if (!stream) return;
  const norms = stream.map((row) => Math.sqrt(row.reduce((a, b) => a + b * b, 0)));
  const mx = Math.max(...norms) || 1;
  norms.forEach((n) => {
    const cell = document.createElement("span");
    cell.className = "spine-cell";
    cell.style.opacity = (0.2 + 0.8 * (n / mx)).toFixed(2);
    strip.appendChild(cell);
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}
