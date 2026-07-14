/* flow.js — the operation diagram (the point of the whole thing).
   Draws each Step as: operand tiles → operator → output tile, with
   arrows, so you can watch data move and sanity-check the shapes.
   Values are shown as a coarse colour texture, not for reading —
   that's what the "inspect" heatmap is for. */

const FLOW = {
  cell: 13,       // px per sampled cell (natural size; SVG scales down to fit)
  maxR: 14,       // sampled rows / cols per tile
  maxC: 22,
  labelH: 18,
  dimH: 15,
  gap: 14,
  opW: 34,
  arrowW: 74,
  pad: 8,
};

function sampleGrid(data, maxR, maxC) {
  const R = data.length, C = data[0].length;
  const rr = Math.min(R, maxR), cc = Math.min(C, maxC);
  const g = [];
  for (let i = 0; i < rr; i++) {
    const ri = Math.floor((i * R) / rr);
    const row = [];
    for (let j = 0; j < cc; j++) row.push(data[ri][Math.floor((j * C) / cc)]);
    g.push(row);
  }
  return { g, R, C, clippedR: R > rr, clippedC: C > cc };
}

// measure a tile → { w, gridW, gridH, draw(x, topOfGrid) -> svg }
function measureTile(t) {
  const cs = FLOW.cell;

  if (t.kind === "char") {
    const w = 64, gridH = 64;
    return { w, gridW: w, gridH, kind: "char", t };
  }
  if (t.kind === "text" || t.kind === "ids") {
    const items = t.kind === "text" ? t.chars : t.ids;
    const n = Math.min(items.length, FLOW.maxC);
    const bw = 22, gridH = 26, gridW = n * (bw + 2);
    return { w: gridW, gridW, gridH, kind: t.kind, items, n, bw, more: items.length > n, t };
  }

  // grid-backed tile (stream / out / weight / vector)
  let R, C, grid = null, seq = t.seq;
  if (t.data) {
    const s = sampleGrid(t.data, FLOW.maxR, FLOW.maxC);
    grid = s.g; R = s.g.length; C = s.g[0].length;
    t._full = t.data;
  } else {
    R = Math.min(t.rows, FLOW.maxR);
    C = Math.min(t.cols, FLOW.maxC);
  }
  const gridW = C * cs, gridH = R * cs;
  return { w: Math.max(gridW, 40), gridW, gridH, kind: "grid", grid, R, C, seq, t };
}

function tileSVG(m, x, bandTop, bandH) {
  const { gridW, gridH } = m;
  const gx = x + (m.w - gridW) / 2;
  const gy = bandTop + (bandH - gridH) / 2;
  const t = m.t;
  let s = "";

  // name label above the band
  s += `<text x="${x + m.w / 2}" y="${bandTop - 6}" class="fl-name">${escapeHtml(t.label || "")}</text>`;

  if (m.kind === "char") {
    s += `<rect x="${gx}" y="${gy}" width="${gridW}" height="${gridH}" class="fl-charbox"/>`;
    s += `<text x="${x + m.w / 2}" y="${gy + gridH / 2}" class="fl-char">${escapeHtml(t.char)}</text>`;
    return s;
  }
  if (m.kind === "text" || m.kind === "ids") {
    for (let i = 0; i < m.n; i++) {
      const bx = gx + i * (m.bw + 2);
      const val = m.kind === "text" ? m.items[i] : String(m.items[i]);
      s += `<rect x="${bx}" y="${gy}" width="${m.bw}" height="${m.gridH}" class="fl-idbox"/>`;
      s += `<text x="${bx + m.bw / 2}" y="${gy + m.gridH / 2}" class="fl-idtext">${escapeHtml(val)}</text>`;
    }
    if (m.more) s += `<text x="${gx + m.n * (m.bw + 2) + 4}" y="${gy + m.gridH / 2}" class="fl-more">…</text>`;
    return s;
  }

  // grid tile
  const cs = FLOW.cell;
  if (m.grid) {
    const mx = maxAbs(m._full || m.grid);
    for (let r = 0; r < m.R; r++)
      for (let c = 0; c < m.C; c++) {
        const v = m.grid[r][c];
        const fill = m.seq ? colorSeq(v) : colorDiverging(v, mx);
        s += `<rect x="${gx + c * cs}" y="${gy + r * cs}" width="${cs - 1}" height="${cs - 1}" fill="${fill}"/>`;
      }
  } else {
    // weight: schematic grid
    s += `<rect x="${gx}" y="${gy}" width="${gridW}" height="${gridH}" class="fl-weight"/>`;
    for (let c = 1; c < m.C; c++)
      s += `<line x1="${gx + c * cs}" y1="${gy}" x2="${gx + c * cs}" y2="${gy + gridH}" class="fl-wline"/>`;
    for (let r = 1; r < m.R; r++)
      s += `<line x1="${gx}" y1="${gy + r * cs}" x2="${gx + gridW}" y2="${gy + r * cs}" class="fl-wline"/>`;
    if (t.hlRows && t.hlRows.length) {
      const bands = [...new Set(t.hlRows.map((i) => Math.min(m.R - 1, Math.floor((i % t.rows) * m.R / t.rows))))].slice(0, 6);
      for (const br of bands)
        s += `<rect x="${gx}" y="${gy + br * cs}" width="${gridW}" height="${cs}" class="fl-hl"/>`;
    }
  }
  s += `<rect x="${gx}" y="${gy}" width="${gridW}" height="${gridH}" class="fl-frame ${t.kind === "out" ? "out" : ""}"/>`;

  // dims below the band
  const dims = m._full ? `${m.t.data.length}×${m.t.data[0].length}` : `${t.rows ?? m.R}×${t.cols ?? m.C}`;
  s += `<text x="${x + m.w / 2}" y="${bandTop + bandH + 12}" class="fl-dim">${dims}</text>`;
  return s;
}

function renderFlow(host, flow) {
  host.innerHTML = "";
  const tiles = flow.operands.map(measureTile);
  const outM = measureTile(flow.out);

  // build the left→right item list: tile, op, tile, ..., arrow, out
  const items = [];
  tiles.forEach((tm, i) => {
    items.push({ type: "tile", m: tm });
    if (i < flow.ops.length) items.push({ type: "op", sym: flow.ops[i] });
  });
  items.push({ type: "arrow", label: flow.arrow || "" });
  items.push({ type: "tile", m: outM });

  const bandH = Math.max(...tiles.map((t) => t.gridH), outM.gridH, 30);
  const bandTop = FLOW.labelH;
  const totalH = bandTop + bandH + FLOW.dimH + FLOW.pad;

  let x = FLOW.pad;
  let svg = "";
  const centerY = bandTop + bandH / 2;

  for (const it of items) {
    if (it.type === "tile") {
      svg += tileSVG(it.m, x, bandTop, bandH);
      x += it.m.w + FLOW.gap;
    } else if (it.type === "op") {
      svg += `<text x="${x + FLOW.opW / 2}" y="${centerY}" class="fl-op">${escapeHtml(it.sym)}</text>`;
      x += FLOW.opW + FLOW.gap;
    } else {
      const x0 = x, x1 = x + FLOW.arrowW;
      svg += `<line x1="${x0}" y1="${centerY}" x2="${x1 - 8}" y2="${centerY}" class="fl-arrow"/>`;
      svg += `<path d="M ${x1 - 8} ${centerY - 5} L ${x1} ${centerY} L ${x1 - 8} ${centerY + 5} Z" class="fl-arrowhead"/>`;
      if (it.label)
        svg += `<text x="${(x0 + x1) / 2}" y="${centerY - 12}" class="fl-arrowlabel">${escapeHtml(it.label)}</text>`;
      x += FLOW.arrowW + FLOW.gap;
    }
  }

  const totalW = x - FLOW.gap + FLOW.pad;
  host.innerHTML =
    `<svg class="flow-svg" viewBox="0 0 ${totalW} ${totalH}" width="${totalW}" height="${totalH}" ` +
    `preserveAspectRatio="xMidYMid meet">${svg}</svg>`;
}
