/* lesson-layernorm.js — the fourth operation: Layer Normalization (LayerNorm₁)
   that prepares the residual stream for the self-attention mechanism.
*/

function renderLayerNormLesson(host, ctx) {
  const { tokens, chars, weights } = ctx;
  const Wt = weights["token_embedding_table.weight"]; // 75×64
  const Wp = weights["position_embedding_table.weight"]; // 64×64
  const gamma = weights["blocks.0.ln1.weight"]; // 64
  const beta = weights["blocks.0.ln1.bias"]; // 64
  const COLS = Wt[0].length; // 64
  const T = tokens.length;
  const EPS = 1e-5;

  const Etok = (i) => Wt[tokens[i]];
  const Epos = (i) => Wp[i];
  const H0 = (i) => Etok(i).map((v, d) => v + Epos(i)[d]);

  // LayerNorm calculation for a given position
  function calcLN(i) {
    const h = H0(i);
    let sum = 0;
    for (let c = 0; c < COLS; c++) sum += h[c];
    const mean = sum / COLS;

    let varSum = 0;
    for (let c = 0; c < COLS; c++) {
      const diff = h[c] - mean;
      varSum += diff * diff;
    }
    const variance = varSum / COLS;
    const std = Math.sqrt(variance + EPS);

    const norm = h.map(v => (v - mean) / std);
    const out = norm.map((v, c) => gamma[c] * v + beta[c]);

    return { h, mean, variance, std, norm, out };
  }

  let active = T - 1;

  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
  });
  host.addEventListener("mouseleave", hideTip);
  host.addEventListener("click", (e) => {
    const p = e.target.dataset && e.target.dataset.pick;
    if (p) { active = +p; draw(); }
  });

  function draw() {
    hideTip();
    host.innerHTML =
      chips() +
      section("1", "the input — h⁰ₜ") +
      inputView(active) +
      section("2", "mean & variance") +
      statsView(active) +
      section("3", "normalize") +
      normView(active) +
      section("4", "scale & shift") +
      affineView(active);
    host.querySelectorAll(".lz-chip").forEach((c) =>
      c.addEventListener("click", () => { active = +c.dataset.i; draw(); })
    );
  }

  function chips() {
    let s =
      '<div class="lz-chips"><span class="lz-chips-label">your sequence — click a position to inspect its LayerNorm</span>' +
      '<div class="lz-chip-row">';
    tokens.forEach((t, i) => {
      s +=
        `<button class="lz-chip${i === active ? " on" : ""}" data-i="${i}" title="position ${i}">` +
        `<span class="lz-chip-g">${escapeHtml(glyph(chars[i]))}</span>` +
        `<span class="lz-chip-id">pos ${i}</span></button>`;
    });
    return s + "</div></div>";
  }

  function inputView(t) {
    const ln = calcLN(t);
    const mx = maxAbs([ln.h]);
    return note("LayerNorm operates on each position <b>independently</b>. It doesn't look across tokens; it looks across the 64 dimensions of a single token's vector.") +
      drawVec("h⁰ₜ", "residual stream", ln.h, mx, "dim");
  }

  function statsView(t) {
    const ln = calcLN(t);
    return note(`First, we find the center and the spread of this specific 64-dimensional vector.<br><br>` +
      `<b>Mean (μ)</b> = sum of all dims / 64 = <b>${ln.mean.toFixed(4)}</b><br>` +
      `<b>Variance (σ²)</b> = average squared distance from mean = <b>${ln.variance.toFixed(4)}</b>`);
  }

  function normView(t) {
    const ln = calcLN(t);
    const mx = maxAbs([ln.norm]);
    return note("We subtract the mean and divide by the standard deviation (√σ²). Now the vector is centered exactly on 0, and its values generally fall between -2 and 2. This prevents extreme values from blowing up the network's calculations.") +
      drawVec("(h - μ) / σ", "normalized", ln.norm, mx, "dim");
  }

  function affineView(t) {
    const ln = calcLN(t);
    const mxOut = maxAbs([ln.out]);
    const mxG = maxAbs([gamma]);
    const mxB = maxAbs([beta]);

    const cw = 11, padL = 100, rowH = 26, gapR = 30, padT = 16;
    const gridW = COLS * cw;
    const w = padL + gridW + 40;
    const h3 = padT + 3 * rowH + 2 * gapR + 20;

    const rows = [
      { y: padT, vec: gamma, lab: "γ", name: "learned weight (64)", mx: mxG, op: "×" },
      { y: padT + rowH + gapR, vec: beta, lab: "β", name: "learned bias (64)", mx: mxB, op: "+" },
      { y: padT + 2 * (rowH + gapR), vec: ln.out, lab: "h̃ₜ", name: "scaled & shifted", mx: mxOut, out: true },
    ];

    let s = "";
    rows.forEach((r, ri) => {
      s += txt(padL - 12, r.y + rowH / 2 + 3, r.lab, r.out ? "lz-slab on" : "lz-slab", "end");
      s += txt(padL - 12, r.y + rowH / 2 + 16, r.name, "lz-subslab", "end");
      for (let c = 0; c < COLS; c++) {
        let tip = `dim ${c}: ${r.lab} = ${r.vec[c].toFixed(4)}`;
        if (r.out) {
            tip = `dim ${c}: ${gamma[c].toFixed(3)} × ${ln.norm[c].toFixed(3)} + ${beta[c].toFixed(3)} = ${r.vec[c].toFixed(3)}`;
        }
        s += `<rect x="${padL + c * cw}" y="${r.y}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(r.vec[c], r.mx)}" ` +
          `data-tip="${tip}"/>`;
      }
      if (r.out) s += `<rect x="${padL}" y="${r.y}" width="${gridW}" height="${rowH}" class="lz-outframe"/>`;
      
      // operator
      if (r.op) {
          s += txt(padL + gridW + 18, r.y + rowH + gapR / 2 + 4, r.op, "lz-op");
      }
    });

    return note("Finally, the network is allowed to reshape this perfect distribution if it needs to. It multiplies by a learned <b>weight (γ)</b> and adds a learned <b>bias (β)</b>. These parameters are unique to this layer, but shared across all positions.") +
      svg(w, h3, s, "h̃ₜ = γ ⊙ norm + β");
  }

  function drawVec(lab, sublab, vec, mx, tipPrefix) {
    const cw = 11, padL = 100, rowH = 26, padT = 16;
    const gridW = COLS * cw;
    const w = padL + gridW + 40;
    const h = padT + rowH + 20;
    let s = "";
    s += txt(padL - 12, padT + rowH / 2 + 3, lab, "lz-slab on", "end");
    s += txt(padL - 12, padT + rowH / 2 + 16, sublab, "lz-subslab", "end");
    for (let c = 0; c < COLS; c++) {
      s += `<rect x="${padL + c * cw}" y="${padT}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(vec[c], mx)}" data-tip="${tipPrefix} ${c} = ${vec[c].toFixed(4)}"/>`;
    }
    s += `<rect x="${padL}" y="${padT}" width="${gridW}" height="${rowH}" class="lz-outframe"/>`;
    return svg(w, h, s, "");
  }

  /* helpers */
  function section(n, title) {
    return `<div class="lz-section"><span class="lz-secnum">${n}</span>${escapeHtml(title)}</div>`;
  }
  function note(html) { return `<div class="lz-note">${html}</div>`; }
  function txt(x, y, str, cls, anchor = "middle") {
    return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${escapeHtml(str)}</text>`;
  }
  function svg(w, h, inner, cap) {
    return `<div class="lz-diagram">${cap ? `<div class="lz-diagram-cap">${escapeHtml(cap)}</div>` : ""}<div class="lz-svgwrap"><svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid meet">${inner}</svg></div></div>`;
  }

  draw();
}
