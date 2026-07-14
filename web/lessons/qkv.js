/* lesson-qkv.js — the fifth operation: Query, Key, Value projections
   for the first self-attention head.
*/

function renderQKVLesson(host, ctx) {
  const { tokens, chars, weights } = ctx;
  const Wt = weights["token_embedding_table.weight"]; // 75×64
  const Wp = weights["position_embedding_table.weight"]; // 64×64
  const ln1_g = weights["blocks.0.ln1.weight"]; // 64
  const ln1_b = weights["blocks.0.ln1.bias"]; // 64
  
  // Projection matrices for head 0 (16x64)
  const Wq = weights["blocks.0.sa.heads.0.query.weight"]; 
  const Wk = weights["blocks.0.sa.heads.0.key.weight"];
  const Wv = weights["blocks.0.sa.heads.0.value.weight"];

  const COLS = Wt[0].length; // 64
  const HEAD_SIZE = Wq.length; // 16
  const T = tokens.length;
  const EPS = 1e-5;

  const Etok = (i) => Wt[tokens[i]];
  const Epos = (i) => Wp[i];
  const H0 = (i) => Etok(i).map((v, d) => v + Epos(i)[d]);

  // Recalculate LayerNorm
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
    const out = norm.map((v, c) => ln1_g[c] * v + ln1_b[c]);

    return out;
  }

  // Calculate Q, K, V
  function calcQKV(i) {
    const h_tilde = calcLN(i);
    const q = [];
    const k = [];
    const v = [];
    for (let row = 0; row < HEAD_SIZE; row++) {
      let q_sum = 0, k_sum = 0, v_sum = 0;
      for (let col = 0; col < COLS; col++) {
        q_sum += h_tilde[col] * Wq[row][col];
        k_sum += h_tilde[col] * Wk[row][col];
        v_sum += h_tilde[col] * Wv[row][col];
      }
      q.push(q_sum);
      k.push(k_sum);
      v.push(v_sum);
    }
    return { h_tilde, q, k, v };
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
      section("1", "the input — h̃ₜ") +
      inputView(active) +
      section("2", "the projection matrices") +
      matrixView() +
      section("3", "the projected vectors — q, k, v") +
      projectionView(active);
    host.querySelectorAll(".lz-chip").forEach((c) =>
      c.addEventListener("click", () => { active = +c.dataset.i; draw(); })
    );
  }

  function chips() {
    let s =
      '<div class="lz-chips"><span class="lz-chips-label">your sequence — click a position to inspect its projections</span>' +
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
    const { h_tilde } = calcQKV(t);
    const mx = maxAbs([h_tilde]);
    
    const cw = 11, padL = 100, rowH = 26, padT = 16;
    const gridW = COLS * cw;
    const w = padL + gridW + 40;
    const h = padT + rowH + 20;
    let s = "";
    s += txt(padL - 12, padT + rowH / 2 + 3, "h̃ₜ", "lz-slab on", "end");
    s += txt(padL - 12, padT + rowH / 2 + 16, "1×64", "lz-subslab", "end");
    for (let c = 0; c < COLS; c++) {
      s += `<rect x="${padL + c * cw}" y="${padT}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(h_tilde[c], mx)}" data-tip="dim ${c} = ${h_tilde[c].toFixed(4)}"/>`;
    }
    s += `<rect x="${padL}" y="${padT}" width="${gridW}" height="${rowH}" class="lz-outframe"/>`;

    return note("This is the normalized residual stream from the previous step. It's the starting point for this token's journey through the attention head.") +
      svg(w, h, s, "");
  }

  function matrixView() {
    const tCol = 4, rowH = 4;
    const padL = 30, padT = 20, gap = 40;
    const matW = COLS * tCol;
    const matH = HEAD_SIZE * rowH;
    
    const w = padL + 3 * matW + 2 * gap + padL;
    const h = padT + matH + 40;
    
    const qMax = maxAbs(Wq);
    const kMax = maxAbs(Wk);
    const vMax = maxAbs(Wv);

    let s = "";
    
    function drawMat(data, x, mx, label) {
        let res = txt(x + matW / 2, padT - 8, label, "lz-cap");
        for (let r = 0; r < HEAD_SIZE; r++) {
            for (let c = 0; c < COLS; c++) {
                res += `<rect x="${x + c * tCol}" y="${padT + r * rowH}" width="${tCol - 0.4}" height="${rowH - 0.4}" fill="${colorDiverging(data[r][c], mx)}" data-tip="${label}[${r}, ${c}] = ${data[r][c].toFixed(4)}"/>`;
            }
        }
        res += `<rect x="${x}" y="${padT}" width="${matW}" height="${matH}" class="lz-heatframe"/>`;
        res += txt(x + matW / 2, padT + matH + 16, "16 × 64", "lz-cap dim");
        return res;
    }

    s += drawMat(Wq, padL, qMax, "W_Q");
    s += drawMat(Wk, padL + matW + gap, kMax, "W_K");
    s += drawMat(Wv, padL + 2 * (matW + gap), vMax, "W_V");

    return note(`Here are the three learned matrices for <b>Head 0</b>. Notice they are wide (64 columns) but short (16 rows). Multiplying our 64-dimensional input by these will project it down into a smaller 16-dimensional "head" subspace. The other 3 heads in this layer have their own matrices just like these.`) +
      svg(w, h, s, "Projection matrices");
  }

  function projectionView(t) {
    const { h_tilde, q, k, v } = calcQKV(t);
    const qMax = maxAbs([q]);
    const kMax = maxAbs([k]);
    const vMax = maxAbs([v]);
    
    const cw = 18, padL = 100, rowH = 26, gapR = 30, padT = 20;
    const gridW = HEAD_SIZE * cw;
    const w = padL + gridW + 40;
    const h3 = padT + 3 * rowH + 2 * gapR + 20;

    const rows = [
      { y: padT, vec: q, lab: "qₜ", name: "query", mx: qMax },
      { y: padT + rowH + gapR, vec: k, lab: "kₜ", name: "key", mx: kMax },
      { y: padT + 2 * (rowH + gapR), vec: v, lab: "vₜ", name: "value", mx: vMax },
    ];

    let s = "";
    rows.forEach((r) => {
      s += txt(padL - 12, r.y + rowH / 2 + 3, r.lab, "lz-slab on", "end");
      s += txt(padL - 12, r.y + rowH / 2 + 16, r.name, "lz-subslab", "end");
      for (let c = 0; c < HEAD_SIZE; c++) {
        s += `<rect x="${padL + c * cw}" y="${r.y}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(r.vec[c], r.mx)}" ` +
          `data-tip="dim ${c}: ${r.lab} = ${r.vec[c].toFixed(4)}"/>`;
      }
      s += `<rect x="${padL}" y="${r.y}" width="${gridW}" height="${rowH}" class="lz-outframe"/>`;
    });

    return note(`The matrix multiplication is a dot product between the input vector and each row of the matrix. We do this three times to create three new 16-dimensional vectors for this position:<br><br>
<b>Query (q)</b>: What this position is looking for in other tokens.<br>
<b>Key (k)</b>: What this position contains (its identity to others).<br>
<b>Value (v)</b>: The actual information this position will share if it gets attended to.`) +
      svg(w, h3, s, "q = h̃ · W_Qᵀ,  k = h̃ · W_Kᵀ,  v = h̃ · W_Vᵀ");
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
