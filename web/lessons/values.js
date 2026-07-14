/* lesson-values.js — the eighth operation: Multiply Values (Attention Output). */

function renderValuesLesson(host, ctx) {
  const { tokens, chars, weights } = ctx;
  const Wt = weights["token_embedding_table.weight"];
  const Wp = weights["position_embedding_table.weight"];
  const ln1_g = weights["blocks.0.ln1.weight"];
  const ln1_b = weights["blocks.0.ln1.bias"];
  
  const Wq = weights["blocks.0.sa.heads.0.query.weight"]; 
  const Wk = weights["blocks.0.sa.heads.0.key.weight"];
  const Wv = weights["blocks.0.sa.heads.0.value.weight"];

  const COLS = Wt[0].length; // 64
  const HEAD_SIZE = Wq.length; // 16
  const T = tokens.length;
  const EPS = 1e-5;

  // Calculate H0
  const Etok = (i) => Wt[tokens[i]];
  const Epos = (i) => Wp[i];
  const H0 = (i) => Etok(i).map((v, d) => v + Epos(i)[d]);

  // Calculate LN
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
    return h.map(v => (v - mean) / std).map((v, c) => ln1_g[c] * v + ln1_b[c]);
  }

  // Precompute Q, K, V
  const Q = [];
  const K = [];
  const V = [];
  for (let i = 0; i < T; i++) {
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
    Q.push(q);
    K.push(k);
    V.push(v);
  }

  // Compute Masked Scores S and Softmax A
  const A = [];
  for (let i = 0; i < T; i++) {
    const s_row = [];
    let maxS = -Infinity;
    for (let j = 0; j < T; j++) {
      let dot = 0;
      for (let d = 0; d < HEAD_SIZE; d++) dot += Q[i][d] * K[j][d];
      const val = j > i ? -Infinity : dot / Math.sqrt(HEAD_SIZE);
      s_row.push(val);
      if (val > maxS) maxS = val;
    }
    
    let expSum = 0;
    const expRow = [];
    for (let j = 0; j < T; j++) {
      const expVal = s_row[j] === -Infinity ? 0 : Math.exp(s_row[j] - maxS);
      expRow.push(expVal);
      expSum += expVal;
    }
    
    const a_row = [];
    for (let j = 0; j < T; j++) {
      a_row.push(expSum === 0 ? 0 : expRow[j] / expSum);
    }
    A.push(a_row);
  }

  // Compute Output O = A * V
  const O = [];
  for (let i = 0; i < T; i++) {
    const o_row = [];
    for (let d = 0; d < HEAD_SIZE; d++) {
      let sum = 0;
      for (let j = 0; j < T; j++) {
        sum += A[i][j] * V[j][d];
      }
      o_row.push(sum);
    }
    O.push(o_row);
  }

  // Setup UI
  let active = T - 1;
  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
  });
  host.addEventListener("mouseleave", hideTip);
  host.addEventListener("click", (e) => {
    const p = e.target.closest(".lz-chip");
    if (p) { active = +p.dataset.i; draw(); }
  });

  function draw() {
    hideTip();
    host.innerHTML =
      chips() +
      section("1", "attention weights (A)") +
      weightsView() +
      section("2", "value vectors (V)") +
      valuesView() +
      section("3", "the weighted sum (O)") +
      outputView();
  }

  function chips() {
    let s =
      '<div class="lz-chips"><span class="lz-chips-label">click a token to see its specific weighted sum calculation</span>' +
      '<div class="lz-chip-row">';
    tokens.forEach((t, i) => {
      s +=
        `<button class="lz-chip${i === active ? " on" : ""}" data-i="${i}" title="position ${i}">` +
        `<span class="lz-chip-g">${escapeHtml(glyph(chars[i]))}</span>` +
        `<span class="lz-chip-id">pos ${i}</span></button>`;
    });
    return s + "</div></div>";
  }

  function weightsView() {
    const cell = Math.max(12, Math.min(30, Math.floor(600 / T)));
    const padL = 40, padT = 40;
    const w = padL + T * cell + 40;
    const h = padT + T * cell + 40;

    let s = "";
    s += txt(padL + (T * cell) / 2, 16, "A (T×T)", "lz-cap");
    
    for (let r = 0; r < T; r++) {
      const isActive = r === active;
      const opacity = isActive ? 1.0 : 0.3;
      s += `<g opacity="${opacity}">`;
      s += txt(padL - 10, padT + r * cell + cell / 2 + 3, glyph(chars[r]), "lz-slab", "end");
      for (let c = 0; c < T; c++) {
        if (r === 0) s += txt(padL + c * cell + cell / 2, padT - 10, glyph(chars[c]), "lz-slab", "middle");
        
        const val = A[r][c];
        const x = padL + c * cell;
        const y = padT + r * cell;
        
        if (c > r) {
          s += `<rect x="${x}" y="${y}" width="${cell - 0.5}" height="${cell - 0.5}" fill="#0b0b09" data-tip="Masked (0.0000)"/>`;
          s += `<line x1="${x}" y1="${y}" x2="${x + cell - 0.5}" y2="${y + cell - 0.5}" stroke="#1f1f1a" stroke-width="1"/>`;
        } else {
          s += `<rect x="${x}" y="${y}" width="${cell - 0.5}" height="${cell - 0.5}" fill="${colorSeq(val)}" data-tip="A[${r}, ${c}] = ${(val * 100).toFixed(1)}%"/>`;
        }
      }
      s += `</g>`;
      if (isActive) {
        s += `<rect x="${padL - 2}" y="${padT + r * cell - 2}" width="${T * cell + 4}" height="${cell + 4}" class="lz-heatframe on" fill="none" pointer-events="none"/>`;
      }
    }
    
    return note(`This is the Softmax probability matrix. The highlighted row represents the active token's "attention budget"—the percentages it has decided to assign to each previous token.`) +
      svg(w, h, s, "");
  }

  function valuesView() {
    const mx = maxAbs(V);
    const cw = 11, rowH = 14;
    const padL = 40, padT = 30;
    const gridW = HEAD_SIZE * cw;
    const w = padL + gridW + 40;
    const h = padT + T * rowH + 20;

    let s = "";
    s += txt(padL + gridW / 2, 12, "V (T×16)", "lz-cap");
    
    for (let r = 0; r < T; r++) {
      const weight = A[active][r];
      // Dim the row based on its weight to visually show the weighting!
      // But also make sure it's fully hidden if masked/zero.
      const opacity = weight === 0 ? 0.1 : 0.2 + 0.8 * (weight / Math.max(...A[active]));
      
      s += `<g opacity="${opacity}">`;
      s += txt(padL - 10, padT + r * rowH + rowH / 2 + 4, glyph(chars[r]), "lz-slab", "end");
      
      for (let c = 0; c < HEAD_SIZE; c++) {
        s += `<rect x="${padL + c * cw}" y="${padT + r * rowH}" width="${cw - 0.5}" height="${rowH - 0.5}" fill="${colorDiverging(V[r][c], mx)}" data-tip="V[${r}, ${c}] = ${V[r][c].toFixed(4)}\nWeight = ${(weight * 100).toFixed(1)}%"/>`;
      }
      s += `</g>`;
    }
    s += `<rect x="${padL}" y="${padT}" width="${gridW}" height="${T * rowH}" class="lz-outframe"/>`;
    s += txt(padL + gridW / 2, padT + T * rowH + 16, "16 dimensions", "lz-cap dim");

    return note(`Here is the V matrix (Values) we computed earlier. For the active token, we take a <b>weighted sum</b> of all these rows, using the percentages from the highlighted row above. Notice how the rows with higher attention weights appear brighter here!`) +
      svg(w, h, s, "");
  }

  function outputView() {
    const mx = maxAbs(O);
    const cw = 18, rowH = 26;
    const padL = 60, padT = 20;
    const gridW = HEAD_SIZE * cw;
    const w = padL + gridW + 40;
    const h = padT + rowH + 20;

    let s = "";
    s += txt(padL - 12, padT + rowH / 2 + 3, `O[${active}]`, "lz-slab on", "end");
    s += txt(padL - 12, padT + rowH / 2 + 16, "1×16", "lz-subslab", "end");
    
    for (let c = 0; c < HEAD_SIZE; c++) {
      s += `<rect x="${padL + c * cw}" y="${padT}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(O[active][c], mx)}" data-tip="O[${active}, ${c}] = ${O[active][c].toFixed(4)}"/>`;
    }
    s += `<rect x="${padL}" y="${padT}" width="${gridW}" height="${rowH}" class="lz-outframe on"/>`;

    return note(`This is the final output of Head 0 for the active token! It has successfully gathered 16-dimensional information from its neighbors (and itself) based on how much attention it decided to pay them.`) +
      svg(w, h, s, "O = A · V");
  }

  /* helpers */
  function maxAbs(mat) {
    let m = 0;
    mat.forEach(r => r.forEach(v => { if (Math.abs(v) > m) m = Math.abs(v); }));
    return m === 0 ? 1 : m;
  }
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
