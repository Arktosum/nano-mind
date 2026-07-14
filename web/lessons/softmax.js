/* lesson-softmax.js — the seventh operation: Softmax (Attention Weights). */

function renderSoftmaxLesson(host, ctx) {
  const { tokens, chars, weights } = ctx;
  const Wt = weights["token_embedding_table.weight"];
  const Wp = weights["position_embedding_table.weight"];
  const ln1_g = weights["blocks.0.ln1.weight"];
  const ln1_b = weights["blocks.0.ln1.bias"];
  
  const Wq = weights["blocks.0.sa.heads.0.query.weight"]; 
  const Wk = weights["blocks.0.sa.heads.0.key.weight"];

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

  // Precompute Q and K
  const Q = [];
  const K = [];
  for (let i = 0; i < T; i++) {
    const h_tilde = calcLN(i);
    const q = [];
    const k = [];
    for (let row = 0; row < HEAD_SIZE; row++) {
      let q_sum = 0, k_sum = 0;
      for (let col = 0; col < COLS; col++) {
        q_sum += h_tilde[col] * Wq[row][col];
        k_sum += h_tilde[col] * Wk[row][col];
      }
      q.push(q_sum);
      k.push(k_sum);
    }
    Q.push(q);
    K.push(k);
  }

  // Compute Masked Scores S
  const S = [];
  for (let i = 0; i < T; i++) {
    const row = [];
    for (let j = 0; j < T; j++) {
      let dot = 0;
      for (let d = 0; d < HEAD_SIZE; d++) dot += Q[i][d] * K[j][d];
      row.push(j > i ? -Infinity : dot / Math.sqrt(HEAD_SIZE));
    }
    S.push(row);
  }

  // Compute Softmax A
  const A = [];
  for (let i = 0; i < T; i++) {
    const row = [];
    let maxS = -Infinity; // For numerical stability
    for (let j = 0; j <= i; j++) if (S[i][j] > maxS) maxS = S[i][j];
    
    let expSum = 0;
    const expRow = [];
    for (let j = 0; j < T; j++) {
      const val = S[i][j] === -Infinity ? 0 : Math.exp(S[i][j] - maxS);
      expRow.push(val);
      expSum += val;
    }
    
    for (let j = 0; j < T; j++) {
      row.push(expSum === 0 ? 0 : expRow[j] / expSum);
    }
    A.push(row);
  }

  // Setup UI
  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
  });
  host.addEventListener("mouseleave", hideTip);

  function draw() {
    hideTip();
    host.innerHTML =
      section("1", "the input — masked scores") +
      inputView() +
      section("2", "the softmax function") +
      softmaxView() +
      section("3", "the output — attention weights") +
      outputView();
  }

  function inputView() {
    return note(`Here is the masked score matrix from the previous step. The upper triangle is filled with −∞ to prevent attending to the future.`) +
      drawHeatmap("S", S, "masked scores S", true, false);
  }

  function softmaxView() {
    return note(`For each row, we exponentiate every score (e<sup>x</sup>) and then divide by the sum of those exponentials. This normalizes the scores so that every row adds up perfectly to 1.0. 
<br><br>Because e<sup>−∞</sup> = 0, the masked future tokens get exactly 0% of the attention. The token is forced to spend its entire 100% "attention budget" on itself and tokens that came before it.`);
  }

  function outputView() {
    return drawHeatmap("A", A, "attention weights A", true, true);
  }

  function drawHeatmap(id, data, subtitle, hasMask, isSeq) {
    let mx = 0;
    if (!isSeq) {
      for (let r = 0; r < T; r++) {
        for (let c = 0; c < T; c++) {
          if (data[r][c] !== -Infinity && Math.abs(data[r][c]) > mx) {
            mx = Math.abs(data[r][c]);
          }
        }
      }
      if (mx === 0) mx = 1;
    }

    const cell = Math.max(12, Math.min(30, Math.floor(600 / T)));
    const padL = 40, padT = 40;
    const w = padL + T * cell + 40;
    const h = padT + T * cell + 40;

    let s = "";
    s += txt(padL + (T * cell) / 2, 16, subtitle, "lz-cap");
    
    for (let r = 0; r < T; r++) {
      s += txt(padL - 10, padT + r * cell + cell / 2 + 3, glyph(chars[r]), "lz-slab", "end");
      for (let c = 0; c < T; c++) {
        s += txt(padL + c * cell + cell / 2, padT - 10, glyph(chars[c]), "lz-slab", "middle");
        
        const val = data[r][c];
        const x = padL + c * cell;
        const y = padT + r * cell;
        
        if (hasMask && c > r) {
          s += `<rect x="${x}" y="${y}" width="${cell - 0.5}" height="${cell - 0.5}" fill="#0b0b09" data-tip="Masked (0.0000)"/>`;
          s += `<line x1="${x}" y1="${y}" x2="${x + cell - 0.5}" y2="${y + cell - 0.5}" stroke="#1f1f1a" stroke-width="1"/>`;
        } else {
          const fill = isSeq ? colorSeq(val) : colorDiverging(val, mx);
          const formatted = isSeq ? (val * 100).toFixed(1) + "%" : val.toFixed(4);
          s += `<rect x="${x}" y="${y}" width="${cell - 0.5}" height="${cell - 0.5}" fill="${fill}" data-tip="${subtitle}[${r}, ${c}] = ${formatted}"/>`;
        }
      }
    }
    s += `<rect x="${padL}" y="${padT}" width="${T * cell}" height="${T * cell}" class="lz-heatframe"/>`;
    s += txt(padL - 10, padT - 10, isSeq ? "Query \\ Attends to" : "Q \\ K", "lz-cap dim", "end");
    
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
