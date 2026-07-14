/* lesson-proj.js — the tenth operation: Output Projection. */

function renderProjLesson(host, ctx) {
  const { tokens, chars, weights } = ctx;
  const Wt = weights["token_embedding_table.weight"];
  const Wp = weights["position_embedding_table.weight"];
  const ln1_g = weights["blocks.0.ln1.weight"];
  const ln1_b = weights["blocks.0.ln1.bias"];
  
  const COLS = Wt[0].length; // 64
  const N_HEADS = 4;
  const HEAD_SIZE = 16;
  const T = tokens.length;
  const EPS = 1e-5;

  const W_proj = weights["blocks.0.sa.proj.weight"];
  const b_proj = weights["blocks.0.sa.proj.bias"];

  // Calculate LN once for all tokens
  function calcLN() {
    const H_tilde = [];
    for (let i = 0; i < T; i++) {
      const etok = Wt[tokens[i]];
      const epos = Wp[i];
      const h = etok.map((v, d) => v + epos[d]);
      
      let sum = 0;
      for (let c = 0; c < COLS; c++) sum += h[c];
      const mean = sum / COLS;
      let varSum = 0;
      for (let c = 0; c < COLS; c++) {
        const diff = h[c] - mean;
        varSum += diff * diff;
      }
      const std = Math.sqrt(varSum / COLS + EPS);
      H_tilde.push(h.map(v => (v - mean) / std).map((v, c) => ln1_g[c] * v + ln1_b[c]));
    }
    return H_tilde;
  }
  const H_tilde = calcLN();

  // Calculate Concatenated Output
  function calcConcatOutput() {
    const O_concat = [];
    for (let i = 0; i < T; i++) O_concat.push(new Array(COLS).fill(0));

    for (let h_idx = 0; h_idx < N_HEADS; h_idx++) {
      const Wq = weights[`blocks.0.sa.heads.${h_idx}.query.weight`]; 
      const Wk = weights[`blocks.0.sa.heads.${h_idx}.key.weight`];
      const Wv = weights[`blocks.0.sa.heads.${h_idx}.value.weight`];
      
      // Q, K, V
      const Q = [], K = [], V = [];
      for (let i = 0; i < T; i++) {
        const q = [], k = [], v = [];
        for (let row = 0; row < HEAD_SIZE; row++) {
          let qs = 0, ks = 0, vs = 0;
          for (let col = 0; col < COLS; col++) {
            qs += H_tilde[i][col] * Wq[row][col];
            ks += H_tilde[i][col] * Wk[row][col];
            vs += H_tilde[i][col] * Wv[row][col];
          }
          q.push(qs); k.push(ks); v.push(vs);
        }
        Q.push(q); K.push(k); V.push(v);
      }
      
      // Scores & Softmax
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
          const val = s_row[j] === -Infinity ? 0 : Math.exp(s_row[j] - maxS);
          expRow.push(val);
          expSum += val;
        }
        const a_row = [];
        for (let j = 0; j < T; j++) a_row.push(expSum === 0 ? 0 : expRow[j] / expSum);
        A.push(a_row);
      }
      
      // Output for this head -> place into concat
      for (let i = 0; i < T; i++) {
        for (let d = 0; d < HEAD_SIZE; d++) {
          let sum = 0;
          for (let j = 0; j < T; j++) sum += A[i][j] * V[j][d];
          O_concat[i][h_idx * HEAD_SIZE + d] = sum;
        }
      }
    }
    return O_concat;
  }
  const O_concat = calcConcatOutput();

  // Compute final Projected Output
  const O_proj = [];
  for (let i = 0; i < T; i++) {
    const row = [];
    for (let c = 0; c < COLS; c++) {
      let sum = b_proj[c];
      // W_proj is 64x64, typically PyTorch Linear is Wx + b where W is out_features x in_features
      // Wait, W_proj[row] corresponds to out_feature row.
      for (let d = 0; d < COLS; d++) {
        sum += O_concat[i][d] * W_proj[c][d];
      }
      row.push(sum);
    }
    O_proj.push(row);
  }

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
      section("1", "input: the concatenated output") +
      inputView() +
      section("2", "weights: projection matrix") +
      weightsView() +
      section("3", "output: projected self-attention") +
      outputView();
  }

  function chips() {
    let s =
      '<div class="lz-chips"><span class="lz-chips-label">select a token</span>' +
      '<div class="lz-chip-row">';
    tokens.forEach((t, i) => {
      s +=
        `<button class="lz-chip${i === active ? " on" : ""}" data-i="${i}" title="position ${i}">` +
        `<span class="lz-chip-g">${escapeHtml(glyph(chars[i]))}</span>` +
        `<span class="lz-chip-id">pos ${i}</span></button>`;
    });
    return s + "</div></div>";
  }

  function inputView() {
    const cw = 11, rowH = 26;
    const padL = 40, padT = 30;
    const gridW = COLS * cw;
    const w = padL + gridW + 40;
    const h = padT + rowH + 30;

    let s = "";
    s += txt(padL + gridW / 2, 12, "O_concat (1×64)", "lz-cap");
    
    let mx = 1;
    O_concat[active].forEach(val => { if (Math.abs(val) > mx) mx = Math.abs(val); });
    
    for (let c = 0; c < COLS; c++) {
      s += `<rect x="${padL + c * cw}" y="${padT}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(O_concat[active][c], mx)}" data-tip="O_concat[${c}] = ${O_concat[active][c].toFixed(4)}"/>`;
    }
    s += `<rect x="${padL}" y="${padT}" width="${gridW}" height="${rowH}" class="lz-outframe on" pointer-events="none"/>`;

    return note(`Here is the 64-dimensional concatenated vector from the previous step.`) +
      svg(w, h, s, "");
  }

  function weightsView() {
    const cw = 6;
    const padL = 40, padT = 30;
    const gridW = COLS * cw;
    const w = padL + gridW + 40;
    const h = padT + COLS * cw + 40;

    let s = "";
    s += txt(padL + gridW / 2, 12, "W_proj (64×64)", "lz-cap");
    
    let mx = 1;
    W_proj.forEach(row => row.forEach(val => { if (Math.abs(val) > mx) mx = Math.abs(val); }));

    // For performance, we'll draw this as a single image or dense SVG
    for (let r = 0; r < COLS; r++) {
      for (let c = 0; c < COLS; c++) {
        s += `<rect x="${padL + c * cw}" y="${padT + r * cw}" width="${cw}" height="${cw}" fill="${colorDiverging(W_proj[r][c], mx)}" data-tip="W_proj[${r}, ${c}] = ${W_proj[r][c].toFixed(4)}"/>`;
      }
    }
    s += `<rect x="${padL}" y="${padT}" width="${gridW}" height="${gridW}" class="lz-outframe" pointer-events="none"/>`;

    return note(`This is the final Linear Projection matrix for the Self-Attention block. It is a 64×64 square matrix. It takes our 64-dimensional concatenated vector, mixes all the information from all 4 heads together, and produces a new, perfectly blended 64-dimensional output.`) +
      svg(w, h, s, "");
  }

  function outputView() {
    const cw = 11, rowH = 26;
    const padL = 40, padT = 30;
    const gridW = COLS * cw;
    const w = padL + gridW + 40;
    const h = padT + rowH + 30;

    let s = "";
    s += txt(padL + gridW / 2, 12, "O_proj = (O_concat · W_proj^T) + b_proj", "lz-cap");
    
    let mx = 1;
    O_proj[active].forEach(val => { if (Math.abs(val) > mx) mx = Math.abs(val); });
    
    for (let c = 0; c < COLS; c++) {
      s += `<rect x="${padL + c * cw}" y="${padT}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(O_proj[active][c], mx)}" data-tip="O_proj[${c}] = ${O_proj[active][c].toFixed(4)}"/>`;
    }
    s += `<rect x="${padL}" y="${padT}" width="${gridW}" height="${rowH}" class="lz-outframe on" pointer-events="none"/>`;

    return note(`This is the final result of the entire Self-Attention mechanism for this token! It is a 64-dimensional vector that represents everything the token learned by attending to its past. This vector is now ready to be added back into the residual stream.`) +
      svg(w, h, s, "");
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
