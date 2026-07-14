/* lesson-resid1.js — the eleventh operation: Residual Add 1. */

function renderResid1Lesson(host, ctx) {
  const { tokens, chars, weights } = ctx;
  const Wt = weights["token_embedding_table.weight"];
  const Wp = weights["position_embedding_table.weight"];
  
  const COLS = Wt[0].length; // 64
  const N_HEADS = 4;
  const HEAD_SIZE = 16;
  const T = tokens.length;
  const EPS = 1e-5;

  // 1. Calculate H0 (Original Residual Stream)
  function calcH0() {
    const H = [];
    for (let i = 0; i < T; i++) {
      const etok = Wt[tokens[i]];
      const epos = Wp[i];
      H.push(etok.map((v, d) => v + epos[d]));
    }
    return H;
  }
  const H0 = calcH0();

  // 2. Calculate LN1
  function calcLN1() {
    const ln1_g = weights["blocks.0.ln1.weight"];
    const ln1_b = weights["blocks.0.ln1.bias"];
    const H_tilde = [];
    for (let i = 0; i < T; i++) {
      const h = H0[i];
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
  const H_tilde = calcLN1();

  // 3. Calculate full SA Block Output
  function calcSAOutput() {
    const O_concat = [];
    for (let i = 0; i < T; i++) O_concat.push(new Array(COLS).fill(0));

    for (let h_idx = 0; h_idx < N_HEADS; h_idx++) {
      const Wq = weights[`blocks.0.sa.heads.${h_idx}.query.weight`]; 
      const Wk = weights[`blocks.0.sa.heads.${h_idx}.key.weight`];
      const Wv = weights[`blocks.0.sa.heads.${h_idx}.value.weight`];
      
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
        let expSum = 0, expRow = [];
        for (let j = 0; j < T; j++) {
          const val = s_row[j] === -Infinity ? 0 : Math.exp(s_row[j] - maxS);
          expRow.push(val);
          expSum += val;
        }
        const a_row = [];
        for (let j = 0; j < T; j++) a_row.push(expSum === 0 ? 0 : expRow[j] / expSum);
        A.push(a_row);
      }
      
      for (let i = 0; i < T; i++) {
        for (let d = 0; d < HEAD_SIZE; d++) {
          let sum = 0;
          for (let j = 0; j < T; j++) sum += A[i][j] * V[j][d];
          O_concat[i][h_idx * HEAD_SIZE + d] = sum;
        }
      }
    }

    const W_proj = weights["blocks.0.sa.proj.weight"];
    const b_proj = weights["blocks.0.sa.proj.bias"];
    const O_proj = [];
    for (let i = 0; i < T; i++) {
      const row = [];
      for (let c = 0; c < COLS; c++) {
        let sum = b_proj[c];
        for (let d = 0; d < COLS; d++) sum += O_concat[i][d] * W_proj[c][d];
        row.push(sum);
      }
      O_proj.push(row);
    }
    return O_proj;
  }
  const O_proj = calcSAOutput();

  // 4. Calculate Residual Add
  const H1 = [];
  for (let i = 0; i < T; i++) {
    const row = [];
    for (let c = 0; c < COLS; c++) {
      row.push(H0[i][c] + O_proj[i][c]);
    }
    H1.push(row);
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
      section("1", "input: the original residual stream") +
      matrixView(H0, "h^0", "lz-cap") +
      section("2", "delta: self-attention output") +
      matrixView(O_proj, "O_proj", "lz-cap dim") +
      section("3", "output: the updated residual stream") +
      matrixView(H1, "h' = h^0 + O_proj", "lz-cap on");
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

  function matrixView(mat, title, capClass) {
    const cw = 11, rowH = 26;
    const padL = 40, padT = 30;
    const gridW = COLS * cw;
    const w = padL + gridW + 40;
    const h = padT + rowH + 30;

    let s = "";
    s += txt(padL + gridW / 2, 12, title, capClass);
    
    let mx = 1;
    mat[active].forEach(val => { if (Math.abs(val) > mx) mx = Math.abs(val); });
    
    for (let c = 0; c < COLS; c++) {
      s += `<rect x="${padL + c * cw}" y="${padT}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(mat[active][c], mx)}" data-tip="${title}[${c}] = ${mat[active][c].toFixed(4)}"/>`;
    }
    
    const frameClass = capClass.includes("on") ? "lz-outframe on" : "lz-outframe";
    s += `<rect x="${padL}" y="${padT}" width="${gridW}" height="${rowH}" class="${frameClass}" pointer-events="none"/>`;

    let noteText = "";
    if (title === "h^0") {
      noteText = "This is our original token embedding, before any Self-Attention happened. It represents the state of the Residual Stream right as it entered Block 0.";
    } else if (title === "O_proj") {
      noteText = "This is the final output of the Self-Attention block. We think of this as a \"delta\"—the update that the attention mechanism wants to apply to the token's understanding.";
    } else {
      noteText = "By simply adding the delta to the original stream, we produce the updated Residual Stream. This is called a \"Skip Connection\". It acts as an information highway, allowing the original data (and gradients during training) to flow unimpeded all the way through the network!";
    }

    return note(noteText) + svg(w, h, s, "");
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
