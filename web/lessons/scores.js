/* lesson-scores.js — the sixth operation: Attention Scores (Q·K^T), Scaling, and Masking. */

function renderScoresLesson(host, ctx) {
  const { chars } = ctx;
  const T = ctx.fwd.T;
  const head0 = ctx.fwd.blocks[0].heads[0];
  const S_raw = head0.scoresRaw;       // T×T raw dot products Q·Kᵀ
  const S_scaled = head0.scoresScaled; // ÷ √d_k
  const S_masked = head0.scoresMasked; // upper triangle = -Infinity

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
      section("1", "the dot product — Q · Kᵀ") +
      rawView() +
      section("2", "scaling") +
      scaleView() +
      section("3", "causal mask") +
      maskView();
  }

  function rawView() {
    return note(`We take the Query vector for every token and take its dot product with the Key vector of every other token. This produces a T×T matrix of raw scores. The higher the score, the more token <i>i</i>'s query aligns with token <i>j</i>'s key.`) +
      drawHeatmap("S_raw", S_raw, "raw scores", false);
  }

  function scaleView() {
    return note(`Notice how large the raw scores can get. When you take the dot product of two vectors, you are adding up 16 independent multiplications. If the original numbers had a variance of 1, adding 16 of them together produces a sum with a <b>variance of 16</b>. As dimensions increase, the scores become more extreme, which pushes the softmax function into flat regions and kills the gradients. To fix this and pull the variance back down to 1, we divide every score by √d<sub>k</sub> = √16 = 4.`) +
      drawHeatmap("S_scaled", S_scaled, "scaled scores", false);
  }

  function maskView() {
    return note(`Our model is autoregressive—it generates text left-to-right. Therefore, a token at position <i>i</i> is absolutely forbidden from attending to a token at position <i>j</i> if <i>j</i> > <i>i</i>. We enforce this by setting all upper-triangular scores to −∞.`) +
      drawHeatmap("S_masked", S_masked, "masked scores", true);
  }

  function drawHeatmap(id, data, subtitle, hasMask) {
    // Find absolute maximum for color scaling, ignoring -Infinity
    let mx = 0;
    for (let r = 0; r < T; r++) {
      for (let c = 0; c < T; c++) {
        if (data[r][c] !== -Infinity && Math.abs(data[r][c]) > mx) {
          mx = Math.abs(data[r][c]);
        }
      }
    }
    // Prevent division by zero
    if (mx === 0) mx = 1;

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
        
        if (val === -Infinity) {
          s += `<rect x="${x}" y="${y}" width="${cell - 0.5}" height="${cell - 0.5}" fill="#0b0b09" data-tip="Q[${r}] · K[${c}] = -∞ (masked)"/>`;
          // Draw a hatch mark or something to show it's masked
          s += `<line x1="${x}" y1="${y}" x2="${x + cell - 0.5}" y2="${y + cell - 0.5}" stroke="#1f1f1a" stroke-width="1"/>`;
        } else {
          s += `<rect x="${x}" y="${y}" width="${cell - 0.5}" height="${cell - 0.5}" fill="${colorDiverging(val, mx)}" data-tip="Q[${r}] · K[${c}] = ${val.toFixed(4)}"/>`;
        }
      }
    }
    s += `<rect x="${padL}" y="${padT}" width="${T * cell}" height="${T * cell}" class="lz-heatframe"/>`;
    s += txt(padL - 10, padT - 10, "Q \\ K", "lz-cap dim", "end");
    
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
