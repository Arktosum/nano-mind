/* lesson-proj.js — the tenth operation: Output Projection. */

function renderProjLesson(host, ctx) {
  const { tokens, chars, weights } = ctx;
  const COLS = 64;
  const T = ctx.fwd.T;
  const W_proj = weights["blocks.0.sa.proj.weight"]; // 64×64, drawn in weightsView
  const b0 = ctx.fwd.blocks[0];
  const O_concat = b0.concat;   // T×64 concatenated heads
  const O_proj = b0.attnProj;   // T×64 projected attention output

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
