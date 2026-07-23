/* lesson-concat.js — the ninth operation: Concat heads. */

function renderConcatLesson(host, ctx) {
  const { tokens, chars } = ctx;
  const COLS = 64, N_HEADS = 4, HEAD_SIZE = 16;
  const T = ctx.fwd.T;
  // the 4 head outputs come from the shared forward pass
  const headOutputs = ctx.fwd.blocks[0].heads.map((h) => h.out); // 4 × (T×16)

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
      section("1", "the 4 parallel head outputs") +
      headsView() +
      section("2", "the concatenated output") +
      concatView();
  }

  function chips() {
    let s =
      '<div class="lz-chips"><span class="lz-chips-label">select a token to view its combined output</span>' +
      '<div class="lz-chip-row">';
    tokens.forEach((t, i) => {
      s +=
        `<button class="lz-chip${i === active ? " on" : ""}" data-i="${i}" title="position ${i}">` +
        `<span class="lz-chip-g">${escapeHtml(glyph(chars[i]))}</span>` +
        `<span class="lz-chip-id">pos ${i}</span></button>`;
    });
    return s + "</div></div>";
  }

  // Define a nice distinct color palette for the 4 heads to help visualize the concatenation
  const headColors = [
    { name: "head 0", stroke: "#e06c75" },
    { name: "head 1", stroke: "#98c379" },
    { name: "head 2", stroke: "#e5c07b" },
    { name: "head 3", stroke: "#61afef" }
  ];

  function headsView() {
    const cw = 14, rowH = 26, gapY = 20, padL = 80, padT = 30;
    const gridW = HEAD_SIZE * cw;
    const w = padL + gridW + 40;
    const h = padT + N_HEADS * rowH + (N_HEADS - 1) * gapY + 40;

    let s = "";
    
    // Find max for color scaling across all heads for consistency
    let mx = 1;
    headOutputs.forEach(O => {
      O.forEach(row => row.forEach(val => {
        if (Math.abs(val) > mx) mx = Math.abs(val);
      }));
    });

    for (let i = 0; i < N_HEADS; i++) {
      const y = padT + i * (rowH + gapY);
      s += txt(padL - 12, y + rowH / 2 + 3, `O_${i}[${active}]`, "lz-slab", "end");
      s += txt(padL - 12, y + rowH / 2 + 16, headColors[i].name, "lz-subslab", "end");
      
      const vec = headOutputs[i][active];
      for (let c = 0; c < HEAD_SIZE; c++) {
        s += `<rect x="${padL + c * cw}" y="${y}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(vec[c], mx)}" data-tip="Head ${i} [${c}] = ${vec[c].toFixed(4)}"/>`;
      }
      // Draw a colored border around each head to differentiate them
      s += `<rect x="${padL}" y="${y}" width="${gridW}" height="${rowH}" fill="none" stroke="${headColors[i].stroke}" stroke-width="1.5" stroke-dasharray="2 2" pointer-events="none"/>`;
    }

    return note(`While we were carefully tracing Head 0, Heads 1, 2, and 3 were simultaneously doing the exact same math with their own unique weight matrices. Each head is learning to look for different patterns. Here are the 4 independent 16-dimensional outputs they produced for the active token.`) +
      svg(w, h, s, "");
  }

  function concatView() {
    let mx = 1;
    const concatVec = [];
    headOutputs.forEach(O => {
      O[active].forEach(val => {
        concatVec.push(val);
        if (Math.abs(val) > mx) mx = Math.abs(val);
      });
    });

    const cw = 11, rowH = 26;
    const padL = 40, padT = 30;
    const gridW = COLS * cw;
    const w = padL + gridW + 40;
    const h = padT + rowH + 60;

    let s = "";
    s += txt(padL + gridW / 2, 12, "O = [ O_0 || O_1 || O_2 || O_3 ]", "lz-cap");
    
    for (let c = 0; c < COLS; c++) {
      const headIdx = Math.floor(c / HEAD_SIZE);
      const subIdx = c % HEAD_SIZE;
      s += `<rect x="${padL + c * cw}" y="${padT}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(concatVec[c], mx)}" data-tip="Concat[${c}]\n(Head ${headIdx}[${subIdx}]) = ${concatVec[c].toFixed(4)}"/>`;
    }

    // Draw the bounding boxes for each head's contribution
    for (let i = 0; i < N_HEADS; i++) {
      const x = padL + i * (HEAD_SIZE * cw);
      s += `<rect x="${x}" y="${padT}" width="${HEAD_SIZE * cw}" height="${rowH}" fill="none" stroke="${headColors[i].stroke}" stroke-width="2" pointer-events="none"/>`;
      s += txt(x + (HEAD_SIZE * cw) / 2, padT + rowH + 20, headColors[i].name, "lz-subslab", "middle");
    }

    return note(`To combine their insights, we literally just glue the four vectors together side-by-side. 4 heads × 16 dimensions = 64 dimensions! We have successfully rebuilt a full 64-dimensional representation of our token, enriched by the context from all 4 attention heads.`) +
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
