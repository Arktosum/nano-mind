/* lesson-resid1.js — the eleventh operation: Residual Add 1. */

export function renderResid1Lesson(host, ctx) {
  const { tokens, chars } = ctx;
  const COLS = 64;
  const T = ctx.fwd.T;
  const b0 = ctx.fwd.blocks[0];
  const H0 = ctx.fwd.h0;      // original residual stream
  const O_proj = b0.attnProj; // self-attention output (the delta)
  const H1 = b0.h1;           // H0 + O_proj

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
