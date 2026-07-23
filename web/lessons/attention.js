/* attention.js — the attention atlas: all 12 heads (3 layers × 4 heads)
   at once, for a chosen query token. Reveals that different heads learn
   to attend to different things. Reads ctx.fwd. */

function renderAttentionAtlasLesson(host, ctx) {
  const F = ctx.fwd;
  const T = F.T;
  const chars = ctx.chars;
  let query = T - 1;

  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
  });
  host.addEventListener("mouseleave", hideTip);

  function draw() {
    hideTip();
    host.innerHTML =
      chipRow(ctx.tokens, ctx.chars, query, "pick the query token — whose attention are we inspecting?") +
      section("1", "every head, every layer") +
      note(`Each row below is one of the model's <b>12 attention heads</b> (3 layers × 4 heads). The cells show how much the query token <b>“${escapeHtml(glyph(chars[query]))}”</b> attends to each earlier token — brighter means more weight. Masked (future) positions are dark.`) +
      atlas() +
      section("2", "reading it") +
      note("Look for <b>specialisation</b>: some heads pour most of their weight onto the single preceding character (a bright cell just left of the diagonal), some fixate on the very first token (a bright left column), and some spread out. Early and late layers tend to attend differently. In a model this small the patterns are noisy, but the division of labour is already visible.");

    host.querySelectorAll(".lz-chip").forEach((c) =>
      c.addEventListener("click", () => { query = +c.dataset.i; draw(); })
    );
  }

  function atlas() {
    const cw = Math.max(14, Math.min(40, Math.floor(560 / T)));
    const rowH = 22, padL = 52, padT = 26, layGap = 8;
    const gridW = T * cw;
    const w = padL + gridW + 30;
    const h = padT + 12 * rowH + 2 * layGap + 20;

    let s = "";
    // column headers: the target tokens
    for (let c = 0; c < T; c++) {
      const isQ = c === query;
      s += `<text x="${padL + c * cw + cw / 2}" y="${padT - 9}" class="${isQ ? "lz-slab on" : "lz-slab"}" text-anchor="middle">${escapeHtml(glyph(chars[c]))}</text>`;
    }

    for (let r = 0; r < 12; r++) {
      const l = Math.floor(r / 4), hd = r % 4;
      const y = padT + r * rowH + l * layGap;
      s += `<text x="${padL - 8}" y="${y + rowH / 2 + 3}" class="lz-slab${hd === 0 ? " on" : ""}" text-anchor="end">L${l}·H${hd}</text>`;
      const row = F.blocks[l].heads[hd].attn[query];
      for (let c = 0; c < T; c++) {
        const v = row[c];
        const fill = c > query ? "#0b0b09" : colorSeq(v);
        s += `<rect x="${padL + c * cw}" y="${y}" width="${cw - 0.6}" height="${rowH - 0.6}" fill="${fill}" ` +
          `data-tip="L${l} H${hd}: “${escapeHtml(glyph(chars[query]))}” → “${escapeHtml(glyph(chars[c]))}” = ${(v * 100).toFixed(1)}%"/>`;
      }
      s += `<rect x="${padL}" y="${y}" width="${gridW}" height="${rowH - 0.6}" class="lz-heatframe"/>`;
    }
    return svg(w, h, s, "rows = 12 heads · columns = tokens the query can attend to");
  }

  draw();
}
