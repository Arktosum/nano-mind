/* blocks.js — stacking blocks: the same structure, three times.
   We visualized block 0 in full; blocks 1 and 2 repeat it with their
   own weights. This lesson shows the residual stream for one token
   deepening block by block, and how its attention shifts with depth.
   Reads ctx.fwd (all 3 blocks already computed). */

export function renderBlocksLesson(host, ctx) {
  const F = ctx.fwd;
  const T = F.T;
  const streams = [F.h0, F.blocks[0].hOut, F.blocks[1].hOut, F.blocks[2].hOut];
  let active = T - 1;

  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
  });
  host.addEventListener("mouseleave", hideTip);

  function draw() {
    hideTip();
    const shared = Math.max(...streams.map((s) => maxAbs([s[active]])));

    host.innerHTML =
      chipRow(ctx.tokens, ctx.chars, active, "your sequence — click a position") +
      section("1", "the same block, three times") +
      note("Everything in the last chapters — LayerNorm, attention, residual, LayerNorm, feed-forward, residual — is <b>one block</b>. This model stacks <b>3 identical blocks</b>. They share the exact structure; each just has its own learned weights and refines the residual stream a little more.") +
      section("2", "the stream deepens") +
      note("Here is one token's 64-dim residual vector after each block. Same slot, progressively transformed — each block reads it, computes a correction, and adds that back. Nothing is thrown away; the representation is <i>edited</i> in place.") +
      sumDiagram([
        { vec: streams[0][active], label: "h⁰", sub: "after embed", max: shared },
        { vec: streams[1][active], label: "h¹", sub: "after block 0", max: shared },
        { vec: streams[2][active], label: "h²", sub: "after block 1", max: shared },
        { vec: streams[3][active], label: "h³", sub: "after block 2", max: shared, out: true },
      ], "residual stream for “" + glyph(ctx.chars[active]) + "”, block by block") +
      section("3", "attention shifts with depth") +
      note("Each block learned its attention from scratch, so the same token looks at different things at different depths. Below: where this token's head 0 attends in each block (brighter = more weight). After block 2, the stream heads to the output stage.") +
      attnByDepth(active);

    host.querySelectorAll(".lz-chip").forEach((c) =>
      c.addEventListener("click", () => { active = +c.dataset.i; draw(); })
    );
  }

  function attnByDepth(t) {
    const cell = Math.max(16, Math.min(34, Math.floor(560 / T)));
    const padL = 90, padT = 24, rowGap = 14, rowH = cell;
    const w = padL + T * cell + 30;
    const h = padT + 3 * (rowH + rowGap) + 16;
    let s = "";
    // column headers = target tokens
    for (let j = 0; j < T; j++)
      s += `<text x="${padL + j * cell + cell / 2}" y="${padT - 8}" class="lz-slab" text-anchor="middle">${escapeHtml(glyph(ctx.chars[j]))}</text>`;
    for (let l = 0; l < 3; l++) {
      const row = F.blocks[l].heads[0].attn[t];
      const y = padT + l * (rowH + rowGap);
      s += `<text x="${padL - 10}" y="${y + rowH / 2 + 3}" class="lz-slab on" text-anchor="end">block ${l}</text>`;
      for (let j = 0; j < T; j++) {
        const v = row[j];
        const fill = j > t ? "#0b0b09" : colorSeq(v);
        s += `<rect x="${padL + j * cell}" y="${y}" width="${cell - 0.6}" height="${rowH - 0.6}" fill="${fill}" data-tip="block ${l}: “${escapeHtml(glyph(ctx.chars[t]))}” → “${escapeHtml(glyph(ctx.chars[j]))}” = ${(v * 100).toFixed(1)}%"/>`;
      }
      s += `<rect x="${padL}" y="${y}" width="${T * cell}" height="${rowH}" class="lz-heatframe"/>`;
    }
    return svg(w, h, s, "head-0 attention from “" + glyph(ctx.chars[t]) + "” at each depth");
  }

  draw();
}
