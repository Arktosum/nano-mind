/* lesson-embedding.js — a focused, richly detailed teaching view of ONE
   operation: the token embedding lookup, taught as one-hot selection.

     char ─▶ one-hot (1×75) × W_E (75×64) = Eᵗᵒᵏ (1×64)

   The one-hot column is row-aligned with the embedding table, so the lit
   selector cell sits exactly beside the row it picks. A second panel shows
   every token's row stacking into the T×64 matrix that leaves this step.

   Hover anything: read a dimension's exact value, or reveal which character
   any table row belongs to. Click a token (top) or a stacked row to inspect. */

function renderEmbeddingLesson(host, ctx) {
  const { tokens, chars, table } = ctx; // table: 75×64
  const ROWS = table.length, COLS = table[0].length;
  const tableMax = maxAbs(table);
  const T = tokens.length;

  let active = T - 1;

  // hover delegation (attached once; survives redraws)
  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
    if (t.dataset && t.dataset.pick !== undefined && e.buttons) {} // no-op
  });
  host.addEventListener("mouseleave", hideTip);
  host.addEventListener("click", (e) => {
    const p = e.target.dataset && e.target.dataset.pick;
    if (p !== undefined && p !== "") { active = +p; draw(); }
  });

  function draw() {
    hideTip();
    host.innerHTML =
      chips() +
      `<div class="lz-stats">W_E &nbsp;=&nbsp; ${ROWS} rows × ${COLS} dims &nbsp;=&nbsp; ` +
      `<b>${(ROWS * COLS).toLocaleString()}</b> learned numbers &nbsp;·&nbsp; ` +
      `random at initialisation, shaped by training</div>` +
      diagram(tokens[active]) +
      stacked();
    host.querySelectorAll(".lz-chip").forEach((c) =>
      c.addEventListener("click", () => { active = +c.dataset.i; draw(); })
    );
  }

  function chips() {
    let s =
      '<div class="lz-chips"><span class="lz-chips-label">your sequence — click a token to inspect its lookup</span>' +
      '<div class="lz-chip-row">';
    tokens.forEach((t, i) => {
      s +=
        `<button class="lz-chip${i === active ? " on" : ""}" data-i="${i}" ` +
        `title="position ${i}">` +
        `<span class="lz-chip-g">${escapeHtml(glyph(chars[i]))}</span>` +
        `<span class="lz-chip-id">${t}</span></button>`;
    });
    return s + "</div></div>";
  }

  /* ── main: char → one-hot × table = vector ─────────────────── */
  function diagram(id) {
    const rowH = 4, tCol = 4, padT = 34, padL = 12;
    const charW = 48, charH = 48, gap = 16, opW = 28, arrowW = 54;
    const idLabW = 22, ohW = 15, outCell = 6, outH = 22;

    const charX = padL;
    const arr1 = charX + charW;
    const idLabX = arr1 + arrowW;
    const ohX = idLabX + idLabW;
    const opX = ohX + ohW + gap;
    const tableX = opX + opW + gap + 34; // room for "row N" label
    const tableW = COLS * tCol;
    const arr2 = tableX + tableW;
    const outX = arr2 + arrowW + 8;
    const outW = COLS * outCell;
    const totalW = outX + outW + 70;

    const tableY = padT;
    const tableH = ROWS * rowH;
    const litY = tableY + id * rowH + rowH / 2;
    const totalH = tableY + tableH + 34;
    const rowMax = maxAbs([table[id]]);

    let s = "";

    // char box
    const charY = Math.min(Math.max(litY - charH / 2, padT), tableY + tableH - charH);
    s += `<rect x="${charX}" y="${charY}" width="${charW}" height="${charH}" class="lz-charbox"/>`;
    s += txt(charX + charW / 2, charY + charH / 2 - 3, glyph(chars[active]), "lz-charglyph");
    s += txt(charX + charW / 2, charY + charH / 2 + 16, "id " + id, "lz-charid");
    s += arrow(arr1, litY, idLabX - 4, litY, "encode");

    // one-hot column, row-aligned with table
    s += txt(ohX + ohW / 2, tableY - 14, "one-hot", "lz-cap");
    s += txt(ohX + ohW / 2, tableY - 3, "1×75", "lz-cap dim");
    for (let r = 0; r < ROWS; r++) {
      const on = r === id;
      s +=
        `<rect x="${ohX}" y="${tableY + r * rowH}" width="${ohW}" height="${rowH - 0.5}" ` +
        `class="${on ? "lz-oh-on" : "lz-oh"}" data-tip="${on ? `one-hot[${r}] = 1` : `one-hot[${r}] = 0`}"/>`;
    }
    s += txt(idLabX + idLabW - 4, tableY + 5, "0", "lz-tick", "end");
    s += txt(idLabX + idLabW - 4, litY + 3, String(id), "lz-tick on", "end");
    s += txt(idLabX + idLabW - 4, tableY + tableH - 1, String(ROWS - 1), "lz-tick", "end");
    s += `<text x="${ohX + ohW / 2}" y="${litY + 3}" class="lz-onehot1" text-anchor="middle">1</text>`;

    // × operator
    s += txt(opX + opW / 2, litY + 7, "×", "lz-op");

    // W_E table (dim non-selected rows)
    s += txt(tableX + tableW / 2, tableY - 14, "W_E", "lz-cap");
    s += txt(tableX + tableW / 2, tableY - 3, `${ROWS}×${COLS}`, "lz-cap dim");
    for (let r = 0; r < ROWS; r++) {
      const dim = r === id ? 1 : 0.26;
      for (let c = 0; c < COLS; c++)
        s += `<rect x="${tableX + c * tCol}" y="${tableY + r * rowH}" width="${tCol - 0.4}" height="${rowH - 0.4}" fill="${colorDiverging(table[r][c], tableMax)}" opacity="${dim}"/>`;
    }
    s += `<rect x="${tableX - 1}" y="${tableY + id * rowH - 1}" width="${tableW + 2}" height="${rowH + 2}" class="lz-rowframe"/>`;
    s += txt(tableX - 8, litY + 3, "row " + id, "lz-rowlab", "end");
    // transparent per-row hover strips → reveal each row's character
    for (let r = 0; r < ROWS; r++)
      s += `<rect x="${tableX}" y="${tableY + r * rowH}" width="${tableW}" height="${rowH}" fill="transparent" ` +
        `data-pick="" data-tip="W_E row ${r} = embedding of &#8216;${escapeHtml(glyph(decode(r)))}&#8217;"/>`;

    // selected row → output
    s += arrow(arr2, litY, outX - 4, litY, "=");

    // output vector — subscript t: this is ONE row, for position `active`
    s += txt(outX + outW / 2, litY - outH / 2 - 16, "Eᵗᵒᵏₜ  ·  t = " + active, "lz-cap");
    s += txt(outX + outW / 2, litY - outH / 2 - 5, "1×64  (one row)", "lz-cap dim");
    for (let c = 0; c < COLS; c++)
      s += `<rect x="${outX + c * outCell}" y="${litY - outH / 2}" width="${outCell - 0.5}" height="${outH}" ` +
        `fill="${colorDiverging(table[id][c], rowMax)}" data-tip="dim ${c} = ${table[id][c].toFixed(4)}"/>`;
    s += `<rect x="${outX}" y="${litY - outH / 2}" width="${outW}" height="${outH}" class="lz-outframe"/>`;
    const preview = table[id].slice(0, 6).map((v) => v.toFixed(2)).join(", ");
    s += txt(outX + outW / 2, litY + outH / 2 + 15, `[ ${preview}, … ]`, "lz-preview");

    return svg(totalW, totalH, s, "the selection, for one token");
  }

  /* ── second panel: every token stacks into T×64 ────────────── */
  function stacked() {
    const rowH = 14, cw = 6, padT = 26, padL = 84;
    const w = padL + COLS * cw + 40;
    const h = padT + T * rowH + 14;
    const mx = maxAbs(tokens.map((id) => table[id]));
    let s = "";
    s += txt(padL + (COLS * cw) / 2, 14, `Eᵗᵒᵏ  (no subscript)  —  ${T}×${COLS}`, "lz-cap");
    for (let i = 0; i < T; i++) {
      const id = tokens[i];
      const y = padT + i * rowH;
      const on = i === active;
      s += txt(padL - 10, y + rowH / 2 + 3, `${glyph(chars[i])}  ${id}`, on ? "lz-slab on" : "lz-slab", "end");
      for (let c = 0; c < COLS; c++)
        s += `<rect x="${padL + c * cw}" y="${y}" width="${cw - 0.5}" height="${rowH - 1.5}" fill="${colorDiverging(table[id][c], mx)}" opacity="${on ? 1 : 0.85}"/>`;
      if (on) s += `<rect x="${padL - 1}" y="${y - 1}" width="${COLS * cw + 2}" height="${rowH - 1.5 + 2}" class="lz-rowframe"/>`;
      // clickable strip
      s += `<rect x="${padL}" y="${y}" width="${COLS * cw}" height="${rowH}" fill="transparent" data-pick="${i}" data-tip="position ${i}: &#8216;${escapeHtml(glyph(chars[i]))}&#8217; → row ${id}"/>`;
    }
    return (
      `<div class="lz-panel-title">Each row here is one <b>Eᵗᵒᵏₜ</b> — exactly the 1×64 vector built ` +
      `above, for one position. Stack all ${T} of them and you get <b>Eᵗᵒᵏ</b> (no subscript), the ` +
      `<b>T×64</b> matrix this step hands to the next — the initial residual stream.</div>` +
      svg(w, h, s, "the full sequence")
    );
  }

  // helpers
  function txt(x, y, str, cls, anchor = "middle") {
    return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${escapeHtml(str)}</text>`;
  }
  function arrow(x0, y0, x1, y1, label) {
    let s = `<line x1="${x0}" y1="${y0}" x2="${x1 - 7}" y2="${y1}" class="lz-arrow"/>`;
    s += `<path d="M ${x1 - 7} ${y1 - 4} L ${x1} ${y1} L ${x1 - 7} ${y1 + 4} Z" class="lz-arrowhead"/>`;
    if (label) s += `<text x="${(x0 + x1) / 2}" y="${y0 - 8}" class="lz-arrowlabel" text-anchor="middle">${escapeHtml(label)}</text>`;
    return s;
  }
  function svg(w, h, inner, cap) {
    return (
      `<div class="lz-diagram">` +
      (cap ? `<div class="lz-diagram-cap">${cap}</div>` : "") +
      `<div class="lz-svgwrap"><svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid meet">${inner}</svg></div></div>`
    );
  }

  draw();
}
