/* lesson-position.js — the third operation: positional embedding + the sum
   that creates the residual stream h⁰.

   Three teaching beats:
     1. the position lookup (same one-hot trick, indexed by position t)
     2. the elementwise sum  h⁰_t = Eᵗᵒᵏ_t + Eᵖᵒˢ_t
     3. the payoff — a repeated character has identical Eᵗᵒᵏ but different
        Eᵖᵒˢ, so its h⁰ differs by position (order is now encoded). */

export function renderPositionSumLesson(host, ctx) {
  const { tokens, chars, weights } = ctx;
  const Wt = weights["token_embedding_table.weight"]; // 75×64
  const Wp = weights["position_embedding_table.weight"]; // 64×64
  const COLS = Wt[0].length; // 64
  const BLOCK = Wp.length; // 64
  const T = tokens.length;

  const Etok = (i) => Wt[tokens[i]];
  const Epos = (i) => Wp[i];
  const H0 = (i) => Etok(i).map((v, d) => v + Epos(i)[d]);
  const wpMax = maxAbs(Wp);

  let active = T - 1;

  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
  });
  host.addEventListener("mouseleave", hideTip);
  host.addEventListener("click", (e) => {
    const p = e.target.dataset && e.target.dataset.pick;
    if (p) { active = +p; draw(); }
  });

  function draw() {
    hideTip();
    host.innerHTML =
      chips() +
      section("1", "the lookup — deliberately the same machine") +
      note("Mechanically this is <b>identical</b> to the token lookup: a one-hot selects a row of a learned table. That sameness is the point — so don't look for a new trick here. The difference is in <i>what the index means</i>, which is section 2.") +
      posLookup(active) +
      section("2", "same machine, opposite meaning") +
      contrastTable() +
      reuseView() +
      sinuView() +
      section("3", "how it learns to encode position") +
      howLearned() +
      section("4", "the sum — fuse “what” with “where”") +
      sumView(active) +
      section("5", "the payoff — order is now encoded") +
      orderView();
    host.querySelectorAll(".lz-chip").forEach((c) =>
      c.addEventListener("click", () => { active = +c.dataset.i; draw(); })
    );
  }

  /* ── 2a. the contrast table (the real differences) ─────────── */
  function contrastTable() {
    const rows = [
      ["indexed by", "identity — <i>which character</i>", "order — <i>which slot</i>"],
      ["rows mean", `75 = the vocabulary`, `64 = max context length`],
      ["a row is reused…", "every time that char appears", "<b>never</b> — once per position"],
      ["index has order?", "no · id 21 vs 22 is arbitrary", "yes · pos 21 precedes 22"],
      ["runs out when…", "never (fixed alphabet)", "text &gt; 64 → we <b>truncate</b>"],
      ["how rows are made", "<b>must</b> be learned", "learned here — or a fixed formula (↓)"],
    ];
    let s =
      '<table class="lz-contrast"><thead><tr><th></th>' +
      '<th>W_E · token</th><th>W_P · position</th></tr></thead><tbody>';
    for (const [k, a, b] of rows)
      s += `<tr><td class="k">${k}</td><td>${a}</td><td class="p">${b}</td></tr>`;
    s += "</tbody></table>";
    s +=
      '<div class="lz-note small">Honest footnote: in a model this small the learned position vectors carry only ' +
      "<b>weak</b> order structure (adjacent-position similarity ≈ 0.07, barely above random). In large models it " +
      "becomes rich and smooth. But the <i>indexing</i> above differs regardless of how well it was trained.</div>";
    return s;
  }

  /* ── 2b. reuse: shared W_E row vs distinct W_P rows ────────── */
  function reuseView() {
    const byId = new Map();
    tokens.forEach((id, i) => { if (!byId.has(id)) byId.set(id, []); byId.get(id).push(i); });
    const rep = [...byId.entries()].find(([, ps]) => ps.length > 1);
    if (!rep) {
      return note("(Type a prompt with a repeated character — e.g. <b>“Hi there”</b> — to see identity get shared while position stays unique.)");
    }
    const [id, ps] = rep;
    const ch = glyph(decode(id));
    const p0 = ps[0], p1 = ps[1];
    const W = 720, H = 190, bw = 118, bh = 34;
    const lx = 20, mx = 300, rx = 560;
    const yTopA = 30, yBotA = 96; // two source cards (identity block)
    const cyA = (yTopA + yBotA) / 2 + bh / 2;
    let s = "";
    // identity block: two cards → ONE shared W_E row
    s += box(lx, yTopA, bw, bh, `“${ch}” @ pos ${p0}`, "src");
    s += box(lx, yBotA, bw, bh, `“${ch}” @ pos ${p1}`, "src");
    s += line(lx + bw, yTopA + bh / 2, mx, cyA);
    s += line(lx + bw, yBotA + bh / 2, mx, cyA);
    s += box(mx, cyA - bh / 2, bw + 26, bh, `W_E  row ${id}`, "shared");
    s += txt(mx + (bw + 26) / 2, cyA + bh / 2 + 15, "one row — identity is SHARED", "lz-reuse-lab shared");

    // position block: two cards → TWO distinct W_P rows
    s += line(lx + bw, yTopA + bh / 2, rx, yTopA + bh / 2);
    s += line(lx + bw, yBotA + bh / 2, rx, yBotA + bh / 2);
    s += box(rx, yTopA, bw, bh, `W_P  row ${p0}`, "distinct");
    s += box(rx, yBotA, bw, bh, `W_P  row ${p1}`, "distinct");
    s += txt(rx + bw / 2, yBotA + bh + 15, "two rows — position is UNIQUE", "lz-reuse-lab distinct");

    return note(`Both <b>“${ch}”</b>s (positions ${p0} and ${p1}) share one <b>W_E</b> row, but land on two different <b>W_P</b> rows:`) +
      svg(W, H, s, "identity shared · position unique");
  }

  function chips() {
    let s =
      '<div class="lz-chips"><span class="lz-chips-label">your sequence — click a position to inspect it</span>' +
      '<div class="lz-chip-row">';
    tokens.forEach((t, i) => {
      s +=
        `<button class="lz-chip${i === active ? " on" : ""}" data-i="${i}" title="position ${i}">` +
        `<span class="lz-chip-g">${escapeHtml(glyph(chars[i]))}</span>` +
        `<span class="lz-chip-id">pos ${i}</span></button>`;
    });
    return s + "</div></div>";
  }

  /* ── 1. position lookup (row t of W_P) ─────────────────────── */
  function posLookup(t) {
    const rowH = 4, tCol = 4, padT = 30, padL = 12;
    const posW = 54, posH = 40, gap = 16, opW = 26, arrowW = 52;
    const idLabW = 26, ohW = 15, outCell = 6, outH = 22;

    const posX = padL;
    const arr1 = posX + posW;
    const idLabX = arr1 + arrowW;
    const ohX = idLabX + idLabW;
    const opX = ohX + ohW + gap;
    const tableX = opX + opW + gap + 34;
    const tableW = COLS * tCol;
    const arr2 = tableX + tableW;
    const outX = arr2 + arrowW + 8;
    const outW = COLS * outCell;
    const totalW = outX + outW + 70;

    const tableY = padT;
    const tableH = BLOCK * rowH;
    const litY = tableY + t * rowH + rowH / 2;
    const totalH = tableY + tableH + 30;
    const rowMax = maxAbs([Wp[t]]);
    let s = "";

    // "pos t" box (the input is the position number itself)
    const boxY = Math.min(Math.max(litY - posH / 2, padT), tableY + tableH - posH);
    s += `<rect x="${posX}" y="${boxY}" width="${posW}" height="${posH}" class="lz-charbox"/>`;
    s += txt(posX + posW / 2, boxY + posH / 2 - 2, "pos " + t, "lz-charglyph small");
    s += txt(posX + posW / 2, boxY + posH / 2 + 14, "“" + glyph(chars[t]) + "”", "lz-charid");
    s += arrow(arr1, litY, idLabX - 4, litY, "one-hot");

    // one-hot over positions
    s += txt(ohX + ohW / 2, tableY - 13, "1×64", "lz-cap dim");
    for (let r = 0; r < BLOCK; r++)
      s += `<rect x="${ohX}" y="${tableY + r * rowH}" width="${ohW}" height="${rowH - 0.5}" class="${r === t ? "lz-oh-on" : "lz-oh"}" data-tip="one-hot[${r}] = ${r === t ? 1 : 0}"/>`;
    s += txt(idLabX + idLabW - 4, tableY + 5, "0", "lz-tick", "end");
    s += txt(idLabX + idLabW - 4, litY + 3, "t=" + t, "lz-tick on", "end");
    s += txt(idLabX + idLabW - 4, tableY + tableH - 1, String(BLOCK - 1), "lz-tick", "end");

    s += txt(opX + opW / 2, litY + 7, "×", "lz-op");

    // W_P table
    s += txt(tableX + tableW / 2, tableY - 13, `W_P  ${BLOCK}×${COLS}`, "lz-cap");
    for (let r = 0; r < BLOCK; r++) {
      const dim = r === t ? 1 : 0.26;
      for (let c = 0; c < COLS; c++)
        s += `<rect x="${tableX + c * tCol}" y="${tableY + r * rowH}" width="${tCol - 0.4}" height="${rowH - 0.4}" fill="${colorDiverging(Wp[r][c], wpMax)}" opacity="${dim}"/>`;
    }
    s += `<rect x="${tableX - 1}" y="${tableY + t * rowH - 1}" width="${tableW + 2}" height="${rowH + 2}" class="lz-rowframe"/>`;
    s += txt(tableX - 8, litY + 3, "row " + t, "lz-rowlab", "end");

    s += arrow(arr2, litY, outX - 4, litY, "=");

    // output Eᵖᵒˢ_t
    s += txt(outX + outW / 2, litY - outH / 2 - 15, "Eᵖᵒˢₜ", "lz-cap");
    s += txt(outX + outW / 2, litY - outH / 2 - 4, "1×64", "lz-cap dim");
    for (let c = 0; c < COLS; c++)
      s += `<rect x="${outX + c * outCell}" y="${litY - outH / 2}" width="${outCell - 0.5}" height="${outH}" fill="${colorDiverging(Wp[t][c], rowMax)}" data-tip="dim ${c} = ${Wp[t][c].toFixed(4)}"/>`;
    s += `<rect x="${outX}" y="${litY - outH / 2}" width="${outW}" height="${outH}" class="lz-outframe"/>`;

    return note("The selector is the <b>position</b> t, not the character — so the lit row is simply row t. Row t of W_P is “what it means to be the t-th token.”") +
      svg(totalW, totalH, s, "Eᵖᵒˢₜ = W_P[t]");
  }

  /* ── 2c. learned vs sinusoidal — make the word a picture ───── */
  function sinuView() {
    const P = 28; // show first 28 positions
    // the classic fixed formula: PE(pos,2i)=sin(pos/10000^(2i/d)), (2i+1)=cos(...)
    const sinu = [];
    for (let pos = 0; pos < P; pos++) {
      const row = [];
      for (let i = 0; i < COLS; i++) {
        const denom = Math.pow(10000, (2 * Math.floor(i / 2)) / COLS);
        row.push(i % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom));
      }
      sinu.push(row);
    }
    const learned = Wp.slice(0, P);
    const lMax = maxAbs(learned);

    const cell = 6, padL = 14, padT = 30, gap = 64;
    const gw = COLS * cell, gh = P * cell;
    const panelW = gw;
    const w = padL + panelW + gap + panelW + padL + 20;
    const h = padT + gh + 28;

    function heat(x, data, mx, title, sub) {
      let s = txt(x + gw / 2, 14, title, "lz-cap");
      s += txt(x + gw / 2, 25, sub, "lz-cap dim");
      for (let r = 0; r < P; r++)
        for (let c = 0; c < COLS; c++)
          s += `<rect x="${x + c * cell}" y="${padT + r * cell}" width="${cell - 0.4}" height="${cell - 0.4}" fill="${colorDiverging(data[r][c], mx)}" data-tip="pos ${r} · dim ${c} = ${data[r][c].toFixed(3)}"/>`;
      s += `<rect x="${x}" y="${padT}" width="${gw}" height="${gh}" class="lz-heatframe"/>`;
      s += txt(x - 4, padT + 6, "pos 0", "lz-tick", "end");
      s += txt(x - 4, padT + gh - 1, "pos " + (P - 1), "lz-tick", "end");
      s += txt(x + gw / 2, padT + gh + 16, "→ 64 dimensions →", "lz-tick");
      return s;
    }
    let s = "";
    s += heat(padL, sinu, 1, "SINUSOIDAL", "fixed formula · not used here");
    s += heat(padL + panelW + gap, learned, lMax, "LEARNED", "this model's actual W_P");
    s += txt(padL + panelW + gap / 2, padT + gh / 2, "vs", "lz-op");

    return (
      note(
        "<b>What “sinusoidal” means:</b> instead of learning W_P, you <i>compute</i> each position's row from " +
        "sine &amp; cosine waves — each column is a wave of a different frequency, so every position gets a unique, " +
        "smooth fingerprint. That's the striped pattern on the left (real values from the formula). This model " +
        "doesn't use it — its learned W_P (right) is comparatively noisy."
      ) +
      formulaBlock() +
      svg(w, h, s, "left: computed by formula · right: learned by this model")
    );
  }

  function formulaBlock() {
    const frac = (n, d) => `<span class="frac"><span class="num">${n}</span><span class="den">${d}</span></span>`;
    return (
      '<div class="lz-formula">' +
      `<div>PE<sub>(pos,&nbsp;2i)</sub> &nbsp;=&nbsp; sin<span class="paren">(</span>&nbsp;${frac("pos", "10000<sup>2i/d</sup>")}&nbsp;<span class="paren">)</span></div>` +
      `<div>PE<sub>(pos,&nbsp;2i+1)</sub> &nbsp;=&nbsp; cos<span class="paren">(</span>&nbsp;${frac("pos", "10000<sup>2i/d</sup>")}&nbsp;<span class="paren">)</span></div>` +
      '<div class="lz-formula-note">d = 64 dims · i indexes dimension-pairs 0…31 · even dims use sin, odd use cos · low i → slow waves, high i → fast waves</div>' +
      "</div>"
    );
  }

  /* ── 3. HOW gradient descent encodes position ─────────────── */
  function howLearned() {
    return (
      '<div class="lz-note key">' +
      "The lookup is the same as always — so <b>where does the positional code come from?</b> Not from this step. " +
      "It's forged during <b>training</b> (which nano-mind doesn't show), by three things working together:" +
      "</div>" +
      '<div class="lz-how">' +
      '<div class="lz-how-item"><span class="n">1</span><b>The objective.</b> The model is trained to predict the ' +
      "next character. Guess wrong → the <i>loss</i> is high.</div>" +
      '<div class="lz-how-item"><span class="n">2</span><b>The pressure.</b> Natural text is ordered: the next ' +
      "character depends on <i>where</i> you are, and reordering the same characters changes their meaning. A model " +
      "that ignores position predicts worse — so any tweak to W_P that lets it use order <b>lowers the loss</b>, and " +
      "gradient descent rolls that way.</div>" +
      '<div class="lz-how-item"><span class="n">3</span><b>The mechanism — parameter sharing.</b> The crux. The one ' +
      "row <b>W_P[t]</b> is reused for the t-th token of <i>every</i> training sequence, so it collects a gradient " +
      "from millions of different position-t tokens. Whatever is <b>consistently true about being t-th</b> " +
      "accumulates in that row; the per-example content pulls in random directions and <b>averages out</b>. The row " +
      "converges to the distilled signal of “being position t.”</div>" +
      "</div>" +
      loopView() +
      note(
        "And you can see a fingerprint of it in the real weights. Below is the <b>length</b> (‖·‖) of each learned " +
        "position row. <b>Position 0 towers over the rest</b> — training discovered that the first character of a " +
        "sequence is statistically special (it opens the text) and handed position 0 a loud, distinctive vector. " +
        "Nobody programmed that; the data pushed it there."
      ) +
      normBars()
    );
  }

  function loopView() {
    const bh = 40, padT = 22;
    const specs = [
      ["W_P[t]", "wp", 92],
      ["h⁰ = Eᵗᵒᵏ + Eᵖᵒˢ", "", 150],
      ["attn → ffn → logits", "", 156],
      ["loss vs true char", "", 132],
    ];
    const gap = 34;
    let xs = [], x = 10;
    specs.forEach((sp) => { xs.push(x); x += sp[2] + gap; });
    const totalW = x - gap + 10;
    const y = padT;
    let s = "";
    // forward arrows
    for (let i = 0; i < specs.length - 1; i++)
      s += line(xs[i] + specs[i][2], y + bh / 2, xs[i + 1], y + bh / 2);
    specs.forEach((sp, i) => { s += box(xs[i], y, sp[2], bh, sp[0], sp[1]); });
    s += txt(xs[0] + specs[0][2] / 2, y + bh + 14, "the thing being learned", "lz-reuse-lab distinct");
    // backprop return arrow (loss → W_P)
    const yb = y + bh + 40;
    const lossCx = xs[3] + specs[3][2] / 2, wpCx = xs[0] + specs[0][2] / 2;
    s += `<line x1="${lossCx}" y1="${y + bh}" x2="${lossCx}" y2="${yb}" class="lz-arrow"/>`;
    s += `<line x1="${lossCx}" y1="${yb}" x2="${wpCx}" y2="${yb}" class="lz-arrow"/>`;
    s += `<line x1="${wpCx}" y1="${yb}" x2="${wpCx}" y2="${y + bh + 6}" class="lz-arrow"/>`;
    s += `<path d="M ${wpCx - 4} ${y + bh + 6} L ${wpCx} ${y + bh} L ${wpCx + 4} ${y + bh + 6} Z" class="lz-arrowhead"/>`;
    s += txt((lossCx + wpCx) / 2, yb - 6, "backprop — ∂loss/∂W_P nudges every position row", "lz-arrowlabel");
    return svg(totalW, yb + 20, s, "one training step · offline · not shown in the forward pass");
  }

  function normBars() {
    const N = Math.min(BLOCK, 32);
    const norms = [];
    for (let i = 0; i < N; i++) { let ss = 0; for (const v of Wp[i]) ss += v * v; norms.push(Math.sqrt(ss)); }
    const mx = Math.max(...norms);
    const bw = 16, g = 5, padL = 42, padT = 12, chartH = 120;
    const w = padL + N * (bw + g) + 12, h = padT + chartH + 32;
    let s = "";
    s += txt(padL - 8, padT + 6, mx.toFixed(0), "lz-tick", "end");
    s += txt(padL - 8, padT + chartH, "0", "lz-tick", "end");
    s += `<line x1="${padL}" y1="${padT + chartH}" x2="${padL + N * (bw + g)}" y2="${padT + chartH}" class="lz-arrow"/>`;
    for (let i = 0; i < N; i++) {
      const bh = chartH * (norms[i] / mx);
      const x = padL + i * (bw + g), yy = padT + chartH - bh;
      s += `<rect x="${x}" y="${yy}" width="${bw}" height="${bh}" fill="${i === 0 ? "#d6401f" : "#4a4a3e"}" data-tip="pos ${i} · ‖W_P‖ = ${norms[i].toFixed(2)}"/>`;
      if (i === 0 || i % 4 === 0) s += txt(x + bw / 2, padT + chartH + 13, String(i), "lz-tick");
    }
    s += txt(padL + (N * (bw + g)) / 2, padT + chartH + 28, "position →", "lz-tick");
    return svg(w, h, s, "‖ W_P[t] ‖ by position — a visible fingerprint of training");
  }

  /* ── 2. the sum: three rows adding ─────────────────────────── */
  function sumView(t) {
    const et = Etok(t), ep = Epos(t), h = H0(t);
    const sMax = Math.max(maxAbs([et]), maxAbs([ep]), maxAbs([h]));
    const cw = 11, padL = 92, rowH = 26, gapR = 30, padT = 16;
    const gridW = COLS * cw;
    const w = padL + gridW + 40;
    const h3 = padT + 3 * rowH + 2 * gapR + 20;

    const rows = [
      { y: padT, vec: et, lab: "Eᵗᵒᵏₜ", name: "what (char)" },
      { y: padT + rowH + gapR, vec: ep, lab: "Eᵖᵒˢₜ", name: "where (pos)" },
      { y: padT + 2 * (rowH + gapR), vec: h, lab: "h⁰ₜ", name: "residual stream", out: true },
    ];
    let s = "";
    rows.forEach((r, ri) => {
      s += txt(padL - 12, r.y + rowH / 2 + 3, r.lab, r.out ? "lz-slab on" : "lz-slab", "end");
      s += txt(padL - 12, r.y + rowH / 2 + 16, r.name, "lz-subslab", "end");
      for (let c = 0; c < COLS; c++)
        s += `<rect x="${padL + c * cw}" y="${r.y}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(r.vec[c], sMax)}" ` +
          `data-tip="dim ${c}: ${et[c].toFixed(3)} + ${ep[c].toFixed(3)} = ${h[c].toFixed(3)}"/>`;
      if (r.out) s += `<rect x="${padL}" y="${r.y}" width="${gridW}" height="${rowH}" class="lz-outframe"/>`;
      // + and = between rows
      if (ri < 2) s += txt(padL + gridW + 18, r.y + rowH + gapR / 2 + 4, ri === 0 ? "+" : "=", "lz-op");
    });

    const call =
      `<div class="lz-callout">dim 0 &nbsp;=&nbsp; <span class="k">${et[0].toFixed(2)}</span>` +
      ` <span class="p">+</span> <span class="k">${ep[0].toFixed(2)}</span>` +
      ` <span class="p">=</span> <span class="h">${h[0].toFixed(2)}</span> &nbsp;·&nbsp; hover any cell for its dimension’s arithmetic</div>`;

    return note("Add the two 1×64 vectors elementwise. No weights, no mixing across dimensions — dim <i>d</i> of the sum is just dim <i>d</i> of each input added together.") +
      svg(w, h3, s, "h⁰ₜ = Eᵗᵒᵏₜ + Eᵖᵒˢₜ") + call;
  }

  /* ── 3. order payoff: repeated char, same Eᵗᵒᵏ, different h⁰ ── */
  function orderView() {
    const byId = new Map();
    tokens.forEach((id, i) => {
      if (!byId.has(id)) byId.set(id, []);
      byId.get(id).push(i);
    });
    const rep = [...byId.entries()].find(([, ps]) => ps.length > 1);

    if (!rep) {
      return note(
        "Your text has no repeated character, so there's nothing to contrast here. Try a prompt like " +
        "<b>“Hi there”</b> (two <b>e</b>'s) or <b>“banana”</b> — identical token embeddings will get " +
        "pulled apart by position."
      );
    }

    const [id, ps] = rep;
    const cols = ps.slice(0, 4);
    const ch = glyph(decode(id));
    const cw = 9, padL = 96, rowH = 20, colGap = 26, padT = 24;
    const blockW = COLS * cw;
    const w = padL + cols.length * (blockW + colGap) + 20;
    const rowsMeta = [
      { lab: "Eᵗᵒᵏ", get: (i) => Etok(i), tag: "identical" },
      { lab: "Eᵖᵒˢ", get: (i) => Epos(i), tag: "differ" },
      { lab: "h⁰", get: (i) => H0(i), tag: "differ", out: true },
    ];
    const h = padT + rowsMeta.length * (rowH + 12) + 24;
    const sMax = Math.max(...cols.flatMap((i) => [maxAbs([Etok(i)]), maxAbs([Epos(i)]), maxAbs([H0(i)])]));
    let s = "";
    // column headers (positions)
    cols.forEach((i, ci) => {
      const x = padL + ci * (blockW + colGap);
      s += txt(x + blockW / 2, 14, `“${ch}” @ pos ${i}`, "lz-cap");
    });
    rowsMeta.forEach((rm, ri) => {
      const y = padT + ri * (rowH + 12);
      s += txt(padL - 12, y + rowH / 2 + 3, rm.lab, rm.out ? "lz-slab on" : "lz-slab", "end");
      s += txt(padL - 12, y + rowH / 2 + 15, rm.tag, rm.tag === "identical" ? "lz-subslab ok" : "lz-subslab diff", "end");
      cols.forEach((i, ci) => {
        const x = padL + ci * (blockW + colGap);
        const vec = rm.get(i);
        for (let c = 0; c < COLS; c++)
          s += `<rect x="${x + c * cw}" y="${y}" width="${cw - 0.5}" height="${rowH}" fill="${colorDiverging(vec[c], sMax)}" data-tip="pos ${i} · dim ${c} = ${vec[c].toFixed(3)}"/>`;
        if (rm.out) s += `<rect x="${x}" y="${y}" width="${blockW}" height="${rowH}" class="lz-outframe"/>`;
      });
    });

    return note(
      `The character <b>“${ch}”</b> appears at ${ps.length} positions (${ps.join(", ")}). Its <b>Eᵗᵒᵏ ` +
      `rows are identical</b> — same character, same lookup. But <b>Eᵖᵒˢ differs</b>, so the <b>h⁰ rows ` +
      `come out different</b>. That difference is the only thing telling the rest of the network these are ` +
      `different positions.`
    ) + svg(w, h, s, "same char · different position → different h⁰");
  }

  /* helpers */
  function section(n, title) {
    return `<div class="lz-section"><span class="lz-secnum">${n}</span>${escapeHtml(title)}</div>`;
  }
  function note(html) { return `<div class="lz-note">${html}</div>`; }
  function txt(x, y, str, cls, anchor = "middle") {
    return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${escapeHtml(str)}</text>`;
  }
  function arrow(x0, y0, x1, y1, label) {
    let s = `<line x1="${x0}" y1="${y0}" x2="${x1 - 7}" y2="${y1}" class="lz-arrow"/>`;
    s += `<path d="M ${x1 - 7} ${y1 - 4} L ${x1} ${y1} L ${x1 - 7} ${y1 + 4} Z" class="lz-arrowhead"/>`;
    if (label) s += `<text x="${(x0 + x1) / 2}" y="${y0 - 8}" class="lz-arrowlabel" text-anchor="middle">${escapeHtml(label)}</text>`;
    return s;
  }
  function box(x, y, w, h, label, cls) {
    return `<rect x="${x}" y="${y}" width="${w}" height="${h}" class="lz-box ${cls}"/>` +
      `<text x="${x + w / 2}" y="${y + h / 2 + 4}" class="lz-boxtext" text-anchor="middle">${escapeHtml(label)}</text>`;
  }
  function line(x0, y0, x1, y1) {
    return `<line x1="${x0}" y1="${y0}" x2="${x1 - 6}" y2="${y1}" class="lz-arrow"/>` +
      `<path d="M ${x1 - 6} ${y1 - 4} L ${x1} ${y1} L ${x1 - 6} ${y1 + 4} Z" class="lz-arrowhead"/>`;
  }
  function svg(w, h, inner, cap) {
    return `<div class="lz-diagram">${cap ? `<div class="lz-diagram-cap">${escapeHtml(cap)}</div>` : ""}<div class="lz-svgwrap"><svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid meet">${inner}</svg></div></div>`;
  }

  draw();
}
