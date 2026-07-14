/* output.js — the output stage: turn the final residual stream into
   an actual next-character prediction. This is the payoff the landing
   page promises. Reads ctx.fwd (all 3 blocks already run). */

function renderOutputLesson(host, ctx) {
  const F = ctx.fwd;
  const T = F.T;
  const last = T - 1;
  const bestChar = VOCAB[F.best];

  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
  });
  host.addEventListener("mouseleave", hideTip);

  const vocabLabels = VOCAB.split("").map(glyph);
  const z = F.znorm[last];
  const logits = F.logits[last];
  const probs = F.probs;

  // top-5 for the callout
  const ranked = probs.map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 5);

  host.innerHTML =
    note(`After 3 blocks, the residual stream for the <b>last</b> token — “${escapeHtml(glyph(ctx.chars[last]))}” at position ${last} — now holds everything the network figured out. Only this row matters for predicting what comes next.`) +
    section("1", "final LayerNorm") +
    note("One last normalization of the last token's 64-dimensional vector before it's read out into vocabulary scores.") +
    vecDiagram(z, { label: "z", sub: "1×64", cap: "z = LayerNorm_f(h³_last)" }) +
    section("2", "language head → 75 logits") +
    note("A learned <b>64→75</b> layer scores every possible next character. One number per vocabulary symbol — raw, unnormalized preferences.") +
    barsDiagram(logits, { labels: vocabLabels, signed: true, highlight: F.best, cap: "ℓ = z · W_lmᵀ + b_lm   (per-character logits)" }) +
    section("3", "softmax → probabilities") +
    note("Exponentiate and normalize the 75 logits into a distribution that sums to 1. The tallest bar is the model's pick.") +
    barsDiagram(probs, { labels: vocabLabels, highlight: F.best, cap: "p = softmax(ℓ)" }) +
    section("4", "the prediction") +
    prediction(ranked) +
    note("With greedy decoding we take the single most probable character, append it to the input, and run the whole forward pass again on the longer sequence. That loop — one character at a time — is how the model generates text.");

  function prediction(top) {
    const rows = top.map(([i, p], r) =>
      `<div class="op-rank${r === 0 ? " win" : ""}"><span class="op-glyph">${escapeHtml(glyph(VOCAB[i]))}</span>` +
      `<span class="op-bar"><span style="width:${(p * 100).toFixed(1)}%"></span></span>` +
      `<span class="op-pct">${(p * 100).toFixed(1)}%</span></div>`
    ).join("");
    return `<div class="op-predict"><div class="op-big">next → <b>“${escapeHtml(glyph(bestChar))}”</b></div>` +
      `<div class="op-top">${rows}</div></div>`;
  }
}
