/* generate.js — autoregressive generation: feed the model its own
   output, one character at a time. Greedy (argmax) decoding, so it's
   deterministic — which also lets you watch it fall into loops, a real
   property of greedy decoding on a tiny model. Reads ctx.weights. */

function renderGenerateLesson(host, ctx) {
  const W = ctx.weights;
  const CTX = 64;
  const MAX = 80;
  const promptTokens = ctx.tokens.slice();
  const promptText = ctx.chars.join("");
  let gen = []; // [{ id, char, prob }]

  function context() {
    const c = promptTokens.concat(gen.map((g) => g.id));
    return c.length > CTX ? c.slice(-CTX) : c;
  }

  function stepOnce() {
    const F = computeForward(W, context());
    gen.push({ id: F.best, char: VOCAB[F.best], prob: F.probs[F.best] });
  }
  function generate(n) {
    for (let i = 0; i < n && gen.length < MAX; i++) stepOnce();
    draw();
  }

  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
  });
  host.addEventListener("mouseleave", hideTip);
  host.addEventListener("click", (e) => {
    const b = e.target.closest("[data-gen]");
    if (!b) return;
    if (b.dataset.gen === "reset") { gen = []; draw(); }
    else generate(+b.dataset.gen);
  });

  // detect a repeating tail (period 1..4 repeated ≥3×) — greedy's classic loop
  function detectLoop() {
    const s = gen.map((g) => g.char).join("");
    for (let p = 1; p <= 4; p++) {
      if (s.length >= p * 3) {
        const tail = s.slice(-p * 3);
        if (tail === tail.slice(0, p).repeat(3)) return true;
      }
    }
    return false;
  }

  function draw() {
    hideTip();
    const ctxLen = context().length;

    // flowing text: prompt (muted) + generated (accent)
    let text = `<span class="gen-prompt">${escapeHtml(promptText)}</span>`;
    gen.forEach((g, i) => {
      text += `<span class="gen-ch" data-tip="char ${i + 1}: “${escapeHtml(glyph(g.char))}” · p = ${(g.prob * 100).toFixed(1)}%">${escapeHtml(g.char === "\n" ? "⏎" : g.char)}</span>`;
    });
    text += `<span class="gen-cursor"></span>`;

    // confidence strip
    let strip = "";
    if (gen.length) {
      strip = '<div class="gen-strip">' +
        gen.map((g, i) =>
          `<span class="gen-bar" data-tip="“${escapeHtml(glyph(g.char))}” · p = ${(g.prob * 100).toFixed(1)}%"><span style="height:${(g.prob * 100).toFixed(0)}%"></span></span>`
        ).join("") + "</div>";
    }

    // what it would pick next, from the current context
    const F = computeForward(W, context());
    const top = F.probs.map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 5);
    const nextList = top.map(([i, p], r) =>
      `<div class="op-rank${r === 0 ? " win" : ""}"><span class="op-glyph">${escapeHtml(glyph(VOCAB[i]))}</span>` +
      `<span class="op-bar"><span style="width:${(p * 100).toFixed(1)}%"></span></span>` +
      `<span class="op-pct">${(p * 100).toFixed(1)}%</span></div>`
    ).join("");

    const looping = detectLoop();

    host.innerHTML =
      section("1", "the loop — predict, append, repeat") +
      note("Generation is just the forward pass, run over and over. Take the model's predicted character, <b>append it to the input</b>, and feed the longer sequence straight back in. Each new character is conditioned on everything before it — including the characters the model itself just wrote.") +
      `<div class="gen-controls">
         <button class="gen-btn" data-gen="1">+1 char</button>
         <button class="gen-btn" data-gen="10">+10</button>
         <button class="gen-btn" data-gen="40">+40</button>
         <button class="gen-btn ghost" data-gen="reset">reset</button>
         <span class="gen-meta">generated <b>${gen.length}</b> · context ${ctxLen}/${CTX}${ctxLen === CTX ? " (full — oldest dropped)" : ""}</span>
       </div>` +
      `<div class="gen-text">${text}</div>` +
      strip +
      note("Hover any generated character (or its bar) for the probability the model gave it. The bars are its <b>confidence</b> at each step — tall where it was sure, short where it hesitated.") +
      section("2", "what it picks next") +
      `<div class="op-predict"><div class="op-big">next → <b>“${escapeHtml(glyph(VOCAB[F.best]))}”</b></div><div class="op-top">${nextList}</div></div>` +
      note(
        "It always takes the single most probable character (greedy decoding), so generation is fully deterministic — same prompt, same output every time." +
        (looping ? " Notice it has settled into a <b>repeating loop</b> — a classic failure of greedy decoding on a small model. Sampling with a temperature (roadmap) breaks these loops." :
          " Keep going and it will often fall into a repeating loop — a classic quirk of greedy decoding that sampling fixes.")
      ) +
      `<div class="gen-window">${escapeHtml(context().map(decode).join(""))}<span class="gen-window-cap">↑ the ${ctxLen} characters currently in the model's context window</span></div>`;
  }

  draw();
}
