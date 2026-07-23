/* generate.js — autoregressive generation with decoding controls.
   Run the forward pass in a loop, feeding the model its own output.
   Choose how the next character is picked: greedy argmax, or sampling
   from the (temperature-scaled) distribution. Sampling is seeded, so a
   given setting reproduces exactly. Reads ctx.weights. */

export function renderGenerateLesson(host, ctx) {
  const W = ctx.weights;
  const CTX = 64, MAX = 120, SEED = 1337;
  const promptTokens = ctx.tokens.slice();
  const promptText = ctx.chars.join("");
  const vocabLabels = VOCAB.split("").map(glyph);
  const softmax = window._fwdMath.softmax;

  let mode = "argmax", temp = 0.8, rng = mulberry32(SEED), gen = [];

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function context() {
    const c = promptTokens.concat(gen.map((g) => g.id));
    return c.length > CTX ? c.slice(-CTX) : c;
  }
  function sampleIdx(probs) {
    const r = rng();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) { cum += probs[i]; if (r < cum) return i; }
    return probs.length - 1;
  }
  function nextProbs(F) {
    return mode === "sample" ? softmax(F.lastLogits.map((v) => v / temp)) : F.probs;
  }
  function stepOnce() {
    const F = computeForward(W, context());
    const probs = nextProbs(F);
    const id = mode === "argmax" ? F.best : sampleIdx(probs);
    gen.push({ id, char: VOCAB[id], prob: probs[id] });
  }
  function generate(n) { for (let i = 0; i < n && gen.length < MAX; i++) stepOnce(); repaint(); }
  function resetGen() { gen = []; rng = mulberry32(SEED); }

  function detectLoop() {
    const s = gen.map((g) => g.char).join("");
    for (let p = 1; p <= 4; p++)
      if (s.length >= p * 3 && s.slice(-p * 3) === s.slice(-p * 3).slice(0, p).repeat(3)) return true;
    return false;
  }

  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
  });
  host.addEventListener("mouseleave", hideTip);

  host.innerHTML =
    section("1", "the loop — predict, append, repeat") +
    note("Generation is just the forward pass, run over and over: take the predicted character, <b>append it to the input</b>, and feed the longer sequence straight back in. Every character is conditioned on everything before it — including the ones the model just wrote.") +
    section("2", "how to pick the next character") +
    note("<b>Argmax</b> always takes the single most likely character — deterministic, but it tends to get stuck repeating. <b>Sampling</b> instead draws a character at random <i>in proportion to</i> the probabilities. <b>Temperature</b> T rescales the logits before softmax (logits ÷ T): T&lt;1 sharpens toward the top choice, T&gt;1 flattens toward random, and T→0 becomes argmax again. Watch the distribution below reshape as you drag it.") +
    controls() +
    `<div id="genOut"></div>`;

  wire();
  repaint();

  function controls() {
    return `<div class="gen-controls">
      <div class="seg" id="modeSeg">
        <button class="seg-btn on" data-mode="argmax">argmax</button>
        <button class="seg-btn" data-mode="sample">sample</button>
      </div>
      <div class="gen-temp off" id="tempWrap">
        <label>temperature</label>
        <input type="range" id="temp" min="0.1" max="1.5" step="0.05" value="0.8" disabled>
        <span id="tempVal">0.80</span>
      </div>
      <div class="seg">
        <button class="gen-btn" data-gen="1">+1</button>
        <button class="gen-btn" data-gen="10">+10</button>
        <button class="gen-btn" data-gen="40">+40</button>
        <button class="gen-btn ghost" data-gen="reset">reset</button>
      </div>
    </div>`;
  }

  function wire() {
    host.querySelectorAll("[data-mode]").forEach((b) =>
      b.addEventListener("click", () => setMode(b.dataset.mode))
    );
    host.querySelectorAll("[data-gen]").forEach((b) =>
      b.addEventListener("click", () => {
        if (b.dataset.gen === "reset") { resetGen(); repaint(); }
        else generate(+b.dataset.gen);
      })
    );
    const t = host.querySelector("#temp");
    t.addEventListener("input", (e) => {
      temp = +e.target.value;
      host.querySelector("#tempVal").textContent = temp.toFixed(2);
      repaint(); // live: distribution reshapes, future steps use new temp
    });
  }

  function setMode(m) {
    mode = m;
    host.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
    const t = host.querySelector("#temp");
    t.disabled = m === "argmax";
    host.querySelector("#tempWrap").classList.toggle("off", m === "argmax");
    resetGen();
    repaint();
  }

  function repaint() { host.querySelector("#genOut").innerHTML = outHTML(); }

  function outHTML() {
    const ctxLen = context().length;
    const F = computeForward(W, context());
    const probs = nextProbs(F);

    let text = `<span class="gen-prompt">${escapeHtml(promptText)}</span>`;
    gen.forEach((g, i) => {
      text += `<span class="gen-ch" data-tip="char ${i + 1}: “${escapeHtml(glyph(g.char))}” · p = ${(g.prob * 100).toFixed(1)}%">${escapeHtml(g.char === "\n" ? "⏎" : g.char)}</span>`;
    });
    text += `<span class="gen-cursor"></span>`;

    const strip = gen.length
      ? '<div class="gen-strip">' + gen.map((g) =>
          `<span class="gen-bar" data-tip="“${escapeHtml(glyph(g.char))}” · p = ${(g.prob * 100).toFixed(1)}%"><span style="height:${(g.prob * 100).toFixed(0)}%"></span></span>`
        ).join("") + "</div>"
      : "";

    const meta = `<div class="gen-meta"><b>${mode}</b>${mode === "sample" ? ` · T = ${temp.toFixed(2)} · seed ${SEED}` : ""} · generated <b>${gen.length}</b> · context ${ctxLen}/${CTX}${ctxLen === CTX ? " (full — oldest dropped)" : ""}</div>`;

    const distCap = mode === "sample" ? `the distribution it samples from · T = ${temp.toFixed(2)}` : "the model's distribution · argmax takes the tallest";
    const dist = barsDiagram(probs, { labels: vocabLabels, highlight: F.best, cap: distCap });

    const loopMsg = mode === "argmax"
      ? (detectLoop()
          ? "It has settled into a <b>repeating loop</b> — the classic failure of greedy decoding. Switch to <b>sample</b> and it breaks out."
          : "Greedy decoding is deterministic — same prompt, same output. Keep going and it will usually fall into a repeating loop.")
      : (detectLoop()
          ? "Even sampling can loop at very low temperature — nudge T up for more variety."
          : "Sampling injects randomness, so repeats are far less likely. Raise T for wilder text, lower it to stay closer to the model's confident guesses.");

    return meta +
      `<div class="gen-text">${text}</div>` + strip +
      note("Hover any generated character or bar for the probability it was given.") +
      section("3", "the next character's distribution") + dist + note(loopMsg) +
      `<div class="gen-window">${escapeHtml(context().map(decode).join(""))}<span class="gen-window-cap">↑ the ${ctxLen} characters currently in the context window</span></div>`;
  }
}
