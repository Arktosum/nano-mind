/* resid2.js — the second residual add (block 0), on real vectors.
   h'' = h' + f, completing block 0. Reads ctx.fwd. */

export function renderResid2Lesson(host, ctx) {
  const b0 = ctx.fwd.blocks[0];
  const T = ctx.fwd.T;
  let active = T - 1;

  host.addEventListener("mousemove", (e) => {
    const t = e.target;
    if (t.dataset && t.dataset.tip) showTip(t.dataset.tip, e.clientX, e.clientY);
    else hideTip();
  });
  host.addEventListener("mouseleave", hideTip);

  function draw() {
    hideTip();
    const hp = b0.h1[active];      // stream after attention
    const f = b0.ffDown[active];   // FFN delta
    const hpp = b0.hOut[active];   // block-0 output
    const sMax = Math.max(maxAbs([hp]), maxAbs([f]), maxAbs([hpp]));

    host.innerHTML =
      chipRow(ctx.tokens, ctx.chars, active, "your sequence — click a position") +
      section("1", "add the FFN's contribution back") +
      note("Exactly like the first residual add: the FFN output is a correction, added onto the stream rather than replacing it. The skip connection keeps information and gradients flowing straight through.") +
      sumDiagram([
        { vec: hp, label: "h′ₜ", sub: "after attention", max: sMax,
          tip: (c) => `dim ${c}: ${hp[c].toFixed(3)} + ${f[c].toFixed(3)} = ${hpp[c].toFixed(3)}`, op: "+" },
        { vec: f, label: "fₜ", sub: "FFN output", max: sMax,
          tip: (c) => `dim ${c}: ${hp[c].toFixed(3)} + ${f[c].toFixed(3)} = ${hpp[c].toFixed(3)}`, op: "=" },
        { vec: hpp, label: "h″ₜ", sub: "block 0 output", max: sMax, out: true,
          tip: (c) => `dim ${c}: ${hp[c].toFixed(3)} + ${f[c].toFixed(3)} = ${hpp[c].toFixed(3)}` },
      ], "h″ₜ = h′ₜ + fₜ") +
      note("That completes <b>Transformer Block 0</b>. This h″ becomes the input to Block 1, which repeats the identical structure with its own weights.");

    host.querySelectorAll(".lz-chip").forEach((c) =>
      c.addEventListener("click", () => { active = +c.dataset.i; draw(); })
    );
  }

  draw();
}
