/* ffwd.js — the feed-forward network (block 0), on real activations.
   Each token is processed independently: 64 → 256, ReLU (which kills
   roughly half the hidden units), 256 → 64. Reads ctx.fwd. */

export function renderFfwdLesson(host, ctx) {
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
    const ln2 = b0.ln2[active], up = b0.ffUp[active], act = b0.ffAct[active], down = b0.ffDown[active];
    const dead = act.filter((v) => v === 0).length;

    host.innerHTML =
      chipRow(ctx.tokens, ctx.chars, active, "your sequence — the FFN runs on each position on its own") +
      section("1", "input — the normalized stream (1×64)") +
      note("The FFN treats each token completely independently — no mixing across positions. This is one token's normalized vector, straight out of LayerNorm 2.") +
      vecDiagram(ln2, { label: "h̃′ₜ", sub: "1×64", cap: "" }) +
      section("2", "expand — project up to 256") +
      note("A learned <b>64→256</b> layer widens the vector into a roomier space where the network can pull apart more feature combinations.") +
      vecDiagram(up, { label: "W_up·h̃′", sub: "1×256", cw: 4, cap: "u = h̃′ · W_upᵀ + b_up" }) +
      section("3", "ReLU — throw away the negatives") +
      note(`ReLU sets every negative value to 0. Here it zeroes <b>${dead} of 256</b> units (~${Math.round((dead / 256) * 100)}%) for this token — the dark cells are “dead”. This sparsity is where the non-linearity lives: only a subset of features survive to describe this token.`) +
      vecDiagram(act, { label: "ReLU(u)", sub: "1×256", cw: 4, max: maxAbs([up]), dead: true, cap: "dark = zeroed" }) +
      section("4", "contract — project back to 64") +
      note("A second learned <b>256→64</b> layer compresses the surviving features back to the residual-stream width. This is the FFN's contribution, ready to be added back in the next step.") +
      vecDiagram(down, { label: "f_t", sub: "1×64", cap: "f = ReLU(u) · W_downᵀ + b_down" });

    host.querySelectorAll(".lz-chip").forEach((c) =>
      c.addEventListener("click", () => { active = +c.dataset.i; draw(); })
    );
  }

  draw();
}
