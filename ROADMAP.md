# nano·mind — roadmap

Where the project is and where it's going. See `README.md` for how to run it and
`docs/DESIGN.md` for the architecture and the original design rationale.

## Where we are

**Lesson mode** is the live experience: a sequence of deep, hand-built teaching chapters,
one operation each, on an infinitely-scrolling page. You type a prompt, and a single correct
forward pass (`web/forward.js`, validated bit-for-bit against the reference engine) feeds every
lesson its real intermediate tensors.

### Lessons shipped (`web/lessons/`)

| # | lesson | covers | interactive |
|---|---|---|---|
| 1 | `embedding` | token embedding as one-hot × table | ✅ |
| 2 | `position` | positional embedding, the sum → h⁰, learned-vs-sinusoidal, how it's learned | ✅ |
| 3 | `layernorm` | LayerNorm 1 (mean/var → normalize → scale/shift) | ✅ |
| 4 | `qkv` | Q, K, V projections (head 0) | ✅ |
| 5 | `scores` | Q·Kᵀ → scale → causal mask | ✅ |
| 6 | `softmax` | attention weights | ✅ |
| 7 | `values` | O = A·V | ✅ |
| 8 | `concat` | join the 4 heads | ✅ |
| 9 | `proj` | attention output projection | ✅ |
| 10 | `resid1` | residual add (attention) | ✅ |
| 11 | `ln2` | LayerNorm 2 | ✅ |
| 12 | `ffwd` | feed-forward 64→256→ReLU→64, with dead-neuron sparsity | ✅ |
| 13 | `resid2` | residual add (FFN) → end of block 0 | ✅ |
| 14 | `recap` | whole-architecture diagram | — |
| 15 | `output` | final LN → logits → softmax → **predicted character** | ✅ |
| 16 | `generate` | **autoregressive loop** — predict, append, repeat; writes text | ✅ |

Plus a contextual **minimap** (`web/render/minimap.js`): whole network / parallel head split /
single-head zoom, shown per lesson.

### Foundations in place
- `web/forward.js` — one shared, correct forward pass (all 3 blocks + output).
- `web/render/lesson-kit.js` — shared SVG helpers + palette, so new lessons don't re-declare them.
- `web/render/tensors.js` — colour maps, heatmaps, bars, hover tooltip.
- The whole thing on one **vermilion "instrument plate"** design system.

## Next up

**Short term**
- [ ] **Migrate lessons 1–11 onto `forward.js` + `lesson-kit.js`.** They still recompute their own
      intermediates and re-declare local helpers. Now that `ctx.fwd` exists and is validated, delete
      the per-lesson math and local `section/note/txt/svg`. Biggest remaining code cleanup.
- [ ] **Voice pass, finish it.** Lessons 4–15 were toned down from the initial breathless draft;
      do a final consistency read against the measured voice of lessons 1–2.
- [ ] Move the minimap from the page bottom to a sticky rail (or up under the equation) so the
      "where am I" context is visible while reading, not only after scrolling.

**Medium term**
- [x] ~~**Autoregressive generation** — the `generate` lesson: drive the predict-append-repeat loop,
      watch it write, with per-character confidence and greedy-loop detection.~~
- [ ] **Temperature / sampling** on top of `generate` — argmax ↔ sample, seeded, to break the loops.
- [ ] **Blocks 1 & 2:** a short "same structure, new weights — here's what changes" lesson rather
      than 13 more identical screens.
- [ ] **Temperature / sampling** controls on the output distribution (argmax ↔ sample, seeded).
- [ ] A per-lesson **"show the exact tensor"** drawer (the parked `player/tensors.js` heatmaps) for
      readers who do want every number.

**Longer term / parked**
- [ ] Re-enable the **fine-grained micro-step player** (`web/player/*`, fully working, ~117 atomic
      steps for a single forward pass) as an optional "expert mode" alongside lesson mode.
- [ ] Attention-pattern explorer: pick a token, watch what every head attends to across layers.
- [ ] Retrain hooks: a bigger context / cleaner corpus without changing the visualizer.

## Known issues / debt
- Lessons 1–11 duplicate math and helpers (migration item above).
- `web/player/*` is not loaded by `index.html`; it's kept for the expert-mode item.
- Blocks 1 and 2 are shown only as collapsed boxes in the minimap/recap.
