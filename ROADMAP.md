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
| 14 | `blocks` | **stacking** — the stream deepening across all 3 blocks; attention by depth | ✅ |
| 15 | `attention` | **atlas** — all 12 heads (3 layers × 4) attending at once, per query token | ✅ |
| 16 | `recap` | whole-architecture diagram | — |
| 17 | `output` | final LN → logits → softmax → **predicted character** | ✅ |
| 18 | `generate` | **autoregressive loop** + **argmax/sample/temperature** decoding | ✅ |
| 19 | `kv_cache` | The KV Cache (Memory bottleneck) | ✅ |
| 20 | `moe` | Mixture of Experts (Sparsity and Routing) | ✅ |
| 21 | `flash_attention` | FlashAttention (SRAM vs HBM optimization) | ✅ |
| 22 | `speculative_decoding` | Speculative Decoding (Draft and Target models) | ✅ |
| 23 | `gqa` | Grouped-Query Attention (Shared KV heads) | ✅ |
| 24 | `paged_attention` | PagedAttention (VRAM fragmentation & Block Tables) | ✅ |
| 25 | `ring_attention` | Ring Attention (Context scaling across GPUs) | ✅ |
| 26 | `mla` | Multi-Head Latent Attention (DeepSeek's KV Cache compression) | ✅ |

Plus a contextual **minimap** (`web/render/minimap.js`): whole network / parallel head split /
single-head zoom, shown per lesson.

### Foundations in place
- **Vite & ES Modules** — strict, scalable module imports (no global namespace pollution).
- **Standardized UI System** — over 300 hardcoded SVG colors extracted into CSS variables.
- `web/forward.js` — one shared, correct forward pass (all 3 blocks + output).
- `web/render/lesson-kit.js` — shared SVG helpers + palette, so new lessons don't re-declare them.
- `web/render/tensors.js` — colour maps, heatmaps, bars, hover tooltip.
- The whole thing on one **vermilion "instrument plate"** design system.

## Next up

**Short term**
- [ ] **Curriculum Expansion:** Add lessons for Rotary Positional Embeddings (RoPE), Quantization (GGUF/AWQ), and Low-Rank Adaptation (LoRA).
- [ ] **Interactive Visuals:** Convert passive SVG animations (MoE, FlashAttention) into interactive sandboxes (e.g. draggable sliders).
- [ ] **Global Dark Mode:** Add a toggle to switch the entire application into dark mode.
- [ ] **Voice pass, finish it.** Lessons 4–15 were toned down from the initial breathless draft; do a final consistency read against the measured voice of lessons 1–2.

**Medium term**
- [x] ~~**Autoregressive generation** — the `generate` lesson: predict-append-repeat, with
      per-character confidence and greedy-loop detection.~~
- [x] ~~**Temperature / sampling** — argmax↔sample toggle + temperature slider (seeded), with the
      next-char distribution reshaping live as you drag T.~~
- [x] ~~**Blocks 1 & 2** — the `blocks` lesson: the residual stream deepening across all 3 blocks
      and how attention shifts with depth.~~
- [ ] **Temperature / sampling** controls on the output distribution (argmax ↔ sample, seeded).
- [ ] A per-lesson **"show the exact tensor"** drawer (the parked `player/tensors.js` heatmaps) for
      readers who do want every number.

**Longer term / parked**
- [ ] Re-enable the **fine-grained micro-step player** (`web/player/*`, fully working, ~117 atomic
      steps for a single forward pass) as an optional "expert mode" alongside lesson mode.
- [x] ~~Attention-pattern explorer — the `attention` atlas: pick a query token, see all 12 heads
      (3 layers × 4) attend at once.~~
- [ ] Retrain hooks: a bigger context / cleaner corpus without changing the visualizer.

## Known issues / debt
- Lessons 1–11 duplicate math and helpers (migration item above).
- `web/player/*` is not loaded by `index.html`; it's kept for the expert-mode item.
- Blocks 1 and 2 are shown only as collapsed boxes in the minimap/recap.
