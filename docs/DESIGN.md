# Nano-Mind — Design Document

> An interactive, rigorous, in-browser visualization of a **decoder-only transformer**,
> stepping through *every operation* of the forward pass with the exact equations and
> the exact numbers, on a **real trained model**.

---

## 1. Vision & guiding principles

The user types a prompt and presses **Step**. Each press advances the computation by one
*atomic operation* — a matmul, a scale, a mask, a softmax, a residual add — and the screen
shows three things locked together:

1. **The equation** for that operation, rendered as real math, with the term currently being
   computed highlighted.
2. **The actual numbers** flowing through it — matrices as heatmaps, vectors as colored rows,
   distributions as bar charts — all hoverable down to the exact float.
3. **A prose explanation** of *what* is happening and *why* it exists in the architecture.

A persistent **minimap** shows the whole architecture so you never lose the forest for the
trees, and you can click any node to jump there.

Two non-negotiable principles:

- **Nothing is hidden.** Every intermediate tensor is inspectable. No framework abstracts the
  math away — it is hand-written nested loops you can read (`math.js`).
- **You can always zoom out.** The minimap + residual-stream spine keep the full picture visible
  while you drill into any single number.

Design decisions already locked:

- **Model:** reuse the existing trained *Alice in Wonderland* char model (Section 2).
- **Step grain:** **fine-grained** — every op is its own step, with its own equation.
- **Stack:** vanilla JS + [KaTeX](https://katex.org/) for math, Canvas/SVG for tensors, no build step.

---

## 2. The model we visualize

A real decoder-only GPT trained on `alice.txt` (147,988 chars), exported to `weights.json`
(3.7 MB) and executed in pure JS. Deliberately tiny so **every number fits on screen.**

| Hyperparameter | Symbol | Value |
|---|---|---|
| Vocabulary (chars) | `vocab_size` | **75** |
| Embedding dim | `d_model` (`N_EMBD`) | **64** |
| Attention heads | `n_head` | **4** |
| Head size | `d_k = d_model / n_head` | **16** |
| Transformer blocks | `n_layer` | **3** |
| Context window | `block_size` | **64** |
| FFN hidden dim | `4 · d_model` | **256** |
| Nonlinearities | — | ReLU (FFN), softmax (attn + output) |
| Norm | — | LayerNorm, **pre-norm** placement |
| Dropout | — | 0 (deterministic — critical for reproducible viz) |

Character-level tokenizer (no BPE) — a gift for visualization: every token *is* a visible
character, so attention axes are labeled with real letters.

### 2.1 Weight tensors (keys in `weights.json`)

| Key pattern | Shape | Role |
|---|---|---|
| `token_embedding_table.weight` | 75 × 64 | token → vector lookup |
| `position_embedding_table.weight` | 64 × 64 | position → vector lookup |
| `blocks.{ℓ}.ln1.{weight,bias}` | 64 | pre-attention LayerNorm |
| `blocks.{ℓ}.sa.heads.{h}.{query,key,value}.weight` | 16 × 64 | per-head Q/K/V (no bias) |
| `blocks.{ℓ}.sa.proj.{weight,bias}` | 64 × 64 | output projection |
| `blocks.{ℓ}.ln2.{weight,bias}` | 64 | pre-FFN LayerNorm |
| `blocks.{ℓ}.ffwd.net.0.{weight,bias}` | 256 × 64 | FFN up-projection |
| `blocks.{ℓ}.ffwd.net.2.{weight,bias}` | 64 × 256 | FFN down-projection |
| `ln_f.{weight,bias}` | 64 | final LayerNorm |
| `lm_head.{weight,bias}` | 75 × 64 | hidden → logits |

> PyTorch `nn.Linear` stores weight as `[out_features, in_features]`, so `math.js:linear`
> computes `x @ Wᵀ + b`. The viz must label axes accordingly.

---

## 3. Architecture: record-then-replay

The key idea. We do **not** try to animate a live forward pass. Instead:

```
 prompt ──▶ instrumented forward pass ──▶ Trace (ordered list of Steps) ──▶ Player UI
             (engine emits every op)         (all intermediates recorded)     (scrub cursor)
```

Running the whole forward pass first and recording each intermediate tensor means the UI is a
pure function of `(trace, cursor)`. Stepping forward/back, jumping, and replaying are all just
moving an integer. No recomputation, no async surprises, trivially deterministic.

### 3.1 The `Step` record

```js
{
  id:        42,                       // global order in the trace
  group:     "layer-1/head-2",         // for the minimap grouping
  stage:     "attention.softmax",      // machine tag
  title:     "Softmax over scores",    // human title
  katex:     "A_{ij}=\\frac{e^{S_{ij}}}{\\sum_k e^{S_{ik}}}",
  highlight: "row",                    // which part of the equation/tensor to emphasize
  prose:     "Each row becomes a probability distribution over past tokens…",
  inputs:    [ { ref: "S", label: "scaled+masked scores" } ],  // provenance
  output:    { tensor: Float32Array-or-nested, shape: [T, T], viz: "heatmap",
               rowLabels: chars, colLabels: chars, domain: [0,1] },
  meta:      { layer: 1, head: 2 }
}
```

`viz` ∈ `gather | vector | matrix | heatmap | bars | dist`. The renderer switches on it.

### 3.2 Instrumenting the engine

`engine.js` already computes every intermediate in readable loops. We refactor it into a
`traceForward(text)` that, instead of only keeping the final result, **pushes a `Step` after
each operation**. `math.js` stays the numeric core; a thin recorder wraps it. The existing
`forward()` can be kept for a fast "just generate" path.

---

## 4. The fine-grained step ladder

Every item below is **one Step** (one press of ▶). `T` = current sequence length.

### Embedding stage
| # | Step | Equation | Output shape |
|---|---|---|---|
| E0 | Tokenize | `xₜ = stoi(cₜ)` | T ids |
| E1 | Token embedding lookup | `Eᵗᵒᵏₜ = W_E[xₜ]` | T × 64 |
| E2 | Positional embedding lookup | `Eᵖᵒˢₜ = W_P[t]` | T × 64 |
| E3 | Sum → residual stream `h⁰` | `h⁰ₜ = Eᵗᵒᵏₜ + Eᵖᵒˢₜ` | T × 64 |

### Per transformer block ℓ = 0,1,2 (input `hˡ`)
| # | Step | Equation |
|---|---|---|
| a | LayerNorm₁: mean & var | `μ = mean(hₜ), σ² = var(hₜ)` |
| b | LayerNorm₁: normalize + affine | `h̃ₜ = γ₁ ⊙ (hₜ−μ)/√(σ²+ε) + β₁` |
| — | **for each head h = 0..3:** | |
| c | Query projection | `Q⁽ʰ⁾ = h̃ · W_Q⁽ʰ⁾ᵀ` (T×16) |
| d | Key projection | `K⁽ʰ⁾ = h̃ · W_K⁽ʰ⁾ᵀ` (T×16) |
| e | Value projection | `V⁽ʰ⁾ = h̃ · W_V⁽ʰ⁾ᵀ` (T×16) |
| f | Attention scores | `S⁽ʰ⁾ = Q⁽ʰ⁾ K⁽ʰ⁾ᵀ` (T×T) |
| g | Scale | `S ← S / √d_k`   (d_k=16 ⇒ ×¼) |
| h | Causal mask | `Sᵢⱼ ← −∞  for j > i` |
| i | Softmax (rows) | `A⁽ʰ⁾ = softmax(S)` (T×T) |
| j | Weighted values | `O⁽ʰ⁾ = A⁽ʰ⁾ V⁽ʰ⁾` (T×16) |
| k | Concat heads | `O = [O⁽⁰⁾‖O⁽¹⁾‖O⁽²⁾‖O⁽³⁾]` (T×64) |
| l | Output projection | `O_proj = O · W_Oᵀ + b_O` |
| m | Residual add | `h' = hˡ + O_proj` |
| n | LayerNorm₂ | `h̃' = γ₂ ⊙ norm(h') + β₂` |
| o | FFN up + ReLU | `u = ReLU(h̃' · W₁ᵀ + b₁)` (T×256) |
| p | FFN down | `f = u · W₂ᵀ + b₂` (T×64) |
| q | Residual add | `hˡ⁺¹ = h' + f` |

### Output stage
| # | Step | Equation |
|---|---|---|
| F1 | Final LayerNorm | `z = γ_f ⊙ norm(h³) + β_f` |
| F2 | LM head → logits | `logits = z · W_lmᵀ + b_lm` (T×75) |
| F3 | Take last position | `ℓ = logits_{T−1}` (75) |
| F4 | Softmax → probabilities | `p = softmax(ℓ)` (75) |
| F5 | Select next token | `ĉ = argmax p` (or sample @ temperature) |
| F6 | Append & autoregress | `x ← x ⧺ ĉ`, loop to E1 |

Step count for a T-token prompt ≈ `4 + 3·(2 + 4·8 + 8) + 5` = **~137 steps** — traversable, and
the minimap + jump-to-node keep it navigable.

---

## 5. UI layout

```
┌───────────────────────────────────────────────────────────────────────────┐
│  nano-mind    [ prompt input........................ ]  [Run]  temp[▁▂▃]  ⓘ │
│  ◀ Prev   ▶ Next   ▷ Play   ⟲ Reset      Step 42 / 137   argmax ○ sample    │
├──────────────┬────────────────────────────────────────────────────────────┤
│              │  layer 1 · head 2 · Softmax over scores                      │
│   MINIMAP    │  ┌──────────────────────────────────────────────────────┐   │
│  (full arch, │  │  EQUATION:  A_ij = e^{S_ij} / Σ_k e^{S_ik}            │   │
│   current    │  └──────────────────────────────────────────────────────┘   │
│   node lit,  │  ┌──────────────────────────────────────────────────────┐   │
│   click to   │  │                                                      │   │
│   jump)      │  │      MAIN STAGE: the tensor for this step             │   │
│              │  │      (heatmap / vector row / bar chart / gather)      │   │
│  ── embed    │  │      hover any cell → exact value + provenance        │   │
│  ▸ block 0   │  │                                                      │   │
│  ▾ block 1   │  └──────────────────────────────────────────────────────┘   │
│     ln1      │  ┌──────────────────────────────────────────────────────┐   │
│     head0-3  │  │  WHY:  Each row of the attention matrix turns raw    │   │
│     ffn      │  │  affinities into a distribution over the tokens the  │   │
│  ▸ block 2   │  │  query is allowed to see (itself + earlier tokens)…  │   │
│  ── output   │  └──────────────────────────────────────────────────────┘   │
├──────────────┴────────────────────────────────────────────────────────────┤
│  RESIDUAL STREAM SPINE:  h  ▓▓▒░▓▒░▓  (T×64, updates as you step)  + genned │
└───────────────────────────────────────────────────────────────────────────┘
```

Panels:

- **Controls bar** — prompt, Run, Prev/Next/Play/Reset, step counter, temperature slider,
  argmax↔sample toggle.
- **Minimap (left rail)** — the whole architecture as a collapsible tree of nodes (embed →
  block 0/1/2 → output), each block expandable to ln1 / heads / ffn. Current step's node is
  highlighted; **click any node to jump** the cursor to its first step.
- **Main stage (center)** — three stacked regions: equation header (KaTeX, term-highlighted),
  the tensor visualization, and the prose "why" panel.
- **Residual-stream spine (bottom)** — a persistent thumbnail of the `T×64` hidden state that
  updates every step, plus the running generated text. This is the "through-line" so you always
  see the state the whole network is transforming.

---

## 6. Tensor visualization components

| `viz` | Used for | Rendering |
|---|---|---|
| `gather` | E1/E2 embedding lookups | the full table with the pulled row highlighted, arrow to the extracted vector |
| `vector` | any T×d row, LN stats | row of colored cells (diverging blue–white–red for signed values) |
| `matrix` | Q/K/V, projections, hidden states | Canvas heatmap, rows = tokens, cols = dims |
| `heatmap` | attention `S`, `A` (T×T) | Canvas, lower-triangular emphasized; masked cells hatched; axes = token chars |
| `bars` | logits, probabilities (75) | bar chart, x = vocab char, hover = value; top-k emphasized |
| `dist` | softmax before/after | paired bars showing raw scores → normalized probs |

Shared behaviors:

- **Hover a cell** → tooltip with exact float + indices + which inputs produced it (provenance).
- **Signed values** use a symmetric diverging scale (blue = negative, red = positive), auto-scaled
  per tensor with the domain shown in a legend.
- **Probabilities / attention** use a sequential scale on `[0,1]`.
- **Accessibility:** color is never the only channel — exact values are always available on hover,
  palette is colorblind-safe, and the equation term highlight is textual.

---

## 7. Interaction model

- **▶ / ◀** advance / rewind the cursor by one Step (keyboard **→ / ←**).
- **▷ Play** auto-advances at an adjustable rate (keyboard **space** = play/pause).
- **Minimap click** jumps the cursor to a node's first step.
- **Hover** any tensor cell for exact value + provenance.
- **Temperature slider + argmax/sample toggle** affect step F5 (selection). Sampling is seeded so
  a given seed is reproducible.
- **Autoregress:** after F6 the chosen token is appended and the trace extends; a **generation
  timeline** shows the produced characters, and you can step through subsequent forward passes.
- **Reset** clears back to the prompt.

---

## 8. Proposed file structure

```
index.html          entry; loads KaTeX (vendored) + modules
style.css           dark theme, extends the existing aesthetic
weights.json        trained weights (3.7 MB) — already present
vocab.js            the 75-char vocab + stoi/itos
math.js             pure linear algebra (already written — keep)
engine.js           traceForward(text) → Trace  (refactor of existing forward)
trace.js            Step schema + recorder helpers
render/
  minimap.js        architecture tree + jump
  stage.js          equation + tensor + prose switcher
  tensors.js        gather/vector/matrix/heatmap/bars/dist renderers
  spine.js          residual-stream thumbnail + generation timeline
app.js              wiring: state {trace, cursor}, controls, keyboard
vendor/katex/       local KaTeX (no CDN, works offline)
```

Everything served as static files (`python -m http.server` or any static host). No build step,
no dependencies beyond a vendored KaTeX. Runs entirely in the browser.

---

## 9. Build roadmap

1. **Scaffold & state** — index.html, controls bar, `{trace, cursor}` store, keyboard nav, KaTeX
   wired, empty stage.
2. **Trace recorder** — refactor `engine.js` into `traceForward`; emit the full Step ladder
   (Section 4); verify step count & shapes against the Python model.
3. **Tensor renderers** — implement the six `viz` types (Section 6) with hover/provenance.
4. **Minimap** — architecture tree, current-node highlight, jump-to.
5. **Residual spine + generation timeline** — the through-line and autoregression loop.
6. **Prose & equations** — author the explanation + KaTeX for all ~137 steps.
7. **Polish** — sampling/temperature, animations between steps, colorblind-safe legend, mobile/
   responsive pass, an "explain like the whole picture" overview mode.

---

## 10. Correctness anchors

Because this teaches, it must be *right*:

- Cross-check the JS `traceForward` against the PyTorch `model.py` on the same input (compare
  final logits to a few decimals). `train.py` / `model.py` remain the source of truth.
- Keep dropout = 0 and use argmax by default so the visualization is deterministic and
  reproducible.
- LayerNorm uses biased variance (`/N`, matching PyTorch) and `ε = 1e-5`, as in `math.js`.
- Attention scale is `1/√d_k = 1/4`; masked positions are exactly `−∞` before softmax.
```
