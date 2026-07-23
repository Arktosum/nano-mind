# nano·mind

An interactive, rigorous, in-browser visualization of a **decoder-only transformer** —
walking through the forward pass **one operation at a time**, with the real equations and the
real numbers, on an actual trained model.

Type a fragment of text and step through what happens to it: every lookup, matrix multiply,
softmax, and residual add, each shown as an operation diagram you can read, with a plain-language
"why" and the exact tensors underneath. Everything runs in the browser — no server, no build step.

The model is a genuinely tiny GPT trained on *Alice in Wonderland* (character-level), deliberately
small so **every number fits on screen**: 75-char vocab, 64 dimensions, 4 heads, 3 layers, context
of 64. It's a real trained network, not random weights.

---

## Run it

The app uses Vite as a development server to handle ES Modules and Hot-Module Reloading:

```bash
cd web
npm install
npm run dev
# open http://localhost:5173
```

Type a prompt (e.g. `Hi there`), hit **trace**, and use ← / → or the on-screen controls to move
between lessons.

---

## How it's built

Two independent sides that meet at one file, `weights.json`:

```
train/  ──trains a tiny GPT, exports──▶  web/weights.json  ──consumed by──▶  web/  (the visualizer)
```

- **`train/`** — the PyTorch side. `model.py` defines the decoder-only GPT; `data.py` prepares the
  char-level dataset from `alice.txt`; `train.py` trains it and exports every weight tensor to
  `../web/weights.json`. You only need this if you want to retrain; the trained weights are already
  committed.
- **`web/`** — the browser app. Built with ES Modules and bundled by Vite. Pure vanilla JS, hand-written math, no heavy dependencies. Equations are
  rendered in plain HTML/CSS (no KaTeX), tensors as inline SVG / canvas.

### `web/` layout

```
web/
├── package.json          npm dependencies (Vite)
├── index.html            app shell: landing + lesson workbench
├── style.css             the "engineering instrument plate" visual system
├── app.js                controller + entry point (imports all lessons)
├── vocab.js              the 75-char vocabulary + tokenizer (encode/decode/glyph)
├── forward.js            one correct forward pass, computed once → ctx.fwd
├── weights.json          the trained weights (exported from train/)
├── render/
│   ├── tensors.js        colour maps, heatmaps, bars, hover tooltip
│   ├── lesson-kit.js     shared SVG helpers + palette used by lessons
│   └── minimap.js        contextual architecture map (network / heads / one head)
├── lessons/              one operation per file (see ROADMAP.md for the full list)
│   ├── embedding.js  position.js  layernorm.js  qkv.js  scores.js  softmax.js
│   ├── values.js  concat.js  proj.js  resid1.js  ln2.js  ffwd.js  resid2.js
│   └── recap.js  output.js
└── player/               PARKED — the full ~117-op micro-step player (not wired)
    ├── math.js  engine.js  trace.js  flow.js  minimap.js
```

Lessons read their real numbers from `ctx.fwd` (built by `forward.js`, validated bit-for-bit
against `player/engine.js`) and draw with the shared helpers in `render/lesson-kit.js`.

### The two "modes"

nano·mind started as a **micro-step player** — one screen that steps through all ~117 atomic
operations of a single forward pass (that code lives in `web/player/`, fully working but parked).
It then pivoted to the current **lesson mode**: a smaller number of deep, hand-crafted teaching
chapters, one operation each, on an infinitely-scrolling page with bespoke interactive diagrams.

Lesson mode is what `index.html` loads today. The player is kept because its recorder + renderers
are the foundation for re-enabling fine-grained stepping later (see `ROADMAP.md`).

### Adding a lesson

Lessons are registered in `web/app.js` in the `LESSONS` array. Each lesson is an ES Module that exports a `render(host)` function along with metadata (`crumb`, `title`, `eq`, `why`). Write the render function in a new `web/lessons/*.js`, `import` it in `app.js`, and push an entry into `LESSONS`. Prev/next navigation and the header/equation/why cards are handled for you.

---

## The model, precisely

| | |
|---|---|
| Vocabulary | 75 characters (char-level, no BPE) |
| `d_model` | 64 |
| Heads | 4 (head size 16) |
| Layers | 3 |
| Context (`block_size`) | 64 |
| FFN hidden | 256 (4 × d_model) |
| Norm / nonlinearity | LayerNorm (pre-norm) / ReLU |
| Positional encoding | **learned** (not sinusoidal) |
| Params | ~0.2 M |

See `docs/DESIGN.md` for the full architecture, the atomic step ladder, and the original design
rationale, and `ROADMAP.md` for what's done and what's next.
