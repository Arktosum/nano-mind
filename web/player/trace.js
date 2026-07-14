/* trace.js — record-then-replay.
   Runs the forward pass and emits one Step per atomic operation
   (see DESIGN.md §4). Each Step carries:
     - eq   : the equation (HTML, no KaTeX)
     - why  : prose
     - flow : an SVG operation diagram spec (operands → operator → output)
     - viz  : the detailed, hoverable output tensor
   The UI is a pure function of (trace, cursor). Reuses math.js. */

const _frac = (n, d) =>
  `<span class="frac"><span class="num">${n}</span><span class="den">${d}</span></span>`;
const _sqrt = (x) => `<span class="sqrt">${x}</span>`;

// flow-tile constructors
const sTile = (data, label) => ({ label, data, kind: "stream" });
const oTile = (data, label) => ({ label, data, kind: "out" });
const wTile = (m, label, extra = {}) => ({ label, rows: m.length, cols: m[0].length, kind: "weight", ...extra });
const vTile = (vec, label, kind = "stream") => ({ label, data: [vec], kind });

function buildTrace(W, text) {
  const N_HEAD = 4,
    HEAD = 16,
    N_LAYER = 3,
    CTX = 64;

  let tokens = encode(text);
  if (tokens.length > CTX) tokens = tokens.slice(-CTX);
  const T = tokens.length;
  const chars = tokens.map(decode);
  const rowLabels = chars.map(glyph);
  const posLabels = tokens.map((_, i) => String(i));
  const vocabLabels = VOCAB.split("").map(glyph);

  const steps = [];
  const push = (s) => steps.push({ id: steps.length, ...s });

  /* ── embedding stage ─────────────────────────────────────── */
  push({
    group: "embed",
    label: "tokenize",
    crumb: "embed / tokenize",
    title: "Text → token ids",
    eq: `x<sub>t</sub> = stoi(c<sub>t</sub>)`,
    why:
      "Every character is looked up in the fixed 75-symbol vocabulary and replaced " +
      "by its integer id. No learning here — just a table lookup that turns text into " +
      "numbers the network can index with.",
    flow: {
      operands: [{ label: "text", chars: rowLabels, kind: "text" }],
      ops: [],
      arrow: "encode",
      out: { label: "ids", ids: tokens, kind: "ids" },
    },
    viz: { type: "tokens", tokens, chars },
  });

  const Wtok = W["token_embedding_table.weight"];
  const tokEmb = tokens.map((id) => Wtok[id].slice());
  push({
    group: "embed",
    label: "token emb",
    crumb: "embed / token embeddings",
    title: "Token embedding lookup",
    eq: `E<sup>tok</sup><sub>t</sub> = W<sub>E</sub>[ x<sub>t</sub> ]`,
    why:
      "Each id selects one row of the learned 75×64 embedding table. That row is a " +
      "64-dimensional vector — the model's learned meaning for that character. Stacking " +
      "the rows gives a T×64 matrix, one vector per position.",
    flow: {
      operands: [wTile(Wtok, "W_E", { hlRows: tokens })],
      ops: [],
      arrow: "gather rows",
      out: oTile(tokEmb, "Eᵗᵒᵏ"),
    },
    viz: { type: "matrix", data: tokEmb, rowLabels, colLabel: "64 dims", diverging: true },
  });

  const Wpos = W["position_embedding_table.weight"];
  const posEmb = tokens.map((_, i) => Wpos[i].slice());
  push({
    group: "embed",
    label: "pos emb",
    crumb: "embed / positional embeddings",
    title: "Positional embedding lookup",
    eq: `E<sup>pos</sup><sub>t</sub> = W<sub>P</sub>[ t ]`,
    why:
      "Self-attention is order-blind: on its own it treats a sequence as a set. So each " +
      "position 0,1,2,… also gets a learned 64-d vector. Position, not character, is what " +
      "this table encodes.",
    flow: {
      operands: [wTile(Wpos, "W_P", { hlRows: tokens.map((_, i) => i) })],
      ops: [],
      arrow: "gather rows 0…T-1",
      out: oTile(posEmb, "Eᵖᵒˢ"),
    },
    viz: { type: "matrix", data: posEmb, rowLabels: posLabels, colLabel: "64 dims", diverging: true },
  });

  let x = add(tokEmb, posEmb);
  push({
    group: "embed",
    label: "sum → h⁰",
    crumb: "embed / sum",
    title: "Add token + position → residual stream",
    eq: `h<sup>0</sup><sub>t</sub> = E<sup>tok</sup><sub>t</sub> + E<sup>pos</sup><sub>t</sub>`,
    why:
      "The two vectors are added elementwise. This sum is the initial residual stream h⁰ " +
      "— the running T×64 state every block reads from and writes back into. Follow this " +
      "matrix along the bottom of the screen; it is the one thing the whole network transforms.",
    flow: {
      operands: [sTile(tokEmb, "Eᵗᵒᵏ"), sTile(posEmb, "Eᵖᵒˢ")],
      ops: ["+"],
      arrow: "",
      out: oTile(clone(x), "h⁰"),
    },
    viz: { type: "matrix", data: clone(x), rowLabels, colLabel: "64 dims", diverging: true },
    stream: clone(x),
  });

  /* ── transformer blocks ──────────────────────────────────── */
  for (let l = 0; l < N_LAYER; l++) {
    const g = `block-${l}`;
    const bp = `blocks.${l}`;

    const ln1 = layerNorm(x, W[`${bp}.ln1.weight`], W[`${bp}.ln1.bias`]);
    push({
      group: g,
      label: "layernorm₁",
      crumb: `block ${l} / layernorm 1`,
      title: "LayerNorm before attention",
      eq: `h&#771; = &#947;<sub>1</sub> &#8857; ${_frac("h − &#956;", _sqrt("&#963;<sup>2</sup> + &#949;"))} + &#946;<sub>1</sub>`,
      why:
        "Each token's 64 values are recentred to mean 0 and rescaled to variance 1, then " +
        "stretched/shifted by learned γ and β. This keeps the numbers entering attention in a " +
        "stable range. Pre-norm: we normalise the input to the sublayer, not its output.",
      flow: { operands: [sTile(clone(x), "h")], ops: [], arrow: "LayerNorm", out: oTile(clone(ln1), "h̃") },
      viz: { type: "matrix", data: clone(ln1), rowLabels, colLabel: "64 dims", diverging: true },
      stream: clone(x),
    });

    const headOuts = [];
    for (let h = 0; h < N_HEAD; h++) {
      const hp = `${bp}.sa.heads.${h}`;
      const Wq = W[`${hp}.query.weight`], Wk = W[`${hp}.key.weight`], Wv = W[`${hp}.value.weight`];
      const q = linear(ln1, Wq, null);
      const k = linear(ln1, Wk, null);
      const v = linear(ln1, Wv, null);
      const hg = `${g}/head-${h}`;

      push({
        group: hg, label: `h${h} · Q`, crumb: `block ${l} / head ${h} / query`,
        title: `Query projection · head ${h}`,
        eq: `Q<sup>(${h})</sup> = h&#771; · W<sub>Q</sub><sup>(${h})&#8868;</sup>`,
        why:
          "Each token asks a question: what am I looking for? The 64-d normalised vector is " +
          "projected down to a 16-d query. A head is a learned subspace — this one has its own " +
          "W_Q, so its questions differ from the other heads'.",
        flow: { operands: [sTile(ln1, "h̃"), wTile(Wq, "W_Q")], ops: ["×"], arrow: "", out: oTile(clone(q), "Q") },
        viz: { type: "matrix", data: clone(q), rowLabels, colLabel: "16 (head dim)", diverging: true },
        stream: clone(x),
      });
      push({
        group: hg, label: `h${h} · K`, crumb: `block ${l} / head ${h} / key`,
        title: `Key projection · head ${h}`,
        eq: `K<sup>(${h})</sup> = h&#771; · W<sub>K</sub><sup>(${h})&#8868;</sup>`,
        why:
          "Each token also advertises a key: what do I offer? Same normalised input, a different " +
          "learned projection to 16 dims. Queries will be compared against these keys to decide " +
          "who attends to whom.",
        flow: { operands: [sTile(ln1, "h̃"), wTile(Wk, "W_K")], ops: ["×"], arrow: "", out: oTile(clone(k), "K") },
        viz: { type: "matrix", data: clone(k), rowLabels, colLabel: "16 (head dim)", diverging: true },
        stream: clone(x),
      });
      push({
        group: hg, label: `h${h} · V`, crumb: `block ${l} / head ${h} / value`,
        title: `Value projection · head ${h}`,
        eq: `V<sup>(${h})</sup> = h&#771; · W<sub>V</sub><sup>(${h})&#8868;</sup>`,
        why:
          "The value is the content a token will actually hand over if attended to — separate " +
          "from the key it advertises. Again a learned 64→16 projection.",
        flow: { operands: [sTile(ln1, "h̃"), wTile(Wv, "W_V")], ops: ["×"], arrow: "", out: oTile(clone(v), "V") },
        viz: { type: "matrix", data: clone(v), rowLabels, colLabel: "16 (head dim)", diverging: true },
        stream: clone(x),
      });

      let scores = qk_dot(q, k);
      push({
        group: hg, label: `h${h} · QKᵀ`, crumb: `block ${l} / head ${h} / scores`,
        title: `Attention scores QKᵀ · head ${h}`,
        eq: `S<sub>ij</sub> = Q<sub>i</sub> · K<sub>j</sub>`,
        why:
          "Row i's query is dotted with every token j's key. A big S(i,j) means token i finds " +
          "token j relevant. This is a raw affinity — not yet a probability, and not yet causal.",
        flow: { operands: [sTile(q, "Q"), sTile(k, "Kᵀ")], ops: ["×"], arrow: "", out: oTile(clone(scores), "S") },
        viz: { type: "matrix", data: clone(scores), rowLabels, colLabels: rowLabels, square: true, diverging: true },
        stream: clone(x),
      });

      const scaled = scores.map((r) => r.map((s) => s / Math.sqrt(HEAD)));
      push({
        group: hg, label: `h${h} · scale`, crumb: `block ${l} / head ${h} / scale`,
        title: `Scale by 1/√d_k · head ${h}`,
        eq: `S &#8592; ${_frac("S", _sqrt("d<sub>k</sub>"))} , &nbsp; d<sub>k</sub>=16 &#8658; &#215;&#188;`,
        why:
          "Dot products grow with dimension; left unchecked they push softmax into near-one-hot " +
          "spikes with vanishing gradients. Dividing by √d_k (here √16 = 4) keeps the variance of " +
          "the scores around 1.",
        flow: { operands: [sTile(scores, "S")], ops: [], arrow: "× ¼", out: oTile(clone(scaled), "S/√dₖ") },
        viz: { type: "matrix", data: clone(scaled), rowLabels, colLabels: rowLabels, square: true, diverging: true },
        stream: clone(x),
      });

      const attn = maskAndSoftmax(scaled);
      push({
        group: hg, label: `h${h} · softmax`, crumb: `block ${l} / head ${h} / mask + softmax`,
        title: `Causal mask + softmax · head ${h}`,
        eq: `A<sub>ij</sub> = ${_frac("e<sup>S<sub>ij</sub></sup>", "&#8721;<sub>k&#8804;i</sub> e<sup>S<sub>ik</sub></sup>")} , &nbsp; A<sub>ij</sub>=0 for j&gt;i`,
        why:
          "First the causal mask sets every future position (j>i) to −∞ so a token can never read " +
          "ahead — this is what makes the model a decoder. Then each row is softmaxed into a " +
          "probability distribution over the tokens it may see. Rows sum to 1; the upper triangle is exactly zero.",
        flow: { operands: [sTile(scaled, "S/√dₖ")], ops: [], arrow: "mask ▸ softmax", out: { ...oTile(clone(attn), "A"), seq: true } },
        viz: { type: "attention", data: clone(attn), rowLabels, colLabels: rowLabels },
        stream: clone(x),
      });

      const out = score_v_dot(attn, v);
      headOuts.push(out);
      push({
        group: hg, label: `h${h} · A·V`, crumb: `block ${l} / head ${h} / weighted values`,
        title: `Weighted sum of values · head ${h}`,
        eq: `O<sup>(${h})</sup><sub>i</sub> = &#8721;<sub>j&#8804;i</sub> A<sub>ij</sub> V<sub>j</sub>`,
        why:
          "Each token builds its output as a weighted average of the value vectors it attended to, " +
          "using the softmax weights. This is the moment information actually moves between " +
          "positions — everything before was just deciding how much.",
        flow: { operands: [{ ...sTile(attn, "A"), seq: true }, sTile(v, "V")], ops: ["×"], arrow: "", out: oTile(clone(out), `O⁽${h}⁾`) },
        viz: { type: "matrix", data: clone(out), rowLabels, colLabel: "16 (head dim)", diverging: true },
        stream: clone(x),
      });
    }

    const concat = concatHeads(headOuts);
    push({
      group: g, label: "concat heads", crumb: `block ${l} / concat heads`,
      title: "Concatenate the 4 heads",
      eq: `O = [ O<sup>(0)</sup> &#8214; O<sup>(1)</sup> &#8214; O<sup>(2)</sup> &#8214; O<sup>(3)</sup> ]`,
      why:
        "The four 16-d head outputs are laid side by side to reform a 64-d vector per token. Each " +
        "quarter of this matrix came from a head that attended differently.",
      flow: {
        operands: headOuts.map((o, i) => sTile(o, `O⁽${i}⁾`)),
        ops: ["‖", "‖", "‖"],
        arrow: "",
        out: oTile(clone(concat), "O"),
      },
      viz: { type: "matrix", data: clone(concat), rowLabels, colLabel: "64 (4×16)", diverging: true, groups: 4 },
      stream: clone(x),
    });

    const proj = linear(concat, W[`${bp}.sa.proj.weight`], W[`${bp}.sa.proj.bias`]);
    push({
      group: g, label: "output proj", crumb: `block ${l} / output projection`,
      title: "Mix heads (output projection)",
      eq: `O<sub>proj</sub> = O · W<sub>O</sub><sup>&#8868;</sup> + b<sub>O</sub>`,
      why:
        "A learned 64×64 layer lets the heads' outputs interact and be remapped back into the " +
        "residual stream's coordinate system, instead of staying in four isolated blocks.",
      flow: { operands: [sTile(concat, "O"), wTile(W[`${bp}.sa.proj.weight`], "W_O")], ops: ["×"], arrow: "+ b_O", out: oTile(clone(proj), "Oₚᵣₒⱼ") },
      viz: { type: "matrix", data: clone(proj), rowLabels, colLabel: "64 dims", diverging: true },
      stream: clone(x),
    });

    const preAttn = clone(x);
    x = add(x, proj);
    push({
      group: g, label: "+ residual", crumb: `block ${l} / residual add`,
      title: "Residual add (attention)",
      eq: `h&#8242; = h + O<sub>proj</sub>`,
      why:
        "Attention's result is added back onto the stream rather than replacing it. The block only " +
        "writes a correction, so information and gradients flow straight through the skip connection " +
        "— the key to training deep stacks.",
      flow: { operands: [sTile(preAttn, "h"), sTile(proj, "Oₚᵣₒⱼ")], ops: ["+"], arrow: "", out: oTile(clone(x), "h′") },
      viz: { type: "matrix", data: clone(x), rowLabels, colLabel: "64 dims", diverging: true },
      stream: clone(x),
    });

    const ln2 = layerNorm(x, W[`${bp}.ln2.weight`], W[`${bp}.ln2.bias`]);
    push({
      group: g, label: "layernorm₂", crumb: `block ${l} / layernorm 2`,
      title: "LayerNorm before feed-forward",
      eq: `h&#771;&#8242; = &#947;<sub>2</sub> &#8857; ${_frac("h&#8242; − &#956;", _sqrt("&#963;<sup>2</sup> + &#949;"))} + &#946;<sub>2</sub>`,
      why: "A second normalisation, now feeding the position-wise feed-forward network. Same operation as LayerNorm₁ with its own learned γ, β.",
      flow: { operands: [sTile(clone(x), "h′")], ops: [], arrow: "LayerNorm", out: oTile(clone(ln2), "h̃′") },
      viz: { type: "matrix", data: clone(ln2), rowLabels, colLabel: "64 dims", diverging: true },
      stream: clone(x),
    });

    const ffUp = relu(linear(ln2, W[`${bp}.ffwd.net.0.weight`], W[`${bp}.ffwd.net.0.bias`]));
    push({
      group: g, label: "ffn ↑ + relu", crumb: `block ${l} / feed-forward up`,
      title: "Feed-forward expand + ReLU",
      eq: `u = ReLU( h&#771;&#8242; · W<sub>1</sub><sup>&#8868;</sup> + b<sub>1</sub> )`,
      why:
        "Each token is processed independently: project 64→256, then ReLU zeroes the negatives. The " +
        "wider hidden layer plus a non-linearity is where the block does its per-token 'thinking', " +
        "selecting and combining features.",
      flow: { operands: [sTile(ln2, "h̃′"), wTile(W[`${bp}.ffwd.net.0.weight`], "W₁")], ops: ["×"], arrow: "+b₁ ▸ ReLU", out: oTile(clone(ffUp), "u") },
      viz: { type: "matrix", data: clone(ffUp), rowLabels, colLabel: "256 (hidden)", diverging: true },
      stream: clone(x),
    });

    const ffDown = linear(ffUp, W[`${bp}.ffwd.net.2.weight`], W[`${bp}.ffwd.net.2.bias`]);
    push({
      group: g, label: "ffn ↓", crumb: `block ${l} / feed-forward down`,
      title: "Feed-forward project back",
      eq: `f = u · W<sub>2</sub><sup>&#8868;</sup> + b<sub>2</sub>`,
      why: "The 256-d hidden activations are projected back to 64 dims so the result fits the residual stream again.",
      flow: { operands: [sTile(ffUp, "u"), wTile(W[`${bp}.ffwd.net.2.weight`], "W₂")], ops: ["×"], arrow: "+ b₂", out: oTile(clone(ffDown), "f") },
      viz: { type: "matrix", data: clone(ffDown), rowLabels, colLabel: "64 dims", diverging: true },
      stream: clone(x),
    });

    const preFF = clone(x);
    x = add(x, ffDown);
    push({
      group: g, label: "+ residual → hˡ⁺¹", crumb: `block ${l} / residual add`,
      title: "Residual add (feed-forward)",
      eq: `h<sup>${l + 1}</sup> = h&#8242; + f`,
      why: `The feed-forward correction is added back, completing block ${l}. The stream carries forward, a little more refined, to the next block.`,
      flow: { operands: [sTile(preFF, "h′"), sTile(ffDown, "f")], ops: ["+"], arrow: "", out: oTile(clone(x), `h${l + 1}`) },
      viz: { type: "matrix", data: clone(x), rowLabels, colLabel: "64 dims", diverging: true },
      stream: clone(x),
    });
  }

  /* ── output stage ────────────────────────────────────────── */
  const zf = layerNorm(x, W["ln_f.weight"], W["ln_f.bias"]);
  push({
    group: "output", label: "final ln", crumb: "output / final layernorm",
    title: "Final LayerNorm",
    eq: `z = &#947;<sub>f</sub> &#8857; norm(h<sup>3</sup>) + &#946;<sub>f</sub>`,
    why: "One last normalisation of the residual stream before it is read out into vocabulary scores.",
    flow: { operands: [sTile(clone(x), "h³")], ops: [], arrow: "LayerNorm", out: oTile(clone(zf), "z") },
    viz: { type: "matrix", data: clone(zf), rowLabels, colLabel: "64 dims", diverging: true },
    stream: clone(x),
  });

  const logits = linear(zf, W["lm_head.weight"], W["lm_head.bias"]);
  push({
    group: "output", label: "lm head", crumb: "output / lm head",
    title: "Project to vocabulary logits",
    eq: `logits = z · W<sub>lm</sub><sup>&#8868;</sup> + b<sub>lm</sub>`,
    why:
      "A learned 64→75 layer scores every position against every possible next character. Only the " +
      "last row matters for generation, but the model computes all of them (that is how it trains on " +
      "every position at once).",
    flow: { operands: [sTile(zf, "z"), wTile(W["lm_head.weight"], "W_lm")], ops: ["×"], arrow: "+ b_lm", out: oTile(clone(logits), "logits") },
    viz: { type: "matrix", data: clone(logits), rowLabels, colLabels: vocabLabels, diverging: true },
    stream: clone(x),
  });

  const lastLogits = logits[T - 1].slice();
  push({
    group: "output", label: "last row", crumb: "output / last position",
    title: "Take the last position's logits",
    eq: `&#8467; = logits<sub>T&#8722;1</sub>`,
    why:
      "To predict what comes next we only need the scores at the final token — its row is the model's " +
      "raw preference over all 75 characters, before normalising.",
    flow: { operands: [sTile(logits, "logits", )], ops: [], arrow: `row T−1 = ${T - 1}`, out: vTile(lastLogits, "ℓ", "out") },
    viz: { type: "bars", values: lastLogits, labels: vocabLabels, signed: true },
    stream: clone(x),
  });

  const probs = softmax1D(lastLogits);
  const best = probs.indexOf(Math.max(...probs));
  push({
    group: "output", label: "softmax", crumb: "output / softmax",
    title: "Softmax → next-character distribution",
    eq: `p = softmax(&#8467;) , &nbsp; p<sub>i</sub> = ${_frac("e<sup>&#8467;<sub>i</sub></sup>", "&#8721;<sub>k</sub> e<sup>&#8467;<sub>k</sub></sup>")}`,
    why:
      "Exponentiate and normalise the 75 logits into a probability distribution that sums to 1. The " +
      "tallest bar is the model's favourite next character; the spread shows how confident it is.",
    flow: { operands: [vTile(lastLogits, "ℓ")], ops: [], arrow: "softmax", out: vTile(probs, "p", "out") },
    viz: { type: "bars", values: probs, labels: vocabLabels, highlight: best },
    stream: clone(x),
    predicted: decode(best),
  });

  push({
    group: "output", label: "select ĉ", crumb: "output / select",
    title: "Pick the next character",
    eq: `c&#770; = argmax<sub>i</sub> p<sub>i</sub> = &#8220;${glyph(decode(best))}&#8221;`,
    why:
      "With greedy (argmax) decoding we take the single most probable character. Append it to the input " +
      "and the whole forward pass runs again on the extended sequence — that loop is autoregressive generation.",
    flow: { operands: [vTile(probs, "p")], ops: [], arrow: "argmax", out: { label: "ĉ", char: glyph(decode(best)), kind: "char" } },
    viz: { type: "bars", values: probs, labels: vocabLabels, highlight: best },
    stream: clone(x),
    predicted: decode(best),
  });

  return { steps, tokens, chars, predicted: decode(best), T };
}

function clone(m) {
  return m.map((r) => (Array.isArray(r) ? r.slice() : r));
}
