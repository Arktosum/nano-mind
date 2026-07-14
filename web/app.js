/* app.js — lesson mode with a small chapter registry.
   Each lesson is one operation of the forward pass, taught on its own
   infinitely-scrolling page. Prev/next move between lessons; the full
   micro-step player (trace.js / flow.js / minimap.js) stays parked. */

const el = (id) => document.getElementById(id);
const dom = {
  body: document.body,
  boot: el("bootStatus"),
  promptInput: el("promptInput"),
  runBtn: el("runBtn"),
  promptForm: el("promptForm"),
  landing: el("landing"),
  workbench: el("workbench"),
  wbPrompt: el("wbPrompt"),
  wbRun: el("wbRun"),
  crumb: el("stageCrumb"),
  title: el("stageTitle"),
  eq: el("equationCard"),
  instrument: el("instrument"),
  why: el("whyText"),
  navPrev: el("lessonPrev"),
  navNext: el("lessonNext"),
  navLabel: el("lessonLabel"),
  foot: el("lessonFoot"),
};

let weights = null;
let ctx = null; // { tokens, chars, weights }
let idx = 0;

/* ── the lessons ──────────────────────────────────────────── */
const LESSONS = [
  {
    crumb: "embed / token embedding — step 2 of the forward pass",
    title: "Token embedding lookup",
    short: "token embedding",
    eq: `<div class="eq">E<sup>tok</sup><sub>t</sub> = W<sub>E</sub>[ x<sub>t</sub> ] = onehot(x<sub>t</sub>) · W<sub>E</sub></div>`,
    why:
      "<p>The tokenizer already turned your text into integer ids. But the raw number <b>21</b> is just " +
      "a name tag — it isn't <i>bigger</i> or <i>smaller</i> than another id in a way the network should " +
      "trust. So each character is handed a richer, learnable representation instead.</p>" +
      "<p>That representation is <b>W_E</b>, a learned <b>75×64</b> table: one row per vocabulary character, " +
      "each row a 64-number vector. The lookup takes the row at the token's id.</p>" +
      "<p>Why is a plain lookup a legitimate network operation? Because picking row <b>21</b> equals " +
      "multiplying a <b>one-hot</b> vector (a single 1 at position 21) by the table — the 1 selects that row, " +
      "the zeros erase the rest. The one-hot column is drawn row-aligned with the table, so " +
      "<b>the lit cell sits beside the row it picks</b>.</p>" +
      "<p class='note'><b>Notation:</b> the diagram builds <b>E<sup>tok</sup><sub>t</sub></b> (subscript " +
      "<i>t</i>) — one 1×64 row for one position. Stack every position to get <b>E<sup>tok</sup></b>, the T×64 matrix.</p>",
    render: (host) =>
      renderEmbeddingLesson(host, { tokens: ctx.tokens, chars: ctx.chars, table: weights["token_embedding_table.weight"] }),
  },
  {
    crumb: "embed / position + sum — step 3 of the forward pass",
    title: "Adding position → the residual stream",
    short: "position + sum",
    eq: `<div class="eq">h<sup>0</sup><sub>t</sub> = E<sup>tok</sup><sub>t</sub> + E<sup>pos</sup><sub>t</sub></div>`,
    why:
      "<p>The next big step, self-attention, is <b>order-blind</b> — it treats the sequence as a set. On its " +
      "own it can't tell your text from a rearrangement of the same characters. So we inject <i>where</i> each " +
      "token sits.</p>" +
      "<p>The lookup uses the <b>exact same machine</b> as token embedding — a one-hot selecting a row of a " +
      "learned table — <i>on purpose</i>. So the interesting part isn't the mechanism; it's that " +
      "<b>token and position tables are opposites in meaning</b>. W_E is indexed by <i>identity</i> and its " +
      "rows are shared (both your <b>e</b>'s reuse one row); W_P is indexed by <i>order</i> and each row is " +
      "used once. W_E has one row per character; W_P has one row per <i>slot</i>, so it hard-caps the context " +
      "length — go past 64 characters and there's no row, which is exactly why the code truncates. Section 2 " +
      "lays the contrast out.</p>" +
      "<p>Then the two are added elementwise into the first residual-stream row, " +
      "<b>h<sup>0</sup><sub>t</sub> = E<sup>tok</sup><sub>t</sub> + E<sup>pos</sup><sub>t</sub></b>, fusing " +
      "<i>what</i> (the character) with <i>where</i> (the position).</p>",
    render: (host) => renderPositionSumLesson(host, ctx),
  },
  {
    crumb: "block 0 / layernorm 1 — step 4 of the forward pass",
    title: "Layer Normalization",
    short: "layernorm 1",
    eq: `<div class="eq">h̃<sub>t</sub> = γ ⊙ <span class="frac"><span class="num">h<sub>t</sub> − μ</span><span class="den">√<span class="paren">(</span>σ² + ε<span class="paren">)</span></span></span> + β</div>`,
    why:
      "<p>Now that we have our initial residual stream (the sum of what the character is and where it is), we are ready to start transforming it. But before we feed these vectors into the self-attention mechanism, we pass them through <b>Layer Normalization</b>.</p>" +
      "<p>Neural networks hate extreme numbers. If some dimensions grow too large, they can drown out other signals or push activation functions into saturated, flat regions where learning stops (vanishing gradients). LayerNorm fixes this by forcing the vector's values to follow a standard distribution: centered at 0, with a spread of 1.</p>" +
      "<p>Crucially, it does this <i>per token</i>. It looks at the 64 dimensions of <b>one</b> token's vector, finds the mean and variance, and normalizes those 64 numbers. It doesn't mix information between different tokens in the sequence. Why across dimensions and not across tokens? Because sequence lengths vary, and in generation, we don't even have the future tokens yet! Normalizing across dimensions ensures a token's representation is self-contained and stable, completely independent of whatever other text happens to surround it.</p>" +
      "<p>Finally, the network applies a learned scale (<b>γ</b>) and shift (<b>β</b>) so it can mold this perfect distribution into whatever shape the attention layer actually needs.</p>",
    render: (host) => renderLayerNormLesson(host, ctx),
  },
  {
    crumb: "block 0 / head 0 / qkv projections — step 5 of the forward pass",
    title: "Query, Key, Value Projections",
    short: "qkv 0",
    eq: `<div class="eq">q<sub>t</sub> = h̃<sub>t</sub> W<sub>Q</sub><sup>T</sup> &nbsp;&nbsp; k<sub>t</sub> = h̃<sub>t</sub> W<sub>K</sub><sup>T</sup> &nbsp;&nbsp; v<sub>t</sub> = h̃<sub>t</sub> W<sub>V</sub><sup>T</sup></div>`,
    why:
      "<p>We have reached the heart of the Transformer: <b>Self-Attention</b>. But attention doesn't operate directly on the 64-dimensional residual stream. Instead, each of the 4 attention heads gets its own unique perspective by projecting the data down into a smaller 16-dimensional subspace.</p>" +
      "<p>For each token, we compute three new vectors by multiplying the normalized residual stream (<b>h̃<sub>t</sub></b>) by three learned matrices (<b>W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub></b>):</p>" +
      "<ul>" +
      "<li><b>Query (q):</b> What am I looking for?</li>" +
      "<li><b>Key (k):</b> What do I contain?</li>" +
      "<li><b>Value (v):</b> If you find me relevant, here is the information I will give you.</li>" +
      "</ul>" +
      "<p>Because this model has 4 heads, this projection happens 4 times in parallel, allowing the network to look for 4 completely different things at the same time. <b>Crucially, this is incredibly fast on hardware.</b> By splitting a large 64-dimensional operation into four smaller 16-dimensional chunks, modern GPUs can distribute the work across thousands of cores simultaneously. This \"head split\" is why Transformers are so absurdly efficient to train!</p>",
    render: (host) => renderQKVLesson(host, ctx),
  },
  {
    crumb: "block 0 / head 0 / attention scores — step 6 of the forward pass",
    title: "Attention Scores (Q · Kᵀ)",
    short: "scores 0",
    eq: `<div class="eq">S = <span class="frac"><span class="num">Q K<sup>T</sup></span><span class="den">√<span class="paren">(</span>d<sub>k</sub><span class="paren">)</span></span></span> <span class="note">(masked)</span></div>`,
    why:
      "<p>Now that every token has broadcasted its <b>Query</b> and its <b>Key</b>, we can calculate how much they align. We do this by taking the dot product of every Query with every Key, forming a T×T matrix.</p>" +
      "<p>But there's a mathematical trap here! If we assume the elements of the Query and Key vectors have a mean of 0 and a variance of 1, their dot product (which sums across all 16 dimensions) will have a mean of 0 but a <b>variance of 16</b>. As we add more dimensions, the variance of the dot product grows, meaning the scores get more and more extreme.</p>" +
      "<p>If we feed these huge raw numbers into the Softmax function later, it will push the function into its flat, saturated regions—giving a probability of almost 1.0 to the highest score and 0 to everything else. This completely kills the gradients during training! To pull the variance back down to a healthy 1, we <b>scale</b> the scores by dividing by the square root of the dimensions (√16 = 4).</p>" +
      "<p>Finally, we apply a <b>causal mask</b>. Since this model generates text autoregressively (one character at a time, left-to-right), token 3 is forbidden from attending to token 4, because token 4 hasn't been generated yet during inference! We enforce this rule by setting all upper-triangular scores (where j > i) to −∞</p>",
    render: (host) => renderScoresLesson(host, ctx),
  },
  {
    crumb: "block 0 / head 0 / attention weights (softmax) — step 7 of the forward pass",
    title: "Attention Weights (Softmax)",
    short: "softmax 0",
    eq: `<div class="eq">A = softmax(S)</div>`,
    why:
      "<p>We have our masked alignment scores, but they are just arbitrary numbers. To make them useful, we pass them through a <b>Softmax</b> function row by row. This turns the scores into a beautiful probability distribution.</p>" +
      "<p>Softmax exponentiates every score and then divides it by the sum of the row. Because e<sup>−∞</sup> = 0, the mathematically enforced causal mask ensures that exactly 0% of the attention is spent on future tokens.</p>" +
      "<p>The remaining valid scores are squashed between 0 and 1 such that every row adds up perfectly to 1.0. You can think of this as each token having a strict 100% 'attention budget' to spend on itself and the tokens that came before it.</p>",
    render: (host) => renderSoftmaxLesson(host, ctx),
  },
  {
    crumb: "block 0 / head 0 / multiply values — step 8 of the forward pass",
    title: "Multiply Values (O = A · V)",
    short: "values 0",
    eq: `<div class="eq">O<sub>0</sub> = A V</div>`,
    why:
      "<p>We finally arrive at the grand payoff of the Self-Attention mechanism: actually gathering information from other tokens!</p>" +
      "<p>Each token has already computed a <b>Value (v)</b> vector—this is the actual \"substance\" or information the token is offering to share. We also just computed the <b>Attention Weights (A)</b>—the strict percentage breakdown of exactly how much each token wants to listen to every other token.</p>" +
      "<p>By multiplying the T×T attention matrix by the T×16 value matrix, every token computes a perfectly <b>weighted sum</b> of the Values from the past. A token that received 80% attention will contribute 80% of its Value vector to the final Output. The resulting T×16 Output matrix represents a new, context-enriched understanding for each token.</p>",
    render: (host) => renderValuesLesson(host, ctx),
  },
  {
    crumb: "block 0 / concat heads — step 9 of the forward pass",
    title: "Concatenate Heads",
    short: "concat 0",
    eq: `<div class="eq">O = [ O<sub>0</sub> || O<sub>1</sub> || O<sub>2</sub> || O<sub>3</sub> ]</div>`,
    why:
      "<p>Up until now, we've only been visualizing <b>Head 0</b>. But while we were doing that, <b>Heads 1, 2, and 3</b> were simultaneously running the exact same math in parallel on the GPU!</p>" +
      "<p>This reveals the true brilliance of Multi-Head Attention. Not only do the different heads learn to look for different linguistic patterns (like nouns vs. punctuation), but because they don't depend on each other, their math can be perfectly parallelized. The GPU crunches all 4 heads independently at lightning speed.</p>" +
      "<p>To combine all these diverse insights, the Transformer simply glues the four 16-dimensional vectors together side-by-side. 4 heads × 16 dimensions perfectly reconstructs our original 64-dimensional vector size. The token is whole again, now deeply enriched by the context of its neighbors.</p>",
    render: (host) => renderConcatLesson(host, ctx),
  },
  {
    crumb: "block 0 / attention output projection — step 10 of the forward pass",
    title: "Output Projection",
    short: "proj 0",
    eq: `<div class="eq">O<sub>proj</sub> = O · W<sub>proj</sub><sup>T</sup> + b<sub>proj</sub></div>`,
    why:
      "<p>This is the absolute final step inside the Self-Attention block. We have our concatenated 64-dimensional vector, but there's a problem: the information from Head 0 is strictly stuck in the first 16 dimensions, Head 1 is stuck in the next 16, and so on.</p>" +
      "<p>To fix this, we pass the vector through one final <b>Linear layer (W<sub>proj</sub>)</b>. This 64×64 weight matrix mixes all the dimensions together, allowing the different heads to finally talk to each other and synthesize their diverse findings into a single, cohesive 64-dimensional conclusion.</p>" +
      "<p>This perfectly blended output represents everything the token learned by attending to its past. It is now ready to be added back into the residual stream!</p>",
    render: (host) => renderProjLesson(host, ctx),
  },
  {
    crumb: "block 0 / residual add — step 11 of the forward pass",
    title: "Residual Add",
    short: "resid 0",
    eq: `<div class="eq">h' = h<sup>0</sup> + O<sub>proj</sub></div>`,
    why:
      "<p>We have officially exited the Self-Attention mechanism! Now we return to the main highway of the Transformer: the <b>Residual Stream</b>.</p>" +
      "<p>Instead of completely overwriting the token's representation, we take the Self-Attention output (which we can think of as a \"delta\" or an update) and simply <b>add</b> it to the original residual stream vector.</p>" +
      "<p>This \"Skip Connection\" is a critical architectural innovation. By allowing the original data (and gradients during training) to flow unimpeded all the way through the network, it solves the vanishing gradient problem and allows Transformers to be built with hundreds of layers.</p>",
    render: (host) => renderResid1Lesson(host, ctx),
  },
  {
    crumb: "block 0 / layernorm 2 — step 12 of the forward pass",
    title: "LayerNorm 2",
    short: "ln2 0",
    eq: `<div class="eq">h̃' = γ<sub>2</sub> ⊙ norm(h') + β<sub>2</sub></div>`,
    why:
      "<p>Now that the residual stream has been updated with context from other tokens, it's time to prepare it for the second major component of the Transformer block: the <b>Feed Forward Network (MLP)</b>.</p>" +
      "<p>Just like we did before Self-Attention, we apply a <b>LayerNorm</b> to the 64-dimensional vector to keep its values stable and well-behaved. The vector is centered to a mean of 0, scaled to a variance of 1, and then transformed by a new set of learned parameters: the scale vector <b>γ<sub>2</sub></b> and the shift vector <b>β<sub>2</sub></b>.</p>" +
      "<p>The sequence is now perfectly prepared for the Feed Forward Network, which will process each token individually to extract deeper meaning.</p>",
    render: (host) => renderLn2Lesson(host, ctx),
  },
  {
    crumb: "block 0 / feed forward — step 13 of the forward pass",
    title: "Feed Forward Network",
    short: "ffwd 0",
    eq: `<div class="eq">FFN(h) = ReLU(h · W<sub>up</sub><sup>T</sup>) · W<sub>down</sub><sup>T</sup></div>`,
    why:
      "<p>While Self-Attention lets tokens communicate with each other, the <b>Feed Forward Network (FFN)</b> gives each token a chance to \"think\" about what it just learned. The FFN processes each token completely independently of the others.</p>" +
      "<p>First, the 64-dimensional vector is projected up into a wider 256-dimensional space (W<sub>up</sub>), allowing the model to represent more complex combinations of features. We apply a <b>ReLU</b> non-linearity to introduce mathematical complexity (thresholding negative values to 0), and then project it back down to 64 dimensions (W<sub>down</sub>).</p>",
    render: (host) => renderFfwdLesson(host, ctx),
  },
  {
    crumb: "block 0 / residual add 2 — step 14 of the forward pass",
    title: "Second Residual Add",
    short: "resid2 0",
    eq: `<div class="eq">h'' = h' + FFN(h̃')</div>`,
    why:
      "<p>Just like we did after Self-Attention, we take the output of the Feed Forward Network and <b>add</b> it directly back into the main residual stream highway.</p>" +
      "<p>This completely finishes <b>Transformer Block 0</b>! The token started as a simple dictionary lookup, absorbed context from the tokens before it via attention, processed that context through an FFN, and safely stored all of this new knowledge in the residual stream.</p>",
    render: (host) => renderResid2Lesson(host, ctx),
  },
  {
    crumb: "block 0 / architecture recap",
    title: "The Big Picture",
    short: "recap",
    eq: `<div class="eq">A high-level view of the Nano-Mind architecture</div>`,
    why:
      "<p>We have just walked through the math for one full cycle of Multi-Head Self Attention and the preparation for the Feed Forward Network.</p>" +
      "<p>But the model actually has <b>3 identical Transformer Blocks</b>! The Residual Stream continues to flow upwards, branching off to be processed and added back in via Skip Connections, until it finally reaches the Output Stage to predict the next token.</p>" +
      "<p>This diagram recaps the components we just built in Block 0, and shows how they fit into the entire network.</p>",
    render: (host) => renderRecapLesson(host, ctx),
  },
];

/* ── boot ─────────────────────────────────────────────────── */
fetch("weights.json")
  .then((r) => {
    if (!r.ok) throw new Error("weights.json not found — serve over http");
    return r.json();
  })
  .then((w) => {
    weights = w;
    dom.boot.textContent = "ready · 3-layer char model";
    dom.boot.classList.add("ready");
    dom.promptInput.disabled = false;
    dom.runBtn.disabled = false;
    dom.promptInput.focus();
  })
  .catch((err) => {
    dom.boot.textContent = err.message;
    dom.boot.classList.add("error");
  });

/* ── landing → lesson ─────────────────────────────────────── */
dom.promptForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = dom.promptInput.value.trim();
  if (!weights || !text) return;
  dom.landing.hidden = true;
  dom.workbench.hidden = false;
  dom.body.dataset.mode = "workbench";
  dom.body.dataset.view = "lesson";
  dom.wbPrompt.value = text;
  buildCtx(text);
  showLesson();
});

dom.wbRun.addEventListener("click", () => {
  const text = dom.wbPrompt.value.trim();
  if (text) { buildCtx(text); showLesson(); }
});
dom.wbPrompt.addEventListener("keydown", (e) => { if (e.key === "Enter") dom.wbRun.click(); });

dom.navPrev.addEventListener("click", () => jump(idx - 1));
dom.navNext.addEventListener("click", () => jump(idx + 1));
window.addEventListener("keydown", (e) => {
  if (dom.workbench.hidden || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowRight") jump(idx + 1);
  if (e.key === "ArrowLeft") jump(idx - 1);
});

/* ── driving ──────────────────────────────────────────────── */
function buildCtx(text) {
  let tokens = encode(text);
  if (tokens.length > 64) tokens = tokens.slice(-64);
  ctx = { tokens, chars: tokens.map(decode), weights };
}

function jump(next) {
  if (next < 0 || next >= LESSONS.length || next === idx) return;
  idx = next;
  showLesson();
  dom.instrument.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showLesson() {
  if (!ctx || !ctx.tokens.length) return;
  const L = LESSONS[idx];
  dom.crumb.textContent = L.crumb;
  dom.title.textContent = L.title;
  dom.eq.innerHTML = L.eq;
  dom.why.innerHTML = L.why;

  // Clear previous event listeners by replacing the node
  const newInst = dom.instrument.cloneNode(false);
  dom.instrument.replaceWith(newInst);
  dom.instrument = newInst;
  
  L.render(dom.instrument);

  // Update Minimap
  const minimapCard = el("minimapCard");
  if (minimapCard && typeof renderMinimap === "function") {
    minimapCard.innerHTML = renderMinimap(L.short);
    minimapCard.style.display = "flex";
  }
  
  dom.navLabel.textContent = `Lesson ${idx + 1} of ${LESSONS.length}`;
  dom.navPrev.disabled = idx === 0;
  dom.navNext.disabled = idx === LESSONS.length - 1;

  const prev = LESSONS[idx - 1], nxt = LESSONS[idx + 1];
  dom.foot.innerHTML =
    (prev ? `<button class="foot-btn" data-dir="-1"><span>← previous</span><b>${prev.short}</b></button>` : "<span></span>") +
    (nxt ? `<button class="foot-btn next" data-dir="1"><span>next →</span><b>${nxt.short}</b></button>` : "<span></span>");
  dom.foot.querySelectorAll(".foot-btn").forEach((b) =>
    b.addEventListener("click", () => jump(idx + Number(b.dataset.dir)))
  );
}
