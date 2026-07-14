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
  L.render(dom.instrument);

  dom.navLabel.textContent = `${idx + 1} / ${LESSONS.length}`;
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
