/* forward.js — one correct forward pass, computed once.
   Every lesson reads its intermediates from here instead of
   recomputing the chain, so the numbers can never drift between
   lessons. Mirrors train/model.py exactly (pre-norm, ReLU FFN,
   biased LayerNorm variance, 1/√d_k attention scale). */

(function () {
  const C = 64, N_HEAD = 4, HEAD = 16, N_LAYER = 3, EPS = 1e-5;

  function lin(x, w, b) {
    // x: [in], w: [out][in], b: [out] | null  ->  [out]
    const out = new Array(w.length);
    for (let o = 0; o < w.length; o++) {
      const wr = w[o];
      let s = b ? b[o] : 0;
      for (let i = 0; i < x.length; i++) s += x[i] * wr[i];
      out[o] = s;
    }
    return out;
  }
  function layernorm(x, g, b) {
    const n = x.length;
    let m = 0; for (const v of x) m += v; m /= n;
    let vv = 0; for (const v of x) { const d = v - m; vv += d * d; } vv /= n;
    const std = Math.sqrt(vv + EPS);
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = ((x[i] - m) / std) * g[i] + b[i];
    return { out, mean: m, variance: vv, std };
  }
  function softmax(x) {
    let mx = -Infinity; for (const v of x) if (v > mx) mx = v;
    let sum = 0; const e = x.map((v) => { const z = Math.exp(v - mx); sum += z; return z; });
    return e.map((v) => v / sum);
  }
  const add = (a, b) => a.map((r, i) => r.map((v, j) => v + b[i][j]));

  function computeForward(W, tokens) {
    const profile = { embed: 0, blocks: [], output: 0 };
    let t0 = performance.now();

    const T = tokens.length;
    const Wtok = W["token_embedding_table.weight"];
    const Wpos = W["position_embedding_table.weight"];

    // embeddings → initial residual stream
    const tokEmb = tokens.map((id) => Wtok[id].slice());
    const posEmb = tokens.map((_, i) => Wpos[i].slice());
    const h0 = add(tokEmb, posEmb);
    
    profile.embed += performance.now() - t0;

    let x = h0.map((r) => r.slice());
    const blocks = [];

    for (let l = 0; l < N_LAYER; l++) {
      const bProf = { ln1: 0, qkv: 0, attn: 0, proj: 0, resid1: 0, ln2: 0, ffwd: 0, resid2: 0 };
      const bp = `blocks.${l}`;
      
      t0 = performance.now();
      const ln1 = x.map((row) => layernorm(row, W[`${bp}.ln1.weight`], W[`${bp}.ln1.bias`]).out);
      bProf.ln1 += performance.now() - t0;

      const heads = [];
      const concat = x.map(() => new Array(C).fill(0));
      for (let h = 0; h < N_HEAD; h++) {
        t0 = performance.now();
        const Wq = W[`${bp}.sa.heads.${h}.query.weight`];
        const Wk = W[`${bp}.sa.heads.${h}.key.weight`];
        const Wv = W[`${bp}.sa.heads.${h}.value.weight`];
        const Q = ln1.map((r) => lin(r, Wq, null));
        const K = ln1.map((r) => lin(r, Wk, null));
        const V = ln1.map((r) => lin(r, Wv, null));
        bProf.qkv += performance.now() - t0;

        t0 = performance.now();
        const scoresRaw = [], scoresScaled = [], scoresMasked = [], attn = [], out = [];
        for (let i = 0; i < T; i++) {
          const raw = [], scaled = [], masked = [];
          for (let j = 0; j < T; j++) {
            let dot = 0; for (let d = 0; d < HEAD; d++) dot += Q[i][d] * K[j][d];
            raw.push(dot);
            const sc = dot / Math.sqrt(HEAD);
            scaled.push(sc);
            masked.push(j > i ? -Infinity : sc);
          }
          scoresRaw.push(raw); scoresScaled.push(scaled); scoresMasked.push(masked);
          attn.push(softmax(masked.map((v) => (v === -Infinity ? -1e9 : v))).map((p, j) => (j > i ? 0 : p)));
        }
        for (let i = 0; i < T; i++) {
          const o = new Array(HEAD).fill(0);
          for (let d = 0; d < HEAD; d++) {
            let s = 0; for (let j = 0; j <= i; j++) s += attn[i][j] * V[j][d];
            o[d] = s;
            concat[i][h * HEAD + d] = s;
          }
          out.push(o);
        }
        heads.push({ Q, K, V, scoresRaw, scoresScaled, scoresMasked, attn, out });
        bProf.attn += performance.now() - t0;
      }

      t0 = performance.now();
      const attnProj = concat.map((r) => lin(r, W[`${bp}.sa.proj.weight`], W[`${bp}.sa.proj.bias`]));
      bProf.proj += performance.now() - t0;
      
      t0 = performance.now();
      const h1 = add(x, attnProj);
      bProf.resid1 += performance.now() - t0;
      
      t0 = performance.now();
      const ln2 = h1.map((row) => layernorm(row, W[`${bp}.ln2.weight`], W[`${bp}.ln2.bias`]).out);
      bProf.ln2 += performance.now() - t0;
      
      t0 = performance.now();
      const ffUp = ln2.map((r) => lin(r, W[`${bp}.ffwd.net.0.weight`], W[`${bp}.ffwd.net.0.bias`]));
      const ffAct = ffUp.map((r) => r.map((v) => Math.max(0, v)));
      const ffDown = ffAct.map((r) => lin(r, W[`${bp}.ffwd.net.2.weight`], W[`${bp}.ffwd.net.2.bias`]));
      bProf.ffwd += performance.now() - t0;
      
      t0 = performance.now();
      const hOut = add(h1, ffDown);
      bProf.resid2 += performance.now() - t0;

      profile.blocks.push(bProf);
      blocks.push({ ln1, heads, concat, attnProj, h1, ln2, ffUp, ffAct, ffDown, hOut });
      x = hOut;
    }

    t0 = performance.now();
    // output stage
    const znorm = x.map((row) => layernorm(row, W["ln_f.weight"], W["ln_f.bias"]).out);
    const logits = znorm.map((r) => lin(r, W["lm_head.weight"], W["lm_head.bias"]));
    const lastLogits = logits[T - 1].slice();
    const probs = softmax(lastLogits);
    let best = 0; for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    profile.output += performance.now() - t0;

    return { T, tokens, tokEmb, posEmb, h0, blocks, znorm, logits, lastLogits, probs, best, profile };
  }

  // expose
  window.computeForward = computeForward;
  window._fwdMath = { lin, layernorm, softmax };
})();
