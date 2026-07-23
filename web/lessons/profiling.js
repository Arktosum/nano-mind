/* profiling.js — Inference Time Profiling */

export function renderProfilingLesson(host, ctx) {
  if (!ctx || !ctx.fwd || !ctx.fwd.profile) {
    host.innerHTML = "<div style='color:red;'>Profiling data not found. Please run the model.</div>";
    return;
  }

  const p = ctx.fwd.profile;
  const labels = {
    embed: "Token & Position Embeddings",
    output: "Output Stage (LayerNorm & Logits)"
  };
  const blockLabels = {
    ln1: "Layer Normalization 1",
    qkv: "Q, K, V Projections",
    attn: "Attention Math (Scores, Mask, Softmax, Values)",
    proj: "Attention Output Projection",
    resid1: "First Residual Add",
    ln2: "Layer Normalization 2",
    ffwd: "Feed Forward Network",
    resid2: "Second Residual Add"
  };
  const blockKeys = ["ln1", "qkv", "attn", "proj", "resid1", "ln2", "ffwd", "resid2"];

  let totalTime = p.embed + p.output;
  p.blocks.forEach(b => {
    blockKeys.forEach(k => { totalTime += b[k]; });
  });

  const data = [];
  data.push({
    key: "embed",
    label: labels.embed,
    time: p.embed,
    pct: totalTime > 0 ? (p.embed / totalTime) * 100 : 0
  });

  p.blocks.forEach((b, i) => {
    blockKeys.forEach(k => {
      data.push({
        key: `b${i}_${k}`,
        label: `Block ${i}: ${blockLabels[k]}`,
        time: b[k],
        pct: totalTime > 0 ? (b[k] / totalTime) * 100 : 0
      });
    });
  });

  data.push({
    key: "output",
    label: labels.output,
    time: p.output,
    pct: totalTime > 0 ? (p.output / totalTime) * 100 : 0
  });

  const sortedData = [...data].sort((a, b) => b.time - a.time); // Descending

  const renderRow = (item) => `
    <div style="display: flex; align-items: center; margin-bottom: 10px; font-size: 13px;">
      <div style="width: 280px; color: #ddd;">${item.label}</div>
      <div style="width: 80px; text-align: right; color: #4dabf7; font-family: monospace; font-weight: bold;">${item.time.toFixed(3)} ms</div>
      <div style="width: 60px; text-align: right; color: #aaa; margin-right: 15px;">${item.pct.toFixed(1)}%</div>
      <div style="flex-grow: 1; background: #222; height: 14px; border-radius: 3px; overflow: hidden; position: relative;">
        <div style="background: ${item.pct > 25 ? '#e03131' : (item.pct > 10 ? '#e67700' : '#4dabf7')}; width: ${item.pct}%; height: 100%; transition: width 0.5s ease-out;"></div>
      </div>
    </div>
  `;

  host.innerHTML = `
    <div class="lz-diagram">
      <div style="padding: 25px; background: #111; border-radius: 8px; border: 1px solid #333; color: #fff; font-family: sans-serif; overflow-y: auto; max-height: 550px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;">
          <h3 style="margin: 0; color: #fff; font-size: 20px;">Inference Time Profiler</h3>
          <div style="font-size: 14px; color: #888; background: #1a1a1a; padding: 6px 12px; border-radius: 4px; border: 1px solid #333;">
            Total Forward Pass Time: <strong style="color: #fff; font-family: monospace; font-size: 16px;">${totalTime.toFixed(3)} ms</strong>
          </div>
        </div>
        
        <div style="display: flex; gap: 40px; flex-direction: column;">
          
          <div>
            <h4 style="color: #aaa; text-transform: uppercase; letter-spacing: 1px; font-size: 12px; margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 8px;">Ordered by Bottleneck (Most Expensive First)</h4>
            ${sortedData.map(renderRow).join("")}
          </div>

          <div>
            <h4 style="color: #aaa; text-transform: uppercase; letter-spacing: 1px; font-size: 12px; margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 8px;">Chronological Order (Forward Pass Flow)</h4>
            ${data.map(renderRow).join("")}
          </div>

        </div>
      </div>
    </div>
  `;
}
