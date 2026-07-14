/* minimap.js — dynamic architecture overview, on the vermilion palette.
   Three contextual panels: the whole network, the parallel head split,
   and a single head's internals — shown only when relevant. */

const MM = {
  accent: "#d6401f",
  activeFill: "rgba(214,64,31,0.16)",
  idleFill: "#17170f",
  idleStroke: "#3a3a30",
  inkOn: "#e8e4da",
  inkOff: "#7d8595",
  wire: "#5a564b",
};

const MHA_STEPS = ["qkv 0", "scores 0", "softmax 0", "values 0", "concat 0", "proj 0"];

function renderMinimap(activeStep) {
  if (MHA_STEPS.includes(activeStep)) {
    return renderMainMinimap(activeStep) + renderMhaParallelMinimap(activeStep) + renderMhaSingleHeadMinimap(activeStep);
  }
  return renderMainMinimap(activeStep);
}

function mmWrap(title, w, h, inner) {
  return `<div class="mm-panel"><div class="mm-title">${title}</div>` +
    `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">` +
    `<defs><marker id="mmA" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">` +
    `<path d="M 0 0 L 10 5 L 0 10 z" fill="${MM.wire}"/></marker></defs>${inner}</svg></div>`;
}

function renderMainMinimap(activeStep) {
  let a = "";
  if (["token embedding", "position + sum"].includes(activeStep)) a = "input";
  else if (activeStep === "layernorm 1") a = "ln1";
  else if (MHA_STEPS.includes(activeStep)) a = "mha";
  else if (activeStep === "resid 0") a = "add1";
  else if (activeStep === "ln2 0") a = "ln2";
  else if (activeStep === "ffwd 0") a = "ffn";
  else if (activeStep === "resid2 0") a = "add2";
  else if (activeStep === "output" || activeStep === "generate") a = "out";

  const fill = (id) => (id === a ? MM.activeFill : MM.idleFill);
  const stroke = (id) => (id === a ? MM.accent : MM.idleStroke);
  const sw = (id) => (id === a ? "2" : "1");
  const tc = (id) => (id === a ? MM.inkOn : MM.inkOff);
  const fw = (id) => (id === a ? "bold" : "normal");
  const box = (id, x, y, w, h, label) =>
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill(id)}" stroke="${stroke(id)}" stroke-width="${sw(id)}"/>` +
    `<text x="${x + w / 2}" y="${y + h / 2 + 3}" fill="${tc(id)}" font-weight="${fw(id)}" text-anchor="middle" font-size="10">${label}</text>`;

  const wire = (d, arrow = true) => `<path d="${d}" fill="none" stroke="${MM.wire}" stroke-width="1.5"${arrow ? ' marker-end="url(#mmA)"' : ""}/>`;

  const s = `
    <line x1="40" y1="20" x2="40" y2="382" stroke="${MM.accent}" stroke-width="3" marker-end="url(#mmA)"/>
    ${box("input", 20, 20, 160, 20, "token + position")}
    <g transform="translate(0,60)">
      <rect x="25" y="0" width="175" height="180" fill="none" stroke="${MM.idleStroke}" stroke-dasharray="2 2"/>
      <text x="35" y="14" fill="${MM.inkOff}" font-size="10">block 0</text>
      ${wire("M 40 20 L 120 20 L 120 30")}
      ${box("ln1", 80, 30, 80, 20, "layernorm 1")}
      ${wire("M 120 50 L 120 60")}
      ${box("mha", 70, 60, 100, 24, "multi-head attn")}
      ${wire("M 120 84 L 120 100 L 48 100")}
      <circle cx="40" cy="100" r="8" fill="${fill("add1")}" stroke="${stroke("add1")}" stroke-width="${sw("add1")}"/>
      <text x="40" y="104" fill="${tc("add1")}" text-anchor="middle" font-size="12" font-weight="bold">+</text>
      ${wire("M 40 115 L 120 115 L 120 125")}
      ${box("ln2", 80, 125, 80, 20, "layernorm 2")}
      ${wire("M 120 145 L 120 155")}
      ${box("ffn", 80, 155, 80, 20, "feed forward")}
      ${wire("M 120 175 L 120 185 L 48 185")}
      <circle cx="40" cy="185" r="8" fill="${fill("add2")}" stroke="${stroke("add2")}" stroke-width="${sw("add2")}"/>
      <text x="40" y="189" fill="${tc("add2")}" text-anchor="middle" font-size="12" font-weight="bold">+</text>
    </g>
    <g transform="translate(0,255)">
      <rect x="25" y="0" width="175" height="22" fill="${MM.idleFill}" stroke="${MM.idleStroke}"/>
      <text x="112" y="15" fill="${MM.inkOff}" text-anchor="middle" font-size="10">block 1 (identical)</text>
    </g>
    <g transform="translate(0,288)">
      <rect x="25" y="0" width="175" height="22" fill="${MM.idleFill}" stroke="${MM.idleStroke}"/>
      <text x="112" y="15" fill="${MM.inkOff}" text-anchor="middle" font-size="10">block 2 (identical)</text>
    </g>
    <g transform="translate(0,325)">
      ${box("out", 25, 0, 175, 40, "")}
      <text x="112" y="17" fill="${tc("out")}" text-anchor="middle" font-size="10" font-weight="${fw("out")}">final LN + language head</text>
      <text x="112" y="31" fill="${tc("out")}" text-anchor="middle" font-size="10" font-weight="${fw("out")}">→ next-token probabilities</text>
    </g>`;
  return mmWrap("the whole network", 220, 400, s);
}

function renderMhaParallelMinimap(activeStep) {
  let a = "";
  if (["qkv 0", "scores 0", "softmax 0", "values 0"].includes(activeStep)) a = "head0";
  else if (activeStep === "concat 0") a = "concat";
  else if (activeStep === "proj 0") a = "proj";

  const fill = (id) => (id === a ? MM.activeFill : MM.idleFill);
  const stroke = (id) => (id === a ? MM.accent : MM.idleStroke);
  const tc = (id) => (id === a ? MM.inkOn : MM.inkOff);
  const wire = (d, arrow = true) => `<path d="${d}" fill="none" stroke="${MM.wire}" stroke-width="1.5"${arrow ? ' marker-end="url(#mmA)"' : ""}/>`;

  const head = (id, x, label) =>
    `<rect x="${x}" y="85" width="40" height="120" fill="${fill(id)}" stroke="${stroke(id)}"/>` +
    `<text x="${x + 20}" y="145" fill="${tc(id)}" text-anchor="middle" font-size="10" transform="rotate(-90 ${x + 20},145)">${label}</text>`;

  const s = `
    <text x="110" y="18" fill="${MM.accent}" font-weight="bold" text-anchor="middle" font-size="11">4 heads run in parallel</text>
    <rect x="60" y="30" width="100" height="20" fill="${MM.idleFill}" stroke="${MM.idleStroke}"/>
    <text x="110" y="44" fill="${MM.inkOff}" text-anchor="middle" font-size="10">from LayerNorm 1</text>
    ${wire("M 110 50 L 110 68 L 32 68 L 32 85")}${wire("M 110 50 L 110 68 L 84 68 L 84 85")}
    ${wire("M 110 50 L 110 68 L 136 68 L 136 85")}${wire("M 110 50 L 110 68 L 188 68 L 188 85")}
    ${head("head0", 12, "head 0")}${head("h1", 64, "head 1")}${head("h2", 116, "head 2")}${head("h3", 168, "head 3")}
    ${wire("M 32 205 L 32 220 L 110 220 L 110 233")}
    <path d="M 84 205 L 84 220 L 110 220" fill="none" stroke="${MM.wire}" stroke-width="1.5"/>
    <path d="M 136 205 L 136 220 L 110 220" fill="none" stroke="${MM.wire}" stroke-width="1.5"/>
    <path d="M 188 205 L 188 220 L 110 220" fill="none" stroke="${MM.wire}" stroke-width="1.5"/>
    <rect x="40" y="233" width="140" height="28" fill="${fill("concat")}" stroke="${stroke("concat")}"/>
    <text x="110" y="251" fill="${tc("concat")}" text-anchor="middle" font-size="10">concat → 64</text>
    ${wire("M 110 261 L 110 279")}
    <rect x="40" y="279" width="140" height="28" fill="${fill("proj")}" stroke="${stroke("proj")}"/>
    <text x="110" y="297" fill="${tc("proj")}" text-anchor="middle" font-size="10">output projection</text>
    ${wire("M 110 307 L 110 328")}
    <text x="110" y="345" fill="${MM.inkOff}" text-anchor="middle" font-size="10">→ residual add</text>`;
  return mmWrap("the parallel split", 220, 360, s);
}

function renderMhaSingleHeadMinimap(activeStep) {
  const a = activeStep.split(" ")[0];
  const internal = ["qkv", "scores", "softmax", "values"].includes(a);
  const fill = (id) => (id === a ? MM.activeFill : MM.idleFill);
  const stroke = (id) => (id === a ? MM.accent : MM.idleStroke);
  const tc = (id) => (id === a ? MM.inkOn : MM.inkOff);
  const wire = (d) => `<path d="${d}" fill="none" stroke="${MM.wire}" stroke-width="1.5" marker-end="url(#mmA)"/>`;
  const box = (id, y, label) =>
    `<rect x="55" y="${y}" width="110" height="28" fill="${fill(id)}" stroke="${stroke(id)}"/>` +
    `<text x="110" y="${y + 18}" fill="${tc(id)}" text-anchor="middle" font-size="10">${label}</text>`;

  const s = `
    <text x="110" y="18" fill="${internal ? MM.accent : MM.inkOff}" font-weight="bold" text-anchor="middle" font-size="11">inside one head</text>
    <rect x="55" y="30" width="110" height="20" fill="${MM.idleFill}" stroke="${MM.idleStroke}"/>
    <text x="110" y="44" fill="${MM.inkOff}" text-anchor="middle" font-size="10">from LayerNorm 1</text>
    ${wire("M 110 50 L 110 72")}${box("qkv", 72, "Q, K, V")}
    ${wire("M 110 100 L 110 122")}${box("scores", 122, "scores + mask")}
    ${wire("M 110 150 L 110 172")}${box("softmax", 172, "softmax")}
    ${wire("M 110 200 L 110 222")}${box("values", 222, "× values")}
    ${wire("M 110 250 L 110 272")}
    <text x="110" y="286" fill="${MM.inkOff}" text-anchor="middle" font-size="10">→ concat</text>`;
  return mmWrap("one head, zoomed in", 220, 300, s);
}
