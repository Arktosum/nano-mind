/* minimap.js — dynamic architecture overview */

function renderMinimap(activeStep) {
  const isMha = ["qkv 0", "scores 0", "softmax 0", "values 0", "concat 0", "proj 0"].includes(activeStep);
  if (isMha) {
    return renderMainMinimap(activeStep) + renderMhaParallelMinimap(activeStep) + renderMhaSingleHeadMinimap(activeStep);
  }
  return renderMainMinimap(activeStep);
}

function renderMainMinimap(activeStep) {
  const w = 220;
  const h = 420;
  
  // Mapping steps to nodes
  let activeNode = "";
  if (["token embedding", "position + sum"].includes(activeStep)) activeNode = "input";
  if (activeStep === "layernorm 1") activeNode = "ln1";
  if (["qkv 0", "scores 0", "softmax 0", "values 0", "concat 0", "proj 0"].includes(activeStep)) activeNode = "mha";
  if (activeStep === "resid 0") activeNode = "add1";
  if (activeStep === "ln2 0") activeNode = "ln2";
  if (activeStep === "ffwd 0") activeNode = "ffn";
  if (activeStep === "resid2 0") activeNode = "add2";

  // Helper to highlight a node
  const col = (id) => (id === activeNode ? "#4dabf7" : "#555");
  const fill = (id) => (id === activeNode ? "#1a365d" : "#2c2c2c");
  const strokeW = (id) => (id === activeNode ? "3" : "1");
  const textCol = (id) => (id === activeNode ? "#fff" : "#aaa");
  const fontW = (id) => (id === activeNode ? "bold" : "normal");

  const s = `
    <defs>
      <marker id="mmArrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
      </marker>
      <marker id="mmArrowBlue" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#4dabf7" />
      </marker>
    </defs>
    
    <!-- Residual Stream Spine -->
    <line x1="40" y1="20" x2="40" y2="380" stroke="#4dabf7" stroke-width="3" marker-end="url(#mmArrowBlue)" />
    
    <!-- Input Stage -->
    <rect x="20" y="20" width="160" height="20" rx="4" fill="${fill("input")}" stroke="${col("input")}" stroke-width="${strokeW("input")}" />
    <text x="100" y="34" fill="${textCol("input")}" font-weight="${fontW("input")}" text-anchor="middle" font-size="10">Token + Pos Embeddings</text>
    
    <!-- Block 0 -->
    <g transform="translate(0, 60)">
      <rect x="25" y="0" width="175" height="180" rx="4" fill="none" stroke="#666" stroke-dasharray="2 2" />
      <text x="35" y="15" fill="#888" font-size="10">Block 0</text>
      
      <!-- MHA Path -->
      <path d="M 40 20 L 120 20 L 120 30" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
      
      <rect x="80" y="30" width="80" height="20" rx="4" fill="${fill("ln1")}" stroke="${col("ln1")}" stroke-width="${strokeW("ln1")}" />
      <text x="120" y="44" fill="${textCol("ln1")}" font-weight="${fontW("ln1")}" text-anchor="middle" font-size="10">LayerNorm 1</text>
      
      <path d="M 120 50 L 120 60" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
      
      <rect x="70" y="60" width="100" height="24" rx="4" fill="${fill("mha")}" stroke="${activeNode === "mha" ? "#4dabf7" : "#e67700"}" stroke-width="${strokeW("mha")}" />
      <text x="120" y="76" fill="${textCol("mha")}" font-weight="${fontW("mha")}" text-anchor="middle" font-size="10">Multi-Head Attn</text>
      
      <!-- Add 1 -->
      <path d="M 120 84 L 120 100 L 48 100" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
      <circle cx="40" cy="100" r="8" fill="${fill("add1")}" stroke="${col("add1")}" stroke-width="${strokeW("add1")}" />
      <text x="40" y="104" fill="${textCol("add1")}" text-anchor="middle" font-size="12" font-weight="bold">+</text>
      
      <!-- FFN Path -->
      <path d="M 40 115 L 120 115 L 120 125" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
      
      <rect x="80" y="125" width="80" height="20" rx="4" fill="${fill("ln2")}" stroke="${col("ln2")}" stroke-width="${strokeW("ln2")}" />
      <text x="120" y="139" fill="${textCol("ln2")}" font-weight="${fontW("ln2")}" text-anchor="middle" font-size="10">LayerNorm 2</text>
      
      <path d="M 120 145 L 120 155" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
      
      <rect x="80" y="155" width="80" height="20" rx="4" fill="${fill("ffn")}" stroke="${activeNode === "ffn" ? "#4dabf7" : "#5c940d"}" stroke-width="${strokeW("ffn")}" />
      <text x="120" y="169" fill="${textCol("ffn")}" font-weight="${fontW("ffn")}" text-anchor="middle" font-size="10">FFN</text>
      
      <!-- Add 2 -->
      <path d="M 120 175 L 120 185 L 48 185" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
      <circle cx="40" cy="185" r="8" fill="${fill("add2")}" stroke="${col("add2")}" stroke-width="${strokeW("add2")}" />
      <text x="40" y="189" fill="${textCol("add2")}" text-anchor="middle" font-size="12" font-weight="bold">+</text>
    </g>
    
    <!-- Block 1 -->
    <g transform="translate(0, 255)">
      <rect x="25" y="0" width="175" height="24" rx="4" fill="#222" stroke="#444" />
      <text x="112" y="16" fill="#888" text-anchor="middle" font-size="10">Transformer Block 1</text>
      <circle cx="40" cy="12" r="8" fill="#2c2c2c" stroke="#555" />
      <text x="40" y="16" fill="#aaa" text-anchor="middle" font-size="12" font-weight="bold">+</text>
    </g>

    <!-- Block 2 -->
    <g transform="translate(0, 290)">
      <rect x="25" y="0" width="175" height="24" rx="4" fill="#222" stroke="#444" />
      <text x="112" y="16" fill="#888" text-anchor="middle" font-size="10">Transformer Block 2</text>
      <circle cx="40" cy="12" r="8" fill="#2c2c2c" stroke="#555" />
      <text x="40" y="16" fill="#aaa" text-anchor="middle" font-size="12" font-weight="bold">+</text>
    </g>
    
    <!-- Output Stage -->
    <g transform="translate(0, 330)">
      <rect x="25" y="0" width="175" height="40" rx="4" fill="#222" stroke="#444" />
      <text x="112" y="18" fill="#888" text-anchor="middle" font-size="10">Output LayerNorm</text>
      <text x="112" y="32" fill="#888" text-anchor="middle" font-size="10">+ Language Head</text>
    </g>
  `;

  return `<div class="minimap-inner" style="background:#0d0d0b; border:1px solid #333; border-radius:8px; padding:10px; width:220px;"><div style="font-size:10px; color:#888; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; text-align:center;">1. Big Picture</div><svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">${s}</svg></div>`;
}

function renderMhaParallelMinimap(activeStep) {
  const w = 220;
  const h = 420;
  
  let activeNode = "";
  if (["qkv 0", "scores 0", "softmax 0", "values 0"].includes(activeStep)) activeNode = "head0";
  if (activeStep === "concat 0") activeNode = "concat";
  if (activeStep === "proj 0") activeNode = "proj";

  const col = (id) => (id === activeNode ? "#4dabf7" : "#e67700");
  const fill = (id) => (id === activeNode ? "#1a365d" : "#2c2c2c");
  const strokeW = (id) => (id === activeNode ? "3" : "1");
  const textCol = (id) => (id === activeNode ? "#fff" : "#aaa");
  const fontW = (id) => (id === activeNode ? "bold" : "normal");

  const headFill = (id) => (id === "head0" ? fill("head0") : "#222");
  const headCol = (id) => (id === "head0" ? col("head0") : "#444");

  const s = `
    <defs>
      <marker id="mmArrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
      </marker>
    </defs>
    
    <text x="110" y="20" fill="#e67700" font-weight="bold" text-anchor="middle" font-size="12">Multi-Head Parallel Map</text>
    
    <rect x="60" y="35" width="100" height="20" rx="4" fill="#222" stroke="#444" />
    <text x="110" y="49" fill="#888" text-anchor="middle" font-size="10">Input from LayerNorm</text>
    
    <!-- Split lines -->
    <path d="M 110 55 L 110 70 L 32 70 L 32 85" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    <path d="M 110 55 L 110 70 L 84 70 L 84 85" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    <path d="M 110 55 L 110 70 L 136 70 L 136 85" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    <path d="M 110 55 L 110 70 L 188 70 L 188 85" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    
    <!-- 4 Parallel Heads -->
    <rect x="12" y="85" width="40" height="120" rx="4" fill="${headFill("head0")}" stroke="${headCol("head0")}" stroke-width="${strokeW("head0")}" />
    <text x="32" y="145" fill="${textCol("head0")}" font-weight="${fontW("head0")}" text-anchor="middle" font-size="10" transform="rotate(-90 32,145)">Head 0</text>
    
    <rect x="64" y="85" width="40" height="120" rx="4" fill="${headFill("head1")}" stroke="${headCol("head1")}" />
    <text x="84" y="145" fill="#888" text-anchor="middle" font-size="10" transform="rotate(-90 84,145)">Head 1</text>
    
    <rect x="116" y="85" width="40" height="120" rx="4" fill="${headFill("head2")}" stroke="${headCol("head2")}" />
    <text x="136" y="145" fill="#888" text-anchor="middle" font-size="10" transform="rotate(-90 136,145)">Head 2</text>
    
    <rect x="168" y="85" width="40" height="120" rx="4" fill="${headFill("head3")}" stroke="${headCol("head3")}" />
    <text x="188" y="145" fill="#888" text-anchor="middle" font-size="10" transform="rotate(-90 188,145)">Head 3</text>
    
    <!-- Merge lines -->
    <path d="M 32 205 L 32 220 L 110 220 L 110 235" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    <path d="M 84 205 L 84 220 L 110 220" fill="none" stroke="#888" stroke-width="1.5" />
    <path d="M 136 205 L 136 220 L 110 220" fill="none" stroke="#888" stroke-width="1.5" />
    <path d="M 188 205 L 188 220 L 110 220" fill="none" stroke="#888" stroke-width="1.5" />
    
    <!-- Concat -->
    <rect x="40" y="235" width="140" height="30" rx="4" fill="${fill("concat")}" stroke="${col("concat")}" stroke-width="${strokeW("concat")}" />
    <text x="110" y="254" fill="${textCol("concat")}" font-weight="${fontW("concat")}" text-anchor="middle" font-size="10">Concat All Heads</text>
    
    <path d="M 110 265 L 110 285" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    
    <!-- Proj -->
    <rect x="40" y="285" width="140" height="30" rx="4" fill="${fill("proj")}" stroke="${col("proj")}" stroke-width="${strokeW("proj")}" />
    <text x="110" y="304" fill="${textCol("proj")}" font-weight="${fontW("proj")}" text-anchor="middle" font-size="10">Output Projection</text>
    
    <path d="M 110 315 L 110 340" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    
    <circle cx="110" cy="350" r="10" fill="#222" stroke="#555" />
    <text x="110" y="355" fill="#aaa" text-anchor="middle" font-size="16" font-weight="bold">+</text>
    <text x="140" y="354" fill="#888" text-anchor="start" font-size="10">To Add 1</text>
  `;

  return `<div class="minimap-inner" style="background:#0d0d0b; border:1px solid #333; border-radius:8px; padding:10px; width:220px;"><div style="font-size:10px; color:#888; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; text-align:center;">2. The Parallel Split</div><svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">${s}</svg></div>`;
}

function renderMhaSingleHeadMinimap(activeStep) {
  const w = 220;
  const h = 420;
  
  // Helper to highlight a node
  // If activeStep is concat or proj, activeNode will be something else that doesn't match the internal steps, thus greying them out.
  const activeNode = activeStep.split(" ")[0]; // "qkv", "scores", "softmax", "values", "concat", "proj"
  
  // Only highlight if it's one of the internal steps
  const isInternal = ["qkv", "scores", "softmax", "values"].includes(activeNode);
  
  const col = (id) => (id === activeNode ? "#4dabf7" : (isInternal ? "#e67700" : "#444"));
  const fill = (id) => (id === activeNode ? "#1a365d" : (isInternal ? "#2c2c2c" : "#222"));
  const strokeW = (id) => (id === activeNode ? "3" : "1");
  const textCol = (id) => (id === activeNode ? "#fff" : (isInternal ? "#aaa" : "#666"));
  const fontW = (id) => (id === activeNode ? "bold" : "normal");

  const s = `
    <defs>
      <marker id="mmArrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
      </marker>
    </defs>
    
    <text x="110" y="20" fill="${isInternal ? "#e67700" : "#666"}" font-weight="bold" text-anchor="middle" font-size="12">Inside Head 0</text>
    
    <rect x="60" y="35" width="100" height="20" rx="4" fill="#222" stroke="#444" />
    <text x="110" y="49" fill="#888" text-anchor="middle" font-size="10">Input from LayerNorm</text>
    
    <path d="M 110 55 L 110 80" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    
    <rect x="40" y="80" width="140" height="30" rx="4" fill="${fill("qkv")}" stroke="${col("qkv")}" stroke-width="${strokeW("qkv")}" />
    <text x="110" y="99" fill="${textCol("qkv")}" font-weight="${fontW("qkv")}" text-anchor="middle" font-size="10">Q, K, V Projections</text>
    
    <path d="M 110 110 L 110 140" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    
    <rect x="60" y="140" width="100" height="30" rx="4" fill="${fill("scores")}" stroke="${col("scores")}" stroke-width="${strokeW("scores")}" />
    <text x="110" y="159" fill="${textCol("scores")}" font-weight="${fontW("scores")}" text-anchor="middle" font-size="10">Attention Scores</text>
    
    <path d="M 110 170 L 110 200" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    
    <rect x="60" y="200" width="100" height="30" rx="4" fill="${fill("softmax")}" stroke="${col("softmax")}" stroke-width="${strokeW("softmax")}" />
    <text x="110" y="219" fill="${textCol("softmax")}" font-weight="${fontW("softmax")}" text-anchor="middle" font-size="10">Softmax & Mask</text>
    
    <path d="M 110 230 L 110 260" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    
    <rect x="60" y="260" width="100" height="30" rx="4" fill="${fill("values")}" stroke="${col("values")}" stroke-width="${strokeW("values")}" />
    <text x="110" y="279" fill="${textCol("values")}" font-weight="${fontW("values")}" text-anchor="middle" font-size="10">Multiply Values</text>
    
    <path d="M 110 290 L 110 320" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#mmArrow)" />
    
    <rect x="60" y="320" width="100" height="20" rx="4" fill="#222" stroke="#444" />
    <text x="110" y="334" fill="#888" text-anchor="middle" font-size="10">To Concat All Heads</text>
  `;

  return `<div class="minimap-inner" style="background:#0d0d0b; border:1px solid #333; border-radius:8px; padding:10px; width:220px;"><div style="font-size:10px; color:#888; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; text-align:center;">3. Single Head Zoom</div><svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">${s}</svg></div>`;
}
