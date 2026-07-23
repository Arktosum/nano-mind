/* mla.js — Multi-Head Latent Attention (DeepSeek) */
function renderMlaLesson(host, ctx) {
  function render() {
    const w = 700;
    const h = 450;
    
    return `
      <div class="lz-diagram">
        <div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333; position:relative;">
          <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" id="mlaSvg">
            <!-- Definitions for arrows -->
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
              </marker>
            </defs>

            <!-- Top Half: Standard MHA -->
            <text x="350" y="30" fill="#e67700" font-weight="bold" text-anchor="middle" font-size="16">Standard Multi-Head Attention (MHA)</text>
            <text x="350" y="50" fill="#aaa" text-anchor="middle" font-size="12" id="mhaStatus">Generating Token 0...</text>
            
            <text x="50" y="80" fill="#fff" font-size="12">GPU Compute</text>
            <rect x="50" y="90" width="100" height="40" rx="4" fill="#2c2c2c" stroke="#e67700" stroke-width="2" />
            <text x="100" y="115" fill="#fff" text-anchor="middle" font-size="12">Calc Q, K, V</text>
            
            <text x="250" y="80" fill="#fff" font-size="12">4 Attention Heads</text>
            <rect x="250" y="90" width="40" height="40" rx="4" fill="#c92a2a" />
            <rect x="300" y="90" width="40" height="40" rx="4" fill="#c92a2a" />
            <rect x="350" y="90" width="40" height="40" rx="4" fill="#c92a2a" />
            <rect x="400" y="90" width="40" height="40" rx="4" fill="#c92a2a" />
            <text x="270" y="115" fill="#fff" font-size="10" text-anchor="middle">H1</text>
            <text x="320" y="115" fill="#fff" font-size="10" text-anchor="middle">H2</text>
            <text x="370" y="115" fill="#fff" font-size="10" text-anchor="middle">H3</text>
            <text x="420" y="115" fill="#fff" font-size="10" text-anchor="middle">H4</text>
            
            <path d="M 150 110 L 240 110" fill="none" stroke="#555" stroke-width="2" marker-end="url(#arrow)" />
            
            <text x="50" y="160" fill="#fff" font-size="12">GPU VRAM (Fills rapidly!)</text>
            <rect x="50" y="170" width="600" height="40" rx="4" fill="#1a365d" stroke="#c92a2a" stroke-width="2" />
            <g id="mhaVram"></g>
            
            <path d="M 345 130 L 345 160" fill="none" stroke="#e67700" stroke-width="2" marker-end="url(#arrow)" />
            
            <!-- Separator -->
            <line x1="50" y1="230" x2="650" y2="230" stroke="#333" stroke-width="2" stroke-dasharray="4 4" />
            
            <!-- Bottom Half: DeepSeek MLA -->
            <text x="350" y="260" fill="#2b8a3e" font-weight="bold" text-anchor="middle" font-size="16">DeepSeek Multi-Head Latent Attention (MLA)</text>
            <text x="350" y="280" fill="#aaa" text-anchor="middle" font-size="12" id="mlaStatus">Generating Token 0...</text>

            <!-- Input x_t -->
            <text x="40" y="295" fill="#fff" font-size="10" text-anchor="middle">Input x_t</text>
            <rect x="30" y="300" width="20" height="80" rx="4" fill="#555" />
            
            <path d="M 50 340 L 70 340" fill="none" stroke="#555" stroke-width="2" marker-end="url(#arrow)" />

            <!-- Down Project Trapezoid (Squeeze) -->
            <polygon points="80,300 160,325 160,355 80,380" fill="#2c2c2c" stroke="#2b8a3e" stroke-width="2" />
            <text x="120" y="345" fill="#fff" text-anchor="middle" font-size="12">W_DKV</text>
            <text x="120" y="365" fill="#aaa" text-anchor="middle" font-size="9">squeeze</text>
            
            <path d="M 160 340 L 190 340" fill="none" stroke="#555" stroke-width="2" marker-end="url(#arrow)" />
            
            <!-- Latent c_t -->
            <rect x="200" y="325" width="20" height="30" rx="2" fill="#4dabf7" />
            <text x="210" y="315" fill="#fff" font-weight="bold" font-size="12" text-anchor="middle">c_t</text>
            
            <path d="M 220 340 L 250 340" fill="none" stroke="#555" stroke-width="2" marker-end="url(#arrow)" />
            
            <!-- Up Project Trapezoid (Expand) -->
            <polygon points="260,325 340,300 340,380 260,355" fill="#2c2c2c" stroke="#4dabf7" stroke-width="2" />
            <text x="300" y="345" fill="#fff" text-anchor="middle" font-size="12">W_UK,V</text>
            <text x="300" y="365" fill="#aaa" text-anchor="middle" font-size="9">expand</text>
            
            <path d="M 340 340 L 370 340" fill="none" stroke="#555" stroke-width="2" marker-end="url(#arrow)" />

            <!-- Unzipped 4 heads -->
            <rect x="380" y="320" width="30" height="40" rx="4" fill="#c92a2a" />
            <rect x="420" y="320" width="30" height="40" rx="4" fill="#c92a2a" />
            <rect x="460" y="320" width="30" height="40" rx="4" fill="#c92a2a" />
            <rect x="500" y="320" width="30" height="40" rx="4" fill="#c92a2a" />
            <text x="395" y="345" fill="#fff" font-size="10" text-anchor="middle">H1</text>
            <text x="435" y="345" fill="#fff" font-size="10" text-anchor="middle">H2</text>
            <text x="475" y="345" fill="#fff" font-size="10" text-anchor="middle">H3</text>
            <text x="515" y="345" fill="#fff" font-size="10" text-anchor="middle">H4</text>
            
            <path d="M 210 365 L 210 390" fill="none" stroke="#2b8a3e" stroke-width="2" marker-end="url(#arrow)" />
            <text x="240" y="380" fill="#2b8a3e" font-size="10" font-weight="bold">save!</text>
            
            <text x="50" y="390" fill="#fff" font-size="12">GPU VRAM (Tiny footprint!)</text>
            <rect x="50" y="400" width="600" height="40" rx="4" fill="#1a365d" stroke="#2b8a3e" stroke-width="2" />
            <g id="mlaVram"></g>
          </svg>
        </div>
        
        <div style="margin-top: 40px; text-align: left; font-family: monospace; font-size: 14px; max-width: 700px; margin-left: auto; margin-right: auto;">
           <!-- Step 1 -->
           <div style="margin-bottom: 30px;">
              <div style="font-size: 16px; color: #4dabf7; font-weight: bold; margin-bottom: 8px;">Step 1: Down-Projection (Compression)</div>
              <div style="padding: 12px; background: #222; border-left: 3px solid #2b8a3e; margin-bottom: 10px; line-height: 1.5;">
                 c<sub>t</sub> = x<sub>t</sub> · W<sub>DKV</sub>
              </div>
              <div style="color: #aaa; line-height: 1.4;">
                 Instead of calculating full Keys and Values for every single head, MLA projects the input token (x<sub>t</sub>) through a <i>down-projection</i> matrix (W<sub>DKV</sub>) into a single, low-dimensional latent vector (c<sub>t</sub>).<br><br>
                 <b>Are these weights trainable?</b> Yes! W<sub>DKV</sub> is a fully learned, trainable parameter matrix that the model learns during training. It learns the optimal way to compress all the multi-head information into a tiny bottleneck.
              </div>
           </div>
           
           <!-- Step 2 -->
           <div style="margin-bottom: 30px;">
              <div style="font-size: 16px; color: #4dabf7; font-weight: bold; margin-bottom: 8px;">Step 2: Store ONLY the Latent</div>
              <div style="padding: 12px; background: #222; border-left: 3px solid #2b8a3e; margin-bottom: 10px; line-height: 1.5;">
                 RAM &larr; c<sub>t</sub>
              </div>
              <div style="color: #aaa; line-height: 1.4;">
                 In Standard MHA, we had to save two long vectors (k<sub>new</sub> and v<sub>new</sub>) for <i>every single head</i> into the KV cache. <br><br>
                 In MLA, <b>we do not save K and V at all!</b> We only save this single, tiny compressed latent vector c<sub>t</sub>. By avoiding storing separate matrices for every head, the memory footprint drops by an order of magnitude.
              </div>
           </div>
           
           <!-- Step 3 -->
           <div style="margin-bottom: 30px;">
              <div style="font-size: 16px; color: #4dabf7; font-weight: bold; margin-bottom: 8px;">Step 3: Up-Projection (Decompression)</div>
              <div style="padding: 12px; background: #222; border-left: 3px solid #2b8a3e; margin-bottom: 10px; line-height: 1.5;">
                 k<sub>t</sub> = c<sub>t</sub> · W<sub>UK</sub><br>
                 v<sub>t</sub> = c<sub>t</sub> · W<sub>UV</sub>
              </div>
              <div style="color: #aaa; line-height: 1.4;">
                 When the model needs to compute attention, it reads the tiny c<sub>t</sub> vectors from RAM and passes them through an <i>up-projection</i> matrix (W<sub>UK</sub>, W<sub>UV</sub>). This mathematically "unzips" the latent vector back into the full, multi-head Keys and Values on the fly! It trades a small amount of compute (multiplication) for a massive amount of memory.
              </div>
           </div>
           
           <!-- References -->
           <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #333; color: #888; font-size: 12px; line-height: 1.5;">
              <b>References:</b><br>
              Multi-Head Latent Attention (MLA) was introduced in <i>"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"</i> by DeepSeek-AI (2024). <a href="https://arxiv.org/abs/2405.04434" target="_blank" style="color: #4dabf7; text-decoration: none;">[arXiv:2405.04434]</a>
           </div>
        </div>
      </div>
    `;
  }
  
  host.innerHTML = render();

  const mhaVram = host.querySelector('#mhaVram');
  const mlaVram = host.querySelector('#mlaVram');
  
  if (!mhaVram || !mlaVram) return;

  const MAX_TOKENS = 12; // fits in the bar
  const TICK_SPEED = 60; // slower to show the difference
  
  let targetToken = 0;
  let ticks = 0;
  
  let animId;
  
  function tick() {
    ticks++;
    if (ticks > TICK_SPEED) {
      ticks = 0;
      targetToken++;
      if (targetToken >= MAX_TOKENS) {
        targetToken = 0; // loop
      }
    }
    
    // MHA logic (4 blocks per token)
    let mhaOut = "";
    for (let i = 0; i <= targetToken; i++) {
      const blockWidth = 40; // 4 heads * 10 width conceptually
      const x = 60 + i * (blockWidth + 5);
      
      if (x + blockWidth < 640) {
        // Render 4 tiny blocks to represent 4 heads
        mhaOut += `<rect x="${x}" y="175" width="8" height="30" rx="2" fill="#c92a2a" />
                   <rect x="${x+10}" y="175" width="8" height="30" rx="2" fill="#c92a2a" />
                   <rect x="${x+20}" y="175" width="8" height="30" rx="2" fill="#c92a2a" />
                   <rect x="${x+30}" y="175" width="8" height="30" rx="2" fill="#c92a2a" />`;
      }
    }
    mhaVram.innerHTML = mhaOut;
    
    // MLA logic (1 block per token)
    let mlaOut = "";
    for (let i = 0; i <= targetToken; i++) {
      const blockWidth = 10;
      const x = 60 + i * (blockWidth + 5);
      
      if (x + blockWidth < 640) {
        // Render 1 tiny block
        mlaOut += `<rect x="${x}" y="405" width="10" height="30" rx="2" fill="#4dabf7" />`;
      }
    }
    mlaVram.innerHTML = mlaOut;
    
    // Pulse animation for active lines
    const mhaStatus = host.querySelector('#mhaStatus');
    const mlaStatus = host.querySelector('#mlaStatus');
    
    mhaStatus.textContent = `Generating Token ${targetToken} (Saving 4x matrices per token)`;
    mlaStatus.textContent = `Generating Token ${targetToken} (Saving 1x latent vector per token)`;
    
    // warning red if full
    if (targetToken >= 10) {
      mhaStatus.style.fill = "#c92a2a";
      mhaStatus.textContent += " — MEMORY FULL!";
    } else {
      mhaStatus.style.fill = "#aaa";
    }
    
    animId = requestAnimationFrame(tick);
    host._animId = animId;
  }
  
  if (host._animId) cancelAnimationFrame(host._animId);
  tick();
}
