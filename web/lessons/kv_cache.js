/* kv_cache.js — The KV Cache explained */
export function renderKvCacheLesson(host, ctx) {
  function render() {
    const w = 700;
    const h = 400;
    
    return `
      <div class="lz-diagram">
        <div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333; position:relative;">
          <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" id="kvCacheSvg">
            <!-- Definitions for arrows -->
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
              </marker>
            </defs>

            <!-- Top Half: Without Cache -->
            <text x="350" y="30" fill="#e67700" font-weight="bold" text-anchor="middle" font-size="16">Without KV Cache (Recomputes Everything)</text>
            <text x="350" y="50" fill="#aaa" text-anchor="middle" font-size="12" id="noCacheStatus">Generating Token 0...</text>
            
            <!-- Compute Core for No Cache -->
            <rect x="50" y="70" width="120" height="60" rx="4" fill="#2c2c2c" stroke="#e67700" stroke-width="2" />
            <text x="110" y="95" fill="#fff" text-anchor="middle" font-size="12">GPU Compute</text>
            <text x="110" y="115" fill="#aaa" text-anchor="middle" font-size="10">Calc Q, K, V</text>
            
            <g id="noCacheTokens"></g>
            
            <!-- Separator -->
            <line x1="50" y1="180" x2="650" y2="180" stroke="#333" stroke-width="2" stroke-dasharray="4 4" />
            
            <!-- Bottom Half: With Cache -->
            <text x="350" y="210" fill="#2b8a3e" font-weight="bold" text-anchor="middle" font-size="16">With KV Cache (Reads Past from RAM)</text>
            <text x="350" y="230" fill="#aaa" text-anchor="middle" font-size="12" id="cacheStatus">Generating Token 0...</text>

            <!-- Compute Core for Cache -->
            <rect x="50" y="250" width="120" height="60" rx="4" fill="#2c2c2c" stroke="#2b8a3e" stroke-width="2" />
            <text x="110" y="275" fill="#fff" text-anchor="middle" font-size="12">GPU Compute</text>
            <text x="110" y="295" fill="#aaa" text-anchor="middle" font-size="10">Calc Q, K, V</text>
            
            <!-- GPU VRAM -->
            <rect x="50" y="330" width="600" height="60" rx="4" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
            <text x="350" y="350" fill="#fff" font-weight="bold" text-anchor="middle" font-size="12">GPU VRAM (The KV Cache)</text>
            <g id="vramBlocks"></g>
            
            <g id="cacheTokens"></g>
          </svg>
        </div>
        
        <div style="margin-top: 40px; text-align: left; font-family: monospace; font-size: 14px; max-width: 700px; margin-left: auto; margin-right: auto;">
           <!-- Step 1 -->
           <div style="margin-bottom: 30px;">
              <div style="font-size: 16px; color: #4dabf7; font-weight: bold; margin-bottom: 8px;">Step 1: Compute New Vectors</div>
              <div style="padding: 12px; background: #222; border-left: 3px solid #e67700; margin-bottom: 10px; line-height: 1.5;">
                 q<sub>new</sub> = x<sub>new</sub> · W<sub>q</sub><br>
                 k<sub>new</sub> = x<sub>new</sub> · W<sub>k</sub><br>
                 v<sub>new</sub> = x<sub>new</sub> · W<sub>v</sub>
              </div>
              <div style="color: #aaa; line-height: 1.4;">
                 Instead of passing the entire historical sequence through the linear projections, we only multiply the <i>single new token</i> (x<sub>new</sub>) by the weight matrices to get its specific Query, Key, and Value vectors.
              </div>
           </div>
           
           <!-- Step 2 -->
           <div style="margin-bottom: 30px;">
              <div style="font-size: 16px; color: #4dabf7; font-weight: bold; margin-bottom: 8px;">Step 2: Update the Cache</div>
              <div style="padding: 12px; background: #222; border-left: 3px solid #e67700; margin-bottom: 10px; line-height: 1.5;">
                 K<sub>cache</sub> = [K<sub>past</sub> , k<sub>new</sub>]<br>
                 V<sub>cache</sub> = [V<sub>past</sub> , v<sub>new</sub>]
              </div>
              <div style="color: #aaa; line-height: 1.4; margin-bottom: 15px;">
                 We read the historical Keys and Values from GPU RAM (K<sub>past</sub>), and physically <b style="color:#fff; background:#e67700; padding: 2px 6px; border-radius:3px;">CONCATENATE</b> our newly computed k<sub>new</sub> and v<sub>new</sub> to the end of them. We then save this updated, slightly longer cache back to RAM for the next turn.
              </div>
              <!-- Concatenation visual -->
              <svg viewBox="0 0 700 80" width="100%" height="80">
                 <rect x="0" y="10" width="200" height="40" rx="4" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
                 <text x="100" y="35" fill="#fff" font-weight="bold" text-anchor="middle">K_past (Tokens 0..N-1)</text>
                 
                 <text x="220" y="35" fill="#fff" font-weight="bold" font-size="20" text-anchor="middle">+</text>
                 
                 <rect x="240" y="10" width="60" height="40" rx="4" fill="#2b8a3e" />
                 <text x="270" y="35" fill="#fff" font-weight="bold" text-anchor="middle">k_new</text>
                 
                 <path d="M 315 30 L 345 30" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
                 
                 <rect x="360" y="10" width="260" height="40" rx="4" fill="#2c2c2c" stroke="#e67700" stroke-width="2" stroke-dasharray="2 2" />
                 <rect x="360" y="10" width="200" height="40" rx="4" fill="#1a365d" />
                 <rect x="560" y="10" width="60" height="40" rx="4" fill="#2b8a3e" />
                 <text x="490" y="35" fill="#fff" font-weight="bold" text-anchor="middle">K_cache (Tokens 0..N)</text>
              </svg>
           </div>
           
           <!-- Step 3 -->
           <div style="margin-bottom: 30px;">
              <div style="font-size: 16px; color: #4dabf7; font-weight: bold; margin-bottom: 8px;">Step 3: Compute Attention</div>
              <div style="padding: 12px; background: #222; border-left: 3px solid #e67700; margin-bottom: 10px; line-height: 1.5;">
                 Attention = softmax( q<sub>new</sub> · K<sub>cache</sub><sup>T</sup> ) · V<sub>cache</sub>
              </div>
              <div style="color: #aaa; line-height: 1.4;">
                 We take our single new Query vector (q<sub>new</sub>) and take the dot product with the <i>entire</i> cached Key matrix. This gives the new token attention scores for every token in the past. We softmax those scores, and multiply by the cached Value matrix to get the final output!
              </div>
           </div>
           
           <!-- The Scaling Problem -->
           <div style="margin-top: 50px; padding: 20px; background: #1a1a1a; border-radius: 8px; border: 1px solid #444;">
              <div style="font-size: 18px; color: #fff; font-weight: bold; margin-bottom: 10px;">The Frontier Bottleneck: Memory</div>
              <div style="color: #aaa; line-height: 1.5; margin-bottom: 15px;">
                 This KV Cache makes generation blazingly fast, but it takes up an immense amount of space. Remember, you don't just need one cache—you need a KV cache for <b>every attention head</b>, across <b>every layer</b> of the model, for <b>every active user</b>.
              </div>
              <div style="font-family: monospace; background: #000; padding: 10px; border-radius: 4px; color: #4dabf7; margin-bottom: 15px;">
                 Cache Size = (Context Length) × (Layers) × (Heads) × (Head Dim) × 2 (for K and V) × 2 (Bytes per float16)
              </div>
              <div style="color: #aaa; line-height: 1.5;">
                 For a model like LLaMA-3 70B operating at a 100k context window, a single user's KV cache requires <b>~40 GB of VRAM</b> just to hold the past context. If you want to serve 100 concurrent users, you need 4 Terabytes of GPU memory <i>just for the cache</i>.<br><br>
                 
                 <b>The DeepSeek Breakthrough:</b> This massive memory footprint is why <b>DeepSeek-V2 / V3</b> shocked the industry by inventing <b>Multi-Head Latent Attention (MLA)</b>. Instead of storing the full K and V matrices, MLA compresses the KV cache into a single, tiny, low-dimensional latent vector and only "decompresses" it when computing attention. This drastically reduces the KV cache size by over <b>90%</b>, allowing them to serve massive models with huge contexts at a fraction of the hardware cost!
              </div>
           </div>
           <!-- References -->
           <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #333; color: #888; font-size: 12px; line-height: 1.5;">
              <b>References:</b><br>
              Fast decoding via Key-Value caching was popularized in early sequence generation papers and is now a universal standard in all LLM inference engines (like vLLM, TensorRT-LLM).
           </div>
        </div>
      </div>
    `;
  }
  
  host.innerHTML = render();

  const noCacheTokens = host.querySelector('#noCacheTokens');
  const noCacheStatus = host.querySelector('#noCacheStatus');
  
  const cacheTokens = host.querySelector('#cacheTokens');
  const vramBlocks = host.querySelector('#vramBlocks');
  const cacheStatus = host.querySelector('#cacheStatus');
  
  if (!noCacheTokens) return;

  const MAX_TOKENS = 8;
  const TICK_SPEED = 20; // smaller is faster

  // State for No Cache
  let ncTargetToken = 0; // the token we are currently trying to generate
  let ncCurrentCompute = 0; // the past token we are currently forced to recompute
  let ncTicks = 0;
  
  // State for Cache
  let cTargetToken = 0;
  let cTicks = 0;
  
  let animId;
  
  function tick() {
    // --- NO CACHE LOGIC ---
    // Takes time proportional to ncTargetToken + 1
    ncTicks++;
    if (ncTicks > TICK_SPEED) {
      ncTicks = 0;
      ncCurrentCompute++;
      if (ncCurrentCompute > ncTargetToken) {
        // finished generating the target token!
        ncTargetToken++;
        ncCurrentCompute = 0;
        if (ncTargetToken >= MAX_TOKENS) {
          ncTargetToken = 0; // loop
        }
      }
    }
    
    noCacheStatus.textContent = `Generating Token ${ncTargetToken} (Recomputing pos ${ncCurrentCompute} / ${ncTargetToken})`;
    
    let ncOut = "";
    for (let i = 0; i < MAX_TOKENS; i++) {
      const x = 200 + i * 50;
      const y = 90;
      if (i < ncTargetToken) {
        // already generated
        ncOut += `<rect x="${x}" y="${y}" width="40" height="20" rx="4" fill="#555" />
                  <text x="${x+20}" y="${y+14}" fill="#fff" font-size="10" text-anchor="middle">T${i}</text>`;
      } else if (i === ncTargetToken) {
        // generating this one
        ncOut += `<rect x="${x}" y="${y}" width="40" height="20" rx="4" fill="#2c2c2c" stroke="#555" stroke-dasharray="2 2" />`;
      }
      
      // highlight the one being computed
      if (i === ncCurrentCompute && ncTargetToken < MAX_TOKENS) {
         ncOut += `<rect x="${x}" y="${y-30}" width="40" height="20" rx="4" fill="#e67700" />
                   <text x="${x+20}" y="${y-16}" fill="#fff" font-size="10" text-anchor="middle">Calc!</text>`;
         // arrow from compute
         ncOut += `<path d="M 170 100 L ${x} ${y-20}" fill="none" stroke="#e67700" stroke-width="2" marker-end="url(#arrow)" />`;
      }
    }
    noCacheTokens.innerHTML = ncOut;
    
    // --- CACHE LOGIC ---
    // Takes exactly 1 tick per token because we don't recompute
    cTicks++;
    if (cTicks > TICK_SPEED) {
      cTicks = 0;
      cTargetToken++;
      if (cTargetToken >= MAX_TOKENS) {
        cTargetToken = 0; // loop
      }
    }
    
    cacheStatus.textContent = `Generating Token ${cTargetToken} (O(1) compute, reading ${cTargetToken} past KV from RAM)`;
    
    let cOut = "";
    let vramOut = "";
    for (let i = 0; i < MAX_TOKENS; i++) {
      const x = 200 + i * 50;
      const y = 270;
      const vx = 100 + i * 50;
      
      if (i < cTargetToken) {
        // already generated
        cOut += `<rect x="${x}" y="${y}" width="40" height="20" rx="4" fill="#555" />
                 <text x="${x+20}" y="${y+14}" fill="#fff" font-size="10" text-anchor="middle">T${i}</text>`;
                 
        // in VRAM
        vramOut += `<rect x="${vx}" y="360" width="40" height="20" rx="4" fill="#4dabf7" opacity="0.8" />
                    <text x="${vx+20}" y="374" fill="#111" font-weight="bold" font-size="10" text-anchor="middle">K,V ${i}</text>`;
                    
        // line from VRAM to compute (always visible during the tick)
        vramOut += `<path d="M ${vx+20} 360 L ${vx+20} 340 L 110 310" fill="none" stroke="#4dabf7" stroke-width="1.5" stroke-dasharray="2 2" />`;
      } else if (i === cTargetToken) {
        // generating this one
        cOut += `<rect x="${x}" y="${y}" width="40" height="20" rx="4" fill="#2c2c2c" stroke="#555" stroke-dasharray="2 2" />`;
        
        // highlight the single compute
        cOut += `<rect x="${x}" y="${y-30}" width="40" height="20" rx="4" fill="#2b8a3e" />
                 <text x="${x+20}" y="${y-16}" fill="#fff" font-size="10" text-anchor="middle">Calc!</text>`;
        // arrow from compute
        cOut += `<path d="M 170 280 L ${x} ${y-20}" fill="none" stroke="#2b8a3e" stroke-width="2" marker-end="url(#arrow)" />`;
        
        // write to vram arrow
        vramOut += `<path d="M 110 310 L ${vx+20} 340 L ${vx+20} 360" fill="none" stroke="#e67700" stroke-width="2" marker-end="url(#arrow)" />
                    <text x="${vx+20}" y="335" fill="#e67700" font-size="10" text-anchor="middle">Save KV</text>`;
      }
    }
    
    cacheTokens.innerHTML = cOut;
    vramBlocks.innerHTML = vramOut;
    
    animId = requestAnimationFrame(tick);
    host._animId = animId;
  }
  
  if (host._animId) cancelAnimationFrame(host._animId);
  tick();
}
