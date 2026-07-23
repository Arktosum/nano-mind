/* speculative_decoding.js — Speculative Decoding */
export function renderSpeculativeDecodingLesson(host, ctx) {
  function render() {
    const w = 700;
    const h = 400;
    
    return `
      <div class="lz-diagram">
        <div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333; position:relative;">
          <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" id="specSvg">
            
            <!-- Draft Model -->
            <rect x="100" y="80" width="120" height="60" rx="8" fill="#333" stroke="#888" stroke-width="2" />
            <text x="160" y="105" fill="#fff" font-weight="bold" text-anchor="middle" font-size="14">Draft Model</text>
            <text x="160" y="125" fill="#aaa" text-anchor="middle" font-size="11">Tiny & Fast (1B)</text>
            
            <!-- Target Model -->
            <rect x="100" y="220" width="120" height="100" rx="8" fill="#1a365d" stroke="#4dabf7" stroke-width="3" />
            <text x="160" y="265" fill="#fff" font-weight="bold" text-anchor="middle" font-size="16">Target Model</text>
            <text x="160" y="285" fill="#4dabf7" text-anchor="middle" font-size="11">Massive & Slow (70B)</text>

            <!-- Sequence text -->
            <text x="350" y="195" fill="#fff" font-weight="bold" text-anchor="middle" font-size="20" id="specText">Alice was beginning to get very </text>
            <g id="specTokens"></g>
            <g id="specChecks"></g>
            
            <text x="350" y="360" fill="#e67700" font-size="14" font-style="italic" text-anchor="middle" id="specStatus">Waiting...</text>

          </svg>
        </div>
        
        <div class="lz-notes" >
          <h3>Speculative Decoding</h3>
          <p class="note-meta"><i>Developed: Nov 2022 (Leviathan, Kalman, Matias)</i></p>
          
          <p class="lz-notes-prob">1. The Problem</p>
          <p>During generation, LLMs are memory-bandwidth bound. Loading massive 70B parameter weights into the compute cores takes 10 milliseconds, but the actual math to predict 1 token takes just 0.1 milliseconds. You waste 99% of your time waiting for weights to load. However, loading the weights to evaluate 5 tokens in parallel takes the <i>exact same 10ms</i>.</p>
          
          <p class="lz-notes-moti">2. The Motivation</p>
          <p>If evaluating 5 tokens costs the same as evaluating 1, what if we had a cheap way to guess the next 5 tokens before we run the massive model?</p>
          
          <p class="lz-notes-impl">3. The Implementation</p>
          <p style="margin-top: 0;">We run a tiny 1B parameter "Draft" model. Because it's small, its weights load instantly, and it rapidly hallucinates 5 tokens. We then pass those 5 tokens into the massive 70B "Target" model in a single batch. The 70B model evaluates all 5 simultaneously. If it agrees with the draft model's math, we accept all 5 tokens at once—generating 5 tokens for the latency cost of 1.</p>
        </div>
      </div>
    `;
  }
  
  host.innerHTML = render();
  
  const tokensGroup = host.querySelector('#specTokens');
  const checksGroup = host.querySelector('#specChecks');
  const statusTxt = host.querySelector('#specStatus');
  if (!tokensGroup) return;
  
  let t = 0;
  let animId;
  const draftWords = ["tired ", "of ", "sitting ", "by "];
  
  function tick() {
    t += 0.015;
    if (t > 4) t = 0;
    
    let tokens = '';
    let checks = '';
    
    let baseY = 195;
    let baseX = 475;
    
    // Phase 1: Drafting
    if (t <= 2) {
      const p = t / 2; // 0 to 1
      statusTxt.textContent = "Draft model rapidly generating 4 tokens...";
      
      let wCount = Math.floor(p * 5); // 0 to 4
      if (wCount > 4) wCount = 4;
      
      for (let i = 0; i < wCount; i++) {
        let x = baseX + (i * 45);
        tokens += `<text x="${x}" y="${baseY}" fill="#888" font-size="20" opacity="0.8">${draftWords[i]}</text>`;
      }
      
      if (wCount > 0) {
        let x = baseX + ((wCount-1) * 45);
        tokens += `<path d="M 230 110 L ${x} 175" fill="none" stroke="#888" stroke-dasharray="4 4" stroke-width="2" />`;
      }
    } 
    // Phase 2: Verifying
    else if (t > 2 && t <= 3) {
      statusTxt.textContent = "Target model verifying all 4 tokens in parallel...";
      for (let i = 0; i < 4; i++) {
        let x = baseX + (i * 45);
        tokens += `<text x="${x}" y="${baseY}" fill="#888" font-size="20">${draftWords[i]}</text>`;
        tokens += `<path d="M 230 270 L ${x+10} 210" fill="none" stroke="#4dabf7" stroke-dasharray="4 4" stroke-width="2" />`;
      }
    }
    // Phase 3: Acceptance
    else {
      statusTxt.textContent = "4 tokens accepted for the cost of 1 forward pass!";
      for (let i = 0; i < 4; i++) {
        let x = baseX + (i * 45);
        tokens += `<text x="${x}" y="${baseY}" fill="#40c057" font-weight="bold" font-size="20">${draftWords[i]}</text>`;
        checks += `<circle cx="${x+10}" cy="${baseY + 15}" r="6" fill="#40c057" />`;
        checks += `<path d="M ${x+6} ${baseY + 15} L ${x+10} ${baseY + 19} L ${x+15} ${baseY + 10}" fill="none" stroke="#fff" stroke-width="2" />`;
      }
    }
    
    tokensGroup.innerHTML = tokens;
    checksGroup.innerHTML = checks;
    
    animId = requestAnimationFrame(tick);
    host._animId = animId;
  }
  
  if (host._animId) cancelAnimationFrame(host._animId);
  tick();
}
