/* flash_attention.js — FlashAttention */
export function renderFlashAttentionLesson(host, ctx) {
  function render() {
    const w = 700;
    const h = 450;
    
    return `
      <div class="lz-diagram">
        <div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333; position:relative;">
          <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" id="faSvg">
            <!-- Definitions -->
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
              </marker>
            </defs>

            <!-- HBM (Main Memory) -->
            <rect x="50" y="50" width="600" height="80" rx="8" class="svg-box" />
            <text x="350" y="75" fill="#fff" font-weight="bold" text-anchor="middle" font-size="16">GPU HBM (High Bandwidth Memory - Huge but Slow)</text>
            
            <rect x="170" y="90" width="60" height="30" rx="4" fill="#333" stroke="#555" />
            <text x="200" y="110" fill="#aaa" font-size="12" text-anchor="middle">Q</text>
            <rect x="240" y="90" width="60" height="30" rx="4" fill="#333" stroke="#555" />
            <text x="270" y="110" fill="#aaa" font-size="12" text-anchor="middle">K</text>
            <rect x="310" y="90" width="60" height="30" rx="4" fill="#333" stroke="#555" />
            <text x="340" y="110" fill="#aaa" font-size="12" text-anchor="middle">V</text>
            
            <rect x="420" y="90" width="120" height="30" rx="4" fill="#4dabf7" stroke="#228be6" opacity="0.3" id="faHbmOutput" />
            <text x="480" y="110" fill="#4dabf7" font-size="12" text-anchor="middle">Output (O)</text>

            <!-- SRAM (On-chip Memory) -->
            <rect x="160" y="240" width="380" height="150" rx="8" class="svg-v" opacity="0.1" />
            <text x="350" y="265" fill="#40c057" font-weight="bold" text-anchor="middle" font-size="16">GPU SRAM (On-chip - Tiny but Blazing Fast)</text>
            
            <!-- SRAM Blocks -->
            <rect x="180" y="300" width="40" height="40" rx="4" fill="#111" stroke="#40c057" stroke-dasharray="2 2" />
            <text x="200" y="325" fill="#fff" font-size="10" text-anchor="middle">Q block</text>
            <rect x="250" y="300" width="40" height="40" rx="4" fill="#111" stroke="#40c057" stroke-dasharray="2 2" />
            <text x="270" y="325" fill="#fff" font-size="10" text-anchor="middle">K block</text>
            <rect x="320" y="300" width="40" height="40" rx="4" fill="#111" stroke="#40c057" stroke-dasharray="2 2" />
            <text x="340" y="325" fill="#fff" font-size="10" text-anchor="middle">V block</text>
            
            <rect x="450" y="300" width="60" height="40" rx="4" fill="#4dabf7" stroke="#228be6" />
            <text x="480" y="325" fill="#fff" font-size="12" text-anchor="middle">O block</text>

            <!-- Data movement arrows -->
            <path d="M 200 130 L 200 290" fill="none" stroke="#888" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrow)" id="faReadQ" />
            <path d="M 270 130 L 270 290" fill="none" stroke="#888" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrow)" id="faReadK" />
            <path d="M 340 130 L 340 290" fill="none" stroke="#888" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrow)" id="faReadV" />
            <path d="M 480 290 L 480 130" fill="none" stroke="#888" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrow)" id="faWriteArrow" />
            
            <!-- Compute node -->
            <circle cx="405" cy="320" r="15" fill="#e67700" />
            <text x="405" y="325" fill="#fff" font-size="16" font-weight="bold" text-anchor="middle">∗</text>
            
            <!-- Animated particles -->
            <g id="faParticles"></g>
          </svg>
        </div>
        
        <div class="lz-notes" >
          <h3>FlashAttention</h3>
          <p class="note-meta"><i>Developed: May 2022 (Dao, Fu, Ermon, Rudra, Ré)</i></p>
          
          <p class="lz-notes-prob">1. The Problem: The Memory Wall</p>
          <p>In GPUs, math (FLOPs) is incredibly fast, but moving data from Main Memory (HBM) to the compute cores is extremely slow. Standard attention computes an $N \\times N$ matrix and writes it to HBM. For 100k tokens, this matrix is 10 Billion elements. Writing and reading this intermediate matrix completely chokes the GPU's memory bandwidth.</p>
          
          <p class="lz-notes-moti">2. The Motivation</p>
          <p>What if we never write the full $N \\times N$ matrix to HBM? What if we could calculate the attention scores in small chunks using the tiny, blazing-fast SRAM right next to the compute cores, and only ever write the final answer?</p>
          
          <p class="lz-notes-impl">3. The Implementation</p>
          <p style="margin-top: 0;">FlashAttention uses a mathematical trick called "Tiling". It loads a block of Queries, Keys, and Values into SRAM. It computes the dot products locally, tracks the running softmax denominator (using an algebraic trick to keep it mathematically identical to standard softmax), multiplies by Values, and writes <i>only</i> the final output vector back to HBM. By avoiding intermediate HBM reads/writes, it speeds up attention by 2-4x.</p>
        </div>
      </div>
    `;
  }
  
  host.innerHTML = render();
  
  const particlesGroup = host.querySelector('#faParticles');
  if (!particlesGroup) return;
  
  let t = 0;
  let animId;
  
  function tick() {
    t += 0.02;
    if (t > 3) t = 0; // 3 phases: Read, Compute, Write
    
    let particles = '';
    
    if (t <= 1) { // Read phase
      const p = t;
      const cy = 130 + (p * 160);
      particles += `<circle cx="200" cy="${cy}" r="4" fill="#40c057" />`;
      particles += `<circle cx="270" cy="${cy}" r="4" fill="#40c057" />`;
      particles += `<circle cx="340" cy="${cy}" r="4" fill="#40c057" />`;
      host.querySelector('#faReadQ').setAttribute('stroke', '#40c057');
      host.querySelector('#faReadK').setAttribute('stroke', '#40c057');
      host.querySelector('#faReadV').setAttribute('stroke', '#40c057');
      host.querySelector('#faWriteArrow').setAttribute('stroke', '#888');
      host.querySelector('#faHbmOutput').setAttribute('opacity', '0.3');
    } else if (t > 1 && t <= 2) { // Compute Phase
      const p = t - 1;
      const glow = Math.sin(p * Math.PI * 4);
      if (glow > 0) {
        particles += `<circle cx="405" cy="320" r="${15 + glow * 10}" fill="#e67700" opacity="0.4" />`;
      }
      host.querySelector('#faReadQ').setAttribute('stroke', '#888');
      host.querySelector('#faReadK').setAttribute('stroke', '#888');
      host.querySelector('#faReadV').setAttribute('stroke', '#888');
      host.querySelector('#faWriteArrow').setAttribute('stroke', '#888');
    } else { // Write Phase
      const p = t - 2;
      const cy = 290 - (p * 160);
      particles += `<circle cx="480" cy="${cy}" r="6" fill="#4dabf7" />`;
      host.querySelector('#faReadQ').setAttribute('stroke', '#888');
      host.querySelector('#faReadK').setAttribute('stroke', '#888');
      host.querySelector('#faReadV').setAttribute('stroke', '#888');
      host.querySelector('#faWriteArrow').setAttribute('stroke', '#4dabf7');
      host.querySelector('#faHbmOutput').setAttribute('opacity', 0.3 + (p * 0.7));
    }
    
    particlesGroup.innerHTML = particles;
    
    animId = requestAnimationFrame(tick);
    host._animId = animId;
  }
  
  if (host._animId) cancelAnimationFrame(host._animId);
  tick();
}
