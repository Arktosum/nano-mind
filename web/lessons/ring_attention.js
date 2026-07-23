/* ring_attention.js — Context Parallelism for 1M context */
export function renderRingAttentionLesson(host, ctx) {
  function render() {
    const w = 700;
    const h = 400;
    
    // We will render an SVG string that will be updated by a timer.
    // 4 GPUs placed in a circle or square.
    // GPU 1: top left (x=150, y=100)
    // GPU 2: top right (x=450, y=100)
    // GPU 3: bottom right (x=450, y=250)
    // GPU 4: bottom left (x=150, y=250)
    
    // The KV blocks moving between them.
    
    return `
      <div class="lz-diagram">
        <div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333; position:relative;">
          <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" id="ringSvg">
            <!-- Definitions for arrows -->
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
              </marker>
            </defs>

            <!-- Connections -->
            <path d="M 230 140 L 410 140" fill="none" stroke="#444" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrow)" />
            <path d="M 490 180 L 490 230" fill="none" stroke="#444" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrow)" />
            <path d="M 410 290 L 230 290" fill="none" stroke="#444" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrow)" />
            <path d="M 150 230 L 150 180" fill="none" stroke="#444" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrow)" />

            <!-- GPUs -->
            <rect x="110" y="100" width="80" height="80" rx="8" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
            <text x="150" y="125" fill="#fff" font-weight="bold" text-anchor="middle" font-size="14">GPU 1</text>
            <text x="150" y="145" fill="#aaa" text-anchor="middle" font-size="12">Holds Q1</text>
            
            <rect x="450" y="100" width="80" height="80" rx="8" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
            <text x="490" y="125" fill="#fff" font-weight="bold" text-anchor="middle" font-size="14">GPU 2</text>
            <text x="490" y="145" fill="#aaa" text-anchor="middle" font-size="12">Holds Q2</text>
            
            <rect x="450" y="250" width="80" height="80" rx="8" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
            <text x="490" y="275" fill="#fff" font-weight="bold" text-anchor="middle" font-size="14">GPU 3</text>
            <text x="490" y="295" fill="#aaa" text-anchor="middle" font-size="12">Holds Q3</text>
            
            <rect x="110" y="250" width="80" height="80" rx="8" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
            <text x="150" y="275" fill="#fff" font-weight="bold" text-anchor="middle" font-size="14">GPU 4</text>
            <text x="150" y="295" fill="#aaa" text-anchor="middle" font-size="12">Holds Q4</text>

            <!-- Dynamic KV Blocks will be injected here -->
            <g id="kvBlocks"></g>
            
            <text x="320" y="215" fill="#e67700" font-weight="bold" text-anchor="middle" font-size="18" id="ringStatus">Computing Local Attention...</text>
          </svg>
        </div>
        
        <div class="lz-notes" >
          <h3 style="color: #fff; margin-top: 0;">Overlapping Communication and Computation</h3>
          <p>The real magic of Ring Attention is how it <strong>hides network latency</strong>. In a naive setup, GPUs calculate attention and then sit idle while waiting for the next Key/Value blocks to travel over the network. This delay becomes a massive bottleneck as you add more GPUs.</p>
          <p>Because modern GPUs have network controllers that operate independently from their math compute cores, Ring Attention can do both at once:</p>
          <ul style="margin-top: 8px; margin-bottom: 12px; padding-left: 20px;">
            <li><strong style="color: #4dabf7;">Computation:</strong> A GPU computes the attention scores for its current block of tokens.</li>
            <li><strong style="color: #e67700;">Communication:</strong> <em>At the exact same time</em>, it transmits its current KV block to the next GPU and receives the incoming KV block from the previous GPU.</li>
          </ul>
          <p style="margin-bottom: 0;">Since the network transfer happens in the background, the new data arrives <em>before</em> the GPU finishes its current math. From the compute cores' perspective, the wait time is practically zero! This eliminates network congestion and allows the model to <strong>scale near-infinitely</strong> to support massive context windows.</p>
        </div>
      </div>
    `;
  }
  
  host.innerHTML = render();

  // Animation logic
  const svg = host.querySelector('#kvBlocks');
  const statusTxt = host.querySelector('#ringStatus');
  
  if (!svg) return;

  const positions = [
    { x: 150, y: 165 }, // GPU 1
    { x: 490, y: 165 }, // GPU 2
    { x: 490, y: 315 }, // GPU 3
    { x: 150, y: 315 }, // GPU 4
  ];
  
  const blocks = [
    { id: 1, label: "KV1", color: "#e67700" },
    { id: 2, label: "KV2", color: "#2b8a3e" },
    { id: 3, label: "KV3", color: "#c92a2a" },
    { id: 4, label: "KV4", color: "#6741d9" },
  ];

  let step = 0; // 0 = computing, 1 = passing, 2 = computing, etc.
  let t = 0;
  
  let animId;
  function tick() {
    t += 0.015;
    
    // Each cycle is: compute (t: 0 to 1), pass (t: 1 to 2)
    const cycleTime = t % 2;
    const isPassing = cycleTime > 1;
    const computePhase = Math.floor(t / 2); // 0, 1, 2, 3
    
    if (computePhase >= 4) {
      t = 0; // reset animation
    }

    if (isPassing) {
      statusTxt.textContent = "Passing KV Blocks to neighbors...";
    } else {
      statusTxt.textContent = "Computing Attention...";
    }

    let out = "";
    
    for (let i = 0; i < 4; i++) {
      // Find where this block currently is.
      // At phase 0, block i is at GPU i.
      // At phase 1, block i is at GPU (i+1)%4.
      const currentGpuIdx = (i + computePhase) % 4;
      const nextGpuIdx = (currentGpuIdx + 1) % 4;
      
      let bx = positions[currentGpuIdx].x;
      let by = positions[currentGpuIdx].y;
      
      if (isPassing) {
        // Interpolate between current and next
        const progress = cycleTime - 1; // 0 to 1
        // Easing function (smoothstep)
        const ease = progress * progress * (3 - 2 * progress);
        
        bx = bx + (positions[nextGpuIdx].x - bx) * ease;
        by = by + (positions[nextGpuIdx].y - by) * ease;
      }
      
      // Draw block
      out += `
        <rect x="${bx - 20}" y="${by - 10}" width="40" height="20" rx="4" fill="${blocks[i].color}" />
        <text x="${bx}" y="${by + 4}" fill="#fff" font-size="10" text-anchor="middle" font-weight="bold">${blocks[i].label}</text>
      `;
      
      // If computing, draw flash/glow on the GPU to show work
      if (!isPassing && cycleTime < 1) {
         const glowAlpha = Math.sin(cycleTime * Math.PI);
         if (glowAlpha > 0) {
           out += `<circle cx="${positions[currentGpuIdx].x}" cy="${positions[currentGpuIdx].y - 30}" r="${20 + glowAlpha * 10}" fill="${blocks[i].color}" opacity="${glowAlpha * 0.3}" />`;
         }
      }
    }
    
    svg.innerHTML = out;
    
    // Store animId on the host so it can be cleaned up if lesson changes
    animId = requestAnimationFrame(tick);
    host._animId = animId;
  }
  
  // Cleanup previous animation if exists
  if (host._animId) cancelAnimationFrame(host._animId);
  tick();
}
