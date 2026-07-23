/* moe.js — Mixture of Experts (MoE) */
export function renderMoeLesson(host, ctx) {
  function render() {
    const w = 700;
    const h = 400;
    
    return `
      <div class="lz-diagram">
        <div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333; position:relative;">
          <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" id="moeSvg">
            <!-- Definitions for arrows -->
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
              </marker>
              <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            <!-- Base connections (Router to Experts) -->
            <g id="routerLines"></g>
            
            <!-- Router -->
            <rect x="300" y="280" width="100" height="40" rx="8" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
            <text x="350" y="305" fill="#fff" font-weight="bold" text-anchor="middle" font-size="14">Router</text>

            <!-- 8 Experts -->
            <g id="expertsGroup"></g>
            
            <!-- Recombiner (Add) -->
            <circle cx="350" cy="80" r="20" fill="#333" stroke="#888" stroke-width="2" />
            <text x="350" y="86" fill="#fff" font-weight="bold" text-anchor="middle" font-size="18">+</text>
            
            <!-- Recombiner lines -->
            <g id="recombineLines"></g>

            <!-- Dynamic Token -->
            <circle cx="350" cy="370" r="12" fill="#e67700" id="moeToken" />
            <text x="350" y="374" fill="#fff" font-size="10" text-anchor="middle" id="moeTokenTxt">x</text>
            
            <!-- Animated tokens going to experts -->
            <g id="moeSubTokens"></g>
            
          </svg>
        </div>
        
        <div class="lz-notes" >
          <h3>Mixture of Experts (MoE)</h3>
          <p class="note-meta"><i>Developed: 2021 (Applied to Transformers by Fedus, Zoph, Shazeer — "Switch Transformers")</i></p>
          
          <p class="lz-notes-prob">1. The Problem</p>
          <p>Model capability scales predictably with parameter count, but activating 100 Billion parameters for every single token costs too much compute (FLOPs) and is too slow for real-time generation.</p>
          
          <p class="lz-notes-moti">2. The Motivation</p>
          <p>A human brain doesn't use the visual cortex to solve a math problem. What if a neural network could conditionally activate only the specialized parts of its brain needed for the current word?</p>
          
          <p class="lz-notes-impl">3. The Implementation</p>
          <p style="margin-top: 0;">The massive Feed-Forward Network (FFN) is shattered into $N$ smaller "Expert" FFNs. A trainable "Router" network predicts a probability distribution over the experts for the incoming token. Only the Top-K (usually 2) experts are multiplied. The outputs are then weighted by the router's probabilities and summed together. This allows a 100B parameter model to run at the speed of a 15B parameter model.</p>
        </div>
      </div>
    `;
  }
  
  host.innerHTML = render();

  const svg = host.querySelector('#moeSvg');
  if (!svg) return;

  const numExperts = 8;
  const experts = [];
  const expertWidth = 60;
  const expertSpacing = 20;
  const totalWidth = (numExperts * expertWidth) + ((numExperts - 1) * expertSpacing);
  const startX = (700 - totalWidth) / 2;
  
  let expertsHTML = '';
  let routerLinesHTML = '';
  let recombineLinesHTML = '';
  
  for (let i = 0; i < numExperts; i++) {
    const cx = startX + (i * (expertWidth + expertSpacing)) + (expertWidth/2);
    experts.push({ cx, cy: 180 });
    
    routerLinesHTML += `<path d="M 350 280 L ${cx} 210" fill="none" stroke="#333" stroke-width="2" id="line_to_${i}" />`;
    recombineLinesHTML += `<path d="M ${cx} 150 L 350 100" fill="none" stroke="#333" stroke-width="2" id="line_from_${i}" />`;
    
    expertsHTML += `
      <rect x="${cx - expertWidth/2}" y="150" width="${expertWidth}" height="60" rx="4" fill="#222" stroke="#444" stroke-width="2" id="expert_${i}" />
      <text x="${cx}" y="180" fill="#666" font-size="12" text-anchor="middle" id="expertTxt_${i}">E${i}</text>
    `;
  }
  
  host.querySelector('#routerLines').innerHTML = routerLinesHTML;
  host.querySelector('#recombineLines').innerHTML = recombineLinesHTML;
  host.querySelector('#expertsGroup').innerHTML = expertsHTML;

  // Animation logic
  let t = 0;
  let activeExperts = [2, 5];
  let animId;
  const token = host.querySelector('#moeToken');
  const tokenTxt = host.querySelector('#moeTokenTxt');
  const subTokensGroup = host.querySelector('#moeSubTokens');
  
  function tick() {
    t += 0.01;
    if (t > 4) {
      t = 0;
      // Pick 2 random new experts
      let e1 = Math.floor(Math.random() * numExperts);
      let e2 = Math.floor(Math.random() * numExperts);
      while (e1 === e2) e2 = Math.floor(Math.random() * numExperts);
      activeExperts = [e1, e2];
      
      // Reset visuals
      for (let i=0; i<numExperts; i++) {
        host.querySelector(`#expert_${i}`).setAttribute('fill', '#222');
        host.querySelector(`#expert_${i}`).setAttribute('stroke', '#444');
        host.querySelector(`#expertTxt_${i}`).setAttribute('fill', '#666');
        host.querySelector(`#line_to_${i}`).setAttribute('stroke', '#333');
        host.querySelector(`#line_from_${i}`).setAttribute('stroke', '#333');
      }
    }
    
    // Phase 0-1: Token moves up to router
    if (t <= 1) {
      token.setAttribute('cy', 370 - (t * 50));
      token.setAttribute('opacity', 1);
      tokenTxt.setAttribute('cy', 374 - (t * 50));
      tokenTxt.setAttribute('opacity', 1);
      subTokensGroup.innerHTML = '';
    }
    // Phase 1-2: Router processing & highlighting experts
    else if (t > 1 && t <= 2) {
      token.setAttribute('cy', 320); // hidden behind router
      token.setAttribute('opacity', 0);
      tokenTxt.setAttribute('opacity', 0);
      
      const p = t - 1;
      activeExperts.forEach(e => {
        host.querySelector(`#line_to_${e}`).setAttribute('stroke', p > 0.5 ? '#4dabf7' : '#333');
        if (p > 0.8) {
          host.querySelector(`#expert_${e}`).setAttribute('fill', '#1a365d');
          host.querySelector(`#expert_${e}`).setAttribute('stroke', '#4dabf7');
          host.querySelector(`#expertTxt_${e}`).setAttribute('fill', '#fff');
        }
      });
      subTokensGroup.innerHTML = '';
    }
    // Phase 2-3: Tokens moving to experts and computing
    else if (t > 2 && t <= 3) {
      const p = t - 2;
      let subs = '';
      activeExperts.forEach(e => {
        const cx = experts[e].cx;
        const cy = 280 - (p * 70);
        subs += `<circle cx="${350 + (cx-350)*p}" cy="${cy}" r="10" fill="#4dabf7" />`;
      });
      subTokensGroup.innerHTML = subs;
    }
    // Phase 3-4: Tokens moving to recombiner
    else if (t > 3 && t <= 4) {
      const p = t - 3;
      let subs = '';
      activeExperts.forEach(e => {
        const cx = experts[e].cx;
        host.querySelector(`#line_from_${e}`).setAttribute('stroke', '#e67700');
        subs += `<circle cx="${cx + (350-cx)*p}" cy="${150 - (p*70)}" r="10" fill="#e67700" />`;
      });
      subTokensGroup.innerHTML = subs;
    }
    
    animId = requestAnimationFrame(tick);
    host._animId = animId;
  }
  
  if (host._animId) cancelAnimationFrame(host._animId);
  tick();
}
