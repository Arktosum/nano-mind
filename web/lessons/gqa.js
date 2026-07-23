/* gqa.js — Grouped-Query Attention */
export function renderGQALesson(host, ctx) {
  function render() {
    const w = 700;
    const h = 400;
    
    return `
      <div class="lz-diagram">
        <div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333; position:relative;">
          <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" id="gqaSvg">
            
            <text x="350" y="30" fill="#fff" font-weight="bold" text-anchor="middle" font-size="16">Multi-Head Attention (MHA) vs Grouped-Query Attention (GQA)</text>
            
            <!-- Standard MHA (Left) -->
            <rect x="50" y="80" width="250" height="280" rx="8" class="svg-box" />
            <text x="175" y="105" fill="#aaa" text-anchor="middle" font-size="14" font-weight="bold">Standard Attention</text>
            <text x="175" y="125" fill="#888" text-anchor="middle" font-size="12">1 Q per 1 K/V</text>
            
            <g id="mhaGroup"></g>
            
            <!-- GQA (Right) -->
            <rect x="400" y="80" width="250" height="280" rx="8" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
            <text x="525" y="105" fill="#4dabf7" text-anchor="middle" font-size="14" font-weight="bold">Grouped-Query Attention</text>
            <text x="525" y="125" fill="#888" text-anchor="middle" font-size="12">4 Qs per 1 K/V (Shrinks KV Cache 75%)</text>
            
            <g id="gqaGroup"></g>

          </svg>
        </div>
        
        <div class="lz-notes" >
          <h3>Grouped-Query Attention (GQA)</h3>
          <p class="note-meta"><i>Developed: May 2023 (Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebron, Sanghai)</i></p>
          
          <p class="lz-notes-prob">1. The Problem</p>
          <p>To generate a new token, the model needs to attend to all past tokens. This means we must store the Key (K) and Value (V) vectors for every token in GPU memory (the KV Cache). In standard Multi-Head Attention, every Query head has its own unique K and V head. For long contexts, this consumes gigabytes of VRAM per user.</p>
          
          <p class="lz-notes-moti">2. The Motivation</p>
          <p>Do we really need 32 unique Key and Value representations? What if multiple Query heads could "look" at the same Key/Value pair without losing too much reasoning quality?</p>
          
          <p class="lz-notes-impl">3. The Implementation</p>
          <p style="margin-top: 0;">GQA divides the Query heads into groups (e.g., 8 groups of 4). Each group of 4 Query heads shares a single Key head and a single Value head. This immediately shrinks the size of the KV cache by 75%, allowing much larger batch sizes and faster generation, which is why Llama 3 relies heavily on it.</p>
        </div>
      </div>
    `;
  }
  
  host.innerHTML = render();
  
  // Render MHA (8 Q, 8 K, 8 V)
  let mha = '';
  for (let i = 0; i < 8; i++) {
    const y = 160 + (i * 25);
    mha += `<rect x="70" y="${y-10}" width="40" height="18" rx="4" fill="#e67700" />`;
    mha += `<text x="90" y="${y+3}" fill="#fff" font-size="10" text-anchor="middle">Q${i}</text>`;
    
    mha += `<rect x="150" y="${y-10}" width="40" height="18" rx="4" fill="#40c057" />`;
    mha += `<text x="170" y="${y+3}" fill="#fff" font-size="10" text-anchor="middle">K${i}</text>`;
    
    mha += `<rect x="230" y="${y-10}" width="40" height="18" rx="4" fill="#be4bdb" />`;
    mha += `<text x="250" y="${y+3}" fill="#fff" font-size="10" text-anchor="middle">V${i}</text>`;
    
    mha += `<path d="M 110 ${y-1} L 150 ${y-1}" fill="none" stroke="#666" stroke-width="1" />`;
  }
  host.querySelector('#mhaGroup').innerHTML = mha;
  
  // Render GQA (8 Q, 2 K, 2 V)
  let gqa = '';
  for (let i = 0; i < 8; i++) {
    const y = 160 + (i * 25);
    gqa += `<rect x="420" y="${y-10}" width="40" height="18" rx="4" fill="#e67700" />`;
    gqa += `<text x="440" y="${y+3}" fill="#fff" font-size="10" text-anchor="middle">Q${i}</text>`;
  }
  
  // Group 0 (Q0-Q3) connects to K0, V0
  gqa += `<rect x="520" y="185" width="40" height="30" rx="4" fill="#40c057" />`;
  gqa += `<text x="540" y="205" fill="#fff" font-size="12" text-anchor="middle" font-weight="bold">K0</text>`;
  gqa += `<rect x="590" y="185" width="40" height="30" rx="4" fill="#be4bdb" />`;
  gqa += `<text x="610" y="205" fill="#fff" font-size="12" text-anchor="middle" font-weight="bold">V0</text>`;
  for (let i=0; i<4; i++) {
    const y = 160 + (i * 25);
    gqa += `<path d="M 460 ${y-1} C 490 ${y-1} 490 200 520 200" class="svg-arrow" />`;
  }
  
  // Group 1 (Q4-Q7) connects to K1, V1
  gqa += `<rect x="520" y="285" width="40" height="30" rx="4" fill="#40c057" />`;
  gqa += `<text x="540" y="305" fill="#fff" font-size="12" text-anchor="middle" font-weight="bold">K1</text>`;
  gqa += `<rect x="590" y="285" width="40" height="30" rx="4" fill="#be4bdb" />`;
  gqa += `<text x="610" y="305" fill="#fff" font-size="12" text-anchor="middle" font-weight="bold">V1</text>`;
  for (let i=4; i<8; i++) {
    const y = 160 + (i * 25);
    gqa += `<path d="M 460 ${y-1} C 490 ${y-1} 490 300 520 300" class="svg-arrow" />`;
  }
  
  host.querySelector('#gqaGroup').innerHTML = gqa;
}
