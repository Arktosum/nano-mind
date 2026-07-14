/* lesson-recap.js — a high-level overview of the full architecture */

function renderRecapLesson(host, ctx) {
  
  function draw() {
    host.innerHTML =
      section("1", "the big picture") +
      note("<p>We just completed the hardest part of the Transformer: the <b>Multi-Head Self-Attention</b>. But there is a bit more to the network! Below is a full map of the Nano-Mind architecture.</p><p>You can see how the Residual Stream acts as a central highway, with information being read from it, processed, and added back in via Skip Connections. You can also see that our model actually stacks <b>3 identical blocks</b> on top of each other, allowing the network to build deep, hierarchical representations of the text!</p>") +
      drawArchitectureSVG();
  }

  function drawArchitectureSVG() {
    const w = 500;
    const h = 950;
    
    let s = `
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
        </marker>
        <marker id="arrowBlue" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#4dabf7" />
        </marker>
      </defs>
      
      <!-- Residual Stream Spine -->
      <line x1="150" y1="50" x2="150" y2="820" stroke="#4dabf7" stroke-width="4" marker-end="url(#arrowBlue)" />
      <text x="70" y="450" transform="rotate(-90 70,450)" fill="#4dabf7" font-weight="bold" font-size="14" text-anchor="middle" letter-spacing="2">RESIDUAL STREAM (64-dim)</text>

      <!-- Input Stage -->
      <rect x="50" y="50" width="200" height="40" rx="4" fill="#333" stroke="#555" />
      <text x="150" y="75" fill="#fff" text-anchor="middle" font-size="14">Token + Pos Embeddings</text>
      
      <!-- Block 0 -->
      <g transform="translate(0, 130)">
        <rect x="100" y="0" width="300" height="360" rx="8" fill="none" stroke="#666" stroke-dasharray="4 4" />
        <text x="120" y="25" fill="#aaa" font-size="16" font-weight="bold">Transformer Block 0</text>
        
        <!-- MHA Path -->
        <path d="M 150 40 L 300 40 L 300 60" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
        <rect x="250" y="60" width="100" height="30" rx="4" fill="#2c2c2c" stroke="#555" />
        <text x="300" y="80" fill="#ddd" text-anchor="middle" font-size="12">LayerNorm 1</text>
        
        <path d="M 300 90 L 300 110" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
        <rect x="230" y="110" width="140" height="40" rx="4" fill="#2c2c2c" stroke="#e67700" stroke-width="2" />
        <text x="300" y="135" fill="#fff" text-anchor="middle" font-size="12" font-weight="bold">Multi-Head Attention</text>
        
        <!-- Add 1 -->
        <path d="M 300 150 L 300 170 L 160 170" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
        <circle cx="150" cy="170" r="10" fill="#2c2c2c" stroke="#4dabf7" stroke-width="2" />
        <text x="150" y="175" fill="#4dabf7" text-anchor="middle" font-size="16" font-weight="bold">+</text>
        
        <!-- FFN Path -->
        <path d="M 150 200 L 300 200 L 300 220" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
        <rect x="250" y="220" width="100" height="30" rx="4" fill="#2c2c2c" stroke="#555" />
        <text x="300" y="240" fill="#ddd" text-anchor="middle" font-size="12">LayerNorm 2</text>
        
        <path d="M 300 250 L 300 270" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
        <rect x="230" y="270" width="140" height="40" rx="4" fill="#2c2c2c" stroke="#5c940d" stroke-width="2" />
        <text x="300" y="287" fill="#fff" text-anchor="middle" font-size="12" font-weight="bold">Feed Forward Net</text>
        <text x="300" y="302" fill="#aaa" text-anchor="middle" font-size="10">(64 → 256 → 64)</text>
        
        <!-- Add 2 -->
        <path d="M 300 310 L 300 330 L 160 330" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
        <circle cx="150" cy="330" r="10" fill="#2c2c2c" stroke="#4dabf7" stroke-width="2" />
        <text x="150" y="335" fill="#4dabf7" text-anchor="middle" font-size="16" font-weight="bold">+</text>
      </g>
      
      <!-- Block 1 -->
      <g transform="translate(0, 510)">
        <rect x="100" y="0" width="300" height="60" rx="8" fill="#222" stroke="#666" />
        <text x="250" y="35" fill="#aaa" text-anchor="middle" font-size="16" font-weight="bold">Transformer Block 1</text>
        <circle cx="150" cy="30" r="10" fill="#2c2c2c" stroke="#4dabf7" stroke-width="2" />
        <text x="150" y="35" fill="#4dabf7" text-anchor="middle" font-size="16" font-weight="bold">+</text>
      </g>
      
      <!-- Block 2 -->
      <g transform="translate(0, 590)">
        <rect x="100" y="0" width="300" height="60" rx="8" fill="#222" stroke="#666" />
        <text x="250" y="35" fill="#aaa" text-anchor="middle" font-size="16" font-weight="bold">Transformer Block 2</text>
        <circle cx="150" cy="30" r="10" fill="#2c2c2c" stroke="#4dabf7" stroke-width="2" />
        <text x="150" y="35" fill="#4dabf7" text-anchor="middle" font-size="16" font-weight="bold">+</text>
      </g>
      
      <!-- Output Stage -->
      <g transform="translate(0, 700)">
        <rect x="100" y="0" width="100" height="30" rx="4" fill="#2c2c2c" stroke="#555" />
        <text x="150" y="20" fill="#ddd" text-anchor="middle" font-size="12">Final LayerNorm</text>
        
        <path d="M 150 30 L 150 50" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
        <rect x="75" y="50" width="150" height="40" rx="4" fill="#2c2c2c" stroke="#e03131" stroke-width="2" />
        <text x="150" y="70" fill="#fff" text-anchor="middle" font-size="12" font-weight="bold">Language Head</text>
        <text x="150" y="85" fill="#aaa" text-anchor="middle" font-size="10">(Linear Projection to Vocab)</text>
        
        <path d="M 150 90 L 150 110" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
        <rect x="100" y="110" width="100" height="30" rx="4" fill="#2c2c2c" stroke="#555" />
        <text x="150" y="130" fill="#ddd" text-anchor="middle" font-size="12">Softmax</text>
        
        <path d="M 150 140 L 150 160" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
        <text x="150" y="175" fill="#fff" text-anchor="middle" font-size="14" font-weight="bold">Next Token Probabilities</text>
      </g>
    `;

    return svg(w, h, s, "");
  }

  /* helpers */
  function section(n, title) {
    return `<div class="lz-section"><span class="lz-secnum">${n}</span>${escapeHtml(title)}</div>`;
  }
  function note(html) { return `<div class="lz-note">${html}</div>`; }
  function svg(w, h, inner, cap) {
    return `<div class="lz-diagram">${cap ? `<div class="lz-diagram-cap">${escapeHtml(cap)}</div>` : ""}<div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333;"><svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid meet">${inner}</svg></div></div>`;
  }

  draw();
}
