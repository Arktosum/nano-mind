/* paged_attention.js — PagedAttention */
export function renderPagedAttentionLesson(host, ctx) {
  function render() {
    const w = 700;
    const h = 400;
    
    return `
      <div class="lz-diagram">
        <div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333; position:relative;">
          <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" id="paSvg">
            
            <text x="350" y="30" fill="#fff" font-weight="bold" text-anchor="middle" font-size="16">PagedAttention (vLLM)</text>
            
            <!-- Logical KV Cache (Contiguous) -->
            <text x="150" y="80" fill="#aaa" text-anchor="middle" font-size="14" font-weight="bold">Logical KV Cache (Contiguous)</text>
            
            <rect x="50" y="100" width="200" height="40" rx="4" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
            <text x="150" y="125" fill="#fff" text-anchor="middle">Tokens 0-15</text>
            <rect x="50" y="150" width="200" height="40" rx="4" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
            <text x="150" y="175" fill="#fff" text-anchor="middle">Tokens 16-31</text>
            <rect x="50" y="200" width="200" height="40" rx="4" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
            <text x="150" y="225" fill="#fff" text-anchor="middle">Tokens 32-47</text>
            
            <!-- Block Table -->
            <text x="350" y="80" fill="#888" text-anchor="middle" font-size="12">Block Table</text>
            <path d="M 255 120 L 320 150" class="svg-arrow" stroke-dasharray="4 4" />
            <path d="M 255 170 L 320 200" class="svg-arrow" stroke-dasharray="4 4" />
            <path d="M 255 220 L 320 300" class="svg-arrow" stroke-dasharray="4 4" />
            
            <rect x="310" y="140" width="80" height="20" rx="2" fill="#222" stroke="#555" />
            <text x="350" y="154" fill="#aaa" font-size="10" text-anchor="middle">Physical: 3</text>
            <rect x="310" y="190" width="80" height="20" rx="2" fill="#222" stroke="#555" />
            <text x="350" y="204" fill="#aaa" font-size="10" text-anchor="middle">Physical: 7</text>
            <rect x="310" y="290" width="80" height="20" rx="2" fill="#222" stroke="#555" />
            <text x="350" y="304" fill="#aaa" font-size="10" text-anchor="middle">Physical: 1</text>

            <!-- Physical VRAM (Fragmented) -->
            <text x="550" y="80" fill="#aaa" text-anchor="middle" font-size="14" font-weight="bold">Physical GPU VRAM (Fragmented)</text>
            
            <g id="vramGrid"></g>

            <path d="M 395 150 L 450 150" class="svg-arrow" />
            <path d="M 395 200 L 450 250" class="svg-arrow" />
            <path d="M 395 300 L 450 100" class="svg-arrow" />

          </svg>
        </div>
        
        <div class="lz-notes" >
          <h3>PagedAttention</h3>
          <p class="note-meta"><i>Developed: Sep 2023 (Kwon et al. / vLLM)</i></p>
          
          <p class="lz-notes-prob">1. The Problem</p>
          <p>In a production server, you don't know how many tokens a user will generate. To be safe, the server pre-allocates a massive contiguous chunk of VRAM for the maximum possible sequence length. If the user stops after 10 tokens, the rest of that massive chunk is wasted. This fragmentation limits servers to very few concurrent users.</p>
          
          <p class="lz-notes-moti">2. The Motivation</p>
          <p>Operating systems solved this decades ago with Virtual Memory—breaking memory into small "pages" that can be scattered anywhere physically, but appear contiguous logically.</p>
          
          <p class="lz-notes-impl">3. The Implementation</p>
          <p style="margin-top: 0;">PagedAttention breaks the KV Cache into small blocks (e.g., 16 tokens). A "Block Table" maps the logical sequence of tokens to scattered physical blocks in VRAM. When a user generates a 17th token, the server allocates just one new physical block wherever it fits. Fragmentation drops to near 0%, allowing servers (like vLLM) to handle 4x more users on the same hardware.</p>
        </div>
      </div>
    `;
  }
  
  host.innerHTML = render();
  
  // Render VRAM Grid (2 columns, 5 rows)
  let vram = '';
  const blocks = [
    { id: 0, active: false }, { id: 1, active: true, map: 2 },
    { id: 2, active: false }, { id: 3, active: true, map: 0 },
    { id: 4, active: false }, { id: 5, active: false },
    { id: 6, active: false }, { id: 7, active: true, map: 1 },
    { id: 8, active: false }, { id: 9, active: false },
  ];
  
  blocks.forEach((b, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 450 + (col * 100);
    const y = 90 + (row * 50);
    
    if (b.active) {
      vram += `<rect x="${x}" y="${y}" width="90" height="40" rx="4" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />`;
      vram += `<text x="${x+45}" y="${y+20}" fill="#fff" font-size="12" text-anchor="middle">Tokens ${b.map * 16}-${(b.map * 16) + 15}</text>`;
      vram += `<text x="${x+45}" y="${y+33}" fill="#4dabf7" font-size="9" text-anchor="middle">Physical Block ${b.id}</text>`;
    } else {
      vram += `<rect x="${x}" y="${y}" width="90" height="40" rx="4" fill="#222" stroke="#444" stroke-width="2" stroke-dasharray="2 2" />`;
      vram += `<text x="${x+45}" y="${y+25}" fill="#555" font-size="12" text-anchor="middle">Empty / Free</text>`;
    }
  });
  
  host.querySelector('#vramGrid').innerHTML = vram;
}
