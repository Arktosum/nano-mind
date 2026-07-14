/* resid2.js — Second Residual Add step */
function renderResid2Lesson(host, ctx) {
  function svg() {
    const w = 500;
    const h = 250;
    
    let s = `
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
        </marker>
      </defs>
      
      <!-- Residual Stream -->
      <rect x="50" y="50" width="100" height="40" rx="4" fill="#222" stroke="#4dabf7" stroke-width="2" />
      <text x="100" y="75" fill="#fff" text-anchor="middle" font-size="14">Stream (h')</text>
      
      <path d="M 150 70 L 230 70" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
      
      <!-- FFN Output -->
      <rect x="50" y="150" width="100" height="40" rx="4" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
      <text x="100" y="175" fill="#fff" text-anchor="middle" font-size="14">FFN Output</text>
      
      <path d="M 150 170 L 250 170 L 250 100" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
      
      <!-- Add -->
      <circle cx="250" cy="70" r="20" fill="#2c2c2c" stroke="#e67700" stroke-width="2" />
      <text x="250" y="78" fill="#e67700" text-anchor="middle" font-size="24" font-weight="bold">+</text>
      
      <path d="M 270 70 L 350 70" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
      
      <!-- Output Stream -->
      <rect x="350" y="50" width="120" height="40" rx="4" fill="#222" stroke="#4dabf7" stroke-width="2" />
      <text x="410" y="75" fill="#fff" text-anchor="middle" font-size="14">New Stream (h'')</text>
    `;
    
    return `<div class="lz-diagram"><div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333;"><svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">${s}</svg></div></div>`;
  }
  
  host.innerHTML = svg();
}
