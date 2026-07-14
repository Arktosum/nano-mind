/* ffwd.js — Feed Forward Network step */
function renderFfwdLesson(host, ctx) {
  function svg() {
    const w = 700;
    const h = 250;
    
    let s = `
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
        </marker>
      </defs>
      
      <!-- Input -->
      <rect x="50" y="100" width="60" height="20" rx="4" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
      <text x="80" y="114" fill="#fff" text-anchor="middle" font-size="12">1 x 64</text>
      <text x="80" y="140" fill="#aaa" text-anchor="middle" font-size="12">Input</text>
      
      <path d="M 110 110 L 150 110" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
      
      <!-- Linear Up -->
      <rect x="150" y="80" width="80" height="60" rx="4" fill="#2c2c2c" stroke="#666" />
      <text x="190" y="105" fill="#fff" text-anchor="middle" font-size="12">W_up</text>
      <text x="190" y="125" fill="#888" text-anchor="middle" font-size="12">64 x 256</text>
      
      <path d="M 230 110 L 270 110" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
      
      <!-- Hidden -->
      <rect x="270" y="70" width="80" height="80" rx="4" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
      <text x="310" y="115" fill="#fff" text-anchor="middle" font-size="12">1 x 256</text>
      <text x="310" y="170" fill="#aaa" text-anchor="middle" font-size="12">Hidden</text>
      
      <path d="M 350 110 L 390 110" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
      
      <!-- Activation -->
      <rect x="390" y="90" width="60" height="40" rx="4" fill="#2b1a10" stroke="#e67700" stroke-width="2" />
      <text x="420" y="115" fill="#fff" text-anchor="middle" font-size="12">ReLU</text>
      
      <path d="M 450 110 L 490 110" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
      
      <!-- Linear Down -->
      <rect x="490" y="80" width="80" height="60" rx="4" fill="#2c2c2c" stroke="#666" />
      <text x="530" y="105" fill="#fff" text-anchor="middle" font-size="12">W_down</text>
      <text x="530" y="125" fill="#888" text-anchor="middle" font-size="12">256 x 64</text>
      
      <path d="M 570 110 L 610 110" fill="none" stroke="#888" stroke-width="2" marker-end="url(#arrow)" />
      
      <!-- Output -->
      <rect x="610" y="100" width="60" height="20" rx="4" fill="#1a365d" stroke="#4dabf7" stroke-width="2" />
      <text x="640" y="114" fill="#fff" text-anchor="middle" font-size="12">1 x 64</text>
      <text x="640" y="140" fill="#aaa" text-anchor="middle" font-size="12">Output</text>
    `;
    
    return `<div class="lz-diagram"><div class="lz-svgwrap" style="background:#111; padding:20px; border-radius:8px; border:1px solid #333;"><svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">${s}</svg></div></div>`;
  }
  
  host.innerHTML = svg();
}
