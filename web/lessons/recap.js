/* recap.js — a high-level map of the whole architecture, on palette.
   Uses the shared kit helpers (section, note, svg). */

function renderRecapLesson(host, ctx) {
  const AC = "#d6401f", INK = "#e8e4da", OFF = "#7d8595", IDLE = "#17170f", LINE = "#5a564b", STROKE = "#3a3a30";

  const inner = `
    <defs>
      <marker id="rcA" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="${LINE}"/>
      </marker>
      <marker id="rcAc" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="${AC}"/>
      </marker>
    </defs>
    <line x1="150" y1="50" x2="150" y2="820" stroke="${AC}" stroke-width="4" marker-end="url(#rcAc)"/>
    <text x="66" y="450" transform="rotate(-90 66,450)" fill="${AC}" font-weight="bold" font-size="13" text-anchor="middle" letter-spacing="2">RESIDUAL STREAM · 64-dim</text>

    <rect x="50" y="50" width="200" height="40" fill="${IDLE}" stroke="${STROKE}"/>
    <text x="150" y="75" fill="${INK}" text-anchor="middle" font-size="14">token + position embeddings</text>

    <g transform="translate(0,130)">
      <rect x="100" y="0" width="300" height="360" fill="none" stroke="${STROKE}" stroke-dasharray="4 4"/>
      <text x="120" y="24" fill="${OFF}" font-size="15" font-weight="bold">transformer block 0</text>
      <path d="M 150 40 L 300 40 L 300 60" fill="none" stroke="${LINE}" stroke-width="2" marker-end="url(#rcA)"/>
      <rect x="250" y="60" width="100" height="30" fill="${IDLE}" stroke="${STROKE}"/>
      <text x="300" y="79" fill="${INK}" text-anchor="middle" font-size="12">LayerNorm 1</text>
      <path d="M 300 90 L 300 110" fill="none" stroke="${LINE}" stroke-width="2" marker-end="url(#rcA)"/>
      <rect x="230" y="110" width="140" height="40" fill="${IDLE}" stroke="${AC}" stroke-width="2"/>
      <text x="300" y="134" fill="${INK}" text-anchor="middle" font-size="12" font-weight="bold">multi-head attention</text>
      <path d="M 300 150 L 300 170 L 160 170" fill="none" stroke="${LINE}" stroke-width="2" marker-end="url(#rcA)"/>
      <circle cx="150" cy="170" r="10" fill="${IDLE}" stroke="${AC}" stroke-width="2"/>
      <text x="150" y="175" fill="${AC}" text-anchor="middle" font-size="16" font-weight="bold">+</text>
      <path d="M 150 200 L 300 200 L 300 220" fill="none" stroke="${LINE}" stroke-width="2" marker-end="url(#rcA)"/>
      <rect x="250" y="220" width="100" height="30" fill="${IDLE}" stroke="${STROKE}"/>
      <text x="300" y="239" fill="${INK}" text-anchor="middle" font-size="12">LayerNorm 2</text>
      <path d="M 300 250 L 300 270" fill="none" stroke="${LINE}" stroke-width="2" marker-end="url(#rcA)"/>
      <rect x="230" y="270" width="140" height="40" fill="${IDLE}" stroke="${AC}" stroke-width="2"/>
      <text x="300" y="288" fill="${INK}" text-anchor="middle" font-size="12" font-weight="bold">feed-forward net</text>
      <text x="300" y="303" fill="${OFF}" text-anchor="middle" font-size="10">64 → 256 → 64</text>
      <path d="M 300 310 L 300 330 L 160 330" fill="none" stroke="${LINE}" stroke-width="2" marker-end="url(#rcA)"/>
      <circle cx="150" cy="330" r="10" fill="${IDLE}" stroke="${AC}" stroke-width="2"/>
      <text x="150" y="335" fill="${AC}" text-anchor="middle" font-size="16" font-weight="bold">+</text>
    </g>

    <g transform="translate(0,510)">
      <rect x="100" y="0" width="300" height="55" fill="${IDLE}" stroke="${STROKE}"/>
      <text x="250" y="32" fill="${OFF}" text-anchor="middle" font-size="14" font-weight="bold">transformer block 1 · identical</text>
      <circle cx="150" cy="27" r="9" fill="${IDLE}" stroke="${AC}" stroke-width="2"/>
      <text x="150" y="32" fill="${AC}" text-anchor="middle" font-size="14" font-weight="bold">+</text>
    </g>
    <g transform="translate(0,585)">
      <rect x="100" y="0" width="300" height="55" fill="${IDLE}" stroke="${STROKE}"/>
      <text x="250" y="32" fill="${OFF}" text-anchor="middle" font-size="14" font-weight="bold">transformer block 2 · identical</text>
      <circle cx="150" cy="27" r="9" fill="${IDLE}" stroke="${AC}" stroke-width="2"/>
      <text x="150" y="32" fill="${AC}" text-anchor="middle" font-size="14" font-weight="bold">+</text>
    </g>

    <g transform="translate(0,690)">
      <rect x="100" y="0" width="100" height="30" fill="${IDLE}" stroke="${STROKE}"/>
      <text x="150" y="19" fill="${INK}" text-anchor="middle" font-size="12">final LayerNorm</text>
      <path d="M 150 30 L 150 50" fill="none" stroke="${LINE}" stroke-width="2" marker-end="url(#rcA)"/>
      <rect x="75" y="50" width="150" height="40" fill="${IDLE}" stroke="${AC}" stroke-width="2"/>
      <text x="150" y="70" fill="${INK}" text-anchor="middle" font-size="12" font-weight="bold">language head</text>
      <text x="150" y="84" fill="${OFF}" text-anchor="middle" font-size="10">linear → 75 vocab logits</text>
      <path d="M 150 90 L 150 110" fill="none" stroke="${LINE}" stroke-width="2" marker-end="url(#rcA)"/>
      <rect x="100" y="110" width="100" height="30" fill="${IDLE}" stroke="${STROKE}"/>
      <text x="150" y="129" fill="${INK}" text-anchor="middle" font-size="12">softmax</text>
      <path d="M 150 140 L 150 160" fill="none" stroke="${LINE}" stroke-width="2" marker-end="url(#rcAc)"/>
      <text x="150" y="176" fill="${AC}" text-anchor="middle" font-size="13" font-weight="bold">next-token probabilities</text>
    </g>`;

  host.innerHTML =
    section("1", "the whole network at a glance") +
    note("We walked through one full Transformer block — LayerNorm, multi-head attention, a residual add, LayerNorm, feed-forward, another residual add. The model stacks <b>3 identical blocks</b>; the residual stream runs top to bottom, read from and added back to at each step, until the output stage turns it into a next-character distribution.") +
    svg(500, 830, inner, "");
}
