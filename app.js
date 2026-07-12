
const input = document.getElementById('promptInput');
const statusBox = document.getElementById('statusBox');
const pipelineBox = document.getElementById('pipelineBox');
const tokensContainer = document.getElementById('tokensContainer');
const embeddingContainer = document.getElementById('embeddingContainer');

// Modal Elements
const modal = document.getElementById('vocabModal');
const btnModal = document.getElementById('viewVocabBtn');
const spanClose = document.getElementById('closeModal');
const vocabGrid = document.getElementById('vocabGrid');

let engine = null;
let currentTokens = [];

// Boot up
fetch('weights.json')
    .then(response => {
        if (!response.ok) throw new Error("Could not find weights.json");
        return response.json();
    })
    .then(weights => {
        engine = new NanoTransformer(weights);
        statusBox.innerHTML = "Engine loaded. 75x64 embedding matrix ready.";
        input.disabled = false;
        populateVocabModal();
        input.focus();
    })
    .catch(err => {
        statusBox.innerHTML = `<span style="color: #ff5252">Error: ${err.message}. Check local server.</span>`;
    });

btnModal.onclick = () => modal.style.display = "block";
spanClose.onclick = () => modal.style.display = "none";
window.onclick = (e) => { if (e.target === modal) modal.style.display = "none"; }

function populateVocabModal() {
    const vocabStr = "\n !()*,-.03:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyzù—‘’“”";
    let html = '';
    for (let i = 0; i < vocabStr.length; i++) {
        let char = vocabStr[i] === '\n' ? '\\n' : vocabStr[i] === ' ' ? 'SPC' : vocabStr[i];
        html += `<div class="vocab-item"><span class="vocab-char">${char}</span><span class="vocab-id">ID: ${i}</span></div>`;
    }
    vocabGrid.innerHTML = html;
}

// Input Loop (Stripped down to just tokenization)
input.addEventListener('input', (e) => {
    if (!engine) return;
    const text = e.target.value;
    
    if (!text) {
        pipelineBox.style.display = 'none';
        return;
    }

    // We still run the forward pass to get the token array, 
    // even though we aren't showing the final probabilities anymore.
    const output = engine.forward(text);
    if (!output) return;
    currentTokens = output.tokens;

    pipelineBox.style.display = 'block';

    tokensContainer.innerHTML = currentTokens.map((t, idx) => {
        const char = engine.decode(t) === ' ' ? 'SPC' : engine.decode(t) === '\n' ? '\\n' : engine.decode(t);
        return `<span class="token-badge" onclick="inspectVector(${t}, this)">${char} (ID: ${t})</span>`;
    }).join('');

    embeddingContainer.innerHTML = '<span class="muted">Select a token above to visualize the matrix extraction.</span>';
});

// The Giant Matrix Visualizer
window.inspectVector = function(tokenId, element) {
    document.querySelectorAll('.token-badge').forEach(el => el.classList.remove('active'));
    element.classList.add('active');

    const tok_matrix = engine.weights['token_embedding_table.weight'];
    const rows = tok_matrix.length;    // 75
    const cols = tok_matrix[0].length; // 64
    
    // Scale up the pixels for visibility
    const scale = 8;
    const matrixHeight = rows * scale; // 600px
    const matrixWidth = cols * scale;  // 512px
    const vectorWidth = 24;            // Fat 1D column

    embeddingContainer.innerHTML = `
        <div class="formula-block">v<sub>tok</sub> = x<sub>one-hot</sub> &times; W<sub>token</sub></div>
        
        <div class="matrix-container">
            <div class="vector-box">
                <span class="vector-label">One-Hot<br>(1&times;${rows})</span>
                <canvas id="oneHotCanvas" width="${vectorWidth}" height="${matrixHeight}" class="matrix-canvas"></canvas>
            </div>
            <span class="math-symbol">&times;</span>
            <div class="vector-box">
                <span class="vector-label">W_token<br>(${rows}&times;${cols})</span>
                <canvas id="matrixCanvas" width="${matrixWidth}" height="${matrixHeight}" class="matrix-canvas"></canvas>
            </div>
            <span class="math-symbol">=</span>
            <div class="vector-box">
                <span class="vector-label">v_tok<br>(1&times;${cols})</span>
                <canvas id="outputCanvas" width="${matrixWidth}" height="${vectorWidth}" class="matrix-canvas"></canvas>
            </div>
        </div>
    `;

    // 1. Draw One-Hot Vector
    const ctxOneHot = document.getElementById('oneHotCanvas').getContext('2d');
    ctxOneHot.fillStyle = '#111';
    ctxOneHot.fillRect(0, 0, vectorWidth, matrixHeight);
    ctxOneHot.fillStyle = '#4CAF50';
    ctxOneHot.fillRect(0, tokenId * scale, vectorWidth, scale);

    // 2. Draw Full Embedding Matrix
    const ctxMatrix = document.getElementById('matrixCanvas').getContext('2d');
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = tok_matrix[r][c];
            const intensity = Math.floor((val + 2) / 4 * 255); 
            
            if (r === tokenId) {
                // The extracted row lights up green
                ctxMatrix.fillStyle = `rgb(0, ${Math.max(120, intensity)}, 0)`;
            } else {
                ctxMatrix.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
            }
            ctxMatrix.fillRect(c * scale, r * scale, scale, scale);
        }
    }

    // 3. Draw Output Vector (The Extracted Row)
    const ctxOut = document.getElementById('outputCanvas').getContext('2d');
    const activeRow = tok_matrix[tokenId];
    for (let c = 0; c < cols; c++) {
        const val = activeRow[c];
        const intensity = Math.floor((val + 2) / 4 * 255);
        ctxOut.fillStyle = `rgb(0, ${Math.max(120, intensity)}, 0)`;
        ctxOut.fillRect(c * scale, 0, scale, vectorWidth);
    }
};