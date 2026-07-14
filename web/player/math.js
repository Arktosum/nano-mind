/**
 * math.js - Pure JavaScript Linear Algebra for Nano-Mind
 * We use simple 1D and 2D arrays to keep the math completely transparent.
 */

// 1. Linear Transformation (x @ W^T + b)
// Matches PyTorch's nn.Linear
function linear(x, w, b) {
    // x shape: [seq_len, in_features]
    // w shape: [out_features, in_features] (PyTorch exports weights transposed!)
    let out = [];
    for (let i = 0; i < x.length; i++) {
        let row = [];
        for (let j = 0; j < w.length; j++) {
            let sum = b ? b[j] : 0;
            for (let k = 0; k < x[i].length; k++) {
                sum += x[i][k] * w[j][k];
            }
            row.push(sum);
        }
        out.push(row);
    }
    return out;
}

// 2. Layer Normalization
// Matches PyTorch's nn.LayerNorm
function layerNorm(x, gamma, beta, eps = 1e-5) {
    let out = [];
    for (let i = 0; i < x.length; i++) {
        // Calculate mean
        let mean = x[i].reduce((a, b) => a + b, 0) / x[i].length;
        // Calculate variance
        let var_sum = x[i].reduce((a, b) => a + Math.pow(b - mean, 2), 0) / x[i].length;
        
        let row = [];
        for (let j = 0; j < x[i].length; j++) {
            let norm = (x[i][j] - mean) / Math.sqrt(var_sum + eps);
            row.push(norm * gamma[j] + beta[j]);
        }
        out.push(row);
    }
    return out;
}

// 3. Attention Math: Query dot Key^T
function qk_dot(q, k) {
    // Both shapes: [seq_len, head_size]
    let out = [];
    for (let i = 0; i < q.length; i++) {
        let row = [];
        for (let j = 0; j < k.length; j++) {
            let sum = 0;
            for (let d = 0; d < q[i].length; d++) {
                sum += q[i][d] * k[j][d];
            }
            row.push(sum);
        }
        out.push(row);
    }
    return out; // Shape: [seq_len, seq_len]
}

// 4. Causal Mask & Softmax (The "Attention" scores)
function maskAndSoftmax(scores) {
    let out = [];
    for (let i = 0; i < scores.length; i++) {
        let row = [];
        let max = -Infinity;
        
        // Causal Masking: Tokens can only look at themselves and the past (j <= i)
        for (let j = 0; j < scores[i].length; j++) {
            if (j > i) {
                row.push(-Infinity);
            } else {
                row.push(scores[i][j]);
                if (scores[i][j] > max) max = scores[i][j];
            }
        }
        
        // Softmax
        let expSum = 0;
        let exps = [];
        for (let j = 0; j < row.length; j++) {
            if (j > i) {
                exps.push(0);
            } else {
                let e = Math.exp(row[j] - max); // Subtract max for numerical stability
                exps.push(e);
                expSum += e;
            }
        }
        out.push(exps.map(e => e / expSum));
    }
    return out; // Output is percentages (0.0 to 1.0)
}

// 5. Attention Math: Scores dot Value
function score_v_dot(scores, v) {
    // scores shape: [seq_len, seq_len]
    // v shape: [seq_len, head_size]
    let out = [];
    for (let i = 0; i < scores.length; i++) {
        let row = [];
        for (let d = 0; d < v[0].length; d++) {
            let sum = 0;
            for (let j = 0; j <= i; j++) {
                sum += scores[i][j] * v[j][d];
            }
            row.push(sum);
        }
        out.push(row);
    }
    return out;
}

// 6. Concatenate multiple Attention Heads back together
function concatHeads(heads_outputs) {
    let seq_len = heads_outputs[0].length;
    let out = [];
    for (let i = 0; i < seq_len; i++) {
        let row = [];
        for (let h = 0; h < heads_outputs.length; h++) {
            row.push(...heads_outputs[h][i]);
        }
        out.push(row);
    }
    return out;
}

// 7. Residual Connection (Vector Addition)
function add(a, b) {
    let out = [];
    for (let i = 0; i < a.length; i++) {
        let row = [];
        for (let j = 0; j < a[i].length; j++) {
            row.push(a[i][j] + b[i][j]);
        }
        out.push(row);
    }
    return out;
}

// 8. ReLU Non-Linearity
function relu(x) {
    let out = [];
    for (let i = 0; i < x.length; i++) {
        out.push(x[i].map(v => Math.max(0, v)));
    }
    return out;
}

// 9. Simple 1D Softmax (For final token prediction)
function softmax1D(x) {
    let max = Math.max(...x);
    let exps = x.map(v => Math.exp(v - max));
    let sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
}