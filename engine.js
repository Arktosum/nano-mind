/**
 * engine.js - The Decoupled Inference Engine
 * Translates characters to tokens and executes the Transformer math.
 */

// Exact vocabulary extracted from the Python training script (75 characters)
const VOCAB = "\n !()*,-.03:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyzù—‘’“”";
const BLOCK_SIZE = 64;
const N_LAYER = 3;
const N_HEAD = 4;

class NanoTransformer {
    constructor(weightsJson) {
        this.weights = weightsJson;
        this.lastAttentionState = null; // We will store the final layer's attention scores here for the UI!
    }

    // Tokenizer mappings
    encode(text) {
        let tokens = [];
        for (let char of text) {
            let idx = VOCAB.indexOf(char);
            if (idx !== -1) tokens.push(idx);
        }
        return tokens;
    }

    decode(tokenId) {
        return VOCAB[tokenId] || "";
    }

    // The Core Forward Pass
    forward(text) {
        let tokens = this.encode(text);
        
        // Truncate context if it exceeds our BLOCK_SIZE
        if (tokens.length > BLOCK_SIZE) {
            tokens = tokens.slice(tokens.length - BLOCK_SIZE);
        }
        
        const seqLen = tokens.length;
        if (seqLen === 0) return null;

        // 1. Embeddings (Lookup tables)
        let x = [];
        for (let i = 0; i < seqLen; i++) {
            let tok_emb = this.weights['token_embedding_table.weight'][tokens[i]];
            let pos_emb = this.weights['position_embedding_table.weight'][i];
            
            // Add token and position embeddings
            let emb = [];
            for (let j = 0; j < tok_emb.length; j++) {
                emb.push(tok_emb[j] + pos_emb[j]);
            }
            x.push(emb);
        }

        // 2. Transformer Blocks
        for (let l = 0; l < N_LAYER; l++) {
            // LayerNorm 1
            let pre_sa = layerNorm(x, this.weights[`blocks.${l}.ln1.weight`], this.weights[`blocks.${l}.ln1.bias`]);
            
            let head_outputs = [];
            let layerAttentionStates = []; // To capture the heatmap data
            
            // Multi-Head Attention
            for (let h = 0; h < N_HEAD; h++) {
                // Linear projections for Q, K, V (Note: bias=null because we set bias=False in PyTorch)
                let q = linear(pre_sa, this.weights[`blocks.${l}.sa.heads.${h}.query.weight`], null);
                let k = linear(pre_sa, this.weights[`blocks.${l}.sa.heads.${h}.key.weight`], null);
                let v = linear(pre_sa, this.weights[`blocks.${l}.sa.heads.${h}.value.weight`], null);
                
                // Q dot K^T
                let scores = qk_dot(q, k);
                
                // Scale by 1 / sqrt(head_size)
                let scale = 1.0 / Math.sqrt(q[0].length);
                for (let r = 0; r < scores.length; r++) {
                    for (let c = 0; c < scores[r].length; c++) scores[r][c] *= scale;
                }
                
                // Masking and Softmax
                let attention = maskAndSoftmax(scores);
                layerAttentionStates.push(attention); // Save for visualizer
                
                // Weighted aggregation of Values
                let out = score_v_dot(attention, v);
                head_outputs.push(out);
            }
            
            // ---> VISUALIZER HOOK: Capture the attention state of the final block! <---
            if (l === N_LAYER - 1) {
                this.lastAttentionState = layerAttentionStates;
            }
            
            // Concat heads and project back
            let sa_concat = concatHeads(head_outputs);
            let sa_out = linear(sa_concat, this.weights[`blocks.${l}.sa.proj.weight`], this.weights[`blocks.${l}.sa.proj.bias`]);
            
            // Residual Connection
            x = add(x, sa_out);
            
            // Feed-Forward Network
            let pre_ffwd = layerNorm(x, this.weights[`blocks.${l}.ln2.weight`], this.weights[`blocks.${l}.ln2.bias`]);
            let ffwd_hidden = relu(linear(pre_ffwd, this.weights[`blocks.${l}.ffwd.net.0.weight`], this.weights[`blocks.${l}.ffwd.net.0.bias`]));
            let ffwd_out = linear(ffwd_hidden, this.weights[`blocks.${l}.ffwd.net.2.weight`], this.weights[`blocks.${l}.ffwd.net.2.bias`]);
            
            // Residual Connection
            x = add(x, ffwd_out);
        }
        
        // 3. Final Output Layer
        x = layerNorm(x, this.weights['ln_f.weight'], this.weights['ln_f.bias']);
        let logits = linear(x, this.weights['lm_head.weight'], this.weights['lm_head.bias']);
        
        // 4. Get the probabilities for the *very last* character in the sequence
        let last_logits = logits[seqLen - 1];
        let probs = softmax1D(last_logits);
        
        // Find the index with the highest probability
        let best_idx = probs.indexOf(Math.max(...probs));
        
        return {
            predictedChar: this.decode(best_idx),
            probabilities: probs,
            tokens: tokens // So the UI knows what characters to label the axes with
        };
    }
}