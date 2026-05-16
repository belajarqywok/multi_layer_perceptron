/**
 * model.js — Pure JavaScript MLP Inference Engine
 *
 * Loads the trained weights from model_weights.js and runs forward propagation
 * to predict enemy aggression score (0 = passive, 1 = very aggressive).
 *
 * Architecture mirrors the C++ MLP:
 *   Input (5) → ReLU Hidden(s) → Linear Output (1)
 *
 * Standardisation: applies the same z-score as the C++ CSVReader used during training.
 *   z = (x - mean) / std
 */

class MLPModel {
    constructor() {
        // Weights are loaded from model_weights.js (included before this file)
        this.W = [];
        this.B = [];
        this._loadWeights();
    }

    _loadWeights() {
        // MODEL_WEIGHTS keys: W0, B0, W1, B1, ...
        // We collect them in order
        let i = 0;
        while (MODEL_WEIGHTS[`W${i}`] !== undefined) {
            this.W.push(MODEL_WEIGHTS[`W${i}`]);
            // Bias may be stored as 1D [n] or 2D [1,n] depending on exporter
            // Normalise to flat 1D array for consistent addition
            const rawB = MODEL_WEIGHTS[`B${i}`];
            this.B.push(Array.isArray(rawB[0]) ? rawB[0] : rawB);
            i++;
        }
        console.log(`[MLPModel] Loaded ${i} layer(s). Architecture: input(${this.W[0].length}) → hidden(${this.W[0][0].length}) → output(${this.W[this.W.length-1][0].length})`);
    }

    /**
     * Z-score standardise raw input vector using training-time statistics.
     * @param {number[]} features  - Raw feature values in FEATURE_NAMES order
     * @returns {number[]} Standardised feature vector
     */
    _standardize(features) {
        return FEATURE_NAMES.map((name, i) =>
            (features[i] - FEATURE_MEANS[name]) / FEATURE_STDS[name]
        );
    }

    /** Element-wise ReLU activation: max(0, x) */
    _relu(vec) { return vec.map(v => Math.max(0, v)); }

    /**
     * Matrix multiply: A [m x k] * B [k x n] → C [m x n]
     * Stored as nested arrays (row-major).
     */
    _matMul(A, B) {
        const m = A.length, k = A[0].length, n = B[0].length;
        const C = Array.from({ length: m }, () => new Float64Array(n));
        for (let i = 0; i < m; i++)
            for (let l = 0; l < k; l++) {
                const a = A[i][l];
                for (let j = 0; j < n; j++)
                    C[i][j] += a * B[l][j];
            }
        return Array.from(C, row => Array.from(row));
    }

    /**
     * Run one forward pass for a single sample.
     * @param {Object} gameState
     *   { rel_x, rel_y, bullet_near, bullet_rel_x, difficulty }
     * @returns {number} Aggression score clamped to [0, 1]
     */
    predict(gameState) {
        const raw = [
            gameState.rel_x,
            gameState.rel_y,
            gameState.bullet_near,
            gameState.bullet_rel_x,
            gameState.difficulty,
        ];

        let A = [this._standardize(raw)];   // shape [1, input_size]

        for (let i = 0; i < this.W.length; i++) {
            // Z = A · W[i]  (then add flat bias B[i])
            let Z = this._matMul(A, this.W[i]);
            const bias = this.B[i];  // flat 1D array [out_size]
            for (let j = 0; j < Z[0].length; j++)
                Z[0][j] += bias[j];

            // Hidden layers use ReLU; final layer is linear (regressor)
            A = (i < this.W.length - 1) ? [this._relu(Z[0])] : Z;
        }

        // Clamp output to [0, 1] — model was trained to predict aggression in that range
        return Math.max(0, Math.min(1, A[0][0]));
    }
}
