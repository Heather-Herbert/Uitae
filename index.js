const fs = require('fs');
const v8 = require('v8');
const math = require('mathjs');
const readline = require('readline');

class Layer {
    constructor(size, nextSize = 0) {
        this.size = size;
        this.x = math.zeros(size);
        this.e = math.zeros(size);
        this.bias = math.zeros(size);
        
        // Weights predicting the layer below (this layer) from the layer above (nextLayer)
        // W[l] has dimensions [size_l, size_l+1]
        if (nextSize > 0) {
            this.w = math.multiply(math.random([size, nextSize]), 0.01);
        } else {
            this.w = null;
        }
    }
}

class Mind {
    constructor(hierarchy = [256, 512, 256], dt = 0.1, lr = 0.01) {
        this.dt = dt;
        this.lr = lr;
        this.layers = [];
        this.errorMovingAverage = new Array(hierarchy.length).fill(0);
        this.alpha = 0.05; // Smoothing for moving average
        this.expansionThreshold = 0.5;

        for (let i = 0; i < hierarchy.length; i++) {
            const size = hierarchy[i];
            const nextSize = i + 1 < hierarchy.length ? hierarchy[i + 1] : 0;
            this.layers.push(new Layer(size, nextSize));
        }
    }

    step(inputVector, iterations = 20) {
        // Clamp input to Layer 0
        this.layers[0].x = math.matrix(inputVector);

        for (let iter = 0; iter < iterations; iter++) {
            // 1. Calculate Errors (Top-Down Predictions)
            for (let l = 0; l < this.layers.length - 1; l++) {
                const currentLayer = this.layers[l];
                const nextLayer = this.layers[l + 1];
                
                // E[l] = X[l] - (W[l] * X[l+1])
                const prediction = math.multiply(currentLayer.w, nextLayer.x);
                currentLayer.e = math.subtract(currentLayer.x, prediction);
                
                // Update error moving average for expansion logic
                const mag = math.norm(currentLayer.e);
                this.errorMovingAverage[l] = (this.alpha * mag) + (1 - this.alpha) * this.errorMovingAverage[l];
            }

            // 2. Update States (Inference)
            // X[l] += dt * (-E[l] + (W[l-1].T * E[l-1]))
            for (let l = 1; l < this.layers.length; l++) {
                const currentLayer = this.layers[l];
                const prevLayer = this.layers[l - 1];
                
                const feedbackDrive = math.multiply(math.transpose(prevLayer.w), prevLayer.e);
                const deltaX = math.multiply(this.dt, math.subtract(feedbackDrive, currentLayer.e));
                currentLayer.x = math.add(currentLayer.x, deltaX);
            }

            // 3. Hebbian Learning (Weight Updates)
            // W[l] += lr * (outerProduct(E[l], X[l+1]))
            for (let l = 0; l < this.layers.length - 1; l++) {
                const currentLayer = this.layers[l];
                const nextLayer = this.layers[l + 1];
                
                // Vector-vector multiplication to get the matrix
                const dw = math.multiply(this.lr, math.multiply(math.reshape(currentLayer.e, [-1, 1]), math.reshape(nextLayer.x, [1, -1])));
                currentLayer.w = math.add(currentLayer.w, dw);
            }

            // 4. Expansion Check
            for (let l = 0; l < this.layers.length; l++) {
                if (this.errorMovingAverage[l] > this.expansionThreshold) {
                    this.expand(l, 8); // Add 8 nodes
                }
            }
        }
    }

    expand(layerIndex, count) {
        console.log(`
[Growth] Expanding Layer ${layerIndex} by ${count} nodes...`);
        const layer = this.layers[layerIndex];
        const oldSize = layer.size;
        const newSize = oldSize + count;

        // Resize X, E, Bias
        const newX = math.zeros(newSize);
        const newE = math.zeros(newSize);
        const newBias = math.zeros(newSize);
        
        // Copy old values (X[l])
        layer.x.forEach((val, idx) => newX.set(idx, val));
        layer.x = newX;
        layer.e = newE;
        layer.bias = newBias;
        layer.size = newSize;

        // Resize W[l] (Predicts this layer from layer above)
        if (layer.w) {
            const newW = math.zeros([newSize, this.layers[layerIndex + 1].size]);
            // Copy old weights and init new with small noise
            for (let i = 0; i < newSize; i++) {
                for (let j = 0; j < this.layers[layerIndex + 1].size; j++) {
                    if (i < oldSize) {
                        newW.set([i, j], layer.w.get([i, j]));
                    } else {
                        newW.set([i, j], (Math.random() - 0.5) * 2 * 1e-4);
                    }
                }
            }
            layer.w = newW;
        }

        // Resize W[l-1] (Predicts layer below from this layer)
        if (layerIndex > 0) {
            const prevLayer = this.layers[layerIndex - 1];
            const newPrevW = math.zeros([prevLayer.size, newSize]);
            for (let i = 0; i < prevLayer.size; i++) {
                for (let j = 0; j < newSize; j++) {
                    if (j < oldSize) {
                        newPrevW.set([i, j], prevLayer.w.get([i, j]));
                    } else {
                        newPrevW.set([i, j], (Math.random() - 0.5) * 2 * 1e-4);
                    }
                }
            }
            prevLayer.w = newPrevW;
        }
        
        this.errorMovingAverage[layerIndex] = 0; // Reset error after expansion
    }

    predict() {
        // Unclamp Layer 0 and return the highest probability byte
        // In this byte-index model, we look at the predicted X[0]
        const outputs = this.layers[0].x.toArray();
        let maxIdx = 0;
        let maxVal = -Infinity;
        for (let i = 0; i < outputs.length; i++) {
            if (outputs[i] > maxVal) {
                maxVal = outputs[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    saveToDisk(path) {
        const data = v8.serialize({
            hierarchy: this.layers.map(l => l.size),
            weights: this.layers.map(l => l.w ? l.w.toArray() : null),
            dt: this.dt,
            lr: this.lr
        });
        fs.writeFileSync(path, data);
    }

    static loadFromDisk(path) {
        if (!fs.existsSync(path)) return null;
        const data = v8.deserialize(fs.readFileSync(path));
        const mind = new Mind(data.hierarchy, data.dt, data.lr);
        data.weights.forEach((wData, i) => {
            if (wData) mind.layers[i].w = math.matrix(wData);
        });
        return mind;
    }
}

// Byte encoding utility
function encode(char) {
    const vec = new Array(256).fill(0);
    const byte = char.charCodeAt(0) % 256;
    vec[byte] = 1.0;
    return vec;
}

function decode(byteIdx) {
    return String.fromCharCode(byteIdx);
}

// Interaction Loop
const MIND_PATH = './mind.bin';
let mind = Mind.loadFromDisk(MIND_PATH) || new Mind([256, 512, 256]);

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

console.log("Loom-Node Recursive Predictive Mind Online.");
console.log("Type to chat. The mind learns character-by-character.");

function prompt() {
    rl.question('> ', (input) => {
        // Process input
        for (let char of input) {
            const vec = encode(char);
            mind.step(vec);
        }
        
        // Generate response
        process.stdout.write('Mind: ');
        let response = "";
        // Step without clamping to generate (free-run)
        // We'll give it a few steps to settle and predict next chars
        for (let i = 0; i < 10; i++) {
            // Predict next based on current internal state
            const nextByte = mind.predict();
            const nextChar = decode(nextByte);
            process.stdout.write(nextChar);
            response += nextChar;
            
            // Feed its own prediction back in
            mind.step(encode(nextChar), 5); 
        }
        console.log("\n");
        
        mind.saveToDisk(MIND_PATH);
        prompt();
    });
}

prompt();
