const fs = require('fs');
const v8 = require('v8');
const math = require('mathjs');

class Layer {
    constructor(size, nextSize = 0) {
        this.size = size;
        this.x = math.zeros(size);
        this.e = math.zeros(size);
        this.bias = math.zeros(size);
        
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
        this.alpha = 0.05;
        this.expansionThreshold = 0.5;

        for (let i = 0; i < hierarchy.length; i++) {
            const size = hierarchy[i];
            const nextSize = i + 1 < hierarchy.length ? hierarchy[i + 1] : 0;
            this.layers.push(new Layer(size, nextSize));
        }
    }

    step(inputVector, iterations = 20) {
        this.layers[0].x = math.matrix(inputVector);

        for (let iter = 0; iter < iterations; iter++) {
            for (let l = 0; l < this.layers.length - 1; l++) {
                const currentLayer = this.layers[l];
                const nextLayer = this.layers[l + 1];
                const prediction = math.multiply(currentLayer.w, nextLayer.x);
                currentLayer.e = math.subtract(currentLayer.x, prediction);
                const mag = math.norm(currentLayer.e);
                this.errorMovingAverage[l] = (this.alpha * mag) + (1 - this.alpha) * this.errorMovingAverage[l];
            }

            for (let l = 1; l < this.layers.length; l++) {
                const currentLayer = this.layers[l];
                const prevLayer = this.layers[l - 1];
                const feedbackDrive = math.multiply(math.transpose(prevLayer.w), prevLayer.e);
                const deltaX = math.multiply(this.dt, math.subtract(feedbackDrive, currentLayer.e));
                currentLayer.x = math.add(currentLayer.x, deltaX);
            }

            for (let l = 0; l < this.layers.length - 1; l++) {
                const currentLayer = this.layers[l];
                const nextLayer = this.layers[l + 1];
                const dw = math.multiply(this.lr, math.multiply(math.reshape(currentLayer.e, [-1, 1]), math.reshape(nextLayer.x, [1, -1])));
                currentLayer.w = math.add(currentLayer.w, dw);
            }

            for (let l = 0; l < this.layers.length; l++) {
                if (this.errorMovingAverage[l] > this.expansionThreshold) {
                    this.expand(l, 8);
                }
            }
        }
    }

    expand(layerIndex, count) {
        const layer = this.layers[layerIndex];
        const oldSize = layer.size;
        const newSize = oldSize + count;

        const newX = math.zeros(newSize);
        const newE = math.zeros(newSize);
        const newBias = math.zeros(newSize);
        
        layer.x.forEach((val, idx) => newX.set(idx, val));
        layer.x = newX;
        layer.e = newE;
        layer.bias = newBias;
        layer.size = newSize;

        if (layer.w) {
            const newW = math.zeros([newSize, this.layers[layerIndex + 1].size]);
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
        
        this.errorMovingAverage[layerIndex] = 0;
    }

    predict() {
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

function encode(char) {
    const vec = new Array(256).fill(0);
    const byte = char.charCodeAt(0) % 256;
    vec[byte] = 1.0;
    return vec;
}

function decode(byteIdx) {
    return String.fromCharCode(byteIdx);
}

module.exports = { Layer, Mind, encode, decode };
