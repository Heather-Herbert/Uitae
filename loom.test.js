const { Mind, encode, decode } = require('./loom');
const fs = require('fs');

describe('Loom-Node Mind', () => {
    const hierarchy = [4, 8, 4];
    const MIND_TEST_PATH = './test_mind.bin';

    afterAll(() => {
        if (fs.existsSync(MIND_TEST_PATH)) {
            fs.unlinkSync(MIND_TEST_PATH);
        }
    });

    test('should initialize with correct hierarchy', () => {
        const mind = new Mind(hierarchy);
        expect(mind.layers.length).toBe(3);
        expect(mind.layers[0].size).toBe(4);
        expect(mind.layers[1].size).toBe(8);
        expect(mind.layers[2].size).toBe(4);
    });

    test('should perform a step and update states', () => {
        const mind = new Mind(hierarchy);
        const input = [1, 0, 0, 0];
        const initialX1 = mind.layers[1].x.toArray();
        
        mind.step(input, 5);
        
        const updatedX1 = mind.layers[1].x.toArray();
        // Expect some change in internal state after steps
        expect(updatedX1).not.toEqual(initialX1);
    });

    test('should encode and decode characters correctly', () => {
        const char = 'A';
        const vec = encode(char);
        expect(vec.length).toBe(256);
        expect(vec[65]).toBe(1.0);
        
        expect(decode(65)).toBe('A');
    });

    test('should save and load from disk', () => {
        const mind = new Mind(hierarchy);
        mind.saveToDisk(MIND_TEST_PATH);
        
        const loadedMind = Mind.loadFromDisk(MIND_TEST_PATH);
        expect(loadedMind).not.toBeNull();
        expect(loadedMind.layers.length).toBe(3);
        expect(loadedMind.layers[1].size).toBe(8);
    });

    test('should expand a layer when error is high', () => {
        const mind = new Mind(hierarchy);
        const oldSize = mind.layers[1].size;
        
        mind.expand(1, 2);
        
        expect(mind.layers[1].size).toBe(oldSize + 2);
        expect(mind.layers[1].x.size()).toEqual([oldSize + 2]);
        // Check if weight matrices were resized
        expect(mind.layers[1].w.size()).toEqual([oldSize + 2, 4]);
        expect(mind.layers[0].w.size()).toEqual([4, oldSize + 2]);
    });

    test('should predict a byte index', () => {
        const mind = new Mind(hierarchy);
        const prediction = mind.predict();
        expect(typeof prediction).toBe('number');
        expect(prediction).toBeGreaterThanOrEqual(0);
        expect(prediction).toBeLessThan(256);
    });
});
