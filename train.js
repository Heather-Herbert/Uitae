const fs = require('fs');
const { Mind, encode } = require('./loom');

const MIND_PATH = './mind.bin';
const ARGS = process.argv.slice(2);

if (ARGS.length < 1) {
    console.log("Usage: node train.js <path_to_text_file>");
    process.exit(1);
}

const filePath = ARGS[0];

if (!fs.existsSync(filePath)) {
    console.error(`File not found: ${filePath}`);
    process.exit(1);
}

const content = fs.readFileSync(filePath, 'utf8');
let mind = Mind.loadFromDisk(MIND_PATH) || new Mind([256, 512, 256]);

console.log(`Training on ${filePath} (${content.length} characters)...`);

for (let i = 0; i < content.length; i++) {
    const char = content[i];
    const vec = encode(char);
    
    // Process character
    mind.step(vec, 10); // Use 10 iterations for speed during batch training
    
    if (i % 100 === 0 && i > 0) {
        const progress = ((i / content.length) * 100).toFixed(2);
        console.log(`Progress: ${progress}% (${i}/${content.length})`);
        // Periodically save
        mind.saveToDisk(MIND_PATH);
    }
}

mind.saveToDisk(MIND_PATH);
console.log("Training complete. Mind saved to disk.");
