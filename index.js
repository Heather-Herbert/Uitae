const readline = require('readline');
const { Mind, encode, decode } = require('./loom');

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
        // Step without clamping to generate (free-run)
        // We'll give it a few steps to settle and predict next chars
        for (let i = 0; i < 10; i++) {
            // Predict next based on current internal state
            const nextByte = mind.predict();
            const nextChar = decode(nextByte);
            process.stdout.write(nextChar);
            
            // Feed its own prediction back in
            mind.step(encode(nextChar), 5); 
        }
        console.log("\n");
        
        mind.saveToDisk(MIND_PATH);
        prompt();
    });
}

prompt();
