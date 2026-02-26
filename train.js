const fs = require('fs');
const path = require('path');
const { Mind, encode } = require('./loom');

const MIND_PATH = './mind.bin';
const ARGS = process.argv.slice(2);

if (ARGS.length < 1) {
    console.log("Usage: node train.js <path_to_file_or_folder>");
    process.exit(1);
}

const inputPath = ARGS[0];

if (!fs.existsSync(inputPath)) {
    console.error(`Path not found: ${inputPath}`);
    process.exit(1);
}

let mind = Mind.loadFromDisk(MIND_PATH) || new Mind([256, 512, 256]);

function getAllFiles(dirPath, arrayOfFiles) {
    const files = fs.readdirSync(dirPath);

    arrayOfFiles = arrayOfFiles || [];

    files.forEach(function(file) {
        if (fs.statSync(path.join(dirPath, file)).isDirectory()) {
            arrayOfFiles = getAllFiles(path.join(dirPath, file), arrayOfFiles);
        } else {
            arrayOfFiles.push(path.join(dirPath, file));
        }
    });

    return arrayOfFiles;
}

const filesToProcess = fs.statSync(inputPath).isDirectory() 
    ? getAllFiles(inputPath) 
    : [inputPath];

console.log(`Found ${filesToProcess.length} file(s) to process.`);

filesToProcess.forEach((filePath, fileIdx) => {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        console.log(`\n[${fileIdx + 1}/${filesToProcess.length}] Training on: ${filePath} (${content.length} characters)`);

        for (let i = 0; i < content.length; i++) {
            const char = content[i];
            const vec = encode(char);
            
            // Process character
            mind.step(vec, 10);
            
            if (i % 500 === 0 && i > 0) {
                const progress = ((i / content.length) * 100).toFixed(2);
                process.stdout.write(`\rProgress: ${progress}% (${i}/${content.length})`);
            }
        }
        console.log(`\nFinished ${filePath}. Saving...`);
        mind.saveToDisk(MIND_PATH);
    } catch (err) {
        console.error(`Error reading ${filePath}: ${err.message}`);
    }
});

mind.saveToDisk(MIND_PATH);
console.log("\nBatch training complete. Mind saved to disk.");
