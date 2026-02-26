const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

const ARGS = process.argv.slice(2);
const DATA_DIR = './data';

if (ARGS.length < 1) {
    console.log("Usage: node scraper.js <url>");
    process.exit(1);
}

const startUrl = ARGS[0];

if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR);
}

async function scrapePage(url) {
    try {
        console.log(`Scraping: ${url}`);
        const response = await axios.get(url, { timeout: 5000 });
        const $ = cheerio.load(response.data);
        
        // Remove scripts, styles, etc.
        $('script, style, nav, footer, header, noscript').remove();
        
        const text = $('body').text().replace(/\s+/g, ' ').trim();
        const filename = url.replace(/[^a-zA-Z0-9]/g, '_').substring(0, 100) + '.txt';
        const filePath = path.join(DATA_DIR, filename);
        
        fs.writeFileSync(filePath, text);
        console.log(`Saved to ${filePath} (${text.length} chars)`);
        
        const links = [];
        $('a').each((i, el) => {
            const href = $(el).attr('href');
            if (href) {
                try {
                    const fullUrl = new URL(href, url).href;
                    if (fullUrl.startsWith('http')) {
                        links.push(fullUrl);
                    }
                } catch (e) {
                    // Ignore invalid URLs
                }
            }
        });
        
        return Array.from(new Set(links));
    } catch (err) {
        console.error(`Error scraping ${url}: ${err.message}`);
        return [];
    }
}

async function main() {
    console.log(`Starting scraper for ${startUrl}...`);
    const subLinks = await scrapePage(startUrl);
    
    console.log(`Found ${subLinks.length} sub-links. Scraping each...`);
    
    // Only one level deep as requested
    for (let i = 0; i < subLinks.length; i++) {
        const link = subLinks[i];
        process.stdout.write(`[${i+1}/${subLinks.length}] `);
        await scrapePage(link);
    }
    
    console.log("\nScraping complete. You can now run the trainer on the './data' directory.");
}

main();
