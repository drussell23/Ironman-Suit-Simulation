const express = require('express');
const compression = require('compression');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 8080;

// Enable compression for better performance
app.use(compression());

// Enable CORS for development
app.use(cors());

// Custom middleware to set correct MIME types for Unity WebGL
app.use((req, res, next) => {
    const url = req.url;
    
    // Set appropriate headers for Unity WebGL files
    if (url.endsWith('.wasm')) {
        res.setHeader('Content-Type', 'application/wasm');
    } else if (url.endsWith('.wasm.gz')) {
        res.setHeader('Content-Type', 'application/wasm');
        res.setHeader('Content-Encoding', 'gzip');
    } else if (url.endsWith('.js.gz')) {
        res.setHeader('Content-Type', 'application/javascript');
        res.setHeader('Content-Encoding', 'gzip');
    } else if (url.endsWith('.data.gz')) {
        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Content-Encoding', 'gzip');
    }
    
    // Enable SharedArrayBuffer for better performance (requires HTTPS in production)
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    
    next();
});

// Serve static files from the IronManExperience directory
const experiencePath = path.join(__dirname, 'IronManExperience');
app.use(express.static(experiencePath));

// Also check for Unity WebGL build
const buildPath = path.join(__dirname, 'IronManExperience', 'Build');
if (fs.existsSync(buildPath)) {
    app.use('/build', express.static(buildPath));
}

// Serve the index.html for the root path
app.get('/', (req, res) => {
    const webIndexPath = path.join(experiencePath, 'index.html');
    const buildIndexPath = path.join(buildPath, 'index.html');
    
    // First try to serve the web-based experience
    if (fs.existsSync(webIndexPath)) {
        res.sendFile(webIndexPath);
    } else if (fs.existsSync(buildIndexPath)) {
        // Fall back to Unity WebGL build if it exists
        res.sendFile(buildIndexPath);
    } else {
        // Send a default page if neither exists
        res.send(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>Iron Man Suit Experience</title>
                <style>
                    body {
                        background-color: #000;
                        color: #00a8ff;
                        font-family: Arial, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                    }
                    .container {
                        text-align: center;
                    }
                    h1 {
                        font-size: 3em;
                        margin-bottom: 0.5em;
                    }
                    p {
                        font-size: 1.2em;
                        margin: 1em 0;
                    }
                    code {
                        background: #111;
                        padding: 0.3em;
                        border-radius: 3px;
                        display: inline-block;
                        margin: 0.5em 0;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Iron Man Suit Experience</h1>
                    <p>Experience files not found!</p>
                    <p>Please check that the IronManExperience directory exists.</p>
                </div>
            </body>
            </html>
        `);
    }
});

// API endpoint for suit telemetry (for future expansion)
app.get('/api/telemetry', (req, res) => {
    res.json({
        status: 'online',
        power: 98,
        altitude: 0,
        velocity: 0,
        systems: {
            jarvis: 'online',
            weapons: 'standby',
            shields: 'active'
        }
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`
╔═══════════════════════════════════════════╗
║       Iron Man Suit Experience            ║
║       Local Development Server            ║
╠═══════════════════════════════════════════╣
║  Server running at:                       ║
║  http://localhost:${PORT}                    ║
║                                           ║
║  JARVIS Status: Online                    ║
║  Arc Reactor: Stable                      ║
║  Systems: All Green                       ║
╚═══════════════════════════════════════════╝

Press Ctrl+C to stop the server
    `);
    
    // Open browser automatically (optional)
    const platform = process.platform;
    const url = `http://localhost:${PORT}`;
    
    if (platform === 'darwin') {
        require('child_process').exec(`open ${url}`);
    } else if (platform === 'win32') {
        require('child_process').exec(`start ${url}`);
    }
});