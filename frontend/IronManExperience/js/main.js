// Main entry point for Iron Man Experience

class IronManExperience {
    constructor() {
        this.systems = {};
        this.isReady = false;
        this.config = {
            skipBoot: false, // Set to true for development
            enableVR: false,
            quality: 'high'
        };
        
        // Check URL parameters
        this.parseURLParams();
        
        // Initialize
        this.init();
    }
    
    parseURLParams() {
        const params = new URLSearchParams(window.location.search);
        
        if (params.has('skipBoot')) {
            this.config.skipBoot = params.get('skipBoot') === 'true';
        }
        
        if (params.has('quality')) {
            this.config.quality = params.get('quality');
        }
        
        if (params.has('debug')) {
            window.DEBUG_MODE = params.get('debug') === 'true';
        }
    }
    
    async init() {
        try {
            // Show loading state
            this.showLoadingState();
            
            // Initialize systems in order
            await this.initializeSystems();
            
            // Start experience
            await this.startExperience();
            
            // Mark as ready
            this.isReady = true;
            
            // Notify parent window (for iframe integration)
            if (window.parent !== window) {
                window.parent.postMessage({ type: 'ironman-ready' }, '*');
            }
            
        } catch (error) {
            console.error('Failed to initialize Iron Man Experience:', error);
            this.showError(error.message);
        }
    }
    
    showLoadingState() {
        // Ensure boot sequence is visible
        const bootElement = Utils.$('#bootSequence');
        if (bootElement) {
            bootElement.style.display = 'block';
            bootElement.classList.add('active');
        }
    }
    
    async initializeSystems() {
        console.log('Initializing Iron Man systems...');
        
        // 1. Audio System (initialize first for boot sounds)
        this.systems.audio = new AudioSystem();
        
        // 2. Boot Sequence
        if (!this.config.skipBoot) {
            this.systems.boot = new BootSequence();
            await new Promise(resolve => {
                this.systems.boot.onComplete = resolve;
                this.systems.boot.start();
            });
        } else {
            // Quick boot for development
            Utils.$('#bootSequence').style.display = 'none';
        }
        
        // 3. HUD System
        this.systems.hud = new HUD();
        window.HUD = this.systems.hud; // Global reference
        
        // 4. JARVIS System
        this.systems.jarvis = new JARVIS();
        window.JARVIS = this.systems.jarvis; // Global reference
        
        // 5. 3D Viewport
        this.systems.viewport = new Viewport();
        window.Viewport = this.systems.viewport; // Global reference
        
        // 6. Mission System is already initialized as window.Missions
        this.systems.missions = window.Missions;
        
        console.log('All systems initialized successfully');
    }
    
    async startExperience() {
        // Fade in HUD interface
        const hudInterface = Utils.$('.hud-interface');
        if (hudInterface) {
            hudInterface.style.display = 'block';
            await Utils.delay(100);
            hudInterface.classList.add('active');
        }
        
        // Set up keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Set up UI interactions
        this.setupUIInteractions();
        
        // Start ambient effects
        this.startAmbientEffects();
        
        // Performance monitoring
        if (window.DEBUG_MODE) {
            this.startPerformanceMonitoring();
        }
        
        console.log('Iron Man Experience started');
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Global shortcuts
            switch(e.key.toLowerCase()) {
                case 'f1':
                    e.preventDefault();
                    this.showHelp();
                    break;
                    
                case 'f11':
                    e.preventDefault();
                    this.toggleFullscreen();
                    break;
                    
                case 'escape':
                    if (this.systems.missions?.missionActive) {
                        this.systems.missions.failMission('Mission aborted');
                    }
                    break;
            }
            
            // Ctrl/Cmd shortcuts
            if (e.ctrlKey || e.metaKey) {
                switch(e.key.toLowerCase()) {
                    case 'm':
                        e.preventDefault();
                        this.toggleMute();
                        break;
                        
                    case 'j':
                        e.preventDefault();
                        this.systems.jarvis?.toggleMinimize();
                        break;
                }
            }
        });
    }
    
    setupUIInteractions() {
        // Add hover sounds to interactive elements
        document.querySelectorAll('button, .weapon-item, .mission-item').forEach(element => {
            element.addEventListener('mouseenter', () => {
                this.systems.audio?.playUISound('hover');
            });
            
            element.addEventListener('click', () => {
                this.systems.audio?.playUISound('click');
            });
        });
        
        // Voice command input
        const voiceInput = Utils.$('#voiceCommand');
        if (voiceInput) {
            voiceInput.addEventListener('focus', () => {
                // Check for speech recognition support
                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                    const micButton = Utils.createElement('button', {
                        className: 'mic-button',
                        innerHTML: '🎤',
                        title: 'Voice input'
                    });
                    
                    micButton.addEventListener('click', () => {
                        this.systems.jarvis?.startVoiceRecognition();
                    });
                    
                    voiceInput.parentElement.appendChild(micButton);
                }
            });
        }
    }
    
    startAmbientEffects() {
        // Periodic system status updates
        setInterval(() => {
            // Simulate power drain
            const currentPower = this.systems.hud?.systemStatus.power || 100;
            if (currentPower > 10) {
                this.systems.hud?.updatePower(currentPower - 0.05);
            }
            
            // Random alerts
            if (Math.random() < 0.01) {
                const alerts = [
                    'Perimeter scan complete',
                    'Satellite uplink established',
                    'Weather conditions optimal',
                    'System optimization complete'
                ];
                this.systems.hud?.showAlert(Utils.randomFrom(alerts), 'info');
            }
        }, 5000);
        
        // Dynamic audio based on activity
        if (this.systems.viewport) {
            setInterval(() => {
                const altitude = this.systems.hud?.altitude || 0;
                const velocity = this.systems.hud?.velocity || 0;
                this.systems.audio?.updateFlightAudio(altitude, velocity);
            }, 100);
        }
    }
    
    startPerformanceMonitoring() {
        const stats = Utils.createElement('div', {
            className: 'performance-stats',
            style: 'position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.8); color: #0f0; padding: 10px; font-family: monospace; font-size: 12px; z-index: 9999;'
        });
        
        document.body.appendChild(stats);
        
        setInterval(() => {
            const fps = this.systems.viewport?.renderer?.info?.render?.fps || 0;
            const memory = performance.memory ? 
                `Memory: ${Math.round(performance.memory.usedJSHeapSize / 1048576)}MB` : '';
            
            stats.innerHTML = `
                FPS: ${fps}<br>
                ${memory}<br>
                Systems: ${Object.keys(this.systems).length}<br>
                Mission: ${this.systems.missions?.missionActive ? 'ACTIVE' : 'IDLE'}
            `;
        }, 1000);
    }
    
    // Utility methods
    showHelp() {
        const help = Utils.createElement('div', {
            className: 'help-overlay',
            innerHTML: `
                <div class="help-content">
                    <h2>IRON MAN SUIT CONTROLS</h2>
                    <div class="help-section">
                        <h3>Movement</h3>
                        <p><kbd>W/A/S/D</kbd> - Move</p>
                        <p><kbd>Q/E</kbd> - Altitude</p>
                        <p><kbd>Shift</kbd> - Boost</p>
                        <p><kbd>Mouse</kbd> - Look</p>
                    </div>
                    <div class="help-section">
                        <h3>Combat</h3>
                        <p><kbd>Left Click</kbd> - Fire Repulsor</p>
                        <p><kbd>Tab</kbd> - Target Lock</p>
                        <p><kbd>1-3</kbd> - Select Weapon</p>
                    </div>
                    <div class="help-section">
                        <h3>Interface</h3>
                        <p><kbd>J</kbd> - JARVIS Command</p>
                        <p><kbd>M</kbd> - Mission Select</p>
                        <p><kbd>H</kbd> - Toggle HUD</p>
                        <p><kbd>ESC</kbd> - Abort Mission</p>
                    </div>
                    <div class="help-section">
                        <h3>Voice Commands</h3>
                        <p>"Status" - System status</p>
                        <p>"Scan" - Environmental scan</p>
                        <p>"Weapons" - Weapon status</p>
                        <p>"Mission" - Mission select</p>
                    </div>
                    <button class="close-help">CLOSE</button>
                </div>
            `
        });
        
        document.body.appendChild(help);
        
        help.querySelector('.close-help').addEventListener('click', () => {
            help.remove();
        });
        
        // Close on ESC
        const closeOnEsc = (e) => {
            if (e.key === 'Escape') {
                help.remove();
                document.removeEventListener('keydown', closeOnEsc);
            }
        };
        document.addEventListener('keydown', closeOnEsc);
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                console.error('Fullscreen error:', err);
            });
        } else {
            document.exitFullscreen();
        }
    }
    
    toggleMute() {
        const isMuted = this.systems.audio?.volumes.master === 0;
        this.systems.audio?.setMasterVolume(isMuted ? 0.8 : 0);
        this.systems.hud?.showAlert(isMuted ? 'AUDIO ENABLED' : 'AUDIO MUTED', 'info');
    }
    
    showError(message) {
        const error = Utils.createElement('div', {
            className: 'error-screen',
            innerHTML: `
                <div class="error-content">
                    <h1>SYSTEM ERROR</h1>
                    <p>${message}</p>
                    <button onclick="location.reload()">RELOAD</button>
                </div>
            `
        });
        
        document.body.appendChild(error);
    }
    
    // Public API for external integration
    executeCommand(command) {
        if (this.systems.jarvis) {
            this.systems.jarvis.processCommand(command);
        }
    }
    
    setQuality(level) {
        this.config.quality = level;
        
        // Adjust viewport quality
        if (this.systems.viewport?.renderer) {
            switch(level) {
                case 'low':
                    this.systems.viewport.renderer.setPixelRatio(1);
                    break;
                case 'medium':
                    this.systems.viewport.renderer.setPixelRatio(window.devicePixelRatio * 0.75);
                    break;
                case 'high':
                    this.systems.viewport.renderer.setPixelRatio(window.devicePixelRatio);
                    break;
            }
        }
    }
}

// Style injection for help and error screens
const style = document.createElement('style');
style.textContent = `
    .help-overlay, .error-screen {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    }
    
    .help-content, .error-content {
        background: rgba(0, 0, 0, 0.8);
        border: 2px solid var(--primary-color);
        border-radius: 10px;
        padding: 40px;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
    }
    
    .help-content h2, .error-content h1 {
        color: var(--primary-color);
        margin-bottom: 20px;
        text-align: center;
    }
    
    .help-section {
        margin-bottom: 20px;
    }
    
    .help-section h3 {
        color: var(--primary-color);
        margin-bottom: 10px;
    }
    
    .help-section p {
        margin: 5px 0;
    }
    
    kbd {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 3px;
        padding: 2px 6px;
        font-family: monospace;
        display: inline-block;
        margin: 0 2px;
    }
    
    .close-help, .error-content button {
        display: block;
        margin: 20px auto 0;
        padding: 10px 30px;
        background: var(--primary-color);
        border: none;
        border-radius: 5px;
        color: black;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .close-help:hover, .error-content button:hover {
        background: var(--secondary-color);
        transform: scale(1.05);
    }
    
    .performance-stats {
        backdrop-filter: blur(5px);
    }
    
    .mic-button {
        position: absolute;
        right: 120px;
        top: 50%;
        transform: translateY(-50%);
        background: transparent;
        border: none;
        font-size: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .mic-button:hover {
        transform: translateY(-50%) scale(1.2);
    }
    
    .mission-briefing {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.95);
        border: 2px solid var(--primary-color);
        border-radius: 10px;
        padding: 30px;
        max-width: 600px;
        z-index: 1000;
    }
    
    .briefing-objectives ul {
        list-style: none;
        padding: 0;
    }
    
    .briefing-objectives li {
        padding: 5px 0;
        padding-left: 20px;
        position: relative;
    }
    
    .briefing-objectives li:before {
        content: '▸';
        position: absolute;
        left: 0;
        color: var(--primary-color);
    }
    
    .briefing-actions {
        display: flex;
        gap: 20px;
        justify-content: center;
        margin-top: 30px;
    }
    
    .briefing-actions button {
        padding: 10px 30px;
        background: transparent;
        border: 1px solid var(--primary-color);
        color: var(--primary-color);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .start-mission {
        background: var(--primary-color) !important;
        color: black !important;
    }
    
    .briefing-actions button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px var(--primary-color);
    }
`;
document.head.appendChild(style);

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.IronManExperience = new IronManExperience();
});

// Handle visibility change
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Pause when tab is hidden
        if (window.AudioSystem) {
            window.AudioSystem.setMasterVolume(0);
        }
    } else {
        // Resume when tab is visible
        if (window.AudioSystem) {
            window.AudioSystem.setMasterVolume(0.8);
        }
    }
});