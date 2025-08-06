// Advanced JARVIS System - Just A Rather Very Intelligent System
// Enhanced AI assistant with contextual awareness and advanced capabilities

class JARVISAdvanced {
    constructor() {
        // Core elements
        this.interface = null;
        this.container = null;
        this.initialized = false;
        
        // State management
        this.state = {
            mode: 'standby', // standby, active, listening, thinking, alert
            context: null,
            lastCommand: null,
            conversation: [],
            systemStatus: {
                power: 98,
                armor: 100,
                temperature: 36.5,
                altitude: 0,
                velocity: 0,
                shields: false,
                weapons: 'standby'
            }
        };
        
        // Advanced features
        this.features = {
            voiceRecognition: false,
            speechSynthesis: false,
            contextualAwareness: true,
            predictiveAnalysis: true,
            emotionalResponse: true
        };
        
        // Command registry with enhanced handlers
        this.commands = {
            // System commands
            'status': {
                handler: () => this.handleStatusCommand(),
                description: 'Full system status report',
                category: 'system'
            },
            'diagnostic': {
                handler: () => this.runDiagnostics(),
                description: 'Run complete system diagnostics',
                category: 'system'
            },
            'power': {
                handler: (args) => this.handlePowerCommand(args),
                description: 'Power management controls',
                category: 'system'
            },
            
            // Combat commands
            'weapons': {
                handler: (args) => this.handleWeaponsCommand(args),
                description: 'Weapons system control',
                category: 'combat'
            },
            'target': {
                handler: (args) => this.handleTargetCommand(args),
                description: 'Target acquisition and tracking',
                category: 'combat'
            },
            'shield': {
                handler: (args) => this.handleShieldCommand(args),
                description: 'Shield system control',
                category: 'combat'
            },
            
            // Navigation commands
            'scan': {
                handler: (args) => this.handleScanCommand(args),
                description: 'Environmental scanning',
                category: 'navigation'
            },
            'route': {
                handler: (args) => this.calculateRoute(args),
                description: 'Route calculation and navigation',
                category: 'navigation'
            },
            'altitude': {
                handler: (args) => this.adjustAltitude(args),
                description: 'Altitude control',
                category: 'navigation'
            },
            
            // Analysis commands
            'analyze': {
                handler: (args) => this.handleAnalysisCommand(args),
                description: 'Deep analysis mode',
                category: 'analysis'
            },
            'predict': {
                handler: (args) => this.runPredictiveAnalysis(args),
                description: 'Predictive analysis',
                category: 'analysis'
            },
            
            // Communication
            'call': {
                handler: (args) => this.initiateCall(args),
                description: 'Communications system',
                category: 'communication'
            },
            'message': {
                handler: (args) => this.sendMessage(args),
                description: 'Send encrypted message',
                category: 'communication'
            },
            
            // Utility commands
            'music': {
                handler: (args) => this.handleMusicCommand(args),
                description: 'Music playback control',
                category: 'utility'
            },
            'reminder': {
                handler: (args) => this.setReminder(args),
                description: 'Set reminders and alerts',
                category: 'utility'
            },
            'search': {
                handler: (args) => this.performSearch(args),
                description: 'Database search',
                category: 'utility'
            }
        };
        
        // Response personality
        this.personality = {
            formal: 0.8,
            humor: 0.2,
            concern: 0.7,
            efficiency: 0.9
        };
        
        // Initialize speech synthesis voices
        this.jarvisVoice = null;
        this.initSpeechSynthesis();
    }
    
    async initialize() {
        if (this.initialized) return;
        
        // Create advanced interface
        this.createInterface();
        
        // Setup event handlers
        this.setupEventHandlers();
        
        // Initialize features
        this.checkFeatureSupport();
        
        // Boot sequence
        await this.bootSequence();
        
        this.initialized = true;
    }
    
    createInterface() {
        // Main container
        this.container = document.createElement('div');
        this.container.className = 'jarvis-advanced-container';
        this.container.innerHTML = `
            <div class="jarvis-hologram">
                <div class="hologram-base">
                    <div class="hologram-ring ring-1"></div>
                    <div class="hologram-ring ring-2"></div>
                    <div class="hologram-ring ring-3"></div>
                </div>
                <div class="hologram-core">
                    <div class="core-sphere">
                        <div class="energy-field"></div>
                        <div class="data-streams">
                            <div class="stream"></div>
                            <div class="stream"></div>
                            <div class="stream"></div>
                        </div>
                    </div>
                </div>
                <div class="hologram-particles"></div>
            </div>
            
            <div class="jarvis-interface-panel">
                <div class="interface-header">
                    <div class="status-indicators">
                        <div class="indicator" data-status="connection"></div>
                        <div class="indicator" data-status="processing"></div>
                        <div class="indicator" data-status="security"></div>
                    </div>
                    <div class="interface-title">J.A.R.V.I.S.</div>
                    <div class="interface-controls">
                        <button class="control-btn" data-action="minimize">_</button>
                        <button class="control-btn" data-action="maximize">□</button>
                        <button class="control-btn" data-action="settings">⚙</button>
                    </div>
                </div>
                
                <div class="conversation-display">
                    <div class="conversation-history"></div>
                    <div class="current-response">
                        <div class="response-avatar">
                            <canvas class="waveform-canvas" width="120" height="60"></canvas>
                        </div>
                        <div class="response-text"></div>
                    </div>
                </div>
                
                <div class="input-section">
                    <div class="input-wrapper">
                        <input type="text" class="command-input" placeholder="Speak or type your command...">
                        <button class="voice-btn" data-action="voice">🎤</button>
                    </div>
                    <div class="suggestions"></div>
                </div>
                
                <div class="quick-actions">
                    <button class="action-btn" data-command="status">Status</button>
                    <button class="action-btn" data-command="scan">Scan</button>
                    <button class="action-btn" data-command="analyze">Analyze</button>
                    <button class="action-btn" data-command="weapons">Weapons</button>
                </div>
            </div>
            
            <!-- Contextual overlay -->
            <div class="jarvis-overlay" style="display: none;">
                <div class="overlay-content"></div>
            </div>
            
            <!-- Alert system -->
            <div class="jarvis-alerts"></div>
        `;
        
        // Add to HUD
        const hudInterface = document.querySelector('.hud-interface');
        if (hudInterface) {
            hudInterface.appendChild(this.container);
        } else {
            document.body.appendChild(this.container);
        }
        
        // Cache elements
        this.elements = {
            hologram: this.container.querySelector('.jarvis-hologram'),
            interface: this.container.querySelector('.jarvis-interface-panel'),
            conversation: this.container.querySelector('.conversation-history'),
            responseText: this.container.querySelector('.response-text'),
            input: this.container.querySelector('.command-input'),
            voiceBtn: this.container.querySelector('.voice-btn'),
            suggestions: this.container.querySelector('.suggestions'),
            overlay: this.container.querySelector('.jarvis-overlay'),
            alerts: this.container.querySelector('.jarvis-alerts'),
            waveform: this.container.querySelector('.waveform-canvas'),
            indicators: this.container.querySelectorAll('.indicator')
        };
        
        // Initialize waveform
        this.initWaveform();
    }
    
    initWaveform() {
        const canvas = this.elements.waveform;
        const ctx = canvas.getContext('2d');
        this.waveformCtx = ctx;
        this.waveformData = new Array(30).fill(0);
        
        this.animateWaveform();
    }
    
    animateWaveform() {
        const ctx = this.waveformCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Update waveform data based on state
        if (this.state.mode === 'listening' || this.state.mode === 'thinking') {
            for (let i = 0; i < this.waveformData.length; i++) {
                this.waveformData[i] = Math.sin(Date.now() * 0.01 + i) * 
                    (this.state.mode === 'thinking' ? 20 : 10) + 
                    Math.random() * 10;
            }
        } else if (this.state.mode === 'active') {
            for (let i = 0; i < this.waveformData.length; i++) {
                this.waveformData[i] *= 0.9;
                if (Math.random() < 0.1) {
                    this.waveformData[i] = Math.random() * 30;
                }
            }
        } else {
            for (let i = 0; i < this.waveformData.length; i++) {
                this.waveformData[i] *= 0.95;
            }
        }
        
        // Draw waveform
        ctx.strokeStyle = '#00ccff';
        ctx.lineWidth = 2;
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#00ccff';
        
        const barWidth = width / this.waveformData.length;
        
        for (let i = 0; i < this.waveformData.length; i++) {
            const x = i * barWidth + barWidth / 2;
            const barHeight = Math.abs(this.waveformData[i]);
            
            ctx.beginPath();
            ctx.moveTo(x, height / 2 - barHeight / 2);
            ctx.lineTo(x, height / 2 + barHeight / 2);
            ctx.stroke();
        }
        
        requestAnimationFrame(() => this.animateWaveform());
    }
    
    setupEventHandlers() {
        // Input handling
        this.elements.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processCommand(e.target.value);
                e.target.value = '';
            }
        });
        
        // Voice button
        this.elements.voiceBtn.addEventListener('click', () => {
            this.toggleVoiceRecognition();
        });
        
        // Quick actions
        this.container.querySelectorAll('.action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const command = btn.dataset.command;
                this.processCommand(command);
            });
        });
        
        // Control buttons
        this.container.querySelectorAll('.control-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.handleControlAction(btn.dataset.action);
            });
        });
        
        // Auto-suggestions
        this.elements.input.addEventListener('input', (e) => {
            this.updateSuggestions(e.target.value);
        });
    }
    
    async bootSequence() {
        this.setState('thinking');
        
        // Initialization messages
        const bootMessages = [
            "Initializing J.A.R.V.I.S. system...",
            "Loading personality matrix...",
            "Establishing neural pathways...",
            "Connecting to Stark Industries network...",
            "Calibrating voice synthesis...",
            "System ready."
        ];
        
        for (const message of bootMessages) {
            await this.displaySystemMessage(message);
            await this.delay(300);
        }
        
        this.setState('active');
        
        // Greeting
        const hour = new Date().getHours();
        let greeting;
        if (hour < 12) greeting = "Good morning";
        else if (hour < 18) greeting = "Good afternoon";
        else greeting = "Good evening";
        
        await this.speak(`${greeting}, sir. All systems are operational.`);
    }
    
    setState(mode) {
        this.state.mode = mode;
        this.container.dataset.mode = mode;
        
        // Update indicators
        this.elements.indicators[0].className = `indicator ${mode === 'active' ? 'active' : ''}`;
        this.elements.indicators[1].className = `indicator ${mode === 'thinking' ? 'active' : ''}`;
        this.elements.indicators[2].className = 'indicator active'; // Always secure
        
        // Update hologram
        this.updateHologram();
    }
    
    updateHologram() {
        const hologram = this.elements.hologram;
        hologram.className = `jarvis-hologram ${this.state.mode}`;
        
        // Add particles for certain states
        if (this.state.mode === 'thinking' || this.state.mode === 'alert') {
            this.addHologramParticles();
        }
    }
    
    addHologramParticles() {
        const container = this.elements.hologram.querySelector('.hologram-particles');
        container.innerHTML = '';
        
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = `${Math.random() * 100}%`;
            particle.style.animationDelay = `${Math.random() * 2}s`;
            particle.style.animationDuration = `${2 + Math.random() * 3}s`;
            container.appendChild(particle);
        }
    }
    
    async processCommand(input) {
        if (!input.trim()) return;
        
        // Add to conversation
        this.addToConversation('user', input);
        
        // Update state
        this.setState('thinking');
        this.state.lastCommand = input;
        
        // Parse command
        const parsed = this.parseCommand(input);
        
        // Execute command
        if (parsed.command && this.commands[parsed.command]) {
            await this.commands[parsed.command].handler(parsed.args);
        } else {
            // Use NLP for complex commands
            await this.interpretNaturalLanguage(input);
        }
        
        // Update state
        this.setState('active');
    }
    
    parseCommand(input) {
        const lower = input.toLowerCase().trim();
        const words = lower.split(' ');
        const command = words[0];
        const args = words.slice(1).join(' ');
        
        // Check for command aliases
        const aliases = {
            'hello': 'greet',
            'hi': 'greet',
            'shoot': 'weapons',
            'fire': 'weapons',
            'fly': 'altitude',
            'shields': 'shield',
            'music': 'music',
            'play': 'music'
        };
        
        const resolvedCommand = aliases[command] || command;
        
        return {
            command: this.commands[resolvedCommand] ? resolvedCommand : null,
            args: args,
            original: input
        };
    }
    
    async interpretNaturalLanguage(input) {
        const lower = input.toLowerCase();
        
        // Context-aware responses
        if (this.state.context === 'combat') {
            if (lower.includes('dodge') || lower.includes('evade')) {
                await this.executeCombatManeuver('evasive');
                return;
            }
        }
        
        // Pattern matching
        const patterns = [
            {
                pattern: /what('s| is) (the )?(.*) (status|level)/i,
                handler: (match) => this.queryStatus(match[3])
            },
            {
                pattern: /how (fast|high|far|much)/i,
                handler: () => this.handleMetricsQuery(input)
            },
            {
                pattern: /show me|display|bring up/i,
                handler: () => this.handleDisplayRequest(input)
            },
            {
                pattern: /calculate|compute|figure out/i,
                handler: () => this.performCalculation(input)
            },
            {
                pattern: /remind me|set (a )?reminder/i,
                handler: () => this.setReminder(input)
            },
            {
                pattern: /call|contact|reach/i,
                handler: () => this.initiateCall(input)
            },
            {
                pattern: /emergency|mayday|help/i,
                handler: () => this.activateEmergencyProtocol()
            },
            {
                pattern: /joke|funny|humor/i,
                handler: () => this.tellJoke()
            }
        ];
        
        // Check patterns
        for (const { pattern, handler } of patterns) {
            const match = input.match(pattern);
            if (match) {
                await handler(match);
                return;
            }
        }
        
        // Default response
        await this.speak("I'm not sure I understand that command, sir. Could you please clarify?");
    }
    
    async speak(text, options = {}) {
        const {
            priority = 'normal',
            emotion = 'neutral',
            speed = 1.0
        } = options;
        
        // Add to conversation
        this.addToConversation('jarvis', text);
        
        // Display text with typing effect
        await this.typeText(text);
        
        // Text-to-speech
        if (this.features.speechSynthesis && this.jarvisVoice) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.voice = this.jarvisVoice;
            utterance.rate = speed;
            utterance.pitch = emotion === 'concerned' ? 0.9 : 1.0;
            utterance.volume = 0.8;
            
            speechSynthesis.speak(utterance);
        }
        
        // Play sound effect
        const audio = document.getElementById('jarvisSound');
        if (audio) {
            audio.play().catch(() => {});
        }
    }
    
    async typeText(text) {
        const element = this.elements.responseText;
        element.textContent = '';
        
        for (let i = 0; i < text.length; i++) {
            element.textContent += text[i];
            await this.delay(30);
        }
    }
    
    addToConversation(speaker, text) {
        const entry = document.createElement('div');
        entry.className = `conversation-entry ${speaker}`;
        entry.innerHTML = `
            <div class="speaker">${speaker.toUpperCase()}</div>
            <div class="message">${text}</div>
            <div class="timestamp">${new Date().toLocaleTimeString()}</div>
        `;
        
        this.elements.conversation.appendChild(entry);
        this.elements.conversation.scrollTop = this.elements.conversation.scrollHeight;
        
        // Keep conversation history
        this.state.conversation.push({ speaker, text, timestamp: Date.now() });
        if (this.state.conversation.length > 100) {
            this.state.conversation.shift();
        }
    }
    
    // Enhanced command handlers
    async handleStatusCommand() {
        const status = this.state.systemStatus;
        const report = `System status report: Power levels at ${status.power}%. 
            Armor integrity: ${status.armor}%. 
            Core temperature: ${status.temperature} degrees Celsius. 
            ${status.shields ? 'Shields are active.' : 'Shields on standby.'} 
            ${this.getSystemAssessment()}`;
        
        await this.speak(report);
        
        // Show visual status
        this.showStatusOverlay();
    }
    
    getSystemAssessment() {
        const status = this.state.systemStatus;
        
        if (status.power < 20) return "Warning: Power critically low.";
        if (status.armor < 50) return "Armor requires immediate attention.";
        if (status.temperature > 40) return "Temperature exceeding optimal range.";
        
        return "All systems operating within normal parameters.";
    }
    
    async runDiagnostics() {
        await this.speak("Initiating full system diagnostic...");
        this.setState('thinking');
        
        // Simulate diagnostic process
        const systems = [
            'Arc Reactor',
            'Repulsor Array',
            'Flight Systems',
            'Targeting Computer',
            'Environmental Controls',
            'Communications Array'
        ];
        
        for (const system of systems) {
            await this.displaySystemMessage(`Checking ${system}...`);
            await this.delay(500);
        }
        
        await this.speak("Diagnostics complete. All systems functioning at optimal capacity.");
        this.setState('active');
    }
    
    async handlePowerCommand(args) {
        if (args.includes('boost') || args.includes('maximum')) {
            this.state.systemStatus.power = Math.min(150, this.state.systemStatus.power + 20);
            await this.speak("Diverting auxiliary power. Arc reactor output increased to maximum.");
        } else if (args.includes('save') || args.includes('conserve')) {
            await this.speak("Entering power conservation mode. Non-essential systems offline.");
        } else {
            await this.speak(`Current power level: ${this.state.systemStatus.power}%`);
        }
    }
    
    async handleWeaponsCommand(args) {
        if (args.includes('ready') || args.includes('online')) {
            this.state.systemStatus.weapons = 'ready';
            await this.speak("Weapons systems online. Repulsors charged and ready.");
            
            // Visual effect
            if (window.HUD) {
                window.HUD.showWeaponStatus();
            }
        } else if (args.includes('fire') || args.includes('engage')) {
            if (this.state.systemStatus.weapons === 'ready') {
                await this.speak("Firing repulsors.");
                this.fireRepulsors();
            } else {
                await this.speak("Weapons systems must be brought online first, sir.");
            }
        } else {
            await this.speak("Weapons on standby. Say 'weapons ready' to arm.");
        }
    }
    
    async handleTargetCommand(args) {
        await this.speak("Activating targeting system...");
        this.showTargetingOverlay();
        
        if (args.includes('lock') || args.includes('acquire')) {
            await this.delay(1500);
            await this.speak("Target acquired. Awaiting your command.");
        }
    }
    
    async handleShieldCommand(args) {
        if (args.includes('up') || args.includes('activate')) {
            this.state.systemStatus.shields = true;
            await this.speak("Energy shields activated. Maximum protection engaged.");
        } else if (args.includes('down') || args.includes('deactivate')) {
            this.state.systemStatus.shields = false;
            await this.speak("Shields deactivated to conserve power.");
        } else {
            const status = this.state.systemStatus.shields ? 'active' : 'inactive';
            await this.speak(`Shield status: ${status}`);
        }
    }
    
    async handleScanCommand(args) {
        await this.speak("Initiating environmental scan...");
        this.setState('thinking');
        
        // Show scanning animation
        this.showScanningOverlay();
        
        await this.delay(3000);
        
        const results = this.generateScanResults();
        await this.speak(results);
        
        this.setState('active');
    }
    
    generateScanResults() {
        const threats = Math.floor(Math.random() * 3);
        const lifeforms = Math.floor(Math.random() * 10) + 5;
        const structures = Math.floor(Math.random() * 5) + 2;
        
        let report = `Scan complete. Detected ${lifeforms} life forms, ${structures} structural elements.`;
        
        if (threats > 0) {
            report += ` Warning: ${threats} potential threat${threats > 1 ? 's' : ''} identified.`;
            this.showAlert('Threat Detected', 'warning');
        } else {
            report += " No immediate threats detected.";
        }
        
        return report;
    }
    
    async handleAnalysisCommand(args) {
        await this.speak("Entering deep analysis mode...");
        this.setState('thinking');
        
        if (args) {
            // Analyze specific target
            await this.speak(`Analyzing ${args}...`);
            await this.delay(2000);
            
            const analysis = this.performAnalysis(args);
            await this.speak(analysis);
        } else {
            await this.speak("Please specify target for analysis.");
        }
        
        this.setState('active');
    }
    
    performAnalysis(target) {
        // Simulated analysis results
        const analyses = {
            'enemy': 'Target appears to be heavily armored. Recommend precision strikes to weak points.',
            'building': 'Structural integrity: 87%. Multiple entry points identified.',
            'weather': 'Atmospheric conditions stable. Visibility optimal for flight operations.',
            'default': `Analysis of ${target} complete. Data stored for future reference.`
        };
        
        for (const [key, result] of Object.entries(analyses)) {
            if (target.includes(key)) return result;
        }
        
        return analyses.default;
    }
    
    async runPredictiveAnalysis(args) {
        await this.speak("Running predictive analysis algorithms...");
        this.setState('thinking');
        
        await this.delay(2000);
        
        const prediction = `Based on current data, probability of success: ${Math.floor(Math.random() * 30) + 70}%. 
            Recommended course of action: Proceed with caution.`;
        
        await this.speak(prediction);
        this.setState('active');
    }
    
    async handleMusicCommand(args) {
        if (args.includes('play') || args.includes('start')) {
            const genre = args.includes('rock') ? 'AC/DC' : 'your preferred playlist';
            await this.speak(`Now playing ${genre}, sir.`);
        } else if (args.includes('stop') || args.includes('pause')) {
            await this.speak("Music paused.");
        } else {
            await this.speak("Would you like me to play something specific, sir?");
        }
    }
    
    async setReminder(args) {
        const time = this.extractTime(args);
        const message = args.replace(/in \d+ \w+/i, '').trim();
        
        await this.speak(`Reminder set: "${message || 'Notification'}" in ${time}.`);
        
        // Actually set the reminder
        setTimeout(() => {
            this.showAlert(`Reminder: ${message}`, 'info');
            this.speak(`Sir, you asked me to remind you: ${message}`);
        }, this.parseTimeToMs(time));
    }
    
    async tellJoke() {
        const jokes = [
            "Why did the robot go on a diet? He had a byte problem.",
            "I would tell you a UDP joke, but you might not get it.",
            "There are only 10 types of people: those who understand binary and those who don't.",
            "Why do programmers prefer dark mode? Because light attracts bugs."
        ];
        
        const joke = jokes[Math.floor(Math.random() * jokes.length)];
        await this.speak(joke, { emotion: 'amused' });
    }
    
    // UI Helper methods
    showStatusOverlay() {
        const overlay = this.elements.overlay;
        const content = overlay.querySelector('.overlay-content');
        
        content.innerHTML = `
            <div class="status-display">
                <h3>SYSTEM STATUS</h3>
                <div class="status-grid">
                    <div class="status-item">
                        <label>Power</label>
                        <div class="status-bar">
                            <div class="status-fill" style="width: ${this.state.systemStatus.power}%"></div>
                        </div>
                        <span>${this.state.systemStatus.power}%</span>
                    </div>
                    <div class="status-item">
                        <label>Armor</label>
                        <div class="status-bar">
                            <div class="status-fill" style="width: ${this.state.systemStatus.armor}%"></div>
                        </div>
                        <span>${this.state.systemStatus.armor}%</span>
                    </div>
                    <div class="status-item">
                        <label>Temperature</label>
                        <span>${this.state.systemStatus.temperature}°C</span>
                    </div>
                    <div class="status-item">
                        <label>Shields</label>
                        <span class="${this.state.systemStatus.shields ? 'active' : 'inactive'}">
                            ${this.state.systemStatus.shields ? 'ACTIVE' : 'STANDBY'}
                        </span>
                    </div>
                </div>
            </div>
        `;
        
        overlay.style.display = 'block';
        setTimeout(() => {
            overlay.style.display = 'none';
        }, 5000);
    }
    
    showScanningOverlay() {
        const overlay = this.elements.overlay;
        const content = overlay.querySelector('.overlay-content');
        
        content.innerHTML = `
            <div class="scanning-display">
                <div class="scan-grid">
                    <div class="scan-line"></div>
                    <div class="scan-targets"></div>
                </div>
                <div class="scan-data">
                    <div>SCANNING...</div>
                    <div class="scan-progress"></div>
                </div>
            </div>
        `;
        
        overlay.style.display = 'block';
        
        // Animate scan
        let progress = 0;
        const interval = setInterval(() => {
            progress += 5;
            content.querySelector('.scan-progress').style.width = `${progress}%`;
            
            if (progress >= 100) {
                clearInterval(interval);
                setTimeout(() => {
                    overlay.style.display = 'none';
                }, 500);
            }
        }, 100);
    }
    
    showTargetingOverlay() {
        if (window.HUD) {
            window.HUD.enableTargeting();
        }
    }
    
    showAlert(message, type = 'info') {
        const alert = document.createElement('div');
        alert.className = `jarvis-alert ${type}`;
        alert.innerHTML = `
            <div class="alert-icon">${this.getAlertIcon(type)}</div>
            <div class="alert-message">${message}</div>
        `;
        
        this.elements.alerts.appendChild(alert);
        
        // Auto remove
        setTimeout(() => {
            alert.classList.add('fade-out');
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    }
    
    getAlertIcon(type) {
        const icons = {
            'info': 'ℹ',
            'warning': '⚠',
            'danger': '☢',
            'success': '✓'
        };
        return icons[type] || icons.info;
    }
    
    async displaySystemMessage(message) {
        const entry = document.createElement('div');
        entry.className = 'system-message';
        entry.textContent = `> ${message}`;
        this.elements.conversation.appendChild(entry);
        this.elements.conversation.scrollTop = this.elements.conversation.scrollHeight;
    }
    
    updateSuggestions(input) {
        if (!input) {
            this.elements.suggestions.innerHTML = '';
            return;
        }
        
        const matches = Object.keys(this.commands).filter(cmd => 
            cmd.startsWith(input.toLowerCase())
        );
        
        this.elements.suggestions.innerHTML = matches.slice(0, 5).map(cmd => `
            <div class="suggestion" data-command="${cmd}">
                <span class="command">${cmd}</span>
                <span class="description">${this.commands[cmd].description}</span>
            </div>
        `).join('');
        
        // Add click handlers
        this.elements.suggestions.querySelectorAll('.suggestion').forEach(sug => {
            sug.addEventListener('click', () => {
                this.elements.input.value = sug.dataset.command;
                this.elements.suggestions.innerHTML = '';
                this.elements.input.focus();
            });
        });
    }
    
    // Voice recognition
    toggleVoiceRecognition() {
        if (!this.features.voiceRecognition) {
            this.speak("Voice recognition is not available in your browser.");
            return;
        }
        
        if (this.recognition && this.recognition.running) {
            this.stopVoiceRecognition();
        } else {
            this.startVoiceRecognition();
        }
    }
    
    startVoiceRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';
        
        this.recognition.onstart = () => {
            this.setState('listening');
            this.elements.voiceBtn.classList.add('active');
        };
        
        this.recognition.onresult = (event) => {
            const last = event.results.length - 1;
            const transcript = event.results[last][0].transcript;
            
            if (event.results[last].isFinal) {
                this.processCommand(transcript);
            } else {
                this.elements.input.value = transcript;
            }
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.stopVoiceRecognition();
        };
        
        this.recognition.onend = () => {
            this.stopVoiceRecognition();
        };
        
        this.recognition.start();
        this.recognition.running = true;
    }
    
    stopVoiceRecognition() {
        if (this.recognition) {
            this.recognition.stop();
            this.recognition.running = false;
        }
        this.setState('active');
        this.elements.voiceBtn.classList.remove('active');
    }
    
    // Speech synthesis initialization
    initSpeechSynthesis() {
        if ('speechSynthesis' in window) {
            this.features.speechSynthesis = true;
            
            // Wait for voices to load
            const loadVoices = () => {
                const voices = speechSynthesis.getVoices();
                
                // Prefer British male voice for JARVIS
                this.jarvisVoice = voices.find(voice => 
                    voice.lang.includes('en-GB') && voice.name.toLowerCase().includes('male')
                ) || voices.find(voice => 
                    voice.lang.includes('en') && voice.name.toLowerCase().includes('male')
                ) || voices[0];
            };
            
            loadVoices();
            speechSynthesis.onvoiceschanged = loadVoices;
        }
    }
    
    checkFeatureSupport() {
        // Voice recognition
        this.features.voiceRecognition = 'SpeechRecognition' in window || 
                                         'webkitSpeechRecognition' in window;
        
        // Update UI based on features
        if (!this.features.voiceRecognition) {
            this.elements.voiceBtn.style.display = 'none';
        }
    }
    
    // Utility methods
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    extractTime(text) {
        const match = text.match(/in (\d+) (\w+)/i);
        if (match) {
            return `${match[1]} ${match[2]}`;
        }
        return '5 minutes';
    }
    
    parseTimeToMs(timeStr) {
        const [amount, unit] = timeStr.split(' ');
        const multipliers = {
            'second': 1000,
            'seconds': 1000,
            'minute': 60000,
            'minutes': 60000,
            'hour': 3600000,
            'hours': 3600000
        };
        return parseInt(amount) * (multipliers[unit] || 60000);
    }
    
    handleControlAction(action) {
        switch (action) {
            case 'minimize':
                this.elements.interface.classList.toggle('minimized');
                break;
            case 'maximize':
                this.elements.interface.classList.toggle('maximized');
                break;
            case 'settings':
                this.showSettings();
                break;
        }
    }
    
    showSettings() {
        // Settings panel implementation
        this.speak("Settings panel not yet implemented, sir.");
    }
    
    // Combat and emergency functions
    async executeCombatManeuver(type) {
        this.state.context = 'combat';
        
        switch (type) {
            case 'evasive':
                await this.speak("Executing evasive maneuvers!");
                if (window.HUD) {
                    window.HUD.activateEvasiveMode();
                }
                break;
            case 'defensive':
                await this.speak("Defensive stance activated. Shields at maximum.");
                this.state.systemStatus.shields = true;
                break;
            case 'offensive':
                await this.speak("Switching to offensive mode. All weapons hot.");
                this.state.systemStatus.weapons = 'ready';
                break;
        }
    }
    
    async activateEmergencyProtocol() {
        this.setState('alert');
        await this.speak("EMERGENCY PROTOCOL ACTIVATED!", { priority: 'urgent', emotion: 'concerned' });
        
        // Emergency actions
        this.state.systemStatus.shields = true;
        this.state.systemStatus.weapons = 'ready';
        
        if (window.HUD) {
            window.HUD.setMode('emergency');
        }
        
        this.showAlert('EMERGENCY MODE ACTIVE', 'danger');
    }
    
    fireRepulsors() {
        // Trigger repulsor effect
        if (window.HUD) {
            window.HUD.triggerRepulsor();
        }
        
        // Sound effect
        const audio = document.getElementById('repulsorSound');
        if (audio) {
            audio.play().catch(() => {});
        }
    }
}

// Initialize and export
window.JARVISAdvanced = JARVISAdvanced;

// Auto-initialize when document is ready
document.addEventListener('DOMContentLoaded', async () => {
    window.JARVIS = new JARVISAdvanced();
    await window.JARVIS.initialize();
});