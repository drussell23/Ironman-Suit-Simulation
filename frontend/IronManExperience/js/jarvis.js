// JARVIS System for Iron Man Experience

class JARVIS {
    constructor() {
        this.interface = Utils.$('.jarvis-interface');
        this.avatar = Utils.$('.jarvis-avatar');
        this.textDisplay = Utils.$('.jarvis-text');
        this.voiceInput = Utils.$('#voiceCommand');
        this.submitButton = Utils.$('#commandSubmit');
        this.jarvisSound = Utils.$('#jarvisSound');

        this.isListening = false;
        this.isSpeaking = false;
        this.commandHistory = [];
        this.maxHistory = 50;

        // Command handlers
        this.commands = {
            'status': () => this.handleStatusCommand(),
            'scan': () => this.handleScanCommand(),
            'weapons': () => this.handleWeaponsCommand(),
            'power': () => this.handlePowerCommand(),
            'mission': () => this.handleMissionCommand(),
            'analysis': () => this.handleAnalysisCommand(),
            'emergency': () => this.handleEmergencyCommand(),
            'help': () => this.handleHelpCommand(),
            'hello': () => this.speak("Hello, sir. How may I assist you?"),
            'time': () => this.speak(`The current time is ${new Date().toLocaleTimeString()}`),
            'date': () => this.speak(`Today is ${new Date().toLocaleDateString()}`),
            'clear': () => this.clearDisplay()
        };

        // Response templates
        this.responses = {
            greeting: [
                "Good evening, sir.",
                "Welcome back, sir.",
                "All systems are at your command, sir."
            ],
            confirmation: [
                "Right away, sir.",
                "Understood.",
                "Processing your request.",
                "Executing command."
            ],
            error: [
                "I'm sorry, I didn't understand that command.",
                "Could you please repeat that, sir?",
                "Command not recognized. Try saying 'help' for available commands."
            ],
            analysis: [
                "Scanning environment...",
                "Analysis in progress...",
                "Running diagnostics..."
            ]
        };

        this.init();
    }

    init() {
        // Set up event listeners
        this.submitButton.addEventListener('click', () => this.processCommand());
        this.voiceInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processCommand();
            }
        });

        // Voice command history navigation
        let historyIndex = -1;
        this.voiceInput.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowUp' && this.commandHistory.length > 0) {
                e.preventDefault();
                historyIndex = Math.min(historyIndex + 1, this.commandHistory.length - 1);
                this.voiceInput.value = this.commandHistory[historyIndex] || '';
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                historyIndex = Math.max(historyIndex - 1, -1);
                this.voiceInput.value = historyIndex >= 0 ? this.commandHistory[historyIndex] : '';
            }
        });

        // Initialize with greeting
        setTimeout(() => {
            this.speak(Utils.randomFrom(this.responses.greeting));
            if (this.jarvisSound) {
                this.jarvisSound.play().catch(() => { });
            }
        }, 1000);
    }

    processCommand() {
        const input = this.voiceInput.value.trim();
        if (!input) return;

        // Add to history
        this.commandHistory.unshift(input);
        if (this.commandHistory.length > this.maxHistory) {
            this.commandHistory.pop();
        }

        // Clear input
        this.voiceInput.value = '';

        // Process command
        const command = input.toLowerCase();
        let handled = false;

        // Check exact matches first
        if (this.commands[command]) {
            this.commands[command]();
            handled = true;
        } else {
            // Check partial matches
            for (const [key, handler] of Object.entries(this.commands)) {
                if (command.includes(key)) {
                    handler();
                    handled = true;
                    break;
                }
            }
        }

        // If no command matched, try to interpret
        if (!handled) {
            this.interpretCommand(command);
        }
    }

    interpretCommand(command) {
        // Natural language processing (simplified)
        if (command.includes('fire') || command.includes('shoot')) {
            this.speak("Weapons require manual authorization, sir.");
            window.HUD?.triggerRepulsor();
        } else if (command.includes('fly') || command.includes('altitude')) {
            this.speak("Flight systems engaged.");
            window.HUD?.updateAltitude(1000);
        } else if (command.includes('target') || command.includes('lock')) {
            this.speak("Scanning for targets...");
            window.HUD?.enableTargeting();
        } else if (command.includes('shield')) {
            this.speak("Shields at maximum power.");
            window.HUD?.activateShields();
        } else if (command.includes('music')) {
            this.speak("Would you like me to play your favorite AC/DC tracks, sir?");
        } else {
            this.speak(Utils.randomFrom(this.responses.error));
        }
    }

    async speak(text, priority = 'normal') {
        if (this.isSpeaking && priority !== 'urgent') return;

        this.isSpeaking = true;
        this.interface.classList.add('speaking');

        // Type out the text
        await Utils.typeWriter(this.textDisplay, text, 30);

        // Use Web Speech API if available
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 0.8;
            utterance.volume = 0.8;

            // Try to use a British male voice
            const voices = speechSynthesis.getVoices();
            const jarvisVoice = voices.find(voice =>
                voice.name.includes('British') && voice.name.includes('Male')
            ) || voices.find(voice =>
                voice.name.includes('UK') && voice.name.includes('Male')
            );

            if (jarvisVoice) {
                utterance.voice = jarvisVoice;
            }

            speechSynthesis.speak(utterance);
        }

        // Keep speaking animation for a bit
        await Utils.delay(1000);

        this.isSpeaking = false;
        this.interface.classList.remove('speaking');
    }

    // Command Handlers
    handleStatusCommand() {
        const status = window.HUD?.getSystemStatus() || {
            power: 98,
            armor: 100,
            temperature: 36.5,
            altitude: 0,
            velocity: 0
        };

        this.speak(`System status: Power at ${status.power}%, armor integrity ${status.armor}%, ` +
            `core temperature ${status.temperature} degrees Celsius. All systems nominal.`);
    }

    handleScanCommand() {
        this.speak("Initiating environmental scan...");
        window.HUD?.startAnalysisMode();

        setTimeout(() => {
            this.speak("Scan complete. No immediate threats detected. " +
                "Three heat signatures identified, structural integrity stable.");
        }, 3000);
    }

    handleWeaponsCommand() {
        this.speak("Weapons systems online. Repulsors charged and ready.");
        window.HUD?.showWeaponStatus();
    }

    handlePowerCommand() {
        this.speak("Diverting power to primary systems. Arc reactor output increased to 120%.");
        window.HUD?.boostPower();
    }

    handleMissionCommand() {
        this.speak("Accessing mission database. Please select your mission parameters.");
        window.Missions?.showMissionSelect();
    }

    handleAnalysisCommand() {
        this.speak("Switching to analysis mode.");
        window.HUD?.setMode('analysis');
    }

    handleEmergencyCommand() {
        this.speak("EMERGENCY PROTOCOL ACTIVATED!", 'urgent');
        window.HUD?.setMode('emergency');
    }

    handleHelpCommand() {
        this.speak("Available commands: status, scan, weapons, power, mission, analysis, " +
            "emergency, time, date, hello, clear. You can also use natural language.");
    }

    clearDisplay() {
        this.textDisplay.textContent = '';
        this.speak("Display cleared.");
    }

    // Voice Recognition (if supported)
    startVoiceRecognition() {
        if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
            this.speak("Voice recognition is not supported in your browser.");
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();

        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            this.isListening = true;
            this.voiceInput.placeholder = "Listening...";
            this.voiceInput.style.borderColor = 'var(--success-color)';
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.voiceInput.value = transcript;
            this.processCommand();
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.speak("I didn't catch that, sir. Please try again.");
        };

        recognition.onend = () => {
            this.isListening = false;
            this.voiceInput.placeholder = "Voice command...";
            this.voiceInput.style.borderColor = '';
        };

        recognition.start();
    }

    // Minimize/Maximize interface
    toggleMinimize() {
        this.interface.classList.toggle('minimized');
    }
}

// Make JARVIS globally available
window.JARVIS = JARVIS;