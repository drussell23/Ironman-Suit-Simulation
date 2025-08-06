// Boot Sequence for Iron Man Experience

class BootSequence {
    constructor() {
        this.bootElement = Utils.$('#bootSequence');
        this.progressBar = Utils.$('.progress-fill');
        this.progressText = Utils.$('.progress-text');
        this.bootStatus = Utils.$('.boot-status');
        this.bootLog = Utils.$('.boot-log');
        this.bootSound = Utils.$('#bootSound');
        this.particles = null;
        
        this.bootSteps = [
            { text: 'Initializing Arc Reactor Core...', duration: 1200, progress: 8, subtext: 'Power output: 3.0 GJ/s' },
            { text: 'Establishing quantum entanglement...', duration: 800, progress: 15, subtext: 'Quantum state: Coherent' },
            { text: 'Loading J.A.R.V.I.S. Neural Matrix...', duration: 1500, progress: 23, subtext: 'AI cores: 8/8 active' },
            { text: 'Calibrating Repulsor Arrays...', duration: 1000, progress: 30, subtext: 'Alignment: 99.97%' },
            { text: 'Initializing flight systems...', duration: 900, progress: 38, subtext: 'Thrust vectoring: Online' },
            { text: 'Establishing Neural Interface...', duration: 1400, progress: 45, subtext: 'Synaptic link: Established' },
            { text: 'Scanning biometric signatures...', duration: 1100, progress: 52, subtext: 'User: Tony Stark - Verified' },
            { text: 'Loading tactical subsystems...', duration: 1000, progress: 60, subtext: 'Combat protocols: Loaded' },
            { text: 'Initializing weapons platform...', duration: 1200, progress: 68, subtext: 'Armament: Ready' },
            { text: 'Activating heads-up display...', duration: 900, progress: 75, subtext: 'HUD refresh rate: 120Hz' },
            { text: 'Synchronizing satellite uplink...', duration: 1100, progress: 82, subtext: 'Connection: Secured' },
            { text: 'Running system diagnostics...', duration: 1300, progress: 90, subtext: 'All systems: Nominal' },
            { text: 'Finalizing boot sequence...', duration: 800, progress: 96, subtext: 'Optimization: Complete' },
            { text: 'MARK 85 SYSTEMS ONLINE', duration: 1500, progress: 100, subtext: 'Welcome back, sir' }
        ];
        
        this.currentStep = 0;
        this.onComplete = null;
    }
    
    async start() {
        // Play boot sound
        if (this.bootSound) {
            this.bootSound.play().catch(() => {
                console.log('Audio autoplay blocked. User interaction required.');
            });
        }
        
        // Show boot sequence
        this.bootElement.classList.add('active');
        
        // Initialize particle effects
        if (window.BootParticles) {
            this.particles = new BootParticles();
            this.particles.init();
        }
        
        // Start boot process
        await this.runBootSteps();
        
        // Complete boot
        await Utils.delay(1000);
        await this.complete();
    }
    
    async runBootSteps() {
        for (const step of this.bootSteps) {
            await this.executeStep(step);
        }
    }
    
    async executeStep(step) {
        // Update status with typing effect
        await this.typeText(this.bootStatus, step.text);
        
        // Add to log with subtext
        this.addLogEntry(step.text, step.subtext);
        
        // Update progress with smooth animation
        await this.updateProgress(step.progress);
        
        // Add random system messages
        if (Math.random() > 0.7) {
            this.addSystemMessage();
        }
        
        // Wait for step duration
        await Utils.delay(step.duration);
    }
    
    async typeText(element, text) {
        element.textContent = '';
        for (let i = 0; i < text.length; i++) {
            element.textContent += text[i];
            await Utils.delay(20);
        }
    }
    
    async updateProgress(targetProgress) {
        const currentProgress = parseFloat(this.progressBar.style.width) || 0;
        const steps = 20;
        const increment = (targetProgress - currentProgress) / steps;
        
        for (let i = 0; i < steps; i++) {
            const progress = currentProgress + (increment * (i + 1));
            this.progressBar.style.width = `${progress}%`;
            this.progressText.textContent = `${Math.floor(progress)}%`;
            await Utils.delay(30);
        }
    }
    
    addLogEntry(text, subtext) {
        const timestamp = new Date().toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit' 
        });
        
        const entry = Utils.createElement('p');
        entry.innerHTML = `<span style="color: #666;">[${timestamp}]</span> ${text}`;
        if (subtext) {
            entry.innerHTML += `<span style="color: #00a8ff; margin-left: 10px; font-size: 0.85em;">${subtext}</span>`;
        }
        
        this.bootLog.appendChild(entry);
        
        // Auto scroll to bottom
        this.bootLog.scrollTop = this.bootLog.scrollHeight;
        
        // Limit log entries
        if (this.bootLog.children.length > 12) {
            this.bootLog.removeChild(this.bootLog.firstChild);
        }
    }
    
    addSystemMessage() {
        const messages = [
            'Memory allocation optimized',
            'Quantum processors synchronized',
            'Nanoparticle assembly verified',
            'Energy distribution balanced',
            'Threat detection online',
            'Environmental sensors calibrated',
            'Backup systems ready',
            'Emergency protocols loaded'
        ];
        
        const message = messages[Math.floor(Math.random() * messages.length)];
        const entry = Utils.createElement('p');
        entry.innerHTML = `<span style="color: #666;">[SYSTEM]</span> <span style="color: #888; font-style: italic;">${message}</span>`;
        entry.style.opacity = '0.7';
        
        this.bootLog.appendChild(entry);
        this.bootLog.scrollTop = this.bootLog.scrollHeight;
    }
    
    async complete() {
        // Add final log entries
        this.addLogEntry('Boot sequence complete.', 'Duration: 14.2 seconds');
        await Utils.delay(500);
        this.addLogEntry('All systems operational.', 'Status: READY');
        await Utils.delay(500);
        
        // Add JARVIS greeting
        const jarvisEntry = Utils.createElement('p');
        jarvisEntry.innerHTML = `<span style="color: #00d4ff; font-weight: bold;">[J.A.R.V.I.S.]</span> Good evening, sir. All systems are operating at peak efficiency.`;
        jarvisEntry.style.fontSize = '1.1em';
        this.bootLog.appendChild(jarvisEntry);
        
        await Utils.delay(1500);
        
        // Add glitch effect before fade
        this.bootElement.classList.add('glitch');
        await Utils.delay(300);
        this.bootElement.classList.remove('glitch');
        
        // Fade out boot sequence with style
        this.bootElement.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
        this.bootElement.style.opacity = '0';
        this.bootElement.style.transform = 'scale(0.95)';
        this.bootElement.style.filter = 'blur(10px)';
        
        await Utils.delay(800);
        
        // Stop particle effects
        if (this.particles) {
            this.particles.stop();
        }
        
        // Hide boot sequence
        this.bootElement.classList.remove('active');
        this.bootElement.style.display = 'none';
        
        // Trigger completion callback
        if (this.onComplete) {
            this.onComplete();
        }
    }
    
    // Quick boot for development
    quickBoot() {
        this.progressBar.style.width = '100%';
        this.progressText.textContent = '100%';
        this.bootStatus.textContent = 'SYSTEMS ONLINE';
        this.complete();
    }
}

// Make BootSequence globally available
window.BootSequence = BootSequence;