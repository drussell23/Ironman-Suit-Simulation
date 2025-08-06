// Boot Sequence for Iron Man Experience

class BootSequence {
    constructor() {
        this.bootElement = Utils.$('#bootSequence');
        this.progressBar = Utils.$('.progress-fill');
        this.progressText = Utils.$('.progress-text');
        this.bootStatus = Utils.$('.boot-status');
        this.bootLog = Utils.$('.boot-log');
        this.bootSound = Utils.$('#bootSound');
        
        this.bootSteps = [
            { text: 'Initializing Arc Reactor...', duration: 1000, progress: 10 },
            { text: 'Loading J.A.R.V.I.S. Core...', duration: 1500, progress: 25 },
            { text: 'Calibrating Repulsor Arrays...', duration: 1200, progress: 40 },
            { text: 'Establishing Neural Link...', duration: 1800, progress: 55 },
            { text: 'Scanning Biometrics...', duration: 1000, progress: 65 },
            { text: 'Loading Weapon Systems...', duration: 1300, progress: 75 },
            { text: 'Activating HUD Interface...', duration: 1100, progress: 85 },
            { text: 'Running System Diagnostics...', duration: 1500, progress: 95 },
            { text: 'SYSTEMS ONLINE', duration: 1000, progress: 100 }
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
        // Update status
        this.bootStatus.textContent = step.text;
        
        // Add to log
        this.addLogEntry(step.text);
        
        // Update progress
        await this.updateProgress(step.progress);
        
        // Wait for step duration
        await Utils.delay(step.duration);
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
    
    addLogEntry(text) {
        const entry = Utils.createElement('p', { textContent: `> ${text}` });
        this.bootLog.appendChild(entry);
        
        // Auto scroll to bottom
        this.bootLog.scrollTop = this.bootLog.scrollHeight;
        
        // Limit log entries
        if (this.bootLog.children.length > 10) {
            this.bootLog.removeChild(this.bootLog.firstChild);
        }
    }
    
    async complete() {
        // Add final log entries
        this.addLogEntry('Boot sequence complete.');
        this.addLogEntry('Welcome back, sir.');
        
        await Utils.delay(1000);
        
        // Fade out boot sequence
        this.bootElement.style.opacity = '0';
        
        await Utils.delay(500);
        
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