// HUD System for Iron Man Experience

class HUD {
    constructor() {
        this.interface = Utils.$('.hud-interface');
        this.hudSound = Utils.$('#hudSound');
        
        // Status elements
        this.powerBar = Utils.$('.power-fill');
        this.powerValue = Utils.$('.status-item .value');
        this.armorBar = Utils.$('.armor-fill');
        this.tempBar = Utils.$('.temp-fill');
        
        // Navigation elements
        this.compass = Utils.$('.compass-ring');
        this.headingText = Utils.$('.heading');
        this.altitudeValue = Utils.$('.altitude-meter .value');
        this.velocityValue = Utils.$('.velocity-meter .value');
        
        // Weapon elements
        this.weaponItems = Utils.$$('.weapon-item');
        this.repulsorSound = Utils.$('#repulsorSound');
        
        // Target elements
        this.targetInfo = Utils.$('.target-info');
        this.noTarget = Utils.$('.no-target');
        this.targetDetails = Utils.$('.target-details');
        
        // Alert elements
        this.alertPanel = Utils.$('.alert-panel');
        this.alertSound = Utils.$('#alertSound');
        
        // Mode indicator
        this.modeIndicator = Utils.$('.mode-indicator');
        
        // State
        this.currentMode = 'STANDARD';
        this.isTargeting = false;
        this.targets = [];
        this.currentHeading = 0;
        this.altitude = 0;
        this.velocity = 0;
        
        // System status
        this.systemStatus = {
            power: 98,
            armor: 100,
            temperature: 36.5,
            shields: true,
            weapons: true
        };
        
        this.init();
    }
    
    init() {
        // Activate HUD
        this.interface.classList.add('active');
        
        // Play HUD activation sound
        if (this.hudSound) {
            this.hudSound.play().catch(() => {});
        }
        
        // Initialize animations
        this.startAnimations();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Start system monitoring
        this.startSystemMonitoring();
    }
    
    startAnimations() {
        // Animate HUD elements sliding in
        Utils.$('.hud-top').classList.add('slide-in-top');
        Utils.$('.hud-left').classList.add('slide-in-left');
        Utils.$('.hud-right').classList.add('slide-in-right');
        Utils.$('.hud-bottom').classList.add('slide-in-bottom');
        
        // Start compass rotation
        this.updateCompass();
        
        // Start horizon indicator
        this.updateHorizon();
    }
    
    setupEventListeners() {
        // Weapon selection
        this.weaponItems.forEach((item, index) => {
            item.addEventListener('click', () => this.selectWeapon(index));
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyPress(e));
    }
    
    handleKeyPress(e) {
        switch(e.key) {
            case '1':
            case '2':
            case '3':
                this.selectWeapon(parseInt(e.key) - 1);
                break;
            case ' ':
                e.preventDefault();
                this.triggerRepulsor();
                break;
            case 'Tab':
                e.preventDefault();
                this.toggleTargeting();
                break;
            case 'm':
                this.cycleMode();
                break;
        }
    }
    
    // System Monitoring
    startSystemMonitoring() {
        setInterval(() => {
            // Simulate system changes
            this.updatePower(this.systemStatus.power - Math.random() * 0.1);
            this.updateTemperature(36.5 + (Math.random() - 0.5) * 0.2);
            
            // Update navigation
            this.updateHeading(this.currentHeading + (Math.random() - 0.5) * 2);
            
            // Check for warnings
            this.checkSystemWarnings();
        }, 1000);
    }
    
    checkSystemWarnings() {
        if (this.systemStatus.power < 20) {
            this.showAlert('LOW POWER WARNING', 'warning');
        }
        
        if (this.systemStatus.temperature > 38) {
            this.showAlert('TEMPERATURE WARNING', 'warning');
        }
        
        if (this.altitude > 10000) {
            this.showAlert('HIGH ALTITUDE WARNING', 'info');
        }
    }
    
    // Status Updates
    updatePower(value) {
        this.systemStatus.power = Utils.clamp(value, 0, 100);
        this.powerBar.style.width = `${this.systemStatus.power}%`;
        
        const powerElement = Utils.$('.status-item:nth-child(1) .value');
        if (powerElement) {
            powerElement.textContent = `${Math.floor(this.systemStatus.power)}%`;
            
            // Update color based on power level
            if (this.systemStatus.power < 20) {
                this.powerBar.style.background = 'linear-gradient(90deg, var(--danger-color), var(--warning-color))';
            } else if (this.systemStatus.power < 50) {
                this.powerBar.style.background = 'linear-gradient(90deg, var(--warning-color), var(--primary-color))';
            }
        }
    }
    
    updateArmor(value) {
        this.systemStatus.armor = Utils.clamp(value, 0, 100);
        this.armorBar.style.width = `${this.systemStatus.armor}%`;
        
        const armorElement = Utils.$('.status-item:nth-child(2) .value');
        if (armorElement) {
            armorElement.textContent = `${Math.floor(this.systemStatus.armor)}%`;
        }
    }
    
    updateTemperature(value) {
        this.systemStatus.temperature = value;
        const tempPercent = Utils.map(value, 35, 40, 0, 100);
        this.tempBar.style.width = `${tempPercent}%`;
        
        const tempElement = Utils.$('.status-item:nth-child(3) .value');
        if (tempElement) {
            tempElement.textContent = `${value.toFixed(1)}°C`;
        }
    }
    
    // Navigation Updates
    updateCompass() {
        setInterval(() => {
            this.compass.style.transform = `rotateZ(${-this.currentHeading}deg)`;
        }, 50);
    }
    
    updateHeading(degrees) {
        this.currentHeading = (degrees + 360) % 360;
        this.headingText.textContent = `${Math.floor(this.currentHeading)}°`;
    }
    
    updateAltitude(meters) {
        this.altitude = Math.max(0, meters);
        this.altitudeValue.textContent = Math.floor(this.altitude);
        
        // Add altitude warning effect
        if (this.altitude > 10000) {
            this.altitudeValue.parentElement.classList.add('altitude-warning');
        } else {
            this.altitudeValue.parentElement.classList.remove('altitude-warning');
        }
    }
    
    updateVelocity(kmh) {
        this.velocity = Math.max(0, kmh);
        this.velocityValue.textContent = Math.floor(this.velocity);
        
        // Update motion blur effect based on speed
        if (this.velocity > 500) {
            document.body.style.filter = `blur(${Utils.map(this.velocity, 500, 1000, 0, 2)}px)`;
        } else {
            document.body.style.filter = 'none';
        }
    }
    
    updateHorizon() {
        const horizonLine = Utils.$('.horizon-line');
        if (!horizonLine) return;
        
        let pitch = 0;
        let roll = 0;
        
        setInterval(() => {
            // Simulate flight dynamics
            pitch += (Math.random() - 0.5) * 0.5;
            roll += (Math.random() - 0.5) * 0.3;
            
            pitch = Utils.clamp(pitch, -30, 30);
            roll = Utils.clamp(roll, -45, 45);
            
            horizonLine.style.transform = `translateY(${pitch * 2}px) rotate(${roll}deg)`;
        }, 100);
    }
    
    // Weapon Systems
    selectWeapon(index) {
        this.weaponItems.forEach((item, i) => {
            if (i === index) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        
        const weaponNames = ['REPULSORS', 'MISSILES', 'LASER'];
        window.JARVIS?.speak(`${weaponNames[index]} selected.`);
    }
    
    triggerRepulsor() {
        // Visual effect
        const reticle = Utils.$('.reticle');
        reticle.classList.add('repulsor-charging');
        
        // Play sound
        if (this.repulsorSound) {
            this.repulsorSound.currentTime = 0;
            this.repulsorSound.play().catch(() => {});
        }
        
        // Fire effect
        setTimeout(() => {
            reticle.classList.remove('repulsor-charging');
            reticle.classList.add('repulsor-fire');
            
            // Screen shake
            Utils.screenShake(5, 200);
            
            // Update power
            this.updatePower(this.systemStatus.power - 2);
            
            setTimeout(() => {
                reticle.classList.remove('repulsor-fire');
            }, 300);
        }, 500);
    }
    
    // Targeting System
    toggleTargeting() {
        this.isTargeting = !this.isTargeting;
        
        if (this.isTargeting) {
            this.enableTargeting();
        } else {
            this.disableTargeting();
        }
    }
    
    enableTargeting() {
        this.isTargeting = true;
        Utils.$('.reticle').classList.add('targeting-active');
        
        // Start scanning for targets
        this.scanForTargets();
        
        window.JARVIS?.speak("Targeting system engaged.");
    }
    
    disableTargeting() {
        this.isTargeting = false;
        Utils.$('.reticle').classList.remove('targeting-active');
        
        // Clear all target indicators
        this.clearTargets();
        
        window.JARVIS?.speak("Targeting system disengaged.");
    }
    
    scanForTargets() {
        if (!this.isTargeting) return;
        
        // Simulate finding targets
        const targetCount = Math.floor(Math.random() * 3) + 1;
        
        for (let i = 0; i < targetCount; i++) {
            this.addTarget({
                id: Utils.uuid(),
                distance: Math.floor(Math.random() * 500) + 100,
                type: Utils.randomFrom(['HOSTILE', 'UNKNOWN', 'CIVILIAN']),
                threat: Math.random()
            });
        }
        
        // Continue scanning
        setTimeout(() => this.scanForTargets(), 3000);
    }
    
    addTarget(targetData) {
        // Create target indicator
        const indicator = Utils.createElement('div', {
            className: 'target-indicator',
            style: `left: ${Math.random() * 80 + 10}%; top: ${Math.random() * 60 + 20}%;`
        });
        
        const distance = Utils.createElement('div', {
            className: 'distance',
            textContent: `${targetData.distance}m`
        });
        
        indicator.appendChild(distance);
        Utils.$('.hud-overlay').appendChild(indicator);
        
        // Add to targets array
        this.targets.push({
            data: targetData,
            element: indicator
        });
        
        // Auto-lock on closest hostile
        if (targetData.type === 'HOSTILE' && this.targets.length === 1) {
            this.lockTarget(targetData);
        }
    }
    
    lockTarget(targetData) {
        this.noTarget.style.display = 'none';
        this.targetDetails.style.display = 'block';
        
        Utils.$('.target-distance').textContent = `DISTANCE: ${targetData.distance}m`;
        Utils.$('.target-type').textContent = `TYPE: ${targetData.type}`;
        Utils.$('.target-threat').textContent = `THREAT: ${Math.floor(targetData.threat * 100)}%`;
        
        // Play lock sound
        if (this.alertSound) {
            this.alertSound.play().catch(() => {});
        }
        
        window.JARVIS?.speak("Target locked.");
    }
    
    clearTargets() {
        this.targets.forEach(target => {
            target.element.remove();
        });
        this.targets = [];
        
        this.noTarget.style.display = 'block';
        this.targetDetails.style.display = 'none';
    }
    
    // Alert System
    showAlert(message, type = 'info') {
        const alert = Utils.createElement('div', {
            className: `alert-message alert-${type}`,
            textContent: message
        });
        
        this.alertPanel.appendChild(alert);
        
        // Play alert sound for warnings
        if (type === 'warning' || type === 'danger') {
            if (this.alertSound) {
                this.alertSound.play().catch(() => {});
            }
        }
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 500);
        }, 5000);
    }
    
    // Mode Management
    setMode(mode) {
        this.currentMode = mode.toUpperCase();
        this.modeIndicator.textContent = `MODE: ${this.currentMode}`;
        
        // Apply mode-specific changes
        switch (this.currentMode) {
            case 'COMBAT':
                this.enableTargeting();
                this.selectWeapon(0);
                document.body.classList.add('combat-mode');
                break;
                
            case 'STEALTH':
                this.interface.style.opacity = '0.3';
                document.body.classList.add('stealth-mode');
                break;
                
            case 'ANALYSIS':
                this.startAnalysisMode();
                break;
                
            case 'EMERGENCY':
                this.startEmergencyMode();
                break;
                
            default:
                this.resetMode();
        }
    }
    
    cycleMode() {
        const modes = ['STANDARD', 'COMBAT', 'STEALTH', 'ANALYSIS'];
        const currentIndex = modes.indexOf(this.currentMode);
        const nextIndex = (currentIndex + 1) % modes.length;
        this.setMode(modes[nextIndex]);
    }
    
    resetMode() {
        this.interface.style.opacity = '1';
        document.body.className = '';
        this.disableTargeting();
    }
    
    startAnalysisMode() {
        const overlay = Utils.$('#analysisOverlay');
        overlay.style.display = 'block';
        
        // Animate scan line
        const scanLine = Utils.$('.scan-line');
        scanLine.style.animation = 'scan 3s linear';
        
        // Show scan results after delay
        setTimeout(() => {
            Utils.$('.scan-results').innerHTML = `
                <h3>SCAN COMPLETE</h3>
                <p>Threats: 0</p>
                <p>Structural Integrity: 100%</p>
                <p>Environmental Hazards: None</p>
                <p>Recommended Action: Continue</p>
            `;
            
            // Add data points
            this.addDataPoints();
        }, 3000);
        
        // Auto-exit after 10 seconds
        setTimeout(() => {
            overlay.style.display = 'none';
            this.setMode('STANDARD');
        }, 10000);
    }
    
    addDataPoints() {
        const container = Utils.$('.data-points');
        container.innerHTML = '';
        
        for (let i = 0; i < 5; i++) {
            const point = Utils.createElement('div', {
                className: 'data-point',
                'data-label': `Point ${i + 1}`,
                style: `left: ${Math.random() * 80 + 10}%; top: ${Math.random() * 60 + 20}%;`
            });
            container.appendChild(point);
        }
    }
    
    startEmergencyMode() {
        const alert = Utils.$('#emergencyAlert');
        alert.style.display = 'block';
        Utils.$('.alert-message').textContent = 'CRITICAL SYSTEM FAILURE';
        
        // Flash effect
        document.body.classList.add('emergency-strobe');
        
        // Alarm sound loop
        if (this.alertSound) {
            this.alertSound.loop = true;
            this.alertSound.play().catch(() => {});
        }
        
        // Exit emergency mode after 5 seconds
        setTimeout(() => {
            alert.style.display = 'none';
            document.body.classList.remove('emergency-strobe');
            this.alertSound.loop = false;
            this.alertSound.pause();
            this.setMode('STANDARD');
        }, 5000);
    }
    
    // Public Methods
    getSystemStatus() {
        return { ...this.systemStatus };
    }
    
    boostPower() {
        this.updatePower(Math.min(100, this.systemStatus.power + 20));
        this.showAlert('POWER BOOST ENGAGED', 'info');
    }
    
    activateShields() {
        this.systemStatus.shields = true;
        document.body.classList.add('shield-active');
        this.showAlert('SHIELDS ONLINE', 'info');
    }
    
    showWeaponStatus() {
        this.weaponItems.forEach(item => {
            item.classList.add('glitch');
            setTimeout(() => item.classList.remove('glitch'), 300);
        });
    }
}

// Make HUD globally available
window.HUD = HUD;