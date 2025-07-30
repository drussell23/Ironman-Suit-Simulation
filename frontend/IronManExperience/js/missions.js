// Mission System for Iron Man Experience

class MissionSystem {
    constructor() {
        this.currentMission = null;
        this.missionActive = false;
        this.missionStartTime = 0;
        this.objectives = [];
        this.score = 0;
        
        // Mission templates
        this.missionTemplates = {
            combat: {
                name: 'HOSTILE TAKEOVER',
                description: 'Eliminate hostile forces at Stark facility',
                difficulty: 3,
                timeLimit: 300, // 5 minutes
                objectives: [
                    { id: 'eliminate_all', description: 'Eliminate all hostiles', type: 'eliminate', target: 10, current: 0 },
                    { id: 'protect_core', description: 'Protect reactor core', type: 'protect', target: 100, current: 100 },
                    { id: 'time_limit', description: 'Complete within time limit', type: 'time', target: 300, current: 0 }
                ],
                rewards: { score: 1000, experience: 100 }
            },
            rescue: {
                name: 'DISASTER RELIEF',
                description: 'Rescue civilians from earthquake zone',
                difficulty: 2,
                timeLimit: 600, // 10 minutes
                objectives: [
                    { id: 'rescue_civilians', description: 'Rescue trapped civilians', type: 'rescue', target: 5, current: 0 },
                    { id: 'medical_supply', description: 'Deliver medical supplies', type: 'deliver', target: 3, current: 0 },
                    { id: 'no_casualties', description: 'Prevent casualties', type: 'protect', target: 100, current: 100 }
                ],
                rewards: { score: 800, experience: 80 }
            },
            recon: {
                name: 'STEALTH RECON',
                description: 'Infiltrate enemy base undetected',
                difficulty: 4,
                timeLimit: 480, // 8 minutes
                objectives: [
                    { id: 'remain_undetected', description: 'Remain undetected', type: 'stealth', target: 100, current: 100 },
                    { id: 'scan_targets', description: 'Scan key targets', type: 'scan', target: 5, current: 0 },
                    { id: 'extract_data', description: 'Extract intelligence data', type: 'collect', target: 3, current: 0 }
                ],
                rewards: { score: 1200, experience: 120 }
            }
        };
        
        // UI elements
        this.missionSelectUI = Utils.$('#missionSelect');
        this.missionItems = Utils.$$('.mission-item');
        this.closeButton = Utils.$('.close-missions');
        this.objectivesList = Utils.$('.objectives-list');
        
        this.init();
    }
    
    init() {
        // Set up event listeners
        this.missionItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const missionType = item.dataset.mission;
                this.selectMission(missionType);
            });
            
            // Add hover sound
            item.addEventListener('mouseenter', () => {
                window.AudioSystem?.playUISound('hover');
            });
        });
        
        if (this.closeButton) {
            this.closeButton.addEventListener('click', () => {
                this.hideMissionSelect();
                window.AudioSystem?.playUISound('click');
            });
        }
    }
    
    showMissionSelect() {
        this.missionSelectUI.style.display = 'block';
        this.missionSelectUI.classList.add('fade-in');
        
        // Pause game time
        if (window.Viewport) {
            window.Viewport.paused = true;
        }
        
        window.AudioSystem?.playUISound('success');
    }
    
    hideMissionSelect() {
        this.missionSelectUI.classList.add('fade-out');
        setTimeout(() => {
            this.missionSelectUI.style.display = 'none';
            this.missionSelectUI.classList.remove('fade-in', 'fade-out');
        }, 500);
        
        // Resume game time
        if (window.Viewport) {
            window.Viewport.paused = false;
        }
    }
    
    selectMission(missionType) {
        const template = this.missionTemplates[missionType];
        if (!template) return;
        
        // Create mission instance
        this.currentMission = {
            ...template,
            type: missionType,
            objectives: template.objectives.map(obj => ({ ...obj })), // Deep copy objectives
            startTime: Date.now(),
            status: 'briefing'
        };
        
        // Show mission briefing
        this.showMissionBriefing();
        
        window.AudioSystem?.playUISound('click');
    }
    
    showMissionBriefing() {
        // Hide mission select
        this.hideMissionSelect();
        
        // Create briefing UI
        const briefing = Utils.createElement('div', {
            className: 'mission-briefing',
            innerHTML: `
                <h2>MISSION BRIEFING</h2>
                <h3>${this.currentMission.name}</h3>
                <p>${this.currentMission.description}</p>
                <div class="briefing-objectives">
                    <h4>OBJECTIVES:</h4>
                    <ul>
                        ${this.currentMission.objectives.map(obj => 
                            `<li>${obj.description}</li>`
                        ).join('')}
                    </ul>
                </div>
                <div class="briefing-info">
                    <p>Time Limit: ${Utils.formatTime(this.currentMission.timeLimit)}</p>
                    <p>Difficulty: ${'★'.repeat(this.currentMission.difficulty)}${'☆'.repeat(5 - this.currentMission.difficulty)}</p>
                </div>
                <div class="briefing-actions">
                    <button class="start-mission">START MISSION</button>
                    <button class="cancel-mission">CANCEL</button>
                </div>
            `
        });
        
        document.body.appendChild(briefing);
        
        // Add event listeners
        briefing.querySelector('.start-mission').addEventListener('click', () => {
            this.startMission();
            briefing.remove();
        });
        
        briefing.querySelector('.cancel-mission').addEventListener('click', () => {
            this.currentMission = null;
            briefing.remove();
            this.showMissionSelect();
        });
        
        // JARVIS briefing
        window.JARVIS?.speak(`Mission briefing: ${this.currentMission.name}. ${this.currentMission.description}`);
    }
    
    startMission() {
        if (!this.currentMission) return;
        
        this.missionActive = true;
        this.missionStartTime = Date.now();
        this.currentMission.status = 'active';
        
        // Update HUD
        window.HUD?.setMode('MISSION');
        this.updateObjectivesDisplay();
        
        // Start mission timer
        this.startMissionTimer();
        
        // Spawn mission elements
        this.spawnMissionElements();
        
        // JARVIS announcement
        window.JARVIS?.speak('Mission commenced. Good luck, sir.', 'urgent');
        window.AudioSystem?.play('powerUp');
    }
    
    updateObjectivesDisplay() {
        if (!this.objectivesList) return;
        
        this.objectivesList.innerHTML = '';
        
        this.currentMission.objectives.forEach(obj => {
            const item = Utils.createElement('div', {
                className: `objective-item ${obj.current >= obj.target ? 'completed' : ''}`,
                innerHTML: `
                    <div class="checkbox"></div>
                    <span>${obj.description}</span>
                    ${obj.type !== 'protect' && obj.type !== 'stealth' ? 
                        `<span class="progress">${obj.current}/${obj.target}</span>` : 
                        `<span class="progress">${Math.floor(obj.current)}%</span>`
                    }
                `
            });
            
            this.objectivesList.appendChild(item);
        });
    }
    
    startMissionTimer() {
        const updateTimer = () => {
            if (!this.missionActive) return;
            
            const elapsed = (Date.now() - this.missionStartTime) / 1000;
            const remaining = this.currentMission.timeLimit - elapsed;
            
            // Update time objective
            const timeObj = this.currentMission.objectives.find(o => o.type === 'time');
            if (timeObj) {
                timeObj.current = elapsed;
            }
            
            // Check time limit
            if (remaining <= 0) {
                this.failMission('Time limit exceeded');
                return;
            }
            
            // Warning at 1 minute
            if (remaining <= 60 && remaining > 59) {
                window.JARVIS?.speak('Warning: One minute remaining.', 'urgent');
                window.HUD?.showAlert('60 SECONDS REMAINING', 'warning');
            }
            
            // Continue timer
            requestAnimationFrame(updateTimer);
        };
        
        updateTimer();
    }
    
    spawnMissionElements() {
        switch (this.currentMission.type) {
            case 'combat':
                this.spawnCombatElements();
                break;
            case 'rescue':
                this.spawnRescueElements();
                break;
            case 'recon':
                this.spawnReconElements();
                break;
        }
    }
    
    spawnCombatElements() {
        // Spawn enemies
        const enemyCount = 10;
        for (let i = 0; i < enemyCount; i++) {
            const position = {
                x: (Math.random() - 0.5) * 500,
                y: Math.random() * 100 + 50,
                z: (Math.random() - 0.5) * 500
            };
            
            // Add to viewport
            if (window.Viewport) {
                window.Viewport.addTarget(position);
            }
            
            // Add to HUD
            window.HUD?.addTarget({
                id: `enemy_${i}`,
                distance: Math.sqrt(position.x ** 2 + position.y ** 2 + position.z ** 2),
                type: 'HOSTILE',
                threat: 0.7
            });
        }
        
        window.JARVIS?.speak('Multiple hostiles detected. Weapons free.');
    }
    
    spawnRescueElements() {
        // Spawn civilians
        const civilianCount = 5;
        const locations = [
            'Building A - Floor 3',
            'Parking Structure',
            'Medical Center',
            'School - West Wing',
            'Shopping District'
        ];
        
        locations.forEach((location, i) => {
            window.HUD?.showAlert(`Civilian detected: ${location}`, 'info');
        });
        
        window.JARVIS?.speak('Civilian heat signatures detected. Proceed with caution.');
    }
    
    spawnReconElements() {
        // Create scan points
        const scanPoints = 5;
        for (let i = 0; i < scanPoints; i++) {
            const point = {
                id: `scan_${i}`,
                name: `Target Alpha-${i + 1}`,
                scanned: false
            };
            
            // Add waypoint
            const position = {
                x: (Math.random() - 0.5) * 800,
                y: Math.random() * 200,
                z: (Math.random() - 0.5) * 800
            };
            
            // In a real implementation, this would add actual scan points
        }
        
        window.JARVIS?.speak('Stealth mode engaged. Avoid detection.');
        window.HUD?.setMode('STEALTH');
    }
    
    // Mission progress tracking
    updateObjective(objectiveId, value) {
        if (!this.missionActive || !this.currentMission) return;
        
        const objective = this.currentMission.objectives.find(o => o.id === objectiveId);
        if (!objective) return;
        
        // Update based on type
        switch (objective.type) {
            case 'eliminate':
            case 'rescue':
            case 'scan':
            case 'collect':
            case 'deliver':
                objective.current = Math.min(objective.current + value, objective.target);
                break;
                
            case 'protect':
            case 'stealth':
                objective.current = Math.max(0, Math.min(100, value));
                break;
        }
        
        // Check if objective completed
        if (objective.current >= objective.target) {
            this.completeObjective(objective);
        }
        
        // Check if objective failed
        if ((objective.type === 'protect' || objective.type === 'stealth') && objective.current <= 0) {
            this.failMission(`Failed: ${objective.description}`);
        }
        
        // Update display
        this.updateObjectivesDisplay();
        
        // Check mission completion
        this.checkMissionCompletion();
    }
    
    completeObjective(objective) {
        window.JARVIS?.speak(`Objective complete: ${objective.description}`);
        window.HUD?.showAlert('OBJECTIVE COMPLETE', 'success');
        window.AudioSystem?.play('uiSuccess');
        
        // Add score
        this.score += 100;
    }
    
    checkMissionCompletion() {
        const allCompleted = this.currentMission.objectives.every(obj => 
            obj.current >= obj.target
        );
        
        if (allCompleted) {
            this.completeMission();
        }
    }
    
    completeMission() {
        this.missionActive = false;
        this.currentMission.status = 'completed';
        
        const completionTime = (Date.now() - this.missionStartTime) / 1000;
        const timeBonus = Math.max(0, this.currentMission.timeLimit - completionTime) * 2;
        const totalScore = this.currentMission.rewards.score + timeBonus + this.score;
        
        // Show completion screen
        const completion = Utils.createElement('div', {
            className: 'mission-complete mission-complete',
            innerHTML: `
                <h2>MISSION COMPLETE</h2>
                <h3>${this.currentMission.name}</h3>
                <div class="completion-stats">
                    <p>Time: ${Utils.formatTime(completionTime)}</p>
                    <p>Score: ${Math.floor(totalScore)}</p>
                    <p>Experience: +${this.currentMission.rewards.experience}</p>
                </div>
                <div class="completion-rating">
                    ${this.getCompletionRating(completionTime)}
                </div>
                <button class="continue-button">CONTINUE</button>
            `
        });
        
        document.body.appendChild(completion);
        
        completion.querySelector('.continue-button').addEventListener('click', () => {
            completion.remove();
            this.resetMission();
        });
        
        // JARVIS congratulations
        window.JARVIS?.speak('Excellent work, sir. Mission accomplished with distinction.');
        window.AudioSystem?.play('powerUp');
    }
    
    failMission(reason) {
        this.missionActive = false;
        this.currentMission.status = 'failed';
        
        // Show failure screen
        const failure = Utils.createElement('div', {
            className: 'mission-failed emergency-alert',
            innerHTML: `
                <h2>MISSION FAILED</h2>
                <p>${reason}</p>
                <button class="retry-button">RETRY</button>
                <button class="abort-button">ABORT</button>
            `
        });
        
        document.body.appendChild(failure);
        
        failure.querySelector('.retry-button').addEventListener('click', () => {
            failure.remove();
            this.restartMission();
        });
        
        failure.querySelector('.abort-button').addEventListener('click', () => {
            failure.remove();
            this.resetMission();
        });
        
        // JARVIS notification
        window.JARVIS?.speak(`Mission failed. ${reason}`, 'urgent');
        window.AudioSystem?.play('systemFailure');
    }
    
    getCompletionRating(completionTime) {
        const timeRatio = completionTime / this.currentMission.timeLimit;
        let stars = 5;
        
        if (timeRatio > 0.9) stars = 1;
        else if (timeRatio > 0.7) stars = 2;
        else if (timeRatio > 0.5) stars = 3;
        else if (timeRatio > 0.3) stars = 4;
        
        return '★'.repeat(stars) + '☆'.repeat(5 - stars);
    }
    
    restartMission() {
        // Reset objectives
        this.currentMission.objectives.forEach(obj => {
            if (obj.type === 'protect' || obj.type === 'stealth') {
                obj.current = 100;
            } else {
                obj.current = 0;
            }
        });
        
        this.score = 0;
        this.startMission();
    }
    
    resetMission() {
        this.missionActive = false;
        this.currentMission = null;
        this.score = 0;
        
        // Clear objectives display
        if (this.objectivesList) {
            this.objectivesList.innerHTML = '';
        }
        
        // Reset HUD
        window.HUD?.setMode('STANDARD');
        
        // Clear spawned elements
        window.HUD?.clearTargets();
    }
    
    // Public API for integration
    onEnemyEliminated() {
        this.updateObjective('eliminate_all', 1);
    }
    
    onCivilianRescued() {
        this.updateObjective('rescue_civilians', 1);
    }
    
    onTargetScanned() {
        this.updateObjective('scan_targets', 1);
    }
    
    onDamageTaken(amount) {
        if (this.currentMission?.type === 'combat') {
            const currentHealth = this.currentMission.objectives.find(o => o.id === 'protect_core')?.current || 100;
            this.updateObjective('protect_core', currentHealth - amount);
        }
    }
    
    onDetected() {
        if (this.currentMission?.type === 'recon') {
            this.updateObjective('remain_undetected', 0);
        }
    }
}

// Make MissionSystem globally available
window.Missions = new MissionSystem();