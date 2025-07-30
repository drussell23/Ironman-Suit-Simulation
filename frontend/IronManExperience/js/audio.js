// Audio System for Iron Man Experience

class AudioSystem {
    constructor() {
        this.context = null;
        this.masterGain = null;
        this.sounds = {};
        this.music = {};
        this.ambientSounds = [];
        
        // Volume settings
        this.volumes = {
            master: 0.8,
            sfx: 1.0,
            music: 0.7,
            voice: 1.0,
            ambient: 0.5
        };
        
        // Audio elements from HTML
        this.audioElements = {
            boot: Utils.$('#bootSound'),
            hud: Utils.$('#hudSound'),
            jarvis: Utils.$('#jarvisSound'),
            alert: Utils.$('#alertSound'),
            repulsor: Utils.$('#repulsorSound')
        };
        
        // Initialize Web Audio API
        this.initWebAudio();
        
        // Spatial audio settings
        this.listener = null;
        this.panner = null;
        
        this.init();
    }
    
    initWebAudio() {
        try {
            window.AudioContext = window.AudioContext || window.webkitAudioContext;
            this.context = new AudioContext();
            
            // Create master gain node
            this.masterGain = this.context.createGain();
            this.masterGain.connect(this.context.destination);
            this.masterGain.gain.value = this.volumes.master;
            
            // Create separate gain nodes for different sound types
            this.sfxGain = this.context.createGain();
            this.sfxGain.connect(this.masterGain);
            this.sfxGain.gain.value = this.volumes.sfx;
            
            this.musicGain = this.context.createGain();
            this.musicGain.connect(this.masterGain);
            this.musicGain.gain.value = this.volumes.music;
            
            this.voiceGain = this.context.createGain();
            this.voiceGain.connect(this.masterGain);
            this.voiceGain.gain.value = this.volumes.voice;
            
            // Set up 3D audio listener
            this.listener = this.context.listener;
            
        } catch (e) {
            console.warn('Web Audio API not supported:', e);
        }
    }
    
    init() {
        // Resume audio context on user interaction
        document.addEventListener('click', () => {
            if (this.context && this.context.state === 'suspended') {
                this.context.resume();
            }
        }, { once: true });
        
        // Load and prepare sounds
        this.loadSounds();
        
        // Start ambient sounds
        setTimeout(() => this.startAmbientSounds(), 2000);
    }
    
    loadSounds() {
        // Define sound library
        const soundLibrary = {
            // UI Sounds
            uiClick: { url: 'assets/audio/ui-click.mp3', volume: 0.5 },
            uiHover: { url: 'assets/audio/ui-hover.mp3', volume: 0.3 },
            uiError: { url: 'assets/audio/ui-error.mp3', volume: 0.6 },
            uiSuccess: { url: 'assets/audio/ui-success.mp3', volume: 0.5 },
            
            // Weapon Sounds
            repulsorCharge: { url: 'assets/audio/repulsor-charge.mp3', volume: 0.7 },
            repulsorFire: { url: 'assets/audio/repulsor-fire.mp3', volume: 0.8 },
            missileLaunch: { url: 'assets/audio/missile-launch.mp3', volume: 0.9 },
            laserBeam: { url: 'assets/audio/laser-beam.mp3', volume: 0.7 },
            
            // Movement Sounds
            thrusterLoop: { url: 'assets/audio/thruster-loop.mp3', volume: 0.5, loop: true },
            sonicBoom: { url: 'assets/audio/sonic-boom.mp3', volume: 1.0 },
            landing: { url: 'assets/audio/landing.mp3', volume: 0.8 },
            
            // System Sounds
            powerUp: { url: 'assets/audio/power-up.mp3', volume: 0.6 },
            powerDown: { url: 'assets/audio/power-down.mp3', volume: 0.6 },
            systemFailure: { url: 'assets/audio/system-failure.mp3', volume: 0.8 },
            
            // Ambient Sounds
            windLoop: { url: 'assets/audio/wind-loop.mp3', volume: 0.3, loop: true },
            cityAmbient: { url: 'assets/audio/city-ambient.mp3', volume: 0.2, loop: true }
        };
        
        // Note: In a real implementation, you would load these files
        // For now, we'll create synthetic sounds
        this.createSyntheticSounds();
    }
    
    createSyntheticSounds() {
        // Create synthetic sound effects using Web Audio API
        this.sounds.uiClick = () => this.playTone(1000, 0.05, 'sine', 0.3);
        this.sounds.uiHover = () => this.playTone(2000, 0.03, 'sine', 0.2);
        this.sounds.uiError = () => this.playTone(200, 0.2, 'sawtooth', 0.4);
        this.sounds.uiSuccess = () => this.playChime([523, 659, 784], 0.1, 0.3);
        
        this.sounds.repulsorCharge = () => this.playSweep(100, 2000, 0.5, 0.5);
        this.sounds.repulsorFire = () => this.playNoise(0.2, 0.8);
        this.sounds.powerUp = () => this.playSweep(200, 1000, 1, 0.4);
        this.sounds.powerDown = () => this.playSweep(1000, 200, 1, 0.4);
        
        // Use HTML audio elements where available
        Object.entries(this.audioElements).forEach(([name, element]) => {
            if (element) {
                this.sounds[name] = () => {
                    element.currentTime = 0;
                    element.volume = this.volumes.sfx * this.volumes.master;
                    element.play().catch(() => {});
                };
            }
        });
    }
    
    // Synthetic sound generators
    playTone(frequency, duration, type = 'sine', volume = 0.5) {
        if (!this.context) return;
        
        const oscillator = this.context.createOscillator();
        const gainNode = this.context.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(this.sfxGain);
        
        oscillator.type = type;
        oscillator.frequency.value = frequency;
        
        gainNode.gain.setValueAtTime(0, this.context.currentTime);
        gainNode.gain.linearRampToValueAtTime(volume, this.context.currentTime + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.context.currentTime + duration);
        
        oscillator.start(this.context.currentTime);
        oscillator.stop(this.context.currentTime + duration);
    }
    
    playChime(frequencies, duration, volume = 0.5) {
        frequencies.forEach((freq, index) => {
            setTimeout(() => {
                this.playTone(freq, duration, 'sine', volume);
            }, index * 100);
        });
    }
    
    playSweep(startFreq, endFreq, duration, volume = 0.5) {
        if (!this.context) return;
        
        const oscillator = this.context.createOscillator();
        const gainNode = this.context.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(this.sfxGain);
        
        oscillator.type = 'sawtooth';
        oscillator.frequency.setValueAtTime(startFreq, this.context.currentTime);
        oscillator.frequency.exponentialRampToValueAtTime(endFreq, this.context.currentTime + duration);
        
        gainNode.gain.setValueAtTime(0, this.context.currentTime);
        gainNode.gain.linearRampToValueAtTime(volume, this.context.currentTime + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.context.currentTime + duration);
        
        oscillator.start(this.context.currentTime);
        oscillator.stop(this.context.currentTime + duration);
    }
    
    playNoise(duration, volume = 0.5) {
        if (!this.context) return;
        
        const bufferSize = this.context.sampleRate * duration;
        const buffer = this.context.createBuffer(1, bufferSize, this.context.sampleRate);
        const output = buffer.getChannelData(0);
        
        for (let i = 0; i < bufferSize; i++) {
            output[i] = Math.random() * 2 - 1;
        }
        
        const noise = this.context.createBufferSource();
        const gainNode = this.context.createGain();
        const filter = this.context.createBiquadFilter();
        
        noise.buffer = buffer;
        noise.connect(filter);
        filter.connect(gainNode);
        gainNode.connect(this.sfxGain);
        
        filter.type = 'lowpass';
        filter.frequency.value = 1000;
        
        gainNode.gain.setValueAtTime(0, this.context.currentTime);
        gainNode.gain.linearRampToValueAtTime(volume, this.context.currentTime + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.context.currentTime + duration);
        
        noise.start(this.context.currentTime);
    }
    
    // Ambient sound system
    startAmbientSounds() {
        // Create wind sound
        this.createWindSound();
        
        // Create engine hum
        this.createEngineHum();
    }
    
    createWindSound() {
        if (!this.context) return;
        
        const noise = this.context.createBufferSource();
        const bufferSize = this.context.sampleRate * 2;
        const buffer = this.context.createBuffer(1, bufferSize, this.context.sampleRate);
        const output = buffer.getChannelData(0);
        
        for (let i = 0; i < bufferSize; i++) {
            output[i] = Math.random() * 2 - 1;
        }
        
        noise.buffer = buffer;
        noise.loop = true;
        
        const filter = this.context.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.value = 400;
        filter.Q.value = 0.5;
        
        const gain = this.context.createGain();
        gain.gain.value = 0.1;
        
        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);
        
        noise.start();
        
        // Modulate wind based on altitude
        this.windGain = gain;
    }
    
    createEngineHum() {
        if (!this.context) return;
        
        const oscillator = this.context.createOscillator();
        oscillator.type = 'sine';
        oscillator.frequency.value = 60;
        
        const gain = this.context.createGain();
        gain.gain.value = 0.05;
        
        oscillator.connect(gain);
        gain.connect(this.masterGain);
        
        oscillator.start();
        
        // Store for later modulation
        this.engineOscillator = oscillator;
        this.engineGain = gain;
    }
    
    // 3D Spatial audio
    play3DSound(soundName, position, volume = 1.0) {
        if (!this.context || !this.sounds[soundName]) return;
        
        const panner = this.context.createPanner();
        panner.panningModel = 'HRTF';
        panner.distanceModel = 'inverse';
        panner.refDistance = 1;
        panner.maxDistance = 10000;
        panner.rolloffFactor = 1;
        panner.coneInnerAngle = 360;
        panner.coneOuterAngle = 0;
        panner.coneOuterGain = 0;
        
        panner.setPosition(position.x, position.y, position.z);
        
        // Connect to master gain
        panner.connect(this.masterGain);
        
        // Play sound through panner
        // (In a real implementation, you would route the sound through the panner)
    }
    
    updateListenerPosition(position, orientation) {
        if (!this.listener) return;
        
        this.listener.setPosition(position.x, position.y, position.z);
        this.listener.setOrientation(
            orientation.forward.x, orientation.forward.y, orientation.forward.z,
            orientation.up.x, orientation.up.y, orientation.up.z
        );
    }
    
    // Dynamic audio based on flight
    updateFlightAudio(altitude, velocity) {
        // Update wind sound based on altitude
        if (this.windGain) {
            const windVolume = Utils.map(altitude, 0, 1000, 0.05, 0.3);
            this.windGain.gain.linearRampToValueAtTime(
                windVolume * this.volumes.ambient,
                this.context.currentTime + 0.1
            );
        }
        
        // Update engine sound based on velocity
        if (this.engineOscillator && this.engineGain) {
            const enginePitch = Utils.map(velocity, 0, 500, 60, 120);
            const engineVolume = Utils.map(velocity, 0, 500, 0.05, 0.2);
            
            this.engineOscillator.frequency.linearRampToValueAtTime(
                enginePitch,
                this.context.currentTime + 0.1
            );
            
            this.engineGain.gain.linearRampToValueAtTime(
                engineVolume * this.volumes.ambient,
                this.context.currentTime + 0.1
            );
        }
        
        // Sonic boom at high speed
        if (velocity > 1000 && !this.sonicBoomPlayed) {
            this.play('sonicBoom');
            this.sonicBoomPlayed = true;
        } else if (velocity < 900) {
            this.sonicBoomPlayed = false;
        }
    }
    
    // Music system
    playMusic(track) {
        // In a real implementation, this would play background music
        console.log(`Playing music track: ${track}`);
    }
    
    stopMusic() {
        // Stop all music
    }
    
    // Public API
    play(soundName, volume = 1.0) {
        if (this.sounds[soundName]) {
            this.sounds[soundName](volume * this.volumes.sfx);
        } else if (this.audioElements[soundName]) {
            this.audioElements[soundName].play().catch(() => {});
        }
    }
    
    setMasterVolume(volume) {
        this.volumes.master = Utils.clamp(volume, 0, 1);
        if (this.masterGain) {
            this.masterGain.gain.value = this.volumes.master;
        }
    }
    
    setSFXVolume(volume) {
        this.volumes.sfx = Utils.clamp(volume, 0, 1);
        if (this.sfxGain) {
            this.sfxGain.gain.value = this.volumes.sfx;
        }
    }
    
    setMusicVolume(volume) {
        this.volumes.music = Utils.clamp(volume, 0, 1);
        if (this.musicGain) {
            this.musicGain.gain.value = this.volumes.music;
        }
    }
    
    // UI feedback sounds
    playUISound(type) {
        switch(type) {
            case 'hover':
                this.play('uiHover');
                break;
            case 'click':
                this.play('uiClick');
                break;
            case 'error':
                this.play('uiError');
                break;
            case 'success':
                this.play('uiSuccess');
                break;
        }
    }
}

// Make AudioSystem globally available
window.AudioSystem = AudioSystem;