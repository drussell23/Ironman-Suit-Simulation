// Cinematic Lighting System for Iron Man Experience
// Movie-quality lighting with dynamic effects and mood

class CinematicLighting {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.lights = {
            key: [],
            fill: [],
            rim: [],
            practical: [],
            effect: []
        };
        
        // Lighting presets
        this.presets = {
            heroic: {
                name: 'Heroic',
                ambientIntensity: 0.3,
                keyIntensity: 2.5,
                fillIntensity: 1.0,
                rimIntensity: 2.0,
                fogDensity: 0.001
            },
            dramatic: {
                name: 'Dramatic',
                ambientIntensity: 0.1,
                keyIntensity: 3.0,
                fillIntensity: 0.5,
                rimIntensity: 3.0,
                fogDensity: 0.003
            },
            workshop: {
                name: 'Workshop',
                ambientIntensity: 0.5,
                keyIntensity: 1.5,
                fillIntensity: 1.2,
                rimIntensity: 1.0,
                fogDensity: 0.0005
            },
            night: {
                name: 'Night Operations',
                ambientIntensity: 0.05,
                keyIntensity: 1.0,
                fillIntensity: 0.3,
                rimIntensity: 2.5,
                fogDensity: 0.005
            }
        };
        
        this.currentPreset = 'heroic';
        this.targetValues = {};
        this.currentValues = {};
        
        this.init();
    }
    
    init() {
        this.setupAmbientLighting();
        this.setupKeyLighting();
        this.setupFillLighting();
        this.setupRimLighting();
        this.setupPracticalLights();
        this.setupEffectLights();
        this.setupFog();
        
        // Apply initial preset
        this.applyPreset(this.currentPreset);
    }
    
    setupAmbientLighting() {
        // Base ambient - very subtle
        const ambientLight = new THREE.AmbientLight(0x202030, 0.3);
        this.scene.add(ambientLight);
        this.lights.ambient = ambientLight;
        
        // Hemisphere light for natural sky/ground variation
        const hemiLight = new THREE.HemisphereLight(
            0x404060,  // Sky color - cool blue
            0x101010,  // Ground color - dark
            0.5
        );
        hemiLight.position.set(0, 100, 0);
        this.scene.add(hemiLight);
        this.lights.hemisphere = hemiLight;
    }
    
    setupKeyLighting() {
        // Main key light - strong directional
        const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
        keyLight.position.set(40, 80, 40);
        keyLight.castShadow = true;
        
        // High quality shadows
        keyLight.shadow.mapSize.width = 4096;
        keyLight.shadow.mapSize.height = 4096;
        keyLight.shadow.camera.near = 0.5;
        keyLight.shadow.camera.far = 200;
        keyLight.shadow.camera.left = -100;
        keyLight.shadow.camera.right = 100;
        keyLight.shadow.camera.top = 100;
        keyLight.shadow.camera.bottom = -100;
        keyLight.shadow.bias = -0.0005;
        keyLight.shadow.normalBias = 0.02;
        keyLight.shadow.radius = 2;
        keyLight.shadow.blurSamples = 25;
        
        this.scene.add(keyLight);
        this.lights.key.push(keyLight);
        
        // Secondary key for cross lighting
        const keyLight2 = new THREE.SpotLight(0xffffff, 1.5, 200, Math.PI / 4, 0.3, 1);
        keyLight2.position.set(-30, 70, 60);
        keyLight2.target.position.set(0, 20, 0);
        keyLight2.castShadow = true;
        keyLight2.shadow.mapSize.width = 2048;
        keyLight2.shadow.mapSize.height = 2048;
        this.scene.add(keyLight2);
        this.scene.add(keyLight2.target);
        this.lights.key.push(keyLight2);
    }
    
    setupFillLighting() {
        // Cool fill light from opposite side
        const fillLight = new THREE.DirectionalLight(0x4080ff, 1.0);
        fillLight.position.set(-60, 40, -30);
        this.scene.add(fillLight);
        this.lights.fill.push(fillLight);
        
        // Soft area light for overall fill
        if (THREE.RectAreaLight) {
            const areaLight = new THREE.RectAreaLight(0x8080ff, 50, 40, 40);
            areaLight.position.set(-70, 50, 0);
            areaLight.rotation.y = Math.PI / 4;
            this.scene.add(areaLight);
            this.lights.fill.push(areaLight);
        }
        
        // Bottom fill to simulate ground bounce
        const bounceLight = new THREE.DirectionalLight(0x443322, 0.3);
        bounceLight.position.set(0, -50, 0);
        bounceLight.target.position.set(0, 0, 0);
        this.scene.add(bounceLight);
        this.scene.add(bounceLight.target);
        this.lights.fill.push(bounceLight);
    }
    
    setupRimLighting() {
        // Strong rim light for edge definition
        const rimLight1 = new THREE.SpotLight(0xff8844, 2.0, 200, Math.PI / 3, 0.5, 1);
        rimLight1.position.set(-50, 60, -60);
        rimLight1.target.position.set(0, 30, 0);
        this.scene.add(rimLight1);
        this.scene.add(rimLight1.target);
        this.lights.rim.push(rimLight1);
        
        // Cool rim from other side
        const rimLight2 = new THREE.SpotLight(0x4488ff, 1.5, 150, Math.PI / 4, 0.3, 1);
        rimLight2.position.set(60, 50, -40);
        rimLight2.target.position.set(0, 30, 0);
        this.scene.add(rimLight2);
        this.scene.add(rimLight2.target);
        this.lights.rim.push(rimLight2);
        
        // Top rim for helmet highlight
        const topRim = new THREE.SpotLight(0xffffff, 1.0, 100, Math.PI / 6, 0.2, 1);
        topRim.position.set(0, 90, -30);
        topRim.target.position.set(0, 50, 0);
        this.scene.add(topRim);
        this.scene.add(topRim.target);
        this.lights.rim.push(topRim);
    }
    
    setupPracticalLights() {
        // Arc reactor spotlight
        const arcSpot = new THREE.SpotLight(0x00ccff, 3.0, 80, Math.PI / 3, 0.2, 1);
        arcSpot.position.set(0, 40, 20);
        arcSpot.target.position.set(0, 40, 60);
        arcSpot.castShadow = false; // No shadows for glow effect
        this.scene.add(arcSpot);
        this.scene.add(arcSpot.target);
        this.lights.practical.push(arcSpot);
        
        // Eye lights
        const leftEyeSpot = new THREE.SpotLight(0x00ccff, 1.0, 50, Math.PI / 4, 0.5, 1);
        leftEyeSpot.position.set(-3, 56, 15);
        leftEyeSpot.target.position.set(-3, 56, 40);
        this.scene.add(leftEyeSpot);
        this.scene.add(leftEyeSpot.target);
        this.lights.practical.push(leftEyeSpot);
        
        const rightEyeSpot = new THREE.SpotLight(0x00ccff, 1.0, 50, Math.PI / 4, 0.5, 1);
        rightEyeSpot.position.set(3, 56, 15);
        rightEyeSpot.target.position.set(3, 56, 40);
        this.scene.add(rightEyeSpot);
        this.scene.add(rightEyeSpot.target);
        this.lights.practical.push(rightEyeSpot);
        
        // Repulsor glow lights
        const leftRepulsorLight = new THREE.PointLight(0x00ccff, 0.5, 30);
        leftRepulsorLight.position.set(-20, 6, 5);
        this.scene.add(leftRepulsorLight);
        this.lights.practical.push(leftRepulsorLight);
        
        const rightRepulsorLight = new THREE.PointLight(0x00ccff, 0.5, 30);
        rightRepulsorLight.position.set(20, 6, 5);
        this.scene.add(rightRepulsorLight);
        this.lights.practical.push(rightRepulsorLight);
    }
    
    setupEffectLights() {
        // Volumetric light shafts (simulated with spotlights)
        const shaft1 = new THREE.SpotLight(0xffffff, 0.5, 300, Math.PI / 8, 0.1, 1);
        shaft1.position.set(100, 150, 100);
        shaft1.target.position.set(0, 0, 0);
        this.scene.add(shaft1);
        this.scene.add(shaft1.target);
        this.lights.effect.push(shaft1);
        
        // Color accent lights for mood
        const accentLight1 = new THREE.PointLight(0x0088ff, 1.0, 80);
        accentLight1.position.set(30, 40, 30);
        this.scene.add(accentLight1);
        this.lights.effect.push(accentLight1);
        
        const accentLight2 = new THREE.PointLight(0xff0066, 0.5, 60);
        accentLight2.position.set(-40, 30, -20);
        this.scene.add(accentLight2);
        this.lights.effect.push(accentLight2);
        
        // Moving light for dynamic effect
        const movingLight = new THREE.SpotLight(0xffffff, 1.0, 150, Math.PI / 6, 0.5, 1);
        movingLight.position.set(0, 80, 0);
        movingLight.target.position.set(0, 0, 0);
        this.scene.add(movingLight);
        this.scene.add(movingLight.target);
        this.lights.effect.push(movingLight);
        this.lights.movingLight = movingLight;
    }
    
    setupFog() {
        // Exponential fog for atmosphere
        this.scene.fog = new THREE.FogExp2(0x000000, 0.001);
    }
    
    applyPreset(presetName) {
        const preset = this.presets[presetName];
        if (!preset) return;
        
        this.currentPreset = presetName;
        
        // Set target values for smooth transition
        this.targetValues = {
            ambientIntensity: preset.ambientIntensity,
            keyIntensity: preset.keyIntensity,
            fillIntensity: preset.fillIntensity,
            rimIntensity: preset.rimIntensity,
            fogDensity: preset.fogDensity
        };
        
        // Initialize current values if not set
        if (!this.currentValues.ambientIntensity) {
            this.currentValues = { ...this.targetValues };
            this.applyLightingValues();
        }
    }
    
    applyLightingValues() {
        // Apply ambient
        if (this.lights.ambient) {
            this.lights.ambient.intensity = this.currentValues.ambientIntensity;
        }
        if (this.lights.hemisphere) {
            this.lights.hemisphere.intensity = this.currentValues.ambientIntensity * 1.5;
        }
        
        // Apply key lights
        this.lights.key.forEach(light => {
            light.intensity = this.currentValues.keyIntensity * 
                (light instanceof THREE.SpotLight ? 0.6 : 1.0);
        });
        
        // Apply fill lights
        this.lights.fill.forEach(light => {
            light.intensity = this.currentValues.fillIntensity * 
                (light instanceof THREE.RectAreaLight ? 50 : 1.0);
        });
        
        // Apply rim lights
        this.lights.rim.forEach(light => {
            light.intensity = this.currentValues.rimIntensity;
        });
        
        // Apply fog
        if (this.scene.fog) {
            this.scene.fog.density = this.currentValues.fogDensity;
        }
    }
    
    update(deltaTime, time) {
        // Smooth transitions between presets
        const lerpSpeed = 2.0 * deltaTime;
        
        Object.keys(this.targetValues).forEach(key => {
            this.currentValues[key] = THREE.MathUtils.lerp(
                this.currentValues[key],
                this.targetValues[key],
                lerpSpeed
            );
        });
        
        this.applyLightingValues();
        
        // Animate practical lights
        this.animatePracticalLights(time);
        
        // Animate effect lights
        this.animateEffectLights(time);
    }
    
    animatePracticalLights(time) {
        // Arc reactor pulse
        const arcPulse = Math.sin(time * 3) * 0.3 + 0.7;
        const arcSpot = this.lights.practical[0];
        if (arcSpot) {
            arcSpot.intensity = 3.0 * arcPulse;
        }
        
        // Eye flicker
        const eyeFlicker = Math.random() < 0.98 ? 1.0 : 0.3;
        this.lights.practical[1].intensity = 1.0 * eyeFlicker;
        this.lights.practical[2].intensity = 1.0 * eyeFlicker;
        
        // Repulsor charge effect
        const repulsorCharge = Math.sin(time * 8) * 0.3 + 0.7;
        this.lights.practical[3].intensity = 0.5 * repulsorCharge;
        this.lights.practical[4].intensity = 0.5 * repulsorCharge;
    }
    
    animateEffectLights(time) {
        // Moving spotlight
        if (this.lights.movingLight) {
            const radius = 50;
            this.lights.movingLight.position.x = Math.sin(time * 0.5) * radius;
            this.lights.movingLight.position.z = Math.cos(time * 0.5) * radius;
            this.lights.movingLight.target.position.x = Math.sin(time * 0.5 + 1) * 20;
            this.lights.movingLight.target.position.z = Math.cos(time * 0.5 + 1) * 20;
        }
        
        // Pulsing accent lights
        this.lights.effect.forEach((light, index) => {
            if (light instanceof THREE.PointLight) {
                const pulse = Math.sin(time * 2 + index * Math.PI) * 0.3 + 0.7;
                light.intensity = light.intensity * pulse;
            }
        });
    }
    
    // Special lighting effects
    activateEmergencyLighting() {
        // Red alert lighting
        this.lights.key.forEach(light => {
            light.color.setHex(0xff0000);
        });
        this.lights.fill.forEach(light => {
            light.intensity *= 0.3;
        });
        
        // Add flashing red lights
        const emergencyLight = new THREE.PointLight(0xff0000, 3.0, 200);
        emergencyLight.position.set(0, 80, 0);
        this.scene.add(emergencyLight);
        this.lights.emergency = emergencyLight;
    }
    
    deactivateEmergencyLighting() {
        // Restore normal lighting
        this.lights.key.forEach(light => {
            light.color.setHex(0xffffff);
        });
        
        if (this.lights.emergency) {
            this.scene.remove(this.lights.emergency);
            delete this.lights.emergency;
        }
        
        this.applyPreset(this.currentPreset);
    }
    
    // Lightning effect for dramatic moments
    createLightningFlash() {
        const flashLight = new THREE.DirectionalLight(0xffffff, 5.0);
        flashLight.position.set(100, 100, 100);
        this.scene.add(flashLight);
        
        // Quick flash
        setTimeout(() => {
            flashLight.intensity = 10.0;
            setTimeout(() => {
                flashLight.intensity = 3.0;
                setTimeout(() => {
                    this.scene.remove(flashLight);
                }, 50);
            }, 50);
        }, 100);
    }
    
    // Set lighting based on time of day
    setTimeOfDay(hour) {
        if (hour >= 6 && hour < 12) {
            // Morning
            this.applyPreset('workshop');
            this.lights.key[0].color.setHex(0xfff5e6);
        } else if (hour >= 12 && hour < 18) {
            // Afternoon
            this.applyPreset('heroic');
        } else if (hour >= 18 && hour < 21) {
            // Evening
            this.applyPreset('dramatic');
            this.lights.key[0].color.setHex(0xffaa66);
        } else {
            // Night
            this.applyPreset('night');
        }
    }
    
    // Get current lighting info for UI
    getLightingInfo() {
        return {
            preset: this.currentPreset,
            presets: Object.keys(this.presets),
            values: this.currentValues
        };
    }
}

// Export
window.CinematicLighting = CinematicLighting;