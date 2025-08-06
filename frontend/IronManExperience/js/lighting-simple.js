// Simplified Lighting System for Iron Man Experience

class EnhancedLighting {
    constructor(scene) {
        this.scene = scene;
        this.lights = {};
        this.init();
    }
    
    init() {
        this.setupBasicLighting();
    }
    
    setupBasicLighting() {
        // Ambient light for overall visibility
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);
        this.lights.ambient = ambientLight;
        
        // Main directional light
        const mainLight = new THREE.DirectionalLight(0xffffff, 1);
        mainLight.position.set(50, 100, 50);
        mainLight.castShadow = true;
        mainLight.shadow.mapSize.width = 2048;
        mainLight.shadow.mapSize.height = 2048;
        mainLight.shadow.camera.near = 0.5;
        mainLight.shadow.camera.far = 500;
        mainLight.shadow.camera.left = -100;
        mainLight.shadow.camera.right = 100;
        mainLight.shadow.camera.top = 100;
        mainLight.shadow.camera.bottom = -100;
        this.scene.add(mainLight);
        this.lights.main = mainLight;
        
        // Fill light from opposite direction
        const fillLight = new THREE.DirectionalLight(0x4080ff, 0.5);
        fillLight.position.set(-50, 50, -50);
        this.scene.add(fillLight);
        this.lights.fill = fillLight;
        
        // Point lights for suit effects
        this.lights.arcReactor = new THREE.PointLight(0x00ffff, 2, 50);
        this.lights.leftEye = new THREE.PointLight(0x00aaff, 1, 30);
        this.lights.rightEye = new THREE.PointLight(0x00aaff, 1, 30);
        
        // Thruster lights
        this.lights.thrusters = {
            leftBoot: new THREE.PointLight(0xffaa00, 0, 40),
            rightBoot: new THREE.PointLight(0xffaa00, 0, 40),
            leftHand: new THREE.PointLight(0xffaa00, 0, 30),
            rightHand: new THREE.PointLight(0xffaa00, 0, 30)
        };
    }
    
    attachToSuit(suit) {
        if (!suit) return;
        
        // Arc reactor light
        const arcReactor = suit.getObjectByName('ArcReactor');
        if (arcReactor && this.lights.arcReactor) {
            arcReactor.add(this.lights.arcReactor);
        }
        
        // Eye lights
        const helmet = suit.getObjectByName('Helmet');
        if (helmet) {
            if (this.lights.leftEye) {
                this.lights.leftEye.position.set(-3, 2, 12);
                helmet.add(this.lights.leftEye);
            }
            if (this.lights.rightEye) {
                this.lights.rightEye.position.set(3, 2, 12);
                helmet.add(this.lights.rightEye);
            }
        }
    }
    
    setThrusterIntensity(intensity) {
        if (this.lights.thrusters) {
            Object.values(this.lights.thrusters).forEach(light => {
                light.intensity = intensity * 3;
            });
        }
    }
    
    update(deltaTime) {
        // Simple pulsing effect for arc reactor
        if (this.lights.arcReactor) {
            const pulse = Math.sin(Date.now() * 0.002) * 0.5 + 0.5;
            this.lights.arcReactor.intensity = 1.5 + pulse;
        }
    }
    
    setCombatMode(enabled) {
        // Adjust lighting for combat mode
        if (enabled) {
            this.lights.ambient.intensity = 0.3;
            this.lights.main.color.setHex(0xff6666);
        } else {
            this.lights.ambient.intensity = 0.5;
            this.lights.main.color.setHex(0xffffff);
        }
    }
}

// Export for use
window.EnhancedLighting = EnhancedLighting;