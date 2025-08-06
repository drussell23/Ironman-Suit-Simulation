// Enhanced Lighting System for Iron Man Experience
// Advanced lighting setup with dynamic effects

class EnhancedLighting {
    constructor(scene) {
        this.scene = scene;
        this.lights = {};
        this.dynamicLights = [];
        this.time = 0;
        
        this.init();
    }
    
    init() {
        this.setupAmbientLighting();
        this.setupDirectionalLighting();
        this.setupPointLights();
        this.setupSpotLights();
        this.setupAreaLights();
        this.setupVolumetricLighting();
    }
    
    setupAmbientLighting() {
        // Hemisphere light for natural sky/ground lighting
        const hemiLight = new THREE.HemisphereLight(0x4080ff, 0x002244, 0.3);
        hemiLight.position.set(0, 500, 0);
        this.scene.add(hemiLight);
        this.lights.hemisphere = hemiLight;
        
        // Subtle ambient for overall visibility
        const ambientLight = new THREE.AmbientLight(0x202040, 0.2);
        this.scene.add(ambientLight);
        this.lights.ambient = ambientLight;
    }
    
    setupDirectionalLighting() {
        // Main directional light (moonlight)
        const moonLight = new THREE.DirectionalLight(0x8899ff, 0.6);
        moonLight.position.set(-200, 300, -100);
        moonLight.target.position.set(0, 0, 0);
        moonLight.castShadow = true;
        
        // High quality shadows
        moonLight.shadow.mapSize.width = 4096;
        moonLight.shadow.mapSize.height = 4096;
        moonLight.shadow.camera.near = 0.5;
        moonLight.shadow.camera.far = 1000;
        moonLight.shadow.camera.left = -200;
        moonLight.shadow.camera.right = 200;
        moonLight.shadow.camera.top = 200;
        moonLight.shadow.camera.bottom = -200;
        moonLight.shadow.bias = -0.0005;
        moonLight.shadow.normalBias = 0.02;
        
        this.scene.add(moonLight);
        this.scene.add(moonLight.target);
        this.lights.moon = moonLight;
        
        // Secondary fill light
        const fillLight = new THREE.DirectionalLight(0xff8844, 0.3);
        fillLight.position.set(100, 200, 200);
        this.scene.add(fillLight);
        this.lights.fill = fillLight;
    }
    
    setupPointLights() {
        // Arc reactor lights (will be attached to suit)
        const arcReactorLight = new THREE.PointLight(0x00ffff, 3, 100);
        arcReactorLight.castShadow = true;
        arcReactorLight.shadow.mapSize.width = 1024;
        arcReactorLight.shadow.mapSize.height = 1024;
        this.lights.arcReactor = arcReactorLight;
        
        // Eye lights (will be attached to helmet)
        const leftEyeLight = new THREE.PointLight(0x00aaff, 2, 50);
        this.lights.leftEye = leftEyeLight;
        
        const rightEyeLight = new THREE.PointLight(0x00aaff, 2, 50);
        this.lights.rightEye = rightEyeLight;
        
        // Thruster lights (will be attached to boots/hands)
        const thrusterColor = 0xffaa00;
        this.lights.thrusters = {
            leftBoot: new THREE.PointLight(thrusterColor, 0, 80),
            rightBoot: new THREE.PointLight(thrusterColor, 0, 80),
            leftHand: new THREE.PointLight(thrusterColor, 0, 60),
            rightHand: new THREE.PointLight(thrusterColor, 0, 60)
        };
    }
    
    setupSpotLights() {
        // Dramatic rim lighting
        const rimLight1 = new THREE.SpotLight(0x4080ff, 1, 300, Math.PI / 4, 0.5, 1);
        rimLight1.position.set(-100, 200, -100);
        rimLight1.target.position.set(0, 100, 0);
        this.scene.add(rimLight1);
        this.scene.add(rimLight1.target);
        this.lights.rim1 = rimLight1;
        
        const rimLight2 = new THREE.SpotLight(0xff4080, 0.5, 300, Math.PI / 4, 0.5, 1);
        rimLight2.position.set(100, 200, -100);
        rimLight2.target.position.set(0, 100, 0);
        this.scene.add(rimLight2);
        this.scene.add(rimLight2.target);
        this.lights.rim2 = rimLight2;
    }
    
    setupAreaLights() {
        // RectAreaLight for soft lighting (requires RectAreaLightUniformsLib)
        if (typeof THREE.RectAreaLight !== 'undefined') {
            const width = 100;
            const height = 100;
            const intensity = 0.5;
            const rectLight = new THREE.RectAreaLight(0xffffff, intensity, width, height);
            rectLight.position.set(0, 300, 100);
            rectLight.lookAt(0, 0, 0);
            this.scene.add(rectLight);
            this.lights.area = rectLight;
        }
    }
    
    setupVolumetricLighting() {
        // Create volumetric light shafts
        this.createLightShaft(new THREE.Vector3(100, 400, -200), 0x4080ff, 0.5);
        this.createLightShaft(new THREE.Vector3(-150, 350, -150), 0x8040ff, 0.3);
    }
    
    createLightShaft(position, color, intensity) {
        const geometry = new THREE.CylinderGeometry(5, 50, 400, 8, 1, true);
        const material = new THREE.ShaderMaterial({
            uniforms: {
                color: { value: new THREE.Color(color) },
                intensity: { value: intensity },
                time: { value: 0 }
            },
            vertexShader: `
                varying vec3 vPosition;
                varying vec2 vUv;
                void main() {
                    vPosition = position;
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 color;
                uniform float intensity;
                uniform float time;
                varying vec3 vPosition;
                varying vec2 vUv;
                
                void main() {
                    float fade = 1.0 - vUv.y;
                    float noise = sin(vPosition.y * 0.1 + time) * 0.1 + 0.9;
                    float alpha = fade * intensity * noise * 0.3;
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });
        
        const shaft = new THREE.Mesh(geometry, material);
        shaft.position.copy(position);
        shaft.rotation.x = Math.PI;
        this.scene.add(shaft);
        this.dynamicLights.push({ mesh: shaft, material: material });
    }
    
    attachToSuit(suit) {
        // Find suit parts and attach lights
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
        
        // Thruster lights
        const feet = suit.getObjectByName('Feet');
        if (feet && this.lights.thrusters) {
            this.lights.thrusters.leftBoot.position.set(-10, 30, -5);
            this.lights.thrusters.rightBoot.position.set(10, 30, -5);
            feet.add(this.lights.thrusters.leftBoot);
            feet.add(this.lights.thrusters.rightBoot);
        }
        
        const hands = suit.getObjectByName('Hands');
        if (hands && this.lights.thrusters) {
            this.lights.thrusters.leftHand.position.set(-22, 95, 5);
            this.lights.thrusters.rightHand.position.set(22, 95, 5);
            hands.add(this.lights.thrusters.leftHand);
            hands.add(this.lights.thrusters.rightHand);
        }
    }
    
    setThrusterIntensity(intensity) {
        // Control thruster light intensity
        if (this.lights.thrusters) {
            Object.values(this.lights.thrusters).forEach(light => {
                light.intensity = intensity * 5;
            });
        }
    }
    
    update(deltaTime) {
        this.time += deltaTime;
        
        // Update dynamic light effects
        this.dynamicLights.forEach(light => {
            light.material.uniforms.time.value = this.time;
        });
        
        // Pulse arc reactor light
        if (this.lights.arcReactor) {
            const pulse = Math.sin(this.time * 2) * 0.5 + 0.5;
            this.lights.arcReactor.intensity = 2 + pulse * 2;
        }
        
        // Flicker eye lights slightly
        if (this.lights.leftEye && this.lights.rightEye) {
            const flicker = Math.random() * 0.1 + 0.9;
            this.lights.leftEye.intensity = 2 * flicker;
            this.lights.rightEye.intensity = 2 * flicker;
        }
        
        // Animate rim lights
        if (this.lights.rim1 && this.lights.rim2) {
            const angle = this.time * 0.5;
            this.lights.rim1.position.x = Math.cos(angle) * 150;
            this.lights.rim1.position.z = Math.sin(angle) * 150;
            
            this.lights.rim2.position.x = Math.cos(angle + Math.PI) * 150;
            this.lights.rim2.position.z = Math.sin(angle + Math.PI) * 150;
        }
    }
    
    setCombatMode(enabled) {
        // Adjust lighting for combat mode
        if (enabled) {
            this.lights.ambient.intensity = 0.1;
            this.lights.hemisphere.intensity = 0.2;
            if (this.lights.rim1) this.lights.rim1.color.setHex(0xff0000);
            if (this.lights.rim2) this.lights.rim2.color.setHex(0xff0000);
        } else {
            this.lights.ambient.intensity = 0.2;
            this.lights.hemisphere.intensity = 0.3;
            if (this.lights.rim1) this.lights.rim1.color.setHex(0x4080ff);
            if (this.lights.rim2) this.lights.rim2.color.setHex(0xff4080);
        }
    }
}

// Export for use
window.EnhancedLighting = EnhancedLighting;