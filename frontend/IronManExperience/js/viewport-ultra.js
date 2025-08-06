// Ultra-Realistic Viewport for Iron Man Experience
// Cinematic quality rendering with advanced effects

class UltraViewport {
    constructor() {
        this.canvas = document.getElementById('viewport');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.suit = null;
        this.composer = null;
        
        // Camera controls
        this.cameraAngle = 0;
        this.cameraDistance = 150;
        this.cameraHeight = 0;
        
        // Effects
        this.particles = [];
        this.lights = {};
        
        this.init();
    }
    
    async init() {
        console.log('UltraViewport: Initializing cinematic experience...');
        
        await this.setupScene();
        this.setupLighting();
        this.createEnvironment();
        this.createSuit();
        this.setupPostProcessing();
        this.setupControls();
        
        // Start render loop
        this.animate();
        
        console.log('UltraViewport: Ready for action!');
    }
    
    async setupScene() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x000011, 0.001);
        
        // Camera with cinematic FOV
        this.camera = new THREE.PerspectiveCamera(
            45,
            window.innerWidth / window.innerHeight,
            0.1,
            2000
        );
        this.camera.position.set(100, 30, 100);
        this.camera.lookAt(0, 0, 0);
        
        // High-quality renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        
        // Enable realistic rendering features
        this.renderer.physicallyCorrectLights = true;
    }
    
    setupLighting() {
        // Dramatic cinematic lighting
        
        // Ambient - very subtle
        const ambientLight = new THREE.AmbientLight(0x020408, 0.2);
        this.scene.add(ambientLight);
        
        // Key light - main dramatic light
        const keyLight = new THREE.DirectionalLight(0xffffff, 3);
        keyLight.position.set(50, 100, 50);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 4096;
        keyLight.shadow.mapSize.height = 4096;
        keyLight.shadow.camera.near = 0.1;
        keyLight.shadow.camera.far = 500;
        keyLight.shadow.camera.left = -100;
        keyLight.shadow.camera.right = 100;
        keyLight.shadow.camera.top = 100;
        keyLight.shadow.camera.bottom = -100;
        keyLight.shadow.bias = -0.0005;
        this.scene.add(keyLight);
        this.lights.key = keyLight;
        
        // Fill light - softer blue tone
        const fillLight = new THREE.DirectionalLight(0x4080ff, 1);
        fillLight.position.set(-100, 50, -50);
        this.scene.add(fillLight);
        this.lights.fill = fillLight;
        
        // Rim light - dramatic edge lighting
        const rimLight = new THREE.SpotLight(0xff6600, 2, 200, Math.PI / 4, 0.5, 1);
        rimLight.position.set(-80, 80, -80);
        rimLight.target.position.set(0, 0, 0);
        this.scene.add(rimLight);
        this.scene.add(rimLight.target);
        this.lights.rim = rimLight;
        
        // Ground bounce light
        const bounceLight = new THREE.DirectionalLight(0x443322, 0.5);
        bounceLight.position.set(0, -50, 0);
        bounceLight.target.position.set(0, 0, 0);
        this.scene.add(bounceLight);
        this.scene.add(bounceLight.target);
        
        // Arc reactor and eye lights will be added by the suit
    }
    
    createEnvironment() {
        // Create a dramatic environment
        
        // Ground plane with reflections
        const groundGeo = new THREE.PlaneGeometry(1000, 1000);
        const groundMat = new THREE.MeshPhysicalMaterial({
            color: 0x050505,
            metalness: 0.9,
            roughness: 0.1,
            envMapIntensity: 0.5
        });
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -50;
        ground.receiveShadow = true;
        this.scene.add(ground);
        
        // Atmospheric particles
        this.createAtmosphericParticles();
        
        // Skybox or gradient background
        this.createSkybox();
    }
    
    createSkybox() {
        // Create gradient sky
        const skyGeo = new THREE.SphereGeometry(800, 32, 32);
        const skyMat = new THREE.ShaderMaterial({
            uniforms: {
                topColor: { value: new THREE.Color(0x000033) },
                bottomColor: { value: new THREE.Color(0x000000) },
                offset: { value: 100 },
                exponent: { value: 0.6 }
            },
            vertexShader: `
                varying vec3 vWorldPosition;
                void main() {
                    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPosition.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 topColor;
                uniform vec3 bottomColor;
                uniform float offset;
                uniform float exponent;
                varying vec3 vWorldPosition;
                void main() {
                    float h = normalize(vWorldPosition + offset).y;
                    gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
                }
            `,
            side: THREE.BackSide
        });
        
        const sky = new THREE.Mesh(skyGeo, skyMat);
        this.scene.add(sky);
        
        // Add stars
        const starsGeo = new THREE.BufferGeometry();
        const starCount = 5000;
        const positions = new Float32Array(starCount * 3);
        
        for (let i = 0; i < starCount * 3; i += 3) {
            const radius = 600 + Math.random() * 200;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            
            positions[i] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i + 1] = radius * Math.sin(phi) * Math.sin(theta);
            positions[i + 2] = radius * Math.cos(phi);
        }
        
        starsGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const starsMat = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.5,
            transparent: true,
            opacity: 0.8
        });
        
        const stars = new THREE.Points(starsGeo, starsMat);
        this.scene.add(stars);
    }
    
    createAtmosphericParticles() {
        // Floating dust particles for atmosphere
        const particleCount = 200;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);
        
        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 200;
            positions[i * 3 + 1] = Math.random() * 100 - 20;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 200;
            
            sizes[i] = Math.random() * 2;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                color: { value: new THREE.Color(0x4488ff) }
            },
            vertexShader: `
                attribute float size;
                varying float vAlpha;
                uniform float time;
                
                void main() {
                    vAlpha = size / 2.0;
                    vec3 pos = position;
                    pos.y += sin(time + position.x * 0.01) * 2.0;
                    
                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_PointSize = size * (300.0 / -mvPosition.z);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform vec3 color;
                varying float vAlpha;
                
                void main() {
                    float dist = distance(gl_PointCoord, vec2(0.5));
                    if (dist > 0.5) discard;
                    
                    float alpha = (1.0 - dist * 2.0) * vAlpha * 0.3;
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        
        const particles = new THREE.Points(geometry, material);
        this.particles.push({ mesh: particles, material: material });
        this.scene.add(particles);
    }
    
    createSuit() {
        // Create the ultra-realistic Iron Man suit
        this.suitSystem = new UltraRealisticIronManSuit(this.scene);
        this.suit = this.suitSystem.getSuit();
        
        // Position suit on pedestal
        this.suit.position.y = -40;
        
        // Add suit-specific lighting
        this.addSuitLighting();
    }
    
    addSuitLighting() {
        // Spotlight focused on suit
        const suitSpot = new THREE.SpotLight(0xffffff, 1, 150, Math.PI / 6, 0.5, 1);
        suitSpot.position.set(0, 100, 50);
        suitSpot.target = this.suit;
        suitSpot.castShadow = true;
        suitSpot.shadow.mapSize.width = 2048;
        suitSpot.shadow.mapSize.height = 2048;
        this.scene.add(suitSpot);
        this.lights.suitSpot = suitSpot;
    }
    
    setupPostProcessing() {
        // Advanced post-processing pipeline
        if (typeof THREE.EffectComposer === 'undefined') {
            console.warn('Post-processing not available');
            return;
        }
        
        this.composer = new THREE.EffectComposer(this.renderer);
        
        // Render pass
        const renderPass = new THREE.RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);
        
        // Bloom for glowing effects
        if (THREE.UnrealBloomPass) {
            const bloomPass = new THREE.UnrealBloomPass(
                new THREE.Vector2(window.innerWidth, window.innerHeight),
                1.5, // strength
                0.4, // radius
                0.85 // threshold
            );
            this.composer.addPass(bloomPass);
        }
        
        // Custom shader pass for additional effects
        const customShader = {
            uniforms: {
                tDiffuse: { value: null },
                time: { value: 0 },
                vignetteAmount: { value: 1.2 },
                scanlineIntensity: { value: 0.05 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform float time;
                uniform float vignetteAmount;
                uniform float scanlineIntensity;
                varying vec2 vUv;
                
                void main() {
                    vec4 color = texture2D(tDiffuse, vUv);
                    
                    // Vignette
                    vec2 uv = vUv * (1.0 - vUv.yx);
                    float vig = uv.x * uv.y * 15.0;
                    vig = pow(vig, vignetteAmount);
                    color.rgb *= vig;
                    
                    // Scanlines
                    float scanline = sin(vUv.y * 800.0 + time * 5.0) * scanlineIntensity;
                    color.rgb += scanline;
                    
                    // Color grading
                    color.rgb = pow(color.rgb, vec3(0.95));
                    color.rgb *= vec3(1.05, 1.0, 0.95); // Slight warm tint
                    
                    gl_FragColor = color;
                }
            `
        };
        
        const customPass = new THREE.ShaderPass(customShader);
        customPass.renderToScreen = true;
        this.composer.addPass(customPass);
        this.customPass = customPass;
    }
    
    setupControls() {
        // Mouse/touch controls for camera orbit
        this.canvas.addEventListener('mousemove', (e) => {
            if (e.buttons === 1) { // Left mouse button
                this.cameraAngle += e.movementX * 0.01;
                this.cameraHeight = Math.max(-50, Math.min(50, 
                    this.cameraHeight - e.movementY * 0.5
                ));
            }
        });
        
        // Zoom with wheel
        this.canvas.addEventListener('wheel', (e) => {
            this.cameraDistance = Math.max(50, Math.min(300, 
                this.cameraDistance + e.deltaY * 0.1
            ));
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            if (this.composer) {
                this.composer.setSize(window.innerWidth, window.innerHeight);
            }
        });
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const time = performance.now() * 0.001;
        const deltaTime = 0.016; // Assume 60fps for now
        
        // Update camera orbit
        this.camera.position.x = Math.cos(this.cameraAngle) * this.cameraDistance;
        this.camera.position.z = Math.sin(this.cameraAngle) * this.cameraDistance;
        this.camera.position.y = 30 + this.cameraHeight;
        this.camera.lookAt(0, 0, 0);
        
        // Rotate suit slowly
        if (this.suit) {
            this.suit.rotation.y += 0.002;
        }
        
        // Update suit system
        if (this.suitSystem) {
            this.suitSystem.update(deltaTime, time);
        }
        
        // Update particles
        this.particles.forEach(particle => {
            if (particle.material.uniforms.time) {
                particle.material.uniforms.time.value = time;
            }
        });
        
        // Update post-processing
        if (this.customPass) {
            this.customPass.uniforms.time.value = time;
        }
        
        // Animate lights
        if (this.lights.rim) {
            this.lights.rim.intensity = 2 + Math.sin(time * 2) * 0.5;
        }
        
        // Render
        if (this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
    }
}

// Replace EnhancedViewport with UltraViewport
window.EnhancedViewport = UltraViewport;