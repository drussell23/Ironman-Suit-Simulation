// 3D Viewport for Iron Man Experience

class Viewport {
    constructor() {
        this.canvas = Utils.$('#viewport');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Objects
        this.suit = null;
        this.environment = null;
        this.particles = [];
        this.targets = [];
        
        // Animation
        this.clock = null;
        this.animationId = null;
        
        // State
        this.isFlying = false;
        this.velocity = new THREE.Vector3(0, 0, 0);
        this.acceleration = new THREE.Vector3(0, 0, 0);
        
        this.init();
    }
    
    init() {
        // Check if Three.js is loaded
        if (typeof THREE === 'undefined') {
            console.warn('Three.js not loaded. Using 2D canvas fallback.');
            this.init2DFallback();
            return;
        }
        
        this.setupScene();
        this.setupLights();
        this.createEnvironment();
        this.createSuit();
        this.setupControls();
        this.setupEventListeners();
        
        // Start render loop
        this.animate();
    }
    
    setupScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.Fog(0x000000, 100, 5000);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            10000
        );
        this.camera.position.set(0, 100, -200);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Clock for animations
        this.clock = new THREE.Clock();
    }
    
    setupLights() {
        // Ambient light
        const ambient = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambient);
        
        // Directional light (sun)
        const directional = new THREE.DirectionalLight(0xffffff, 1);
        directional.position.set(100, 200, 50);
        directional.castShadow = true;
        directional.shadow.camera.near = 0.1;
        directional.shadow.camera.far = 500;
        directional.shadow.camera.left = -200;
        directional.shadow.camera.right = 200;
        directional.shadow.camera.top = 200;
        directional.shadow.camera.bottom = -200;
        this.scene.add(directional);
        
        // Point lights for city effect
        for (let i = 0; i < 10; i++) {
            const light = new THREE.PointLight(
                new THREE.Color().setHSL(Math.random(), 0.7, 0.5),
                1,
                200
            );
            light.position.set(
                (Math.random() - 0.5) * 1000,
                Math.random() * 100,
                (Math.random() - 0.5) * 1000
            );
            this.scene.add(light);
        }
    }
    
    createEnvironment() {
        // Ground plane
        const groundGeometry = new THREE.PlaneGeometry(5000, 5000);
        const groundMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x222222,
            side: THREE.DoubleSide
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);
        
        // Grid helper
        const grid = new THREE.GridHelper(2000, 50, 0x00a8ff, 0x004466);
        this.scene.add(grid);
        
        // Buildings
        const buildingGeometry = new THREE.BoxGeometry(1, 1, 1);
        const buildingGroup = new THREE.Group();
        
        for (let i = 0; i < 50; i++) {
            const material = new THREE.MeshPhongMaterial({
                color: new THREE.Color().setHSL(0.6, 0.1, 0.2 + Math.random() * 0.2),
                emissive: 0x001122,
                emissiveIntensity: 0.2
            });
            
            const building = new THREE.Mesh(buildingGeometry, material);
            const scale = {
                x: 20 + Math.random() * 50,
                y: 50 + Math.random() * 200,
                z: 20 + Math.random() * 50
            };
            
            building.scale.set(scale.x, scale.y, scale.z);
            building.position.set(
                (Math.random() - 0.5) * 1000,
                scale.y / 2,
                (Math.random() - 0.5) * 1000
            );
            building.castShadow = true;
            building.receiveShadow = true;
            
            buildingGroup.add(building);
        }
        
        this.scene.add(buildingGroup);
        this.environment = buildingGroup;
        
        // Skybox
        this.createSkybox();
        
        // Particles for atmosphere
        this.createParticles();
    }
    
    createSkybox() {
        const skyGeometry = new THREE.SphereGeometry(5000, 32, 32);
        const skyMaterial = new THREE.ShaderMaterial({
            uniforms: {
                topColor: { value: new THREE.Color(0x000033) },
                bottomColor: { value: new THREE.Color(0x000000) },
                offset: { value: 33 },
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
        
        const sky = new THREE.Mesh(skyGeometry, skyMaterial);
        this.scene.add(sky);
    }
    
    createParticles() {
        const particleCount = 1000;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 2000;
            positions[i + 1] = Math.random() * 1000;
            positions[i + 2] = (Math.random() - 0.5) * 2000;
            
            const color = new THREE.Color();
            color.setHSL(0.6, 0.5, 0.5 + Math.random() * 0.5);
            colors[i] = color.r;
            colors[i + 1] = color.g;
            colors[i + 2] = color.b;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 2,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
        });
        
        const particles = new THREE.Points(geometry, material);
        this.scene.add(particles);
        this.particles.push(particles);
    }
    
    createSuit() {
        // Simplified Iron Man suit representation
        const suitGroup = new THREE.Group();
        
        // Body
        const bodyGeometry = new THREE.CylinderGeometry(10, 8, 30, 8);
        const bodyMaterial = new THREE.MeshPhongMaterial({
            color: 0xff0000,
            emissive: 0x441111,
            shininess: 100,
            metalness: 0.8
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        suitGroup.add(body);
        
        // Arc reactor
        const reactorGeometry = new THREE.CircleGeometry(3, 16);
        const reactorMaterial = new THREE.MeshBasicMaterial({
            color: 0x00a8ff,
            emissive: 0x00a8ff,
            emissiveIntensity: 2
        });
        const reactor = new THREE.Mesh(reactorGeometry, reactorMaterial);
        reactor.position.z = 8.1;
        suitGroup.add(reactor);
        
        // Repulsor glow
        const glowGeometry = new THREE.SphereGeometry(5, 16, 16);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0x00a8ff,
            transparent: true,
            opacity: 0.3
        });
        const leftRepulsor = new THREE.Mesh(glowGeometry, glowMaterial);
        leftRepulsor.position.set(-15, -15, 0);
        suitGroup.add(leftRepulsor);
        
        const rightRepulsor = new THREE.Mesh(glowGeometry, glowMaterial);
        rightRepulsor.position.set(15, -15, 0);
        suitGroup.add(rightRepulsor);
        
        // Add suit to scene
        suitGroup.position.y = 100;
        this.scene.add(suitGroup);
        this.suit = suitGroup;
    }
    
    setupControls() {
        // Simple camera follow controls
        this.controls = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            boost: false,
            mouseX: 0,
            mouseY: 0
        };
    }
    
    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        document.addEventListener('keyup', (e) => this.onKeyUp(e));
        
        // Mouse controls
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
    }
    
    onWindowResize() {
        if (this.camera && this.renderer) {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        }
    }
    
    onKeyDown(e) {
        switch(e.key.toLowerCase()) {
            case 'w': this.controls.forward = true; break;
            case 's': this.controls.backward = true; break;
            case 'a': this.controls.left = true; break;
            case 'd': this.controls.right = true; break;
            case 'q': this.controls.up = true; break;
            case 'e': this.controls.down = true; break;
            case 'shift': this.controls.boost = true; break;
        }
    }
    
    onKeyUp(e) {
        switch(e.key.toLowerCase()) {
            case 'w': this.controls.forward = false; break;
            case 's': this.controls.backward = false; break;
            case 'a': this.controls.left = false; break;
            case 'd': this.controls.right = false; break;
            case 'q': this.controls.up = false; break;
            case 'e': this.controls.down = false; break;
            case 'shift': this.controls.boost = false; break;
        }
    }
    
    onMouseMove(e) {
        this.controls.mouseX = (e.clientX / window.innerWidth) * 2 - 1;
        this.controls.mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
    }
    
    updateMovement(delta) {
        if (!this.suit) return;
        
        const speed = this.controls.boost ? 500 : 200;
        const moveSpeed = speed * delta;
        
        // Update acceleration based on controls
        this.acceleration.set(0, 0, 0);
        
        if (this.controls.forward) this.acceleration.z += 1;
        if (this.controls.backward) this.acceleration.z -= 1;
        if (this.controls.left) this.acceleration.x -= 1;
        if (this.controls.right) this.acceleration.x += 1;
        if (this.controls.up) this.acceleration.y += 1;
        if (this.controls.down) this.acceleration.y -= 1;
        
        // Normalize and scale acceleration
        if (this.acceleration.length() > 0) {
            this.acceleration.normalize().multiplyScalar(moveSpeed);
            this.isFlying = true;
        } else {
            this.isFlying = false;
        }
        
        // Apply acceleration to velocity
        this.velocity.add(this.acceleration);
        
        // Apply drag
        this.velocity.multiplyScalar(0.95);
        
        // Update position
        this.suit.position.add(this.velocity);
        
        // Rotate based on mouse
        this.suit.rotation.y = -this.controls.mouseX * Math.PI * 0.5;
        this.suit.rotation.x = this.controls.mouseY * Math.PI * 0.25;
        
        // Update camera to follow suit
        const cameraOffset = new THREE.Vector3(0, 50, -150);
        cameraOffset.applyQuaternion(this.suit.quaternion);
        this.camera.position.lerp(
            this.suit.position.clone().add(cameraOffset),
            0.1
        );
        this.camera.lookAt(this.suit.position);
        
        // Update HUD values
        const altitude = Math.max(0, this.suit.position.y);
        const velocity = this.velocity.length() * 3.6; // Convert to km/h
        
        if (window.HUD) {
            window.HUD.updateAltitude(altitude);
            window.HUD.updateVelocity(velocity);
        }
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        const delta = this.clock.getDelta();
        
        // Update movement
        this.updateMovement(delta);
        
        // Animate particles
        this.particles.forEach(particle => {
            particle.rotation.y += delta * 0.1;
        });
        
        // Render scene
        this.renderer.render(this.scene, this.camera);
    }
    
    // 2D Canvas fallback
    init2DFallback() {
        const ctx = this.canvas.getContext('2d');
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        
        // Simple 2D visualization
        const render2D = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Draw grid
            ctx.strokeStyle = 'rgba(0, 168, 255, 0.2)';
            ctx.lineWidth = 1;
            
            const gridSize = 50;
            for (let x = 0; x < this.canvas.width; x += gridSize) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, this.canvas.height);
                ctx.stroke();
            }
            
            for (let y = 0; y < this.canvas.height; y += gridSize) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(this.canvas.width, y);
                ctx.stroke();
            }
            
            // Draw horizon line
            const horizon = this.canvas.height * 0.6;
            ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, horizon);
            ctx.lineTo(this.canvas.width, horizon);
            ctx.stroke();
            
            // Continue animation
            requestAnimationFrame(render2D);
        };
        
        render2D();
    }
    
    // Public methods
    addTarget(position) {
        if (!this.scene) return;
        
        const geometry = new THREE.SphereGeometry(10, 16, 16);
        const material = new THREE.MeshPhongMaterial({
            color: 0xff0000,
            emissive: 0xff0000,
            emissiveIntensity: 0.5
        });
        
        const target = new THREE.Mesh(geometry, material);
        target.position.copy(position);
        
        this.scene.add(target);
        this.targets.push(target);
    }
    
    fireRepulsor() {
        if (!this.suit) return;
        
        // Create repulsor beam
        const geometry = new THREE.CylinderGeometry(2, 5, 100, 8);
        const material = new THREE.MeshBasicMaterial({
            color: 0x00a8ff,
            transparent: true,
            opacity: 0.8
        });
        
        const beam = new THREE.Mesh(geometry, material);
        beam.position.copy(this.suit.position);
        beam.rotation.copy(this.suit.rotation);
        beam.translateZ(50);
        
        this.scene.add(beam);
        
        // Animate and remove
        const animateBeam = () => {
            beam.translateZ(10);
            beam.material.opacity *= 0.95;
            
            if (beam.material.opacity > 0.01) {
                requestAnimationFrame(animateBeam);
            } else {
                this.scene.remove(beam);
            }
        };
        
        animateBeam();
    }
}

// Make Viewport globally available
window.Viewport = Viewport;