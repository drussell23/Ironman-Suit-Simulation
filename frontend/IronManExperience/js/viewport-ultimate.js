// Ultimate Iron Man Viewport - Combining Ultra-Realistic Suit with Perfect Lighting

class UltimateViewport {
    constructor() {
        this.canvas = document.getElementById('viewport');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.composer = null;
        
        // Suit and controls
        this.suitSystem = null;
        this.environmentSystem = null;
        this.lightingSystem = null;
        this.controls = {
            autoRotate: true,
            rotationSpeed: 0.002,
            cameraRadius: 150,
            cameraHeight: 30,
            cameraAngle: 0
        };
        
        this.init();
    }
    
    async init() {
        console.log('UltimateViewport: Initializing ultimate experience...');
        
        // Setup core components
        this.setupRenderer();
        this.setupScene();
        this.setupCamera();
        this.setupCinematicLighting();
        this.createCinematicEnvironment();
        
        // Create the ultra-realistic suit
        await this.createUltraRealisticSuit();
        
        // Setup post-processing
        this.setupPostProcessing();
        
        // Setup controls
        this.setupInteraction();
        
        // Start animation
        this.animate();
        
        console.log('UltimateViewport: JARVIS ready, sir!');
    }
    
    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false,
            powerPreference: "high-performance"
        });
        
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        
        // Enable shadows
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Tone mapping for realistic colors
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.3;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        
        // Enable physically correct lighting
        this.renderer.physicallyCorrectLights = true;
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        
        // Dark environment with subtle fog
        this.scene.background = new THREE.Color(0x050505);
        this.scene.fog = new THREE.FogExp2(0x050505, 0.002);
    }
    
    setupCamera() {
        this.camera = new THREE.PerspectiveCamera(
            45,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        
        // Initial camera position
        this.updateCameraPosition();
    }
    
    updateCameraPosition() {
        const radius = this.controls.cameraRadius;
        const height = this.controls.cameraHeight;
        const angle = this.controls.cameraAngle;
        
        this.camera.position.x = Math.cos(angle) * radius;
        this.camera.position.z = Math.sin(angle) * radius;
        this.camera.position.y = height;
        
        this.camera.lookAt(0, 20, 0);
    }
    
    setupCinematicLighting() {
        // Try to use cinematic lighting system
        try {
            if (window.CinematicLighting) {
                console.log('Setting up cinematic lighting...');
                this.lightingSystem = new CinematicLighting(this.scene, this.camera);
                return;
            }
        } catch (error) {
            console.warn('Cinematic lighting failed, using basic lighting...', error);
        }
        
        // Fallback to basic lighting
        this.setupBasicLighting();
    }
    
    setupBasicLighting() {
        // 1. Base ambient light - subtle overall illumination
        const ambientLight = new THREE.AmbientLight(0x202030, 0.5);
        this.scene.add(ambientLight);
        
        // 2. Hemisphere light for sky/ground color variation
        const hemiLight = new THREE.HemisphereLight(0x303060, 0x101020, 0.5);
        hemiLight.position.set(0, 100, 0);
        this.scene.add(hemiLight);
        
        // 3. Main key light - strong directional
        const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
        keyLight.position.set(40, 80, 40);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 4096;
        keyLight.shadow.mapSize.height = 4096;
        keyLight.shadow.camera.near = 1;
        keyLight.shadow.camera.far = 200;
        keyLight.shadow.camera.left = -80;
        keyLight.shadow.camera.right = 80;
        keyLight.shadow.camera.top = 80;
        keyLight.shadow.camera.bottom = -80;
        keyLight.shadow.bias = -0.0005;
        keyLight.shadow.normalBias = 0.02;
        this.scene.add(keyLight);
        
        // 4. Fill light - softer, colored
        const fillLight = new THREE.DirectionalLight(0x4080ff, 1.0);
        fillLight.position.set(-60, 40, -30);
        this.scene.add(fillLight);
        
        // 5. Rim/Back light for edge definition
        const rimLight = new THREE.SpotLight(0xff8844, 2.0, 200, Math.PI / 3, 0.5, 1);
        rimLight.position.set(-50, 60, -60);
        rimLight.target.position.set(0, 20, 0);
        this.scene.add(rimLight);
        this.scene.add(rimLight.target);
        
        // 6. Front spot for face/chest emphasis
        const frontSpot = new THREE.SpotLight(0xffffff, 1.5, 150, Math.PI / 6, 0.2, 1);
        frontSpot.position.set(0, 50, 80);
        frontSpot.target.position.set(0, 35, 0);
        frontSpot.castShadow = true;
        frontSpot.shadow.mapSize.width = 2048;
        frontSpot.shadow.mapSize.height = 2048;
        this.scene.add(frontSpot);
        this.scene.add(frontSpot.target);
        
        // 7. Ground bounce light
        const bounceLight = new THREE.DirectionalLight(0x332211, 0.3);
        bounceLight.position.set(0, -50, 0);
        bounceLight.target.position.set(0, 0, 0);
        this.scene.add(bounceLight);
        this.scene.add(bounceLight.target);
        
        // 8. Accent lights for dramatic effect
        const accentLight1 = new THREE.PointLight(0x00aaff, 1, 80);
        accentLight1.position.set(30, 40, 30);
        this.scene.add(accentLight1);
        
        const accentLight2 = new THREE.PointLight(0xff0066, 0.5, 60);
        accentLight2.position.set(-40, 30, -20);
        this.scene.add(accentLight2);
    }
    
    createCinematicEnvironment() {
        // Try to create cinematic environment
        try {
            if (window.CinematicEnvironment) {
                console.log('Creating cinematic environment...');
                this.environmentSystem = new CinematicEnvironment(this.scene);
                return;
            }
        } catch (error) {
            console.warn('Cinematic environment failed, using basic environment...', error);
        }
        
        // Fallback to basic environment
        this.createBasicEnvironment();
    }
    
    createBasicEnvironment() {
        // 1. Reflective floor
        const floorGeometry = new THREE.PlaneGeometry(300, 300);
        const floorMaterial = new THREE.MeshStandardMaterial({
            color: 0x0a0a0a,
            metalness: 0.8,
            roughness: 0.2,
            envMapIntensity: 0.5
        });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -2;
        floor.receiveShadow = true;
        this.scene.add(floor);
        
        // 2. Display platform
        const platformGeometry = new THREE.CylinderGeometry(50, 55, 3, 32);
        const platformMaterial = new THREE.MeshStandardMaterial({
            color: 0x151515,
            metalness: 0.9,
            roughness: 0.1
        });
        const platform = new THREE.Mesh(platformGeometry, platformMaterial);
        platform.position.y = -0.5;
        platform.receiveShadow = true;
        platform.castShadow = true;
        this.scene.add(platform);
        
        // 3. Platform ring light
        const ringGeometry = new THREE.TorusGeometry(52, 1, 8, 32);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: 0x00aaff,
            emissive: 0x00aaff
        });
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 1.5;
        this.scene.add(ring);
        
        // 4. Subtle grid
        const gridHelper = new THREE.GridHelper(300, 60, 0x222222, 0x111111);
        gridHelper.position.y = -1.9;
        this.scene.add(gridHelper);
    }
    
    async createUltraRealisticSuit() {
        // First, let's try the ultimate realistic suit
        try {
            if (window.UltimateRealisticIronManSuit) {
                console.log('Creating ultimate realistic suit...');
                this.suitSystem = new UltimateRealisticIronManSuit(this.scene);
                return;
            }
        } catch (error) {
            console.warn('Ultimate realistic suit failed, trying ultra-realistic...', error);
        }
        
        // Try the ultra-realistic suit
        try {
            if (window.UltraRealisticIronManSuit) {
                console.log('Creating ultra-realistic suit...');
                this.suitSystem = new UltraRealisticIronManSuit(this.scene);
                return;
            }
        } catch (error) {
            console.warn('Ultra-realistic suit failed, falling back...', error);
        }
        
        // Fallback: Create a well-lit visible suit
        this.createFallbackSuit();
    }
    
    createFallbackSuit() {
        console.log('Creating fallback suit with enhanced visibility...');
        
        const suit = new THREE.Group();
        
        // Enhanced materials with better visibility
        const materials = {
            armorRed: new THREE.MeshPhysicalMaterial({
                color: 0xcc0000,
                metalness: 0.8,
                roughness: 0.2,
                clearcoat: 1.0,
                clearcoatRoughness: 0.1,
                emissive: 0x330000,
                emissiveIntensity: 0.1
            }),
            armorGold: new THREE.MeshPhysicalMaterial({
                color: 0xffcc00,
                metalness: 0.9,
                roughness: 0.1,
                clearcoat: 0.8,
                clearcoatRoughness: 0.05,
                emissive: 0x443300,
                emissiveIntensity: 0.15
            }),
            arcReactor: new THREE.MeshPhysicalMaterial({
                color: 0x00ffff,
                emissive: 0x00ffff,
                emissiveIntensity: 3.0,
                metalness: 0.0,
                roughness: 0.0,
                transparent: true,
                opacity: 0.9
            }),
            eyes: new THREE.MeshBasicMaterial({
                color: 0x00ccff,
                emissive: 0x00ccff
            })
        };
        
        // Build suit parts with better proportions
        
        // Helmet
        const helmet = new THREE.Group();
        
        const helmetGeo = new THREE.SphereGeometry(10, 32, 24);
        helmetGeo.scale(1, 1.1, 0.95);
        const helmetMesh = new THREE.Mesh(helmetGeo, materials.armorRed);
        helmetMesh.castShadow = true;
        helmet.add(helmetMesh);
        
        // Faceplate
        const faceplateGeo = new THREE.BoxGeometry(8, 9, 2);
        const faceplate = new THREE.Mesh(faceplateGeo, materials.armorGold);
        faceplate.position.set(0, -1, 9);
        helmet.add(faceplate);
        
        // Eyes
        const eyeGeo = new THREE.BoxGeometry(2.5, 1.2, 0.5);
        const leftEye = new THREE.Mesh(eyeGeo, materials.eyes);
        leftEye.position.set(-3, 1, 10);
        helmet.add(leftEye);
        
        const rightEye = new THREE.Mesh(eyeGeo, materials.eyes);
        rightEye.position.set(3, 1, 10);
        helmet.add(rightEye);
        
        // Eye lights
        const leftEyeLight = new THREE.PointLight(0x00ccff, 0.5, 20);
        leftEyeLight.position.set(-3, 1, 11);
        helmet.add(leftEyeLight);
        
        const rightEyeLight = new THREE.PointLight(0x00ccff, 0.5, 20);
        rightEyeLight.position.set(3, 1, 11);
        helmet.add(rightEyeLight);
        
        helmet.position.y = 55;
        suit.add(helmet);
        
        // Torso
        const torsoGeo = new THREE.BoxGeometry(24, 30, 14);
        torsoGeo.scale(1.1, 1, 1);
        const torso = new THREE.Mesh(torsoGeo, materials.armorRed);
        torso.position.y = 35;
        torso.castShadow = true;
        suit.add(torso);
        
        // Chest plates
        const chestPlates = [
            { x: 0, y: 40, z: 8, w: 10, h: 12 },
            { x: -8, y: 35, z: 7.5, w: 6, h: 10 },
            { x: 8, y: 35, z: 7.5, w: 6, h: 10 }
        ];
        
        chestPlates.forEach(plate => {
            const plateGeo = new THREE.BoxGeometry(plate.w, plate.h, 2);
            const plateMesh = new THREE.Mesh(plateGeo, materials.armorGold);
            plateMesh.position.set(plate.x, plate.y, plate.z);
            suit.add(plateMesh);
        });
        
        // Arc Reactor
        const arcReactorGroup = new THREE.Group();
        
        const reactorRingGeo = new THREE.TorusGeometry(4, 1, 8, 16);
        const reactorRing = new THREE.Mesh(reactorRingGeo, materials.armorGold);
        arcReactorGroup.add(reactorRing);
        
        const reactorCoreGeo = new THREE.SphereGeometry(3, 16, 16);
        const reactorCore = new THREE.Mesh(reactorCoreGeo, materials.arcReactor);
        arcReactorGroup.add(reactorCore);
        
        const reactorLight = new THREE.PointLight(0x00ffff, 2, 50);
        reactorLight.position.z = 2;
        arcReactorGroup.add(reactorLight);
        
        arcReactorGroup.position.set(0, 40, 9);
        suit.add(arcReactorGroup);
        
        // Shoulders
        const shoulderGeo = new THREE.SphereGeometry(8, 16, 12);
        shoulderGeo.scale(1.2, 1, 1);
        
        const leftShoulder = new THREE.Mesh(shoulderGeo, materials.armorRed);
        leftShoulder.position.set(-18, 48, 0);
        leftShoulder.castShadow = true;
        suit.add(leftShoulder);
        
        const rightShoulder = new THREE.Mesh(shoulderGeo, materials.armorRed);
        rightShoulder.position.set(18, 48, 0);
        rightShoulder.castShadow = true;
        suit.add(rightShoulder);
        
        // Arms
        const upperArmGeo = new THREE.CylinderGeometry(5, 6, 20, 8);
        const lowerArmGeo = new THREE.CylinderGeometry(4, 5, 20, 8);
        
        // Left arm
        const leftUpperArm = new THREE.Mesh(upperArmGeo, materials.armorRed);
        leftUpperArm.position.set(-18, 35, 0);
        leftUpperArm.castShadow = true;
        suit.add(leftUpperArm);
        
        const leftLowerArm = new THREE.Mesh(lowerArmGeo, materials.armorRed);
        leftLowerArm.position.set(-18, 20, 0);
        leftLowerArm.castShadow = true;
        suit.add(leftLowerArm);
        
        // Right arm
        const rightUpperArm = new THREE.Mesh(upperArmGeo, materials.armorRed);
        rightUpperArm.position.set(18, 35, 0);
        rightUpperArm.castShadow = true;
        suit.add(rightUpperArm);
        
        const rightLowerArm = new THREE.Mesh(lowerArmGeo, materials.armorRed);
        rightLowerArm.position.set(18, 20, 0);
        rightLowerArm.castShadow = true;
        suit.add(rightLowerArm);
        
        // Hands with repulsors
        const handGeo = new THREE.BoxGeometry(7, 8, 5);
        
        const leftHand = new THREE.Mesh(handGeo, materials.armorGold);
        leftHand.position.set(-18, 10, 0);
        suit.add(leftHand);
        
        const rightHand = new THREE.Mesh(handGeo, materials.armorGold);
        rightHand.position.set(18, 10, 0);
        suit.add(rightHand);
        
        // Repulsors
        const repulsorGeo = new THREE.CylinderGeometry(2.5, 2.5, 1, 16);
        
        const leftRepulsor = new THREE.Mesh(repulsorGeo, materials.arcReactor);
        leftRepulsor.rotation.x = Math.PI / 2;
        leftRepulsor.position.set(-18, 10, 3);
        suit.add(leftRepulsor);
        
        const rightRepulsor = new THREE.Mesh(repulsorGeo, materials.arcReactor);
        rightRepulsor.rotation.x = Math.PI / 2;
        rightRepulsor.position.set(18, 10, 3);
        suit.add(rightRepulsor);
        
        // Waist
        const waistGeo = new THREE.BoxGeometry(20, 10, 12);
        const waist = new THREE.Mesh(waistGeo, materials.armorRed);
        waist.position.y = 20;
        waist.castShadow = true;
        suit.add(waist);
        
        // Belt
        const beltGeo = new THREE.BoxGeometry(22, 3, 13);
        const belt = new THREE.Mesh(beltGeo, materials.armorGold);
        belt.position.y = 20;
        suit.add(belt);
        
        // Legs
        const thighGeo = new THREE.CylinderGeometry(6, 7, 20, 8);
        const shinGeo = new THREE.CylinderGeometry(5, 6, 20, 8);
        
        // Left leg
        const leftThigh = new THREE.Mesh(thighGeo, materials.armorRed);
        leftThigh.position.set(-8, 10, 0);
        leftThigh.castShadow = true;
        suit.add(leftThigh);
        
        const leftShin = new THREE.Mesh(shinGeo, materials.armorRed);
        leftShin.position.set(-8, -5, 0);
        leftShin.castShadow = true;
        suit.add(leftShin);
        
        // Right leg
        const rightThigh = new THREE.Mesh(thighGeo, materials.armorRed);
        rightThigh.position.set(8, 10, 0);
        rightThigh.castShadow = true;
        suit.add(rightThigh);
        
        const rightShin = new THREE.Mesh(shinGeo, materials.armorRed);
        rightShin.position.set(8, -5, 0);
        rightShin.castShadow = true;
        suit.add(rightShin);
        
        // Boots
        const bootGeo = new THREE.BoxGeometry(9, 8, 14);
        
        const leftBoot = new THREE.Mesh(bootGeo, materials.armorRed);
        leftBoot.position.set(-8, -15, 2);
        leftBoot.castShadow = true;
        suit.add(leftBoot);
        
        const rightBoot = new THREE.Mesh(bootGeo, materials.armorRed);
        rightBoot.position.set(8, -15, 2);
        rightBoot.castShadow = true;
        suit.add(rightBoot);
        
        // Thrusters
        const thrusterGeo = new THREE.CylinderGeometry(2, 3, 4, 8);
        
        const leftThruster = new THREE.Mesh(thrusterGeo, materials.armorGold);
        leftThruster.position.set(-8, -19, -4);
        leftThruster.rotation.x = Math.PI / 2;
        suit.add(leftThruster);
        
        const rightThruster = new THREE.Mesh(thrusterGeo, materials.armorGold);
        rightThruster.position.set(8, -19, -4);
        rightThruster.rotation.x = Math.PI / 2;
        suit.add(rightThruster);
        
        // Add panel lines
        this.addPanelLines(suit);
        
        // Position suit
        suit.position.y = 20;
        
        // Store reference
        this.suit = suit;
        this.scene.add(suit);
        
        // Store for animation
        this.suitParts = {
            arcReactor: arcReactorGroup,
            helmet: helmet,
            leftEye: leftEye,
            rightEye: rightEye
        };
    }
    
    addPanelLines(group) {
        // Add edge geometry for panel lines
        group.traverse((child) => {
            if (child.isMesh && child.geometry) {
                const edges = new THREE.EdgesGeometry(child.geometry, 30);
                const lineMaterial = new THREE.LineBasicMaterial({ 
                    color: 0x000000,
                    transparent: true,
                    opacity: 0.3
                });
                const lineSegments = new THREE.LineSegments(edges, lineMaterial);
                lineSegments.position.copy(child.position);
                lineSegments.rotation.copy(child.rotation);
                lineSegments.scale.copy(child.scale);
                group.add(lineSegments);
            }
        });
    }
    
    setupPostProcessing() {
        // Simple bloom effect for glowing parts
        if (typeof THREE.EffectComposer === 'undefined') {
            console.log('Post-processing not available, skipping...');
            return;
        }
        
        this.composer = new THREE.EffectComposer(this.renderer);
        
        const renderPass = new THREE.RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);
        
        // Bloom pass
        if (THREE.UnrealBloomPass) {
            const bloomPass = new THREE.UnrealBloomPass(
                new THREE.Vector2(window.innerWidth, window.innerHeight),
                1.0,  // strength
                0.4,  // radius  
                0.85  // threshold
            );
            this.composer.addPass(bloomPass);
        }
    }
    
    setupInteraction() {
        // Mouse controls
        let mouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        
        this.canvas.addEventListener('mousedown', (e) => {
            mouseDown = true;
            mouseX = e.clientX;
            mouseY = e.clientY;
            this.controls.autoRotate = false;
        });
        
        this.canvas.addEventListener('mouseup', () => {
            mouseDown = false;
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (mouseDown) {
                const deltaX = e.clientX - mouseX;
                const deltaY = e.clientY - mouseY;
                
                this.controls.cameraAngle -= deltaX * 0.01;
                this.controls.cameraHeight = Math.max(10, Math.min(80, 
                    this.controls.cameraHeight + deltaY * 0.5
                ));
                
                mouseX = e.clientX;
                mouseY = e.clientY;
            }
        });
        
        // Mouse wheel zoom
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.controls.cameraRadius = Math.max(80, Math.min(250,
                this.controls.cameraRadius + e.deltaY * 0.1
            ));
        });
        
        // Double click to toggle auto-rotate
        this.canvas.addEventListener('dblclick', () => {
            this.controls.autoRotate = !this.controls.autoRotate;
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
        
        // Keyboard controls
        window.addEventListener('keydown', (e) => {
            switch(e.key) {
                case ' ':
                    this.controls.autoRotate = !this.controls.autoRotate;
                    break;
                case 'r':
                case 'R':
                    // Reset camera
                    this.controls.cameraRadius = 150;
                    this.controls.cameraHeight = 30;
                    this.controls.cameraAngle = 0;
                    break;
            }
        });
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const time = performance.now() * 0.001;
        const deltaTime = 0.016;
        
        // Update camera
        if (this.controls.autoRotate) {
            this.controls.cameraAngle += this.controls.rotationSpeed;
        }
        this.updateCameraPosition();
        
        // Animate suit parts
        if (this.suitParts) {
            // Pulse arc reactor
            if (this.suitParts.arcReactor) {
                const pulse = Math.sin(time * 3) * 0.1 + 0.9;
                this.suitParts.arcReactor.scale.set(pulse, pulse, pulse);
                
                // Rotate arc reactor core
                const core = this.suitParts.arcReactor.children[1];
                if (core) {
                    core.rotation.x = time * 0.5;
                    core.rotation.y = time * 0.7;
                }
            }
            
            // Flicker eyes occasionally
            if (Math.random() < 0.01) {
                const flicker = Math.random() * 0.5 + 0.5;
                if (this.suitParts.leftEye) {
                    this.suitParts.leftEye.material.emissiveIntensity = flicker;
                }
                if (this.suitParts.rightEye) {
                    this.suitParts.rightEye.material.emissiveIntensity = flicker;
                }
            }
        }
        
        // Update suit system if using ultra-realistic
        if (this.suitSystem && this.suitSystem.update) {
            this.suitSystem.update(deltaTime, time);
        }
        
        // Update environment system
        if (this.environmentSystem && this.environmentSystem.update) {
            this.environmentSystem.update(deltaTime, time);
        }
        
        // Update lighting system
        if (this.lightingSystem && this.lightingSystem.update) {
            this.lightingSystem.update(deltaTime, time);
        }
        
        // Render
        if (this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
    }
}

// Replace EnhancedViewport
window.EnhancedViewport = UltimateViewport;