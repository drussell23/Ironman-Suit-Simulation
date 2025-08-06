// Enhanced 3D Viewport for Iron Man Experience
// Advanced rendering, physics, and visual effects

class EnhancedViewport {
    constructor() {
        this.canvas = Utils.$('#viewport');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.composer = null;
        this.controls = null;
        
        // Objects
        this.suit = null;
        this.suitParts = {};
        this.environment = null;
        this.particles = [];
        this.targets = [];
        this.effects = {};
        
        // Enhanced systems
        this.materials = null;
        this.suitBuilder = null;
        this.lighting = null;
        this.postProcessing = null;
        
        // Animation
        this.clock = new THREE.Clock();
        this.animationMixer = null;
        this.animations = {};
        
        // Physics
        this.velocity = new THREE.Vector3(0, 0, 0);
        this.acceleration = new THREE.Vector3(0, 0, 0);
        this.rotation = new THREE.Euler(0, 0, 0);
        
        // State
        this.state = {
            isFlying: false,
            isAssembling: false,
            powerLevel: 100,
            altitude: 0,
            speed: 0,
            boost: false,
            combatMode: false
        };
        
        // Camera modes
        this.cameraMode = 'third-person'; // 'third-person', 'first-person', 'cinematic'
        this.cameraOffset = new THREE.Vector3(0, 100, -300);
        
        this.init();
    }
    
    async init() {
        if (typeof THREE === 'undefined') {
            console.warn('Three.js not loaded. Using 2D canvas fallback.');
            this.init2DFallback();
            return;
        }
        
        await this.setupScene();
        await this.initEnhancedSystems();
        this.createEnvironment();
        await this.createDetailedSuit();
        this.setupControls();
        this.setupPostProcessing();
        this.setupEventListeners();
        
        // Start render loop
        this.animate();
    }
    
    async setupScene() {
        // Scene with advanced fog
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x000011, 0.0002);
        
        // Camera with cinematic settings
        this.camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            0.1,
            20000
        );
        this.camera.position.set(0, 200, -500);
        this.camera.lookAt(0, 100, 0);
        
        // Advanced renderer
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
        this.renderer.toneMappingExposure = 1.0;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
    }
    
    async initEnhancedSystems() {
        // Initialize PBR materials
        this.materials = new IronManMaterials();
        await this.materials.init();
        
        // Initialize suit builder
        this.suitBuilder = new IronManSuitBuilder(this.materials);
        
        // Initialize enhanced lighting
        this.lighting = new EnhancedLighting(this.scene);
    }
    
    createCityLights() {
        const cityLights = new THREE.Group();
        
        // Create volumetric light beams
        for (let i = 0; i < 20; i++) {
            const color = new THREE.Color().setHSL(
                0.1 + Math.random() * 0.1,
                0.8,
                0.5 + Math.random() * 0.3
            );
            
            const light = new THREE.SpotLight(color, 2, 1000, Math.PI / 6, 0.5, 1);
            light.position.set(
                (Math.random() - 0.5) * 2000,
                50 + Math.random() * 200,
                (Math.random() - 0.5) * 2000
            );
            light.target.position.set(
                light.position.x,
                0,
                light.position.z
            );
            
            cityLights.add(light);
            cityLights.add(light.target);
        }
        
        this.scene.add(cityLights);
        this.effects.cityLights = cityLights;
    }
    
    createEnvironment() {
        // Create detailed city environment
        const city = new THREE.Group();
        
        // Ground with reflective material
        const groundGeometry = new THREE.PlaneGeometry(10000, 10000);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x111122,
            roughness: 0.8,
            metalness: 0.2,
            envMapIntensity: 0.5
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        city.add(ground);
        
        // Animated grid
        this.createAnimatedGrid();
        
        // Detailed buildings with windows
        this.createDetailedBuildings(city);
        
        // Sky dome with stars
        this.createAdvancedSkybox();
        
        // Atmospheric particles
        this.createAtmosphericEffects();
        
        this.scene.add(city);
        this.environment = city;
    }
    
    createAnimatedGrid() {
        const gridHelper = new THREE.GridHelper(4000, 100, 0x00a8ff, 0x003366);
        gridHelper.material.opacity = 0.3;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);
        
        // Animated scanning grid
        const scanGrid = gridHelper.clone();
        scanGrid.material = scanGrid.material.clone();
        scanGrid.material.opacity = 0.1;
        this.scene.add(scanGrid);
        
        this.effects.scanGrid = scanGrid;
    }
    
    createDetailedBuildings(cityGroup) {
        const buildingGeometries = [
            new THREE.BoxGeometry(1, 1, 1),
            new THREE.CylinderGeometry(0.5, 0.5, 1, 8),
            new THREE.ConeGeometry(0.5, 1, 4)
        ];
        
        for (let i = 0; i < 100; i++) {
            const geometry = buildingGeometries[Math.floor(Math.random() * buildingGeometries.length)];
            
            // Building material with emissive windows
            const material = new THREE.MeshStandardMaterial({
                color: new THREE.Color().setHSL(0.6, 0.1, 0.2 + Math.random() * 0.1),
                roughness: 0.7,
                metalness: 0.3,
                emissive: new THREE.Color(0x001144),
                emissiveIntensity: 0.2
            });
            
            const building = new THREE.Mesh(geometry, material);
            const scale = {
                x: 30 + Math.random() * 70,
                y: 100 + Math.random() * 400,
                z: 30 + Math.random() * 70
            };
            
            building.scale.set(scale.x, scale.y, scale.z);
            building.position.set(
                (Math.random() - 0.5) * 3000,
                scale.y / 2,
                (Math.random() - 0.5) * 3000
            );
            building.castShadow = true;
            building.receiveShadow = true;
            
            // Add window lights
            this.addBuildingWindows(building, scale);
            
            cityGroup.add(building);
        }
    }
    
    addBuildingWindows(building, scale) {
        const windowsGroup = new THREE.Group();
        const windowSize = 5;
        const windowSpacing = 15;
        
        // Create window texture
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        
        // Draw window pattern
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, 64, 64);
        ctx.fillStyle = '#ffaa00';
        ctx.fillRect(8, 8, 48, 48);
        
        const windowTexture = new THREE.CanvasTexture(canvas);
        windowTexture.repeat.set(scale.x / windowSpacing, scale.y / windowSpacing);
        windowTexture.wrapS = THREE.RepeatWrapping;
        windowTexture.wrapT = THREE.RepeatWrapping;
        
        const windowMaterial = new THREE.MeshBasicMaterial({
            map: windowTexture,
            transparent: true,
            opacity: 0.8
        });
        
        // Apply windows to building sides
        const windowMesh = new THREE.Mesh(
            new THREE.PlaneGeometry(scale.x * 0.98, scale.y * 0.98),
            windowMaterial
        );
        windowMesh.position.z = scale.z / 2 + 0.1;
        windowsGroup.add(windowMesh);
        
        building.add(windowsGroup);
    }
    
    createAdvancedSkybox() {
        // Create star field
        const starsGeometry = new THREE.BufferGeometry();
        const starsCount = 10000;
        const positions = new Float32Array(starsCount * 3);
        const colors = new Float32Array(starsCount * 3);
        const sizes = new Float32Array(starsCount);
        
        for (let i = 0; i < starsCount; i++) {
            const i3 = i * 3;
            const radius = 5000 + Math.random() * 5000;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            
            positions[i3] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
            positions[i3 + 2] = radius * Math.cos(phi);
            
            const color = new THREE.Color();
            color.setHSL(0.6, 0.2, 0.7 + Math.random() * 0.3);
            colors[i3] = color.r;
            colors[i3 + 1] = color.g;
            colors[i3 + 2] = color.b;
            
            sizes[i] = Math.random() * 2;
        }
        
        starsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        starsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        starsGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        const starsMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 }
            },
            vertexShader: `
                attribute float size;
                varying vec3 vColor;
                uniform float time;
                
                void main() {
                    vColor = color;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    float twinkle = sin(time + position.x * 0.1) * 0.5 + 0.5;
                    gl_PointSize = size * (300.0 / -mvPosition.z) * (0.5 + twinkle * 0.5);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                
                void main() {
                    float dist = distance(gl_PointCoord, vec2(0.5));
                    if (dist > 0.5) discard;
                    
                    float intensity = 1.0 - dist * 2.0;
                    intensity = pow(intensity, 3.0);
                    
                    gl_FragColor = vec4(vColor * intensity, intensity);
                }
            `,
            vertexColors: true,
            transparent: true,
            blending: THREE.AdditiveBlending
        });
        
        const stars = new THREE.Points(starsGeometry, starsMaterial);
        this.scene.add(stars);
        this.effects.stars = stars;
    }
    
    createAtmosphericEffects() {
        // Fog particles
        const fogGeometry = new THREE.BufferGeometry();
        const fogCount = 500;
        const fogPositions = new Float32Array(fogCount * 3);
        const fogOpacity = new Float32Array(fogCount);
        
        for (let i = 0; i < fogCount; i++) {
            fogPositions[i * 3] = (Math.random() - 0.5) * 3000;
            fogPositions[i * 3 + 1] = Math.random() * 500;
            fogPositions[i * 3 + 2] = (Math.random() - 0.5) * 3000;
            fogOpacity[i] = Math.random() * 0.5;
        }
        
        fogGeometry.setAttribute('position', new THREE.BufferAttribute(fogPositions, 3));
        fogGeometry.setAttribute('opacity', new THREE.BufferAttribute(fogOpacity, 1));
        
        const fogMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                fogColor: { value: new THREE.Color(0x0040aa) }
            },
            vertexShader: `
                attribute float opacity;
                varying float vOpacity;
                uniform float time;
                
                void main() {
                    vOpacity = opacity;
                    vec3 pos = position;
                    pos.x += sin(time * 0.1 + position.y * 0.01) * 20.0;
                    pos.z += cos(time * 0.1 + position.y * 0.01) * 20.0;
                    
                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_PointSize = 100.0 * (1.0 / -mvPosition.z);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform vec3 fogColor;
                varying float vOpacity;
                
                void main() {
                    float dist = distance(gl_PointCoord, vec2(0.5));
                    if (dist > 0.5) discard;
                    
                    float intensity = 1.0 - dist * 2.0;
                    intensity = pow(intensity, 2.0);
                    
                    gl_FragColor = vec4(fogColor, vOpacity * intensity * 0.3);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        
        const fog = new THREE.Points(fogGeometry, fogMaterial);
        this.scene.add(fog);
        this.effects.fog = fog;
    }
    
    async createDetailedSuit() {
        // Create highly detailed Iron Man suit using the enhanced suit builder
        this.suit = this.suitBuilder.buildSuit();
        
        // Position suit
        this.suit.position.set(0, 100, 0);
        this.scene.add(this.suit);
        
        // Attach lights to suit
        this.lighting.attachToSuit(this.suit);
        
        // Setup animation mixer
        this.animationMixer = new THREE.AnimationMixer(this.suit);
        
        // Create suit animations
        this.createSuitAnimations();
        
        // Store suit parts reference
        this.suitParts = this.suitBuilder.parts;
    }
    
    createTorso() {
        const torsoGroup = new THREE.Group();
        
        // Main body with custom geometry
        const bodyShape = new THREE.Shape();
        bodyShape.moveTo(0, -30);
        bodyShape.bezierCurveTo(15, -30, 20, -20, 20, 0);
        bodyShape.bezierCurveTo(20, 20, 15, 30, 0, 30);
        bodyShape.bezierCurveTo(-15, 30, -20, 20, -20, 0);
        bodyShape.bezierCurveTo(-20, -20, -15, -30, 0, -30);
        
        const extrudeSettings = {
            depth: 15,
            bevelEnabled: true,
            bevelSegments: 5,
            steps: 2,
            bevelSize: 2,
            bevelThickness: 1
        };
        
        const bodyGeometry = new THREE.ExtrudeGeometry(bodyShape, extrudeSettings);
        const bodyMaterial = this.createSuitMaterial(0xcc0000, 0xff0000);
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.castShadow = true;
        body.receiveShadow = true;
        torsoGroup.add(body);
        
        // Add panel lines
        this.addPanelLines(torsoGroup, bodyGeometry);
        
        // Add chest details
        this.addChestDetails(torsoGroup);
        
        return torsoGroup;
    }
    
    createHelmet() {
        const helmetGroup = new THREE.Group();
        
        // Main helmet shape
        const helmetGeometry = new THREE.SphereGeometry(15, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.75);
        const helmetMaterial = this.createSuitMaterial(0xcc0000, 0xff0000);
        const helmet = new THREE.Mesh(helmetGeometry, helmetMaterial);
        helmet.position.y = 40;
        helmet.castShadow = true;
        helmetGroup.add(helmet);
        
        // Faceplate
        const faceplateGeometry = new THREE.SphereGeometry(14.5, 16, 16, 0, Math.PI * 2, Math.PI * 0.25, Math.PI * 0.5);
        const faceplateMaterial = this.createSuitMaterial(0xffaa00, 0xffdd00, 0.9, 0.1);
        const faceplate = new THREE.Mesh(faceplateGeometry, faceplateMaterial);
        faceplate.position.y = 40;
        helmetGroup.add(faceplate);
        
        // Eye slits
        const eyeGeometry = new THREE.BoxGeometry(5, 2, 1);
        const eyeMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ffff,
            emissive: 0x00ffff,
            emissiveIntensity: 2,
            metalness: 1,
            roughness: 0
        });
        
        const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        leftEye.position.set(-5, 42, -14);
        helmetGroup.add(leftEye);
        
        const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        rightEye.position.set(5, 42, -14);
        helmetGroup.add(rightEye);
        
        // Add glow effect to eyes
        this.addGlowEffect(leftEye, 0x00ffff, 0.5);
        this.addGlowEffect(rightEye, 0x00ffff, 0.5);
        
        return helmetGroup;
    }
    
    createArm(side) {
        const armGroup = new THREE.Group();
        const xOffset = side === 'left' ? -25 : 25;
        
        // Upper arm
        const upperArmGeometry = new THREE.CylinderGeometry(5, 6, 25, 8);
        const upperArmMaterial = this.createSuitMaterial(0xcc0000, 0xff0000);
        const upperArm = new THREE.Mesh(upperArmGeometry, upperArmMaterial);
        upperArm.position.set(xOffset, 10, 0);
        upperArm.castShadow = true;
        armGroup.add(upperArm);
        
        // Elbow joint
        const elbowGeometry = new THREE.SphereGeometry(6, 16, 16);
        const elbowMaterial = this.createSuitMaterial(0x888888, 0xaaaaaa, 1, 0);
        const elbow = new THREE.Mesh(elbowGeometry, elbowMaterial);
        elbow.position.set(xOffset, -2, 0);
        armGroup.add(elbow);
        
        // Lower arm
        const lowerArmGeometry = new THREE.CylinderGeometry(4, 5, 25, 8);
        const lowerArmMaterial = this.createSuitMaterial(0xcc0000, 0xff0000);
        const lowerArm = new THREE.Mesh(lowerArmGeometry, lowerArmMaterial);
        lowerArm.position.set(xOffset, -15, 0);
        lowerArm.castShadow = true;
        armGroup.add(lowerArm);
        
        // Hand with repulsor
        const handGroup = this.createHand(side);
        handGroup.position.set(xOffset, -28, 0);
        armGroup.add(handGroup);
        
        return armGroup;
    }
    
    createHand(side) {
        const handGroup = new THREE.Group();
        
        // Palm
        const palmGeometry = new THREE.BoxGeometry(8, 8, 4);
        const palmMaterial = this.createSuitMaterial(0xcc0000, 0xff0000);
        const palm = new THREE.Mesh(palmGeometry, palmMaterial);
        handGroup.add(palm);
        
        // Repulsor
        const repulsorGeometry = new THREE.CircleGeometry(3, 32);
        const repulsorMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ccff,
            emissive: 0x00ccff,
            emissiveIntensity: 2,
            metalness: 1,
            roughness: 0
        });
        const repulsor = new THREE.Mesh(repulsorGeometry, repulsorMaterial);
        repulsor.position.z = -2.1;
        handGroup.add(repulsor);
        
        // Add repulsor glow
        this.addGlowEffect(repulsor, 0x00ccff, 1);
        
        // Store reference for effects
        if (side === 'left') {
            this.effects.leftRepulsor = repulsor;
        } else {
            this.effects.rightRepulsor = repulsor;
        }
        
        return handGroup;
    }
    
    createLeg(side) {
        const legGroup = new THREE.Group();
        const xOffset = side === 'left' ? -10 : 10;
        
        // Upper leg
        const upperLegGeometry = new THREE.CylinderGeometry(7, 8, 35, 8);
        const upperLegMaterial = this.createSuitMaterial(0xcc0000, 0xff0000);
        const upperLeg = new THREE.Mesh(upperLegGeometry, upperLegMaterial);
        upperLeg.position.set(xOffset, -45, 0);
        upperLeg.castShadow = true;
        legGroup.add(upperLeg);
        
        // Knee joint
        const kneeGeometry = new THREE.SphereGeometry(7, 16, 16);
        const kneeMaterial = this.createSuitMaterial(0x888888, 0xaaaaaa, 1, 0);
        const knee = new THREE.Mesh(kneeGeometry, kneeMaterial);
        knee.position.set(xOffset, -63, 0);
        legGroup.add(knee);
        
        // Lower leg
        const lowerLegGeometry = new THREE.CylinderGeometry(6, 7, 35, 8);
        const lowerLegMaterial = this.createSuitMaterial(0xcc0000, 0xff0000);
        const lowerLeg = new THREE.Mesh(lowerLegGeometry, lowerLegMaterial);
        lowerLeg.position.set(xOffset, -80, 0);
        lowerLeg.castShadow = true;
        legGroup.add(lowerLeg);
        
        // Boot with thruster
        const bootGroup = this.createBoot(side);
        bootGroup.position.set(xOffset, -98, 0);
        legGroup.add(bootGroup);
        
        return legGroup;
    }
    
    createBoot(side) {
        const bootGroup = new THREE.Group();
        
        // Boot
        const bootGeometry = new THREE.BoxGeometry(10, 8, 15);
        const bootMaterial = this.createSuitMaterial(0xcc0000, 0xff0000);
        const boot = new THREE.Mesh(bootGeometry, bootMaterial);
        bootGroup.add(boot);
        
        // Thruster
        const thrusterGeometry = new THREE.CylinderGeometry(3, 4, 5, 16);
        const thrusterMaterial = new THREE.MeshStandardMaterial({
            color: 0x0088ff,
            emissive: 0x0088ff,
            emissiveIntensity: 1,
            metalness: 1,
            roughness: 0.2
        });
        const thruster = new THREE.Mesh(thrusterGeometry, thrusterMaterial);
        thruster.position.y = -6;
        thruster.rotation.x = Math.PI;
        bootGroup.add(thruster);
        
        // Store reference
        if (side === 'left') {
            this.effects.leftThruster = thruster;
        } else {
            this.effects.rightThruster = thruster;
        }
        
        return bootGroup;
    }
    
    createArcReactor() {
        const reactorGroup = new THREE.Group();
        
        // Outer ring
        const ringGeometry = new THREE.TorusGeometry(8, 2, 8, 32);
        const ringMaterial = this.createSuitMaterial(0x888888, 0xcccccc, 1, 0);
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.position.z = -8;
        reactorGroup.add(ring);
        
        // Core
        const coreGeometry = new THREE.SphereGeometry(6, 32, 32);
        const coreMaterial = new THREE.MeshStandardMaterial({
            color: 0x00ccff,
            emissive: 0x00ccff,
            emissiveIntensity: 3,
            metalness: 1,
            roughness: 0
        });
        const core = new THREE.Mesh(coreGeometry, coreMaterial);
        core.position.z = -8;
        reactorGroup.add(core);
        
        // Add pulsing glow
        this.addGlowEffect(core, 0x00ccff, 2);
        this.effects.arcReactor = core;
        
        // Energy lines
        const lineCount = 8;
        for (let i = 0; i < lineCount; i++) {
            const angle = (i / lineCount) * Math.PI * 2;
            const lineGeometry = new THREE.BoxGeometry(1, 10, 1);
            const lineMaterial = new THREE.MeshStandardMaterial({
                color: 0x00ccff,
                emissive: 0x00ccff,
                emissiveIntensity: 1
            });
            const line = new THREE.Mesh(lineGeometry, lineMaterial);
            line.position.x = Math.cos(angle) * 5;
            line.position.y = Math.sin(angle) * 5;
            line.position.z = -8;
            line.rotation.z = angle;
            reactorGroup.add(line);
        }
        
        return reactorGroup;
    }
    
    createThrusters() {
        const thrustersGroup = new THREE.Group();
        
        // Back thrusters
        const backThrusterGeometry = new THREE.CylinderGeometry(4, 5, 10, 16);
        const backThrusterMaterial = new THREE.MeshStandardMaterial({
            color: 0x0088ff,
            emissive: 0x0088ff,
            emissiveIntensity: 1,
            metalness: 1,
            roughness: 0.2
        });
        
        const leftBackThruster = new THREE.Mesh(backThrusterGeometry, backThrusterMaterial);
        leftBackThruster.position.set(-15, -10, 10);
        leftBackThruster.rotation.x = Math.PI / 2;
        thrustersGroup.add(leftBackThruster);
        
        const rightBackThruster = new THREE.Mesh(backThrusterGeometry, backThrusterMaterial);
        rightBackThruster.position.set(15, -10, 10);
        rightBackThruster.rotation.x = Math.PI / 2;
        thrustersGroup.add(rightBackThruster);
        
        this.effects.backThrusters = [leftBackThruster, rightBackThruster];
        
        return thrustersGroup;
    }
    
    createSuitMaterial(color, emissive, metalness = 0.8, roughness = 0.2) {
        return new THREE.MeshStandardMaterial({
            color: color,
            emissive: emissive,
            emissiveIntensity: 0.1,
            metalness: metalness,
            roughness: roughness,
            envMapIntensity: 1
        });
    }
    
    addPanelLines(group, geometry) {
        const edges = new THREE.EdgesGeometry(geometry, 30);
        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0x000000,
            linewidth: 2
        });
        const lines = new THREE.LineSegments(edges, lineMaterial);
        group.add(lines);
    }
    
    addChestDetails(torsoGroup) {
        // Add various tech details to the chest
        const detailCount = 5;
        for (let i = 0; i < detailCount; i++) {
            const detailGeometry = new THREE.BoxGeometry(
                2 + Math.random() * 3,
                2 + Math.random() * 3,
                1
            );
            const detailMaterial = this.createSuitMaterial(0x666666, 0x888888, 1, 0.1);
            const detail = new THREE.Mesh(detailGeometry, detailMaterial);
            detail.position.set(
                (Math.random() - 0.5) * 15,
                (Math.random() - 0.5) * 20,
                -8
            );
            torsoGroup.add(detail);
        }
    }
    
    addGlowEffect(mesh, color, intensity) {
        // Create glow sprite
        const spriteMaterial = new THREE.SpriteMaterial({
            map: this.createGlowTexture(),
            color: color,
            blending: THREE.AdditiveBlending,
            opacity: 0.8
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.multiplyScalar(intensity * 20);
        mesh.add(sprite);
    }
    
    createGlowTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const context = canvas.getContext('2d');
        
        const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
        gradient.addColorStop(0, 'rgba(255,255,255,1)');
        gradient.addColorStop(0.2, 'rgba(255,255,255,0.8)');
        gradient.addColorStop(0.4, 'rgba(255,255,255,0.5)');
        gradient.addColorStop(1, 'rgba(255,255,255,0)');
        
        context.fillStyle = gradient;
        context.fillRect(0, 0, 64, 64);
        
        return new THREE.CanvasTexture(canvas);
    }
    
    createSuitAnimations() {
        // Assembly animation
        const assemblyClip = this.createAssemblyAnimation();
        this.animations.assembly = this.animationMixer.clipAction(assemblyClip);
        
        // Hover animation
        const hoverClip = this.createHoverAnimation();
        this.animations.hover = this.animationMixer.clipAction(hoverClip);
        
        // Combat stance animation
        const combatClip = this.createCombatAnimation();
        this.animations.combat = this.animationMixer.clipAction(combatClip);
    }
    
    createAssemblyAnimation() {
        const duration = 3;
        const tracks = [];
        
        // Animate each part flying in
        Object.entries(this.suitParts).forEach(([name, part], index) => {
            const delay = index * 0.2;
            const startPos = part.position.clone();
            startPos.add(new THREE.Vector3(
                (Math.random() - 0.5) * 200,
                (Math.random() - 0.5) * 200,
                (Math.random() - 0.5) * 200
            ));
            
            const posTrack = new THREE.VectorKeyframeTrack(
                `.suitParts[${name}].position`,
                [delay, delay + 1],
                [startPos.x, startPos.y, startPos.z, part.position.x, part.position.y, part.position.z]
            );
            
            const rotTrack = new THREE.QuaternionKeyframeTrack(
                `.suitParts[${name}].quaternion`,
                [delay, delay + 1],
                [
                    Math.random(), Math.random(), Math.random(), Math.random(),
                    part.quaternion.x, part.quaternion.y, part.quaternion.z, part.quaternion.w
                ]
            );
            
            tracks.push(posTrack, rotTrack);
        });
        
        return new THREE.AnimationClip('assembly', duration, tracks);
    }
    
    createHoverAnimation() {
        const duration = 4;
        const tracks = [];
        
        // Gentle floating motion
        const posTrack = new THREE.VectorKeyframeTrack(
            '.position',
            [0, 1, 2, 3, 4],
            [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, -10, 0, 0, 0, 0]
        );
        
        // Slight rotation
        const rotTrack = new THREE.NumberKeyframeTrack(
            '.rotation[y]',
            [0, 2, 4],
            [0, Math.PI * 0.1, 0]
        );
        
        tracks.push(posTrack, rotTrack);
        
        return new THREE.AnimationClip('hover', duration, tracks);
    }
    
    createCombatAnimation() {
        const duration = 2;
        const tracks = [];
        
        // Combat ready pose
        const leftArmTrack = new THREE.QuaternionKeyframeTrack(
            '.suitParts[leftArm].quaternion',
            [0, 0.5, 1],
            [
                0, 0, 0, 1,
                0.5, 0, 0, 0.866,
                0, 0, 0, 1
            ]
        );
        
        const rightArmTrack = new THREE.QuaternionKeyframeTrack(
            '.suitParts[rightArm].quaternion',
            [0, 0.5, 1],
            [
                0, 0, 0, 1,
                -0.5, 0, 0, 0.866,
                0, 0, 0, 1
            ]
        );
        
        tracks.push(leftArmTrack, rightArmTrack);
        
        return new THREE.AnimationClip('combat', duration, tracks);
    }
    
    setupPostProcessing() {
        // Initialize enhanced post-processing
        if (window.PostProcessingEffects) {
            this.postProcessing = new PostProcessingEffects(this.renderer, this.scene, this.camera);
        } else {
            console.warn('Post-processing not available');
        }
    }
    
    setupControls() {
        this.controls = {
            movement: {
                forward: false,
                backward: false,
                left: false,
                right: false,
                up: false,
                down: false,
                boost: false
            },
            mouse: {
                x: 0,
                y: 0,
                deltaX: 0,
                deltaY: 0
            },
            actions: {
                fire: false,
                switchCamera: false,
                assemble: false
            }
        };
        
        // Mouse look sensitivity
        this.mouseSensitivity = 0.002;
    }
    
    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
        document.addEventListener('keyup', (e) => this.onKeyUp(e));
        
        // Mouse controls
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
        document.addEventListener('mousedown', (e) => this.onMouseDown(e));
        document.addEventListener('mouseup', (e) => this.onMouseUp(e));
        
        // Touch controls for mobile
        if ('ontouchstart' in window) {
            this.setupTouchControls();
        }
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        
        if (this.postProcessing) {
            this.postProcessing.setSize(window.innerWidth, window.innerHeight);
        } else if (this.composer) {
            this.composer.setSize(window.innerWidth, window.innerHeight);
        }
    }
    
    onKeyDown(e) {
        switch(e.key.toLowerCase()) {
            // Movement
            case 'w': this.controls.movement.forward = true; break;
            case 's': this.controls.movement.backward = true; break;
            case 'a': this.controls.movement.left = true; break;
            case 'd': this.controls.movement.right = true; break;
            case 'q': this.controls.movement.up = true; break;
            case 'e': this.controls.movement.down = true; break;
            case 'shift': this.controls.movement.boost = true; break;
            
            // Actions
            case ' ': this.controls.actions.fire = true; break;
            case 'c': this.switchCameraMode(); break;
            case 'r': this.assemblesuit(); break;
            case 'f': this.toggleCombatMode(); break;
        }
    }
    
    onKeyUp(e) {
        switch(e.key.toLowerCase()) {
            case 'w': this.controls.movement.forward = false; break;
            case 's': this.controls.movement.backward = false; break;
            case 'a': this.controls.movement.left = false; break;
            case 'd': this.controls.movement.right = false; break;
            case 'q': this.controls.movement.up = false; break;
            case 'e': this.controls.movement.down = false; break;
            case 'shift': this.controls.movement.boost = false; break;
            case ' ': this.controls.actions.fire = false; break;
        }
    }
    
    onMouseMove(e) {
        const deltaX = e.movementX || e.mozMovementX || e.webkitMovementX || 0;
        const deltaY = e.movementY || e.mozMovementY || e.webkitMovementY || 0;
        
        this.controls.mouse.deltaX = deltaX;
        this.controls.mouse.deltaY = deltaY;
        this.controls.mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
        this.controls.mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    }
    
    onMouseDown(e) {
        if (e.button === 0) { // Left click
            this.controls.actions.fire = true;
        }
    }
    
    onMouseUp(e) {
        if (e.button === 0) {
            this.controls.actions.fire = false;
        }
    }
    
    setupTouchControls() {
        // Virtual joystick for mobile
        const joystickContainer = document.createElement('div');
        joystickContainer.className = 'virtual-joystick';
        document.body.appendChild(joystickContainer);
        
        // Touch event handlers
        let touchStartX = 0;
        let touchStartY = 0;
        
        joystickContainer.addEventListener('touchstart', (e) => {
            const touch = e.touches[0];
            touchStartX = touch.clientX;
            touchStartY = touch.clientY;
        });
        
        joystickContainer.addEventListener('touchmove', (e) => {
            const touch = e.touches[0];
            const deltaX = touch.clientX - touchStartX;
            const deltaY = touch.clientY - touchStartY;
            
            // Convert touch delta to movement
            this.controls.movement.forward = deltaY < -20;
            this.controls.movement.backward = deltaY > 20;
            this.controls.movement.left = deltaX < -20;
            this.controls.movement.right = deltaX > 20;
        });
        
        joystickContainer.addEventListener('touchend', () => {
            this.controls.movement.forward = false;
            this.controls.movement.backward = false;
            this.controls.movement.left = false;
            this.controls.movement.right = false;
        });
    }
    
    switchCameraMode() {
        const modes = ['third-person', 'first-person', 'cinematic'];
        const currentIndex = modes.indexOf(this.cameraMode);
        this.cameraMode = modes[(currentIndex + 1) % modes.length];
        
        switch(this.cameraMode) {
            case 'first-person':
                this.cameraOffset.set(0, 5, 5);
                break;
            case 'cinematic':
                this.cameraOffset.set(100, 150, -300);
                break;
            default: // third-person
                this.cameraOffset.set(0, 100, -300);
        }
    }
    
    assemblesuit() {
        if (this.state.isAssembling) return;
        
        this.state.isAssembling = true;
        this.animations.assembly.reset();
        this.animations.assembly.play();
        
        // Play assembly sound
        if (window.AudioManager) {
            window.AudioManager.play('suit-assembly');
        }
        
        setTimeout(() => {
            this.state.isAssembling = false;
        }, 3000);
    }
    
    toggleCombatMode() {
        this.state.combatMode = !this.state.combatMode;
        
        if (this.state.combatMode) {
            this.animations.combat.reset();
            this.animations.combat.play();
        } else {
            this.animations.combat.stop();
        }
    }
    
    updateMovement(delta) {
        if (!this.suit) return;
        
        const speed = this.controls.movement.boost ? 800 : 300;
        const moveSpeed = speed * delta;
        
        // Calculate movement direction
        const movement = new THREE.Vector3();
        
        if (this.controls.movement.forward) movement.z += 1;
        if (this.controls.movement.backward) movement.z -= 1;
        if (this.controls.movement.left) movement.x -= 1;
        if (this.controls.movement.right) movement.x += 1;
        if (this.controls.movement.up) movement.y += 1;
        if (this.controls.movement.down) movement.y -= 1;
        
        // Apply movement in world space
        if (movement.length() > 0) {
            movement.normalize();
            movement.multiplyScalar(moveSpeed);
            
            // Transform movement to suit's local space
            movement.applyQuaternion(this.suit.quaternion);
            
            this.acceleration.copy(movement);
            this.state.isFlying = true;
            
            // Update thruster intensity based on movement
            if (this.lighting) {
                const thrusterIntensity = this.controls.movement.boost ? 1.0 : 0.5;
                this.lighting.setThrusterIntensity(thrusterIntensity);
            }
        } else {
            this.acceleration.multiplyScalar(0.9);
            this.state.isFlying = false;
            
            // Dim thrusters when not moving
            if (this.lighting) {
                this.lighting.setThrusterIntensity(0.1);
            }
        }
        
        // Apply physics
        this.velocity.add(this.acceleration.multiplyScalar(delta));
        this.velocity.multiplyScalar(0.98); // Air resistance
        
        // Update position
        this.suit.position.add(this.velocity.clone().multiplyScalar(delta));
        
        // Apply mouse look rotation
        this.rotation.y -= this.controls.mouse.deltaX * this.mouseSensitivity;
        this.rotation.x -= this.controls.mouse.deltaY * this.mouseSensitivity;
        this.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.rotation.x));
        
        this.suit.rotation.set(this.rotation.x, this.rotation.y, this.rotation.z);
        
        // Reset mouse delta
        this.controls.mouse.deltaX = 0;
        this.controls.mouse.deltaY = 0;
        
        // Update camera
        this.updateCamera();
        
        // Update HUD
        this.updateHUD();
        
        // Update effects
        this.updateEffects(delta);
    }
    
    updateCamera() {
        const desiredPosition = this.suit.position.clone().add(
            this.cameraOffset.clone().applyQuaternion(this.suit.quaternion)
        );
        
        // Smooth camera movement
        this.camera.position.lerp(desiredPosition, 0.1);
        
        // Look at suit with offset for better view
        const lookTarget = this.suit.position.clone();
        lookTarget.y += 20;
        this.camera.lookAt(lookTarget);
    }
    
    updateHUD() {
        // Calculate values
        this.state.altitude = Math.max(0, this.suit.position.y);
        this.state.speed = this.velocity.length() * 3.6; // Convert to km/h
        
        // Update HUD display
        if (window.HUD) {
            window.HUD.updateAltitude(this.state.altitude);
            window.HUD.updateVelocity(this.state.speed);
            window.HUD.updatePower(this.state.powerLevel);
        }
    }
    
    updateEffects(delta) {
        const time = this.clock.getElapsedTime();
        
        // Arc reactor pulsing
        if (this.effects.arcReactor) {
            const pulse = Math.sin(time * 3) * 0.5 + 0.5;
            this.effects.arcReactor.material.emissiveIntensity = 2 + pulse * 2;
        }
        
        // Thruster effects when flying
        if (this.state.isFlying) {
            this.updateThrusterEffects(delta);
        }
        
        // Repulsor charging when firing
        if (this.controls.actions.fire) {
            this.updateRepulsorEffects(delta);
        }
        
        // Update shader uniforms
        if (this.effects.stars) {
            this.effects.stars.material.uniforms.time.value = time;
        }
        
        if (this.effects.fog) {
            this.effects.fog.material.uniforms.time.value = time;
        }
        
        // Scan grid animation
        if (this.effects.scanGrid) {
            this.effects.scanGrid.position.y = Math.sin(time * 0.5) * 50;
            this.effects.scanGrid.material.opacity = 0.1 + Math.sin(time) * 0.05;
        }
    }
    
    updateThrusterEffects(delta) {
        // Create thruster particles
        const thrusterPositions = [
            this.effects.leftThruster,
            this.effects.rightThruster,
            ...this.effects.backThrusters
        ];
        
        thrusterPositions.forEach(thruster => {
            if (!thruster) return;
            
            // Create particle
            const particle = this.createThrusterParticle();
            particle.position.copy(thruster.getWorldPosition(new THREE.Vector3()));
            this.scene.add(particle);
            
            // Animate particle
            const animateParticle = () => {
                particle.position.y -= 5;
                particle.scale.multiplyScalar(1.1);
                particle.material.opacity *= 0.95;
                
                if (particle.material.opacity > 0.01) {
                    requestAnimationFrame(animateParticle);
                } else {
                    this.scene.remove(particle);
                }
            };
            
            animateParticle();
        });
    }
    
    createThrusterParticle() {
        const geometry = new THREE.SphereGeometry(2, 8, 8);
        const material = new THREE.MeshBasicMaterial({
            color: this.controls.movement.boost ? 0xff8800 : 0x0088ff,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });
        return new THREE.Mesh(geometry, material);
    }
    
    updateRepulsorEffects(delta) {
        const repulsors = [this.effects.leftRepulsor, this.effects.rightRepulsor];
        
        repulsors.forEach(repulsor => {
            if (!repulsor) return;
            
            // Charge effect
            repulsor.material.emissiveIntensity = Math.min(5, repulsor.material.emissiveIntensity + delta * 10);
            
            // Fire beam
            if (repulsor.material.emissiveIntensity >= 5) {
                this.fireRepulsorBeam(repulsor);
                repulsor.material.emissiveIntensity = 2;
            }
        });
    }
    
    fireRepulsorBeam(repulsor) {
        const beamGeometry = new THREE.CylinderGeometry(1, 3, 200, 8);
        const beamMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ccff,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending
        });
        
        const beam = new THREE.Mesh(beamGeometry, beamMaterial);
        beam.position.copy(repulsor.getWorldPosition(new THREE.Vector3()));
        beam.quaternion.copy(this.suit.quaternion);
        beam.rotateX(-Math.PI / 2);
        beam.translateY(100);
        
        this.scene.add(beam);
        
        // Play sound
        if (window.AudioManager) {
            window.AudioManager.play('repulsor-fire');
        }
        
        // Animate beam
        const animateBeam = () => {
            beam.translateY(20);
            beam.scale.x *= 0.95;
            beam.scale.z *= 0.95;
            beam.material.opacity *= 0.9;
            
            if (beam.material.opacity > 0.01) {
                requestAnimationFrame(animateBeam);
            } else {
                this.scene.remove(beam);
            }
        };
        
        animateBeam();
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const delta = Math.min(this.clock.getDelta(), 0.1); // Cap delta to prevent large jumps
        const elapsed = this.clock.getElapsedTime();
        
        // Update materials time
        if (this.materials) {
            this.materials.updateTime(elapsed);
        }
        
        // Update lighting
        if (this.lighting) {
            this.lighting.update(delta);
        }
        
        // Update animations
        if (this.animationMixer) {
            this.animationMixer.update(delta);
        }
        
        // Update movement and physics
        this.updateMovement(delta);
        
        // Update particles
        this.particles.forEach(particle => {
            particle.rotation.y += delta * 0.1;
        });
        
        // Render with post-processing
        if (this.postProcessing) {
            this.postProcessing.render();
        } else if (this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    // 2D Fallback (enhanced)
    init2DFallback() {
        const ctx = this.canvas.getContext('2d');
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        
        // Enhanced 2D visualization
        const render2D = () => {
            // Clear with gradient
            const gradient = ctx.createLinearGradient(0, 0, 0, this.canvas.height);
            gradient.addColorStop(0, '#000033');
            gradient.addColorStop(1, '#000000');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Draw perspective grid
            ctx.strokeStyle = 'rgba(0, 168, 255, 0.3)';
            ctx.lineWidth = 1;
            
            const horizon = this.canvas.height * 0.5;
            const vanishX = this.canvas.width * 0.5;
            
            // Vertical lines
            for (let x = -20; x <= 20; x++) {
                ctx.beginPath();
                ctx.moveTo(vanishX + x * 100, horizon);
                ctx.lineTo(vanishX + x * 200, this.canvas.height);
                ctx.stroke();
            }
            
            // Horizontal lines
            for (let y = 0; y < 10; y++) {
                const yPos = horizon + Math.pow(y / 10, 2) * (this.canvas.height - horizon);
                ctx.beginPath();
                ctx.moveTo(0, yPos);
                ctx.lineTo(this.canvas.width, yPos);
                ctx.stroke();
            }
            
            // Draw suit silhouette
            const suitX = this.canvas.width / 2;
            const suitY = this.canvas.height / 2;
            
            ctx.save();
            ctx.translate(suitX, suitY);
            
            // Body
            ctx.fillStyle = '#cc0000';
            ctx.fillRect(-20, -40, 40, 60);
            
            // Arc reactor
            ctx.beginPath();
            ctx.arc(0, -10, 10, 0, Math.PI * 2);
            ctx.fillStyle = '#00ccff';
            ctx.fill();
            
            // Glow effect
            ctx.shadowBlur = 20;
            ctx.shadowColor = '#00ccff';
            ctx.fill();
            
            ctx.restore();
            
            // HUD elements
            ctx.font = '16px monospace';
            ctx.fillStyle = '#00ff00';
            ctx.fillText('ALT: 0m', 20, 30);
            ctx.fillText('SPD: 0km/h', 20, 50);
            ctx.fillText('PWR: 100%', 20, 70);
            
            requestAnimationFrame(render2D);
        };
        
        render2D();
    }
    
    // Public API
    addTarget(position) {
        const targetGeometry = new THREE.OctahedronGeometry(10, 0);
        const targetMaterial = new THREE.MeshStandardMaterial({
            color: 0xff0000,
            emissive: 0xff0000,
            emissiveIntensity: 0.5,
            metalness: 0.8,
            roughness: 0.2
        });
        
        const target = new THREE.Mesh(targetGeometry, targetMaterial);
        target.position.copy(position);
        
        // Add rotation animation
        target.userData.update = (delta) => {
            target.rotation.x += delta * 2;
            target.rotation.y += delta * 3;
        };
        
        this.scene.add(target);
        this.targets.push(target);
        
        // Add target marker
        this.addTargetMarker(target);
    }
    
    addTargetMarker(target) {
        const markerGeometry = new THREE.RingGeometry(15, 20, 32);
        const markerMaterial = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide
        });
        
        const marker = new THREE.Mesh(markerGeometry, markerMaterial);
        marker.lookAt(this.camera.position);
        target.add(marker);
    }
    
    dispose() {
        // Clean up resources
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        // Dispose of geometries and materials
        this.scene.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
        
        // Dispose renderer
        this.renderer.dispose();
    }
}

// Make EnhancedViewport globally available
window.EnhancedViewport = EnhancedViewport;