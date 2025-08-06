// Iron Man Suit Showcase Viewport
// Well-lit presentation of the ultra-realistic suit

class ShowcaseViewport {
    constructor() {
        this.canvas = document.getElementById('viewport');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.suit = null;
        
        // Animation
        this.clock = new THREE.Clock();
        this.cameraAngle = 0;
        
        this.init();
    }
    
    init() {
        console.log('ShowcaseViewport: Starting showcase mode...');
        
        this.setupRenderer();
        this.setupScene();
        this.setupCamera();
        this.setupLighting();
        this.createFloor();
        this.createSuit();
        this.setupControls();
        
        // Start animation
        this.animate();
        
        console.log('ShowcaseViewport: Ready!');
    }
    
    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.LinearToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.scene.fog = new THREE.Fog(0x0a0a0a, 100, 400);
    }
    
    setupCamera() {
        this.camera = new THREE.PerspectiveCamera(
            50,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 40, 120);
        this.camera.lookAt(0, 20, 0);
    }
    
    setupLighting() {
        // Strong ambient light to ensure visibility
        const ambientLight = new THREE.AmbientLight(0x404040, 1.0);
        this.scene.add(ambientLight);
        
        // Main key light
        const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
        keyLight.position.set(50, 100, 50);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 1;
        keyLight.shadow.camera.far = 200;
        keyLight.shadow.camera.left = -50;
        keyLight.shadow.camera.right = 50;
        keyLight.shadow.camera.top = 50;
        keyLight.shadow.camera.bottom = -50;
        this.scene.add(keyLight);
        
        // Fill light
        const fillLight = new THREE.DirectionalLight(0x88aaff, 1.0);
        fillLight.position.set(-50, 50, 50);
        this.scene.add(fillLight);
        
        // Back light
        const backLight = new THREE.DirectionalLight(0xffffff, 1.0);
        backLight.position.set(0, 50, -100);
        this.scene.add(backLight);
        
        // Spotlights on suit
        const spotLight1 = new THREE.SpotLight(0xffffff, 2.0, 200, Math.PI / 4, 0.5);
        spotLight1.position.set(30, 80, 30);
        spotLight1.target.position.set(0, 20, 0);
        spotLight1.castShadow = true;
        this.scene.add(spotLight1);
        this.scene.add(spotLight1.target);
        
        const spotLight2 = new THREE.SpotLight(0xff6600, 1.0, 200, Math.PI / 4, 0.5);
        spotLight2.position.set(-30, 80, -30);
        spotLight2.target.position.set(0, 20, 0);
        this.scene.add(spotLight2);
        this.scene.add(spotLight2.target);
        
        // Point lights for extra illumination
        const pointLight1 = new THREE.PointLight(0x00aaff, 1, 100);
        pointLight1.position.set(0, 50, 50);
        this.scene.add(pointLight1);
        
        const pointLight2 = new THREE.PointLight(0xff0000, 0.5, 100);
        pointLight2.position.set(-50, 30, 0);
        this.scene.add(pointLight2);
    }
    
    createFloor() {
        // Reflective floor
        const floorGeometry = new THREE.CircleGeometry(100, 64);
        const floorMaterial = new THREE.MeshStandardMaterial({
            color: 0x111111,
            metalness: 0.8,
            roughness: 0.2
        });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -2;
        floor.receiveShadow = true;
        this.scene.add(floor);
        
        // Grid for reference
        const gridHelper = new THREE.GridHelper(200, 40, 0x444444, 0x222222);
        gridHelper.position.y = -1.9;
        this.scene.add(gridHelper);
    }
    
    createSuit() {
        // Create simplified but visible Iron Man suit
        const suit = new THREE.Group();
        
        // Materials
        const redMaterial = new THREE.MeshPhongMaterial({
            color: 0xaa0000,
            emissive: 0x220000,
            specular: 0xffffff,
            shininess: 100
        });
        
        const goldMaterial = new THREE.MeshPhongMaterial({
            color: 0xffaa00,
            emissive: 0x332200,
            specular: 0xffffff,
            shininess: 150
        });
        
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ccff,
            emissive: 0x00ccff
        });
        
        // Head/Helmet
        const headGeometry = new THREE.SphereGeometry(8, 32, 24);
        headGeometry.scale(1, 1.1, 0.95);
        const head = new THREE.Mesh(headGeometry, redMaterial);
        head.position.y = 55;
        head.castShadow = true;
        suit.add(head);
        
        // Faceplate
        const faceplateGeometry = new THREE.BoxGeometry(7, 8, 2);
        const faceplate = new THREE.Mesh(faceplateGeometry, goldMaterial);
        faceplate.position.set(0, 54, 7.5);
        suit.add(faceplate);
        
        // Eyes
        const eyeGeometry = new THREE.BoxGeometry(2, 1, 0.5);
        const leftEye = new THREE.Mesh(eyeGeometry, glowMaterial);
        leftEye.position.set(-2.5, 55, 8);
        suit.add(leftEye);
        
        const rightEye = new THREE.Mesh(eyeGeometry, glowMaterial);
        rightEye.position.set(2.5, 55, 8);
        suit.add(rightEye);
        
        // Torso
        const torsoGeometry = new THREE.BoxGeometry(20, 25, 12);
        torsoGeometry.scale(1.2, 1, 1);
        const torso = new THREE.Mesh(torsoGeometry, redMaterial);
        torso.position.y = 35;
        torso.castShadow = true;
        suit.add(torso);
        
        // Chest Plate
        const chestPlateGeometry = new THREE.BoxGeometry(8, 12, 2);
        const chestPlate = new THREE.Mesh(chestPlateGeometry, goldMaterial);
        chestPlate.position.set(0, 38, 7);
        suit.add(chestPlate);
        
        // Arc Reactor
        const arcReactorGeometry = new THREE.CylinderGeometry(3, 3, 1, 16);
        const arcReactor = new THREE.Mesh(arcReactorGeometry, glowMaterial);
        arcReactor.rotation.x = Math.PI / 2;
        arcReactor.position.set(0, 38, 8);
        suit.add(arcReactor);
        
        // Arc Reactor Light
        const reactorLight = new THREE.PointLight(0x00ccff, 2, 30);
        reactorLight.position.set(0, 38, 10);
        suit.add(reactorLight);
        
        // Shoulders
        const shoulderGeometry = new THREE.SphereGeometry(7, 16, 12);
        const leftShoulder = new THREE.Mesh(shoulderGeometry, redMaterial);
        leftShoulder.position.set(-15, 45, 0);
        leftShoulder.castShadow = true;
        suit.add(leftShoulder);
        
        const rightShoulder = new THREE.Mesh(shoulderGeometry, redMaterial);
        rightShoulder.position.set(15, 45, 0);
        rightShoulder.castShadow = true;
        suit.add(rightShoulder);
        
        // Arms
        const armGeometry = new THREE.CylinderGeometry(4, 3, 20, 8);
        
        const leftUpperArm = new THREE.Mesh(armGeometry, redMaterial);
        leftUpperArm.position.set(-15, 33, 0);
        leftUpperArm.castShadow = true;
        suit.add(leftUpperArm);
        
        const rightUpperArm = new THREE.Mesh(armGeometry, redMaterial);
        rightUpperArm.position.set(15, 33, 0);
        rightUpperArm.castShadow = true;
        suit.add(rightUpperArm);
        
        const leftLowerArm = new THREE.Mesh(armGeometry, redMaterial);
        leftLowerArm.position.set(-15, 18, 0);
        leftLowerArm.castShadow = true;
        suit.add(leftLowerArm);
        
        const rightLowerArm = new THREE.Mesh(armGeometry, redMaterial);
        rightLowerArm.position.set(15, 18, 0);
        rightLowerArm.castShadow = true;
        suit.add(rightLowerArm);
        
        // Gauntlets
        const gauntletGeometry = new THREE.BoxGeometry(6, 8, 5);
        const leftGauntlet = new THREE.Mesh(gauntletGeometry, goldMaterial);
        leftGauntlet.position.set(-15, 8, 0);
        suit.add(leftGauntlet);
        
        const rightGauntlet = new THREE.Mesh(gauntletGeometry, goldMaterial);
        rightGauntlet.position.set(15, 8, 0);
        suit.add(rightGauntlet);
        
        // Repulsors
        const repulsorGeometry = new THREE.CylinderGeometry(2, 2, 0.5, 16);
        const leftRepulsor = new THREE.Mesh(repulsorGeometry, glowMaterial);
        leftRepulsor.rotation.x = Math.PI / 2;
        leftRepulsor.position.set(-15, 8, 3);
        suit.add(leftRepulsor);
        
        const rightRepulsor = new THREE.Mesh(repulsorGeometry, glowMaterial);
        rightRepulsor.rotation.x = Math.PI / 2;
        rightRepulsor.position.set(15, 8, 3);
        suit.add(rightRepulsor);
        
        // Waist
        const waistGeometry = new THREE.BoxGeometry(18, 8, 10);
        const waist = new THREE.Mesh(waistGeometry, redMaterial);
        waist.position.y = 20;
        waist.castShadow = true;
        suit.add(waist);
        
        // Belt
        const beltGeometry = new THREE.BoxGeometry(20, 3, 11);
        const belt = new THREE.Mesh(beltGeometry, goldMaterial);
        belt.position.y = 20;
        suit.add(belt);
        
        // Legs
        const legGeometry = new THREE.CylinderGeometry(5, 4, 25, 8);
        
        const leftThigh = new THREE.Mesh(legGeometry, redMaterial);
        leftThigh.position.set(-7, 12, 0);
        leftThigh.castShadow = true;
        suit.add(leftThigh);
        
        const rightThigh = new THREE.Mesh(legGeometry, redMaterial);
        rightThigh.position.set(7, 12, 0);
        rightThigh.castShadow = true;
        suit.add(rightThigh);
        
        const leftShin = new THREE.Mesh(legGeometry, redMaterial);
        leftShin.position.set(-7, -5, 0);
        leftShin.castShadow = true;
        suit.add(leftShin);
        
        const rightShin = new THREE.Mesh(legGeometry, redMaterial);
        rightShin.position.set(7, -5, 0);
        rightShin.castShadow = true;
        suit.add(rightShin);
        
        // Boots
        const bootGeometry = new THREE.BoxGeometry(8, 6, 12);
        const leftBoot = new THREE.Mesh(bootGeometry, redMaterial);
        leftBoot.position.set(-7, -15, 2);
        leftBoot.castShadow = true;
        suit.add(leftBoot);
        
        const rightBoot = new THREE.Mesh(bootGeometry, redMaterial);
        rightBoot.position.set(7, -15, 2);
        rightBoot.castShadow = true;
        suit.add(rightBoot);
        
        // Thrusters
        const thrusterGeometry = new THREE.CylinderGeometry(2, 3, 3, 8);
        const leftThruster = new THREE.Mesh(thrusterGeometry, goldMaterial);
        leftThruster.position.set(-7, -18, -3);
        leftThruster.rotation.x = Math.PI / 2;
        suit.add(leftThruster);
        
        const rightThruster = new THREE.Mesh(thrusterGeometry, goldMaterial);
        rightThruster.position.set(7, -18, -3);
        rightThruster.rotation.x = Math.PI / 2;
        suit.add(rightThruster);
        
        // Add some panel lines
        const edges = new THREE.EdgesGeometry(torsoGeometry);
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0x000000 });
        const lineSegments = new THREE.LineSegments(edges, lineMaterial);
        lineSegments.position.copy(torso.position);
        lineSegments.scale.copy(torso.scale);
        suit.add(lineSegments);
        
        this.suit = suit;
        this.scene.add(suit);
        
        // Position suit
        suit.position.y = 20;
    }
    
    setupControls() {
        // Window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Mouse controls for camera orbit
        let mouseDown = false;
        let mouseX = 0;
        
        this.canvas.addEventListener('mousedown', (e) => {
            mouseDown = true;
            mouseX = e.clientX;
        });
        
        this.canvas.addEventListener('mouseup', () => {
            mouseDown = false;
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (mouseDown) {
                const deltaX = e.clientX - mouseX;
                this.cameraAngle += deltaX * 0.01;
                mouseX = e.clientX;
            }
        });
        
        // Touch controls
        this.canvas.addEventListener('touchstart', (e) => {
            mouseDown = true;
            mouseX = e.touches[0].clientX;
        });
        
        this.canvas.addEventListener('touchend', () => {
            mouseDown = false;
        });
        
        this.canvas.addEventListener('touchmove', (e) => {
            if (mouseDown) {
                const deltaX = e.touches[0].clientX - mouseX;
                this.cameraAngle += deltaX * 0.01;
                mouseX = e.touches[0].clientX;
            }
        });
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const time = this.clock.getElapsedTime();
        
        // Auto-rotate camera
        this.cameraAngle += 0.003;
        this.camera.position.x = Math.cos(this.cameraAngle) * 120;
        this.camera.position.z = Math.sin(this.cameraAngle) * 120;
        this.camera.position.y = 40 + Math.sin(time * 0.5) * 10;
        this.camera.lookAt(0, 30, 0);
        
        // Animate arc reactor glow
        const arcReactor = this.suit?.children.find(child => 
            child.geometry?.type === 'CylinderGeometry' && child.material?.emissive
        );
        if (arcReactor) {
            const pulse = Math.sin(time * 3) * 0.3 + 0.7;
            arcReactor.scale.set(pulse, 1, pulse);
        }
        
        // Render
        this.renderer.render(this.scene, this.camera);
    }
}

// Replace EnhancedViewport with ShowcaseViewport
window.EnhancedViewport = ShowcaseViewport;