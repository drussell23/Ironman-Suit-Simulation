// Cinematic Environment for Iron Man Experience
// High-tech lab/hangar with dramatic lighting and atmosphere

class CinematicEnvironment {
    constructor(scene) {
        this.scene = scene;
        this.elements = {};
        this.lights = [];
        this.animatedElements = [];
        
        this.init();
    }
    
    init() {
        this.createFloor();
        this.createPlatform();
        this.createWalls();
        this.createCeiling();
        this.addEnvironmentDetails();
        this.addAtmosphericEffects();
        this.setupEnvironmentLighting();
    }
    
    createFloor() {
        // Multi-layered floor with reflections
        const floorGroup = new THREE.Group();
        
        // Base floor - polished concrete
        const floorGeometry = new THREE.PlaneGeometry(500, 500, 50, 50);
        
        // Add subtle height variation
        const positions = floorGeometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const z = positions.getZ(i);
            const noise = Math.sin(x * 0.05) * Math.cos(z * 0.05) * 0.2;
            positions.setY(i, noise);
        }
        floorGeometry.computeVertexNormals();
        
        const floorMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x0a0a0a,
            metalness: 0.7,
            roughness: 0.15,
            clearcoat: 0.8,
            clearcoatRoughness: 0.1,
            reflectivity: 0.9,
            envMapIntensity: 0.5
        });
        
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -2;
        floor.receiveShadow = true;
        floorGroup.add(floor);
        
        // Grid overlay
        const gridTexture = this.createGridTexture();
        const gridMaterial = new THREE.MeshBasicMaterial({
            map: gridTexture,
            transparent: true,
            opacity: 0.3,
            blending: THREE.AdditiveBlending
        });
        
        const gridOverlay = new THREE.Mesh(
            new THREE.PlaneGeometry(500, 500),
            gridMaterial
        );
        gridOverlay.rotation.x = -Math.PI / 2;
        gridOverlay.position.y = -1.9;
        floorGroup.add(gridOverlay);
        
        // Floor markings
        this.addFloorMarkings(floorGroup);
        
        this.scene.add(floorGroup);
        this.elements.floor = floorGroup;
    }
    
    createGridTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Background
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, 512, 512);
        
        // Grid lines
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.5;
        
        // Major grid
        for (let i = 0; i <= 512; i += 64) {
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, 512);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(512, i);
            ctx.stroke();
        }
        
        // Minor grid
        ctx.globalAlpha = 0.2;
        for (let i = 0; i <= 512; i += 16) {
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, 512);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(512, i);
            ctx.stroke();
        }
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
        texture.repeat.set(20, 20);
        
        return texture;
    }
    
    addFloorMarkings(floorGroup) {
        // Warning stripes
        const stripeGeometry = new THREE.PlaneGeometry(10, 80);
        const stripeMaterial = new THREE.MeshPhysicalMaterial({
            color: 0xffaa00,
            emissive: 0xff6600,
            emissiveIntensity: 0.2,
            metalness: 0.8,
            roughness: 0.2
        });
        
        // Create warning area around platform
        for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 4) {
            const stripe = new THREE.Mesh(stripeGeometry, stripeMaterial);
            stripe.rotation.x = -Math.PI / 2;
            stripe.rotation.z = angle;
            stripe.position.x = Math.cos(angle) * 80;
            stripe.position.z = Math.sin(angle) * 80;
            stripe.position.y = -1.8;
            floorGroup.add(stripe);
        }
        
        // Directional arrows
        const arrowShape = new THREE.Shape();
        arrowShape.moveTo(0, -5);
        arrowShape.lineTo(3, 0);
        arrowShape.lineTo(1, 0);
        arrowShape.lineTo(1, 5);
        arrowShape.lineTo(-1, 5);
        arrowShape.lineTo(-1, 0);
        arrowShape.lineTo(-3, 0);
        arrowShape.closePath();
        
        const arrowGeometry = new THREE.ShapeGeometry(arrowShape);
        const arrowMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            emissive: 0x00ff00,
            emissiveIntensity: 0.5
        });
        
        for (let i = 0; i < 4; i++) {
            const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
            arrow.rotation.x = -Math.PI / 2;
            arrow.rotation.z = i * Math.PI / 2;
            arrow.position.x = Math.cos(i * Math.PI / 2) * 120;
            arrow.position.z = Math.sin(i * Math.PI / 2) * 120;
            arrow.position.y = -1.8;
            floorGroup.add(arrow);
        }
    }
    
    createPlatform() {
        const platformGroup = new THREE.Group();
        
        // Main platform - multi-level design
        const basePlatformGeometry = new THREE.CylinderGeometry(55, 60, 4, 64);
        const basePlatformMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x1a1a1a,
            metalness: 0.9,
            roughness: 0.1,
            clearcoat: 0.5,
            clearcoatRoughness: 0.05
        });
        const basePlatform = new THREE.Mesh(basePlatformGeometry, basePlatformMaterial);
        basePlatform.position.y = -0.5;
        basePlatform.receiveShadow = true;
        basePlatform.castShadow = true;
        platformGroup.add(basePlatform);
        
        // Upper platform ring
        const upperRingGeometry = new THREE.CylinderGeometry(52, 54, 2, 64);
        const upperRing = new THREE.Mesh(upperRingGeometry, basePlatformMaterial);
        upperRing.position.y = 2;
        platformGroup.add(upperRing);
        
        // Platform lights - animated ring
        const lightRingGeometry = new THREE.TorusGeometry(53, 1.5, 16, 64);
        const lightRingMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x00aaff,
            emissive: 0x00aaff,
            emissiveIntensity: 2.0,
            metalness: 0.0,
            roughness: 0.0,
            transparent: true,
            opacity: 0.8
        });
        const lightRing = new THREE.Mesh(lightRingGeometry, lightRingMaterial);
        lightRing.rotation.x = -Math.PI / 2;
        lightRing.position.y = 3;
        platformGroup.add(lightRing);
        this.animatedElements.push({ mesh: lightRing, type: 'pulse' });
        
        // Inner details
        const detailRings = [
            { radius: 45, height: 0.5, y: 3.5 },
            { radius: 35, height: 0.3, y: 3.7 },
            { radius: 25, height: 0.2, y: 3.8 }
        ];
        
        detailRings.forEach(ring => {
            const ringGeo = new THREE.RingGeometry(ring.radius - 1, ring.radius, 64);
            const ringMat = new THREE.MeshPhysicalMaterial({
                color: 0x333333,
                metalness: 0.8,
                roughness: 0.2,
                emissive: 0x001122,
                emissiveIntensity: 0.1
            });
            const ringMesh = new THREE.Mesh(ringGeo, ringMat);
            ringMesh.rotation.x = -Math.PI / 2;
            ringMesh.position.y = ring.y;
            platformGroup.add(ringMesh);
        });
        
        // Tech panels
        this.addTechPanels(platformGroup);
        
        // Platform light
        const platformLight = new THREE.PointLight(0x00aaff, 1, 100);
        platformLight.position.y = 5;
        platformGroup.add(platformLight);
        this.lights.push(platformLight);
        
        this.scene.add(platformGroup);
        this.elements.platform = platformGroup;
    }
    
    addTechPanels(platformGroup) {
        const panelGeometry = new THREE.BoxGeometry(8, 0.5, 8);
        const panelMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x002244,
            metalness: 0.7,
            roughness: 0.3,
            emissive: 0x001133,
            emissiveIntensity: 0.3
        });
        
        // Add tech panels around the platform
        for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 3) {
            const panel = new THREE.Mesh(panelGeometry, panelMaterial);
            panel.position.x = Math.cos(angle) * 40;
            panel.position.z = Math.sin(angle) * 40;
            panel.position.y = 3.5;
            panel.rotation.y = angle;
            platformGroup.add(panel);
            
            // Add small indicator light
            const lightGeometry = new THREE.SphereGeometry(0.5, 16, 16);
            const lightMaterial = new THREE.MeshBasicMaterial({
                color: 0x00ff00,
                emissive: 0x00ff00
            });
            const light = new THREE.Mesh(lightGeometry, lightMaterial);
            light.position.x = Math.cos(angle) * 40;
            light.position.z = Math.sin(angle) * 40;
            light.position.y = 4.5;
            platformGroup.add(light);
            this.animatedElements.push({ mesh: light, type: 'blink' });
        }
    }
    
    createWalls() {
        // Create futuristic lab walls
        const wallGroup = new THREE.Group();
        
        // Wall material
        const wallMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x0a0a0a,
            metalness: 0.6,
            roughness: 0.4,
            clearcoat: 0.3,
            clearcoatRoughness: 0.2
        });
        
        // Back wall with displays
        const backWallGeometry = new THREE.BoxGeometry(300, 100, 5);
        const backWall = new THREE.Mesh(backWallGeometry, wallMaterial);
        backWall.position.set(0, 50, -150);
        backWall.receiveShadow = true;
        wallGroup.add(backWall);
        
        // Add holographic displays
        this.addHolographicDisplays(wallGroup);
        
        // Side walls with structural elements
        const sideWallGeometry = new THREE.BoxGeometry(5, 100, 200);
        
        const leftWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
        leftWall.position.set(-150, 50, -50);
        leftWall.receiveShadow = true;
        wallGroup.add(leftWall);
        
        const rightWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
        rightWall.position.set(150, 50, -50);
        rightWall.receiveShadow = true;
        wallGroup.add(rightWall);
        
        // Structural pillars
        this.addStructuralPillars(wallGroup);
        
        this.scene.add(wallGroup);
        this.elements.walls = wallGroup;
    }
    
    addHolographicDisplays(wallGroup) {
        // Create holographic display screens
        const displayGeometry = new THREE.PlaneGeometry(40, 30);
        const displayMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x0088ff,
            emissive: 0x0044ff,
            emissiveIntensity: 0.5,
            metalness: 0.0,
            roughness: 0.0,
            transparent: true,
            opacity: 0.7,
            side: THREE.DoubleSide
        });
        
        // Center display
        const centerDisplay = new THREE.Mesh(displayGeometry, displayMaterial);
        centerDisplay.position.set(0, 60, -147);
        wallGroup.add(centerDisplay);
        
        // Side displays
        const leftDisplay = new THREE.Mesh(displayGeometry, displayMaterial);
        leftDisplay.position.set(-60, 60, -147);
        leftDisplay.rotation.y = 0.2;
        wallGroup.add(leftDisplay);
        
        const rightDisplay = new THREE.Mesh(displayGeometry, displayMaterial);
        rightDisplay.position.set(60, 60, -147);
        rightDisplay.rotation.y = -0.2;
        wallGroup.add(rightDisplay);
        
        // Add scan line effect
        displays = [centerDisplay, leftDisplay, rightDisplay];
        displays.forEach(display => {
            this.animatedElements.push({ mesh: display, type: 'scanline' });
        });
    }
    
    addStructuralPillars(wallGroup) {
        const pillarGeometry = new THREE.BoxGeometry(10, 100, 10);
        const pillarMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x1a1a1a,
            metalness: 0.8,
            roughness: 0.3
        });
        
        // Corner pillars
        const positions = [
            { x: -145, z: -145 },
            { x: 145, z: -145 },
            { x: -145, z: 45 },
            { x: 145, z: 45 }
        ];
        
        positions.forEach(pos => {
            const pillar = new THREE.Mesh(pillarGeometry, pillarMaterial);
            pillar.position.set(pos.x, 50, pos.z);
            pillar.castShadow = true;
            wallGroup.add(pillar);
            
            // Add accent light strip
            const lightStripGeometry = new THREE.BoxGeometry(2, 90, 2);
            const lightStripMaterial = new THREE.MeshBasicMaterial({
                color: 0x00aaff,
                emissive: 0x00aaff
            });
            const lightStrip = new THREE.Mesh(lightStripGeometry, lightStripMaterial);
            lightStrip.position.set(pos.x, 50, pos.z + 6);
            wallGroup.add(lightStrip);
        });
    }
    
    createCeiling() {
        const ceilingGroup = new THREE.Group();
        
        // Main ceiling
        const ceilingGeometry = new THREE.PlaneGeometry(300, 300);
        const ceilingMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x050505,
            metalness: 0.4,
            roughness: 0.6
        });
        const ceiling = new THREE.Mesh(ceilingGeometry, ceilingMaterial);
        ceiling.rotation.x = Math.PI / 2;
        ceiling.position.y = 100;
        ceilingGroup.add(ceiling);
        
        // Ceiling lights grid
        const lightFixtureGeometry = new THREE.BoxGeometry(20, 2, 20);
        const lightFixtureMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x222222,
            metalness: 0.7,
            roughness: 0.3
        });
        
        for (let x = -100; x <= 100; x += 50) {
            for (let z = -100; z <= 0; z += 50) {
                const fixture = new THREE.Mesh(lightFixtureGeometry, lightFixtureMaterial);
                fixture.position.set(x, 98, z);
                ceilingGroup.add(fixture);
                
                // Light panel
                const panelGeometry = new THREE.PlaneGeometry(18, 18);
                const panelMaterial = new THREE.MeshBasicMaterial({
                    color: 0xffffff,
                    emissive: 0xffffff,
                    emissiveIntensity: 0.8
                });
                const panel = new THREE.Mesh(panelGeometry, panelMaterial);
                panel.rotation.x = -Math.PI / 2;
                panel.position.set(x, 97, z);
                ceilingGroup.add(panel);
                
                // Actual light
                const ceilingLight = new THREE.RectAreaLight(0xffffff, 20, 18, 18);
                ceilingLight.position.set(x, 97, z);
                ceilingLight.rotation.x = -Math.PI / 2;
                ceilingGroup.add(ceilingLight);
            }
        }
        
        this.scene.add(ceilingGroup);
        this.elements.ceiling = ceilingGroup;
    }
    
    addEnvironmentDetails() {
        // Tool racks
        this.addToolRacks();
        
        // Computer terminals
        this.addComputerTerminals();
        
        // Robotic arms
        this.addRoboticArms();
        
        // Storage units
        this.addStorageUnits();
    }
    
    addToolRacks() {
        const rackGroup = new THREE.Group();
        
        // Rack frame
        const frameGeometry = new THREE.BoxGeometry(40, 60, 5);
        const frameMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x333333,
            metalness: 0.8,
            roughness: 0.3
        });
        
        const rack = new THREE.Mesh(frameGeometry, frameMaterial);
        rack.position.set(-120, 30, -140);
        rackGroup.add(rack);
        
        // Tools (simplified)
        const toolMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x666666,
            metalness: 0.9,
            roughness: 0.2
        });
        
        for (let i = 0; i < 5; i++) {
            const toolGeometry = new THREE.CylinderGeometry(1, 1.5, 20, 8);
            const tool = new THREE.Mesh(toolGeometry, toolMaterial);
            tool.position.set(-120 + (i - 2) * 8, 30, -138);
            tool.rotation.z = Math.random() * 0.2 - 0.1;
            rackGroup.add(tool);
        }
        
        this.scene.add(rackGroup);
    }
    
    addComputerTerminals() {
        const terminalGroup = new THREE.Group();
        
        // Desk
        const deskGeometry = new THREE.BoxGeometry(60, 3, 30);
        const deskMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x1a1a1a,
            metalness: 0.7,
            roughness: 0.3
        });
        const desk = new THREE.Mesh(deskGeometry, deskMaterial);
        desk.position.set(100, 35, -120);
        terminalGroup.add(desk);
        
        // Monitors
        for (let i = 0; i < 3; i++) {
            const monitorGeometry = new THREE.BoxGeometry(20, 15, 1);
            const monitorMaterial = new THREE.MeshPhysicalMaterial({
                color: 0x000000,
                emissive: 0x001144,
                emissiveIntensity: 0.5,
                metalness: 0.0,
                roughness: 0.1
            });
            const monitor = new THREE.Mesh(monitorGeometry, monitorMaterial);
            monitor.position.set(80 + i * 20, 45, -120);
            monitor.rotation.y = -0.3 + i * 0.2;
            terminalGroup.add(monitor);
            
            // Screen glow
            const screenLight = new THREE.RectAreaLight(0x0088ff, 5, 18, 13);
            screenLight.position.copy(monitor.position);
            screenLight.position.z += 1;
            screenLight.rotation.copy(monitor.rotation);
            terminalGroup.add(screenLight);
        }
        
        this.scene.add(terminalGroup);
    }
    
    addRoboticArms() {
        const armGroup = new THREE.Group();
        
        // Base
        const baseGeometry = new THREE.CylinderGeometry(8, 10, 5, 16);
        const baseMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x444444,
            metalness: 0.8,
            roughness: 0.3
        });
        const base = new THREE.Mesh(baseGeometry, baseMaterial);
        base.position.set(80, 2.5, 0);
        armGroup.add(base);
        
        // Arm segments
        const segmentMaterial = new THREE.MeshPhysicalMaterial({
            color: 0xcccccc,
            metalness: 0.9,
            roughness: 0.2
        });
        
        // Lower arm
        const lowerArmGeometry = new THREE.CylinderGeometry(3, 4, 30, 8);
        const lowerArm = new THREE.Mesh(lowerArmGeometry, segmentMaterial);
        lowerArm.position.set(80, 20, 0);
        lowerArm.rotation.z = 0.3;
        armGroup.add(lowerArm);
        
        // Upper arm
        const upperArmGeometry = new THREE.CylinderGeometry(2, 3, 25, 8);
        const upperArm = new THREE.Mesh(upperArmGeometry, segmentMaterial);
        upperArm.position.set(85, 40, 0);
        upperArm.rotation.z = -0.5;
        armGroup.add(upperArm);
        
        // Gripper
        const gripperGeometry = new THREE.BoxGeometry(8, 8, 8);
        const gripper = new THREE.Mesh(gripperGeometry, segmentMaterial);
        gripper.position.set(87, 50, 0);
        armGroup.add(gripper);
        
        this.scene.add(armGroup);
        this.animatedElements.push({ mesh: armGroup, type: 'roboticArm' });
    }
    
    addStorageUnits() {
        const storageGroup = new THREE.Group();
        
        const unitGeometry = new THREE.BoxGeometry(30, 80, 20);
        const unitMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x1a1a1a,
            metalness: 0.7,
            roughness: 0.4
        });
        
        // Create multiple storage units
        for (let i = 0; i < 3; i++) {
            const unit = new THREE.Mesh(unitGeometry, unitMaterial);
            unit.position.set(-130, 40, -80 + i * 35);
            storageGroup.add(unit);
            
            // Add detail panels
            const panelGeometry = new THREE.BoxGeometry(25, 15, 0.5);
            const panelMaterial = new THREE.MeshPhysicalMaterial({
                color: 0x003366,
                emissive: 0x001133,
                emissiveIntensity: 0.2,
                metalness: 0.5,
                roughness: 0.3
            });
            
            for (let j = 0; j < 4; j++) {
                const panel = new THREE.Mesh(panelGeometry, panelMaterial);
                panel.position.set(-130, 20 + j * 20, -70 + i * 35);
                storageGroup.add(panel);
            }
        }
        
        this.scene.add(storageGroup);
    }
    
    addAtmosphericEffects() {
        // Fog for depth
        this.scene.fog = new THREE.FogExp2(0x000000, 0.002);
        
        // Particle dust
        const particleGeometry = new THREE.BufferGeometry();
        const particleCount = 1000;
        const positions = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 300;
            positions[i * 3 + 1] = Math.random() * 100;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 300;
        }
        
        particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const particleMaterial = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.5,
            transparent: true,
            opacity: 0.3,
            blending: THREE.AdditiveBlending
        });
        
        const particles = new THREE.Points(particleGeometry, particleMaterial);
        this.scene.add(particles);
        this.elements.particles = particles;
        this.animatedElements.push({ mesh: particles, type: 'float' });
    }
    
    setupEnvironmentLighting() {
        // Ambient light for base illumination
        const ambientLight = new THREE.AmbientLight(0x101020, 0.4);
        this.scene.add(ambientLight);
        
        // Main overhead lights
        const mainLight1 = new THREE.SpotLight(0xffffff, 100, 200, Math.PI / 3, 0.5);
        mainLight1.position.set(0, 90, 0);
        mainLight1.target.position.set(0, 0, 0);
        mainLight1.castShadow = true;
        mainLight1.shadow.mapSize.width = 2048;
        mainLight1.shadow.mapSize.height = 2048;
        this.scene.add(mainLight1);
        this.scene.add(mainLight1.target);
        
        // Colored accent lights
        const accentLight1 = new THREE.PointLight(0x0088ff, 50, 150);
        accentLight1.position.set(-100, 50, -100);
        this.scene.add(accentLight1);
        
        const accentLight2 = new THREE.PointLight(0xff0044, 30, 100);
        accentLight2.position.set(100, 40, -50);
        this.scene.add(accentLight2);
        
        // Floor up-lighting
        const floorLight1 = new THREE.SpotLight(0x00aaff, 50, 100, Math.PI / 4, 0.5);
        floorLight1.position.set(-50, 5, -50);
        floorLight1.target.position.set(-50, 100, -50);
        this.scene.add(floorLight1);
        this.scene.add(floorLight1.target);
        
        const floorLight2 = new THREE.SpotLight(0x00aaff, 50, 100, Math.PI / 4, 0.5);
        floorLight2.position.set(50, 5, -50);
        floorLight2.target.position.set(50, 100, -50);
        this.scene.add(floorLight2);
        this.scene.add(floorLight2.target);
    }
    
    update(deltaTime, time) {
        // Animate elements
        this.animatedElements.forEach(element => {
            switch (element.type) {
                case 'pulse':
                    const pulse = Math.sin(time * 2) * 0.1 + 0.9;
                    element.mesh.material.emissiveIntensity = 2.0 * pulse;
                    break;
                    
                case 'blink':
                    element.mesh.material.emissiveIntensity = 
                        Math.random() < 0.98 ? 1.0 : 0.0;
                    break;
                    
                case 'scanline':
                    // Create scanline effect on displays
                    const scanPos = (time * 0.5) % 1;
                    element.mesh.material.emissiveIntensity = 0.5 + Math.sin(scanPos * Math.PI) * 0.3;
                    break;
                    
                case 'float':
                    // Floating particles
                    const positions = element.mesh.geometry.attributes.position;
                    for (let i = 0; i < positions.count; i++) {
                        const y = positions.getY(i);
                        positions.setY(i, y + Math.sin(time + i) * 0.01);
                    }
                    positions.needsUpdate = true;
                    break;
                    
                case 'roboticArm':
                    // Simple robotic arm movement
                    element.mesh.rotation.y = Math.sin(time * 0.3) * 0.5;
                    break;
            }
        });
        
        // Update platform lights intensity
        this.lights.forEach((light, index) => {
            if (light.parent === this.elements.platform) {
                light.intensity = 1 + Math.sin(time * 3 + index) * 0.2;
            }
        });
    }
}

// Export
window.CinematicEnvironment = CinematicEnvironment;