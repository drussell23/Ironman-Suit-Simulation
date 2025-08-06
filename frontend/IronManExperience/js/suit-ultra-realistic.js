// Ultra-Realistic Iron Man Suit Builder
// Movie-quality 3D model with advanced materials and effects

class UltraRealisticIronManSuit {
    constructor(scene) {
        this.scene = scene;
        this.suit = new THREE.Group();
        this.parts = {};
        this.materials = {};
        this.effects = [];
        this.animationMixers = [];
        
        this.init();
    }
    
    init() {
        this.createMaterials();
        this.buildSuit();
        this.addEffects();
        this.setupAnimations();
    }
    
    createMaterials() {
        // Create ultra-realistic PBR materials
        
        // Main armor - deep metallic red with car paint finish
        this.materials.armorRed = new THREE.MeshPhysicalMaterial({
            color: 0x8B0000,
            metalness: 0.9,
            roughness: 0.15,
            clearcoat: 1.0,
            clearcoatRoughness: 0.05,
            reflectivity: 0.9,
            emissive: 0x220000,
            emissiveIntensity: 0.05,
            envMapIntensity: 1.5
        });
        
        // Gold titanium alloy
        this.materials.armorGold = new THREE.MeshPhysicalMaterial({
            color: 0xFFD700,
            metalness: 0.95,
            roughness: 0.1,
            clearcoat: 0.8,
            clearcoatRoughness: 0.1,
            reflectivity: 1.0,
            emissive: 0x443300,
            emissiveIntensity: 0.1,
            envMapIntensity: 2.0
        });
        
        // Arc reactor - plasma energy core
        this.materials.arcReactor = new THREE.MeshPhysicalMaterial({
            color: 0x00E5FF,
            emissive: 0x00E5FF,
            emissiveIntensity: 4.0,
            metalness: 0.2,
            roughness: 0.0,
            transmission: 0.8,
            thickness: 0.5,
            ior: 1.5,
            transparent: true,
            opacity: 0.9
        });
        
        // Glowing eyes with subsurface scattering
        this.materials.eyes = new THREE.MeshPhysicalMaterial({
            color: 0xFFFFFF,
            emissive: 0x00CCFF,
            emissiveIntensity: 5.0,
            metalness: 0.0,
            roughness: 0.0,
            transmission: 0.5,
            thickness: 0.2,
            transparent: true
        });
        
        // Carbon fiber joints
        this.materials.carbonFiber = new THREE.MeshPhysicalMaterial({
            color: 0x1A1A1A,
            metalness: 0.4,
            roughness: 0.6,
            clearcoat: 0.5,
            clearcoatRoughness: 0.3,
            normalScale: new THREE.Vector2(2, 2)
        });
        
        // Brushed metal details
        this.materials.brushedMetal = new THREE.MeshPhysicalMaterial({
            color: 0x666666,
            metalness: 0.8,
            roughness: 0.3,
            anisotropy: 0.8,
            anisotropyRotation: Math.PI / 4
        });
    }
    
    buildSuit() {
        // Build with movie-accurate proportions
        this.parts.helmet = this.buildHelmet();
        this.parts.chestplate = this.buildChestplate();
        this.parts.shoulders = this.buildShoulders();
        this.parts.arms = this.buildArms();
        this.parts.gauntlets = this.buildGauntlets();
        this.parts.waist = this.buildWaist();
        this.parts.legs = this.buildLegs();
        this.parts.boots = this.buildBoots();
        
        // Assemble all parts
        Object.values(this.parts).forEach(part => {
            this.suit.add(part);
        });
        
        // Apply overall transforms
        this.suit.scale.set(0.5, 0.5, 0.5);
        this.suit.position.y = -20;
        
        this.scene.add(this.suit);
    }
    
    buildHelmet() {
        const helmet = new THREE.Group();
        helmet.name = 'Helmet';
        
        // Main helmet shape using lathe geometry for smooth curves
        const helmetPoints = [];
        for (let i = 0; i <= 20; i++) {
            const t = i / 20;
            const angle = t * Math.PI * 0.75;
            const radius = 12 * Math.sin(angle) * (1 - t * 0.2);
            const y = 15 * Math.cos(angle) + 5;
            helmetPoints.push(new THREE.Vector2(radius, y));
        }
        
        const helmetGeometry = new THREE.LatheGeometry(helmetPoints, 32);
        const helmetMesh = new THREE.Mesh(helmetGeometry, this.materials.armorRed);
        helmetMesh.castShadow = true;
        helmetMesh.receiveShadow = true;
        helmet.add(helmetMesh);
        
        // Faceplate with complex geometry
        const faceplateShape = new THREE.Shape();
        faceplateShape.moveTo(-8, 10);
        faceplateShape.bezierCurveTo(-10, 5, -10, -5, -8, -10);
        faceplateShape.lineTo(-6, -12);
        faceplateShape.bezierCurveTo(-4, -13, 4, -13, 6, -12);
        faceplateShape.lineTo(8, -10);
        faceplateShape.bezierCurveTo(10, -5, 10, 5, 8, 10);
        faceplateShape.closePath();
        
        const faceplateGeometry = new THREE.ExtrudeGeometry(faceplateShape, {
            depth: 3,
            bevelEnabled: true,
            bevelSegments: 3,
            steps: 1,
            bevelSize: 0.5,
            bevelThickness: 0.5
        });
        
        const faceplate = new THREE.Mesh(faceplateGeometry, this.materials.armorGold);
        faceplate.position.set(0, 5, 11);
        faceplate.scale.set(1, 0.8, 1);
        helmet.add(faceplate);
        
        // Advanced eye design
        const eyeGroup = this.createAdvancedEyes();
        helmet.add(eyeGroup);
        
        // Add panel lines
        this.addPanelLines(helmet, helmetGeometry, 0x000000, 0.5);
        
        // Antenna and details
        const antennaGeo = new THREE.CylinderGeometry(0.3, 0.2, 5, 8);
        const antenna = new THREE.Mesh(antennaGeo, this.materials.brushedMetal);
        antenna.position.set(8, 15, -2);
        antenna.rotation.z = -0.3;
        helmet.add(antenna);
        
        helmet.position.y = 45;
        return helmet;
    }
    
    createAdvancedEyes() {
        const eyeGroup = new THREE.Group();
        
        // Eye shape with glow layers
        const eyeShape = new THREE.Shape();
        eyeShape.moveTo(0, 0);
        eyeShape.bezierCurveTo(2, 0, 3, 0.5, 3, 1);
        eyeShape.bezierCurveTo(3, 1.5, 2, 2, 0, 2);
        eyeShape.bezierCurveTo(-2, 2, -3, 1.5, -3, 1);
        eyeShape.bezierCurveTo(-3, 0.5, -2, 0, 0, 0);
        
        const eyeGeometry = new THREE.ExtrudeGeometry(eyeShape, {
            depth: 0.5,
            bevelEnabled: true,
            bevelSegments: 2,
            steps: 1,
            bevelSize: 0.1,
            bevelThickness: 0.1
        });
        
        // Left eye
        const leftEye = new THREE.Mesh(eyeGeometry, this.materials.eyes);
        leftEye.position.set(-4, 3, 12);
        eyeGroup.add(leftEye);
        
        // Right eye
        const rightEye = new THREE.Mesh(eyeGeometry, this.materials.eyes);
        rightEye.position.set(4, 3, 12);
        eyeGroup.add(rightEye);
        
        // Eye glow effect
        const glowGeometry = new THREE.PlaneGeometry(4, 2);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0x00CCFF,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending
        });
        
        const leftGlow = new THREE.Mesh(glowGeometry, glowMaterial);
        leftGlow.position.set(-4, 3, 12.5);
        eyeGroup.add(leftGlow);
        
        const rightGlow = new THREE.Mesh(glowGeometry, glowMaterial);
        rightGlow.position.set(4, 3, 12.5);
        eyeGroup.add(rightGlow);
        
        return eyeGroup;
    }
    
    buildChestplate() {
        const chestplate = new THREE.Group();
        chestplate.name = 'Chestplate';
        
        // Complex chest geometry
        const chestShape = new THREE.Shape();
        chestShape.moveTo(0, 20);
        chestShape.bezierCurveTo(-15, 20, -20, 15, -20, 0);
        chestShape.bezierCurveTo(-20, -15, -15, -20, 0, -20);
        chestShape.bezierCurveTo(15, -20, 20, -15, 20, 0);
        chestShape.bezierCurveTo(20, 15, 15, 20, 0, 20);
        
        const chestGeometry = new THREE.ExtrudeGeometry(chestShape, {
            depth: 15,
            bevelEnabled: true,
            bevelSegments: 5,
            steps: 2,
            bevelSize: 2,
            bevelThickness: 1
        });
        
        const chestMesh = new THREE.Mesh(chestGeometry, this.materials.armorRed);
        chestMesh.castShadow = true;
        chestplate.add(chestMesh);
        
        // Chest armor plates with details
        const platePositions = [
            { x: 0, y: 5, z: 8 },
            { x: -10, y: 0, z: 7 },
            { x: 10, y: 0, z: 7 },
            { x: -8, y: -10, z: 6 },
            { x: 8, y: -10, z: 6 }
        ];
        
        platePositions.forEach(pos => {
            const plateGeo = new THREE.BoxGeometry(8, 10, 2);
            const plate = new THREE.Mesh(plateGeo, this.materials.armorGold);
            plate.position.copy(pos);
            plate.rotation.z = pos.x * 0.02;
            chestplate.add(plate);
            
            // Add rivets
            for (let i = 0; i < 4; i++) {
                const rivetGeo = new THREE.SphereGeometry(0.3, 8, 8);
                const rivet = new THREE.Mesh(rivetGeo, this.materials.brushedMetal);
                rivet.position.set(
                    pos.x + (i % 2 - 0.5) * 6,
                    pos.y + (Math.floor(i / 2) - 0.5) * 8,
                    pos.z + 1.2
                );
                chestplate.add(rivet);
            }
        });
        
        // Arc Reactor housing
        const reactorHousing = this.buildArcReactor();
        reactorHousing.position.set(0, 5, 10);
        chestplate.add(reactorHousing);
        
        // Add mechanical details
        this.addMechanicalDetails(chestplate);
        
        chestplate.position.y = 15;
        return chestplate;
    }
    
    buildArcReactor() {
        const reactor = new THREE.Group();
        
        // Outer ring with segments
        const ringGeo = new THREE.TorusGeometry(6, 1.5, 8, 12);
        const outerRing = new THREE.Mesh(ringGeo, this.materials.brushedMetal);
        reactor.add(outerRing);
        
        // Inner rings
        for (let i = 0; i < 3; i++) {
            const innerRingGeo = new THREE.TorusGeometry(4 - i * 1.2, 0.3, 6, 32);
            const innerRing = new THREE.Mesh(innerRingGeo, this.materials.armorGold);
            innerRing.position.z = i * 0.5;
            reactor.add(innerRing);
        }
        
        // Energy core
        const coreGeo = new THREE.IcosahedronGeometry(3, 2);
        const core = new THREE.Mesh(coreGeo, this.materials.arcReactor);
        reactor.add(core);
        
        // Energy particles
        const particleCount = 20;
        const particleGeo = new THREE.SphereGeometry(0.2, 6, 6);
        
        for (let i = 0; i < particleCount; i++) {
            const particle = new THREE.Mesh(particleGeo, this.materials.arcReactor);
            const angle = (i / particleCount) * Math.PI * 2;
            const radius = 2 + Math.random() * 2;
            particle.position.set(
                Math.cos(angle) * radius,
                Math.sin(angle) * radius,
                Math.random() * 2 - 1
            );
            reactor.add(particle);
        }
        
        // Point light
        const reactorLight = new THREE.PointLight(0x00E5FF, 3, 50);
        reactorLight.position.z = 2;
        reactor.add(reactorLight);
        
        return reactor;
    }
    
    buildShoulders() {
        const shoulders = new THREE.Group();
        shoulders.name = 'Shoulders';
        
        const shoulderPositions = [-22, 22];
        
        shoulderPositions.forEach(x => {
            // Shoulder pad with complex shape
            const shoulderShape = new THREE.Shape();
            shoulderShape.moveTo(0, 0);
            shoulderShape.bezierCurveTo(5, 0, 8, 3, 8, 6);
            shoulderShape.bezierCurveTo(8, 9, 5, 12, 0, 12);
            shoulderShape.bezierCurveTo(-5, 12, -8, 9, -8, 6);
            shoulderShape.bezierCurveTo(-8, 3, -5, 0, 0, 0);
            
            const shoulderGeo = new THREE.ExtrudeGeometry(shoulderShape, {
                depth: 10,
                bevelEnabled: true,
                bevelSegments: 3,
                steps: 1,
                bevelSize: 1,
                bevelThickness: 1
            });
            
            const shoulder = new THREE.Mesh(shoulderGeo, this.materials.armorRed);
            shoulder.position.set(x, 20, 0);
            shoulder.rotation.z = x > 0 ? -0.3 : 0.3;
            shoulder.castShadow = true;
            shoulders.add(shoulder);
            
            // Shoulder armor plate
            const plateGeo = new THREE.BoxGeometry(12, 8, 10);
            const plate = new THREE.Mesh(plateGeo, this.materials.armorGold);
            plate.position.set(x * 1.1, 23, 0);
            plate.rotation.z = x > 0 ? -0.2 : 0.2;
            shoulders.add(plate);
            
            // Missile pod detail
            const podGeo = new THREE.CylinderGeometry(1.5, 1, 4, 8);
            for (let i = 0; i < 3; i++) {
                const pod = new THREE.Mesh(podGeo, this.materials.brushedMetal);
                pod.position.set(
                    x + (x > 0 ? 2 : -2),
                    22 + i * 2,
                    -3
                );
                pod.rotation.x = Math.PI / 2;
                shoulders.add(pod);
            }
        });
        
        return shoulders;
    }
    
    buildArms() {
        const arms = new THREE.Group();
        arms.name = 'Arms';
        
        const armPositions = [-20, 20];
        
        armPositions.forEach(x => {
            // Upper arm with muscle definition
            const upperArmGeo = new THREE.CylinderGeometry(5, 6, 20, 12, 1);
            const positions = upperArmGeo.attributes.position;
            
            // Add muscle bulge
            for (let i = 0; i < positions.count; i++) {
                const y = positions.getY(i);
                const angle = Math.atan2(positions.getZ(i), positions.getX(i));
                const bulge = Math.sin(y * 0.3) * 0.5;
                const radius = Math.sqrt(
                    positions.getX(i) ** 2 + positions.getZ(i) ** 2
                );
                positions.setX(i, Math.cos(angle) * (radius + bulge));
                positions.setZ(i, Math.sin(angle) * (radius + bulge));
            }
            upperArmGeo.computeVertexNormals();
            
            const upperArm = new THREE.Mesh(upperArmGeo, this.materials.armorRed);
            upperArm.position.set(x, 0, 0);
            upperArm.castShadow = true;
            arms.add(upperArm);
            
            // Elbow joint with mechanical detail
            const elbowGroup = new THREE.Group();
            
            const elbowBall = new THREE.Mesh(
                new THREE.SphereGeometry(4, 16, 16),
                this.materials.carbonFiber
            );
            elbowGroup.add(elbowBall);
            
            // Hydraulic pistons
            for (let i = 0; i < 3; i++) {
                const pistonGeo = new THREE.CylinderGeometry(0.5, 0.5, 6, 8);
                const piston = new THREE.Mesh(pistonGeo, this.materials.brushedMetal);
                const angle = (i / 3) * Math.PI * 2;
                piston.position.set(
                    Math.cos(angle) * 3,
                    0,
                    Math.sin(angle) * 3
                );
                piston.lookAt(0, 0, 0);
                elbowGroup.add(piston);
            }
            
            elbowGroup.position.set(x, -12, 0);
            arms.add(elbowGroup);
            
            // Lower arm
            const lowerArmGeo = new THREE.CylinderGeometry(4, 5, 20, 12);
            const lowerArm = new THREE.Mesh(lowerArmGeo, this.materials.armorRed);
            lowerArm.position.set(x, -25, 0);
            lowerArm.castShadow = true;
            arms.add(lowerArm);
            
            // Forearm weapon systems
            const weaponPanel = new THREE.Mesh(
                new THREE.BoxGeometry(6, 12, 4),
                this.materials.armorGold
            );
            weaponPanel.position.set(x, -25, 4);
            arms.add(weaponPanel);
            
            // Weapon barrels
            for (let i = 0; i < 3; i++) {
                const barrelGeo = new THREE.CylinderGeometry(0.3, 0.4, 8, 6);
                const barrel = new THREE.Mesh(barrelGeo, this.materials.brushedMetal);
                barrel.position.set(
                    x + (i - 1) * 1.5,
                    -25,
                    6
                );
                barrel.rotation.x = Math.PI / 2;
                arms.add(barrel);
            }
        });
        
        return arms;
    }
    
    buildGauntlets() {
        const gauntlets = new THREE.Group();
        gauntlets.name = 'Gauntlets';
        
        const handPositions = [-20, 20];
        
        handPositions.forEach(x => {
            // Hand base
            const handGeo = new THREE.BoxGeometry(8, 10, 5);
            const hand = new THREE.Mesh(handGeo, this.materials.armorRed);
            hand.position.set(x, -40, 0);
            gauntlets.add(hand);
            
            // Repulsor emitter with complex design
            const repulsorGroup = new THREE.Group();
            
            // Outer ring
            const outerRingGeo = new THREE.TorusGeometry(3, 0.5, 8, 16);
            const outerRing = new THREE.Mesh(outerRingGeo, this.materials.armorGold);
            repulsorGroup.add(outerRing);
            
            // Inner components
            const componentCount = 8;
            for (let i = 0; i < componentCount; i++) {
                const compGeo = new THREE.BoxGeometry(0.5, 0.5, 0.5);
                const comp = new THREE.Mesh(compGeo, this.materials.brushedMetal);
                const angle = (i / componentCount) * Math.PI * 2;
                comp.position.set(
                    Math.cos(angle) * 2,
                    Math.sin(angle) * 2,
                    0
                );
                repulsorGroup.add(comp);
            }
            
            // Emitter core
            const emitterGeo = new THREE.ConeGeometry(2, 2, 16);
            const emitter = new THREE.Mesh(emitterGeo, this.materials.arcReactor);
            emitter.rotation.x = Math.PI;
            repulsorGroup.add(emitter);
            
            repulsorGroup.position.set(x, -40, 3);
            repulsorGroup.rotation.x = Math.PI / 2;
            gauntlets.add(repulsorGroup);
            
            // Fingers
            const fingerPositions = [
                { x: -2, y: -45, z: 0 },
                { x: -0.7, y: -46, z: 0 },
                { x: 0.7, y: -46, z: 0 },
                { x: 2, y: -45, z: 0 }
            ];
            
            fingerPositions.forEach(pos => {
                const fingerGeo = new THREE.CylinderGeometry(0.8, 0.6, 4, 6);
                const finger = new THREE.Mesh(fingerGeo, this.materials.armorRed);
                finger.position.set(x + pos.x, pos.y, pos.z);
                gauntlets.add(finger);
                
                // Finger joints
                const jointGeo = new THREE.SphereGeometry(0.7, 8, 8);
                const joint = new THREE.Mesh(jointGeo, this.materials.carbonFiber);
                joint.position.set(x + pos.x, pos.y - 2, pos.z);
                gauntlets.add(joint);
            });
        });
        
        return gauntlets;
    }
    
    buildWaist() {
        const waist = new THREE.Group();
        waist.name = 'Waist';
        
        // Pelvis armor
        const pelvisGeo = new THREE.BoxGeometry(25, 12, 15);
        const pelvis = new THREE.Mesh(pelvisGeo, this.materials.armorRed);
        pelvis.position.y = -10;
        waist.add(pelvis);
        
        // Belt with utility modules
        const beltGeo = new THREE.TorusGeometry(14, 2, 6, 24);
        const belt = new THREE.Mesh(beltGeo, this.materials.armorGold);
        belt.position.y = -10;
        belt.rotation.x = Math.PI / 2;
        belt.scale.y = 0.5;
        waist.add(belt);
        
        // Utility pouches
        const pouchPositions = [
            { x: -12, angle: Math.PI },
            { x: -8, angle: Math.PI * 0.8 },
            { x: 8, angle: Math.PI * 1.2 },
            { x: 12, angle: 0 }
        ];
        
        pouchPositions.forEach(pos => {
            const pouchGeo = new THREE.BoxGeometry(3, 4, 2);
            const pouch = new THREE.Mesh(pouchGeo, this.materials.carbonFiber);
            pouch.position.set(pos.x, -10, 8);
            pouch.rotation.y = pos.angle;
            waist.add(pouch);
        });
        
        return waist;
    }
    
    buildLegs() {
        const legs = new THREE.Group();
        legs.name = 'Legs';
        
        const legPositions = [-8, 8];
        
        legPositions.forEach(x => {
            // Thigh with anatomical shape
            const thighShape = [];
            for (let i = 0; i <= 10; i++) {
                const t = i / 10;
                const radius = 6 - t * 1.5;
                const y = -15 - t * 20;
                thighShape.push(new THREE.Vector2(radius, y));
            }
            
            const thighGeo = new THREE.LatheGeometry(thighShape, 12);
            const thigh = new THREE.Mesh(thighGeo, this.materials.armorRed);
            thigh.position.x = x;
            thigh.castShadow = true;
            legs.add(thigh);
            
            // Thigh armor plates
            const thighPlateGeo = new THREE.BoxGeometry(8, 15, 6);
            const thighPlate = new THREE.Mesh(thighPlateGeo, this.materials.armorGold);
            thighPlate.position.set(x, -25, 5);
            legs.add(thighPlate);
            
            // Knee assembly
            const kneeGroup = new THREE.Group();
            
            // Main knee cap
            const kneeCapGeo = new THREE.SphereGeometry(5, 16, 12, 0, Math.PI);
            const kneeCap = new THREE.Mesh(kneeCapGeo, this.materials.brushedMetal);
            kneeCap.rotation.x = -Math.PI / 2;
            kneeGroup.add(kneeCap);
            
            // Knee pistons
            for (let i = 0; i < 2; i++) {
                const pistonGeo = new THREE.CylinderGeometry(0.8, 0.8, 8, 8);
                const piston = new THREE.Mesh(pistonGeo, this.materials.carbonFiber);
                piston.position.set(i * 4 - 2, 0, -3);
                piston.rotation.z = i * 0.4 - 0.2;
                kneeGroup.add(piston);
            }
            
            kneeGroup.position.set(x, -40, 0);
            legs.add(kneeGroup);
            
            // Shin
            const shinGeo = new THREE.CylinderGeometry(4.5, 5.5, 25, 12);
            const shin = new THREE.Mesh(shinGeo, this.materials.armorRed);
            shin.position.set(x, -55, 0);
            shin.castShadow = true;
            legs.add(shin);
            
            // Shin armor
            const shinPlateGeo = new THREE.BoxGeometry(7, 20, 5);
            const shinPlate = new THREE.Mesh(shinPlateGeo, this.materials.armorGold);
            shinPlate.position.set(x, -55, 4);
            legs.add(shinPlate);
        });
        
        return legs;
    }
    
    buildBoots() {
        const boots = new THREE.Group();
        boots.name = 'Boots';
        
        const bootPositions = [-8, 8];
        
        bootPositions.forEach(x => {
            // Boot base with aerodynamic design
            const bootShape = new THREE.Shape();
            bootShape.moveTo(-4, 0);
            bootShape.lineTo(-4, -5);
            bootShape.bezierCurveTo(-4, -8, -2, -10, 2, -10);
            bootShape.lineTo(8, -10);
            bootShape.bezierCurveTo(10, -10, 12, -8, 12, -5);
            bootShape.lineTo(12, 0);
            bootShape.closePath();
            
            const bootGeo = new THREE.ExtrudeGeometry(bootShape, {
                depth: 8,
                bevelEnabled: true,
                bevelSegments: 2,
                steps: 1,
                bevelSize: 0.5,
                bevelThickness: 0.5
            });
            
            const boot = new THREE.Mesh(bootGeo, this.materials.armorRed);
            boot.position.set(x, -70, 5);
            boot.rotation.x = Math.PI / 2;
            boots.add(boot);
            
            // Thruster assembly
            const thrusterGroup = new THREE.Group();
            
            // Main thruster
            const thrusterHousingGeo = new THREE.CylinderGeometry(4, 5, 6, 12);
            const thrusterHousing = new THREE.Mesh(
                thrusterHousingGeo,
                this.materials.brushedMetal
            );
            thrusterGroup.add(thrusterHousing);
            
            // Thruster nozzles
            const nozzleCount = 3;
            for (let i = 0; i < nozzleCount; i++) {
                const nozzleGeo = new THREE.ConeGeometry(1.5, 3, 8);
                const nozzle = new THREE.Mesh(nozzleGeo, this.materials.carbonFiber);
                const angle = (i / nozzleCount) * Math.PI * 2;
                nozzle.position.set(
                    Math.cos(angle) * 2,
                    -3,
                    Math.sin(angle) * 2
                );
                nozzle.rotation.x = Math.PI;
                thrusterGroup.add(nozzle);
            }
            
            // Thruster glow
            const glowGeo = new THREE.SphereGeometry(3, 16, 16);
            const glowMat = new THREE.MeshBasicMaterial({
                color: 0xFF6600,
                transparent: true,
                opacity: 0.6,
                blending: THREE.AdditiveBlending
            });
            const glow = new THREE.Mesh(glowGeo, glowMat);
            glow.position.y = -4;
            thrusterGroup.add(glow);
            
            thrusterGroup.position.set(x, -75, -5);
            thrusterGroup.rotation.x = -Math.PI / 6;
            boots.add(thrusterGroup);
            
            // Stabilizer fins
            for (let i = 0; i < 2; i++) {
                const finGeo = new THREE.BoxGeometry(1, 8, 4);
                const fin = new THREE.Mesh(finGeo, this.materials.armorGold);
                fin.position.set(
                    x + (i * 6 - 3),
                    -72,
                    -8
                );
                fin.rotation.z = i * 0.4 - 0.2;
                boots.add(fin);
            }
        });
        
        return boots;
    }
    
    addPanelLines(group, geometry, color = 0x000000, opacity = 0.3) {
        const edges = new THREE.EdgesGeometry(geometry, 15);
        const lineMaterial = new THREE.LineBasicMaterial({
            color: color,
            transparent: true,
            opacity: opacity
        });
        const lines = new THREE.LineSegments(edges, lineMaterial);
        group.add(lines);
    }
    
    addMechanicalDetails(group) {
        // Add various mechanical details like vents, bolts, panels
        const detailCount = 20;
        
        for (let i = 0; i < detailCount; i++) {
            const detailType = Math.floor(Math.random() * 3);
            
            switch (detailType) {
                case 0: // Vents
                    const ventGeo = new THREE.BoxGeometry(3, 0.5, 1);
                    const vent = new THREE.Mesh(ventGeo, this.materials.carbonFiber);
                    vent.position.set(
                        (Math.random() - 0.5) * 30,
                        (Math.random() - 0.5) * 30,
                        8 + Math.random() * 2
                    );
                    group.add(vent);
                    break;
                    
                case 1: // Bolts
                    const boltGeo = new THREE.CylinderGeometry(0.3, 0.3, 0.5, 6);
                    const bolt = new THREE.Mesh(boltGeo, this.materials.brushedMetal);
                    bolt.position.set(
                        (Math.random() - 0.5) * 35,
                        (Math.random() - 0.5) * 35,
                        10
                    );
                    bolt.rotation.x = Math.PI / 2;
                    group.add(bolt);
                    break;
                    
                case 2: // Small panels
                    const panelGeo = new THREE.BoxGeometry(
                        2 + Math.random() * 2,
                        2 + Math.random() * 2,
                        0.5
                    );
                    const panel = new THREE.Mesh(panelGeo, this.materials.carbonFiber);
                    panel.position.set(
                        (Math.random() - 0.5) * 30,
                        (Math.random() - 0.5) * 30,
                        9
                    );
                    group.add(panel);
                    break;
            }
        }
    }
    
    addEffects() {
        // Add particle systems and other effects
        this.createThrusterParticles();
        this.createEnergyField();
        this.createHolographicDisplay();
    }
    
    createThrusterParticles() {
        // Particle system for thruster effects
        const particleCount = 100;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);
        
        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = 0;
            positions[i * 3 + 1] = 0;
            positions[i * 3 + 2] = 0;
            
            colors[i * 3] = 1;
            colors[i * 3 + 1] = 0.6;
            colors[i * 3 + 2] = 0;
            
            sizes[i] = Math.random() * 2 + 1;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        const material = new THREE.PointsMaterial({
            size: 2,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true,
            opacity: 0.8
        });
        
        const particles = new THREE.Points(geometry, material);
        this.effects.push({ type: 'thruster', particles: particles });
    }
    
    createEnergyField() {
        // Energy shield effect
        const shieldGeo = new THREE.SphereGeometry(50, 32, 32);
        const shieldMat = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                color: { value: new THREE.Color(0x0088FF) },
                opacity: { value: 0.1 }
            },
            vertexShader: `
                varying vec3 vNormal;
                varying vec3 vPosition;
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    vPosition = position;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 color;
                uniform float opacity;
                varying vec3 vNormal;
                varying vec3 vPosition;
                
                void main() {
                    float fresnel = 1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0));
                    fresnel = pow(fresnel, 3.0);
                    
                    float wave = sin(vPosition.y * 0.1 + time * 2.0) * 0.5 + 0.5;
                    float hexPattern = sin(vPosition.x * 10.0) * sin(vPosition.y * 10.0) * sin(vPosition.z * 10.0);
                    
                    float alpha = fresnel * opacity * wave * (0.5 + hexPattern * 0.5);
                    gl_FragColor = vec4(color, alpha);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });
        
        const shield = new THREE.Mesh(shieldGeo, shieldMat);
        shield.visible = false; // Enable during combat
        this.effects.push({ type: 'shield', mesh: shield });
    }
    
    createHolographicDisplay() {
        // HUD holographic projections
        const holoGeo = new THREE.PlaneGeometry(20, 20);
        const holoMat = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                texture: { value: null }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform sampler2D texture;
                varying vec2 vUv;
                
                void main() {
                    vec2 uv = vUv;
                    uv.y += sin(uv.x * 50.0 + time * 5.0) * 0.01;
                    
                    vec4 color = vec4(0.0, 1.0, 1.0, 1.0);
                    float scanline = sin(uv.y * 100.0 + time * 10.0) * 0.04 + 0.96;
                    float flicker = sin(time * 30.0) * 0.03 + 0.97;
                    
                    color.a = scanline * flicker * 0.6;
                    gl_FragColor = color;
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });
        
        const holo = new THREE.Mesh(holoGeo, holoMat);
        holo.position.set(30, 20, 0);
        holo.visible = false; // Enable when needed
        this.effects.push({ type: 'hologram', mesh: holo });
    }
    
    setupAnimations() {
        // Create animation clips for various actions
        this.createIdleAnimation();
        this.createFlightAnimation();
        this.createCombatAnimation();
        this.createAssemblyAnimation();
    }
    
    createIdleAnimation() {
        // Subtle breathing and reactor pulsing
        const tracks = [];
        
        // Chest breathing
        const chestTrack = new THREE.KeyframeTrack(
            '.scale',
            [0, 2, 4],
            [1, 1, 1, 1.02, 1.02, 1.02, 1, 1, 1]
        );
        tracks.push(chestTrack);
        
        const idleClip = new THREE.AnimationClip('idle', 4, tracks);
        return idleClip;
    }
    
    createFlightAnimation() {
        // Flight pose adjustments
        const tracks = [];
        
        // Lean forward
        const rotationTrack = new THREE.KeyframeTrack(
            '.rotation[x]',
            [0, 0.5],
            [0, -0.3]
        );
        tracks.push(rotationTrack);
        
        const flightClip = new THREE.AnimationClip('flight', 0.5, tracks);
        return flightClip;
    }
    
    createCombatAnimation() {
        // Combat ready stance
        const tracks = [];
        
        // Arms positioning
        const leftArmTrack = new THREE.KeyframeTrack(
            'Arms.rotation[z]',
            [0, 0.3],
            [0, -0.5]
        );
        tracks.push(leftArmTrack);
        
        const combatClip = new THREE.AnimationClip('combat', 0.3, tracks);
        return combatClip;
    }
    
    createAssemblyAnimation() {
        // Suit assembly sequence
        const duration = 3;
        const tracks = [];
        
        // Parts flying in from different positions
        const parts = ['Helmet', 'Chestplate', 'Arms', 'Legs', 'Boots'];
        
        parts.forEach((part, index) => {
            const delay = index * 0.5;
            const posTrack = new THREE.KeyframeTrack(
                `${part}.position`,
                [0, delay, delay + 0.5],
                [
                    0, 100, 0,  // Start position
                    0, 100, 0,  // Hold
                    0, 0, 0     // Final position
                ]
            );
            tracks.push(posTrack);
        });
        
        const assemblyClip = new THREE.AnimationClip('assembly', duration, tracks);
        return assemblyClip;
    }
    
    update(deltaTime, elapsedTime) {
        // Update all animations and effects
        this.animationMixers.forEach(mixer => {
            mixer.update(deltaTime);
        });
        
        // Update shader uniforms
        this.effects.forEach(effect => {
            if (effect.mesh && effect.mesh.material.uniforms.time) {
                effect.mesh.material.uniforms.time.value = elapsedTime;
            }
        });
        
        // Rotate arc reactor core
        const arcReactor = this.suit.getObjectByName('ArcReactor');
        if (arcReactor) {
            const core = arcReactor.children.find(child => 
                child.geometry && child.geometry.type === 'IcosahedronGeometry'
            );
            if (core) {
                core.rotation.x = elapsedTime * 0.5;
                core.rotation.y = elapsedTime * 0.7;
            }
        }
        
        // Pulse arc reactor light
        const reactorLight = arcReactor?.children.find(child => 
            child instanceof THREE.PointLight
        );
        if (reactorLight) {
            reactorLight.intensity = 2 + Math.sin(elapsedTime * 3) * 0.5;
        }
    }
    
    enableCombatMode() {
        // Activate combat systems
        const shield = this.effects.find(e => e.type === 'shield');
        if (shield) shield.mesh.visible = true;
        
        // Change materials to combat mode
        this.materials.armorRed.emissiveIntensity = 0.2;
        this.materials.eyes.emissiveIntensity = 8.0;
    }
    
    disableCombatMode() {
        // Deactivate combat systems
        const shield = this.effects.find(e => e.type === 'shield');
        if (shield) shield.mesh.visible = false;
        
        // Reset materials
        this.materials.armorRed.emissiveIntensity = 0.05;
        this.materials.eyes.emissiveIntensity = 5.0;
    }
    
    getSuit() {
        return this.suit;
    }
}

// Export for use
window.UltraRealisticIronManSuit = UltraRealisticIronManSuit;