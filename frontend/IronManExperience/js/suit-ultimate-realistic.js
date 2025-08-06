// Ultimate Realistic Iron Man Suit - Cinema Quality
// Advanced geometry, PBR materials, and movie-accurate details

class UltimateRealisticIronManSuit {
    constructor(scene) {
        this.scene = scene;
        this.suit = new THREE.Group();
        this.suit.name = 'IronManSuit';
        
        // Suit components
        this.parts = {};
        this.materials = {};
        this.lights = [];
        this.animationMixers = [];
        
        // Animation states
        this.animationState = {
            arcReactorPulse: 0,
            eyeGlow: 1.0,
            repulsorCharge: 0,
            thrusterGlow: 0,
            time: 0
        };
        
        this.init();
    }
    
    init() {
        console.log('Building Ultimate Iron Man Suit...');
        this.createAdvancedMaterials();
        this.buildDetailedSuit();
        this.addSuitLighting();
        this.addDetailTextures();
        this.scene.add(this.suit);
    }
    
    createAdvancedMaterials() {
        // Main armor - Premium automotive metallic red
        this.materials.armorRed = new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(0.55, 0.05, 0.05),
            metalness: 0.85,
            roughness: 0.12,
            clearcoat: 1.0,
            clearcoatRoughness: 0.03,
            reflectivity: 0.95,
            sheen: 1.0,
            sheenRoughness: 0.05,
            sheenColor: new THREE.Color(1, 0.2, 0.2),
            emissive: new THREE.Color(0.1, 0, 0),
            emissiveIntensity: 0.1,
            envMapIntensity: 1.8
        });
        
        // Gold titanium alloy - Highly reflective gold
        this.materials.armorGold = new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(0.9, 0.7, 0.15),
            metalness: 0.92,
            roughness: 0.08,
            clearcoat: 0.9,
            clearcoatRoughness: 0.05,
            reflectivity: 1.0,
            sheen: 1.0,
            sheenRoughness: 0.1,
            sheenColor: new THREE.Color(1, 0.9, 0.4),
            emissive: new THREE.Color(0.2, 0.15, 0),
            emissiveIntensity: 0.15,
            envMapIntensity: 2.2
        });
        
        // Arc Reactor - Advanced plasma energy
        this.materials.arcReactor = new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(0.3, 0.9, 1.0),
            emissive: new THREE.Color(0, 0.8, 1.0),
            emissiveIntensity: 5.0,
            metalness: 0.1,
            roughness: 0.0,
            transmission: 0.9,
            thickness: 1.0,
            ior: 1.45,
            attenuationDistance: 0.5,
            attenuationColor: new THREE.Color(0, 0.8, 1),
            transparent: true,
            opacity: 0.95,
            side: THREE.DoubleSide
        });
        
        // Eyes - Holographic display
        this.materials.eyes = new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(1, 1, 1),
            emissive: new THREE.Color(0, 0.7, 1.0),
            emissiveIntensity: 6.0,
            metalness: 0.0,
            roughness: 0.0,
            transmission: 0.6,
            thickness: 0.3,
            transparent: true,
            opacity: 0.9
        });
        
        // Dark metal for joints and mechanical parts
        this.materials.darkMetal = new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(0.08, 0.08, 0.08),
            metalness: 0.9,
            roughness: 0.4,
            clearcoat: 0.3,
            clearcoatRoughness: 0.2
        });
        
        // Repulsor glow material
        this.materials.repulsor = new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(0.5, 0.9, 1.0),
            emissive: new THREE.Color(0, 0.8, 1.0),
            emissiveIntensity: 8.0,
            metalness: 0.0,
            roughness: 0.0,
            transmission: 0.8,
            transparent: true,
            opacity: 0.9
        });
        
        // Silver accents
        this.materials.silver = new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(0.75, 0.75, 0.75),
            metalness: 0.95,
            roughness: 0.15,
            envMapIntensity: 2.0
        });
    }
    
    buildDetailedSuit() {
        // Build suit with movie-accurate proportions and details
        
        // HELMET
        this.parts.helmet = this.buildDetailedHelmet();
        this.parts.helmet.position.y = 55;
        this.suit.add(this.parts.helmet);
        
        // TORSO
        this.parts.torso = this.buildDetailedTorso();
        this.parts.torso.position.y = 35;
        this.suit.add(this.parts.torso);
        
        // SHOULDERS
        this.parts.leftShoulder = this.buildShoulder('left');
        this.parts.leftShoulder.position.set(-20, 48, 0);
        this.suit.add(this.parts.leftShoulder);
        
        this.parts.rightShoulder = this.buildShoulder('right');
        this.parts.rightShoulder.position.set(20, 48, 0);
        this.suit.add(this.parts.rightShoulder);
        
        // ARMS
        this.buildArms();
        
        // WAIST & PELVIS
        this.parts.waist = this.buildWaist();
        this.parts.waist.position.y = 20;
        this.suit.add(this.parts.waist);
        
        // LEGS
        this.buildLegs();
        
        // Add panel lines and detail
        this.addPanelDetails();
    }
    
    buildDetailedHelmet() {
        const helmet = new THREE.Group();
        
        // Main helmet shape - more accurate proportions
        const helmetGeometry = new THREE.SphereGeometry(10, 64, 48);
        helmetGeometry.scale(1.0, 1.15, 0.92);
        
        // Modify vertices for more accurate shape
        const positions = helmetGeometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const y = positions.getY(i);
            const z = positions.getZ(i);
            
            // Flatten the back
            if (z < -8) {
                positions.setZ(i, z * 0.7);
            }
            
            // Create jawline
            if (y < -2 && Math.abs(x) > 6) {
                positions.setY(i, y * 0.85);
            }
        }
        helmetGeometry.computeVertexNormals();
        
        const helmetMesh = new THREE.Mesh(helmetGeometry, this.materials.armorRed);
        helmetMesh.castShadow = true;
        helmetMesh.receiveShadow = true;
        helmet.add(helmetMesh);
        
        // Faceplate - more detailed
        const faceplateGroup = new THREE.Group();
        
        // Main faceplate
        const faceplateGeometry = new THREE.BoxGeometry(8.5, 10, 0.5);
        faceplateGeometry.translate(0, -1, 9.5);
        const faceplate = new THREE.Mesh(faceplateGeometry, this.materials.armorGold);
        faceplateGroup.add(faceplate);
        
        // Faceplate details - forehead piece
        const foreheadGeometry = new THREE.BoxGeometry(7, 3, 0.3);
        foreheadGeometry.translate(0, 3.5, 9.7);
        const forehead = new THREE.Mesh(foreheadGeometry, this.materials.armorGold);
        faceplateGroup.add(forehead);
        
        // Chin piece
        const chinGeometry = new THREE.BoxGeometry(6, 3.5, 0.3);
        chinGeometry.translate(0, -5, 9.7);
        const chin = new THREE.Mesh(chinGeometry, this.materials.armorGold);
        faceplateGroup.add(chin);
        
        helmet.add(faceplateGroup);
        
        // Eyes - more realistic shape and glow
        const eyeGroup = new THREE.Group();
        
        // Eye shapes - trapezoid-like
        const eyeShape = new THREE.Shape();
        eyeShape.moveTo(-1.5, 0.5);
        eyeShape.lineTo(-1.2, -0.5);
        eyeShape.lineTo(1.2, -0.5);
        eyeShape.lineTo(1.5, 0.5);
        eyeShape.closePath();
        
        const eyeGeometry = new THREE.ExtrudeGeometry(eyeShape, {
            depth: 0.5,
            bevelEnabled: true,
            bevelThickness: 0.1,
            bevelSize: 0.1,
            bevelSegments: 8
        });
        
        // Left eye
        const leftEye = new THREE.Mesh(eyeGeometry, this.materials.eyes);
        leftEye.position.set(-3.2, 1.5, 10);
        leftEye.rotation.z = -0.05;
        eyeGroup.add(leftEye);
        
        // Right eye
        const rightEye = new THREE.Mesh(eyeGeometry, this.materials.eyes);
        rightEye.position.set(3.2, 1.5, 10);
        rightEye.rotation.z = 0.05;
        eyeGroup.add(rightEye);
        
        // Eye glow lights
        const leftEyeLight = new THREE.PointLight(0x00ccff, 0.8, 30);
        leftEyeLight.position.set(-3.2, 1.5, 11);
        eyeGroup.add(leftEyeLight);
        this.lights.push(leftEyeLight);
        
        const rightEyeLight = new THREE.PointLight(0x00ccff, 0.8, 30);
        rightEyeLight.position.set(3.2, 1.5, 11);
        eyeGroup.add(rightEyeLight);
        this.lights.push(rightEyeLight);
        
        helmet.add(eyeGroup);
        this.parts.eyes = eyeGroup;
        
        // Helmet details - air intakes
        const intakeGeometry = new THREE.BoxGeometry(2, 4, 1);
        const leftIntake = new THREE.Mesh(intakeGeometry, this.materials.darkMetal);
        leftIntake.position.set(-8.5, 0, 3);
        leftIntake.rotation.y = -0.3;
        helmet.add(leftIntake);
        
        const rightIntake = new THREE.Mesh(intakeGeometry, this.materials.darkMetal);
        rightIntake.position.set(8.5, 0, 3);
        rightIntake.rotation.y = 0.3;
        helmet.add(rightIntake);
        
        // Antenna/sensor
        const antennaGeometry = new THREE.CylinderGeometry(0.2, 0.3, 3, 8);
        const antenna = new THREE.Mesh(antennaGeometry, this.materials.silver);
        antenna.position.set(7, 8, -2);
        antenna.rotation.z = -0.2;
        helmet.add(antenna);
        
        return helmet;
    }
    
    buildDetailedTorso() {
        const torso = new THREE.Group();
        
        // Main chest piece - more anatomically correct
        const chestGeometry = new THREE.BoxGeometry(26, 32, 16);
        
        // Modify for more realistic shape
        const positions = chestGeometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const y = positions.getY(i);
            const z = positions.getZ(i);
            
            // Taper towards waist
            if (y < -10) {
                const factor = 1 - (Math.abs(y + 10) / 22) * 0.3;
                positions.setX(i, x * factor);
            }
            
            // Round the chest
            if (z > 6 && Math.abs(x) > 10) {
                positions.setZ(i, z - Math.abs(x - 10) * 0.2);
            }
        }
        chestGeometry.computeVertexNormals();
        
        const chestMesh = new THREE.Mesh(chestGeometry, this.materials.armorRed);
        chestMesh.castShadow = true;
        chestMesh.receiveShadow = true;
        torso.add(chestMesh);
        
        // Chest armor plates
        this.addChestPlates(torso);
        
        // Arc Reactor housing
        const arcReactorGroup = this.buildArcReactor();
        arcReactorGroup.position.set(0, 5, 9);
        torso.add(arcReactorGroup);
        this.parts.arcReactor = arcReactorGroup;
        
        // Back details
        this.addBackDetails(torso);
        
        // Side panels
        const sidePanelGeometry = new THREE.BoxGeometry(3, 20, 12);
        const leftSidePanel = new THREE.Mesh(sidePanelGeometry, this.materials.darkMetal);
        leftSidePanel.position.set(-14, 0, 0);
        torso.add(leftSidePanel);
        
        const rightSidePanel = new THREE.Mesh(sidePanelGeometry, this.materials.darkMetal);
        rightSidePanel.position.set(14, 0, 0);
        torso.add(rightSidePanel);
        
        return torso;
    }
    
    buildArcReactor() {
        const arcReactor = new THREE.Group();
        
        // Outer housing
        const housingGeometry = new THREE.CylinderGeometry(5.5, 5.5, 2, 32);
        const housing = new THREE.Mesh(housingGeometry, this.materials.silver);
        housing.rotation.x = Math.PI / 2;
        arcReactor.add(housing);
        
        // Inner rings
        for (let i = 0; i < 3; i++) {
            const radius = 4.5 - i * 1.2;
            const ringGeometry = new THREE.TorusGeometry(radius, 0.3, 8, 32);
            const ring = new THREE.Mesh(ringGeometry, this.materials.armorGold);
            ring.position.z = 0.5 + i * 0.3;
            arcReactor.add(ring);
        }
        
        // Core
        const coreGeometry = new THREE.SphereGeometry(2, 32, 32);
        const core = new THREE.Mesh(coreGeometry, this.materials.arcReactor);
        core.position.z = 1;
        arcReactor.add(core);
        
        // Energy field
        const fieldGeometry = new THREE.ConeGeometry(3, 1, 32, 1, true);
        const field = new THREE.Mesh(fieldGeometry, this.materials.arcReactor);
        field.rotation.x = -Math.PI / 2;
        field.position.z = 1.5;
        field.material.opacity = 0.5;
        arcReactor.add(field);
        
        // Arc reactor light
        const reactorLight = new THREE.PointLight(0x00ccff, 3, 60);
        reactorLight.position.z = 3;
        arcReactor.add(reactorLight);
        this.lights.push(reactorLight);
        
        // Spot light for focused glow
        const reactorSpot = new THREE.SpotLight(0x00ccff, 2, 40, Math.PI / 4, 0.5);
        reactorSpot.position.z = 2;
        reactorSpot.target.position.z = 10;
        arcReactor.add(reactorSpot);
        arcReactor.add(reactorSpot.target);
        
        return arcReactor;
    }
    
    addChestPlates(torso) {
        // Upper chest plates
        const upperPlateGeometry = new THREE.BoxGeometry(10, 8, 1.5);
        
        const centerPlate = new THREE.Mesh(upperPlateGeometry, this.materials.armorGold);
        centerPlate.position.set(0, 10, 8.5);
        torso.add(centerPlate);
        
        // Angled side plates
        const sidePlateGeometry = new THREE.BoxGeometry(6, 10, 1.5);
        
        const leftPlate = new THREE.Mesh(sidePlateGeometry, this.materials.armorGold);
        leftPlate.position.set(-9, 5, 8);
        leftPlate.rotation.y = 0.2;
        torso.add(leftPlate);
        
        const rightPlate = new THREE.Mesh(sidePlateGeometry, this.materials.armorGold);
        rightPlate.position.set(9, 5, 8);
        rightPlate.rotation.y = -0.2;
        torso.add(rightPlate);
        
        // Lower chest details
        const lowerPlateGeometry = new THREE.BoxGeometry(18, 4, 1);
        const lowerPlate = new THREE.Mesh(lowerPlateGeometry, this.materials.armorGold);
        lowerPlate.position.set(0, -8, 8.5);
        torso.add(lowerPlate);
        
        // Chest vents
        const ventGeometry = new THREE.BoxGeometry(2, 6, 1);
        for (let i = 0; i < 3; i++) {
            const leftVent = new THREE.Mesh(ventGeometry, this.materials.darkMetal);
            leftVent.position.set(-11, -2 - i * 3, 7.5);
            torso.add(leftVent);
            
            const rightVent = new THREE.Mesh(ventGeometry, this.materials.darkMetal);
            rightVent.position.set(11, -2 - i * 3, 7.5);
            torso.add(rightVent);
        }
    }
    
    addBackDetails(torso) {
        // Back armor plates
        const backPlateGeometry = new THREE.BoxGeometry(20, 25, 1.5);
        const backPlate = new THREE.Mesh(backPlateGeometry, this.materials.armorRed);
        backPlate.position.set(0, 2, -8);
        torso.add(backPlate);
        
        // Jet pack mount points
        const mountGeometry = new THREE.CylinderGeometry(3, 3, 2, 16);
        const leftMount = new THREE.Mesh(mountGeometry, this.materials.darkMetal);
        leftMount.position.set(-8, 8, -8.5);
        leftMount.rotation.x = Math.PI / 2;
        torso.add(leftMount);
        
        const rightMount = new THREE.Mesh(mountGeometry, this.materials.darkMetal);
        rightMount.position.set(8, 8, -8.5);
        rightMount.rotation.x = Math.PI / 2;
        torso.add(rightMount);
        
        // Spine detail
        const spineGeometry = new THREE.BoxGeometry(3, 24, 2);
        const spine = new THREE.Mesh(spineGeometry, this.materials.silver);
        spine.position.set(0, 0, -9);
        torso.add(spine);
    }
    
    buildShoulder(side) {
        const shoulder = new THREE.Group();
        
        // Main shoulder pad - rounded design
        const shoulderGeometry = new THREE.SphereGeometry(9, 32, 24);
        shoulderGeometry.scale(1.3, 1.0, 1.0);
        
        // Cut bottom half
        const positions = shoulderGeometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const y = positions.getY(i);
            if (y < -2) {
                positions.setY(i, -2);
            }
        }
        shoulderGeometry.computeVertexNormals();
        
        const shoulderMesh = new THREE.Mesh(shoulderGeometry, this.materials.armorRed);
        shoulderMesh.castShadow = true;
        shoulder.add(shoulderMesh);
        
        // Shoulder plate detail
        const plateGeometry = new THREE.BoxGeometry(8, 10, 1);
        const plate = new THREE.Mesh(plateGeometry, this.materials.armorGold);
        plate.position.set(side === 'left' ? -4 : 4, 0, 7);
        plate.rotation.y = side === 'left' ? -0.3 : 0.3;
        shoulder.add(plate);
        
        // Shoulder joint
        const jointGeometry = new THREE.SphereGeometry(5, 16, 16);
        const joint = new THREE.Mesh(jointGeometry, this.materials.darkMetal);
        joint.position.y = -5;
        shoulder.add(joint);
        
        return shoulder;
    }
    
    buildArms() {
        // Left arm
        this.buildArm('left', -20);
        
        // Right arm  
        this.buildArm('right', 20);
    }
    
    buildArm(side, xPos) {
        // Upper arm
        const upperArmGroup = new THREE.Group();
        
        // Bicep
        const bicepGeometry = new THREE.CylinderGeometry(5.5, 6.5, 18, 16);
        const bicep = new THREE.Mesh(bicepGeometry, this.materials.armorRed);
        bicep.castShadow = true;
        upperArmGroup.add(bicep);
        
        // Upper arm details
        const upperDetailGeometry = new THREE.BoxGeometry(4, 12, 1);
        const upperDetail = new THREE.Mesh(upperDetailGeometry, this.materials.armorGold);
        upperDetail.position.set(0, 0, 5.5);
        upperArmGroup.add(upperDetail);
        
        upperArmGroup.position.set(xPos, 33, 0);
        this.suit.add(upperArmGroup);
        
        // Elbow joint
        const elbowGeometry = new THREE.SphereGeometry(4.5, 16, 16);
        const elbow = new THREE.Mesh(elbowGeometry, this.materials.darkMetal);
        elbow.position.set(xPos, 24, 0);
        this.suit.add(elbow);
        
        // Forearm
        const forearmGroup = new THREE.Group();
        
        const forearmGeometry = new THREE.CylinderGeometry(4, 5, 18, 16);
        const forearm = new THREE.Mesh(forearmGeometry, this.materials.armorRed);
        forearm.castShadow = true;
        forearmGroup.add(forearm);
        
        // Forearm panel
        const forearmPanelGeometry = new THREE.BoxGeometry(3.5, 14, 1);
        const forearmPanel = new THREE.Mesh(forearmPanelGeometry, this.materials.armorGold);
        forearmPanel.position.set(0, 0, 4.5);
        forearmGroup.add(forearmPanel);
        
        forearmGroup.position.set(xPos, 15, 0);
        this.suit.add(forearmGroup);
        
        // Hand and repulsor
        this.buildHand(side, xPos);
    }
    
    buildHand(side, xPos) {
        const handGroup = new THREE.Group();
        
        // Palm
        const palmGeometry = new THREE.BoxGeometry(7, 8, 5);
        const palm = new THREE.Mesh(palmGeometry, this.materials.armorGold);
        handGroup.add(palm);
        
        // Fingers (simplified)
        const fingerGeometry = new THREE.BoxGeometry(1.2, 4, 1);
        for (let i = 0; i < 4; i++) {
            const finger = new THREE.Mesh(fingerGeometry, this.materials.armorRed);
            finger.position.set(-2.5 + i * 1.7, -5, 0);
            handGroup.add(finger);
        }
        
        // Thumb
        const thumbGeometry = new THREE.BoxGeometry(1.5, 3, 1.2);
        const thumb = new THREE.Mesh(thumbGeometry, this.materials.armorRed);
        thumb.position.set(side === 'left' ? 3.5 : -3.5, -2, 0);
        thumb.rotation.z = side === 'left' ? -0.5 : 0.5;
        handGroup.add(thumb);
        
        // Repulsor
        const repulsorGroup = new THREE.Group();
        
        const repulsorRingGeometry = new THREE.TorusGeometry(2.5, 0.5, 8, 16);
        const repulsorRing = new THREE.Mesh(repulsorRingGeometry, this.materials.silver);
        repulsorGroup.add(repulsorRing);
        
        const repulsorCoreGeometry = new THREE.CylinderGeometry(2, 2, 0.5, 16);
        const repulsorCore = new THREE.Mesh(repulsorCoreGeometry, this.materials.repulsor);
        repulsorCore.rotation.x = Math.PI / 2;
        repulsorGroup.add(repulsorCore);
        
        // Repulsor light
        const repulsorLight = new THREE.SpotLight(0x00ccff, 1, 30, Math.PI / 3, 0.5);
        repulsorLight.position.z = 1;
        repulsorLight.target.position.z = 10;
        repulsorGroup.add(repulsorLight);
        repulsorGroup.add(repulsorLight.target);
        this.lights.push(repulsorLight);
        
        repulsorGroup.position.z = 3;
        handGroup.add(repulsorGroup);
        
        handGroup.position.set(xPos, 6, 0);
        this.suit.add(handGroup);
        
        if (side === 'left') {
            this.parts.leftRepulsor = repulsorGroup;
        } else {
            this.parts.rightRepulsor = repulsorGroup;
        }
    }
    
    buildWaist() {
        const waist = new THREE.Group();
        
        // Core waist piece
        const waistGeometry = new THREE.BoxGeometry(22, 10, 14);
        const waistMesh = new THREE.Mesh(waistGeometry, this.materials.armorRed);
        waistMesh.castShadow = true;
        waist.add(waistMesh);
        
        // Belt
        const beltGeometry = new THREE.BoxGeometry(24, 4, 15);
        const belt = new THREE.Mesh(beltGeometry, this.materials.armorGold);
        belt.position.y = 0;
        waist.add(belt);
        
        // Belt buckle
        const buckleGeometry = new THREE.BoxGeometry(6, 5, 1);
        const buckle = new THREE.Mesh(buckleGeometry, this.materials.silver);
        buckle.position.set(0, 0, 7.5);
        waist.add(buckle);
        
        // Hip joints
        const hipJointGeometry = new THREE.SphereGeometry(5, 16, 16);
        const leftHip = new THREE.Mesh(hipJointGeometry, this.materials.darkMetal);
        leftHip.position.set(-9, -3, 0);
        waist.add(leftHip);
        
        const rightHip = new THREE.Mesh(hipJointGeometry, this.materials.darkMetal);
        rightHip.position.set(9, -3, 0);
        waist.add(rightHip);
        
        // Pelvis armor
        const pelvisGeometry = new THREE.BoxGeometry(18, 8, 12);
        const pelvis = new THREE.Mesh(pelvisGeometry, this.materials.armorRed);
        pelvis.position.y = -7;
        waist.add(pelvis);
        
        return waist;
    }
    
    buildLegs() {
        // Left leg
        this.buildLeg('left', -9);
        
        // Right leg
        this.buildLeg('right', 9);
    }
    
    buildLeg(side, xPos) {
        // Thigh
        const thighGroup = new THREE.Group();
        
        const thighGeometry = new THREE.CylinderGeometry(6.5, 7.5, 20, 16);
        const thigh = new THREE.Mesh(thighGeometry, this.materials.armorRed);
        thigh.castShadow = true;
        thighGroup.add(thigh);
        
        // Thigh plate
        const thighPlateGeometry = new THREE.BoxGeometry(5, 16, 1);
        const thighPlate = new THREE.Mesh(thighPlateGeometry, this.materials.armorGold);
        thighPlate.position.set(0, 0, 6.5);
        thighGroup.add(thighPlate);
        
        thighGroup.position.set(xPos, 8, 0);
        this.suit.add(thighGroup);
        
        // Knee
        const kneeGroup = new THREE.Group();
        
        const kneeGeometry = new THREE.SphereGeometry(5, 16, 16);
        const knee = new THREE.Mesh(kneeGeometry, this.materials.darkMetal);
        kneeGroup.add(knee);
        
        // Knee cap
        const kneeCapGeometry = new THREE.BoxGeometry(6, 6, 2);
        const kneeCap = new THREE.Mesh(kneeCapGeometry, this.materials.armorGold);
        kneeCap.position.z = 4.5;
        kneeGroup.add(kneeCap);
        
        kneeGroup.position.set(xPos, -2, 0);
        this.suit.add(kneeGroup);
        
        // Shin
        const shinGroup = new THREE.Group();
        
        const shinGeometry = new THREE.CylinderGeometry(5, 6, 18, 16);
        const shin = new THREE.Mesh(shinGeometry, this.materials.armorRed);
        shin.castShadow = true;
        shinGroup.add(shin);
        
        // Shin guard
        const shinGuardGeometry = new THREE.BoxGeometry(4.5, 14, 1.5);
        const shinGuard = new THREE.Mesh(shinGuardGeometry, this.materials.armorGold);
        shinGuard.position.set(0, 0, 5.5);
        shinGroup.add(shinGuard);
        
        shinGroup.position.set(xPos, -11, 0);
        this.suit.add(shinGroup);
        
        // Boot
        this.buildBoot(side, xPos);
    }
    
    buildBoot(side, xPos) {
        const bootGroup = new THREE.Group();
        
        // Main boot
        const bootGeometry = new THREE.BoxGeometry(9, 8, 14);
        const boot = new THREE.Mesh(bootGeometry, this.materials.armorRed);
        boot.castShadow = true;
        bootGroup.add(boot);
        
        // Boot toe
        const toeGeometry = new THREE.BoxGeometry(8, 6, 4);
        const toe = new THREE.Mesh(toeGeometry, this.materials.armorGold);
        toe.position.set(0, -1, 7);
        bootGroup.add(toe);
        
        // Ankle detail
        const ankleGeometry = new THREE.CylinderGeometry(4.5, 5, 4, 16);
        const ankle = new THREE.Mesh(ankleGeometry, this.materials.darkMetal);
        ankle.position.y = 5;
        bootGroup.add(ankle);
        
        // Heel thruster
        const thrusterGroup = new THREE.Group();
        
        const thrusterHousingGeometry = new THREE.CylinderGeometry(3, 3.5, 5, 16);
        const thrusterHousing = new THREE.Mesh(thrusterHousingGeometry, this.materials.silver);
        thrusterHousing.rotation.x = Math.PI / 2;
        thrusterGroup.add(thrusterHousing);
        
        const thrusterCoreGeometry = new THREE.CylinderGeometry(2.5, 2.5, 4, 16);
        const thrusterCore = new THREE.Mesh(thrusterCoreGeometry, this.materials.repulsor);
        thrusterCore.rotation.x = Math.PI / 2;
        thrusterGroup.add(thrusterCore);
        
        // Thruster light
        const thrusterLight = new THREE.SpotLight(0xff4400, 0.5, 20, Math.PI / 3, 0.5);
        thrusterLight.position.z = -2;
        thrusterLight.target.position.z = -10;
        thrusterGroup.add(thrusterLight);
        thrusterGroup.add(thrusterLight.target);
        this.lights.push(thrusterLight);
        
        thrusterGroup.position.set(0, -2, -6);
        bootGroup.add(thrusterGroup);
        
        bootGroup.position.set(xPos, -24, 2);
        this.suit.add(bootGroup);
        
        if (side === 'left') {
            this.parts.leftThruster = thrusterGroup;
        } else {
            this.parts.rightThruster = thrusterGroup;
        }
    }
    
    addPanelDetails() {
        // Add edge lines to all armor pieces for panel definition
        this.suit.traverse((child) => {
            if (child.isMesh && (child.material === this.materials.armorRed || 
                               child.material === this.materials.armorGold)) {
                // Create edge geometry
                const edges = new THREE.EdgesGeometry(child.geometry, 30);
                const lineMaterial = new THREE.LineBasicMaterial({ 
                    color: 0x000000,
                    transparent: true,
                    opacity: 0.2,
                    linewidth: 1
                });
                const lineSegments = new THREE.LineSegments(edges, lineMaterial);
                child.add(lineSegments);
            }
        });
    }
    
    addSuitLighting() {
        // Additional ambient lights for the suit
        const suitLight1 = new THREE.PointLight(0xffffff, 0.3, 100);
        suitLight1.position.set(0, 60, 50);
        this.suit.add(suitLight1);
        
        const suitLight2 = new THREE.PointLight(0x4080ff, 0.2, 80);
        suitLight2.position.set(-40, 30, -30);
        this.suit.add(suitLight2);
    }
    
    addDetailTextures() {
        // Could add normal maps, roughness maps, etc. here if textures were available
        // For now, we're using procedural materials
    }
    
    update(deltaTime, time) {
        this.animationState.time = time;
        
        // Animate arc reactor pulse
        if (this.parts.arcReactor) {
            const pulse = Math.sin(time * 3) * 0.1 + 0.9;
            this.parts.arcReactor.scale.set(pulse, pulse, pulse);
            
            // Rotate arc reactor rings
            const rings = this.parts.arcReactor.children;
            rings.forEach((child, index) => {
                if (child.geometry && child.geometry.type === 'TorusGeometry') {
                    child.rotation.z = time * (0.5 + index * 0.2) * (index % 2 ? 1 : -1);
                }
            });
            
            // Update arc reactor light intensity
            const reactorLight = this.lights.find(light => light.parent === this.parts.arcReactor);
            if (reactorLight) {
                reactorLight.intensity = 3 + Math.sin(time * 4) * 0.5;
            }
        }
        
        // Eye glow variation
        if (this.parts.eyes) {
            const eyeFlicker = Math.random() < 0.005 ? Math.random() * 0.3 : 0;
            this.parts.eyes.children.forEach((child) => {
                if (child.material && child.material.emissive) {
                    child.material.emissiveIntensity = 6.0 + eyeFlicker;
                }
            });
        }
        
        // Repulsor charging effect
        const repulsorGlow = Math.sin(time * 8) * 0.3 + 0.7;
        [this.parts.leftRepulsor, this.parts.rightRepulsor].forEach(repulsor => {
            if (repulsor) {
                const core = repulsor.children.find(child => 
                    child.material === this.materials.repulsor
                );
                if (core) {
                    core.material.emissiveIntensity = 8.0 * repulsorGlow;
                }
            }
        });
        
        // Subtle suit breathing animation
        const breathe = Math.sin(time * 0.5) * 0.002 + 1;
        this.suit.scale.y = breathe;
        
        // Update all lights
        this.lights.forEach((light, index) => {
            if (light.type === 'SpotLight' && light.parent && 
                (light.parent === this.parts.leftRepulsor || 
                 light.parent === this.parts.rightRepulsor)) {
                // Repulsor lights
                light.intensity = 1.0 * repulsorGlow;
            }
        });
    }
    
    // Activation sequence
    async activate() {
        // Could implement a cool activation sequence here
        console.log('Iron Man suit systems online!');
    }
    
    // Get suit reference
    getSuit() {
        return this.suit;
    }
}

// Export
window.UltimateRealisticIronManSuit = UltimateRealisticIronManSuit;