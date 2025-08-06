// Iron Man Suit Builder - Detailed 3D Model Construction

class IronManSuitBuilder {
    constructor(materials) {
        this.materials = materials;
        this.parts = {};
    }
    
    buildSuit() {
        const suit = new THREE.Group();
        suit.name = 'IronManSuit';
        
        // Build all parts
        this.parts = {
            helmet: this.buildHelmet(),
            torso: this.buildTorso(),
            shoulders: this.buildShoulders(),
            arms: this.buildArms(),
            hands: this.buildHands(),
            pelvis: this.buildPelvis(),
            legs: this.buildLegs(),
            feet: this.buildFeet(),
            arcReactor: this.buildArcReactor(),
            details: this.buildDetails()
        };
        
        // Assemble the suit
        Object.values(this.parts).forEach(part => {
            if (part) suit.add(part);
        });
        
        // Scale to proper size (roughly 2 meters tall)
        suit.scale.set(0.01, 0.01, 0.01);
        
        return suit;
    }
    
    buildHelmet() {
        const helmet = new THREE.Group();
        helmet.name = 'Helmet';
        
        // Main helmet shape
        const helmetGeo = new THREE.SphereGeometry(12, 32, 24);
        helmetGeo.scale(1, 1.1, 0.95);
        
        // Modify vertices for Iron Man helmet shape
        const positions = helmetGeo.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const y = positions.getY(i);
            const z = positions.getZ(i);
            
            // Flatten the back
            if (z < -8) {
                positions.setZ(i, z * 0.7);
            }
            
            // Create face plate indent
            if (y < 0 && z > 5) {
                positions.setZ(i, z * 0.85);
            }
        }
        
        helmetGeo.computeVertexNormals();
        
        const helmetMesh = new THREE.Mesh(helmetGeo, this.materials.getMaterial('armorRed'));
        helmetMesh.castShadow = true;
        helmetMesh.receiveShadow = true;
        helmet.add(helmetMesh);
        
        // Face plate
        const facePlateGeo = new THREE.BoxGeometry(10, 12, 2);
        facePlateGeo.translate(0, -2, 11);
        const facePlate = new THREE.Mesh(facePlateGeo, this.materials.getMaterial('armorGold'));
        facePlate.castShadow = true;
        helmet.add(facePlate);
        
        // Eyes
        const eyeGeo = new THREE.BoxGeometry(3, 1.5, 0.5);
        const leftEye = new THREE.Mesh(eyeGeo, this.materials.getMaterial('eyes'));
        leftEye.position.set(-3, 2, 11.5);
        helmet.add(leftEye);
        
        const rightEye = new THREE.Mesh(eyeGeo, this.materials.getMaterial('eyes'));
        rightEye.position.set(3, 2, 11.5);
        helmet.add(rightEye);
        
        // Add eye glow effect
        const eyeGlowGeo = new THREE.PlaneGeometry(4, 2);
        const eyeGlowMat = new THREE.MeshBasicMaterial({
            color: 0x00aaff,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide
        });
        
        const leftGlow = new THREE.Mesh(eyeGlowGeo, eyeGlowMat);
        leftGlow.position.set(-3, 2, 12);
        helmet.add(leftGlow);
        
        const rightGlow = new THREE.Mesh(eyeGlowGeo, eyeGlowMat);
        rightGlow.position.set(3, 2, 12);
        helmet.add(rightGlow);
        
        helmet.position.y = 185;
        
        return helmet;
    }
    
    buildTorso() {
        const torso = new THREE.Group();
        torso.name = 'Torso';
        
        // Upper chest
        const chestGeo = new THREE.BoxGeometry(35, 30, 20);
        chestGeo.scale(1, 1.2, 1);
        
        // Modify for muscular shape
        const positions = chestGeo.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const y = positions.getY(i);
            const z = positions.getZ(i);
            
            // Pectoral muscles
            if (y > 5 && Math.abs(x) > 10 && z > 0) {
                positions.setZ(i, z + 3);
            }
            
            // Taper at waist
            if (y < -10) {
                positions.setX(i, x * 0.85);
            }
        }
        chestGeo.computeVertexNormals();
        
        const chestMesh = new THREE.Mesh(chestGeo, this.materials.getMaterial('armorRed'));
        chestMesh.castShadow = true;
        chestMesh.receiveShadow = true;
        torso.add(chestMesh);
        
        // Chest armor plates
        const plateSizeX = 12;
        const plateSizeY = 18;
        const plateGeo = new THREE.BoxGeometry(plateSizeX, plateSizeY, 2);
        
        // Center chest plate (houses arc reactor)
        const centerPlate = new THREE.Mesh(plateGeo, this.materials.getMaterial('armorGold'));
        centerPlate.position.set(0, 5, 11);
        torso.add(centerPlate);
        
        // Side plates
        const leftPlate = new THREE.Mesh(plateGeo, this.materials.getMaterial('armorGold'));
        leftPlate.position.set(-15, 0, 8);
        leftPlate.rotation.y = -0.3;
        torso.add(leftPlate);
        
        const rightPlate = new THREE.Mesh(plateGeo, this.materials.getMaterial('armorGold'));
        rightPlate.position.set(15, 0, 8);
        rightPlate.rotation.y = 0.3;
        torso.add(rightPlate);
        
        // Abdominal section
        const abdomenGeo = new THREE.CylinderGeometry(14, 12, 20, 8, 1);
        const abdomen = new THREE.Mesh(abdomenGeo, this.materials.getMaterial('armorRed'));
        abdomen.position.y = -25;
        abdomen.castShadow = true;
        torso.add(abdomen);
        
        // Ab plates
        for (let i = 0; i < 4; i++) {
            const abPlateGeo = new THREE.BoxGeometry(10, 4, 1);
            const abPlate = new THREE.Mesh(abPlateGeo, this.materials.getMaterial('joints'));
            abPlate.position.set(0, -20 - i * 5, 12);
            torso.add(abPlate);
        }
        
        torso.position.y = 150;
        
        return torso;
    }
    
    buildShoulders() {
        const shoulders = new THREE.Group();
        shoulders.name = 'Shoulders';
        
        const shoulderGeo = new THREE.SphereGeometry(10, 16, 12);
        shoulderGeo.scale(1.2, 1, 1);
        
        // Left shoulder
        const leftShoulder = new THREE.Mesh(shoulderGeo, this.materials.getMaterial('armorRed'));
        leftShoulder.position.set(-22, 165, 0);
        leftShoulder.castShadow = true;
        shoulders.add(leftShoulder);
        
        // Right shoulder
        const rightShoulder = new THREE.Mesh(shoulderGeo, this.materials.getMaterial('armorRed'));
        rightShoulder.position.set(22, 165, 0);
        rightShoulder.castShadow = true;
        shoulders.add(rightShoulder);
        
        // Shoulder armor plates
        const plateGeo = new THREE.BoxGeometry(15, 8, 12);
        plateGeo.rotateZ(0.3);
        
        const leftPlate = new THREE.Mesh(plateGeo, this.materials.getMaterial('armorGold'));
        leftPlate.position.set(-25, 168, 0);
        shoulders.add(leftPlate);
        
        const rightPlate = new THREE.Mesh(plateGeo, this.materials.getMaterial('armorGold'));
        rightPlate.position.set(25, 168, 0);
        rightPlate.scale.x = -1;
        shoulders.add(rightPlate);
        
        return shoulders;
    }
    
    buildArms() {
        const arms = new THREE.Group();
        arms.name = 'Arms';
        
        // Arm segments for better articulation
        const upperArmGeo = new THREE.CylinderGeometry(7, 6, 30, 8);
        const lowerArmGeo = new THREE.CylinderGeometry(6, 5, 30, 8);
        
        // Left arm
        const leftUpperArm = new THREE.Mesh(upperArmGeo, this.materials.getMaterial('armorRed'));
        leftUpperArm.position.set(-22, 145, 0);
        leftUpperArm.castShadow = true;
        arms.add(leftUpperArm);
        
        const leftElbow = new THREE.Mesh(
            new THREE.SphereGeometry(6, 8, 6),
            this.materials.getMaterial('joints')
        );
        leftElbow.position.set(-22, 130, 0);
        arms.add(leftElbow);
        
        const leftLowerArm = new THREE.Mesh(lowerArmGeo, this.materials.getMaterial('armorRed'));
        leftLowerArm.position.set(-22, 115, 0);
        leftLowerArm.castShadow = true;
        arms.add(leftLowerArm);
        
        // Right arm
        const rightUpperArm = new THREE.Mesh(upperArmGeo, this.materials.getMaterial('armorRed'));
        rightUpperArm.position.set(22, 145, 0);
        rightUpperArm.castShadow = true;
        arms.add(rightUpperArm);
        
        const rightElbow = new THREE.Mesh(
            new THREE.SphereGeometry(6, 8, 6),
            this.materials.getMaterial('joints')
        );
        rightElbow.position.set(22, 130, 0);
        arms.add(rightElbow);
        
        const rightLowerArm = new THREE.Mesh(lowerArmGeo, this.materials.getMaterial('armorRed'));
        rightLowerArm.position.set(22, 115, 0);
        rightLowerArm.castShadow = true;
        arms.add(rightLowerArm);
        
        // Armor details
        const armPlateGeo = new THREE.BoxGeometry(8, 15, 8);
        
        // Forearm plates
        const leftForearmPlate = new THREE.Mesh(armPlateGeo, this.materials.getMaterial('armorGold'));
        leftForearmPlate.position.set(-22, 115, 4);
        arms.add(leftForearmPlate);
        
        const rightForearmPlate = new THREE.Mesh(armPlateGeo, this.materials.getMaterial('armorGold'));
        rightForearmPlate.position.set(22, 115, 4);
        arms.add(rightForearmPlate);
        
        return arms;
    }
    
    buildHands() {
        const hands = new THREE.Group();
        hands.name = 'Hands';
        
        // Palm
        const palmGeo = new THREE.BoxGeometry(8, 10, 4);
        
        // Left hand
        const leftPalm = new THREE.Mesh(palmGeo, this.materials.getMaterial('armorRed'));
        leftPalm.position.set(-22, 95, 0);
        hands.add(leftPalm);
        
        // Repulsor in palm
        const repulsorGeo = new THREE.CylinderGeometry(3, 3, 1, 16);
        const leftRepulsor = new THREE.Mesh(repulsorGeo, this.materials.getMaterial('arcReactor'));
        leftRepulsor.position.set(-22, 95, 2.5);
        leftRepulsor.rotation.x = Math.PI / 2;
        hands.add(leftRepulsor);
        
        // Right hand
        const rightPalm = new THREE.Mesh(palmGeo, this.materials.getMaterial('armorRed'));
        rightPalm.position.set(22, 95, 0);
        hands.add(rightPalm);
        
        const rightRepulsor = new THREE.Mesh(repulsorGeo, this.materials.getMaterial('arcReactor'));
        rightRepulsor.position.set(22, 95, 2.5);
        rightRepulsor.rotation.x = Math.PI / 2;
        hands.add(rightRepulsor);
        
        // Simplified fingers
        const fingerGeo = new THREE.BoxGeometry(1.5, 6, 1.5);
        
        for (let i = 0; i < 4; i++) {
            // Left hand fingers
            const leftFinger = new THREE.Mesh(fingerGeo, this.materials.getMaterial('armorRed'));
            leftFinger.position.set(-24 + i * 2, 88, 0);
            hands.add(leftFinger);
            
            // Right hand fingers
            const rightFinger = new THREE.Mesh(fingerGeo, this.materials.getMaterial('armorRed'));
            rightFinger.position.set(20 + i * 2, 88, 0);
            hands.add(rightFinger);
        }
        
        return hands;
    }
    
    buildPelvis() {
        const pelvis = new THREE.Group();
        pelvis.name = 'Pelvis';
        
        const pelvisGeo = new THREE.BoxGeometry(30, 15, 18);
        const pelvisMesh = new THREE.Mesh(pelvisGeo, this.materials.getMaterial('armorRed'));
        pelvisMesh.position.y = 115;
        pelvisMesh.castShadow = true;
        pelvis.add(pelvisMesh);
        
        // Belt detail
        const beltGeo = new THREE.BoxGeometry(32, 4, 20);
        const belt = new THREE.Mesh(beltGeo, this.materials.getMaterial('armorGold'));
        belt.position.y = 115;
        pelvis.add(belt);
        
        return pelvis;
    }
    
    buildLegs() {
        const legs = new THREE.Group();
        legs.name = 'Legs';
        
        const thighGeo = new THREE.CylinderGeometry(8, 7, 35, 8);
        const shinGeo = new THREE.CylinderGeometry(7, 6, 35, 8);
        
        // Left leg
        const leftThigh = new THREE.Mesh(thighGeo, this.materials.getMaterial('armorRed'));
        leftThigh.position.set(-10, 90, 0);
        leftThigh.castShadow = true;
        legs.add(leftThigh);
        
        const leftKnee = new THREE.Mesh(
            new THREE.SphereGeometry(7, 8, 6),
            this.materials.getMaterial('joints')
        );
        leftKnee.position.set(-10, 72, 0);
        legs.add(leftKnee);
        
        const leftShin = new THREE.Mesh(shinGeo, this.materials.getMaterial('armorRed'));
        leftShin.position.set(-10, 55, 0);
        leftShin.castShadow = true;
        legs.add(leftShin);
        
        // Right leg
        const rightThigh = new THREE.Mesh(thighGeo, this.materials.getMaterial('armorRed'));
        rightThigh.position.set(10, 90, 0);
        rightThigh.castShadow = true;
        legs.add(rightThigh);
        
        const rightKnee = new THREE.Mesh(
            new THREE.SphereGeometry(7, 8, 6),
            this.materials.getMaterial('joints')
        );
        rightKnee.position.set(10, 72, 0);
        legs.add(rightKnee);
        
        const rightShin = new THREE.Mesh(shinGeo, this.materials.getMaterial('armorRed'));
        rightShin.position.set(10, 55, 0);
        rightShin.castShadow = true;
        legs.add(rightShin);
        
        // Thigh armor plates
        const thighPlateGeo = new THREE.BoxGeometry(10, 20, 10);
        
        const leftThighPlate = new THREE.Mesh(thighPlateGeo, this.materials.getMaterial('armorGold'));
        leftThighPlate.position.set(-10, 90, 5);
        legs.add(leftThighPlate);
        
        const rightThighPlate = new THREE.Mesh(thighPlateGeo, this.materials.getMaterial('armorGold'));
        rightThighPlate.position.set(10, 90, 5);
        legs.add(rightThighPlate);
        
        return legs;
    }
    
    buildFeet() {
        const feet = new THREE.Group();
        feet.name = 'Feet';
        
        const footGeo = new THREE.BoxGeometry(10, 8, 20);
        
        // Left foot
        const leftFoot = new THREE.Mesh(footGeo, this.materials.getMaterial('armorRed'));
        leftFoot.position.set(-10, 34, 5);
        leftFoot.castShadow = true;
        feet.add(leftFoot);
        
        // Left boot thruster
        const thrusterGeo = new THREE.CylinderGeometry(4, 5, 3, 8);
        const leftThruster = new THREE.Mesh(thrusterGeo, this.materials.getMaterial('thruster'));
        leftThruster.position.set(-10, 30, -5);
        leftThruster.rotation.x = Math.PI / 2;
        feet.add(leftThruster);
        
        // Right foot
        const rightFoot = new THREE.Mesh(footGeo, this.materials.getMaterial('armorRed'));
        rightFoot.position.set(10, 34, 5);
        rightFoot.castShadow = true;
        feet.add(rightFoot);
        
        // Right boot thruster
        const rightThruster = new THREE.Mesh(thrusterGeo, this.materials.getMaterial('thruster'));
        rightThruster.position.set(10, 30, -5);
        rightThruster.rotation.x = Math.PI / 2;
        feet.add(rightThruster);
        
        return feet;
    }
    
    buildArcReactor() {
        const arcReactor = new THREE.Group();
        arcReactor.name = 'ArcReactor';
        
        // Outer ring
        const outerRingGeo = new THREE.TorusGeometry(6, 1, 8, 16);
        const outerRing = new THREE.Mesh(outerRingGeo, this.materials.getMaterial('armorGold'));
        arcReactor.add(outerRing);
        
        // Inner core
        const coreGeo = new THREE.SphereGeometry(4, 16, 16);
        const core = new THREE.Mesh(coreGeo, this.materials.getMaterial('arcReactor'));
        arcReactor.add(core);
        
        // Energy glow
        const glowGeo = new THREE.PlaneGeometry(12, 12);
        const glowMat = new THREE.MeshBasicMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide
        });
        const glow = new THREE.Mesh(glowGeo, glowMat);
        glow.position.z = 1;
        arcReactor.add(glow);
        
        // Point light for arc reactor
        const reactorLight = new THREE.PointLight(0x00ffff, 2, 50);
        reactorLight.position.z = 5;
        arcReactor.add(reactorLight);
        
        arcReactor.position.set(0, 155, 13);
        
        return arcReactor;
    }
    
    buildDetails() {
        const details = new THREE.Group();
        details.name = 'Details';
        
        // Add various small details, panels, etc.
        // This could be expanded with more intricate details
        
        return details;
    }
}

// Export for use
window.IronManSuitBuilder = IronManSuitBuilder;