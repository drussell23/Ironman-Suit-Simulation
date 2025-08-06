// Simplified Iron Man Suit Builder
// Creates a visible 3D Iron Man suit model

class IronManSuitBuilder {
    constructor(materials) {
        this.materials = materials || {
            getMaterial: (name) => {
                // Fallback materials if materials system fails
                const fallbackMaterials = {
                    armorRed: new THREE.MeshPhongMaterial({ color: 0xcc0000, shininess: 100 }),
                    armorGold: new THREE.MeshPhongMaterial({ color: 0xffcc00, shininess: 150 }),
                    arcReactor: new THREE.MeshBasicMaterial({ color: 0x00ffff }),
                    eyes: new THREE.MeshBasicMaterial({ color: 0x00aaff }),
                    joints: new THREE.MeshPhongMaterial({ color: 0x333333 })
                };
                return fallbackMaterials[name] || fallbackMaterials.armorRed;
            }
        };
        this.parts = {};
    }
    
    buildSuit() {
        const suit = new THREE.Group();
        suit.name = 'IronManSuit';
        
        // Build all parts with simpler geometry
        this.parts = {
            helmet: this.buildHelmet(),
            torso: this.buildTorso(),
            arms: this.buildArms(),
            legs: this.buildLegs(),
            arcReactor: this.buildArcReactor()
        };
        
        // Assemble the suit
        Object.values(this.parts).forEach(part => {
            if (part) suit.add(part);
        });
        
        return suit;
    }
    
    buildHelmet() {
        const helmet = new THREE.Group();
        helmet.name = 'Helmet';
        
        // Main helmet - simple sphere
        const helmetGeo = new THREE.SphereGeometry(12, 16, 12);
        const helmetMesh = new THREE.Mesh(helmetGeo, this.materials.getMaterial('armorRed'));
        helmetMesh.scale.y = 1.1;
        helmet.add(helmetMesh);
        
        // Face plate - simple box
        const facePlateGeo = new THREE.BoxGeometry(10, 10, 2);
        const facePlate = new THREE.Mesh(facePlateGeo, this.materials.getMaterial('armorGold'));
        facePlate.position.set(0, -2, 10);
        helmet.add(facePlate);
        
        // Eyes - simple boxes
        const eyeGeo = new THREE.BoxGeometry(3, 1.5, 0.5);
        const leftEye = new THREE.Mesh(eyeGeo, this.materials.getMaterial('eyes'));
        leftEye.position.set(-3, 2, 11);
        helmet.add(leftEye);
        
        const rightEye = new THREE.Mesh(eyeGeo, this.materials.getMaterial('eyes'));
        rightEye.position.set(3, 2, 11);
        helmet.add(rightEye);
        
        helmet.position.y = 50;
        
        return helmet;
    }
    
    buildTorso() {
        const torso = new THREE.Group();
        torso.name = 'Torso';
        
        // Main body - simple box
        const chestGeo = new THREE.BoxGeometry(30, 35, 18);
        const chestMesh = new THREE.Mesh(chestGeo, this.materials.getMaterial('armorRed'));
        torso.add(chestMesh);
        
        // Center chest plate for arc reactor
        const plateGeo = new THREE.BoxGeometry(12, 15, 2);
        const centerPlate = new THREE.Mesh(plateGeo, this.materials.getMaterial('armorGold'));
        centerPlate.position.set(0, 5, 10);
        torso.add(centerPlate);
        
        torso.position.y = 20;
        
        return torso;
    }
    
    buildArms() {
        const arms = new THREE.Group();
        arms.name = 'Arms';
        
        // Simple cylinder arms
        const armGeo = new THREE.CylinderGeometry(5, 4, 30, 8);
        
        // Left arm
        const leftArm = new THREE.Mesh(armGeo, this.materials.getMaterial('armorRed'));
        leftArm.position.set(-20, 15, 0);
        arms.add(leftArm);
        
        // Right arm  
        const rightArm = new THREE.Mesh(armGeo, this.materials.getMaterial('armorRed'));
        rightArm.position.set(20, 15, 0);
        arms.add(rightArm);
        
        // Simple hands
        const handGeo = new THREE.BoxGeometry(7, 7, 4);
        
        const leftHand = new THREE.Mesh(handGeo, this.materials.getMaterial('armorRed'));
        leftHand.position.set(-20, -2, 0);
        arms.add(leftHand);
        
        const rightHand = new THREE.Mesh(handGeo, this.materials.getMaterial('armorRed'));
        rightHand.position.set(20, -2, 0);
        arms.add(rightHand);
        
        return arms;
    }
    
    buildLegs() {
        const legs = new THREE.Group();
        legs.name = 'Legs';
        
        // Simple cylinder legs
        const legGeo = new THREE.CylinderGeometry(6, 5, 35, 8);
        
        // Left leg
        const leftLeg = new THREE.Mesh(legGeo, this.materials.getMaterial('armorRed'));
        leftLeg.position.set(-10, -20, 0);
        legs.add(leftLeg);
        
        // Right leg
        const rightLeg = new THREE.Mesh(legGeo, this.materials.getMaterial('armorRed'));
        rightLeg.position.set(10, -20, 0);
        legs.add(rightLeg);
        
        // Simple feet
        const footGeo = new THREE.BoxGeometry(10, 5, 15);
        
        const leftFoot = new THREE.Mesh(footGeo, this.materials.getMaterial('armorRed'));
        leftFoot.position.set(-10, -40, 2);
        legs.add(leftFoot);
        
        const rightFoot = new THREE.Mesh(footGeo, this.materials.getMaterial('armorRed'));
        rightFoot.position.set(10, -40, 2);
        legs.add(rightFoot);
        
        return legs;
    }
    
    buildArcReactor() {
        const arcReactor = new THREE.Group();
        arcReactor.name = 'ArcReactor';
        
        // Simple glowing circle
        const reactorGeo = new THREE.CylinderGeometry(4, 4, 1, 16);
        const reactor = new THREE.Mesh(reactorGeo, this.materials.getMaterial('arcReactor'));
        reactor.rotation.x = Math.PI / 2;
        reactor.position.set(0, 25, 11);
        arcReactor.add(reactor);
        
        // Add a point light
        const light = new THREE.PointLight(0x00ffff, 1, 30);
        light.position.set(0, 25, 12);
        arcReactor.add(light);
        
        return arcReactor;
    }
}

// Export for use
window.IronManSuitBuilder = IronManSuitBuilder;