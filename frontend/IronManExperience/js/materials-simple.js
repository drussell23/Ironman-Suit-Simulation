// Simplified PBR Materials for Iron Man Suit
// Fallback materials that work without complex setup

class IronManMaterials {
    constructor() {
        this.materials = {};
        this.init();
    }
    
    init() {
        this.createMaterials();
    }
    
    createMaterials() {
        // Main armor material - highly reflective metal with red and gold
        this.materials.armorRed = new THREE.MeshStandardMaterial({
            color: 0xaa0000,
            metalness: 0.9,
            roughness: 0.2,
            emissive: 0x220000,
            emissiveIntensity: 0.1
        });
        
        this.materials.armorGold = new THREE.MeshStandardMaterial({
            color: 0xffaa00,
            metalness: 0.95,
            roughness: 0.1,
            emissive: 0x332200,
            emissiveIntensity: 0.2
        });
        
        // Arc Reactor material - emissive energy core
        this.materials.arcReactor = new THREE.MeshStandardMaterial({
            color: 0x00ffff,
            emissive: 0x00ffff,
            emissiveIntensity: 2.0,
            metalness: 0.5,
            roughness: 0.0
        });
        
        // Eye lights material
        this.materials.eyes = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            emissive: 0x00aaff,
            emissiveIntensity: 3.0,
            metalness: 0.0,
            roughness: 0.0
        });
        
        // Thruster material - hot glowing effect
        this.materials.thruster = new THREE.MeshBasicMaterial({
            color: 0xffaa00,
            transparent: true,
            opacity: 0.8
        });
        
        // Joint material - darker metal
        this.materials.joints = new THREE.MeshStandardMaterial({
            color: 0x333333,
            metalness: 0.8,
            roughness: 0.4
        });
        
        // Basic glow material for effects
        this.materials.glow = new THREE.MeshBasicMaterial({
            color: 0x00aaff,
            transparent: true,
            opacity: 0.5,
            side: THREE.DoubleSide
        });
    }
    
    updateTime(time) {
        // Update any time-based effects here
    }
    
    getMaterial(name) {
        return this.materials[name] || this.materials.armorRed;
    }
}

// Export for use in viewport
window.IronManMaterials = IronManMaterials;