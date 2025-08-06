// Debug viewport to test 3D rendering

class DebugViewport {
    constructor() {
        this.canvas = document.getElementById('viewport');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.suit = null;
        
        this.init();
    }
    
    init() {
        console.log('DebugViewport: Initializing...');
        
        // Basic Three.js setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000033);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 0, 100);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(50, 50, 50);
        this.scene.add(directionalLight);
        
        // Create simple Iron Man suit
        this.createSimpleSuit();
        
        // Start animation
        this.animate();
        
        console.log('DebugViewport: Initialization complete');
    }
    
    createSimpleSuit() {
        console.log('DebugViewport: Creating suit...');
        
        this.suit = new THREE.Group();
        
        // Red material for armor
        const redMaterial = new THREE.MeshPhongMaterial({
            color: 0xcc0000,
            specular: 0xffffff,
            shininess: 100
        });
        
        // Gold material for accents
        const goldMaterial = new THREE.MeshPhongMaterial({
            color: 0xffcc00,
            specular: 0xffffff,
            shininess: 150
        });
        
        // Blue glow material
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ccff
        });
        
        // Head
        const headGeometry = new THREE.SphereGeometry(10, 16, 12);
        const head = new THREE.Mesh(headGeometry, redMaterial);
        head.position.y = 35;
        this.suit.add(head);
        
        // Face plate
        const faceGeometry = new THREE.BoxGeometry(8, 8, 2);
        const face = new THREE.Mesh(faceGeometry, goldMaterial);
        face.position.set(0, 33, 9);
        this.suit.add(face);
        
        // Eyes
        const eyeGeometry = new THREE.BoxGeometry(2, 1, 0.5);
        const leftEye = new THREE.Mesh(eyeGeometry, glowMaterial);
        leftEye.position.set(-2.5, 35, 10);
        this.suit.add(leftEye);
        
        const rightEye = new THREE.Mesh(eyeGeometry, glowMaterial);
        rightEye.position.set(2.5, 35, 10);
        this.suit.add(rightEye);
        
        // Body
        const bodyGeometry = new THREE.BoxGeometry(20, 25, 12);
        const body = new THREE.Mesh(bodyGeometry, redMaterial);
        body.position.y = 10;
        this.suit.add(body);
        
        // Arc Reactor
        const arcGeometry = new THREE.CylinderGeometry(3, 3, 1, 16);
        const arcReactor = new THREE.Mesh(arcGeometry, glowMaterial);
        arcReactor.rotation.x = Math.PI / 2;
        arcReactor.position.set(0, 15, 7);
        this.suit.add(arcReactor);
        
        // Arms
        const armGeometry = new THREE.CylinderGeometry(3, 2.5, 20, 8);
        
        const leftArm = new THREE.Mesh(armGeometry, redMaterial);
        leftArm.position.set(-15, 10, 0);
        this.suit.add(leftArm);
        
        const rightArm = new THREE.Mesh(armGeometry, redMaterial);
        rightArm.position.set(15, 10, 0);
        this.suit.add(rightArm);
        
        // Legs
        const legGeometry = new THREE.CylinderGeometry(4, 3, 25, 8);
        
        const leftLeg = new THREE.Mesh(legGeometry, redMaterial);
        leftLeg.position.set(-7, -15, 0);
        this.suit.add(leftLeg);
        
        const rightLeg = new THREE.Mesh(legGeometry, redMaterial);
        rightLeg.position.set(7, -15, 0);
        this.suit.add(rightLeg);
        
        this.scene.add(this.suit);
        
        console.log('DebugViewport: Suit created and added to scene');
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Rotate suit
        if (this.suit) {
            this.suit.rotation.y += 0.01;
        }
        
        this.renderer.render(this.scene, this.camera);
    }
}

// Replace EnhancedViewport with DebugViewport for testing
window.EnhancedViewport = DebugViewport;