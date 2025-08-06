// PBR Materials for Iron Man Suit
// Physically Based Rendering materials for realistic metallic surfaces

class IronManMaterials {
    constructor() {
        this.textureLoader = new THREE.TextureLoader();
        this.cubeTextureLoader = new THREE.CubeTextureLoader();
        this.materials = {};
        this.envMap = null;
        
        this.init();
    }
    
    async init() {
        // Load environment map for reflections
        await this.loadEnvironmentMap();
        
        // Create all materials
        this.createMaterials();
    }
    
    async loadEnvironmentMap() {
        // Create a procedural environment map since we don't have textures
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(90, 1, 0.1, 1000);
        
        // Create gradient sky for reflections
        const skyGeo = new THREE.SphereGeometry(500, 32, 32);
        const skyMat = new THREE.ShaderMaterial({
            uniforms: {
                topColor: { value: new THREE.Color(0x0077ff) },
                bottomColor: { value: new THREE.Color(0x000033) },
                offset: { value: 33 },
                exponent: { value: 0.6 }
            },
            vertexShader: `
                varying vec3 vWorldPosition;
                void main() {
                    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPosition.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 topColor;
                uniform vec3 bottomColor;
                uniform float offset;
                uniform float exponent;
                varying vec3 vWorldPosition;
                void main() {
                    float h = normalize(vWorldPosition + offset).y;
                    gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
                }
            `,
            side: THREE.BackSide
        });
        
        const sky = new THREE.Mesh(skyGeo, skyMat);
        scene.add(sky);
        
        // Add some lights to the environment
        const light1 = new THREE.PointLight(0xffffff, 2, 100);
        light1.position.set(50, 50, 50);
        scene.add(light1);
        
        const light2 = new THREE.PointLight(0x00a8ff, 1, 100);
        light2.position.set(-50, 30, -50);
        scene.add(light2);
        
        // Render to cube render target
        const cubeRenderTarget = new THREE.WebGLCubeRenderTarget(256, {
            format: THREE.RGBFormat,
            generateMipmaps: true,
            minFilter: THREE.LinearMipmapLinearFilter
        });
        
        const cubeCamera = new THREE.CubeCamera(0.1, 1000, cubeRenderTarget);
        
        // Store environment map
        this.envMap = cubeRenderTarget.texture;
    }
    
    createMaterials() {
        // Main armor material - highly reflective metal with red and gold
        this.materials.armorRed = new THREE.MeshStandardMaterial({
            color: 0xcc0000,
            metalness: 0.95,
            roughness: 0.15,
            envMap: this.envMap,
            envMapIntensity: 1.5,
            normalScale: new THREE.Vector2(0.5, 0.5)
        });
        
        this.materials.armorGold = new THREE.MeshStandardMaterial({
            color: 0xffd700,
            metalness: 0.98,
            roughness: 0.1,
            envMap: this.envMap,
            envMapIntensity: 2.0,
            normalScale: new THREE.Vector2(0.3, 0.3)
        });
        
        // Arc Reactor material - emissive energy core
        this.materials.arcReactor = new THREE.MeshStandardMaterial({
            color: 0x00ffff,
            emissive: 0x00ffff,
            emissiveIntensity: 2.0,
            metalness: 0.5,
            roughness: 0.0,
            envMap: this.envMap,
            transparent: true,
            opacity: 0.9
        });
        
        // Eye lights material
        this.materials.eyes = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            emissive: 0x00aaff,
            emissiveIntensity: 3.0,
            metalness: 0.0,
            roughness: 0.0,
            transparent: true,
            opacity: 0.95
        });
        
        // Thruster material - hot glowing effect
        this.materials.thruster = new THREE.MeshStandardMaterial({
            color: 0xffaa00,
            emissive: 0xffaa00,
            emissiveIntensity: 4.0,
            metalness: 0.0,
            roughness: 1.0,
            transparent: true,
            opacity: 0.8
        });
        
        // Joint material - darker metal
        this.materials.joints = new THREE.MeshStandardMaterial({
            color: 0x333333,
            metalness: 0.8,
            roughness: 0.4,
            envMap: this.envMap,
            envMapIntensity: 0.5
        });
        
        // Glass/visor material
        this.materials.visor = new THREE.MeshPhysicalMaterial({
            color: 0x001122,
            metalness: 0.0,
            roughness: 0.0,
            transmission: 0.9,
            thickness: 0.5,
            envMap: this.envMap,
            envMapIntensity: 1.0,
            clearcoat: 1.0,
            clearcoatRoughness: 0.0,
            transparent: true,
            opacity: 0.8,
            ior: 1.5
        });
        
        // Energy shield material
        this.materials.shield = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                color: { value: new THREE.Color(0x00a8ff) },
                opacity: { value: 0.3 }
            },
            vertexShader: `
                varying vec2 vUv;
                varying vec3 vNormal;
                void main() {
                    vUv = uv;
                    vNormal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 color;
                uniform float opacity;
                varying vec2 vUv;
                varying vec3 vNormal;
                void main() {
                    float fresnel = pow(1.0 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
                    float wave = sin(vUv.y * 20.0 + time * 2.0) * 0.5 + 0.5;
                    gl_FragColor = vec4(color, opacity * fresnel * wave);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });
        
        // Holographic display material
        this.materials.hologram = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                color: { value: new THREE.Color(0x00ffff) }
            },
            vertexShader: `
                varying vec2 vUv;
                varying vec3 vPosition;
                void main() {
                    vUv = uv;
                    vPosition = position;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 color;
                varying vec2 vUv;
                varying vec3 vPosition;
                void main() {
                    float scanline = sin(vPosition.y * 50.0 + time * 5.0) * 0.1 + 0.9;
                    float flicker = sin(time * 30.0) * 0.05 + 0.95;
                    float alpha = scanline * flicker * 0.8;
                    gl_FragColor = vec4(color * scanline, alpha);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: false
        });
    }
    
    updateTime(time) {
        // Update time-based uniforms
        if (this.materials.shield) {
            this.materials.shield.uniforms.time.value = time;
        }
        if (this.materials.hologram) {
            this.materials.hologram.uniforms.time.value = time;
        }
    }
    
    getMaterial(name) {
        return this.materials[name] || this.materials.armorRed;
    }
}

// Export for use in viewport
window.IronManMaterials = IronManMaterials;