// Lighting improvements for better visibility

function applyLightingFix(viewport) {
    if (!viewport || !viewport.scene) return;
    
    // Increase ambient light significantly
    viewport.scene.traverse((child) => {
        if (child.type === 'AmbientLight') {
            child.intensity = 1.2; // Increase from 0.4
            child.color = new THREE.Color(0x808090); // Slightly blue-tinted
        }
    });
    
    // Add additional fill light
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
    fillLight.position.set(-100, 200, -100);
    viewport.scene.add(fillLight);
    
    // Add a rim light for better suit visibility
    const rimLight = new THREE.DirectionalLight(0x4080ff, 0.4);
    rimLight.position.set(0, 100, -200);
    viewport.scene.add(rimLight);
    
    // Brighten the fog
    viewport.scene.fog = new THREE.FogExp2(0x001144, 0.0001); // Lighter fog
    
    // Set brighter clear color
    viewport.renderer.setClearColor(0x001144); // Dark blue instead of black
    
    // Increase tone mapping exposure
    viewport.renderer.toneMappingExposure = 1.5;
    
    // Make suit materials brighter
    if (viewport.suit) {
        viewport.suit.traverse((child) => {
            if (child.isMesh && child.material) {
                // Only modify emissive for materials that support it
                if (child.material.isStandardMaterial || child.material.isPhongMaterial) {
                    if (child.material.emissive && child.material.emissiveIntensity !== undefined) {
                        child.material.emissiveIntensity = Math.min(child.material.emissiveIntensity * 2, 3);
                    }
                    // Make metallic surfaces less dark
                    if (child.material.metalness > 0.5) {
                        child.material.metalness = 0.6;
                        child.material.roughness = 0.3;
                    }
                }
                // For basic materials, just brighten the color
                else if (child.material.isMeshBasicMaterial) {
                    const currentColor = child.material.color;
                    // Brighten the color by 50%
                    child.material.color = new THREE.Color(
                        Math.min(currentColor.r * 1.5, 1),
                        Math.min(currentColor.g * 1.5, 1),
                        Math.min(currentColor.b * 1.5, 1)
                    );
                }
            }
        });
    }
    
    // Brighten arc reactor and repulsors
    if (viewport.effects) {
        if (viewport.effects.arcReactor && viewport.effects.arcReactor.material) {
            const mat = viewport.effects.arcReactor.material;
            if (mat.emissiveIntensity !== undefined) {
                mat.emissiveIntensity = 5;
            } else if (mat.color) {
                // Brighten basic material
                mat.color = new THREE.Color(0x00ffff);
            }
        }
        ['leftRepulsor', 'rightRepulsor'].forEach(name => {
            if (viewport.effects[name] && viewport.effects[name].material) {
                const mat = viewport.effects[name].material;
                if (mat.emissiveIntensity !== undefined) {
                    mat.emissiveIntensity = 3;
                } else if (mat.color) {
                    // Brighten basic material
                    mat.color = new THREE.Color(0x00ffff);
                }
            }
        });
    }
    
    // Add ground reflection
    const groundMirror = new THREE.Mesh(
        new THREE.PlaneGeometry(2000, 2000),
        new THREE.MeshStandardMaterial({
            color: 0x111122,
            metalness: 0.5,
            roughness: 0.8,
            envMapIntensity: 1
        })
    );
    groundMirror.rotation.x = -Math.PI / 2;
    groundMirror.position.y = -1;
    viewport.scene.add(groundMirror);
}

// Day mode preset
function setDayMode(viewport) {
    if (!viewport || !viewport.scene) return;
    
    // Bright sky
    viewport.scene.fog = new THREE.Fog(0x87CEEB, 100, 5000);
    viewport.renderer.setClearColor(0x87CEEB);
    
    // Sun light
    viewport.scene.traverse((child) => {
        if (child.type === 'DirectionalLight' && child.intensity > 0.7) {
            child.intensity = 1.2;
            child.color = new THREE.Color(0xfff4e6);
        }
    });
    
    // Bright ambient
    viewport.scene.traverse((child) => {
        if (child.type === 'AmbientLight') {
            child.intensity = 0.8;
            child.color = new THREE.Color(0xffffff);
        }
    });
}

// Night mode preset (brighter than default)
function setBrightNightMode(viewport) {
    if (!viewport || !viewport.scene) return;
    
    // Less dark fog
    viewport.scene.fog = new THREE.FogExp2(0x000033, 0.00015);
    viewport.renderer.setClearColor(0x000033);
    
    // Moonlight
    viewport.scene.traverse((child) => {
        if (child.type === 'DirectionalLight' && child.intensity > 0.7) {
            child.intensity = 0.6;
            child.color = new THREE.Color(0xaabbff);
        }
    });
    
    // Higher ambient for night
    viewport.scene.traverse((child) => {
        if (child.type === 'AmbientLight') {
            child.intensity = 0.5;
            child.color = new THREE.Color(0x404080);
        }
    });
}

// Auto-apply lighting fix when viewport is ready
if (window.viewport) {
    applyLightingFix(window.viewport);
}

// Export functions
window.LightingFix = {
    apply: applyLightingFix,
    setDayMode: setDayMode,
    setNightMode: setBrightNightMode
};