// Post-Processing Effects for Iron Man Experience
// Advanced visual effects using Three.js post-processing

class PostProcessingEffects {
    constructor(renderer, scene, camera) {
        this.renderer = renderer;
        this.scene = scene;
        this.camera = camera;
        this.composer = null;
        this.passes = {};
        
        this.init();
    }
    
    init() {
        // Check if EffectComposer is available
        if (typeof THREE.EffectComposer === 'undefined') {
            console.warn('Post-processing not available. Including fallback shaders.');
            this.initFallbackComposer();
            return;
        }
        
        this.setupComposer();
        this.addPasses();
    }
    
    initFallbackComposer() {
        // Create a simple fallback that applies basic effects
        this.composer = {
            render: () => {
                this.renderer.render(this.scene, this.camera);
            },
            setSize: (width, height) => {
                // No-op for fallback
            }
        };
    }
    
    setupComposer() {
        // Create render targets with proper settings
        const renderTarget = new THREE.WebGLRenderTarget(
            window.innerWidth,
            window.innerHeight,
            {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                stencilBuffer: false
            }
        );
        
        // Initialize composer
        this.composer = new THREE.EffectComposer(this.renderer, renderTarget);
        
        // Add render pass
        const renderPass = new THREE.RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);
        this.passes.render = renderPass;
    }
    
    addPasses() {
        // Bloom pass for glowing effects
        this.addBloomPass();
        
        // Film grain and scan lines for cinematic look
        this.addFilmPass();
        
        // FXAA for anti-aliasing
        this.addFXAAPass();
        
        // Vignette effect
        this.addVignettePass();
        
        // Optional: Motion blur for fast movements
        // this.addMotionBlurPass();
    }
    
    addBloomPass() {
        if (typeof THREE.UnrealBloomPass !== 'undefined') {
            const bloomPass = new THREE.UnrealBloomPass(
                new THREE.Vector2(window.innerWidth, window.innerHeight),
                1.5, // strength
                0.4, // radius
                0.85 // threshold
            );
            this.composer.addPass(bloomPass);
            this.passes.bloom = bloomPass;
        } else {
            // Fallback custom bloom shader
            this.addCustomBloomPass();
        }
    }
    
    addCustomBloomPass() {
        const bloomShader = {
            uniforms: {
                tDiffuse: { value: null },
                bloomStrength: { value: 1.5 },
                bloomRadius: { value: 0.4 },
                bloomThreshold: { value: 0.85 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform float bloomStrength;
                uniform float bloomRadius;
                uniform float bloomThreshold;
                varying vec2 vUv;
                
                vec3 extractBright(vec3 color) {
                    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
                    return brightness > bloomThreshold ? color : vec3(0.0);
                }
                
                void main() {
                    vec4 texel = texture2D(tDiffuse, vUv);
                    vec3 brightColor = extractBright(texel.rgb);
                    
                    // Simple blur
                    vec3 blur = vec3(0.0);
                    float total = 0.0;
                    for (float x = -4.0; x <= 4.0; x += 1.0) {
                        for (float y = -4.0; y <= 4.0; y += 1.0) {
                            float weight = exp(-(x*x + y*y) / (2.0 * bloomRadius * bloomRadius));
                            vec2 offset = vec2(x, y) * 0.002;
                            blur += extractBright(texture2D(tDiffuse, vUv + offset).rgb) * weight;
                            total += weight;
                        }
                    }
                    blur /= total;
                    
                    gl_FragColor = vec4(texel.rgb + blur * bloomStrength, texel.a);
                }
            `
        };
        
        const bloomPass = new THREE.ShaderPass(bloomShader);
        this.composer.addPass(bloomPass);
        this.passes.bloom = bloomPass;
    }
    
    addFilmPass() {
        const filmShader = {
            uniforms: {
                tDiffuse: { value: null },
                time: { value: 0 },
                nIntensity: { value: 0.5 },
                sIntensity: { value: 0.05 },
                sCount: { value: 2048 },
                grayscale: { value: 0 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform float time;
                uniform float nIntensity;
                uniform float sIntensity;
                uniform float sCount;
                uniform float grayscale;
                varying vec2 vUv;
                
                void main() {
                    vec4 cTextureScreen = texture2D(tDiffuse, vUv);
                    
                    // Film grain
                    float dx = rand(vUv + time);
                    vec3 cResult = cTextureScreen.rgb + cTextureScreen.rgb * clamp(0.1 + dx, 0.0, 1.0);
                    
                    // Scanlines
                    vec2 sc = vec2(sin(vUv.y * sCount), cos(vUv.y * sCount));
                    cResult += cTextureScreen.rgb * vec3(sc.x, sc.y, sc.x) * sIntensity;
                    
                    // Convert to grayscale if needed
                    if (grayscale > 0.0) {
                        float gray = dot(cResult, vec3(0.299, 0.587, 0.114));
                        cResult = mix(cResult, vec3(gray), grayscale);
                    }
                    
                    gl_FragColor = vec4(cResult, cTextureScreen.a);
                }
                
                float rand(vec2 co) {
                    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
                }
            `
        };
        
        const filmPass = new THREE.ShaderPass(filmShader);
        filmPass.uniforms.nIntensity.value = 0.15;
        filmPass.uniforms.sIntensity.value = 0.03;
        this.composer.addPass(filmPass);
        this.passes.film = filmPass;
    }
    
    addFXAAPass() {
        if (typeof THREE.ShaderPass !== 'undefined' && typeof THREE.FXAAShader !== 'undefined') {
            const fxaaPass = new THREE.ShaderPass(THREE.FXAAShader);
            const pixelRatio = this.renderer.getPixelRatio();
            fxaaPass.material.uniforms['resolution'].value.x = 1 / (window.innerWidth * pixelRatio);
            fxaaPass.material.uniforms['resolution'].value.y = 1 / (window.innerHeight * pixelRatio);
            this.composer.addPass(fxaaPass);
            this.passes.fxaa = fxaaPass;
        }
    }
    
    addVignettePass() {
        const vignetteShader = {
            uniforms: {
                tDiffuse: { value: null },
                offset: { value: 1.0 },
                darkness: { value: 1.3 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform float offset;
                uniform float darkness;
                varying vec2 vUv;
                
                void main() {
                    vec4 texel = texture2D(tDiffuse, vUv);
                    vec2 uv = (vUv - vec2(0.5)) * vec2(offset);
                    float vignette = 1.0 - dot(uv, uv);
                    gl_FragColor = vec4(mix(texel.rgb, texel.rgb * vignette, darkness), texel.a);
                }
            `
        };
        
        const vignettePass = new THREE.ShaderPass(vignetteShader);
        vignettePass.renderToScreen = true; // This should be the last pass
        this.composer.addPass(vignettePass);
        this.passes.vignette = vignettePass;
    }
    
    render() {
        if (this.composer) {
            // Update time-based uniforms
            if (this.passes.film) {
                this.passes.film.uniforms.time.value = performance.now() * 0.001;
            }
            
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    setSize(width, height) {
        if (this.composer && this.composer.setSize) {
            this.composer.setSize(width, height);
            
            // Update FXAA resolution
            if (this.passes.fxaa) {
                const pixelRatio = this.renderer.getPixelRatio();
                this.passes.fxaa.material.uniforms['resolution'].value.x = 1 / (width * pixelRatio);
                this.passes.fxaa.material.uniforms['resolution'].value.y = 1 / (height * pixelRatio);
            }
        }
    }
    
    setCombatMode(enabled) {
        // Adjust post-processing for combat mode
        if (this.passes.bloom) {
            this.passes.bloom.strength = enabled ? 2.0 : 1.5;
        }
        if (this.passes.film) {
            this.passes.film.uniforms.nIntensity.value = enabled ? 0.3 : 0.15;
            this.passes.film.uniforms.sIntensity.value = enabled ? 0.1 : 0.03;
        }
        if (this.passes.vignette) {
            this.passes.vignette.uniforms.darkness.value = enabled ? 1.5 : 1.3;
        }
    }
    
    setQuality(quality) {
        // Adjust post-processing quality
        switch (quality) {
            case 'low':
                if (this.passes.bloom) this.passes.bloom.enabled = false;
                if (this.passes.fxaa) this.passes.fxaa.enabled = false;
                break;
            case 'medium':
                if (this.passes.bloom) this.passes.bloom.enabled = true;
                if (this.passes.fxaa) this.passes.fxaa.enabled = false;
                break;
            case 'high':
                if (this.passes.bloom) this.passes.bloom.enabled = true;
                if (this.passes.fxaa) this.passes.fxaa.enabled = true;
                break;
        }
    }
}

// Export for use
window.PostProcessingEffects = PostProcessingEffects;