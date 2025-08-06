// Particle effects for boot sequence
class BootParticles {
    constructor() {
        this.container = document.querySelector('.particle-container');
        this.particles = [];
        this.particleCount = 50;
        this.active = false;
    }
    
    init() {
        if (!this.container) return;
        
        this.active = true;
        this.createParticles();
        this.animate();
    }
    
    createParticles() {
        for (let i = 0; i < this.particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'boot-particle';
            
            // Random starting position in a circle around the arc reactor
            const angle = (Math.PI * 2 * i) / this.particleCount;
            const radius = 150 + Math.random() * 50;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;
            
            particle.style.cssText = `
                position: absolute;
                width: 4px;
                height: 4px;
                background: ${this.getRandomColor()};
                border-radius: 50%;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%) translate(${x}px, ${y}px);
                box-shadow: 0 0 10px currentColor;
                opacity: 0;
            `;
            
            this.container.appendChild(particle);
            
            this.particles.push({
                element: particle,
                x: x,
                y: y,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                life: Math.random() * 100,
                maxLife: 100,
                delay: Math.random() * 2000
            });
        }
    }
    
    getRandomColor() {
        const colors = [
            '#00a8ff', // Primary blue
            '#00d4ff', // Light blue
            '#0080ff', // Medium blue
            '#ffffff', // White
            '#40c4ff'  // Cyan
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }
    
    animate() {
        if (!this.active) return;
        
        this.particles.forEach(particle => {
            if (particle.delay > 0) {
                particle.delay -= 16; // Assuming 60fps
                return;
            }
            
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Add some attraction to center
            const distanceFromCenter = Math.sqrt(particle.x * particle.x + particle.y * particle.y);
            if (distanceFromCenter > 50) {
                particle.vx -= particle.x * 0.001;
                particle.vy -= particle.y * 0.001;
            }
            
            // Update life
            particle.life -= 0.5;
            
            // Reset particle if dead
            if (particle.life <= 0) {
                const angle = Math.random() * Math.PI * 2;
                const radius = 150 + Math.random() * 50;
                particle.x = Math.cos(angle) * radius;
                particle.y = Math.sin(angle) * radius;
                particle.vx = (Math.random() - 0.5) * 2;
                particle.vy = (Math.random() - 0.5) * 2;
                particle.life = particle.maxLife;
                particle.element.style.background = this.getRandomColor();
            }
            
            // Update element
            const opacity = particle.life / particle.maxLife;
            particle.element.style.transform = `translate(-50%, -50%) translate(${particle.x}px, ${particle.y}px)`;
            particle.element.style.opacity = opacity * 0.8;
        });
        
        requestAnimationFrame(() => this.animate());
    }
    
    stop() {
        this.active = false;
        this.particles.forEach(particle => {
            particle.element.style.transition = 'opacity 0.5s ease-out';
            particle.element.style.opacity = '0';
        });
        
        setTimeout(() => {
            this.particles.forEach(particle => {
                particle.element.remove();
            });
            this.particles = [];
        }, 500);
    }
}

// Initialize particles when boot sequence starts
window.BootParticles = BootParticles;