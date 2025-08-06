// Audio context fix for autoplay policy

document.addEventListener('DOMContentLoaded', () => {
    // Add click handler to resume audio context
    const resumeAudio = () => {
        if (window.AudioSystem && window.AudioSystem.context) {
            const ctx = window.AudioSystem.context;
            if (ctx.state === 'suspended') {
                ctx.resume().then(() => {
                    console.log('Audio context resumed');
                    // Remove the listener after first interaction
                    document.removeEventListener('click', resumeAudio);
                    document.removeEventListener('keydown', resumeAudio);
                });
            }
        }
    };
    
    // Listen for user interaction
    document.addEventListener('click', resumeAudio);
    document.addEventListener('keydown', resumeAudio);
    
    // Also add a visible prompt if needed
    if (window.AudioSystem && window.AudioSystem.context && window.AudioSystem.context.state === 'suspended') {
        const prompt = document.createElement('div');
        prompt.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 168, 255, 0.9);
            color: black;
            padding: 15px 30px;
            border-radius: 5px;
            font-weight: bold;
            z-index: 10000;
            cursor: pointer;
            animation: pulse 2s infinite;
        `;
        prompt.textContent = 'Click anywhere to enable audio';
        prompt.onclick = () => {
            resumeAudio();
            prompt.remove();
        };
        document.body.appendChild(prompt);
        
        // Add pulse animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0% { transform: translateX(-50%) scale(1); }
                50% { transform: translateX(-50%) scale(1.05); }
                100% { transform: translateX(-50%) scale(1); }
            }
        `;
        document.head.appendChild(style);
    }
});