// Utility functions for Iron Man Experience

const Utils = {
    // DOM helpers
    $: (selector) => document.querySelector(selector),
    $$: (selector) => document.querySelectorAll(selector),
    
    // Create element with attributes
    createElement: (tag, attributes = {}, children = []) => {
        const element = document.createElement(tag);
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'textContent') {
                element.textContent = value;
            } else {
                element.setAttribute(key, value);
            }
        });
        children.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else {
                element.appendChild(child);
            }
        });
        return element;
    },
    
    // Animation helpers
    animate: (element, className, duration = 1000) => {
        return new Promise(resolve => {
            element.classList.add(className);
            setTimeout(() => {
                element.classList.remove(className);
                resolve();
            }, duration);
        });
    },
    
    // Type writer effect
    typeWriter: async (element, text, speed = 50) => {
        element.textContent = '';
        element.classList.add('typing');
        
        for (let i = 0; i < text.length; i++) {
            element.textContent += text.charAt(i);
            await Utils.delay(speed);
        }
        
        element.classList.remove('typing');
    },
    
    // Delay helper
    delay: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
    
    // Random number in range
    random: (min, max) => Math.random() * (max - min) + min,
    
    // Format number with leading zeros
    pad: (num, size) => String(num).padStart(size, '0'),
    
    // Clamp value between min and max
    clamp: (value, min, max) => Math.min(Math.max(value, min), max),
    
    // Linear interpolation
    lerp: (start, end, t) => start + (end - start) * t,
    
    // Map value from one range to another
    map: (value, inMin, inMax, outMin, outMax) => {
        return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
    },
    
    // Format time as MM:SS
    formatTime: (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${Utils.pad(mins, 2)}:${Utils.pad(secs, 2)}`;
    },
    
    // Get distance between two points
    distance: (x1, y1, x2, y2) => {
        return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    },
    
    // Get angle between two points
    angle: (x1, y1, x2, y2) => {
        return Math.atan2(y2 - y1, x2 - x1);
    },
    
    // Convert degrees to radians
    toRadians: (degrees) => degrees * (Math.PI / 180),
    
    // Convert radians to degrees
    toDegrees: (radians) => radians * (180 / Math.PI),
    
    // Throttle function calls
    throttle: (func, limit) => {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    // Debounce function calls
    debounce: (func, delay) => {
        let timeoutId;
        return function() {
            const args = arguments;
            const context = this;
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(context, args), delay);
        };
    },
    
    // Local storage helpers
    storage: {
        get: (key) => {
            try {
                return JSON.parse(localStorage.getItem(key));
            } catch (e) {
                return null;
            }
        },
        set: (key, value) => {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (e) {
                return false;
            }
        },
        remove: (key) => localStorage.removeItem(key),
        clear: () => localStorage.clear()
    },
    
    // Event emitter
    createEventEmitter: () => {
        const events = {};
        return {
            on: (event, callback) => {
                if (!events[event]) events[event] = [];
                events[event].push(callback);
            },
            off: (event, callback) => {
                if (events[event]) {
                    events[event] = events[event].filter(cb => cb !== callback);
                }
            },
            emit: (event, data) => {
                if (events[event]) {
                    events[event].forEach(callback => callback(data));
                }
            }
        };
    },
    
    // Screen shake effect
    screenShake: (intensity = 10, duration = 500) => {
        const body = document.body;
        const startTime = Date.now();
        
        const shake = () => {
            const elapsed = Date.now() - startTime;
            if (elapsed < duration) {
                const x = (Math.random() - 0.5) * intensity;
                const y = (Math.random() - 0.5) * intensity;
                body.style.transform = `translate(${x}px, ${y}px)`;
                requestAnimationFrame(shake);
            } else {
                body.style.transform = 'translate(0, 0)';
            }
        };
        
        shake();
    },
    
    // Check if element is in viewport
    isInViewport: (element) => {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= window.innerHeight &&
            rect.right <= window.innerWidth
        );
    },
    
    // Get random item from array
    randomFrom: (array) => array[Math.floor(Math.random() * array.length)],
    
    // Shuffle array
    shuffle: (array) => {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    },
    
    // Generate UUID
    uuid: () => {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
};

// Make Utils globally available
window.Utils = Utils;