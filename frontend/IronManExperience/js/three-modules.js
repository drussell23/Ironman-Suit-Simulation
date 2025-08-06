// Three.js ES6 module loader
// This file helps with the migration from the deprecated three.min.js

// For now, we'll continue using the minified version but prepare for ES6 modules
// In the future, replace with:
// import * as THREE from 'https://unpkg.com/three@0.150.0/build/three.module.js';

// Suppress the deprecation warning
if (typeof console !== 'undefined' && console.warn) {
    const originalWarn = console.warn;
    console.warn = function(...args) {
        if (args[0] && typeof args[0] === 'string' && args[0].includes('three.js') && args[0].includes('deprecated')) {
            // Suppress three.js deprecation warning
            return;
        }
        originalWarn.apply(console, args);
    };
}