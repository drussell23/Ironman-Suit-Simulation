# Iron Man Experience - Web Interface

A fully immersive web-based Iron Man suit experience featuring JARVIS AI, interactive HUD, mission simulations, and 3D visualization.

## Features

- **JARVIS AI System**: Voice commands and natural language processing
- **Interactive HUD**: Real-time suit status, navigation, targeting, and weapon systems
- **Mission Simulations**: Combat, rescue, and reconnaissance missions
- **3D Visualization**: WebGL-powered viewport with flight controls
- **Immersive Audio**: Dynamic sound effects and spatial audio
- **Responsive Design**: Works on desktop and mobile browsers

## Quick Start

### Option 1: Using the Express Server

```bash
cd frontend
npm install
npm run serve
```

Then open http://localhost:8080

### Option 2: Python Simple Server

```bash
cd frontend/IronManExperience
python -m http.server 8080
```

### Option 3: Direct File Access

Open `frontend/IronManExperience/index.html` directly in Chrome or Firefox.

## Controls

### Keyboard
- **WASD**: Movement
- **Q/E**: Altitude control  
- **Shift**: Boost
- **Space/Click**: Fire repulsors
- **Tab**: Target lock
- **J**: JARVIS voice command
- **M**: Mission select
- **H**: Toggle HUD
- **1-3**: Select weapons
- **F1**: Help
- **ESC**: Abort mission

### Voice Commands
- "status" - System status report
- "scan" - Environmental scan
- "weapons" - Weapon systems control
- "mission" - Open mission select
- "analysis" - Analysis mode
- "emergency" - Emergency protocol

## URL Parameters

- `?skipBoot=true` - Skip boot sequence (for development)
- `?quality=low|medium|high` - Graphics quality setting
- `?debug=true` - Show performance stats

## Browser Requirements

- WebGL support (Chrome 90+, Firefox 88+, Safari 14+)
- ES6 JavaScript support
- Hardware acceleration recommended
- Microphone access for voice commands (optional)

## Architecture

```
index.html          - Main HTML structure
css/
  ├── main.css      - Core styles and boot sequence
  ├── hud.css       - HUD interface styles
  ├── jarvis.css    - JARVIS interface styles
  └── animations.css - Animation keyframes
js/
  ├── utils.js      - Utility functions
  ├── boot-sequence.js - Boot animation
  ├── jarvis.js     - JARVIS AI system
  ├── hud.js        - HUD management
  ├── viewport.js   - 3D visualization
  ├── audio.js      - Audio system
  ├── missions.js   - Mission logic
  └── main.js       - Main application
assets/
  ├── audio/        - Sound effects (not included)
  ├── images/       - UI graphics
  └── fonts/        - Custom fonts
```

## Customization

### Adding New Missions

Edit `js/missions.js` and add to `missionTemplates`:

```javascript
newMission: {
    name: 'MISSION NAME',
    description: 'Mission description',
    difficulty: 1-5,
    timeLimit: seconds,
    objectives: [...],
    rewards: { score: 1000, experience: 100 }
}
```

### Custom JARVIS Commands

Edit `js/jarvis.js` and add to `commands` object:

```javascript
'custom': () => this.handleCustomCommand()
```

### HUD Modifications

Edit `css/hud.css` for styling and `js/hud.js` for functionality.

## Performance Tips

- Use `?quality=low` on slower devices
- Close other browser tabs
- Enable hardware acceleration in browser settings
- Use Chrome for best WebGL performance

## Known Issues

- Audio may require user interaction to start (browser security)
- Voice recognition requires HTTPS in production
- Some mobile browsers have limited WebGL support

## License

For demonstration and educational purposes only.
Stark Industries © 2024