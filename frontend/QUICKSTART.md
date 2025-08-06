# Iron Man Suit Experience - Quick Start Guide

## 🚀 Getting Started

### 1. Unity Setup
1. Open Unity Hub and load the project from `IronManSuitSim3D/`
2. Open the scene: `Assets/Scenes/SampleScene.unity`
3. Add the IronManSceneSetup component:
   - Create empty GameObject
   - Add Component > Iron Man Sim > Frontend > Iron Man Scene Setup
   - Click "Setup Iron Man Experience" in inspector

### 2. Quick Test in Unity
- Press Play button
- Use controls:
  - **WASD** - Move
  - **Mouse** - Look around
  - **Shift** - Boost
  - **Left Click** - Fire repulsors
  - **Tab** - Target lock
  - **J** - JARVIS command
  - **H** - Toggle HUD

### 3. Build for WebGL
#### Option A: Unity Editor
1. File > Build Settings
2. Switch Platform to WebGL
3. Menu: IronMan > Build > Quick WebGL Build
4. Wait for build to complete (5-10 minutes)

#### Option B: Command Line
```bash
cd frontend
./build-webgl.sh
```

### 4. Test Locally
```bash
cd frontend
npm install
npm run serve
```

Open browser: http://localhost:8080

## 🎮 Controls

### Keyboard
- **Movement**: WASD or Arrow Keys
- **Look**: Mouse (hold right-click in browser)
- **Fire**: Left Mouse / Space
- **Boost**: Shift
- **Target Lock**: Tab
- **JARVIS**: J
- **HUD Toggle**: H
- **Weapon Select**: 1-9
- **Mission Abort**: ESC

### Touch (Mobile/Tablet)
- **Look**: Swipe
- **Fire**: Tap
- **Target**: Double tap on enemy
- **Zoom**: Pinch

## 🎯 Features

### HUD Modes
1. **Standard** - Basic flight HUD
2. **Combat** - Weapon systems active
3. **Mission** - Objectives displayed
4. **Analysis** - Environmental scan
5. **Emergency** - Critical alerts
6. **Stealth** - Minimal HUD

### Mission Types
- **Combat** - Eliminate hostiles
- **Rescue** - Save civilians
- **Recon** - Stealth reconnaissance
- **Defense** - Protect objectives
- **Training** - Practice exercises

### JARVIS Commands
- "status" - System status
- "scan area" - Environmental scan
- "weapons online" - Activate weapons
- "power boost" - Increase power
- "analysis" - Switch to analysis mode
- "mission start" - Begin mission
- "emergency" - Emergency protocol

## 🛠️ Customization

### Adding Audio
1. Import audio clips to `Assets/Audio/`
2. Assign in AudioManager component:
   - Boot sounds
   - Weapon effects
   - Music tracks
   - JARVIS voice

### Custom Missions
1. Create mission prefabs in `Assets/Prefabs/Missions/`
2. Add to MissionSystem templates
3. Configure objectives and scenarios

### WebGL Template
Edit `frontend/IronManExperience/TemplateData/` for custom:
- Loading screen
- Progress bar
- Browser UI

## 📱 Browser Compatibility

### Recommended
- Chrome 90+
- Firefox 88+
- Edge 90+

### Mobile
- Chrome Android
- Safari iOS 14+

### Requirements
- WebGL 2.0
- Hardware acceleration
- 2GB+ RAM

## 🐛 Troubleshooting

### Unity Issues
- **Scripts not compiling**: Reimport All (Assets > Reimport All)
- **Menu items missing**: Restart Unity
- **WebGL build fails**: Check Player Settings > WebGL

### Browser Issues
- **Black screen**: Enable hardware acceleration
- **Low performance**: Lower quality settings
- **Audio not playing**: User interaction required first

### Server Issues
- **Port in use**: Change PORT in server.js
- **MIME type errors**: Update server headers
- **CORS errors**: Check server CORS settings

## 🚀 Production Deployment

### Optimize Build
1. Unity: Edit > Project Settings > Player
2. Publishing Settings:
   - Compression: Gzip
   - Decompression Fallback: enabled
3. Build with IL2CPP for better performance

### Deploy to Web
1. Upload contents of `Build/` folder
2. Configure server for:
   - Gzip compression
   - Proper MIME types
   - CORS headers (if needed)

### CDN Setup
- Host `.data` files on CDN
- Update `index.html` with CDN URLs
- Enable caching headers

## 📞 Support

- GitHub Issues: [Report bugs](https://github.com/starkindustries/ironman/issues)
- Documentation: See `/docs` folder
- Unity Forums: Search "Iron Man Experience"

---

**Remember**: With great power comes great responsibility. Use the Iron Man suit wisely!

*JARVIS is watching.* 🤖