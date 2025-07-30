# Iron Man UI Animation System

This folder contains a comprehensive animation system for the Iron Man suit's user interface, providing cinematic and interactive animations that bring the HUD to life.

## Animation Components

### 1. UIAnimationController.cs
**Core animation engine** that provides:
- Centralized animation management
- Reusable animation methods (Float, Vector3, Color)
- Preset animations (FadeIn, FadeOut, ScaleBounce, SlideIn)
- Animation queuing and cancellation
- Easing curve support

### 2. HUDAnimations.cs
**HUD-specific animations** including:
- Boot sequence with cascading panel activation
- Scanning line effects (vertical/horizontal)
- Dynamic targeting system with reticle management
- Radar sweep animation with blip generation
- Data stream with typewriter effect
- Power down/shutdown sequences
- Screen glitch effects

### 3. MenuTransitions.cs
**Menu navigation animations** featuring:
- Multiple transition types (Fade, Slide, IronManAssemble, HolographicWipe, ReactorBurst)
- Per-element animation with staggered timing
- Iron Man themed effects (assembly/disassembly)
- Support for complex menu hierarchies
- Audio integration for transition sounds

### 4. AlertAnimations.cs
**Alert and warning system** with:
- Multiple alert types (Critical, Warning, Damage, Missile, etc.)
- Various animation styles (Pulse, Flash, EdgeGlow, Glitch)
- Screen effects (shake, flash, vignette)
- Alert queuing and priority management
- Customizable per-alert configurations

### 5. HolographicEffects.cs
**Holographic UI effects** providing:
- Material-based hologram rendering
- Animated scanlines and grid overlays
- Noise and distortion effects
- Build/dismiss animations
- Projection effects from point to point
- Glitch effects for damaged states

### 6. PowerSequenceAnimations.cs
**Power-up/down sequences** featuring:
- Arc reactor initialization animation
- Energy distribution visualization
- Multi-stage power sequences
- System status messaging
- Emergency shutdown procedures
- Synchronized audio and visual effects

### 7. DataVisualizationAnimations.cs
**Data visualization components** including:
- Line graphs with real-time updates
- Animated bar charts
- Radar charts with mesh generation
- 3D holographic graphs
- Waveform displays
- Performance metrics
- Data stream visualization

### 8. JARVISInterfaceAnimations.cs
**JARVIS-specific animations** with:
- Voice waveform visualization
- Avatar state management
- Typewriter text effects with word highlighting
- Analysis panel animations
- Status indicators and lights
- Listening/speaking state visualization

## Usage Examples

### Basic Animation
```csharp
// Get the animation controller
UIAnimationController animController = UIAnimationController.Instance;

// Fade in a UI element
animController.FadeIn(canvasGroup, duration: 0.5f, onComplete: () => {
    Debug.Log("Fade complete!");
});

// Animate a custom value
animController.AnimateFloat("MyAnimation", 0f, 100f, 1f, 
    (value) => slider.value = value);
```

### HUD Boot Sequence
```csharp
// Play the full boot sequence
HUDAnimations hudAnims = GetComponent<HUDAnimations>();
hudAnims.PlayBootSequence();

// Add a target
TargetReticle target = hudAnims.AddTarget(enemyPosition, "HOSTILE");
```

### Menu Transitions
```csharp
// Transition between menus
MenuTransitions menuTransitions = GetComponent<MenuTransitions>();
menuTransitions.TransitionToMenu("MainMenu", TransitionType.IronManAssemble);
```

### Alerts
```csharp
// Show critical alert
AlertAnimations alerts = GetComponent<AlertAnimations>();
alerts.ShowCriticalAlert("MISSILE LOCK DETECTED", duration: 5f);

// Trigger screen effect
alerts.TriggerScreenEffect(ScreenEffect.EdgeGlow, Color.red, 2f);
```

### Holographic Effects
```csharp
// Apply hologram effect to UI
HolographicEffects holo = GetComponent<HolographicEffects>();
holo.ApplyHologramEffect(uiPanel, animated: true);

// Trigger glitch
holo.TriggerHologramGlitch(uiPanel, duration: 0.5f);
```

### Power Sequences
```csharp
// Start power-up
PowerSequenceAnimations power = GetComponent<PowerSequenceAnimations>();
power.StartPowerUpSequence();

// Emergency shutdown
power.EmergencyShutdown();
```

### Data Visualization
```csharp
// Update line graph
DataVisualizationAnimations dataViz = GetComponent<DataVisualizationAnimations>();
dataViz.AddLineGraphData(sensorValue);

// Update bar chart
List<GraphData> data = new List<GraphData> {
    new GraphData { label = "Power", value = 85f, color = Color.cyan },
    new GraphData { label = "Speed", value = 120f, color = Color.green }
};
dataViz.UpdateBarChart(data);
```

### JARVIS
```csharp
// Activate JARVIS
JARVISInterfaceAnimations jarvis = GetComponent<JARVISInterfaceAnimations>();
jarvis.ActivateJARVIS();

// Make JARVIS speak
jarvis.Speak("Suit power at maximum capacity, sir.", AnimationType.System);

// Show analysis
Dictionary<string, float> analysisData = new Dictionary<string, float> {
    { "Threat Level", 8.5f },
    { "Distance", 1200f }
};
jarvis.ShowAnalysis("COMBAT", analysisData);
```

## Animation Best Practices

1. **Performance**: Use object pooling for frequently created elements (targets, alerts)
2. **Consistency**: Maintain consistent animation durations across similar UI elements
3. **Feedback**: Always provide visual feedback for user actions
4. **Accessibility**: Ensure critical information is readable even during animations
5. **Mobile**: Test animations on target hardware for performance

## Customization

Most animation parameters are exposed in the Inspector:
- Animation curves for custom easing
- Colors for different states
- Durations and speeds
- Audio clips for feedback
- Material references for effects

## Integration with Backend

The animation system can be integrated with backend data:
- Real-time sensor data → Data visualizations
- System alerts → Alert animations
- Power levels → Power sequences
- AI responses → JARVIS animations

## Performance Considerations

- Animations use coroutines for efficiency
- Object pooling reduces allocation overhead  
- Material instancing prevents conflicts
- LOD system can disable complex effects on lower-end hardware