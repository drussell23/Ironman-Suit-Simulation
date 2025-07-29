# Unity Setup Guide for Iron Man Suit Simulation

## Prerequisites

1. **Unity Hub** - Download from https://unity.com/download
2. **Unity Editor** - Version 2022.3.x LTS (recommended)
3. **Visual Studio** or **VSCode** for C# editing

## Project Setup

### 1. Open the Project

1. Launch Unity Hub
2. Click "Open" → "Add project from disk"
3. Navigate to: `/Users/derekjrussell/Documents/repos/IronMan/simulation/unity_simulation_ui/Iron-Man-Suit-Simulation`
4. Select the folder and click "Open"

### 2. Install Required Packages

The project uses these Unity packages:
- Universal Render Pipeline (URP)
- Input System
- TextMeshPro
- AI Navigation

If prompted, allow Unity to install/update these packages.

### 3. Project Structure

```
Assets/
├── Scripts/
│   ├── Controllers/      # Input and control systems
│   ├── Managers/        # High-level game systems
│   ├── Services/        # Backend communication
│   ├── SimulationModules/ # Physics and simulation
│   └── UI/             # HUD and displays
├── Scenes/
│   └── SampleScene.unity # Main simulation scene
├── Plugins/            # Native libraries (physics plugin)
└── Models/            # 3D models and animations
```

## Quick Start

### 1. Open the Main Scene

1. In Project window: `Assets/Scenes/SampleScene.unity`
2. Double-click to open

### 2. Configure Backend Connection

Create a new script to connect to the Python backend:

**File:** `Assets/Scripts/Services/BackendConnector.cs`

```csharp
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System.Text;

[System.Serializable]
public class FlightData
{
    public float[] position;
    public float[] velocity;
    public float[] rotation;
    public float altitude;
    public float thrust;
}

public class BackendConnector : MonoBehaviour
{
    [Header("Connection Settings")]
    [SerializeField] private string backendUrl = "http://localhost:8000";
    [SerializeField] private float updateInterval = 0.1f; // 10 Hz
    
    [Header("References")]
    [SerializeField] private Transform ironManSuit;
    
    private FlightData currentFlightData;
    private Coroutine updateCoroutine;
    
    void Start()
    {
        // Start connection to backend
        updateCoroutine = StartCoroutine(ConnectToBackend());
    }
    
    IEnumerator ConnectToBackend()
    {
        while (true)
        {
            // Get flight data from backend
            yield return StartCoroutine(GetFlightData());
            
            // Apply to suit
            if (currentFlightData != null && ironManSuit != null)
            {
                ApplyFlightData();
            }
            
            yield return new WaitForSeconds(updateInterval);
        }
    }
    
    IEnumerator GetFlightData()
    {
        using (UnityWebRequest request = UnityWebRequest.Get(backendUrl + "/api/flight/state"))
        {
            yield return request.SendWebRequest();
            
            if (request.result == UnityWebRequest.Result.Success)
            {
                currentFlightData = JsonUtility.FromJson<FlightData>(request.downloadHandler.text);
            }
            else
            {
                Debug.LogWarning($"Backend connection failed: {request.error}");
            }
        }
    }
    
    void ApplyFlightData()
    {
        // Update position
        ironManSuit.position = new Vector3(
            currentFlightData.position[0],
            currentFlightData.position[1],
            currentFlightData.position[2]
        );
        
        // Update rotation based on velocity
        if (currentFlightData.velocity.Length >= 3)
        {
            Vector3 velocity = new Vector3(
                currentFlightData.velocity[0],
                currentFlightData.velocity[1],
                currentFlightData.velocity[2]
            );
            
            if (velocity.magnitude > 0.1f)
            {
                ironManSuit.rotation = Quaternion.LookRotation(velocity);
            }
        }
    }
    
    // Send control commands back to backend
    public void SendControlCommand(string command, float value)
    {
        StartCoroutine(SendCommand(command, value));
    }
    
    IEnumerator SendCommand(string command, float value)
    {
        var data = new { command = command, value = value };
        string json = JsonUtility.ToJson(data);
        
        using (UnityWebRequest request = new UnityWebRequest(backendUrl + "/api/control", "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            
            yield return request.SendWebRequest();
            
            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"Command failed: {request.error}");
            }
        }
    }
}
```

### 3. Basic Iron Man Suit Controller

**File:** `Assets/Scripts/Controllers/IronManController.cs`

```csharp
using UnityEngine;
using UnityEngine.InputSystem;

public class IronManController : MonoBehaviour
{
    [Header("Movement Settings")]
    [SerializeField] private float thrustPower = 1000f;
    [SerializeField] private float rotationSpeed = 90f;
    [SerializeField] private float maxSpeed = 100f;
    
    [Header("Components")]
    [SerializeField] private Rigidbody rb;
    [SerializeField] private BackendConnector backend;
    [SerializeField] private ParticleSystem[] thrusters;
    
    [Header("Input")]
    private Vector2 moveInput;
    private float thrustInput;
    private float yawInput;
    
    void Start()
    {
        if (rb == null) rb = GetComponent<Rigidbody>();
        if (backend == null) backend = GetComponent<BackendConnector>();
        
        // Configure physics
        rb.useGravity = true;
        rb.drag = 0.5f;
        rb.angularDrag = 2f;
    }
    
    void Update()
    {
        // Get input (using new Input System)
        HandleInput();
        
        // Update thruster visuals
        UpdateThrusters();
    }
    
    void FixedUpdate()
    {
        // Apply physics
        ApplyThrust();
        ApplyRotation();
        
        // Limit speed
        if (rb.velocity.magnitude > maxSpeed)
        {
            rb.velocity = rb.velocity.normalized * maxSpeed;
        }
    }
    
    void HandleInput()
    {
        // Get input from keyboard/gamepad
        var keyboard = Keyboard.current;
        if (keyboard != null)
        {
            // WASD for movement
            moveInput = new Vector2(
                keyboard.dKey.ReadValue() - keyboard.aKey.ReadValue(),
                keyboard.wKey.ReadValue() - keyboard.sKey.ReadValue()
            );
            
            // Space for thrust
            thrustInput = keyboard.spaceKey.ReadValue();
            
            // Q/E for yaw
            yawInput = keyboard.eKey.ReadValue() - keyboard.qKey.ReadValue();
        }
    }
    
    void ApplyThrust()
    {
        // Main thrust
        if (thrustInput > 0)
        {
            Vector3 thrust = transform.up * thrustPower * thrustInput;
            rb.AddForce(thrust);
            
            // Send to backend
            backend?.SendControlCommand("thrust", thrustPower * thrustInput);
        }
        
        // Directional thrust
        if (moveInput.magnitude > 0)
        {
            Vector3 move = transform.right * moveInput.x + transform.forward * moveInput.y;
            rb.AddForce(move * thrustPower * 0.5f);
        }
    }
    
    void ApplyRotation()
    {
        // Pitch and roll based on movement
        float pitch = -moveInput.y * rotationSpeed * Time.deltaTime;
        float roll = -moveInput.x * rotationSpeed * 0.5f * Time.deltaTime;
        float yaw = yawInput * rotationSpeed * Time.deltaTime;
        
        transform.Rotate(pitch, yaw, roll);
    }
    
    void UpdateThrusters()
    {
        // Update particle effects based on thrust
        foreach (var thruster in thrusters)
        {
            var emission = thruster.emission;
            emission.enabled = thrustInput > 0;
            
            var main = thruster.main;
            main.startSpeed = 10f * thrustInput;
        }
    }
}
```

### 4. Setting Up the Iron Man Suit

1. **Create the Suit GameObject:**
   - Right-click in Hierarchy → Create Empty
   - Name it "IronManSuit"
   - Add Components:
     - Rigidbody
     - IronManController (script)
     - BackendConnector (script)

2. **Add Visual Model:**
   - Add a child GameObject with a basic model (Capsule for testing)
   - Scale and position appropriately

3. **Add Thrusters:**
   - Create child GameObjects for thrusters
   - Add Particle System component to each
   - Configure for jet/flame effect

### 5. UI Setup

Create a simple HUD to display telemetry:

**File:** `Assets/Scripts/UI/HUD.cs`

```csharp
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class HUD : MonoBehaviour
{
    [Header("UI Elements")]
    [SerializeField] private TextMeshProUGUI altitudeText;
    [SerializeField] private TextMeshProUGUI speedText;
    [SerializeField] private TextMeshProUGUI thrustText;
    [SerializeField] private Slider thrustBar;
    
    [Header("References")]
    [SerializeField] private Transform ironManSuit;
    [SerializeField] private Rigidbody suitRigidbody;
    
    void Update()
    {
        if (ironManSuit == null) return;
        
        // Update altitude
        float altitude = ironManSuit.position.y;
        altitudeText.text = $"ALT: {altitude:F1}m";
        
        // Update speed
        if (suitRigidbody != null)
        {
            float speed = suitRigidbody.velocity.magnitude;
            speedText.text = $"SPD: {speed:F1}m/s";
        }
        
        // Update thrust (get from controller)
        var controller = ironManSuit.GetComponent<IronManController>();
        if (controller != null)
        {
            // You'd need to expose thrust value from controller
            thrustText.text = $"THR: {0:F0}%";
        }
    }
}
```

## Running the Simulation

### Standalone Unity Mode

1. Click Play button in Unity
2. Use controls:
   - **WASD**: Directional movement
   - **Space**: Main thrust
   - **Q/E**: Yaw rotation
   - **Mouse**: Look around

### Connected to Python Backend

1. Start the Python backend first:
   ```bash
   cd /Users/derekjrussell/Documents/repos/IronMan
   python backend/api/main.py
   ```

2. In Unity, ensure BackendConnector URL is set to `http://localhost:8000`

3. Click Play - Unity will connect to the backend

## Troubleshooting

### Common Issues

1. **"Can't find script" errors**
   - Right-click Scripts folder → Reimport

2. **Pink/Missing materials**
   - Switch to URP materials in Window → Rendering → Render Pipeline Converter

3. **Backend connection fails**
   - Check Python backend is running
   - Verify firewall isn't blocking port 8000
   - Check URL in BackendConnector

4. **Physics feels wrong**
   - Adjust Rigidbody mass (100kg default)
   - Tweak drag values
   - Check Time.fixedDeltaTime (0.02 recommended)

### Performance Optimization

1. **Quality Settings:**
   - Edit → Project Settings → Quality
   - Use "Medium" for testing

2. **Physics Settings:**
   - Edit → Project Settings → Physics
   - Default Solver Iterations: 6
   - Default Solver Velocity Iterations: 1

## Next Steps

1. **Add More Features:**
   - Weapon systems (repulsors)
   - Jarvis AI integration
   - Damage system
   - Environmental effects

2. **Improve Visuals:**
   - Import Iron Man 3D model
   - Add shader effects
   - Implement HUD overlay

3. **VR Support:**
   - Install XR Plugin Management
   - Add VR hand controllers
   - Implement gesture controls

## Resources

- Unity Learn: https://learn.unity.com
- Universal RP docs: https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@latest
- Input System: https://docs.unity3d.com/Packages/com.unity.inputsystem@latest