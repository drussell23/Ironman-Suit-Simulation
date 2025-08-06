using UnityEngine;
using UnityEngine.SceneManagement;

namespace IronManSim.Frontend
{
    /// <summary>
    /// Automatically sets up the Iron Man experience scene
    /// </summary>
    public class IronManSceneSetup : MonoBehaviour
    {
        [Header("Auto Setup")]
        [SerializeField] private bool autoSetupOnStart = true;
        [SerializeField] private bool createTestEnvironment = true;
        
        void Start()
        {
            if (autoSetupOnStart)
            {
                SetupIronManExperience();
            }
        }
        
        [ContextMenu("Setup Iron Man Experience")]
        public void SetupIronManExperience()
        {
            Debug.Log("Setting up Iron Man Experience...");
            
            // Create experience manager if not exists
            if (IronManExperienceManager.Instance == null)
            {
                GameObject managerObj = new GameObject("Iron Man Experience Manager");
                managerObj.AddComponent<IronManExperienceManager>();
            }
            
            // Setup camera
            SetupCamera();
            
            // Create test environment
            if (createTestEnvironment)
            {
                CreateTestEnvironment();
            }
            
            // Configure for WebGL if needed
            #if UNITY_WEBGL
            ConfigureWebGL();
            #endif
            
            Debug.Log("Iron Man Experience setup complete!");
        }
        
        private void SetupCamera()
        {
            if (Camera.main == null)
            {
                GameObject camObj = new GameObject("Main Camera");
                Camera cam = camObj.AddComponent<Camera>();
                cam.tag = "MainCamera";
                cam.transform.position = new Vector3(0, 100, -200);
                cam.transform.rotation = Quaternion.Euler(15, 0, 0);
                
                // Add audio listener
                camObj.AddComponent<AudioListener>();
                
                // Add skybox
                cam.clearFlags = CameraClearFlags.Skybox;
            }
        }
        
        private void CreateTestEnvironment()
        {
            // Create ground
            GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.localScale = new Vector3(100, 1, 100);
            ground.GetComponent<Renderer>().material.color = new Color(0.2f, 0.2f, 0.2f);
            
            // Create some buildings
            for (int i = 0; i < 10; i++)
            {
                GameObject building = GameObject.CreatePrimitive(PrimitiveType.Cube);
                building.name = $"Building_{i}";
                
                float height = Random.Range(50, 200);
                float width = Random.Range(20, 50);
                
                building.transform.localScale = new Vector3(width, height, width);
                building.transform.position = new Vector3(
                    Random.Range(-300, 300),
                    height / 2,
                    Random.Range(-300, 300)
                );
                
                building.GetComponent<Renderer>().material.color = new Color(0.3f, 0.3f, 0.4f);
            }
            
            // Create some test targets
            for (int i = 0; i < 5; i++)
            {
                GameObject target = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                target.name = $"Target_{i}";
                target.tag = "Enemy";
                
                target.transform.localScale = Vector3.one * 5;
                target.transform.position = new Vector3(
                    Random.Range(-200, 200),
                    Random.Range(50, 150),
                    Random.Range(-200, 200)
                );
                
                target.GetComponent<Renderer>().material.color = Color.red;
                
                // Add simple movement
                target.AddComponent<SimpleTargetMovement>();
            }
            
            // Add lighting
            if (RenderSettings.sun == null)
            {
                GameObject lightObj = new GameObject("Directional Light");
                Light light = lightObj.AddComponent<Light>();
                light.type = LightType.Directional;
                light.intensity = 1.2f;
                light.transform.rotation = Quaternion.Euler(45, -30, 0);
            }
        }
        
        private void ConfigureWebGL()
        {
            // Adjust quality for web
            QualitySettings.SetQualityLevel(2); // Medium quality
            Application.targetFrameRate = 60;
            
            // Add WebGL communication bridge
            GameObject webglBridge = new GameObject("WebGL Bridge");
            webglBridge.AddComponent<WebGLBridge>();
        }
    }
    
    /// <summary>
    /// Simple movement for test targets
    /// </summary>
    public class SimpleTargetMovement : MonoBehaviour
    {
        private Vector3 startPosition;
        private float moveSpeed;
        private float moveRadius;
        
        void Start()
        {
            startPosition = transform.position;
            moveSpeed = Random.Range(1f, 3f);
            moveRadius = Random.Range(20f, 50f);
        }
        
        void Update()
        {
            float x = Mathf.Sin(Time.time * moveSpeed) * moveRadius;
            float z = Mathf.Cos(Time.time * moveSpeed) * moveRadius;
            
            transform.position = startPosition + new Vector3(x, 0, z);
        }
    }
    
    /// <summary>
    /// Bridge for WebGL JavaScript communication
    /// </summary>
    public class WebGLBridge : MonoBehaviour
    {
        void Start()
        {
            #if UNITY_WEBGL && !UNITY_EDITOR
            // Register with JavaScript
            Application.ExternalCall("UnityReady");
            #endif
        }
        
        // Called from JavaScript
        public void SendCommand(string command)
        {
            if (IronManExperienceManager.Instance != null)
            {
                IronManExperienceManager.Instance.ProcessVoiceCommand(command);
            }
        }
        
        // Called from JavaScript
        public void SetQuality(int level)
        {
            QualitySettings.SetQualityLevel(level);
        }
    }
}