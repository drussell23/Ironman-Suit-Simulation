using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine.SceneManagement;

namespace IronManSim.Frontend
{
    /// <summary>
    /// Main manager for the Iron Man suit experience interface
    /// </summary>
    public class IronManExperienceManager : MonoBehaviour
    {
        [Header("Experience Modes")]
        [SerializeField] private ExperienceMode currentMode = ExperienceMode.HUD;
        [SerializeField] private bool autoStartExperience = true;
        
        public enum ExperienceMode
        {
            Loading,
            Bootup,
            HUD,
            Mission,
            Combat,
            Analysis,
            Emergency
        }
        
        [Header("Core Systems")]
        [SerializeField] private JARVISSystem jarvisSystem;
        [SerializeField] private HUDManager hudManager;
        [SerializeField] private MissionSystem missionSystem;
        [SerializeField] private InputManager inputManager;
        [SerializeField] private AudioManager audioManager;
        
        [Header("Cameras")]
        [SerializeField] private Camera mainCamera;
        [SerializeField] private Camera hudCamera;
        [SerializeField] private float defaultFOV = 60f;
        [SerializeField] private float combatFOV = 75f;
        
        [Header("UI Containers")]
        [SerializeField] private Canvas mainCanvas;
        [SerializeField] private Canvas hudCanvas;
        [SerializeField] private Canvas overlayCanvas;
        
        [Header("Experience Settings")]
        [SerializeField] private float bootSequenceDuration = 5f;
        [SerializeField] private bool enableVoiceCommands = true;
        [SerializeField] private bool enableMotionEffects = true;
        
        private static IronManExperienceManager instance;
        private bool isInitialized = false;
        
        public static IronManExperienceManager Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = FindObjectOfType<IronManExperienceManager>();
                }
                return instance;
            }
        }
        
        void Awake()
        {
            if (instance == null)
            {
                instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else if (instance != this)
            {
                Destroy(gameObject);
            }
            
            InitializeSystems();
        }
        
        void Start()
        {
            if (autoStartExperience)
            {
                StartExperience();
            }
        }
        
        #region Initialization
        
        private void InitializeSystems()
        {
            // Create systems if not assigned
            if (jarvisSystem == null)
            {
                jarvisSystem = GetComponentInChildren<JARVISSystem>();
                if (jarvisSystem == null)
                {
                    GameObject jarvisObj = new GameObject("JARVIS System");
                    jarvisObj.transform.SetParent(transform);
                    jarvisSystem = jarvisObj.AddComponent<JARVISSystem>();
                }
            }
            
            if (hudManager == null)
            {
                hudManager = GetComponentInChildren<HUDManager>();
                if (hudManager == null)
                {
                    GameObject hudObj = new GameObject("HUD Manager");
                    hudObj.transform.SetParent(transform);
                    hudManager = hudObj.AddComponent<HUDManager>();
                }
            }
            
            if (missionSystem == null)
            {
                missionSystem = GetComponentInChildren<MissionSystem>();
                if (missionSystem == null)
                {
                    GameObject missionObj = new GameObject("Mission System");
                    missionObj.transform.SetParent(transform);
                    missionSystem = missionObj.AddComponent<MissionSystem>();
                }
            }
            
            SetupCameras();
            SetupCanvases();
            
            isInitialized = true;
        }
        
        private void SetupCameras()
        {
            if (mainCamera == null)
            {
                mainCamera = Camera.main;
                if (mainCamera == null)
                {
                    GameObject camObj = new GameObject("Main Camera");
                    mainCamera = camObj.AddComponent<Camera>();
                    mainCamera.tag = "MainCamera";
                }
            }
            
            // Setup HUD camera for UI overlay
            if (hudCamera == null)
            {
                GameObject hudCamObj = new GameObject("HUD Camera");
                hudCamObj.transform.SetParent(mainCamera.transform);
                hudCamera = hudCamObj.AddComponent<Camera>();
                hudCamera.clearFlags = CameraClearFlags.Depth;
                hudCamera.cullingMask = 1 << LayerMask.NameToLayer("UI");
                hudCamera.depth = 1;
            }
        }
        
        private void SetupCanvases()
        {
            // Main canvas for 3D world UI
            if (mainCanvas == null)
            {
                GameObject mainCanvasObj = new GameObject("Main Canvas");
                mainCanvas = mainCanvasObj.AddComponent<Canvas>();
                mainCanvas.renderMode = RenderMode.ScreenSpaceCamera;
                mainCanvas.worldCamera = mainCamera;
                mainCanvasObj.AddComponent<CanvasScaler>();
                mainCanvasObj.AddComponent<GraphicRaycaster>();
            }
            
            // HUD canvas for overlay UI
            if (hudCanvas == null)
            {
                GameObject hudCanvasObj = new GameObject("HUD Canvas");
                hudCanvas = hudCanvasObj.AddComponent<Canvas>();
                hudCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
                hudCanvas.sortingOrder = 100;
                hudCanvasObj.AddComponent<CanvasScaler>();
                hudCanvasObj.AddComponent<GraphicRaycaster>();
            }
            
            // Overlay canvas for effects
            if (overlayCanvas == null)
            {
                GameObject overlayCanvasObj = new GameObject("Overlay Canvas");
                overlayCanvas = overlayCanvasObj.AddComponent<Canvas>();
                overlayCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
                overlayCanvas.sortingOrder = 200;
                overlayCanvasObj.AddComponent<CanvasScaler>();
                overlayCanvasObj.AddComponent<GraphicRaycaster>();
            }
        }
        
        #endregion
        
        #region Experience Control
        
        public void StartExperience()
        {
            if (!isInitialized) return;
            
            StartCoroutine(ExperienceStartSequence());
        }
        
        private IEnumerator ExperienceStartSequence()
        {
            // Set loading mode
            SetExperienceMode(ExperienceMode.Loading);
            
            // Initialize all systems
            yield return new WaitForSeconds(0.5f);
            
            // Start boot sequence
            SetExperienceMode(ExperienceMode.Bootup);
            yield return StartCoroutine(BootSequence());
            
            // Activate HUD
            SetExperienceMode(ExperienceMode.HUD);
            hudManager.ActivateHUD();
            
            // Start JARVIS
            jarvisSystem.Initialize();
            jarvisSystem.Speak("Good evening, sir. All systems are online and ready.");
            
            // Enable input
            if (inputManager != null)
            {
                inputManager.EnableInput();
            }
        }
        
        private IEnumerator BootSequence()
        {
            // Play boot sound
            if (audioManager != null)
            {
                audioManager.PlayBootSequence();
            }
            
            // Show boot UI
            hudManager.ShowBootSequence();
            
            // Simulate system initialization
            float elapsed = 0;
            while (elapsed < bootSequenceDuration)
            {
                elapsed += Time.deltaTime;
                float progress = elapsed / bootSequenceDuration;
                
                hudManager.UpdateBootProgress(progress);
                
                // Add boot messages
                if (progress > 0.2f && progress < 0.25f)
                {
                    hudManager.AddBootMessage("Initializing Arc Reactor...");
                }
                else if (progress > 0.4f && progress < 0.45f)
                {
                    hudManager.AddBootMessage("Calibrating Repulsor Arrays...");
                }
                else if (progress > 0.6f && progress < 0.65f)
                {
                    hudManager.AddBootMessage("Loading J.A.R.V.I.S...");
                }
                else if (progress > 0.8f && progress < 0.85f)
                {
                    hudManager.AddBootMessage("Establishing Neural Link...");
                }
                
                yield return null;
            }
            
            hudManager.AddBootMessage("SYSTEMS ONLINE");
            yield return new WaitForSeconds(1f);
        }
        
        public void SetExperienceMode(ExperienceMode mode)
        {
            currentMode = mode;
            
            // Update systems based on mode
            switch (mode)
            {
                case ExperienceMode.HUD:
                    hudManager.SetHUDMode(HUDManager.HUDMode.Standard);
                    mainCamera.fieldOfView = defaultFOV;
                    break;
                    
                case ExperienceMode.Mission:
                    hudManager.SetHUDMode(HUDManager.HUDMode.Mission);
                    missionSystem.ShowMissionInterface();
                    break;
                    
                case ExperienceMode.Combat:
                    hudManager.SetHUDMode(HUDManager.HUDMode.Combat);
                    mainCamera.fieldOfView = combatFOV;
                    hudManager.EnableTargeting();
                    break;
                    
                case ExperienceMode.Analysis:
                    hudManager.SetHUDMode(HUDManager.HUDMode.Analysis);
                    hudManager.ShowAnalysisOverlay();
                    break;
                    
                case ExperienceMode.Emergency:
                    hudManager.SetHUDMode(HUDManager.HUDMode.Emergency);
                    hudManager.ShowEmergencyAlerts();
                    break;
            }
            
            // Notify JARVIS of mode change
            if (jarvisSystem != null && mode != ExperienceMode.Loading)
            {
                jarvisSystem.OnModeChanged(mode);
            }
        }
        
        #endregion
        
        #region Mission Control
        
        public void StartMission(string missionId)
        {
            SetExperienceMode(ExperienceMode.Mission);
            missionSystem.LoadMission(missionId);
        }
        
        public void EndMission()
        {
            missionSystem.EndCurrentMission();
            SetExperienceMode(ExperienceMode.HUD);
        }
        
        #endregion
        
        #region Input Handling
        
        public void ProcessVoiceCommand(string command)
        {
            if (!enableVoiceCommands) return;
            
            jarvisSystem.ProcessVoiceCommand(command);
        }
        
        public void ProcessGesture(string gesture)
        {
            switch (gesture.ToLower())
            {
                case "fist":
                    hudManager.TriggerRepulsor();
                    break;
                case "palm":
                    hudManager.ShowHandInterface();
                    break;
                case "swipe_left":
                    hudManager.CycleHUDMode(-1);
                    break;
                case "swipe_right":
                    hudManager.CycleHUDMode(1);
                    break;
            }
        }
        
        #endregion
        
        #region WebGL Specific
        
        /// <summary>
        /// Called from JavaScript when running in browser
        /// </summary>
        public void OnWebGLReady()
        {
            Debug.Log("Iron Man Experience ready in WebGL");
            
            // Adjust settings for web
            QualitySettings.pixelLightCount = 2;
            QualitySettings.shadows = ShadowQuality.HardOnly;
            
            // Start experience
            StartExperience();
        }
        
        /// <summary>
        /// Handle browser resize
        /// </summary>
        public void OnBrowserResize(int width, int height)
        {
            // Adjust UI scaling
            CanvasScaler[] scalers = FindObjectsOfType<CanvasScaler>();
            foreach (var scaler in scalers)
            {
                scaler.referenceResolution = new Vector2(width, height);
            }
        }
        
        #endregion
        
        #region Public API
        
        public void ToggleHUD()
        {
            hudManager.ToggleVisibility();
        }
        
        public void SetHUDOpacity(float opacity)
        {
            hudManager.SetOpacity(opacity);
        }
        
        public void TriggerAlert(string message, AlertLevel level)
        {
            hudManager.ShowAlert(message, level);
            
            if (level == AlertLevel.Critical)
            {
                jarvisSystem.Speak($"Critical alert: {message}");
            }
        }
        
        public void UpdateSuitStatus(SuitStatus status)
        {
            hudManager.UpdateSuitStatus(status);
        }
        
        #endregion
    }
    
    [System.Serializable]
    public class SuitStatus
    {
        public float powerLevel = 100f;
        public float armorIntegrity = 100f;
        public float coreTemperature = 36.5f;
        public Vector3 velocity = Vector3.zero;
        public float altitude = 0f;
        public int activeTargets = 0;
        public bool weaponsOnline = true;
        public bool shieldsActive = true;
    }
    
    public enum AlertLevel
    {
        Info,
        Warning,
        Critical,
        Emergency
    }
}