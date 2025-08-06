using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.EventSystems;

namespace IronManSim.Frontend
{
    /// <summary>
    /// Manages input for the Iron Man suit experience across different platforms
    /// </summary>
    public class InputManager : MonoBehaviour
    {
        [Header("Input Configuration")]
        [SerializeField] private bool enableMouseInput = true;
        [SerializeField] private bool enableKeyboardInput = true;
        [SerializeField] private bool enableTouchInput = true;
        [SerializeField] private bool enableGamepadInput = false;
        
        [Header("Mouse Settings")]
        [SerializeField] private float mouseSensitivity = 2f;
        [SerializeField] private bool invertY = false;
        [SerializeField] private bool lockCursor = true;
        
        [Header("Touch Settings")]
        [SerializeField] private float touchSensitivity = 1f;
        [SerializeField] private float pinchZoomSpeed = 0.1f;
        [SerializeField] private float doubleTapTime = 0.3f;
        
        [Header("Key Bindings")]
        [SerializeField] private KeyCode boostKey = KeyCode.LeftShift;
        [SerializeField] private KeyCode fireKey = KeyCode.Mouse0;
        [SerializeField] private KeyCode targetLockKey = KeyCode.Tab;
        [SerializeField] private KeyCode jarvisKey = KeyCode.J;
        [SerializeField] private KeyCode hudToggleKey = KeyCode.H;
        
        // Input state
        private bool inputEnabled = false;
        private Vector2 lookInput;
        private Vector2 moveInput;
        private bool isBoosting;
        private bool isFiring;
        
        // Touch handling
        private Dictionary<int, Touch> activeTouches = new Dictionary<int, Touch>();
        private float lastTapTime;
        private Vector2 lastTapPosition;
        
        // References
        private IronManExperienceManager experienceManager;
        
        #region Initialization
        
        void Awake()
        {
            experienceManager = IronManExperienceManager.Instance;
            SetupPlatformSpecificInput();
        }
        
        void Start()
        {
            // Configure cursor for WebGL
            #if UNITY_WEBGL && !UNITY_EDITOR
            lockCursor = false; // Don't lock cursor in browser
            Cursor.visible = true;
            #else
            if (lockCursor)
            {
                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible = false;
            }
            #endif
        }
        
        private void SetupPlatformSpecificInput()
        {
            // Detect platform and adjust settings
            #if UNITY_IOS || UNITY_ANDROID
            enableTouchInput = true;
            enableMouseInput = false;
            enableKeyboardInput = false;
            #elif UNITY_WEBGL
            enableTouchInput = Input.touchSupported;
            enableMouseInput = true;
            enableKeyboardInput = true;
            #endif
        }
        
        #endregion
        
        #region Input Handling
        
        void Update()
        {
            if (!inputEnabled) return;
            
            // Handle different input types
            if (enableKeyboardInput)
            {
                HandleKeyboardInput();
            }
            
            if (enableMouseInput)
            {
                HandleMouseInput();
            }
            
            if (enableTouchInput && Input.touchCount > 0)
            {
                HandleTouchInput();
            }
            
            if (enableGamepadInput)
            {
                HandleGamepadInput();
            }
            
            // Process common actions
            ProcessMovement();
            ProcessActions();
        }
        
        private void HandleKeyboardInput()
        {
            // Movement
            moveInput.x = Input.GetAxis("Horizontal");
            moveInput.y = Input.GetAxis("Vertical");
            
            // Boost
            isBoosting = Input.GetKey(boostKey);
            
            // Fire
            if (Input.GetKeyDown(fireKey))
            {
                StartFiring();
            }
            else if (Input.GetKeyUp(fireKey))
            {
                StopFiring();
            }
            
            // Target lock
            if (Input.GetKeyDown(targetLockKey))
            {
                ToggleTargetLock();
            }
            
            // JARVIS
            if (Input.GetKeyDown(jarvisKey))
            {
                ToggleJARVIS();
            }
            
            // HUD toggle
            if (Input.GetKeyDown(hudToggleKey))
            {
                experienceManager.ToggleHUD();
            }
            
            // Number keys for weapon selection
            for (int i = 1; i <= 9; i++)
            {
                if (Input.GetKeyDown(KeyCode.Alpha0 + i))
                {
                    SelectWeapon(i - 1);
                }
            }
            
            // Mission abort (ESC)
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                if (experienceManager.missionSystem.IsMissionActive())
                {
                    experienceManager.missionSystem.AbortCurrentMission();
                }
            }
        }
        
        private void HandleMouseInput()
        {
            // Look input
            if (!Cursor.visible || Input.GetMouseButton(1)) // Right click to look
            {
                lookInput.x = Input.GetAxis("Mouse X") * mouseSensitivity;
                lookInput.y = Input.GetAxis("Mouse Y") * mouseSensitivity * (invertY ? 1 : -1);
            }
            
            // Fire with left click
            if (Input.GetMouseButtonDown(0) && !EventSystem.current.IsPointerOverGameObject())
            {
                StartFiring();
            }
            else if (Input.GetMouseButtonUp(0))
            {
                StopFiring();
            }
            
            // Scroll wheel for zoom/thrust
            float scroll = Input.GetAxis("Mouse ScrollWheel");
            if (scroll != 0)
            {
                AdjustThrust(scroll);
            }
        }
        
        private void HandleTouchInput()
        {
            // Update active touches
            foreach (Touch touch in Input.touches)
            {
                activeTouches[touch.fingerId] = touch;
                
                switch (touch.phase)
                {
                    case TouchPhase.Began:
                        OnTouchBegan(touch);
                        break;
                        
                    case TouchPhase.Moved:
                        OnTouchMoved(touch);
                        break;
                        
                    case TouchPhase.Ended:
                    case TouchPhase.Canceled:
                        OnTouchEnded(touch);
                        activeTouches.Remove(touch.fingerId);
                        break;
                }
            }
            
            // Handle gestures
            if (Input.touchCount == 2)
            {
                HandlePinchGesture();
            }
        }
        
        private void OnTouchBegan(Touch touch)
        {
            // Check for double tap
            if (Time.time - lastTapTime < doubleTapTime && 
                Vector2.Distance(touch.position, lastTapPosition) < 50)
            {
                OnDoubleTap(touch.position);
            }
            
            lastTapTime = Time.time;
            lastTapPosition = touch.position;
            
            // Check if touching UI
            if (EventSystem.current.IsPointerOverGameObject(touch.fingerId))
            {
                return;
            }
            
            // Start firing if in combat mode
            if (experienceManager.currentMode == IronManExperienceManager.ExperienceMode.Combat)
            {
                StartFiring();
            }
        }
        
        private void OnTouchMoved(Touch touch)
        {
            // Use first touch for look controls
            if (touch.fingerId == 0)
            {
                lookInput = touch.deltaPosition * touchSensitivity * Time.deltaTime;
            }
        }
        
        private void OnTouchEnded(Touch touch)
        {
            if (touch.fingerId == 0)
            {
                lookInput = Vector2.zero;
            }
            
            StopFiring();
        }
        
        private void OnDoubleTap(Vector2 position)
        {
            // Double tap to target
            Ray ray = Camera.main.ScreenPointToRay(position);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit, 1000f))
            {
                if (hit.collider.CompareTag("Enemy") || hit.collider.CompareTag("Target"))
                {
                    experienceManager.hudManager.LockTarget(hit.collider.gameObject);
                }
            }
        }
        
        private void HandlePinchGesture()
        {
            Touch touch1 = Input.GetTouch(0);
            Touch touch2 = Input.GetTouch(1);
            
            // Calculate pinch
            Vector2 touch1PrevPos = touch1.position - touch1.deltaPosition;
            Vector2 touch2PrevPos = touch2.position - touch2.deltaPosition;
            
            float prevMagnitude = (touch1PrevPos - touch2PrevPos).magnitude;
            float currentMagnitude = (touch1.position - touch2.position).magnitude;
            
            float difference = currentMagnitude - prevMagnitude;
            
            // Zoom/thrust adjustment
            AdjustThrust(difference * pinchZoomSpeed);
        }
        
        private void HandleGamepadInput()
        {
            // Left stick for movement
            moveInput.x = Input.GetAxis("Gamepad Horizontal");
            moveInput.y = Input.GetAxis("Gamepad Vertical");
            
            // Right stick for look
            lookInput.x = Input.GetAxis("Gamepad Look X") * mouseSensitivity;
            lookInput.y = Input.GetAxis("Gamepad Look Y") * mouseSensitivity * (invertY ? 1 : -1);
            
            // Triggers for fire
            if (Input.GetAxis("Gamepad Fire") > 0.1f)
            {
                StartFiring();
            }
            else
            {
                StopFiring();
            }
            
            // Buttons
            if (Input.GetButtonDown("Gamepad Boost"))
            {
                isBoosting = true;
            }
            else if (Input.GetButtonUp("Gamepad Boost"))
            {
                isBoosting = false;
            }
        }
        
        #endregion
        
        #region Action Processing
        
        private void ProcessMovement()
        {
            // Apply movement to camera or character
            if (Camera.main != null)
            {
                Transform cam = Camera.main.transform;
                
                // Rotation
                cam.Rotate(Vector3.up, lookInput.x);
                cam.Rotate(Vector3.right, -lookInput.y);
                
                // Movement
                Vector3 movement = new Vector3(moveInput.x, 0, moveInput.y);
                movement = cam.TransformDirection(movement);
                
                float speed = isBoosting ? 20f : 10f;
                cam.position += movement * speed * Time.deltaTime;
            }
        }
        
        private void ProcessActions()
        {
            if (isFiring)
            {
                // Continuous firing logic
                if (Time.frameCount % 5 == 0) // Fire every 5 frames
                {
                    experienceManager.hudManager.TriggerRepulsor();
                }
            }
        }
        
        #endregion
        
        #region Input Actions
        
        private void StartFiring()
        {
            isFiring = true;
            experienceManager.hudManager.TriggerRepulsor();
        }
        
        private void StopFiring()
        {
            isFiring = false;
        }
        
        private void ToggleTargetLock()
        {
            if (experienceManager.currentMode == IronManExperienceManager.ExperienceMode.Combat)
            {
                // Find nearest target
                GameObject[] targets = GameObject.FindGameObjectsWithTag("Enemy");
                GameObject nearest = null;
                float nearestDist = float.MaxValue;
                
                foreach (var target in targets)
                {
                    float dist = Vector3.Distance(Camera.main.transform.position, target.transform.position);
                    if (dist < nearestDist)
                    {
                        nearest = target;
                        nearestDist = dist;
                    }
                }
                
                if (nearest != null)
                {
                    experienceManager.hudManager.LockTarget(nearest);
                }
            }
        }
        
        private void ToggleJARVIS()
        {
            // Simple voice command simulation
            string[] sampleCommands = {
                "status",
                "scan area",
                "weapons online",
                "power boost",
                "analysis"
            };
            
            string command = sampleCommands[Random.Range(0, sampleCommands.Length)];
            experienceManager.ProcessVoiceCommand(command);
        }
        
        private void SelectWeapon(int index)
        {
            experienceManager.hudManager.ShowAlert($"Weapon {index + 1} selected", AlertLevel.Info);
        }
        
        private void AdjustThrust(float delta)
        {
            // Adjust flight speed/altitude
            if (Camera.main != null)
            {
                Camera.main.transform.position += Vector3.up * delta * 10f;
            }
        }
        
        #endregion
        
        #region Public Interface
        
        public void EnableInput()
        {
            inputEnabled = true;
        }
        
        public void DisableInput()
        {
            inputEnabled = false;
            lookInput = Vector2.zero;
            moveInput = Vector2.zero;
            isFiring = false;
            isBoosting = false;
        }
        
        public Vector2 GetLookInput()
        {
            return lookInput;
        }
        
        public Vector2 GetMoveInput()
        {
            return moveInput;
        }
        
        public bool IsBoosting()
        {
            return isBoosting;
        }
        
        public bool IsFiring()
        {
            return isFiring;
        }
        
        #endregion
        
        #region WebGL Specific
        
        /// <summary>
        /// Called from JavaScript for browser input
        /// </summary>
        public void OnBrowserKeyPress(string key)
        {
            switch (key.ToLower())
            {
                case "space":
                    isBoosting = !isBoosting;
                    break;
                case "f":
                    ToggleFullscreen();
                    break;
            }
        }
        
        private void ToggleFullscreen()
        {
            #if UNITY_WEBGL && !UNITY_EDITOR
            Screen.fullScreen = !Screen.fullScreen;
            #endif
        }
        
        #endregion
    }
}