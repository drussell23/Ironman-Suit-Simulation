using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using System.Linq;

namespace IronManSim.Frontend
{
    /// <summary>
    /// Manages all HUD elements for the Iron Man suit interface
    /// </summary>
    public class HUDManager : MonoBehaviour
    {
        [Header("HUD Modes")]
        [SerializeField] private HUDMode currentMode = HUDMode.Standard;
        [SerializeField] private float modeTransitionDuration = 0.5f;
        
        public enum HUDMode
        {
            Off,
            Standard,
            Combat,
            Mission,
            Analysis,
            Emergency,
            Stealth
        }
        
        [Header("Core HUD Elements")]
        [SerializeField] private GameObject hudCanvas;
        [SerializeField] private CanvasGroup hudCanvasGroup;
        [SerializeField] private float hudOpacity = 0.8f;
        
        [Header("Status Displays")]
        [SerializeField] private Slider powerLevelSlider;
        [SerializeField] private TextMeshProUGUI powerLevelText;
        [SerializeField] private Slider armorIntegritySlider;
        [SerializeField] private TextMeshProUGUI armorIntegrityText;
        [SerializeField] private TextMeshProUGUI velocityText;
        [SerializeField] private TextMeshProUGUI altitudeText;
        [SerializeField] private TextMeshProUGUI temperatureText;
        
        [Header("Compass & Navigation")]
        [SerializeField] private RectTransform compassBar;
        [SerializeField] private TextMeshProUGUI headingText;
        [SerializeField] private GameObject waypointPrefab;
        [SerializeField] private Transform waypointContainer;
        private List<GameObject> activeWaypoints = new List<GameObject>();
        
        [Header("Targeting System")]
        [SerializeField] private GameObject targetingReticle;
        [SerializeField] private GameObject targetLockIndicator;
        [SerializeField] private GameObject threatIndicatorPrefab;
        [SerializeField] private Transform threatContainer;
        [SerializeField] private int maxTargets = 10;
        private List<TargetInfo> activeTargets = new List<TargetInfo>();
        
        [Header("Weapon Systems")]
        [SerializeField] private GameObject weaponPanel;
        [SerializeField] private Image repulsorChargeLeft;
        [SerializeField] private Image repulsorChargeRight;
        [SerializeField] private TextMeshProUGUI missileCountText;
        [SerializeField] private GameObject weaponSelectWheel;
        
        [Header("Alert System")]
        [SerializeField] private GameObject alertPanel;
        [SerializeField] private TextMeshProUGUI alertText;
        [SerializeField] private Image alertBackground;
        [SerializeField] private float alertDisplayDuration = 3f;
        private Queue<AlertMessage> alertQueue = new Queue<AlertMessage>();
        
        [Header("Analysis Mode")]
        [SerializeField] private GameObject analysisOverlay;
        [SerializeField] private GameObject scanLineEffect;
        [SerializeField] private TextMeshProUGUI scanResultText;
        [SerializeField] private GameObject dataPointPrefab;
        
        [Header("Boot Sequence")]
        [SerializeField] private GameObject bootSequencePanel;
        [SerializeField] private Slider bootProgressBar;
        [SerializeField] private TextMeshProUGUI bootStatusText;
        [SerializeField] private ScrollRect bootLogScroll;
        [SerializeField] private TextMeshProUGUI bootLogText;
        
        [Header("Effects")]
        [SerializeField] private Material hologramMaterial;
        [SerializeField] private ParticleSystem hudParticles;
        [SerializeField] private AnimationCurve glitchCurve;
        [SerializeField] private float glitchIntensity = 0.1f;
        
        [Header("Colors")]
        [SerializeField] private Color normalColor = new Color(0.2f, 0.8f, 1f);
        [SerializeField] private Color warningColor = Color.yellow;
        [SerializeField] private Color criticalColor = Color.red;
        [SerializeField] private Color emergencyColor = new Color(1f, 0.2f, 0f);
        
        // Private variables
        private bool isInitialized = false;
        private Coroutine alertCoroutine;
        private float currentHeading = 0f;
        private bool targetingEnabled = false;
        
        public class TargetInfo
        {
            public GameObject targetObject;
            public GameObject hudIndicator;
            public float distance;
            public float threatLevel;
            public bool isLocked;
        }
        
        public class AlertMessage
        {
            public string message;
            public AlertLevel level;
            public float duration;
        }
        
        #region Initialization
        
        void Awake()
        {
            SetupHUDComponents();
        }
        
        void Start()
        {
            if (hudCanvasGroup != null)
            {
                hudCanvasGroup.alpha = 0f;
            }
        }
        
        private void SetupHUDComponents()
        {
            if (hudCanvas == null)
            {
                CreateHUDStructure();
            }
            
            // Start background processes
            StartCoroutine(UpdateCompass());
            StartCoroutine(ProcessAlertQueue());
            
            isInitialized = true;
        }
        
        private void CreateHUDStructure()
        {
            // Create main HUD canvas
            GameObject hudObj = new GameObject("HUD Canvas");
            hudCanvas = hudObj;
            
            Canvas canvas = hudObj.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 10;
            
            hudObj.AddComponent<CanvasScaler>();
            hudObj.AddComponent<GraphicRaycaster>();
            hudCanvasGroup = hudObj.AddComponent<CanvasGroup>();
            
            // Create HUD structure
            CreateStatusPanel();
            CreateCompass();
            CreateTargetingSystem();
            CreateWeaponPanel();
            CreateAlertSystem();
            CreateAnalysisOverlay();
        }
        
        #endregion
        
        #region HUD Activation
        
        public void ActivateHUD()
        {
            StartCoroutine(HUDBootSequence());
        }
        
        private IEnumerator HUDBootSequence()
        {
            // Fade in HUD
            float elapsed = 0f;
            while (elapsed < 1f)
            {
                elapsed += Time.deltaTime;
                hudCanvasGroup.alpha = Mathf.Lerp(0f, hudOpacity, elapsed);
                yield return null;
            }
            
            // Activate default elements
            SetHUDMode(HUDMode.Standard);
        }
        
        public void SetHUDMode(HUDMode mode)
        {
            if (currentMode == mode) return;
            
            StartCoroutine(TransitionToMode(mode));
        }
        
        private IEnumerator TransitionToMode(HUDMode newMode)
        {
            // Fade out current mode elements
            yield return StartCoroutine(FadeHUDElements(false));
            
            currentMode = newMode;
            
            // Configure elements for new mode
            ConfigureHUDForMode(newMode);
            
            // Fade in new mode elements
            yield return StartCoroutine(FadeHUDElements(true));
        }
        
        private void ConfigureHUDForMode(HUDMode mode)
        {
            // Reset all elements
            weaponPanel.SetActive(false);
            analysisOverlay.SetActive(false);
            targetingReticle.SetActive(false);
            
            switch (mode)
            {
                case HUDMode.Standard:
                    // Basic HUD elements only
                    break;
                    
                case HUDMode.Combat:
                    weaponPanel.SetActive(true);
                    targetingReticle.SetActive(true);
                    targetingEnabled = true;
                    break;
                    
                case HUDMode.Mission:
                    // Show waypoints and objectives
                    break;
                    
                case HUDMode.Analysis:
                    analysisOverlay.SetActive(true);
                    StartCoroutine(RunAnalysisScan());
                    break;
                    
                case HUDMode.Emergency:
                    // Flash critical elements
                    StartCoroutine(EmergencyMode());
                    break;
                    
                case HUDMode.Stealth:
                    // Minimal HUD
                    hudCanvasGroup.alpha = 0.3f;
                    break;
            }
        }
        
        #endregion
        
        #region Status Updates
        
        public void UpdateSuitStatus(SuitStatus status)
        {
            if (!isInitialized) return;
            
            // Update power level
            if (powerLevelSlider != null)
            {
                powerLevelSlider.value = status.powerLevel / 100f;
                powerLevelText.text = $"{status.powerLevel:F0}%";
                
                // Color based on level
                Color powerColor = status.powerLevel > 50 ? normalColor :
                                 status.powerLevel > 20 ? warningColor : criticalColor;
                powerLevelSlider.fillRect.GetComponent<Image>().color = powerColor;
            }
            
            // Update armor integrity
            if (armorIntegritySlider != null)
            {
                armorIntegritySlider.value = status.armorIntegrity / 100f;
                armorIntegrityText.text = $"{status.armorIntegrity:F0}%";
                
                Color armorColor = status.armorIntegrity > 70 ? normalColor :
                                 status.armorIntegrity > 30 ? warningColor : criticalColor;
                armorIntegritySlider.fillRect.GetComponent<Image>().color = armorColor;
            }
            
            // Update velocity and altitude
            if (velocityText != null)
            {
                float speedKmh = status.velocity.magnitude * 3.6f;
                velocityText.text = $"{speedKmh:F0} km/h";
            }
            
            if (altitudeText != null)
            {
                altitudeText.text = $"{status.altitude:F0} m";
            }
            
            if (temperatureText != null)
            {
                temperatureText.text = $"{status.coreTemperature:F1}°C";
                Color tempColor = Mathf.Abs(status.coreTemperature - 36.5f) < 2f ? normalColor : warningColor;
                temperatureText.color = tempColor;
            }
        }
        
        #endregion
        
        #region Compass & Navigation
        
        private IEnumerator UpdateCompass()
        {
            while (true)
            {
                if (compassBar != null && Camera.main != null)
                {
                    // Update heading
                    currentHeading = Camera.main.transform.eulerAngles.y;
                    headingText.text = $"{currentHeading:F0}°";
                    
                    // Rotate compass bar
                    compassBar.localRotation = Quaternion.Euler(0, 0, currentHeading);
                }
                
                yield return new WaitForSeconds(0.1f);
            }
        }
        
        public void AddWaypoint(Vector3 worldPosition, string label)
        {
            if (waypointPrefab == null || waypointContainer == null) return;
            
            GameObject waypoint = Instantiate(waypointPrefab, waypointContainer);
            waypoint.GetComponentInChildren<TextMeshProUGUI>().text = label;
            
            // Position waypoint on HUD based on world position
            StartCoroutine(UpdateWaypointPosition(waypoint, worldPosition));
            activeWaypoints.Add(waypoint);
        }
        
        private IEnumerator UpdateWaypointPosition(GameObject waypoint, Vector3 worldPos)
        {
            while (waypoint != null)
            {
                Vector3 screenPos = Camera.main.WorldToScreenPoint(worldPos);
                if (screenPos.z > 0)
                {
                    waypoint.SetActive(true);
                    waypoint.transform.position = screenPos;
                }
                else
                {
                    waypoint.SetActive(false);
                }
                
                yield return null;
            }
        }
        
        #endregion
        
        #region Targeting System
        
        public void EnableTargeting()
        {
            targetingEnabled = true;
            targetingReticle.SetActive(true);
            StartCoroutine(TargetingScan());
        }
        
        public void DisableTargeting()
        {
            targetingEnabled = false;
            targetingReticle.SetActive(false);
            ClearTargets();
        }
        
        private IEnumerator TargetingScan()
        {
            while (targetingEnabled)
            {
                // Scan for targets
                Collider[] potentialTargets = Physics.OverlapSphere(
                    Camera.main.transform.position, 
                    500f, 
                    LayerMask.GetMask("Enemy", "Target")
                );
                
                // Update target list
                UpdateTargetList(potentialTargets);
                
                // Update target indicators
                UpdateTargetIndicators();
                
                yield return new WaitForSeconds(0.1f);
            }
        }
        
        private void UpdateTargetList(Collider[] potentialTargets)
        {
            // Remove invalid targets
            activeTargets.RemoveAll(t => t.targetObject == null);
            
            // Add new targets
            foreach (var collider in potentialTargets)
            {
                if (activeTargets.Count >= maxTargets) break;
                
                if (!activeTargets.Any(t => t.targetObject == collider.gameObject))
                {
                    AddTarget(collider.gameObject);
                }
            }
        }
        
        private void AddTarget(GameObject target)
        {
            if (threatIndicatorPrefab == null) return;
            
            GameObject indicator = Instantiate(threatIndicatorPrefab, threatContainer);
            
            TargetInfo targetInfo = new TargetInfo
            {
                targetObject = target,
                hudIndicator = indicator,
                distance = Vector3.Distance(Camera.main.transform.position, target.transform.position),
                threatLevel = CalculateThreatLevel(target),
                isLocked = false
            };
            
            activeTargets.Add(targetInfo);
        }
        
        private float CalculateThreatLevel(GameObject target)
        {
            // Simple threat calculation
            float distance = Vector3.Distance(Camera.main.transform.position, target.transform.position);
            float threat = 1f - (distance / 500f);
            
            // Check if target has weapons
            if (target.GetComponent<WeaponSystem>() != null)
            {
                threat *= 2f;
            }
            
            return Mathf.Clamp01(threat);
        }
        
        private void UpdateTargetIndicators()
        {
            foreach (var target in activeTargets)
            {
                if (target.targetObject == null || target.hudIndicator == null) continue;
                
                // Update position
                Vector3 screenPos = Camera.main.WorldToScreenPoint(target.targetObject.transform.position);
                if (screenPos.z > 0)
                {
                    target.hudIndicator.SetActive(true);
                    target.hudIndicator.transform.position = screenPos;
                    
                    // Update distance
                    target.distance = Vector3.Distance(
                        Camera.main.transform.position, 
                        target.targetObject.transform.position
                    );
                    
                    // Update UI
                    var texts = target.hudIndicator.GetComponentsInChildren<TextMeshProUGUI>();
                    if (texts.Length > 0)
                    {
                        texts[0].text = $"{target.distance:F0}m";
                    }
                    
                    // Update color based on threat
                    Image indicator = target.hudIndicator.GetComponent<Image>();
                    if (indicator != null)
                    {
                        indicator.color = Color.Lerp(normalColor, criticalColor, target.threatLevel);
                    }
                }
                else
                {
                    target.hudIndicator.SetActive(false);
                }
            }
        }
        
        public void LockTarget(GameObject target)
        {
            var targetInfo = activeTargets.FirstOrDefault(t => t.targetObject == target);
            if (targetInfo != null)
            {
                targetInfo.isLocked = true;
                
                // Show lock indicator
                if (targetLockIndicator != null)
                {
                    targetLockIndicator.SetActive(true);
                    targetLockIndicator.transform.position = targetInfo.hudIndicator.transform.position;
                }
                
                ShowAlert("Target Locked", AlertLevel.Info);
            }
        }
        
        private void ClearTargets()
        {
            foreach (var target in activeTargets)
            {
                if (target.hudIndicator != null)
                {
                    Destroy(target.hudIndicator);
                }
            }
            activeTargets.Clear();
        }
        
        #endregion
        
        #region Weapon Systems
        
        public void TriggerRepulsor(bool isLeft = true)
        {
            StartCoroutine(RepulsorFire(isLeft));
        }
        
        private IEnumerator RepulsorFire(bool isLeft)
        {
            Image repulsor = isLeft ? repulsorChargeLeft : repulsorChargeRight;
            if (repulsor == null) yield break;
            
            // Charge effect
            float chargeTime = 0.5f;
            float elapsed = 0f;
            
            while (elapsed < chargeTime)
            {
                elapsed += Time.deltaTime;
                float charge = elapsed / chargeTime;
                repulsor.fillAmount = charge;
                repulsor.color = Color.Lerp(normalColor, Color.white, charge);
                yield return null;
            }
            
            // Fire effect
            repulsor.color = Color.white;
            yield return new WaitForSeconds(0.1f);
            
            // Cooldown
            elapsed = 0f;
            while (elapsed < 1f)
            {
                elapsed += Time.deltaTime;
                repulsor.fillAmount = elapsed;
                repulsor.color = Color.Lerp(criticalColor, normalColor, elapsed);
                yield return null;
            }
        }
        
        public void UpdateMissileCount(int count)
        {
            if (missileCountText != null)
            {
                missileCountText.text = $"Missiles: {count}";
                missileCountText.color = count > 0 ? normalColor : criticalColor;
            }
        }
        
        public void ShowWeaponWheel()
        {
            if (weaponSelectWheel != null)
            {
                weaponSelectWheel.SetActive(true);
                Time.timeScale = 0.3f; // Slow motion during selection
            }
        }
        
        public void HideWeaponWheel()
        {
            if (weaponSelectWheel != null)
            {
                weaponSelectWheel.SetActive(false);
                Time.timeScale = 1f;
            }
        }
        
        #endregion
        
        #region Alert System
        
        public void ShowAlert(string message, AlertLevel level)
        {
            AlertMessage alert = new AlertMessage
            {
                message = message,
                level = level,
                duration = alertDisplayDuration
            };
            
            alertQueue.Enqueue(alert);
        }
        
        private IEnumerator ProcessAlertQueue()
        {
            while (true)
            {
                if (alertQueue.Count > 0 && alertPanel != null)
                {
                    AlertMessage alert = alertQueue.Dequeue();
                    yield return StartCoroutine(DisplayAlert(alert));
                }
                
                yield return new WaitForSeconds(0.1f);
            }
        }
        
        private IEnumerator DisplayAlert(AlertMessage alert)
        {
            // Set alert appearance
            alertText.text = alert.message;
            
            Color alertColor = normalColor;
            switch (alert.level)
            {
                case AlertLevel.Warning:
                    alertColor = warningColor;
                    break;
                case AlertLevel.Critical:
                    alertColor = criticalColor;
                    break;
                case AlertLevel.Emergency:
                    alertColor = emergencyColor;
                    break;
            }
            
            alertBackground.color = new Color(alertColor.r, alertColor.g, alertColor.b, 0.8f);
            alertText.color = Color.white;
            
            // Animate in
            alertPanel.SetActive(true);
            CanvasGroup alertGroup = alertPanel.GetComponent<CanvasGroup>();
            if (alertGroup == null)
            {
                alertGroup = alertPanel.AddComponent<CanvasGroup>();
            }
            
            float elapsed = 0f;
            while (elapsed < 0.3f)
            {
                elapsed += Time.deltaTime;
                alertGroup.alpha = Mathf.Lerp(0f, 1f, elapsed / 0.3f);
                yield return null;
            }
            
            // Hold
            yield return new WaitForSeconds(alert.duration);
            
            // Animate out
            elapsed = 0f;
            while (elapsed < 0.3f)
            {
                elapsed += Time.deltaTime;
                alertGroup.alpha = Mathf.Lerp(1f, 0f, elapsed / 0.3f);
                yield return null;
            }
            
            alertPanel.SetActive(false);
        }
        
        public void ShowEmergencyAlerts()
        {
            ShowAlert("CRITICAL SYSTEM FAILURE", AlertLevel.Emergency);
            ShowAlert("IMMEDIATE ACTION REQUIRED", AlertLevel.Emergency);
            
            // Flash HUD elements
            StartCoroutine(EmergencyMode());
        }
        
        private IEnumerator EmergencyMode()
        {
            float flashDuration = 10f;
            float elapsed = 0f;
            
            while (elapsed < flashDuration && currentMode == HUDMode.Emergency)
            {
                elapsed += Time.deltaTime;
                
                // Flash HUD opacity
                float flash = Mathf.Sin(elapsed * 10f) * 0.5f + 0.5f;
                hudCanvasGroup.alpha = Mathf.Lerp(0.5f, 1f, flash);
                
                // Tint HUD red
                foreach (var graphic in hudCanvas.GetComponentsInChildren<Graphic>())
                {
                    graphic.color = Color.Lerp(graphic.color, emergencyColor, Time.deltaTime * 2f);
                }
                
                yield return null;
            }
        }
        
        #endregion
        
        #region Analysis Mode
        
        public void ShowAnalysisOverlay()
        {
            if (analysisOverlay != null)
            {
                analysisOverlay.SetActive(true);
                StartCoroutine(RunAnalysisScan());
            }
        }
        
        private IEnumerator RunAnalysisScan()
        {
            if (scanLineEffect == null) yield break;
            
            // Animate scan line
            RectTransform scanRect = scanLineEffect.GetComponent<RectTransform>();
            float scanDuration = 2f;
            float elapsed = 0f;
            
            while (elapsed < scanDuration)
            {
                elapsed += Time.deltaTime;
                float progress = elapsed / scanDuration;
                
                // Move scan line
                scanRect.anchorMin = new Vector2(0, 1f - progress);
                scanRect.anchorMax = new Vector2(1, 1f - progress + 0.05f);
                
                yield return null;
            }
            
            // Show results
            DisplayScanResults();
        }
        
        private void DisplayScanResults()
        {
            if (scanResultText != null)
            {
                scanResultText.text = "SCAN COMPLETE\n" +
                                    "Threats: 0\n" +
                                    "Structural Integrity: 100%\n" +
                                    "Environmental Hazards: None\n" +
                                    "Recommended Action: Continue";
            }
            
            // Add data points
            for (int i = 0; i < 5; i++)
            {
                Vector2 randomPos = new Vector2(
                    Random.Range(100f, Screen.width - 100f),
                    Random.Range(100f, Screen.height - 100f)
                );
                
                CreateDataPoint(randomPos, $"Data Point {i + 1}");
            }
        }
        
        private void CreateDataPoint(Vector2 screenPos, string label)
        {
            if (dataPointPrefab == null) return;
            
            GameObject dataPoint = Instantiate(dataPointPrefab, analysisOverlay.transform);
            dataPoint.transform.position = screenPos;
            
            TextMeshProUGUI text = dataPoint.GetComponentInChildren<TextMeshProUGUI>();
            if (text != null)
            {
                text.text = label;
            }
            
            // Animate data point
            StartCoroutine(AnimateDataPoint(dataPoint));
        }
        
        private IEnumerator AnimateDataPoint(GameObject dataPoint)
        {
            CanvasGroup group = dataPoint.GetComponent<CanvasGroup>();
            if (group == null)
            {
                group = dataPoint.AddComponent<CanvasGroup>();
            }
            
            // Fade in
            float elapsed = 0f;
            while (elapsed < 0.5f)
            {
                elapsed += Time.deltaTime;
                group.alpha = Mathf.Lerp(0f, 1f, elapsed / 0.5f);
                dataPoint.transform.localScale = Vector3.Lerp(Vector3.zero, Vector3.one, elapsed / 0.5f);
                yield return null;
            }
            
            // Pulse
            while (dataPoint != null)
            {
                float pulse = Mathf.Sin(Time.time * 2f) * 0.1f + 0.9f;
                dataPoint.transform.localScale = Vector3.one * pulse;
                yield return null;
            }
        }
        
        #endregion
        
        #region Boot Sequence
        
        public void ShowBootSequence()
        {
            if (bootSequencePanel != null)
            {
                bootSequencePanel.SetActive(true);
                bootLogText.text = "";
            }
        }
        
        public void UpdateBootProgress(float progress)
        {
            if (bootProgressBar != null)
            {
                bootProgressBar.value = progress;
                bootStatusText.text = $"INITIALIZING... {progress * 100:F0}%";
            }
        }
        
        public void AddBootMessage(string message)
        {
            if (bootLogText != null)
            {
                bootLogText.text += $"> {message}\n";
                
                // Scroll to bottom
                Canvas.ForceUpdateCanvases();
                bootLogScroll.verticalNormalizedPosition = 0f;
            }
        }
        
        #endregion
        
        #region Effects
        
        private IEnumerator FadeHUDElements(bool fadeIn)
        {
            float start = fadeIn ? 0f : 1f;
            float end = fadeIn ? 1f : 0f;
            float elapsed = 0f;
            
            while (elapsed < modeTransitionDuration)
            {
                elapsed += Time.deltaTime;
                float t = elapsed / modeTransitionDuration;
                
                // Apply fade to all HUD elements
                foreach (var graphic in hudCanvas.GetComponentsInChildren<Graphic>())
                {
                    Color color = graphic.color;
                    color.a = Mathf.Lerp(start, end, t);
                    graphic.color = color;
                }
                
                yield return null;
            }
        }
        
        public void ApplyGlitchEffect(float intensity = 1f)
        {
            StartCoroutine(GlitchEffect(intensity));
        }
        
        private IEnumerator GlitchEffect(float intensity)
        {
            float duration = 0.5f;
            float elapsed = 0f;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float glitch = glitchCurve.Evaluate(elapsed / duration) * intensity;
                
                // Apply random offset to HUD elements
                foreach (var rect in hudCanvas.GetComponentsInChildren<RectTransform>())
                {
                    if (rect != hudCanvas.GetComponent<RectTransform>())
                    {
                        rect.anchoredPosition += new Vector2(
                            Random.Range(-glitch, glitch) * 10f,
                            Random.Range(-glitch, glitch) * 10f
                        );
                    }
                }
                
                // Flicker opacity
                hudCanvasGroup.alpha = hudOpacity + Random.Range(-glitch, glitch) * 0.5f;
                
                yield return new WaitForSeconds(0.02f);
                
                // Reset positions
                foreach (var rect in hudCanvas.GetComponentsInChildren<RectTransform>())
                {
                    if (rect != hudCanvas.GetComponent<RectTransform>())
                    {
                        rect.anchoredPosition = Vector2.zero;
                    }
                }
            }
            
            hudCanvasGroup.alpha = hudOpacity;
        }
        
        #endregion
        
        #region Public Interface
        
        public void ToggleVisibility()
        {
            if (currentMode == HUDMode.Off)
            {
                SetHUDMode(HUDMode.Standard);
            }
            else
            {
                SetHUDMode(HUDMode.Off);
            }
        }
        
        public void SetOpacity(float opacity)
        {
            hudOpacity = Mathf.Clamp01(opacity);
            if (hudCanvasGroup != null)
            {
                hudCanvasGroup.alpha = hudOpacity;
            }
        }
        
        public void CycleHUDMode(int direction)
        {
            int currentIndex = (int)currentMode;
            int modeCount = System.Enum.GetValues(typeof(HUDMode)).Length;
            
            currentIndex += direction;
            if (currentIndex < 0) currentIndex = modeCount - 1;
            if (currentIndex >= modeCount) currentIndex = 0;
            
            SetHUDMode((HUDMode)currentIndex);
        }
        
        public void ShowHandInterface()
        {
            // Show palm-based interface
            ShowAlert("Hand Interface Activated", AlertLevel.Info);
        }
        
        #endregion
        
        #region Component Creation (Editor Support)
        
        private void CreateStatusPanel()
        {
            GameObject panel = new GameObject("Status Panel");
            panel.transform.SetParent(hudCanvas.transform);
            
            RectTransform rect = panel.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0, 0.8f);
            rect.anchorMax = new Vector2(0.2f, 1f);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            
            // Add background
            Image bg = panel.AddComponent<Image>();
            bg.color = new Color(0, 0, 0, 0.5f);
        }
        
        private void CreateCompass()
        {
            GameObject compass = new GameObject("Compass");
            compass.transform.SetParent(hudCanvas.transform);
            
            RectTransform rect = compass.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.4f, 0.9f);
            rect.anchorMax = new Vector2(0.6f, 0.95f);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
        }
        
        private void CreateTargetingSystem()
        {
            GameObject targeting = new GameObject("Targeting System");
            targeting.transform.SetParent(hudCanvas.transform);
            
            targetingReticle = new GameObject("Targeting Reticle");
            targetingReticle.transform.SetParent(targeting.transform);
            
            Image reticle = targetingReticle.AddComponent<Image>();
            reticle.color = normalColor;
            
            targetingReticle.SetActive(false);
        }
        
        private void CreateWeaponPanel()
        {
            weaponPanel = new GameObject("Weapon Panel");
            weaponPanel.transform.SetParent(hudCanvas.transform);
            
            RectTransform rect = weaponPanel.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.8f, 0.7f);
            rect.anchorMax = new Vector2(1f, 0.9f);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            
            weaponPanel.SetActive(false);
        }
        
        private void CreateAlertSystem()
        {
            alertPanel = new GameObject("Alert Panel");
            alertPanel.transform.SetParent(hudCanvas.transform);
            
            RectTransform rect = alertPanel.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.3f, 0.7f);
            rect.anchorMax = new Vector2(0.7f, 0.8f);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            
            alertBackground = alertPanel.AddComponent<Image>();
            alertBackground.color = new Color(1, 0, 0, 0.8f);
            
            GameObject textObj = new GameObject("Alert Text");
            textObj.transform.SetParent(alertPanel.transform);
            alertText = textObj.AddComponent<TextMeshProUGUI>();
            alertText.text = "";
            alertText.alignment = TextAlignmentOptions.Center;
            
            alertPanel.SetActive(false);
        }
        
        private void CreateAnalysisOverlay()
        {
            analysisOverlay = new GameObject("Analysis Overlay");
            analysisOverlay.transform.SetParent(hudCanvas.transform);
            
            RectTransform rect = analysisOverlay.AddComponent<RectTransform>();
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.one;
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            
            analysisOverlay.SetActive(false);
        }
        
        #endregion
    }
    
    // Support class
    public class WeaponSystem : MonoBehaviour
    {
        // Placeholder for weapon system detection
    }
}