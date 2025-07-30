using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;

namespace IronManSim.UI.Animations
{
    /// <summary>
    /// Manages alert and warning animations for the Iron Man HUD
    /// </summary>
    public class AlertAnimations : MonoBehaviour
    {
        [System.Serializable]
        public class AlertConfig
        {
            public AlertType type;
            public Color primaryColor = Color.red;
            public Color secondaryColor = Color.yellow;
            public float pulseSpeed = 2f;
            public float pulseIntensity = 0.5f;
            public AudioClip alertSound;
            public string defaultMessage = "ALERT";
        }
        
        public enum AlertType
        {
            Critical,
            Warning,
            Caution,
            SystemFailure,
            LowPower,
            Damage,
            Missile,
            Proximity,
            TargetLock
        }
        
        public enum AlertAnimation
        {
            Pulse,
            Flash,
            EdgeGlow,
            ScreenShake,
            RadialWipe,
            DigitalGlitch,
            HologramDistort
        }
        
        [Header("Alert Display")]
        [SerializeField] private GameObject alertPanelPrefab;
        [SerializeField] private Transform alertContainer;
        [SerializeField] private int maxSimultaneousAlerts = 5;
        
        [Header("Screen Effects")]
        [SerializeField] private Image screenOverlay;
        [SerializeField] private Image edgeVignette;
        [SerializeField] private Material glitchMaterial;
        [SerializeField] private float screenShakeIntensity = 10f;
        
        [Header("Alert Configurations")]
        [SerializeField] private List<AlertConfig> alertConfigs = new List<AlertConfig>();
        
        [Header("Audio")]
        [SerializeField] private AudioSource alertAudioSource;
        [SerializeField] private AudioClip defaultAlertSound;
        
        private Dictionary<AlertType, AlertConfig> alertLookup = new Dictionary<AlertType, AlertConfig>();
        private Queue<GameObject> alertPool = new Queue<GameObject>();
        private List<ActiveAlert> activeAlerts = new List<ActiveAlert>();
        private UIAnimationController animController;
        private Coroutine screenShakeCoroutine;
        
        private class ActiveAlert
        {
            public GameObject alertObject;
            public AlertType type;
            public Coroutine animationCoroutine;
            public float startTime;
        }
        
        void Start()
        {
            animController = UIAnimationController.Instance;
            
            // Build alert lookup
            foreach (var config in alertConfigs)
            {
                alertLookup[config.type] = config;
            }
            
            // Initialize alert pool
            InitializeAlertPool();
            
            // Setup screen effects
            if (screenOverlay != null)
            {
                screenOverlay.color = new Color(1, 1, 1, 0);
            }
            
            if (edgeVignette != null)
            {
                edgeVignette.color = new Color(1, 0, 0, 0);
            }
        }
        
        #region Public Methods
        
        /// <summary>
        /// Show an alert with specified animation
        /// </summary>
        public void ShowAlert(AlertType type, string message = null, float duration = 3f, AlertAnimation animation = AlertAnimation.Pulse)
        {
            StartCoroutine(ShowAlertCoroutine(type, message, duration, animation));
        }
        
        /// <summary>
        /// Show critical system alert with multiple effects
        /// </summary>
        public void ShowCriticalAlert(string message, float duration = 5f)
        {
            ShowAlert(AlertType.Critical, message, duration, AlertAnimation.Pulse);
            TriggerScreenEffect(ScreenEffect.EdgeGlow, Color.red, duration);
            TriggerScreenShake(duration * 0.5f, screenShakeIntensity);
        }
        
        /// <summary>
        /// Clear all active alerts
        /// </summary>
        public void ClearAllAlerts()
        {
            foreach (var alert in activeAlerts)
            {
                if (alert.animationCoroutine != null)
                {
                    StopCoroutine(alert.animationCoroutine);
                }
                ReturnAlertToPool(alert.alertObject);
            }
            activeAlerts.Clear();
        }
        
        /// <summary>
        /// Clear specific alert type
        /// </summary>
        public void ClearAlert(AlertType type)
        {
            var alertsToRemove = activeAlerts.FindAll(a => a.type == type);
            foreach (var alert in alertsToRemove)
            {
                if (alert.animationCoroutine != null)
                {
                    StopCoroutine(alert.animationCoroutine);
                }
                ReturnAlertToPool(alert.alertObject);
                activeAlerts.Remove(alert);
            }
        }
        
        #endregion
        
        #region Alert Display
        
        private void InitializeAlertPool()
        {
            if (alertPanelPrefab == null) return;
            
            for (int i = 0; i < maxSimultaneousAlerts; i++)
            {
                GameObject alert = Instantiate(alertPanelPrefab, alertContainer);
                alert.SetActive(false);
                alertPool.Enqueue(alert);
            }
        }
        
        private IEnumerator ShowAlertCoroutine(AlertType type, string message, float duration, AlertAnimation animation)
        {
            // Get alert configuration
            AlertConfig config = alertLookup.ContainsKey(type) ? 
                alertLookup[type] : 
                new AlertConfig { type = type, primaryColor = Color.red };
            
            // Play alert sound
            PlayAlertSound(config.alertSound ?? defaultAlertSound);
            
            // Get alert panel from pool
            GameObject alertPanel = GetAlertFromPool();
            if (alertPanel == null) yield break;
            
            // Setup alert panel
            SetupAlertPanel(alertPanel, config, message ?? config.defaultMessage);
            
            // Create active alert entry
            ActiveAlert activeAlert = new ActiveAlert
            {
                alertObject = alertPanel,
                type = type,
                startTime = Time.time
            };
            activeAlerts.Add(activeAlert);
            
            // Start animation
            activeAlert.animationCoroutine = StartCoroutine(
                AnimateAlert(alertPanel, animation, config, duration));
            
            // Wait for duration
            yield return new WaitForSeconds(duration);
            
            // Fade out and return to pool
            CanvasGroup canvasGroup = alertPanel.GetComponent<CanvasGroup>();
            if (canvasGroup != null)
            {
                animController.FadeOut(canvasGroup, 0.3f, () =>
                {
                    ReturnAlertToPool(alertPanel);
                    activeAlerts.Remove(activeAlert);
                });
            }
        }
        
        private void SetupAlertPanel(GameObject panel, AlertConfig config, string message)
        {
            panel.SetActive(true);
            
            // Set alert text
            TextMeshProUGUI alertText = panel.GetComponentInChildren<TextMeshProUGUI>();
            if (alertText != null)
            {
                alertText.text = message;
                alertText.color = config.primaryColor;
            }
            
            // Set alert icon/border colors
            Image[] images = panel.GetComponentsInChildren<Image>();
            foreach (var image in images)
            {
                if (image.gameObject.name.Contains("Border") || 
                    image.gameObject.name.Contains("Icon"))
                {
                    image.color = config.primaryColor;
                }
            }
            
            // Position alert (stack them)
            RectTransform rect = panel.GetComponent<RectTransform>();
            int alertIndex = activeAlerts.Count;
            rect.anchoredPosition = new Vector2(0, -alertIndex * 80);
        }
        
        private GameObject GetAlertFromPool()
        {
            if (alertPool.Count > 0)
            {
                return alertPool.Dequeue();
            }
            
            // If no alerts available, remove oldest
            if (activeAlerts.Count > 0)
            {
                var oldest = activeAlerts[0];
                if (oldest.animationCoroutine != null)
                {
                    StopCoroutine(oldest.animationCoroutine);
                }
                activeAlerts.RemoveAt(0);
                return oldest.alertObject;
            }
            
            return null;
        }
        
        private void ReturnAlertToPool(GameObject alert)
        {
            alert.SetActive(false);
            alertPool.Enqueue(alert);
        }
        
        #endregion
        
        #region Alert Animations
        
        private IEnumerator AnimateAlert(GameObject alertPanel, AlertAnimation animation, AlertConfig config, float duration)
        {
            switch (animation)
            {
                case AlertAnimation.Pulse:
                    yield return PulseAnimation(alertPanel, config, duration);
                    break;
                    
                case AlertAnimation.Flash:
                    yield return FlashAnimation(alertPanel, config, duration);
                    break;
                    
                case AlertAnimation.EdgeGlow:
                    yield return EdgeGlowAnimation(alertPanel, config, duration);
                    break;
                    
                case AlertAnimation.RadialWipe:
                    yield return RadialWipeAnimation(alertPanel, config, duration);
                    break;
                    
                case AlertAnimation.DigitalGlitch:
                    yield return DigitalGlitchAnimation(alertPanel, config, duration);
                    break;
                    
                case AlertAnimation.HologramDistort:
                    yield return HologramDistortAnimation(alertPanel, config, duration);
                    break;
            }
        }
        
        private IEnumerator PulseAnimation(GameObject alertPanel, AlertConfig config, float duration)
        {
            CanvasGroup canvasGroup = alertPanel.GetComponent<CanvasGroup>();
            if (canvasGroup == null)
            {
                canvasGroup = alertPanel.AddComponent<CanvasGroup>();
            }
            
            float elapsed = 0;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float alpha = Mathf.PingPong(elapsed * config.pulseSpeed, 1f);
                alpha = Mathf.Lerp(1f - config.pulseIntensity, 1f, alpha);
                canvasGroup.alpha = alpha;
                
                // Also pulse scale slightly
                float scale = Mathf.Lerp(1f, 1.05f, alpha);
                alertPanel.transform.localScale = Vector3.one * scale;
                
                yield return null;
            }
        }
        
        private IEnumerator FlashAnimation(GameObject alertPanel, AlertConfig config, float duration)
        {
            Image[] images = alertPanel.GetComponentsInChildren<Image>();
            TextMeshProUGUI[] texts = alertPanel.GetComponentsInChildren<TextMeshProUGUI>();
            
            float elapsed = 0;
            bool isOn = true;
            float flashInterval = 0.2f;
            float lastFlash = 0;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                
                if (elapsed - lastFlash > flashInterval)
                {
                    isOn = !isOn;
                    lastFlash = elapsed;
                    
                    // Toggle visibility
                    foreach (var image in images)
                    {
                        image.enabled = isOn;
                    }
                    foreach (var text in texts)
                    {
                        text.enabled = isOn;
                    }
                }
                
                yield return null;
            }
            
            // Ensure everything is visible at end
            foreach (var image in images)
            {
                image.enabled = true;
            }
            foreach (var text in texts)
            {
                text.enabled = true;
            }
        }
        
        private IEnumerator EdgeGlowAnimation(GameObject alertPanel, AlertConfig config, float duration)
        {
            // Find or create edge glow image
            Image edgeGlow = null;
            foreach (Transform child in alertPanel.transform)
            {
                if (child.name.Contains("EdgeGlow"))
                {
                    edgeGlow = child.GetComponent<Image>();
                    break;
                }
            }
            
            if (edgeGlow == null)
            {
                GameObject glowObj = new GameObject("EdgeGlow");
                glowObj.transform.SetParent(alertPanel.transform, false);
                edgeGlow = glowObj.AddComponent<Image>();
                edgeGlow.color = config.primaryColor;
                
                RectTransform glowRect = glowObj.GetComponent<RectTransform>();
                glowRect.anchorMin = Vector2.zero;
                glowRect.anchorMax = Vector2.one;
                glowRect.sizeDelta = new Vector2(10, 10);
                glowRect.anchoredPosition = Vector2.zero;
            }
            
            float elapsed = 0;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float glow = Mathf.Sin(elapsed * config.pulseSpeed * Mathf.PI) * 0.5f + 0.5f;
                
                Color glowColor = config.primaryColor;
                glowColor.a = glow * 0.8f;
                edgeGlow.color = glowColor;
                
                yield return null;
            }
        }
        
        private IEnumerator RadialWipeAnimation(GameObject alertPanel, AlertConfig config, float duration)
        {
            // This would use a custom shader with radial wipe
            // For now, simulate with rotation
            float elapsed = 0;
            float rotationSpeed = 360f / duration;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                alertPanel.transform.Rotate(0, 0, rotationSpeed * Time.deltaTime);
                
                // Pulse alpha
                CanvasGroup group = alertPanel.GetComponent<CanvasGroup>();
                if (group != null)
                {
                    group.alpha = Mathf.PingPong(elapsed * 2, 1);
                }
                
                yield return null;
            }
            
            alertPanel.transform.rotation = Quaternion.identity;
        }
        
        private IEnumerator DigitalGlitchAnimation(GameObject alertPanel, AlertConfig config, float duration)
        {
            RectTransform rect = alertPanel.GetComponent<RectTransform>();
            Vector2 originalPosition = rect.anchoredPosition;
            
            float elapsed = 0;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                
                // Random position offset
                if (Random.Range(0f, 1f) < 0.1f) // 10% chance per frame
                {
                    rect.anchoredPosition = originalPosition + new Vector2(
                        Random.Range(-5f, 5f),
                        Random.Range(-5f, 5f)
                    );
                    
                    // Color shift
                    Image[] images = alertPanel.GetComponentsInChildren<Image>();
                    foreach (var image in images)
                    {
                        Color glitchColor = image.color;
                        glitchColor.r = Random.Range(0.8f, 1f);
                        glitchColor.g = Random.Range(0f, 0.2f);
                        glitchColor.b = Random.Range(0f, 0.2f);
                        image.color = glitchColor;
                    }
                }
                else
                {
                    rect.anchoredPosition = originalPosition;
                    
                    // Reset colors
                    Image[] images = alertPanel.GetComponentsInChildren<Image>();
                    foreach (var image in images)
                    {
                        image.color = config.primaryColor;
                    }
                }
                
                yield return new WaitForSeconds(0.05f);
            }
            
            rect.anchoredPosition = originalPosition;
        }
        
        private IEnumerator HologramDistortAnimation(GameObject alertPanel, AlertConfig config, float duration)
        {
            // Apply hologram material if available
            if (glitchMaterial != null)
            {
                Image mainImage = alertPanel.GetComponent<Image>();
                if (mainImage != null)
                {
                    mainImage.material = glitchMaterial;
                }
            }
            
            float elapsed = 0;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                
                // Update shader properties
                if (glitchMaterial != null)
                {
                    glitchMaterial.SetFloat("_DistortionAmount", 
                        Mathf.Sin(elapsed * config.pulseSpeed) * 0.1f);
                    glitchMaterial.SetFloat("_ChromaticAberration", 
                        Random.Range(0f, 0.05f));
                }
                
                // Scale distortion
                float scaleX = 1f + Mathf.Sin(elapsed * 10) * 0.02f;
                float scaleY = 1f + Mathf.Cos(elapsed * 10) * 0.02f;
                alertPanel.transform.localScale = new Vector3(scaleX, scaleY, 1);
                
                yield return null;
            }
            
            // Reset
            alertPanel.transform.localScale = Vector3.one;
            if (glitchMaterial != null)
            {
                Image mainImage = alertPanel.GetComponent<Image>();
                if (mainImage != null)
                {
                    mainImage.material = null;
                }
            }
        }
        
        #endregion
        
        #region Screen Effects
        
        public enum ScreenEffect
        {
            Flash,
            EdgeGlow,
            FullScreenTint,
            Vignette
        }
        
        public void TriggerScreenEffect(ScreenEffect effect, Color color, float duration)
        {
            switch (effect)
            {
                case ScreenEffect.Flash:
                    StartCoroutine(ScreenFlash(color, duration));
                    break;
                    
                case ScreenEffect.EdgeGlow:
                    StartCoroutine(EdgeGlow(color, duration));
                    break;
                    
                case ScreenEffect.FullScreenTint:
                    StartCoroutine(ScreenTint(color, duration));
                    break;
                    
                case ScreenEffect.Vignette:
                    StartCoroutine(VignetteEffect(color, duration));
                    break;
            }
        }
        
        public void TriggerScreenShake(float duration, float intensity)
        {
            if (screenShakeCoroutine != null)
            {
                StopCoroutine(screenShakeCoroutine);
            }
            screenShakeCoroutine = StartCoroutine(ScreenShake(duration, intensity));
        }
        
        private IEnumerator ScreenFlash(Color color, float duration)
        {
            if (screenOverlay == null) yield break;
            
            screenOverlay.color = color;
            animController.AnimateFloat($"ScreenFlash",
                color.a, 0f, duration,
                (alpha) =>
                {
                    Color c = screenOverlay.color;
                    c.a = alpha;
                    screenOverlay.color = c;
                });
            
            yield return new WaitForSeconds(duration);
        }
        
        private IEnumerator EdgeGlow(Color color, float duration)
        {
            if (edgeVignette == null) yield break;
            
            float elapsed = 0;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float intensity = Mathf.PingPong(elapsed * 2, 1) * 0.8f;
                
                Color c = color;
                c.a = intensity;
                edgeVignette.color = c;
                
                yield return null;
            }
            
            // Fade out
            animController.AnimateFloat($"EdgeGlowFade",
                edgeVignette.color.a, 0f, 0.3f,
                (alpha) =>
                {
                    Color c = edgeVignette.color;
                    c.a = alpha;
                    edgeVignette.color = c;
                });
        }
        
        private IEnumerator ScreenTint(Color color, float duration)
        {
            if (screenOverlay == null) yield break;
            
            Color startColor = screenOverlay.color;
            color.a = 0.3f; // Semi-transparent
            
            animController.AnimateColor($"ScreenTint",
                startColor, color, duration * 0.2f,
                (c) => screenOverlay.color = c);
            
            yield return new WaitForSeconds(duration * 0.8f);
            
            animController.AnimateColor($"ScreenTintFade",
                screenOverlay.color, new Color(1, 1, 1, 0), duration * 0.2f,
                (c) => screenOverlay.color = c);
        }
        
        private IEnumerator VignetteEffect(Color color, float duration)
        {
            if (edgeVignette == null) yield break;
            
            animController.AnimateFloat($"Vignette",
                0f, 0.8f, duration * 0.3f,
                (alpha) =>
                {
                    Color c = color;
                    c.a = alpha;
                    edgeVignette.color = c;
                });
            
            yield return new WaitForSeconds(duration * 0.7f);
            
            animController.AnimateFloat($"VignetteFade",
                edgeVignette.color.a, 0f, duration * 0.3f,
                (alpha) =>
                {
                    Color c = edgeVignette.color;
                    c.a = alpha;
                    edgeVignette.color = c;
                });
        }
        
        private IEnumerator ScreenShake(float duration, float intensity)
        {
            Vector3 originalPosition = transform.position;
            float elapsed = 0;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                
                float percentComplete = elapsed / duration;
                float damper = 1f - Mathf.Clamp01(percentComplete);
                
                float offsetX = Random.Range(-1f, 1f) * intensity * damper;
                float offsetY = Random.Range(-1f, 1f) * intensity * damper;
                
                transform.position = originalPosition + new Vector3(offsetX, offsetY, 0);
                
                yield return null;
            }
            
            transform.position = originalPosition;
            screenShakeCoroutine = null;
        }
        
        #endregion
        
        #region Utility Methods
        
        private void PlayAlertSound(AudioClip clip)
        {
            if (alertAudioSource != null && clip != null)
            {
                alertAudioSource.PlayOneShot(clip);
            }
        }
        
        #endregion
    }
}