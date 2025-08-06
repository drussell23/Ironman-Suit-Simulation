using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;

namespace IronManSim.UI.Animations
{
    /// <summary>
    /// Specialized animations for the Iron Man HUD elements
    /// </summary>
    public class HUDAnimations : MonoBehaviour
    {
        [Header("HUD Elements")]
        [SerializeField] private RectTransform hudContainer;
        [SerializeField] private List<RectTransform> hudPanels = new List<RectTransform>();
        [SerializeField] private List<Image> hudBorders = new List<Image>();
        [SerializeField] private List<TextMeshProUGUI> hudTexts = new List<TextMeshProUGUI>();
        
        [Header("Boot Sequence")]
        [SerializeField] private float bootDuration = 3f;
        [SerializeField] private AnimationCurve bootCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
        
        [Header("Scanning Effect")]
        [SerializeField] private RectTransform scanLine;
        [SerializeField] private Image scanLineImage;
        [SerializeField] private float scanSpeed = 2f;
        [SerializeField] private Color scanColor = new Color(0, 1, 1, 0.5f);
        
        [Header("Targeting System")]
        [SerializeField] private GameObject targetReticlePrefab;
        [SerializeField] private int maxTargets = 10;
        private Queue<GameObject> targetPool = new Queue<GameObject>();
        private List<TargetReticle> activeTargets = new List<TargetReticle>();
        
        [Header("Radar Animation")]
        [SerializeField] private RectTransform radarSweep;
        [SerializeField] private float radarSweepSpeed = 60f; // degrees per second
        [SerializeField] private Image radarBlip;
        
        [Header("Data Stream")]
        [SerializeField] private TextMeshProUGUI dataStreamText;
        [SerializeField] private float dataStreamSpeed = 50f; // characters per second
        private Queue<string> dataQueue = new Queue<string>();
        private Coroutine dataStreamCoroutine;
        
        private UIAnimationController animController;
        
        void Start()
        {
            animController = UIAnimationController.Instance;
            InitializeTargetPool();
        }
        
        #region Boot Sequence
        
        /// <summary>
        /// Plays the full HUD boot sequence
        /// </summary>
        public void PlayBootSequence()
        {
            StartCoroutine(BootSequenceCoroutine());
        }
        
        private IEnumerator BootSequenceCoroutine()
        {
            // Phase 1: Initialize HUD container
            hudContainer.gameObject.SetActive(true);
            CanvasGroup hudCanvasGroup = hudContainer.GetComponent<CanvasGroup>();
            if (hudCanvasGroup == null)
            {
                hudCanvasGroup = hudContainer.gameObject.AddComponent<CanvasGroup>();
            }
            hudCanvasGroup.alpha = 0;
            
            // Fade in main container
            animController.FadeIn(hudCanvasGroup, bootDuration * 0.2f);
            yield return new WaitForSeconds(bootDuration * 0.2f);
            
            // Phase 2: Boot individual panels with stagger
            float panelDelay = bootDuration * 0.5f / hudPanels.Count;
            foreach (var panel in hudPanels)
            {
                StartCoroutine(BootPanel(panel, 0.5f));
                yield return new WaitForSeconds(panelDelay);
            }
            
            // Phase 3: Activate borders with glow effect
            foreach (var border in hudBorders)
            {
                StartCoroutine(ActivateBorder(border, 0.3f));
            }
            
            // Phase 4: Type out system messages
            yield return new WaitForSeconds(0.5f);
            AddDataStreamMessage("JARVIS SYSTEM ONLINE");
            AddDataStreamMessage("MARK 85 SUIT INITIALIZED");
            AddDataStreamMessage("ALL SYSTEMS OPERATIONAL");
            
            // Phase 5: Start continuous animations
            StartRadarSweep();
            StartScanLineAnimation();
        }
        
        private IEnumerator BootPanel(RectTransform panel, float duration)
        {
            // Scale from 0 to 1 with overshoot
            panel.localScale = Vector3.zero;
            panel.gameObject.SetActive(true);
            
            float elapsed = 0;
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float t = elapsed / duration;
                float scale = bootCurve.Evaluate(t) * 1.1f;
                if (t > 0.8f) scale = Mathf.Lerp(1.1f, 1f, (t - 0.8f) / 0.2f);
                panel.localScale = Vector3.one * scale;
                yield return null;
            }
            
            panel.localScale = Vector3.one;
        }
        
        private IEnumerator ActivateBorder(Image border, float duration)
        {
            Color startColor = border.color;
            startColor.a = 0;
            border.color = startColor;
            
            Color targetColor = startColor;
            targetColor.a = 1;
            
            animController.AnimateColor($"BorderGlow_{border.GetInstanceID()}", 
                startColor, targetColor, duration,
                (color) => border.color = color);
            
            yield return null;
        }
        
        #endregion
        
        #region Scanning Effects
        
        private void StartScanLineAnimation()
        {
            if (scanLine != null)
            {
                StartCoroutine(ScanLineCoroutine());
            }
        }
        
        private IEnumerator ScanLineCoroutine()
        {
            while (true)
            {
                // Vertical scan
                float startY = -Screen.height / 2;
                float endY = Screen.height / 2;
                
                animController.AnimateFloat("ScanLine_Vertical", startY, endY, scanSpeed,
                    (y) => 
                    {
                        scanLine.anchoredPosition = new Vector2(scanLine.anchoredPosition.x, y);
                        
                        // Fade based on position
                        float alpha = 1f - Mathf.Abs(y / (Screen.height / 2));
                        Color color = scanLineImage.color;
                        color.a = alpha * 0.5f;
                        scanLineImage.color = color;
                    });
                
                yield return new WaitForSeconds(scanSpeed + 0.5f);
                
                // Horizontal scan
                float startX = -Screen.width / 2;
                float endX = Screen.width / 2;
                
                animController.AnimateFloat("ScanLine_Horizontal", startX, endX, scanSpeed,
                    (x) => 
                    {
                        scanLine.anchoredPosition = new Vector2(x, scanLine.anchoredPosition.y);
                        
                        // Fade based on position
                        float alpha = 1f - Mathf.Abs(x / (Screen.width / 2));
                        Color color = scanLineImage.color;
                        color.a = alpha * 0.5f;
                        scanLineImage.color = color;
                    });
                
                yield return new WaitForSeconds(scanSpeed + 0.5f);
            }
        }
        
        #endregion
        
        #region Targeting System
        
        private void InitializeTargetPool()
        {
            if (targetReticlePrefab == null) return;
            
            for (int i = 0; i < maxTargets; i++)
            {
                GameObject target = Instantiate(targetReticlePrefab, hudContainer);
                target.SetActive(false);
                targetPool.Enqueue(target);
            }
        }
        
        /// <summary>
        /// Add a new target to track
        /// </summary>
        public TargetReticle AddTarget(Vector3 worldPosition, string targetInfo = "")
        {
            if (targetPool.Count == 0) return null;
            
            GameObject targetObj = targetPool.Dequeue();
            targetObj.SetActive(true);
            
            TargetReticle reticle = targetObj.GetComponent<TargetReticle>();
            if (reticle == null)
            {
                reticle = targetObj.AddComponent<TargetReticle>();
            }
            
            reticle.Initialize(worldPosition, targetInfo);
            activeTargets.Add(reticle);
            
            // Animate target appearance
            RectTransform targetRect = targetObj.GetComponent<RectTransform>();
            targetRect.localScale = Vector3.zero;
            animController.AnimateVector3($"TargetAppear_{targetObj.GetInstanceID()}",
                Vector3.zero, Vector3.one, 0.3f,
                (scale) => targetRect.localScale = scale,
                null,
                AnimationCurve.EaseInOut(0, 0, 1, 1));
            
            return reticle;
        }
        
        /// <summary>
        /// Remove a target
        /// </summary>
        public void RemoveTarget(TargetReticle target)
        {
            if (target == null || !activeTargets.Contains(target)) return;
            
            activeTargets.Remove(target);
            
            // Animate target disappearance
            RectTransform targetRect = target.GetComponent<RectTransform>();
            animController.AnimateVector3($"TargetDisappear_{target.GetInstanceID()}",
                targetRect.localScale, Vector3.zero, 0.2f,
                (scale) => targetRect.localScale = scale,
                () =>
                {
                    target.gameObject.SetActive(false);
                    targetPool.Enqueue(target.gameObject);
                });
        }
        
        #endregion
        
        #region Radar Animation
        
        private void StartRadarSweep()
        {
            if (radarSweep != null)
            {
                StartCoroutine(RadarSweepCoroutine());
            }
        }
        
        private IEnumerator RadarSweepCoroutine()
        {
            while (true)
            {
                radarSweep.Rotate(0, 0, -radarSweepSpeed * Time.deltaTime);
                
                // Create blip effect every full rotation
                if (Mathf.Abs(radarSweep.rotation.eulerAngles.z) < 1f)
                {
                    CreateRadarBlip();
                }
                
                yield return null;
            }
        }
        
        private void CreateRadarBlip()
        {
            if (radarBlip != null)
            {
                GameObject blip = Instantiate(radarBlip.gameObject, radarBlip.transform.parent);
                Image blipImage = blip.GetComponent<Image>();
                
                // Random position on radar
                RectTransform blipRect = blip.GetComponent<RectTransform>();
                float distance = Random.Range(0, 100f);
                float angle = Random.Range(0, 360f) * Mathf.Deg2Rad;
                blipRect.anchoredPosition = new Vector2(
                    Mathf.Cos(angle) * distance,
                    Mathf.Sin(angle) * distance
                );
                
                // Fade out blip
                animController.AnimateColor($"RadarBlip_{blip.GetInstanceID()}",
                    blipImage.color, new Color(blipImage.color.r, blipImage.color.g, blipImage.color.b, 0),
                    2f,
                    (color) => blipImage.color = color,
                    () => Destroy(blip));
            }
        }
        
        #endregion
        
        #region Data Stream
        
        /// <summary>
        /// Add a message to the data stream
        /// </summary>
        public void AddDataStreamMessage(string message)
        {
            dataQueue.Enqueue(message);
            
            if (dataStreamCoroutine == null)
            {
                dataStreamCoroutine = StartCoroutine(ProcessDataStream());
            }
        }
        
        private IEnumerator ProcessDataStream()
        {
            while (dataQueue.Count > 0)
            {
                string message = dataQueue.Dequeue();
                yield return StartCoroutine(TypewriterEffect(dataStreamText, message));
                yield return new WaitForSeconds(0.5f);
            }
            
            dataStreamCoroutine = null;
        }
        
        private IEnumerator TypewriterEffect(TextMeshProUGUI textComponent, string message)
        {
            textComponent.text = "";
            
            foreach (char letter in message)
            {
                textComponent.text += letter;
                yield return new WaitForSeconds(1f / dataStreamSpeed);
            }
        }
        
        #endregion
        
        #region Power Down
        
        /// <summary>
        /// Plays the HUD shutdown sequence
        /// </summary>
        public void PlayShutdownSequence()
        {
            StartCoroutine(ShutdownSequenceCoroutine());
        }
        
        private IEnumerator ShutdownSequenceCoroutine()
        {
            // Stop all continuous animations
            StopAllCoroutines();
            
            // Add shutdown message
            AddDataStreamMessage("SYSTEMS SHUTTING DOWN...");
            
            // Glitch effect
            yield return StartCoroutine(GlitchEffect(0.5f));
            
            // Fade out all panels
            foreach (var panel in hudPanels)
            {
                CanvasGroup panelGroup = panel.GetComponent<CanvasGroup>();
                if (panelGroup == null)
                {
                    panelGroup = panel.gameObject.AddComponent<CanvasGroup>();
                }
                
                animController.FadeOut(panelGroup, 0.3f);
            }
            
            yield return new WaitForSeconds(0.5f);
            
            // Final fade out
            CanvasGroup hudCanvasGroup = hudContainer.GetComponent<CanvasGroup>();
            animController.FadeOut(hudCanvasGroup, 0.5f, () => hudContainer.gameObject.SetActive(false));
        }
        
        private IEnumerator GlitchEffect(float duration)
        {
            float elapsed = 0;
            Vector3 originalPosition = hudContainer.anchoredPosition;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                
                // Random position offset
                float offsetX = Random.Range(-5f, 5f);
                float offsetY = Random.Range(-5f, 5f);
                hudContainer.anchoredPosition = originalPosition + new Vector3(offsetX, offsetY, 0);
                
                // Random color shift
                foreach (var border in hudBorders)
                {
                    Color glitchColor = border.color;
                    glitchColor.r = Random.Range(0.8f, 1f);
                    border.color = glitchColor;
                }
                
                yield return new WaitForSeconds(0.05f);
            }
            
            hudContainer.anchoredPosition = originalPosition;
        }
        
        #endregion
    }
    
    /// <summary>
    /// Component for individual target reticles
    /// </summary>
    public class TargetReticle : MonoBehaviour
    {
        private Vector3 worldPosition;
        private string targetInfo;
        private TextMeshProUGUI infoText;
        private Image reticleImage;
        private RectTransform rectTransform;
        
        public void Initialize(Vector3 position, string info)
        {
            worldPosition = position;
            targetInfo = info;
            
            rectTransform = GetComponent<RectTransform>();
            reticleImage = GetComponent<Image>();
            infoText = GetComponentInChildren<TextMeshProUGUI>();
            
            if (infoText != null)
            {
                infoText.text = info;
            }
        }
        
        void Update()
        {
            // Update screen position based on world position
            Vector3 screenPos = Camera.main.WorldToScreenPoint(worldPosition);
            rectTransform.position = screenPos;
            
            // Rotate reticle
            rectTransform.Rotate(0, 0, 45f * Time.deltaTime);
            
            // Hide if behind camera
            bool isVisible = screenPos.z > 0;
            gameObject.SetActive(isVisible);
        }
    }
}