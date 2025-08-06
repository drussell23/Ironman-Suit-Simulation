using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using System.Text;

namespace IronManSim.UI.Animations
{
    /// <summary>
    /// JARVIS-specific interface animations and effects
    /// </summary>
    public class JARVISInterfaceAnimations : MonoBehaviour
    {
        [System.Serializable]
        public class VoiceWaveform
        {
            public LineRenderer waveformLine;
            public int resolution = 64;
            public float amplitude = 50f;
            public float frequency = 2f;
            public Color activeColor = new Color(0.2f, 0.8f, 1f);
            public Color inactiveColor = new Color(0.1f, 0.3f, 0.4f);
        }
        
        [System.Serializable]
        public class ResponseAnimation
        {
            public string triggerPhrase;
            public AnimationType animationType;
            public float duration = 1f;
            public AudioClip responseSound;
        }
        
        public enum AnimationType
        {
            Standard,
            Alert,
            Analysis,
            Combat,
            System,
            Humor
        }
        
        [Header("JARVIS Avatar")]
        [SerializeField] private Image jarvisAvatar;
        [SerializeField] private List<Sprite> avatarStates = new List<Sprite>();
        [SerializeField] private ParticleSystem avatarParticles;
        [SerializeField] private float avatarPulseSpeed = 1f;
        
        [Header("Voice Visualization")]
        [SerializeField] private VoiceWaveform primaryWaveform;
        [SerializeField] private VoiceWaveform secondaryWaveform;
        [SerializeField] private Image voiceOrb;
        [SerializeField] private float orbPulseIntensity = 0.3f;
        
        [Header("Text Display")]
        [SerializeField] private TextMeshProUGUI jarvisTextDisplay;
        [SerializeField] private TextMeshProUGUI subtitleDisplay;
        [SerializeField] private float typewriterSpeed = 50f;
        [SerializeField] private float wordHighlightDuration = 0.2f;
        [SerializeField] private Color highlightColor = Color.yellow;
        
        [Header("Response Animations")]
        [SerializeField] private List<ResponseAnimation> responseAnimations = new List<ResponseAnimation>();
        [SerializeField] private GameObject analysisPanelPrefab;
        [SerializeField] private Transform analysisContainer;
        
        [Header("Holographic Display")]
        [SerializeField] private RectTransform holographicDisplay;
        [SerializeField] private Material hologramMaterial;
        [SerializeField] private float hologramNoiseAmount = 0.02f;
        
        [Header("Status Indicators")]
        [SerializeField] private Image statusRing;
        [SerializeField] private List<Image> statusLights = new List<Image>();
        [SerializeField] private Color listeningColor = Color.green;
        [SerializeField] private Color processingColor = Color.yellow;
        [SerializeField] private Color speakingColor = Color.cyan;
        [SerializeField] private Color errorColor = Color.red;
        
        [Header("Audio")]
        [SerializeField] private AudioSource jarvisAudioSource;
        [SerializeField] private AudioClip activationSound;
        [SerializeField] private AudioClip deactivationSound;
        [SerializeField] private AudioClip processingSound;
        
        private UIAnimationController animController;
        private Coroutine currentSpeechCoroutine;
        private Coroutine waveformCoroutine;
        private Queue<string> messageQueue = new Queue<string>();
        private bool isActive = false;
        private JarvisState currentState = JarvisState.Idle;
        
        public enum JarvisState
        {
            Idle,
            Listening,
            Processing,
            Speaking,
            Alert,
            Error
        }
        
        void Start()
        {
            animController = UIAnimationController.Instance;
            InitializeJARVIS();
        }
        
        #region Initialization
        
        private void InitializeJARVIS()
        {
            // Set initial state
            SetJarvisState(JarvisState.Idle);
            
            // Initialize waveforms
            if (primaryWaveform.waveformLine != null)
            {
                primaryWaveform.waveformLine.positionCount = primaryWaveform.resolution;
            }
            
            if (secondaryWaveform.waveformLine != null)
            {
                secondaryWaveform.waveformLine.positionCount = secondaryWaveform.resolution;
            }
            
            // Start ambient animations
            StartCoroutine(AmbientAnimationCoroutine());
        }
        
        #endregion
        
        #region Public Methods
        
        /// <summary>
        /// Activate JARVIS interface
        /// </summary>
        public void ActivateJARVIS()
        {
            if (isActive) return;
            
            isActive = true;
            StartCoroutine(ActivationSequence());
        }
        
        /// <summary>
        /// Deactivate JARVIS interface
        /// </summary>
        public void DeactivateJARVIS()
        {
            if (!isActive) return;
            
            isActive = false;
            StartCoroutine(DeactivationSequence());
        }
        
        /// <summary>
        /// Make JARVIS speak with animated text and voice visualization
        /// </summary>
        public void Speak(string message, AnimationType animationType = AnimationType.Standard)
        {
            messageQueue.Enqueue(message);
            
            if (currentSpeechCoroutine == null)
            {
                currentSpeechCoroutine = StartCoroutine(ProcessSpeechQueue(animationType));
            }
        }
        
        /// <summary>
        /// Show JARVIS is listening
        /// </summary>
        public void StartListening()
        {
            SetJarvisState(JarvisState.Listening);
            StartVoiceVisualization(true);
        }
        
        /// <summary>
        /// Stop listening mode
        /// </summary>
        public void StopListening()
        {
            SetJarvisState(JarvisState.Idle);
            StartVoiceVisualization(false);
        }
        
        /// <summary>
        /// Show analysis animation
        /// </summary>
        public void ShowAnalysis(string analysisType, Dictionary<string, float> data)
        {
            StartCoroutine(AnalysisAnimationCoroutine(analysisType, data));
        }
        
        #endregion
        
        #region Activation/Deactivation
        
        private IEnumerator ActivationSequence()
        {
            PlaySound(activationSound);
            
            // Fade in avatar
            if (jarvisAvatar != null)
            {
                jarvisAvatar.gameObject.SetActive(true);
                CanvasGroup avatarGroup = jarvisAvatar.GetComponent<CanvasGroup>();
                if (avatarGroup == null)
                {
                    avatarGroup = jarvisAvatar.gameObject.AddComponent<CanvasGroup>();
                }
                
                avatarGroup.alpha = 0;
                animController.FadeIn(avatarGroup, 0.5f);
            }
            
            // Activate status ring with spin
            if (statusRing != null)
            {
                statusRing.gameObject.SetActive(true);
                StartCoroutine(SpinStatusRing());
                
                // Animate ring scale
                statusRing.transform.localScale = Vector3.zero;
                animController.AnimateVector3("StatusRingScale",
                    Vector3.zero, Vector3.one, 0.5f,
                    (scale) => statusRing.transform.localScale = scale,
                    null,
                    AnimationCurve.EaseInOut(0, 0, 1, 1));
            }
            
            // Light up status lights sequentially
            yield return StartCoroutine(ActivateStatusLights());
            
            // Initialize voice orb
            if (voiceOrb != null)
            {
                voiceOrb.gameObject.SetActive(true);
                StartCoroutine(PulseVoiceOrb());
            }
            
            // Show welcome message
            yield return new WaitForSeconds(0.5f);
            Speak("Good evening, sir. All systems are operational.", AnimationType.Standard);
            
            SetJarvisState(JarvisState.Idle);
        }
        
        private IEnumerator DeactivationSequence()
        {
            PlaySound(deactivationSound);
            
            // Stop ongoing animations
            StopAllCoroutines();
            
            // Fade out displays
            if (jarvisTextDisplay != null)
            {
                StartCoroutine(FadeOutText(jarvisTextDisplay));
            }
            
            if (subtitleDisplay != null)
            {
                StartCoroutine(FadeOutText(subtitleDisplay));
            }
            
            // Deactivate status lights
            yield return StartCoroutine(DeactivateStatusLights());
            
            // Scale down status ring
            if (statusRing != null)
            {
                animController.AnimateVector3("StatusRingScaleDown",
                    statusRing.transform.localScale, Vector3.zero, 0.3f,
                    (scale) => statusRing.transform.localScale = scale,
                    () => statusRing.gameObject.SetActive(false));
            }
            
            // Fade out avatar
            if (jarvisAvatar != null)
            {
                CanvasGroup avatarGroup = jarvisAvatar.GetComponent<CanvasGroup>();
                animController.FadeOut(avatarGroup, 0.5f,
                    () => jarvisAvatar.gameObject.SetActive(false));
            }
            
            yield return new WaitForSeconds(0.5f);
            
            SetJarvisState(JarvisState.Idle);
        }
        
        #endregion
        
        #region Speech and Text
        
        private IEnumerator ProcessSpeechQueue(AnimationType animationType)
        {
            while (messageQueue.Count > 0)
            {
                string message = messageQueue.Dequeue();
                SetJarvisState(JarvisState.Speaking);
                
                // Start voice visualization
                StartVoiceVisualization(true);
                
                // Display text with typewriter effect
                yield return StartCoroutine(TypewriterEffect(message));
                
                // Continue visualization for a moment
                yield return new WaitForSeconds(0.5f);
                
                // Stop voice visualization
                StartVoiceVisualization(false);
                SetJarvisState(JarvisState.Idle);
                
                yield return new WaitForSeconds(0.2f);
            }
            
            currentSpeechCoroutine = null;
        }
        
        private IEnumerator TypewriterEffect(string message)
        {
            if (jarvisTextDisplay == null) yield break;
            
            jarvisTextDisplay.text = "";
            StringBuilder displayText = new StringBuilder();
            
            // Split into words for better highlighting
            string[] words = message.Split(' ');
            
            for (int w = 0; w < words.Length; w++)
            {
                string word = words[w];
                
                // Add word character by character
                for (int i = 0; i < word.Length; i++)
                {
                    displayText.Append(word[i]);
                    jarvisTextDisplay.text = displayText.ToString();
                    
                    // Play typing sound effect
                    if (Random.Range(0f, 1f) < 0.3f)
                    {
                        // Play subtle click sound
                    }
                    
                    yield return new WaitForSeconds(1f / typewriterSpeed);
                }
                
                // Add space after word (except last word)
                if (w < words.Length - 1)
                {
                    displayText.Append(" ");
                    jarvisTextDisplay.text = displayText.ToString();
                }
                
                // Highlight important words
                if (IsImportantWord(word))
                {
                    yield return StartCoroutine(HighlightWord(jarvisTextDisplay, displayText.Length - word.Length, word.Length));
                }
            }
            
            // Also show in subtitle if available
            if (subtitleDisplay != null)
            {
                subtitleDisplay.text = message;
                StartCoroutine(FadeInText(subtitleDisplay));
            }
        }
        
        private bool IsImportantWord(string word)
        {
            string[] importantWords = { "alert", "warning", "critical", "system", "detected", "activated", "online", "offline" };
            return System.Array.Exists(importantWords, w => word.ToLower().Contains(w));
        }
        
        private IEnumerator HighlightWord(TextMeshProUGUI textComponent, int startIndex, int length)
        {
            // This would use TextMeshPro's rich text or vertex colors
            // For now, flash the entire text
            Color originalColor = textComponent.color;
            
            animController.AnimateColor($"WordHighlight_{startIndex}",
                originalColor, highlightColor, wordHighlightDuration * 0.5f,
                (color) => textComponent.color = color);
            
            yield return new WaitForSeconds(wordHighlightDuration * 0.5f);
            
            animController.AnimateColor($"WordHighlightRevert_{startIndex}",
                highlightColor, originalColor, wordHighlightDuration * 0.5f,
                (color) => textComponent.color = color);
        }
        
        #endregion
        
        #region Voice Visualization
        
        private void StartVoiceVisualization(bool active)
        {
            if (waveformCoroutine != null)
            {
                StopCoroutine(waveformCoroutine);
            }
            
            if (active)
            {
                waveformCoroutine = StartCoroutine(AnimateWaveforms());
            }
            else
            {
                // Fade out waveforms
                StartCoroutine(FadeOutWaveforms());
            }
        }
        
        private IEnumerator AnimateWaveforms()
        {
            float time = 0;
            
            while (true)
            {
                time += Time.deltaTime;
                
                // Update primary waveform
                UpdateWaveform(primaryWaveform, time, 1f);
                
                // Update secondary waveform (inverse)
                UpdateWaveform(secondaryWaveform, time, -1f);
                
                yield return null;
            }
        }
        
        private void UpdateWaveform(VoiceWaveform waveform, float time, float direction)
        {
            if (waveform.waveformLine == null) return;
            
            Vector3[] positions = new Vector3[waveform.resolution];
            float step = 2f * Mathf.PI / waveform.resolution;
            
            for (int i = 0; i < waveform.resolution; i++)
            {
                float x = (i - waveform.resolution / 2f) * 10f;
                
                // Create complex waveform
                float y = 0;
                y += Mathf.Sin((i * step + time * waveform.frequency) * 1f) * waveform.amplitude * 0.5f;
                y += Mathf.Sin((i * step + time * waveform.frequency) * 2f) * waveform.amplitude * 0.3f;
                y += Mathf.Sin((i * step + time * waveform.frequency) * 3f) * waveform.amplitude * 0.2f;
                
                // Add random noise for voice effect
                y += Random.Range(-waveform.amplitude * 0.1f, waveform.amplitude * 0.1f);
                
                y *= direction;
                
                positions[i] = new Vector3(x, y, 0);
            }
            
            waveform.waveformLine.SetPositions(positions);
            
            // Update color based on state
            Color targetColor = currentState == JarvisState.Speaking ? 
                waveform.activeColor : waveform.inactiveColor;
            
            waveform.waveformLine.startColor = targetColor;
            waveform.waveformLine.endColor = targetColor;
        }
        
        private IEnumerator FadeOutWaveforms()
        {
            float duration = 0.3f;
            float elapsed = 0;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float t = 1f - (elapsed / duration);
                
                // Scale down amplitude
                if (primaryWaveform.waveformLine != null)
                {
                    Color color = primaryWaveform.inactiveColor;
                    color.a = t;
                    primaryWaveform.waveformLine.startColor = color;
                    primaryWaveform.waveformLine.endColor = color;
                }
                
                if (secondaryWaveform.waveformLine != null)
                {
                    Color color = secondaryWaveform.inactiveColor;
                    color.a = t;
                    secondaryWaveform.waveformLine.startColor = color;
                    secondaryWaveform.waveformLine.endColor = color;
                }
                
                yield return null;
            }
        }
        
        #endregion
        
        #region State Management
        
        private void SetJarvisState(JarvisState state)
        {
            currentState = state;
            
            // Update status ring color
            if (statusRing != null)
            {
                Color targetColor = GetStateColor(state);
                animController.AnimateColor("StatusRingColor",
                    statusRing.color, targetColor, 0.3f,
                    (color) => statusRing.color = color);
            }
            
            // Update avatar state
            UpdateAvatarState(state);
            
            // Update particle effects
            UpdateParticleEffects(state);
        }
        
        private Color GetStateColor(JarvisState state)
        {
            switch (state)
            {
                case JarvisState.Listening:
                    return listeningColor;
                case JarvisState.Processing:
                    return processingColor;
                case JarvisState.Speaking:
                    return speakingColor;
                case JarvisState.Alert:
                case JarvisState.Error:
                    return errorColor;
                default:
                    return Color.white;
            }
        }
        
        private void UpdateAvatarState(JarvisState state)
        {
            if (jarvisAvatar == null || avatarStates.Count == 0) return;
            
            int stateIndex = (int)state;
            if (stateIndex < avatarStates.Count && avatarStates[stateIndex] != null)
            {
                jarvisAvatar.sprite = avatarStates[stateIndex];
            }
        }
        
        private void UpdateParticleEffects(JarvisState state)
        {
            if (avatarParticles == null) return;
            
            var emission = avatarParticles.emission;
            
            switch (state)
            {
                case JarvisState.Speaking:
                case JarvisState.Processing:
                    emission.rateOverTime = 20f;
                    break;
                case JarvisState.Alert:
                    emission.rateOverTime = 50f;
                    break;
                default:
                    emission.rateOverTime = 5f;
                    break;
            }
        }
        
        #endregion
        
        #region Analysis Animations
        
        private IEnumerator AnalysisAnimationCoroutine(string analysisType, Dictionary<string, float> data)
        {
            SetJarvisState(JarvisState.Processing);
            PlaySound(processingSound);
            
            // Create analysis panel
            GameObject panel = analysisPanelPrefab != null ?
                Instantiate(analysisPanelPrefab, analysisContainer) :
                CreateDefaultAnalysisPanel();
            
            RectTransform panelRect = panel.GetComponent<RectTransform>();
            
            // Animate panel appearance
            panelRect.localScale = Vector3.zero;
            animController.AnimateVector3("AnalysisPanelScale",
                Vector3.zero, Vector3.one, 0.3f,
                (scale) => panelRect.localScale = scale,
                null,
                AnimationCurve.EaseInOut(0, 0, 1, 1));
            
            // Display analysis type
            TextMeshProUGUI titleText = panel.transform.Find("Title")?.GetComponent<TextMeshProUGUI>();
            if (titleText != null)
            {
                titleText.text = $"ANALYSIS: {analysisType.ToUpper()}";
            }
            
            // Animate data points
            yield return new WaitForSeconds(0.3f);
            
            Transform dataContainer = panel.transform.Find("DataContainer");
            if (dataContainer != null)
            {
                foreach (var kvp in data)
                {
                    CreateDataPoint(dataContainer, kvp.Key, kvp.Value);
                    yield return new WaitForSeconds(0.1f);
                }
            }
            
            // Hold for viewing
            yield return new WaitForSeconds(3f);
            
            // Dismiss panel
            animController.AnimateVector3("AnalysisPanelDismiss",
                panelRect.localScale, Vector3.zero, 0.3f,
                (scale) => panelRect.localScale = scale,
                () => Destroy(panel));
            
            SetJarvisState(JarvisState.Idle);
        }
        
        private GameObject CreateDefaultAnalysisPanel()
        {
            GameObject panel = new GameObject("AnalysisPanel");
            panel.transform.SetParent(analysisContainer, false);
            
            RectTransform rect = panel.AddComponent<RectTransform>();
            rect.sizeDelta = new Vector2(400, 300);
            
            Image bg = panel.AddComponent<Image>();
            bg.color = new Color(0, 0, 0, 0.8f);
            
            // Add title
            GameObject titleObj = new GameObject("Title");
            titleObj.transform.SetParent(panel.transform, false);
            TextMeshProUGUI title = titleObj.AddComponent<TextMeshProUGUI>();
            title.text = "ANALYSIS";
            title.fontSize = 24;
            title.alignment = TextAlignmentOptions.Center;
            
            RectTransform titleRect = titleObj.GetComponent<RectTransform>();
            titleRect.anchorMin = new Vector2(0, 0.8f);
            titleRect.anchorMax = new Vector2(1, 1);
            titleRect.sizeDelta = Vector2.zero;
            titleRect.anchoredPosition = Vector2.zero;
            
            // Add data container
            GameObject dataContainer = new GameObject("DataContainer");
            dataContainer.transform.SetParent(panel.transform, false);
            RectTransform dataRect = dataContainer.AddComponent<RectTransform>();
            dataRect.anchorMin = new Vector2(0.1f, 0.1f);
            dataRect.anchorMax = new Vector2(0.9f, 0.7f);
            dataRect.sizeDelta = Vector2.zero;
            dataRect.anchoredPosition = Vector2.zero;
            
            return panel;
        }
        
        private void CreateDataPoint(Transform container, string label, float value)
        {
            GameObject dataPoint = new GameObject($"Data_{label}");
            dataPoint.transform.SetParent(container, false);
            
            TextMeshProUGUI text = dataPoint.AddComponent<TextMeshProUGUI>();
            text.text = $"{label}: {value:F1}";
            text.fontSize = 16;
            
            // Animate appearance
            CanvasGroup group = dataPoint.AddComponent<CanvasGroup>();
            group.alpha = 0;
            animController.FadeIn(group, 0.2f);
            
            // Position based on child count
            RectTransform rect = dataPoint.GetComponent<RectTransform>();
            rect.anchoredPosition = new Vector2(0, -container.childCount * 30);
        }
        
        #endregion
        
        #region Ambient Animations
        
        private IEnumerator AmbientAnimationCoroutine()
        {
            while (true)
            {
                // Pulse voice orb
                if (voiceOrb != null && isActive)
                {
                    float pulse = Mathf.Sin(Time.time * avatarPulseSpeed) * orbPulseIntensity + (1f - orbPulseIntensity);
                    voiceOrb.transform.localScale = Vector3.one * pulse;
                }
                
                // Update holographic noise
                if (hologramMaterial != null)
                {
                    hologramMaterial.SetFloat("_NoiseAmount", 
                        hologramNoiseAmount * (1f + Mathf.Sin(Time.time * 2f) * 0.2f));
                }
                
                yield return null;
            }
        }
        
        private IEnumerator SpinStatusRing()
        {
            while (statusRing != null && statusRing.gameObject.activeInHierarchy)
            {
                statusRing.transform.Rotate(0, 0, 30f * Time.deltaTime);
                yield return null;
            }
        }
        
        private IEnumerator PulseVoiceOrb()
        {
            while (voiceOrb != null && voiceOrb.gameObject.activeInHierarchy)
            {
                // Base pulse
                float basePulse = Mathf.Sin(Time.time * 2f) * 0.1f + 0.9f;
                
                // Additional pulse when speaking
                float speakingPulse = currentState == JarvisState.Speaking ? 
                    Mathf.Sin(Time.time * 10f) * 0.2f : 0f;
                
                float totalScale = basePulse + speakingPulse;
                voiceOrb.transform.localScale = Vector3.one * totalScale;
                
                // Update color intensity
                Color orbColor = voiceOrb.color;
                orbColor.a = currentState == JarvisState.Speaking ? 1f : 0.7f;
                voiceOrb.color = orbColor;
                
                yield return null;
            }
        }
        
        #endregion
        
        #region Status Lights
        
        private IEnumerator ActivateStatusLights()
        {
            foreach (var light in statusLights)
            {
                if (light != null)
                {
                    light.gameObject.SetActive(true);
                    
                    // Fade in
                    CanvasGroup group = light.GetComponent<CanvasGroup>();
                    if (group == null)
                    {
                        group = light.gameObject.AddComponent<CanvasGroup>();
                    }
                    
                    group.alpha = 0;
                    animController.FadeIn(group, 0.2f);
                    
                    // Set color
                    light.color = listeningColor;
                    
                    yield return new WaitForSeconds(0.1f);
                }
            }
        }
        
        private IEnumerator DeactivateStatusLights()
        {
            for (int i = statusLights.Count - 1; i >= 0; i--)
            {
                if (statusLights[i] != null)
                {
                    Image light = statusLights[i];
                    
                    // Fade out
                    CanvasGroup group = light.GetComponent<CanvasGroup>();
                    animController.FadeOut(group, 0.2f,
                        () => light.gameObject.SetActive(false));
                    
                    yield return new WaitForSeconds(0.05f);
                }
            }
        }
        
        #endregion
        
        #region Utility Methods
        
        private void PlaySound(AudioClip clip)
        {
            if (jarvisAudioSource != null && clip != null)
            {
                jarvisAudioSource.PlayOneShot(clip);
            }
        }
        
        private IEnumerator FadeInText(TextMeshProUGUI text)
        {
            Color c = text.color;
            c.a = 0;
            text.color = c;
            
            animController.AnimateFloat($"TextFadeIn_{text.GetInstanceID()}",
                0, 1, 0.3f,
                (alpha) =>
                {
                    Color color = text.color;
                    color.a = alpha;
                    text.color = color;
                });
            
            yield return new WaitForSeconds(0.3f);
        }
        
        private IEnumerator FadeOutText(TextMeshProUGUI text)
        {
            animController.AnimateFloat($"TextFadeOut_{text.GetInstanceID()}",
                text.color.a, 0, 0.3f,
                (alpha) =>
                {
                    Color color = text.color;
                    color.a = alpha;
                    text.color = color;
                });
            
            yield return new WaitForSeconds(0.3f);
        }
        
        #endregion
    }
}