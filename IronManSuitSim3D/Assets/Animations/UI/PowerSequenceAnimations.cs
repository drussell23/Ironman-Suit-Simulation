using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine.Events;

namespace IronManSim.UI.Animations
{
    /// <summary>
    /// Manages power-up and power-down sequences for the Iron Man suit UI
    /// </summary>
    public class PowerSequenceAnimations : MonoBehaviour
    {
        [System.Serializable]
        public class PowerStage
        {
            public string stageName;
            public float duration = 1f;
            public List<GameObject> activateObjects = new List<GameObject>();
            public List<GameObject> deactivateObjects = new List<GameObject>();
            public UnityEvent onStageStart;
            public UnityEvent onStageComplete;
            public AudioClip stageSound;
        }
        
        [Header("Arc Reactor")]
        [SerializeField] private Image arcReactorCore;
        [SerializeField] private Image arcReactorGlow;
        [SerializeField] private List<Image> arcReactorRings = new List<Image>();
        [SerializeField] private float reactorSpinSpeed = 90f;
        [SerializeField] private AnimationCurve reactorPowerCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
        
        [Header("Power Stages")]
        [SerializeField] private List<PowerStage> powerUpStages = new List<PowerStage>();
        [SerializeField] private List<PowerStage> powerDownStages = new List<PowerStage>();
        
        [Header("System Status")]
        [SerializeField] private TextMeshProUGUI systemStatusText;
        [SerializeField] private List<string> bootMessages = new List<string>
        {
            "INITIALIZING ARC REACTOR...",
            "CALIBRATING REPULSOR ARRAY...",
            "LOADING J.A.R.V.I.S...",
            "SYNCHRONIZING SERVO MOTORS...",
            "ESTABLISHING NEURAL LINK...",
            "SYSTEMS ONLINE"
        };
        
        [Header("Visual Effects")]
        [SerializeField] private ParticleSystem energyParticles;
        [SerializeField] private Light reactorLight;
        [SerializeField] private float maxLightIntensity = 5f;
        [SerializeField] private Color powerUpColor = new Color(0.2f, 0.8f, 1f);
        [SerializeField] private Color powerDownColor = new Color(1f, 0.2f, 0.2f);
        
        [Header("Energy Flow")]
        [SerializeField] private List<Image> energyFlowLines = new List<Image>();
        [SerializeField] private Material energyFlowMaterial;
        [SerializeField] private float energyFlowSpeed = 2f;
        
        [Header("HUD Elements")]
        [SerializeField] private CanvasGroup mainHUDGroup;
        [SerializeField] private List<CanvasGroup> hudSections = new List<CanvasGroup>();
        [SerializeField] private float hudFadeDelay = 0.1f;
        
        [Header("Audio")]
        [SerializeField] private AudioSource powerAudioSource;
        [SerializeField] private AudioClip reactorStartupSound;
        [SerializeField] private AudioClip reactorShutdownSound;
        [SerializeField] private AudioClip powerUpCompleteSound;
        [SerializeField] private AudioClip emergencyShutdownSound;
        
        private UIAnimationController animController;
        private Coroutine currentSequence;
        private bool isPoweredOn = false;
        private float currentPowerLevel = 0f;
        
        void Start()
        {
            animController = UIAnimationController.Instance;
            
            // Initial state
            SetPowerLevel(0f);
            if (mainHUDGroup != null)
            {
                mainHUDGroup.alpha = 0;
            }
        }
        
        #region Public Methods
        
        /// <summary>
        /// Start the power-up sequence
        /// </summary>
        public void StartPowerUpSequence()
        {
            if (currentSequence != null)
            {
                StopCoroutine(currentSequence);
            }
            
            currentSequence = StartCoroutine(PowerUpSequenceCoroutine());
        }
        
        /// <summary>
        /// Start the power-down sequence
        /// </summary>
        public void StartPowerDownSequence()
        {
            if (currentSequence != null)
            {
                StopCoroutine(currentSequence);
            }
            
            currentSequence = StartCoroutine(PowerDownSequenceCoroutine());
        }
        
        /// <summary>
        /// Emergency shutdown - immediate power off
        /// </summary>
        public void EmergencyShutdown()
        {
            if (currentSequence != null)
            {
                StopCoroutine(currentSequence);
            }
            
            StartCoroutine(EmergencyShutdownCoroutine());
        }
        
        /// <summary>
        /// Set power level directly (0-1)
        /// </summary>
        public void SetPowerLevel(float level)
        {
            currentPowerLevel = Mathf.Clamp01(level);
            UpdatePowerVisuals(currentPowerLevel);
        }
        
        #endregion
        
        #region Power Up Sequence
        
        private IEnumerator PowerUpSequenceCoroutine()
        {
            isPoweredOn = false;
            
            // Play startup sound
            PlaySound(reactorStartupSound);
            
            // Phase 1: Arc Reactor Initialization
            yield return StartCoroutine(InitializeArcReactor());
            
            // Phase 2: Energy Distribution
            yield return StartCoroutine(DistributeEnergy());
            
            // Phase 3: Execute power-up stages
            for (int i = 0; i < powerUpStages.Count; i++)
            {
                yield return StartCoroutine(ExecutePowerStage(powerUpStages[i], i));
            }
            
            // Phase 4: HUD Activation
            yield return StartCoroutine(ActivateHUD());
            
            // Complete
            isPoweredOn = true;
            PlaySound(powerUpCompleteSound);
            
            if (systemStatusText != null)
            {
                systemStatusText.text = "SYSTEMS ONLINE";
                StartCoroutine(PulseText(systemStatusText, 2f));
            }
        }
        
        private IEnumerator InitializeArcReactor()
        {
            // Start reactor spin
            if (arcReactorCore != null)
            {
                StartCoroutine(SpinReactor(true));
            }
            
            // Animate reactor brightness
            float duration = 2f;
            animController.AnimateFloat("ReactorPower", 0f, 1f, duration,
                (power) =>
                {
                    SetPowerLevel(power * 0.3f); // 30% power during initialization
                    
                    // Pulse effect
                    float pulse = Mathf.Sin(Time.time * 10) * 0.1f + 0.9f;
                    if (arcReactorGlow != null)
                    {
                        Color glowColor = powerUpColor;
                        glowColor.a = power * pulse;
                        arcReactorGlow.color = glowColor;
                    }
                },
                null,
                reactorPowerCurve);
            
            // Activate rings sequentially
            float ringDelay = duration / arcReactorRings.Count;
            foreach (var ring in arcReactorRings)
            {
                StartCoroutine(FadeInRing(ring, 0.3f));
                yield return new WaitForSeconds(ringDelay);
            }
            
            yield return new WaitForSeconds(0.5f);
        }
        
        private IEnumerator FadeInRing(Image ring, float duration)
        {
            Color startColor = ring.color;
            startColor.a = 0;
            ring.color = startColor;
            
            animController.AnimateColor($"Ring_{ring.GetInstanceID()}",
                startColor, powerUpColor, duration,
                (color) => ring.color = color);
            
            yield return new WaitForSeconds(duration);
        }
        
        private IEnumerator DistributeEnergy()
        {
            // Show energy flowing from reactor to systems
            foreach (var flowLine in energyFlowLines)
            {
                if (flowLine != null && energyFlowMaterial != null)
                {
                    flowLine.material = new Material(energyFlowMaterial);
                    StartCoroutine(AnimateEnergyFlow(flowLine));
                }
            }
            
            // Increase power level
            animController.AnimateFloat("PowerDistribution", currentPowerLevel, 0.7f, 1.5f,
                (power) => SetPowerLevel(power));
            
            // Show status messages
            if (systemStatusText != null && bootMessages.Count > 0)
            {
                for (int i = 0; i < Mathf.Min(3, bootMessages.Count); i++)
                {
                    systemStatusText.text = bootMessages[i];
                    yield return new WaitForSeconds(0.5f);
                }
            }
            
            yield return new WaitForSeconds(0.5f);
        }
        
        private IEnumerator AnimateEnergyFlow(Image flowLine)
        {
            Material flowMat = flowLine.material;
            float offset = 0;
            
            while (isPoweredOn || currentSequence != null)
            {
                offset += Time.deltaTime * energyFlowSpeed;
                flowMat.SetTextureOffset("_MainTex", new Vector2(offset, 0));
                
                // Pulse brightness
                float brightness = Mathf.Sin(Time.time * 2) * 0.2f + 0.8f;
                flowMat.SetFloat("_Brightness", brightness * currentPowerLevel);
                
                yield return null;
            }
        }
        
        private IEnumerator ExecutePowerStage(PowerStage stage, int stageIndex)
        {
            // Fire start event
            stage.onStageStart?.Invoke();
            
            // Play stage sound
            if (stage.stageSound != null)
            {
                PlaySound(stage.stageSound);
            }
            
            // Update status
            if (systemStatusText != null && stageIndex < bootMessages.Count)
            {
                systemStatusText.text = bootMessages[stageIndex + 3]; // Skip first 3 messages
            }
            
            // Activate objects
            foreach (var obj in stage.activateObjects)
            {
                if (obj != null)
                {
                    obj.SetActive(true);
                    
                    // Fade in if has CanvasGroup
                    CanvasGroup group = obj.GetComponent<CanvasGroup>();
                    if (group != null)
                    {
                        group.alpha = 0;
                        animController.FadeIn(group, stage.duration * 0.5f);
                    }
                }
            }
            
            // Deactivate objects
            foreach (var obj in stage.deactivateObjects)
            {
                if (obj != null)
                {
                    CanvasGroup group = obj.GetComponent<CanvasGroup>();
                    if (group != null)
                    {
                        animController.FadeOut(group, stage.duration * 0.5f,
                            () => obj.SetActive(false));
                    }
                    else
                    {
                        obj.SetActive(false);
                    }
                }
            }
            
            yield return new WaitForSeconds(stage.duration);
            
            // Fire complete event
            stage.onStageComplete?.Invoke();
        }
        
        private IEnumerator ActivateHUD()
        {
            // Fade in main HUD
            if (mainHUDGroup != null)
            {
                animController.FadeIn(mainHUDGroup, 0.5f);
            }
            
            // Activate HUD sections with cascade effect
            foreach (var section in hudSections)
            {
                if (section != null)
                {
                    section.alpha = 0;
                    animController.FadeIn(section, 0.3f);
                    
                    // Scale bounce effect
                    RectTransform rect = section.GetComponent<RectTransform>();
                    if (rect != null)
                    {
                        rect.localScale = Vector3.zero;
                        animController.AnimateVector3($"HUDSection_{section.GetInstanceID()}",
                            Vector3.zero, Vector3.one, 0.3f,
                            (scale) => rect.localScale = scale,
                            null,
                            AnimationCurve.EaseInOut(0, 0, 1, 1));
                    }
                    
                    yield return new WaitForSeconds(hudFadeDelay);
                }
            }
            
            // Final power level
            animController.AnimateFloat("FinalPower", currentPowerLevel, 1f, 0.5f,
                (power) => SetPowerLevel(power));
        }
        
        #endregion
        
        #region Power Down Sequence
        
        private IEnumerator PowerDownSequenceCoroutine()
        {
            isPoweredOn = false;
            
            // Play shutdown sound
            PlaySound(reactorShutdownSound);
            
            // Update status
            if (systemStatusText != null)
            {
                systemStatusText.text = "SHUTTING DOWN...";
            }
            
            // Phase 1: Deactivate HUD
            yield return StartCoroutine(DeactivateHUD());
            
            // Phase 2: Execute power-down stages
            for (int i = 0; i < powerDownStages.Count; i++)
            {
                yield return StartCoroutine(ExecutePowerStage(powerDownStages[i], i));
            }
            
            // Phase 3: Drain energy
            yield return StartCoroutine(DrainEnergy());
            
            // Phase 4: Shutdown reactor
            yield return StartCoroutine(ShutdownReactor());
            
            // Complete
            currentPowerLevel = 0f;
            UpdatePowerVisuals(0f);
            
            if (systemStatusText != null)
            {
                systemStatusText.text = "SYSTEM OFFLINE";
                yield return new WaitForSeconds(1f);
                systemStatusText.text = "";
            }
        }
        
        private IEnumerator DeactivateHUD()
        {
            // Deactivate HUD sections in reverse order
            for (int i = hudSections.Count - 1; i >= 0; i--)
            {
                if (hudSections[i] != null)
                {
                    animController.FadeOut(hudSections[i], 0.2f);
                    yield return new WaitForSeconds(hudFadeDelay * 0.5f);
                }
            }
            
            // Fade out main HUD
            if (mainHUDGroup != null)
            {
                animController.FadeOut(mainHUDGroup, 0.3f);
            }
            
            yield return new WaitForSeconds(0.3f);
        }
        
        private IEnumerator DrainEnergy()
        {
            // Stop energy flow animations
            foreach (var flowLine in energyFlowLines)
            {
                if (flowLine != null)
                {
                    CanvasGroup group = flowLine.GetComponent<CanvasGroup>();
                    if (group == null)
                    {
                        group = flowLine.gameObject.AddComponent<CanvasGroup>();
                    }
                    animController.FadeOut(group, 0.5f);
                }
            }
            
            // Reduce power level
            animController.AnimateFloat("PowerDrain", currentPowerLevel, 0.2f, 1f,
                (power) => SetPowerLevel(power));
            
            yield return new WaitForSeconds(1f);
        }
        
        private IEnumerator ShutdownReactor()
        {
            // Fade out rings
            foreach (var ring in arcReactorRings)
            {
                animController.AnimateColor($"RingShutdown_{ring.GetInstanceID()}",
                    ring.color, new Color(ring.color.r, ring.color.g, ring.color.b, 0),
                    0.3f,
                    (color) => ring.color = color);
            }
            
            // Reduce reactor glow
            if (arcReactorGlow != null)
            {
                animController.AnimateFloat("ReactorGlowShutdown", arcReactorGlow.color.a, 0f, 1f,
                    (alpha) =>
                    {
                        Color c = arcReactorGlow.color;
                        c.a = alpha;
                        arcReactorGlow.color = c;
                    });
            }
            
            // Slow down and stop reactor spin
            yield return new WaitForSeconds(0.5f);
            StopAllCoroutines();
            
            // Final power off
            SetPowerLevel(0f);
        }
        
        #endregion
        
        #region Emergency Shutdown
        
        private IEnumerator EmergencyShutdownCoroutine()
        {
            // Play emergency sound
            PlaySound(emergencyShutdownSound);
            
            // Flash warning
            if (systemStatusText != null)
            {
                systemStatusText.text = "EMERGENCY SHUTDOWN";
                systemStatusText.color = Color.red;
                StartCoroutine(FlashText(systemStatusText, 0.1f, 5));
            }
            
            // Instantly hide all UI
            if (mainHUDGroup != null)
            {
                mainHUDGroup.alpha = 0;
            }
            
            // Glitch effect
            yield return StartCoroutine(EmergencyGlitchEffect(0.5f));
            
            // Instant power off
            SetPowerLevel(0f);
            isPoweredOn = false;
            
            // Hide all elements
            foreach (var stage in powerUpStages)
            {
                foreach (var obj in stage.activateObjects)
                {
                    if (obj != null) obj.SetActive(false);
                }
            }
            
            // Stop all animations
            StopAllCoroutines();
        }
        
        private IEnumerator EmergencyGlitchEffect(float duration)
        {
            float elapsed = 0;
            Transform hudTransform = mainHUDGroup?.transform ?? transform;
            Vector3 originalPos = hudTransform.position;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                
                // Shake
                hudTransform.position = originalPos + (Vector3)Random.insideUnitCircle * 10f;
                
                // Random power fluctuations
                SetPowerLevel(Random.Range(0f, currentPowerLevel));
                
                yield return new WaitForSeconds(0.02f);
            }
            
            hudTransform.position = originalPos;
        }
        
        #endregion
        
        #region Visual Updates
        
        private void UpdatePowerVisuals(float powerLevel)
        {
            // Update arc reactor brightness
            if (arcReactorCore != null)
            {
                Color coreColor = Color.Lerp(Color.black, powerUpColor, powerLevel);
                arcReactorCore.color = coreColor;
            }
            
            // Update particle effects
            if (energyParticles != null)
            {
                var emission = energyParticles.emission;
                emission.rateOverTime = powerLevel * 50f;
                
                var main = energyParticles.main;
                main.startSpeed = powerLevel * 5f;
            }
            
            // Update light intensity
            if (reactorLight != null)
            {
                reactorLight.intensity = powerLevel * maxLightIntensity;
                reactorLight.color = Color.Lerp(powerDownColor, powerUpColor, powerLevel);
            }
        }
        
        private IEnumerator SpinReactor(bool spin)
        {
            while (spin && (isPoweredOn || currentSequence != null))
            {
                if (arcReactorCore != null)
                {
                    arcReactorCore.transform.Rotate(0, 0, reactorSpinSpeed * Time.deltaTime * currentPowerLevel);
                }
                
                // Counter-rotate rings
                for (int i = 0; i < arcReactorRings.Count; i++)
                {
                    if (arcReactorRings[i] != null)
                    {
                        float direction = i % 2 == 0 ? 1 : -1;
                        arcReactorRings[i].transform.Rotate(0, 0, 
                            reactorSpinSpeed * 0.5f * direction * Time.deltaTime * currentPowerLevel);
                    }
                }
                
                yield return null;
            }
        }
        
        #endregion
        
        #region Text Effects
        
        private IEnumerator PulseText(TextMeshProUGUI text, float duration)
        {
            float elapsed = 0;
            Color originalColor = text.color;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float alpha = Mathf.PingPong(elapsed * 2, 1);
                text.color = new Color(originalColor.r, originalColor.g, originalColor.b, alpha);
                yield return null;
            }
            
            text.color = originalColor;
        }
        
        private IEnumerator FlashText(TextMeshProUGUI text, float interval, int flashes)
        {
            for (int i = 0; i < flashes; i++)
            {
                text.enabled = !text.enabled;
                yield return new WaitForSeconds(interval);
            }
            text.enabled = true;
        }
        
        #endregion
        
        #region Utility
        
        private void PlaySound(AudioClip clip)
        {
            if (powerAudioSource != null && clip != null)
            {
                powerAudioSource.PlayOneShot(clip);
            }
        }
        
        #endregion
    }
}