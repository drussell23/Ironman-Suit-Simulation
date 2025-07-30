using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using System.Linq;

namespace IronManSim.Frontend
{
    /// <summary>
    /// JARVIS AI system for the Iron Man experience
    /// </summary>
    public class JARVISSystem : MonoBehaviour
    {
        [Header("JARVIS Configuration")]
        [SerializeField] private string jarvisName = "J.A.R.V.I.S.";
        [SerializeField] private float responseDelay = 0.5f;
        [SerializeField] private bool enableVoiceOutput = true;
        [SerializeField] private bool enableTextOutput = true;
        
        [Header("UI Elements")]
        [SerializeField] private GameObject jarvisUIContainer;
        [SerializeField] private TextMeshProUGUI jarvisText;
        [SerializeField] private TextMeshProUGUI subtitleText;
        [SerializeField] private Image jarvisAvatar;
        [SerializeField] private Image voiceWaveform;
        [SerializeField] private GameObject commandHistoryPanel;
        
        [Header("Voice Settings")]
        [SerializeField] private AudioSource voiceAudioSource;
        [SerializeField] private List<AudioClip> greetingSounds;
        [SerializeField] private List<AudioClip> alertSounds;
        [SerializeField] private List<AudioClip> confirmationSounds;
        
        [Header("Animation")]
        [SerializeField] private float waveformSpeed = 2f;
        [SerializeField] private float textTypeSpeed = 50f;
        [SerializeField] private AnimationCurve waveformCurve;
        
        // Command processing
        private Dictionary<string, System.Action<string[]>> commandHandlers;
        private List<string> commandHistory = new List<string>();
        private Queue<JARVISMessage> messageQueue = new Queue<JARVISMessage>();
        private Coroutine messageProcessingCoroutine;
        
        // State
        private bool isInitialized = false;
        private bool isSpeaking = false;
        private IronManExperienceManager.ExperienceMode currentMode;
        
        public class JARVISMessage
        {
            public string text;
            public MessageType type;
            public AudioClip audioClip;
            public float priority;
            
            public enum MessageType
            {
                Info,
                Alert,
                Warning,
                Response,
                System
            }
        }
        
        void Awake()
        {
            InitializeCommandHandlers();
        }
        
        #region Initialization
        
        public void Initialize()
        {
            if (isInitialized) return;
            
            SetupUI();
            
            // Start message processing
            messageProcessingCoroutine = StartCoroutine(ProcessMessageQueue());
            
            isInitialized = true;
            
            // Initial greeting
            Speak("Systems initialized. Good evening, sir.", JARVISMessage.MessageType.System);
        }
        
        private void SetupUI()
        {
            if (jarvisUIContainer == null)
            {
                CreateJARVISUI();
            }
            
            // Start waveform animation
            if (voiceWaveform != null)
            {
                StartCoroutine(AnimateWaveform());
            }
        }
        
        private void CreateJARVISUI()
        {
            // Create JARVIS UI container
            jarvisUIContainer = new GameObject("JARVIS UI");
            jarvisUIContainer.transform.SetParent(IronManExperienceManager.Instance.transform);
            
            // Add components
            RectTransform rect = jarvisUIContainer.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0, 0.7f);
            rect.anchorMax = new Vector2(0.3f, 1f);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            
            // Create text display
            GameObject textObj = new GameObject("JARVIS Text");
            textObj.transform.SetParent(jarvisUIContainer.transform);
            jarvisText = textObj.AddComponent<TextMeshProUGUI>();
            jarvisText.text = "";
            jarvisText.fontSize = 18;
            jarvisText.color = new Color(0.2f, 0.8f, 1f);
            
            RectTransform textRect = textObj.GetComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = new Vector2(20, 20);
            textRect.offsetMax = new Vector2(-20, -20);
        }
        
        #endregion
        
        #region Command Handling
        
        private void InitializeCommandHandlers()
        {
            commandHandlers = new Dictionary<string, System.Action<string[]>>
            {
                { "status", HandleStatusCommand },
                { "scan", HandleScanCommand },
                { "target", HandleTargetCommand },
                { "power", HandlePowerCommand },
                { "weapons", HandleWeaponsCommand },
                { "navigation", HandleNavigationCommand },
                { "emergency", HandleEmergencyCommand },
                { "analysis", HandleAnalysisCommand },
                { "mission", HandleMissionCommand },
                { "help", HandleHelpCommand }
            };
        }
        
        public void ProcessVoiceCommand(string command)
        {
            if (string.IsNullOrEmpty(command)) return;
            
            // Add to history
            commandHistory.Add(command);
            if (commandHistory.Count > 50)
            {
                commandHistory.RemoveAt(0);
            }
            
            // Parse command
            string[] parts = command.ToLower().Split(' ');
            string mainCommand = parts[0];
            
            // Find handler
            if (commandHandlers.ContainsKey(mainCommand))
            {
                commandHandlers[mainCommand](parts);
            }
            else
            {
                // Try to find partial match
                var matchingCommand = commandHandlers.Keys.FirstOrDefault(k => k.StartsWith(mainCommand));
                if (matchingCommand != null)
                {
                    commandHandlers[matchingCommand](parts);
                }
                else
                {
                    // Use AI to interpret
                    InterpretCommand(command);
                }
            }
        }
        
        private void InterpretCommand(string command)
        {
            // Simple interpretation logic
            command = command.ToLower();
            
            if (command.Contains("hello") || command.Contains("hi"))
            {
                Speak("Hello, sir. How may I assist you?", JARVISMessage.MessageType.Response);
            }
            else if (command.Contains("time"))
            {
                Speak($"The current time is {System.DateTime.Now:h:mm tt}", JARVISMessage.MessageType.Response);
            }
            else if (command.Contains("weather"))
            {
                Speak("Weather data is currently unavailable in simulation mode.", JARVISMessage.MessageType.Response);
            }
            else if (command.Contains("suit") && command.Contains("status"))
            {
                HandleStatusCommand(new string[] { "status" });
            }
            else if (command.Contains("fire") || command.Contains("shoot"))
            {
                Speak("Weapons systems require manual authorization.", JARVISMessage.MessageType.Alert);
            }
            else
            {
                Speak($"I'm not sure how to interpret '{command}'. Try saying 'help' for available commands.", 
                    JARVISMessage.MessageType.Response);
            }
        }
        
        #endregion
        
        #region Command Handlers
        
        private void HandleStatusCommand(string[] args)
        {
            var status = new SuitStatus();
            string statusReport = $"Suit status: Power at {status.powerLevel}%, " +
                                $"armor integrity {status.armorIntegrity}%, " +
                                $"core temperature {status.coreTemperature}°C. " +
                                $"All systems nominal.";
            
            Speak(statusReport, JARVISMessage.MessageType.Info);
            IronManExperienceManager.Instance.UpdateSuitStatus(status);
        }
        
        private void HandleScanCommand(string[] args)
        {
            if (args.Length > 1 && args[1] == "area")
            {
                Speak("Initiating area scan. Stand by...", JARVISMessage.MessageType.System);
                IronManExperienceManager.Instance.SetExperienceMode(IronManExperienceManager.ExperienceMode.Analysis);
                
                // Simulate scan
                StartCoroutine(PerformAreaScan());
            }
            else
            {
                Speak("Specify scan type: 'scan area', 'scan target', or 'scan systems'", 
                    JARVISMessage.MessageType.Response);
            }
        }
        
        private void HandleTargetCommand(string[] args)
        {
            if (args.Length > 1)
            {
                if (args[1] == "lock")
                {
                    Speak("Target lock engaged. Awaiting fire command.", JARVISMessage.MessageType.Alert);
                }
                else if (args[1] == "release")
                {
                    Speak("Target lock released.", JARVISMessage.MessageType.Info);
                }
            }
        }
        
        private void HandlePowerCommand(string[] args)
        {
            if (args.Length > 1)
            {
                if (args[1] == "divert" && args.Length > 3)
                {
                    string from = args[2];
                    string to = args[3];
                    Speak($"Diverting power from {from} to {to} systems.", JARVISMessage.MessageType.System);
                }
                else if (args[1] == "boost")
                {
                    Speak("Engaging power boost. Arc reactor output increased to 120%.", 
                        JARVISMessage.MessageType.Alert);
                }
            }
        }
        
        private void HandleWeaponsCommand(string[] args)
        {
            if (args.Length > 1)
            {
                if (args[1] == "online")
                {
                    Speak("Weapons systems online. Repulsors charged.", JARVISMessage.MessageType.System);
                }
                else if (args[1] == "offline")
                {
                    Speak("Disabling weapons systems.", JARVISMessage.MessageType.System);
                }
            }
        }
        
        private void HandleNavigationCommand(string[] args)
        {
            Speak("Navigation system ready. Awaiting destination coordinates.", JARVISMessage.MessageType.Response);
        }
        
        private void HandleEmergencyCommand(string[] args)
        {
            Speak("EMERGENCY PROTOCOL ACTIVATED!", JARVISMessage.MessageType.Alert);
            IronManExperienceManager.Instance.SetExperienceMode(IronManExperienceManager.ExperienceMode.Emergency);
        }
        
        private void HandleAnalysisCommand(string[] args)
        {
            Speak("Switching to analysis mode.", JARVISMessage.MessageType.System);
            IronManExperienceManager.Instance.SetExperienceMode(IronManExperienceManager.ExperienceMode.Analysis);
        }
        
        private void HandleMissionCommand(string[] args)
        {
            if (args.Length > 1)
            {
                if (args[1] == "start")
                {
                    Speak("Mission interface activated. Select mission parameters.", JARVISMessage.MessageType.System);
                    IronManExperienceManager.Instance.SetExperienceMode(IronManExperienceManager.ExperienceMode.Mission);
                }
                else if (args[1] == "abort")
                {
                    Speak("Mission aborted. Returning to standard operations.", JARVISMessage.MessageType.Alert);
                    IronManExperienceManager.Instance.EndMission();
                }
            }
        }
        
        private void HandleHelpCommand(string[] args)
        {
            string helpText = "Available commands: status, scan, target, power, weapons, " +
                            "navigation, emergency, analysis, mission. " +
                            "Say 'help [command]' for specific information.";
            
            Speak(helpText, JARVISMessage.MessageType.Info);
        }
        
        #endregion
        
        #region Speech System
        
        public void Speak(string text, JARVISMessage.MessageType type = JARVISMessage.MessageType.Info)
        {
            JARVISMessage message = new JARVISMessage
            {
                text = text,
                type = type,
                priority = GetPriorityForType(type)
            };
            
            // Select appropriate audio
            switch (type)
            {
                case JARVISMessage.MessageType.Alert:
                case JARVISMessage.MessageType.Warning:
                    if (alertSounds.Count > 0)
                    {
                        message.audioClip = alertSounds[Random.Range(0, alertSounds.Count)];
                    }
                    break;
                case JARVISMessage.MessageType.Response:
                    if (confirmationSounds.Count > 0)
                    {
                        message.audioClip = confirmationSounds[Random.Range(0, confirmationSounds.Count)];
                    }
                    break;
            }
            
            messageQueue.Enqueue(message);
        }
        
        private float GetPriorityForType(JARVISMessage.MessageType type)
        {
            switch (type)
            {
                case JARVISMessage.MessageType.Alert:
                    return 1f;
                case JARVISMessage.MessageType.Warning:
                    return 0.8f;
                case JARVISMessage.MessageType.System:
                    return 0.6f;
                case JARVISMessage.MessageType.Response:
                    return 0.4f;
                default:
                    return 0.2f;
            }
        }
        
        private IEnumerator ProcessMessageQueue()
        {
            while (true)
            {
                if (messageQueue.Count > 0 && !isSpeaking)
                {
                    JARVISMessage message = messageQueue.Dequeue();
                    yield return StartCoroutine(SpeakMessage(message));
                }
                
                yield return new WaitForSeconds(0.1f);
            }
        }
        
        private IEnumerator SpeakMessage(JARVISMessage message)
        {
            isSpeaking = true;
            
            // Play audio if available
            if (enableVoiceOutput && voiceAudioSource != null && message.audioClip != null)
            {
                voiceAudioSource.PlayOneShot(message.audioClip);
            }
            
            // Display text
            if (enableTextOutput && jarvisText != null)
            {
                yield return StartCoroutine(TypewriterEffect(message.text));
            }
            
            // Update subtitle
            if (subtitleText != null)
            {
                subtitleText.text = message.text;
            }
            
            // Wait for completion
            float speakDuration = message.text.Length / textTypeSpeed + 1f;
            yield return new WaitForSeconds(speakDuration);
            
            isSpeaking = false;
        }
        
        private IEnumerator TypewriterEffect(string text)
        {
            jarvisText.text = "";
            
            foreach (char letter in text)
            {
                jarvisText.text += letter;
                yield return new WaitForSeconds(1f / textTypeSpeed);
            }
        }
        
        #endregion
        
        #region Animations
        
        private IEnumerator AnimateWaveform()
        {
            if (voiceWaveform == null) yield break;
            
            Material waveformMat = voiceWaveform.material;
            float time = 0;
            
            while (true)
            {
                time += Time.deltaTime * waveformSpeed;
                
                // Animate based on speaking state
                float amplitude = isSpeaking ? 1f : 0.2f;
                float frequency = isSpeaking ? 10f : 2f;
                
                // Update shader properties or scale
                float wave = Mathf.Sin(time * frequency) * amplitude;
                voiceWaveform.transform.localScale = new Vector3(1f + wave * 0.1f, 1f + wave * 0.2f, 1f);
                
                // Update color intensity
                Color color = voiceWaveform.color;
                color.a = 0.5f + wave * 0.5f;
                voiceWaveform.color = color;
                
                yield return null;
            }
        }
        
        #endregion
        
        #region Simulation Methods
        
        private IEnumerator PerformAreaScan()
        {
            yield return new WaitForSeconds(2f);
            
            Speak("Scan complete. Detected: 3 heat signatures, 2 vehicles, " +
                  "no immediate threats. Environmental conditions stable.", 
                  JARVISMessage.MessageType.Info);
            
            // Return to HUD mode
            yield return new WaitForSeconds(3f);
            IronManExperienceManager.Instance.SetExperienceMode(IronManExperienceManager.ExperienceMode.HUD);
        }
        
        #endregion
        
        #region Mode Handling
        
        public void OnModeChanged(IronManExperienceManager.ExperienceMode newMode)
        {
            currentMode = newMode;
            
            switch (newMode)
            {
                case IronManExperienceManager.ExperienceMode.Combat:
                    Speak("Combat mode engaged. Weapons hot.", JARVISMessage.MessageType.Alert);
                    break;
                    
                case IronManExperienceManager.ExperienceMode.Mission:
                    Speak("Mission parameters loaded. Good luck, sir.", JARVISMessage.MessageType.System);
                    break;
                    
                case IronManExperienceManager.ExperienceMode.Emergency:
                    Speak("Emergency protocols active. All non-critical systems disabled.", 
                        JARVISMessage.MessageType.Alert);
                    break;
            }
        }
        
        #endregion
    }
}