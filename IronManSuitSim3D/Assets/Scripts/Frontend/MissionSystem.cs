using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using System.Linq;

namespace IronManSim.Frontend
{
    /// <summary>
    /// Manages mission simulations and objectives for the Iron Man experience
    /// </summary>
    public class MissionSystem : MonoBehaviour
    {
        [Header("Mission Configuration")]
        [SerializeField] private List<MissionTemplate> availableMissions;
        [SerializeField] private Mission currentMission;
        [SerializeField] private float missionUpdateInterval = 1f;
        
        [Header("UI Elements")]
        [SerializeField] private GameObject missionSelectionPanel;
        [SerializeField] private GameObject missionBriefingPanel;
        [SerializeField] private GameObject missionHUDPanel;
        [SerializeField] private GameObject missionCompletePanel;
        [SerializeField] private Transform missionListContainer;
        [SerializeField] private GameObject missionItemPrefab;
        
        [Header("Mission HUD")]
        [SerializeField] private TextMeshProUGUI missionNameText;
        [SerializeField] private TextMeshProUGUI missionTimerText;
        [SerializeField] private Transform objectiveListContainer;
        [SerializeField] private GameObject objectivePrefab;
        [SerializeField] private Slider missionProgressBar;
        [SerializeField] private TextMeshProUGUI missionStatusText;
        
        [Header("Briefing Elements")]
        [SerializeField] private TextMeshProUGUI briefingTitleText;
        [SerializeField] private TextMeshProUGUI briefingDescriptionText;
        [SerializeField] private Image briefingMapImage;
        [SerializeField] private Transform briefingObjectivesList;
        [SerializeField] private Button startMissionButton;
        [SerializeField] private Button abortMissionButton;
        
        [Header("Mission Types")]
        [SerializeField] private List<MissionScenario> combatScenarios;
        [SerializeField] private List<MissionScenario> rescueScenarios;
        [SerializeField] private List<MissionScenario> reconScenarios;
        [SerializeField] private List<MissionScenario> defenseScenarios;
        
        // Mission state
        private bool isMissionActive = false;
        private float missionStartTime;
        private Dictionary<string, ObjectiveUI> objectiveUIElements = new Dictionary<string, ObjectiveUI>();
        private Coroutine missionUpdateCoroutine;
        
        // Events
        public System.Action<Mission> OnMissionStarted;
        public System.Action<Mission> OnMissionCompleted;
        public System.Action<Mission> OnMissionFailed;
        public System.Action<Objective> OnObjectiveCompleted;
        
        [System.Serializable]
        public class Mission
        {
            public string id;
            public string name;
            public string description;
            public MissionType type;
            public float timeLimit; // 0 for unlimited
            public List<Objective> objectives;
            public MissionStatus status = MissionStatus.NotStarted;
            public float completionTime;
            public int score;
            public Dictionary<string, object> missionData = new Dictionary<string, object>();
        }
        
        [System.Serializable]
        public class Objective
        {
            public string id;
            public string description;
            public ObjectiveType type;
            public bool isRequired = true;
            public bool isCompleted = false;
            public float progress = 0f;
            public float targetValue = 1f;
            public Vector3 location;
            public GameObject targetObject;
        }
        
        [System.Serializable]
        public class MissionTemplate
        {
            public string templateId;
            public string name;
            public string description;
            public MissionType type;
            public Sprite thumbnail;
            public int difficulty;
            public float estimatedDuration;
            public List<string> requiredSystems;
        }
        
        [System.Serializable]
        public class MissionScenario
        {
            public string scenarioId;
            public GameObject scenarioPrefab;
            public List<Vector3> spawnPoints;
            public List<GameObject> enemyPrefabs;
            public List<GameObject> civilianPrefabs;
            public GameObject objectivePrefab;
        }
        
        public enum MissionType
        {
            Combat,
            Rescue,
            Recon,
            Defense,
            Training,
            Emergency
        }
        
        public enum MissionStatus
        {
            NotStarted,
            InProgress,
            Completed,
            Failed,
            Aborted
        }
        
        public enum ObjectiveType
        {
            Eliminate,
            Rescue,
            Reach,
            Defend,
            Scan,
            Survive,
            Collect,
            Escort
        }
        
        public class ObjectiveUI
        {
            public GameObject uiElement;
            public TextMeshProUGUI descriptionText;
            public Slider progressBar;
            public Image checkmark;
        }
        
        #region Initialization
        
        void Awake()
        {
            LoadMissionTemplates();
            SetupUI();
        }
        
        private void LoadMissionTemplates()
        {
            if (availableMissions == null || availableMissions.Count == 0)
            {
                // Create default mission templates
                availableMissions = new List<MissionTemplate>
                {
                    CreateMissionTemplate("mission_01", "Hostile Takeover", MissionType.Combat, 
                        "Eliminate hostile forces that have taken control of Stark Industries facility.", 3, 600f),
                    
                    CreateMissionTemplate("mission_02", "Disaster Relief", MissionType.Rescue,
                        "Respond to earthquake disaster zone and rescue trapped civilians.", 2, 900f),
                    
                    CreateMissionTemplate("mission_03", "Stealth Reconnaissance", MissionType.Recon,
                        "Infiltrate enemy base and gather intelligence without detection.", 4, 1200f),
                    
                    CreateMissionTemplate("mission_04", "City Defense", MissionType.Defense,
                        "Protect the city from incoming missile attack.", 5, 480f),
                    
                    CreateMissionTemplate("mission_05", "Training Simulation", MissionType.Training,
                        "Complete basic flight and combat training exercises.", 1, 300f)
                };
            }
        }
        
        private MissionTemplate CreateMissionTemplate(string id, string name, MissionType type, 
            string description, int difficulty, float duration)
        {
            return new MissionTemplate
            {
                templateId = id,
                name = name,
                type = type,
                description = description,
                difficulty = difficulty,
                estimatedDuration = duration,
                requiredSystems = GetRequiredSystemsForType(type)
            };
        }
        
        private List<string> GetRequiredSystemsForType(MissionType type)
        {
            List<string> systems = new List<string> { "HUD", "JARVIS" };
            
            switch (type)
            {
                case MissionType.Combat:
                    systems.AddRange(new[] { "Weapons", "Targeting", "Shields" });
                    break;
                case MissionType.Rescue:
                    systems.AddRange(new[] { "Sensors", "Medical" });
                    break;
                case MissionType.Recon:
                    systems.AddRange(new[] { "Stealth", "Sensors", "Recording" });
                    break;
                case MissionType.Defense:
                    systems.AddRange(new[] { "Weapons", "Shields", "Targeting" });
                    break;
            }
            
            return systems;
        }
        
        private void SetupUI()
        {
            if (startMissionButton != null)
            {
                startMissionButton.onClick.AddListener(StartSelectedMission);
            }
            
            if (abortMissionButton != null)
            {
                abortMissionButton.onClick.AddListener(AbortCurrentMission);
            }
        }
        
        #endregion
        
        #region Mission Selection
        
        public void ShowMissionInterface()
        {
            if (missionSelectionPanel == null)
            {
                CreateMissionSelectionUI();
            }
            
            PopulateMissionList();
            missionSelectionPanel.SetActive(true);
        }
        
        private void PopulateMissionList()
        {
            // Clear existing items
            foreach (Transform child in missionListContainer)
            {
                Destroy(child.gameObject);
            }
            
            // Create mission items
            foreach (var template in availableMissions)
            {
                CreateMissionListItem(template);
            }
        }
        
        private void CreateMissionListItem(MissionTemplate template)
        {
            GameObject item = missionItemPrefab != null ? 
                Instantiate(missionItemPrefab, missionListContainer) : 
                CreateDefaultMissionItem();
            
            // Set mission info
            TextMeshProUGUI nameText = item.GetComponentInChildren<TextMeshProUGUI>();
            if (nameText != null)
            {
                nameText.text = template.name;
            }
            
            // Add difficulty stars
            Transform difficultyContainer = item.transform.Find("Difficulty");
            if (difficultyContainer != null)
            {
                for (int i = 0; i < 5; i++)
                {
                    difficultyContainer.GetChild(i).gameObject.SetActive(i < template.difficulty);
                }
            }
            
            // Add click handler
            Button button = item.GetComponent<Button>();
            if (button != null)
            {
                button.onClick.AddListener(() => SelectMission(template));
            }
        }
        
        private GameObject CreateDefaultMissionItem()
        {
            GameObject item = new GameObject("Mission Item");
            item.transform.SetParent(missionListContainer);
            
            // Add components
            RectTransform rect = item.AddComponent<RectTransform>();
            rect.sizeDelta = new Vector2(400, 80);
            
            Button button = item.AddComponent<Button>();
            Image bg = item.AddComponent<Image>();
            bg.color = new Color(0.1f, 0.1f, 0.1f, 0.8f);
            
            // Add text
            GameObject textObj = new GameObject("Name");
            textObj.transform.SetParent(item.transform);
            TextMeshProUGUI text = textObj.AddComponent<TextMeshProUGUI>();
            text.text = "Mission";
            text.fontSize = 18;
            
            return item;
        }
        
        private void SelectMission(MissionTemplate template)
        {
            ShowMissionBriefing(template);
        }
        
        #endregion
        
        #region Mission Briefing
        
        private void ShowMissionBriefing(MissionTemplate template)
        {
            if (missionBriefingPanel == null)
            {
                CreateBriefingUI();
            }
            
            // Update briefing content
            if (briefingTitleText != null)
            {
                briefingTitleText.text = template.name;
            }
            
            if (briefingDescriptionText != null)
            {
                briefingDescriptionText.text = template.description;
            }
            
            // Generate mission from template
            currentMission = GenerateMission(template);
            
            // Show objectives
            PopulateBriefingObjectives();
            
            // Show briefing
            missionSelectionPanel.SetActive(false);
            missionBriefingPanel.SetActive(true);
        }
        
        private Mission GenerateMission(MissionTemplate template)
        {
            Mission mission = new Mission
            {
                id = System.Guid.NewGuid().ToString(),
                name = template.name,
                description = template.description,
                type = template.type,
                timeLimit = template.estimatedDuration,
                objectives = GenerateObjectives(template),
                status = MissionStatus.NotStarted
            };
            
            return mission;
        }
        
        private List<Objective> GenerateObjectives(MissionTemplate template)
        {
            List<Objective> objectives = new List<Objective>();
            
            switch (template.type)
            {
                case MissionType.Combat:
                    objectives.Add(CreateObjective("eliminate_primary", "Eliminate all hostile forces", 
                        ObjectiveType.Eliminate, true, 10f));
                    objectives.Add(CreateObjective("minimize_damage", "Minimize collateral damage", 
                        ObjectiveType.Defend, false));
                    break;
                    
                case MissionType.Rescue:
                    objectives.Add(CreateObjective("rescue_civilians", "Rescue trapped civilians", 
                        ObjectiveType.Rescue, true, 5f));
                    objectives.Add(CreateObjective("medical_assist", "Provide medical assistance", 
                        ObjectiveType.Reach, false));
                    break;
                    
                case MissionType.Recon:
                    objectives.Add(CreateObjective("scan_targets", "Scan target locations", 
                        ObjectiveType.Scan, true, 3f));
                    objectives.Add(CreateObjective("remain_undetected", "Remain undetected", 
                        ObjectiveType.Survive, true));
                    break;
                    
                case MissionType.Defense:
                    objectives.Add(CreateObjective("defend_position", "Defend the target area", 
                        ObjectiveType.Defend, true, template.estimatedDuration));
                    objectives.Add(CreateObjective("intercept_missiles", "Intercept incoming missiles", 
                        ObjectiveType.Eliminate, true, 20f));
                    break;
                    
                case MissionType.Training:
                    objectives.Add(CreateObjective("complete_course", "Complete obstacle course", 
                        ObjectiveType.Reach, true));
                    objectives.Add(CreateObjective("target_practice", "Hit all targets", 
                        ObjectiveType.Eliminate, true, 5f));
                    break;
            }
            
            return objectives;
        }
        
        private Objective CreateObjective(string id, string description, ObjectiveType type, 
            bool required, float target = 1f)
        {
            return new Objective
            {
                id = id,
                description = description,
                type = type,
                isRequired = required,
                targetValue = target,
                location = GenerateObjectiveLocation(type)
            };
        }
        
        private Vector3 GenerateObjectiveLocation(ObjectiveType type)
        {
            // Generate random location within mission area
            float range = 500f;
            return new Vector3(
                Random.Range(-range, range),
                Random.Range(50f, 200f),
                Random.Range(-range, range)
            );
        }
        
        private void PopulateBriefingObjectives()
        {
            if (briefingObjectivesList == null || currentMission == null) return;
            
            // Clear existing
            foreach (Transform child in briefingObjectivesList)
            {
                Destroy(child.gameObject);
            }
            
            // Add objectives
            foreach (var objective in currentMission.objectives)
            {
                GameObject objItem = new GameObject($"Objective_{objective.id}");
                objItem.transform.SetParent(briefingObjectivesList);
                
                TextMeshProUGUI text = objItem.AddComponent<TextMeshProUGUI>();
                text.text = $"• {objective.description}";
                text.fontSize = 16;
                text.color = objective.isRequired ? Color.white : Color.gray;
                
                if (objective.isRequired)
                {
                    text.text += " [REQUIRED]";
                }
            }
        }
        
        #endregion
        
        #region Mission Execution
        
        private void StartSelectedMission()
        {
            if (currentMission == null) return;
            
            // Hide briefing
            missionBriefingPanel.SetActive(false);
            
            // Start mission
            StartMission(currentMission);
        }
        
        public void StartMission(Mission mission)
        {
            currentMission = mission;
            currentMission.status = MissionStatus.InProgress;
            isMissionActive = true;
            missionStartTime = Time.time;
            
            // Setup mission environment
            SetupMissionEnvironment();
            
            // Show mission HUD
            ShowMissionHUD();
            
            // Start mission update loop
            missionUpdateCoroutine = StartCoroutine(MissionUpdateLoop());
            
            // Notify systems
            OnMissionStarted?.Invoke(currentMission);
            
            // JARVIS announcement
            IronManExperienceManager.Instance.jarvisSystem.Speak(
                $"Mission {currentMission.name} is now active. Good luck, sir.", 
                JARVISSystem.JARVISMessage.MessageType.System
            );
        }
        
        private void SetupMissionEnvironment()
        {
            // Spawn mission-specific elements based on type
            MissionScenario scenario = GetScenarioForMission(currentMission);
            
            if (scenario != null && scenario.scenarioPrefab != null)
            {
                GameObject missionEnvironment = Instantiate(scenario.scenarioPrefab);
                missionEnvironment.name = $"Mission_{currentMission.id}_Environment";
                
                // Spawn enemies
                if (currentMission.type == MissionType.Combat || currentMission.type == MissionType.Defense)
                {
                    SpawnEnemies(scenario);
                }
                
                // Spawn civilians
                if (currentMission.type == MissionType.Rescue)
                {
                    SpawnCivilians(scenario);
                }
                
                // Setup objective markers
                SetupObjectiveMarkers();
            }
        }
        
        private MissionScenario GetScenarioForMission(Mission mission)
        {
            List<MissionScenario> scenarios = null;
            
            switch (mission.type)
            {
                case MissionType.Combat:
                    scenarios = combatScenarios;
                    break;
                case MissionType.Rescue:
                    scenarios = rescueScenarios;
                    break;
                case MissionType.Recon:
                    scenarios = reconScenarios;
                    break;
                case MissionType.Defense:
                    scenarios = defenseScenarios;
                    break;
            }
            
            if (scenarios != null && scenarios.Count > 0)
            {
                return scenarios[Random.Range(0, scenarios.Count)];
            }
            
            return null;
        }
        
        private void SpawnEnemies(MissionScenario scenario)
        {
            if (scenario.enemyPrefabs == null || scenario.enemyPrefabs.Count == 0) return;
            
            foreach (var objective in currentMission.objectives.Where(o => o.type == ObjectiveType.Eliminate))
            {
                int enemyCount = Mathf.RoundToInt(objective.targetValue);
                List<GameObject> enemies = new List<GameObject>();
                
                for (int i = 0; i < enemyCount; i++)
                {
                    GameObject enemyPrefab = scenario.enemyPrefabs[Random.Range(0, scenario.enemyPrefabs.Count)];
                    Vector3 spawnPos = scenario.spawnPoints.Count > 0 ? 
                        scenario.spawnPoints[Random.Range(0, scenario.spawnPoints.Count)] : 
                        objective.location + Random.insideUnitSphere * 50f;
                    
                    GameObject enemy = Instantiate(enemyPrefab, spawnPos, Quaternion.identity);
                    enemy.tag = "Enemy";
                    enemies.Add(enemy);
                }
                
                // Store enemies for tracking
                currentMission.missionData[$"enemies_{objective.id}"] = enemies;
            }
        }
        
        private void SpawnCivilians(MissionScenario scenario)
        {
            if (scenario.civilianPrefabs == null || scenario.civilianPrefabs.Count == 0) return;
            
            foreach (var objective in currentMission.objectives.Where(o => o.type == ObjectiveType.Rescue))
            {
                int civilianCount = Mathf.RoundToInt(objective.targetValue);
                List<GameObject> civilians = new List<GameObject>();
                
                for (int i = 0; i < civilianCount; i++)
                {
                    GameObject civilianPrefab = scenario.civilianPrefabs[Random.Range(0, scenario.civilianPrefabs.Count)];
                    Vector3 spawnPos = objective.location + Random.insideUnitSphere * 30f;
                    
                    GameObject civilian = Instantiate(civilianPrefab, spawnPos, Quaternion.identity);
                    civilian.tag = "Civilian";
                    civilians.Add(civilian);
                }
                
                currentMission.missionData[$"civilians_{objective.id}"] = civilians;
            }
        }
        
        private void SetupObjectiveMarkers()
        {
            foreach (var objective in currentMission.objectives)
            {
                if (objective.location != Vector3.zero)
                {
                    // Add waypoint in HUD
                    IronManExperienceManager.Instance.hudManager.AddWaypoint(
                        objective.location, 
                        objective.description
                    );
                }
            }
        }
        
        #endregion
        
        #region Mission HUD
        
        private void ShowMissionHUD()
        {
            if (missionHUDPanel == null)
            {
                CreateMissionHUD();
            }
            
            missionHUDPanel.SetActive(true);
            
            // Update mission info
            if (missionNameText != null)
            {
                missionNameText.text = currentMission.name;
            }
            
            // Create objective UI elements
            CreateObjectiveUIElements();
            
            // Update initial status
            UpdateMissionHUD();
        }
        
        private void CreateObjectiveUIElements()
        {
            objectiveUIElements.Clear();
            
            // Clear existing
            foreach (Transform child in objectiveListContainer)
            {
                Destroy(child.gameObject);
            }
            
            // Create UI for each objective
            foreach (var objective in currentMission.objectives)
            {
                GameObject objUI = objectivePrefab != null ? 
                    Instantiate(objectivePrefab, objectiveListContainer) : 
                    CreateDefaultObjectiveUI();
                
                ObjectiveUI ui = new ObjectiveUI
                {
                    uiElement = objUI,
                    descriptionText = objUI.GetComponentInChildren<TextMeshProUGUI>(),
                    progressBar = objUI.GetComponentInChildren<Slider>(),
                    checkmark = objUI.transform.Find("Checkmark")?.GetComponent<Image>()
                };
                
                if (ui.descriptionText != null)
                {
                    ui.descriptionText.text = objective.description;
                }
                
                if (ui.progressBar != null)
                {
                    ui.progressBar.gameObject.SetActive(objective.targetValue > 1);
                    ui.progressBar.maxValue = objective.targetValue;
                    ui.progressBar.value = 0;
                }
                
                if (ui.checkmark != null)
                {
                    ui.checkmark.gameObject.SetActive(false);
                }
                
                objectiveUIElements[objective.id] = ui;
            }
        }
        
        private GameObject CreateDefaultObjectiveUI()
        {
            GameObject obj = new GameObject("Objective");
            obj.transform.SetParent(objectiveListContainer);
            
            RectTransform rect = obj.AddComponent<RectTransform>();
            rect.sizeDelta = new Vector2(300, 30);
            
            // Add text
            GameObject textObj = new GameObject("Text");
            textObj.transform.SetParent(obj.transform);
            TextMeshProUGUI text = textObj.AddComponent<TextMeshProUGUI>();
            text.fontSize = 14;
            
            return obj;
        }
        
        private void UpdateMissionHUD()
        {
            if (!isMissionActive) return;
            
            // Update timer
            if (currentMission.timeLimit > 0 && missionTimerText != null)
            {
                float timeRemaining = currentMission.timeLimit - (Time.time - missionStartTime);
                if (timeRemaining < 0) timeRemaining = 0;
                
                int minutes = Mathf.FloorToInt(timeRemaining / 60);
                int seconds = Mathf.FloorToInt(timeRemaining % 60);
                missionTimerText.text = $"{minutes:00}:{seconds:00}";
                
                // Color based on time
                if (timeRemaining < 60)
                {
                    missionTimerText.color = Color.red;
                }
                else if (timeRemaining < 180)
                {
                    missionTimerText.color = Color.yellow;
                }
            }
            
            // Update objectives
            foreach (var objective in currentMission.objectives)
            {
                if (objectiveUIElements.ContainsKey(objective.id))
                {
                    var ui = objectiveUIElements[objective.id];
                    
                    if (ui.progressBar != null && objective.targetValue > 1)
                    {
                        ui.progressBar.value = objective.progress;
                    }
                    
                    if (objective.isCompleted && ui.checkmark != null)
                    {
                        ui.checkmark.gameObject.SetActive(true);
                        if (ui.descriptionText != null)
                        {
                            ui.descriptionText.color = Color.green;
                        }
                    }
                }
            }
            
            // Update overall progress
            if (missionProgressBar != null)
            {
                float totalProgress = currentMission.objectives.Count(o => o.isCompleted);
                float requiredCount = currentMission.objectives.Count(o => o.isRequired);
                missionProgressBar.value = totalProgress / requiredCount;
            }
        }
        
        #endregion
        
        #region Mission Logic
        
        private IEnumerator MissionUpdateLoop()
        {
            while (isMissionActive && currentMission != null)
            {
                // Check objectives
                CheckObjectiveProgress();
                
                // Update HUD
                UpdateMissionHUD();
                
                // Check mission status
                CheckMissionCompletion();
                
                // Check time limit
                if (currentMission.timeLimit > 0)
                {
                    float elapsed = Time.time - missionStartTime;
                    if (elapsed > currentMission.timeLimit)
                    {
                        FailMission("Time limit exceeded");
                    }
                }
                
                yield return new WaitForSeconds(missionUpdateInterval);
            }
        }
        
        private void CheckObjectiveProgress()
        {
            foreach (var objective in currentMission.objectives)
            {
                if (objective.isCompleted) continue;
                
                switch (objective.type)
                {
                    case ObjectiveType.Eliminate:
                        CheckEliminateObjective(objective);
                        break;
                        
                    case ObjectiveType.Rescue:
                        CheckRescueObjective(objective);
                        break;
                        
                    case ObjectiveType.Reach:
                        CheckReachObjective(objective);
                        break;
                        
                    case ObjectiveType.Defend:
                        CheckDefendObjective(objective);
                        break;
                        
                    case ObjectiveType.Scan:
                        CheckScanObjective(objective);
                        break;
                        
                    case ObjectiveType.Survive:
                        CheckSurviveObjective(objective);
                        break;
                }
            }
        }
        
        private void CheckEliminateObjective(Objective objective)
        {
            string enemyKey = $"enemies_{objective.id}";
            if (currentMission.missionData.ContainsKey(enemyKey))
            {
                List<GameObject> enemies = currentMission.missionData[enemyKey] as List<GameObject>;
                if (enemies != null)
                {
                    // Count destroyed enemies
                    int destroyed = enemies.Count(e => e == null || !e.activeInHierarchy);
                    objective.progress = destroyed;
                    
                    if (destroyed >= objective.targetValue)
                    {
                        CompleteObjective(objective);
                    }
                }
            }
        }
        
        private void CheckRescueObjective(Objective objective)
        {
            string civilianKey = $"civilians_{objective.id}";
            if (currentMission.missionData.ContainsKey(civilianKey))
            {
                List<GameObject> civilians = currentMission.missionData[civilianKey] as List<GameObject>;
                if (civilians != null)
                {
                    // Check rescued civilians (simplified - check if they're near the player)
                    int rescued = 0;
                    foreach (var civilian in civilians.Where(c => c != null))
                    {
                        float distance = Vector3.Distance(
                            Camera.main.transform.position, 
                            civilian.transform.position
                        );
                        
                        if (distance < 10f)
                        {
                            rescued++;
                            civilian.SetActive(false); // Simple rescue
                        }
                    }
                    
                    objective.progress = rescued;
                    
                    if (rescued >= objective.targetValue)
                    {
                        CompleteObjective(objective);
                    }
                }
            }
        }
        
        private void CheckReachObjective(Objective objective)
        {
            if (objective.location != Vector3.zero)
            {
                float distance = Vector3.Distance(Camera.main.transform.position, objective.location);
                if (distance < 20f)
                {
                    CompleteObjective(objective);
                }
            }
        }
        
        private void CheckDefendObjective(Objective objective)
        {
            // For defense objectives, track time survived
            objective.progress = Time.time - missionStartTime;
            
            if (objective.progress >= objective.targetValue)
            {
                CompleteObjective(objective);
            }
        }
        
        private void CheckScanObjective(Objective objective)
        {
            // Simplified scan check - would integrate with analysis mode
            if (IronManExperienceManager.Instance.currentMode == 
                IronManExperienceManager.ExperienceMode.Analysis)
            {
                objective.progress += Time.deltaTime;
                
                if (objective.progress >= objective.targetValue)
                {
                    CompleteObjective(objective);
                }
            }
        }
        
        private void CheckSurviveObjective(Objective objective)
        {
            // Check if player is still "alive"
            // For now, always true unless mission failed
            objective.progress = 1f;
        }
        
        private void CompleteObjective(Objective objective)
        {
            objective.isCompleted = true;
            objective.progress = objective.targetValue;
            
            // Notify
            OnObjectiveCompleted?.Invoke(objective);
            
            // JARVIS notification
            IronManExperienceManager.Instance.jarvisSystem.Speak(
                $"Objective complete: {objective.description}", 
                JARVISSystem.JARVISMessage.MessageType.Info
            );
            
            // Visual/audio feedback
            PlayObjectiveCompleteEffect();
        }
        
        private void CheckMissionCompletion()
        {
            // Check if all required objectives are complete
            bool allRequiredComplete = currentMission.objectives
                .Where(o => o.isRequired)
                .All(o => o.isCompleted);
            
            if (allRequiredComplete)
            {
                CompleteMission();
            }
        }
        
        #endregion
        
        #region Mission Completion
        
        private void CompleteMission()
        {
            if (!isMissionActive) return;
            
            isMissionActive = false;
            currentMission.status = MissionStatus.Completed;
            currentMission.completionTime = Time.time - missionStartTime;
            
            // Calculate score
            CalculateMissionScore();
            
            // Stop update loop
            if (missionUpdateCoroutine != null)
            {
                StopCoroutine(missionUpdateCoroutine);
            }
            
            // Show completion screen
            ShowMissionComplete();
            
            // Notify
            OnMissionCompleted?.Invoke(currentMission);
            
            // JARVIS congratulations
            IronManExperienceManager.Instance.jarvisSystem.Speak(
                $"Excellent work, sir. Mission {currentMission.name} completed successfully.", 
                JARVISSystem.JARVISMessage.MessageType.System
            );
        }
        
        private void FailMission(string reason)
        {
            if (!isMissionActive) return;
            
            isMissionActive = false;
            currentMission.status = MissionStatus.Failed;
            
            // Stop update loop
            if (missionUpdateCoroutine != null)
            {
                StopCoroutine(missionUpdateCoroutine);
            }
            
            // Show failure screen
            ShowMissionFailed(reason);
            
            // Notify
            OnMissionFailed?.Invoke(currentMission);
            
            // JARVIS notification
            IronManExperienceManager.Instance.jarvisSystem.Speak(
                $"Mission failed: {reason}", 
                JARVISSystem.JARVISMessage.MessageType.Alert
            );
        }
        
        public void AbortCurrentMission()
        {
            if (!isMissionActive || currentMission == null) return;
            
            isMissionActive = false;
            currentMission.status = MissionStatus.Aborted;
            
            // Cleanup
            CleanupMission();
            
            // Return to HUD
            IronManExperienceManager.Instance.SetExperienceMode(
                IronManExperienceManager.ExperienceMode.HUD
            );
        }
        
        private void CalculateMissionScore()
        {
            int baseScore = 1000;
            
            // Time bonus
            if (currentMission.timeLimit > 0)
            {
                float timeRatio = currentMission.completionTime / currentMission.timeLimit;
                int timeBonus = Mathf.RoundToInt((1f - timeRatio) * 500);
                baseScore += Mathf.Max(0, timeBonus);
            }
            
            // Optional objectives
            int optionalCompleted = currentMission.objectives
                .Count(o => !o.isRequired && o.isCompleted);
            baseScore += optionalCompleted * 200;
            
            // Accuracy/efficiency bonuses would go here
            
            currentMission.score = baseScore;
        }
        
        #endregion
        
        #region UI Display
        
        private void ShowMissionComplete()
        {
            if (missionCompletePanel == null)
            {
                CreateCompletionUI();
            }
            
            // Hide mission HUD
            missionHUDPanel.SetActive(false);
            
            // Show completion
            missionCompletePanel.SetActive(true);
            
            // Update completion info
            Transform content = missionCompletePanel.transform;
            
            TextMeshProUGUI titleText = content.Find("Title")?.GetComponent<TextMeshProUGUI>();
            if (titleText != null)
            {
                titleText.text = "MISSION COMPLETE";
            }
            
            TextMeshProUGUI scoreText = content.Find("Score")?.GetComponent<TextMeshProUGUI>();
            if (scoreText != null)
            {
                scoreText.text = $"Score: {currentMission.score}";
            }
            
            TextMeshProUGUI timeText = content.Find("Time")?.GetComponent<TextMeshProUGUI>();
            if (timeText != null)
            {
                int minutes = Mathf.FloorToInt(currentMission.completionTime / 60);
                int seconds = Mathf.FloorToInt(currentMission.completionTime % 60);
                timeText.text = $"Time: {minutes:00}:{seconds:00}";
            }
            
            // Auto-hide after delay
            StartCoroutine(AutoHideCompletion());
        }
        
        private void ShowMissionFailed(string reason)
        {
            if (missionCompletePanel == null)
            {
                CreateCompletionUI();
            }
            
            // Hide mission HUD
            missionHUDPanel.SetActive(false);
            
            // Show failure
            missionCompletePanel.SetActive(true);
            
            // Update failure info
            Transform content = missionCompletePanel.transform;
            
            TextMeshProUGUI titleText = content.Find("Title")?.GetComponent<TextMeshProUGUI>();
            if (titleText != null)
            {
                titleText.text = "MISSION FAILED";
                titleText.color = Color.red;
            }
            
            TextMeshProUGUI reasonText = content.Find("Score")?.GetComponent<TextMeshProUGUI>();
            if (reasonText != null)
            {
                reasonText.text = reason;
            }
            
            // Auto-hide after delay
            StartCoroutine(AutoHideCompletion());
        }
        
        private IEnumerator AutoHideCompletion()
        {
            yield return new WaitForSeconds(5f);
            
            missionCompletePanel.SetActive(false);
            
            // Cleanup
            CleanupMission();
            
            // Return to HUD
            IronManExperienceManager.Instance.SetExperienceMode(
                IronManExperienceManager.ExperienceMode.HUD
            );
        }
        
        #endregion
        
        #region Cleanup
        
        private void CleanupMission()
        {
            // Remove spawned objects
            GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
            foreach (var enemy in enemies)
            {
                Destroy(enemy);
            }
            
            GameObject[] civilians = GameObject.FindGameObjectsWithTag("Civilian");
            foreach (var civilian in civilians)
            {
                Destroy(civilian);
            }
            
            // Clear mission data
            if (currentMission != null)
            {
                currentMission.missionData.Clear();
            }
            
            // Clear UI
            objectiveUIElements.Clear();
            
            // Hide panels
            if (missionHUDPanel != null)
            {
                missionHUDPanel.SetActive(false);
            }
        }
        
        #endregion
        
        #region Effects
        
        private void PlayObjectiveCompleteEffect()
        {
            // Visual effect
            IronManExperienceManager.Instance.hudManager.ApplyGlitchEffect(0.2f);
            
            // Audio would go here
        }
        
        #endregion
        
        #region UI Creation
        
        private void CreateMissionSelectionUI()
        {
            missionSelectionPanel = new GameObject("Mission Selection");
            missionSelectionPanel.transform.SetParent(IronManExperienceManager.Instance.hudCanvas.transform);
            
            RectTransform rect = missionSelectionPanel.AddComponent<RectTransform>();
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.one;
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            
            // Add background
            Image bg = missionSelectionPanel.AddComponent<Image>();
            bg.color = new Color(0, 0, 0, 0.8f);
            
            // Create list container
            missionListContainer = new GameObject("Mission List").transform;
            missionListContainer.SetParent(missionSelectionPanel.transform);
        }
        
        private void CreateBriefingUI()
        {
            missionBriefingPanel = new GameObject("Mission Briefing");
            missionBriefingPanel.transform.SetParent(IronManExperienceManager.Instance.hudCanvas.transform);
            
            RectTransform rect = missionBriefingPanel.AddComponent<RectTransform>();
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.one;
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            
            // Add components
            briefingTitleText = CreateTextElement("Title", missionBriefingPanel.transform);
            briefingDescriptionText = CreateTextElement("Description", missionBriefingPanel.transform);
            
            // Create start button
            GameObject buttonObj = new GameObject("Start Button");
            buttonObj.transform.SetParent(missionBriefingPanel.transform);
            startMissionButton = buttonObj.AddComponent<Button>();
            Image buttonBg = buttonObj.AddComponent<Image>();
            buttonBg.color = new Color(0.2f, 0.8f, 0.2f);
            
            TextMeshProUGUI buttonText = CreateTextElement("Text", buttonObj.transform);
            buttonText.text = "START MISSION";
        }
        
        private void CreateMissionHUD()
        {
            missionHUDPanel = new GameObject("Mission HUD");
            missionHUDPanel.transform.SetParent(IronManExperienceManager.Instance.hudCanvas.transform);
            
            RectTransform rect = missionHUDPanel.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0, 0.7f);
            rect.anchorMax = new Vector2(0.3f, 1f);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            
            // Create elements
            missionNameText = CreateTextElement("Mission Name", missionHUDPanel.transform);
            missionTimerText = CreateTextElement("Timer", missionHUDPanel.transform);
            
            // Objective list
            objectiveListContainer = new GameObject("Objectives").transform;
            objectiveListContainer.SetParent(missionHUDPanel.transform);
        }
        
        private void CreateCompletionUI()
        {
            missionCompletePanel = new GameObject("Mission Complete");
            missionCompletePanel.transform.SetParent(IronManExperienceManager.Instance.hudCanvas.transform);
            
            RectTransform rect = missionCompletePanel.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.3f, 0.3f);
            rect.anchorMax = new Vector2(0.7f, 0.7f);
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
            
            // Add background
            Image bg = missionCompletePanel.AddComponent<Image>();
            bg.color = new Color(0, 0, 0, 0.9f);
            
            // Add text elements
            CreateTextElement("Title", missionCompletePanel.transform);
            CreateTextElement("Score", missionCompletePanel.transform);
            CreateTextElement("Time", missionCompletePanel.transform);
        }
        
        private TextMeshProUGUI CreateTextElement(string name, Transform parent)
        {
            GameObject textObj = new GameObject(name);
            textObj.transform.SetParent(parent);
            
            TextMeshProUGUI text = textObj.AddComponent<TextMeshProUGUI>();
            text.fontSize = 18;
            text.color = Color.white;
            
            return text;
        }
        
        #endregion
        
        #region Public API
        
        public void LoadMission(string missionId)
        {
            MissionTemplate template = availableMissions.FirstOrDefault(m => m.templateId == missionId);
            if (template != null)
            {
                ShowMissionBriefing(template);
            }
        }
        
        public void EndCurrentMission()
        {
            if (isMissionActive)
            {
                AbortCurrentMission();
            }
        }
        
        public Mission GetCurrentMission()
        {
            return currentMission;
        }
        
        public bool IsMissionActive()
        {
            return isMissionActive;
        }
        
        #endregion
    }
}