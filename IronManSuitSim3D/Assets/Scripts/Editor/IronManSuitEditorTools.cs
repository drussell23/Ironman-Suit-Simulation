using UnityEngine;
using UnityEditor;
using IronManSim.Models;

namespace IronManSim.Editor
{
    /// <summary>
    /// Custom editor tools for Iron Man suit creation and setup
    /// </summary>
    public class IronManSuitEditorTools : EditorWindow
    {
        private IronManSuitModelGenerator.SuitModel selectedModel = IronManSuitModelGenerator.SuitModel.Mark85;
        private Color primaryColor = new Color(0.8f, 0.1f, 0.1f);
        private Color secondaryColor = new Color(0.9f, 0.75f, 0.1f);
        private Color accentColor = new Color(0.2f, 0.8f, 1f);
        
        private GameObject currentSuit;
        private IronManSuitModelGenerator generator;
        private IronManSuitRig rig;
        private SuitComponentLibrary componentLibrary;
        
        private bool showModelSettings = true;
        private bool showRigSettings = true;
        private bool showComponentSettings = true;
        private bool showMaterialSettings = true;
        
        private Vector2 scrollPos;
        
        [MenuItem("IronMan/Suit Editor")]
        public static void ShowWindow()
        {
            GetWindow<IronManSuitEditorTools>("Iron Man Suit Editor");
        }
        
        void OnGUI()
        {
            scrollPos = EditorGUILayout.BeginScrollView(scrollPos);
            
            // Header
            EditorGUILayout.Space();
            GUILayout.Label("Iron Man Suit Editor", EditorStyles.boldLabel);
            EditorGUILayout.Space();
            
            // Model Generation Section
            showModelSettings = EditorGUILayout.BeginFoldoutHeaderGroup(showModelSettings, "Model Generation");
            if (showModelSettings)
            {
                DrawModelGenerationSection();
            }
            EditorGUILayout.EndFoldoutHeaderGroup();
            
            EditorGUILayout.Space();
            
            // Rig Setup Section
            showRigSettings = EditorGUILayout.BeginFoldoutHeaderGroup(showRigSettings, "Rig Setup");
            if (showRigSettings)
            {
                DrawRigSetupSection();
            }
            EditorGUILayout.EndFoldoutHeaderGroup();
            
            EditorGUILayout.Space();
            
            // Component Library Section
            showComponentSettings = EditorGUILayout.BeginFoldoutHeaderGroup(showComponentSettings, "Component Library");
            if (showComponentSettings)
            {
                DrawComponentLibrarySection();
            }
            EditorGUILayout.EndFoldoutHeaderGroup();
            
            EditorGUILayout.Space();
            
            // Material Settings Section
            showMaterialSettings = EditorGUILayout.BeginFoldoutHeaderGroup(showMaterialSettings, "Material Settings");
            if (showMaterialSettings)
            {
                DrawMaterialSettingsSection();
            }
            EditorGUILayout.EndFoldoutHeaderGroup();
            
            EditorGUILayout.EndScrollView();
        }
        
        #region Model Generation
        
        private void DrawModelGenerationSection()
        {
            EditorGUI.indentLevel++;
            
            // Model selection
            selectedModel = (IronManSuitModelGenerator.SuitModel)EditorGUILayout.EnumPopup("Suit Model", selectedModel);
            
            // Color settings
            primaryColor = EditorGUILayout.ColorField("Primary Color (Red)", primaryColor);
            secondaryColor = EditorGUILayout.ColorField("Secondary Color (Gold)", secondaryColor);
            accentColor = EditorGUILayout.ColorField("Accent Color (Arc Reactor)", accentColor);
            
            EditorGUILayout.Space();
            
            // Generation buttons
            EditorGUILayout.BeginHorizontal();
            
            if (GUILayout.Button("Generate New Suit", GUILayout.Height(30)))
            {
                GenerateNewSuit();
            }
            
            if (currentSuit != null)
            {
                if (GUILayout.Button("Update Current Suit", GUILayout.Height(30)))
                {
                    UpdateCurrentSuit();
                }
            }
            
            EditorGUILayout.EndHorizontal();
            
            if (currentSuit != null)
            {
                EditorGUILayout.Space();
                EditorGUILayout.HelpBox($"Current Suit: {currentSuit.name}", MessageType.Info);
                
                if (GUILayout.Button("Select Suit"))
                {
                    Selection.activeGameObject = currentSuit;
                }
            }
            
            EditorGUI.indentLevel--;
        }
        
        private void GenerateNewSuit()
        {
            // Create suit object
            GameObject suitObject = new GameObject("IronManSuit");
            currentSuit = suitObject;
            
            // Add generator component
            generator = suitObject.AddComponent<IronManSuitModelGenerator>();
            
            // Configure generator
            var config = new IronManSuitModelGenerator.SuitConfiguration
            {
                suitModel = selectedModel,
                primaryColor = primaryColor,
                secondaryColor = secondaryColor,
                accentColor = accentColor,
                modelScale = 1.8f,
                meshResolution = 32,
                generateColliders = true,
                generateRigidBodies = true,
                generatePanelLines = true,
                generateRivets = true,
                generateServos = true
            };
            
            // Apply configuration (would need to expose this in the generator)
            generator.GenerateSuit();
            
            // Select the created suit
            Selection.activeGameObject = suitObject;
            
            EditorUtility.DisplayDialog("Suit Generated", 
                $"Successfully generated {selectedModel} suit!", "OK");
        }
        
        private void UpdateCurrentSuit()
        {
            if (generator != null)
            {
                generator.GenerateSuit();
                EditorUtility.DisplayDialog("Suit Updated", 
                    "Suit model has been updated with new settings.", "OK");
            }
        }
        
        #endregion
        
        #region Rig Setup
        
        private void DrawRigSetupSection()
        {
            EditorGUI.indentLevel++;
            
            if (currentSuit == null)
            {
                EditorGUILayout.HelpBox("Generate a suit first to set up rigging.", MessageType.Warning);
                EditorGUI.indentLevel--;
                return;
            }
            
            rig = currentSuit.GetComponent<IronManSuitRig>();
            
            if (rig == null)
            {
                if (GUILayout.Button("Add Rig Component"))
                {
                    rig = currentSuit.AddComponent<IronManSuitRig>();
                    rig.SetupRig();
                }
            }
            else
            {
                EditorGUILayout.BeginHorizontal();
                
                if (GUILayout.Button("Setup Rig"))
                {
                    rig.SetupRig();
                }
                
                if (GUILayout.Button("Create IK Targets"))
                {
                    CreateIKTargets();
                }
                
                EditorGUILayout.EndHorizontal();
                
                // IK Settings
                EditorGUILayout.Space();
                EditorGUILayout.LabelField("IK Settings", EditorStyles.boldLabel);
                
                if (GUILayout.Button("Toggle IK Visualization"))
                {
                    // Toggle gizmos
                }
            }
            
            EditorGUI.indentLevel--;
        }
        
        private void CreateIKTargets()
        {
            if (rig == null) return;
            
            // Create IK target objects
            GameObject ikTargets = new GameObject("IK_Targets");
            ikTargets.transform.SetParent(currentSuit.transform);
            
            // Left hand target
            GameObject leftHandTarget = new GameObject("LeftHand_Target");
            leftHandTarget.transform.SetParent(ikTargets.transform);
            leftHandTarget.transform.position = new Vector3(-0.5f, 1f, 0.3f);
            AddTargetGizmo(leftHandTarget);
            
            // Right hand target
            GameObject rightHandTarget = new GameObject("RightHand_Target");
            rightHandTarget.transform.SetParent(ikTargets.transform);
            rightHandTarget.transform.position = new Vector3(0.5f, 1f, 0.3f);
            AddTargetGizmo(rightHandTarget);
            
            // Left foot target
            GameObject leftFootTarget = new GameObject("LeftFoot_Target");
            leftFootTarget.transform.SetParent(ikTargets.transform);
            leftFootTarget.transform.position = new Vector3(-0.2f, 0, 0);
            AddTargetGizmo(leftFootTarget);
            
            // Right foot target
            GameObject rightFootTarget = new GameObject("RightFoot_Target");
            rightFootTarget.transform.SetParent(ikTargets.transform);
            rightFootTarget.transform.position = new Vector3(0.2f, 0, 0);
            AddTargetGizmo(rightFootTarget);
            
            EditorUtility.DisplayDialog("IK Targets Created", 
                "IK target objects have been created. Assign them to the rig component.", "OK");
        }
        
        private void AddTargetGizmo(GameObject target)
        {
            // Add a visual gizmo component (would need to create this)
            // For now, just add a small sphere
            GameObject gizmo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            gizmo.transform.SetParent(target.transform);
            gizmo.transform.localScale = Vector3.one * 0.05f;
            
            // Make it semi-transparent
            MeshRenderer renderer = gizmo.GetComponent<MeshRenderer>();
            renderer.material = new Material(Shader.Find("Standard"));
            renderer.material.color = new Color(1f, 1f, 0f, 0.5f);
        }
        
        #endregion
        
        #region Component Library
        
        private void DrawComponentLibrarySection()
        {
            EditorGUI.indentLevel++;
            
            if (currentSuit == null)
            {
                EditorGUILayout.HelpBox("Generate a suit first to add components.", MessageType.Warning);
                EditorGUI.indentLevel--;
                return;
            }
            
            componentLibrary = currentSuit.GetComponent<SuitComponentLibrary>();
            
            if (componentLibrary == null)
            {
                if (GUILayout.Button("Add Component Library"))
                {
                    componentLibrary = currentSuit.AddComponent<SuitComponentLibrary>();
                }
            }
            else
            {
                EditorGUILayout.LabelField("Quick Add Components", EditorStyles.boldLabel);
                
                // Weapons
                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Add Repulsors"))
                {
                    AddRepulsors();
                }
                if (GUILayout.Button("Add Unibeam"))
                {
                    AddUnibeam();
                }
                if (GUILayout.Button("Add Missiles"))
                {
                    AddMissilePods();
                }
                EditorGUILayout.EndHorizontal();
                
                // Thrusters
                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Add Boot Thrusters"))
                {
                    AddBootThrusters();
                }
                if (GUILayout.Button("Add Back Thrusters"))
                {
                    AddBackThrusters();
                }
                EditorGUILayout.EndHorizontal();
                
                // Details
                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Add Panel Lines"))
                {
                    AddPanelLines();
                }
                if (GUILayout.Button("Add Vents"))
                {
                    AddVents();
                }
                if (GUILayout.Button("Add Sensors"))
                {
                    AddSensors();
                }
                EditorGUILayout.EndHorizontal();
            }
            
            EditorGUI.indentLevel--;
        }
        
        private void AddRepulsors()
        {
            // Add repulsors to hands
            Transform leftHand = currentSuit.transform.Find("*/LeftHand");
            Transform rightHand = currentSuit.transform.Find("*/RightHand");
            
            if (leftHand != null)
            {
                GameObject leftRepulsor = componentLibrary.GetComponent("Standard Repulsor");
                if (leftRepulsor != null)
                {
                    componentLibrary.AttachComponent(leftRepulsor, leftHand, 
                        new Vector3(0, -0.05f, 0.02f), Vector3.zero);
                }
            }
            
            if (rightHand != null)
            {
                GameObject rightRepulsor = componentLibrary.GetComponent("Standard Repulsor");
                if (rightRepulsor != null)
                {
                    componentLibrary.AttachComponent(rightRepulsor, rightHand, 
                        new Vector3(0, -0.05f, 0.02f), Vector3.zero);
                }
            }
        }
        
        private void AddUnibeam()
        {
            Transform chest = currentSuit.transform.Find("*/ChestPlate");
            if (chest != null)
            {
                GameObject unibeam = componentLibrary.GetComponent("Heavy Repulsor");
                if (unibeam != null)
                {
                    componentLibrary.AttachComponent(unibeam, chest, 
                        new Vector3(0, 0, 0.16f), Vector3.zero);
                    unibeam.name = "Unibeam";
                }
            }
        }
        
        private void AddMissilePods()
        {
            Transform leftShoulder = currentSuit.transform.Find("*/LeftShoulder");
            Transform rightShoulder = currentSuit.transform.Find("*/RightShoulder");
            
            if (leftShoulder != null)
            {
                GameObject leftPod = componentLibrary.GetComponent("Shoulder Missile Pod");
                if (leftPod != null)
                {
                    componentLibrary.AttachComponent(leftPod, leftShoulder, 
                        new Vector3(-0.1f, 0.05f, 0), new Vector3(0, -30, 0));
                }
            }
            
            if (rightShoulder != null)
            {
                GameObject rightPod = componentLibrary.GetComponent("Shoulder Missile Pod");
                if (rightPod != null)
                {
                    componentLibrary.AttachComponent(rightPod, rightShoulder, 
                        new Vector3(0.1f, 0.05f, 0), new Vector3(0, 30, 0));
                }
            }
        }
        
        private void AddBootThrusters()
        {
            // Implementation for boot thrusters
        }
        
        private void AddBackThrusters()
        {
            // Implementation for back thrusters
        }
        
        private void AddPanelLines()
        {
            // Implementation for panel lines
        }
        
        private void AddVents()
        {
            // Implementation for vents
        }
        
        private void AddSensors()
        {
            // Implementation for sensors
        }
        
        #endregion
        
        #region Material Settings
        
        private void DrawMaterialSettingsSection()
        {
            EditorGUI.indentLevel++;
            
            if (currentSuit == null)
            {
                EditorGUILayout.HelpBox("Generate a suit first to modify materials.", MessageType.Warning);
                EditorGUI.indentLevel--;
                return;
            }
            
            EditorGUILayout.LabelField("Material Presets", EditorStyles.boldLabel);
            
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Classic"))
            {
                ApplyMaterialPreset("Classic");
            }
            if (GUILayout.Button("Stealth"))
            {
                ApplyMaterialPreset("Stealth");
            }
            if (GUILayout.Button("Gold Titanium"))
            {
                ApplyMaterialPreset("GoldTitanium");
            }
            if (GUILayout.Button("Battle Damaged"))
            {
                ApplyMaterialPreset("BattleDamaged");
            }
            EditorGUILayout.EndHorizontal();
            
            EditorGUILayout.Space();
            
            // Material properties
            EditorGUILayout.LabelField("Material Properties", EditorStyles.boldLabel);
            
            if (GUILayout.Button("Create PBR Materials"))
            {
                CreatePBRMaterials();
            }
            
            EditorGUI.indentLevel--;
        }
        
        private void ApplyMaterialPreset(string presetName)
        {
            switch (presetName)
            {
                case "Classic":
                    primaryColor = new Color(0.8f, 0.1f, 0.1f);
                    secondaryColor = new Color(0.9f, 0.75f, 0.1f);
                    break;
                    
                case "Stealth":
                    primaryColor = new Color(0.1f, 0.1f, 0.1f);
                    secondaryColor = new Color(0.3f, 0.3f, 0.3f);
                    break;
                    
                case "GoldTitanium":
                    primaryColor = new Color(0.9f, 0.75f, 0.1f);
                    secondaryColor = new Color(0.8f, 0.8f, 0.8f);
                    break;
                    
                case "BattleDamaged":
                    primaryColor = new Color(0.5f, 0.05f, 0.05f);
                    secondaryColor = new Color(0.6f, 0.5f, 0.05f);
                    break;
            }
            
            UpdateSuitMaterials();
        }
        
        private void UpdateSuitMaterials()
        {
            if (currentSuit == null) return;
            
            MeshRenderer[] renderers = currentSuit.GetComponentsInChildren<MeshRenderer>();
            
            foreach (var renderer in renderers)
            {
                // Update material colors based on component type
                // This is simplified - in practice you'd have more sophisticated material assignment
                if (renderer.material != null)
                {
                    if (renderer.name.Contains("Eye") || renderer.name.Contains("Reactor") || renderer.name.Contains("Repulsor"))
                    {
                        renderer.material.color = accentColor;
                        renderer.material.SetColor("_EmissionColor", accentColor * 5f);
                        renderer.material.EnableKeyword("_EMISSION");
                    }
                    else if (renderer.name.Contains("Secondary") || renderer.name.Contains("Gold"))
                    {
                        renderer.material.color = secondaryColor;
                    }
                    else
                    {
                        renderer.material.color = primaryColor;
                    }
                }
            }
        }
        
        private void CreatePBRMaterials()
        {
            // Create folder for materials
            string folderPath = "Assets/Materials/IronManSuit";
            if (!AssetDatabase.IsValidFolder(folderPath))
            {
                AssetDatabase.CreateFolder("Assets/Materials", "IronManSuit");
            }
            
            // Create primary material
            Material primaryMat = new Material(Shader.Find("Standard"));
            primaryMat.name = "IronMan_Primary";
            primaryMat.color = primaryColor;
            primaryMat.SetFloat("_Metallic", 0.8f);
            primaryMat.SetFloat("_Glossiness", 0.9f);
            AssetDatabase.CreateAsset(primaryMat, $"{folderPath}/IronMan_Primary.mat");
            
            // Create secondary material
            Material secondaryMat = new Material(Shader.Find("Standard"));
            secondaryMat.name = "IronMan_Secondary";
            secondaryMat.color = secondaryColor;
            secondaryMat.SetFloat("_Metallic", 0.9f);
            secondaryMat.SetFloat("_Glossiness", 0.95f);
            AssetDatabase.CreateAsset(secondaryMat, $"{folderPath}/IronMan_Secondary.mat");
            
            // Create emissive material
            Material emissiveMat = new Material(Shader.Find("Standard"));
            emissiveMat.name = "IronMan_Emissive";
            emissiveMat.color = accentColor;
            emissiveMat.SetColor("_EmissionColor", accentColor * 5f);
            emissiveMat.EnableKeyword("_EMISSION");
            emissiveMat.SetFloat("_Metallic", 0.5f);
            emissiveMat.SetFloat("_Glossiness", 1f);
            AssetDatabase.CreateAsset(emissiveMat, $"{folderPath}/IronMan_Emissive.mat");
            
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            
            EditorUtility.DisplayDialog("Materials Created", 
                "PBR materials have been created in Assets/Materials/IronManSuit", "OK");
        }
        
        #endregion
    }
}