using UnityEngine;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace IronManSim.Art
{
    /// <summary>
    /// Advanced Iron Man suit designer with detailed mesh generation,
    /// materials, textures, and visual effects
    /// </summary>
    [ExecuteInEditMode]
    public class IronManSuitDesigner : MonoBehaviour
    {
        [Header("Suit Design Presets")]
        [SerializeField] private SuitDesignPreset currentPreset = SuitDesignPreset.MarkIII;
        
        [Header("Color Scheme")]
        [SerializeField] private Color primaryColor = new Color(0.7f, 0.1f, 0.1f); // Deep red
        [SerializeField] private Color secondaryColor = new Color(0.9f, 0.75f, 0.1f); // Gold
        [SerializeField] private Color tertiaryColor = new Color(0.3f, 0.3f, 0.3f); // Dark gray
        [SerializeField] private Color arcReactorColor = new Color(0.3f, 0.8f, 1f); // Cyan blue
        [SerializeField] private Color eyeGlowColor = new Color(1f, 1f, 0.9f); // Bright white
        
        [Header("Material Properties")]
        [SerializeField] private float metallic = 0.9f;
        [SerializeField] private float smoothness = 0.85f;
        [SerializeField] private float normalStrength = 1f;
        [SerializeField] private float emissiveIntensity = 3f;
        
        [Header("Detail Settings")]
        [SerializeField] private bool addPanelLines = true;
        [SerializeField] private bool addRivets = true;
        [SerializeField] private bool addBattleDamage = false;
        [SerializeField] private float damageAmount = 0.3f;
        
        [Header("Advanced Options")]
        [SerializeField] private bool generateHighPolyDetails = false;
        [SerializeField] private bool createLODs = true;
        [SerializeField] private int textureResolution = 2048;
        
        private GameObject suitRoot;
        private Dictionary<string, Material> suitMaterials = new Dictionary<string, Material>();
        
        public enum SuitDesignPreset
        {
            MarkI,      // Cave-built prototype
            MarkII,     // Silver prototype
            MarkIII,    // Classic red and gold
            MarkVII,    // Avengers suit
            MarkXLII,   // Modular suit
            MarkL,      // Infinity War nano-suit
            MarkLXXXV,  // Endgame suit
            Stealth,    // Black stealth variant
            Hulkbuster, // Heavy armor variant
        }
        
        #if UNITY_EDITOR
        [MenuItem("IronMan/Art/Create Detailed Suit Model")]
        public static void CreateDetailedSuit()
        {
            GameObject suitObj = new GameObject("IronManSuit_Detailed");
            IronManSuitDesigner designer = suitObj.AddComponent<IronManSuitDesigner>();
            designer.BuildDetailedSuit();
            Selection.activeGameObject = suitObj;
        }
        #endif
        
        [ContextMenu("Build Detailed Suit")]
        public void BuildDetailedSuit()
        {
            // Clean up existing suit
            if (suitRoot != null)
            {
                if (Application.isPlaying)
                    Destroy(suitRoot);
                else
                    DestroyImmediate(suitRoot);
            }
            
            // Apply preset colors
            ApplyPresetColors();
            
            // Create root
            suitRoot = new GameObject("IronManSuit_Model");
            suitRoot.transform.SetParent(transform);
            suitRoot.transform.localPosition = Vector3.zero;
            suitRoot.transform.localRotation = Quaternion.identity;
            
            // Create materials
            CreateSuitMaterials();
            
            // Build suit components with detailed geometry
            BuildHelmetDetailed();
            BuildTorsoDetailed();
            BuildArmsDetailed();
            BuildLegsDetailed();
            
            // Add detail elements
            if (addPanelLines)
                AddPanelLineDetails();
            
            if (addRivets)
                AddRivetDetails();
            
            if (addBattleDamage)
                ApplyBattleDamage();
            
            // Create LODs if requested
            if (createLODs)
                GenerateLODs();
        }
        
        private void ApplyPresetColors()
        {
            switch (currentPreset)
            {
                case SuitDesignPreset.MarkI:
                    primaryColor = new Color(0.4f, 0.4f, 0.4f); // Gray metal
                    secondaryColor = new Color(0.3f, 0.3f, 0.3f); // Dark gray
                    metallic = 0.7f;
                    smoothness = 0.4f;
                    break;
                    
                case SuitDesignPreset.MarkII:
                    primaryColor = new Color(0.8f, 0.8f, 0.8f); // Silver
                    secondaryColor = new Color(0.7f, 0.7f, 0.7f); // Light silver
                    metallic = 0.95f;
                    smoothness = 0.9f;
                    break;
                    
                case SuitDesignPreset.Stealth:
                    primaryColor = new Color(0.1f, 0.1f, 0.1f); // Black
                    secondaryColor = new Color(0.2f, 0.2f, 0.2f); // Dark gray
                    arcReactorColor = new Color(0.1f, 0.3f, 0.5f); // Dim blue
                    metallic = 0.8f;
                    smoothness = 0.6f;
                    break;
                    
                case SuitDesignPreset.Hulkbuster:
                    primaryColor = new Color(0.8f, 0.1f, 0.1f); // Bright red
                    secondaryColor = new Color(0.9f, 0.8f, 0.1f); // Bright gold
                    transform.localScale = Vector3.one * 1.5f; // Larger scale
                    break;
                    
                // Default Mark III colors already set
            }
        }
        
        private void CreateSuitMaterials()
        {
            // Primary armor material
            Material primaryMat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            primaryMat.name = "IronMan_Primary";
            primaryMat.color = primaryColor;
            primaryMat.SetFloat("_Metallic", metallic);
            primaryMat.SetFloat("_Smoothness", smoothness);
            suitMaterials["Primary"] = primaryMat;
            
            // Secondary armor material
            Material secondaryMat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            secondaryMat.name = "IronMan_Secondary";
            secondaryMat.color = secondaryColor;
            secondaryMat.SetFloat("_Metallic", metallic);
            secondaryMat.SetFloat("_Smoothness", smoothness);
            suitMaterials["Secondary"] = secondaryMat;
            
            // Tertiary detail material
            Material tertiaryMat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            tertiaryMat.name = "IronMan_Details";
            tertiaryMat.color = tertiaryColor;
            tertiaryMat.SetFloat("_Metallic", 0.8f);
            tertiaryMat.SetFloat("_Smoothness", 0.6f);
            suitMaterials["Tertiary"] = tertiaryMat;
            
            // Arc Reactor emissive material
            Material arcReactorMat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            arcReactorMat.name = "IronMan_ArcReactor";
            arcReactorMat.color = arcReactorColor;
            arcReactorMat.EnableKeyword("_EMISSION");
            arcReactorMat.SetColor("_EmissionColor", arcReactorColor * emissiveIntensity);
            arcReactorMat.SetFloat("_Metallic", 1f);
            arcReactorMat.SetFloat("_Smoothness", 1f);
            suitMaterials["ArcReactor"] = arcReactorMat;
            
            // Eye glow material
            Material eyeMat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            eyeMat.name = "IronMan_Eyes";
            eyeMat.color = eyeGlowColor;
            eyeMat.EnableKeyword("_EMISSION");
            eyeMat.SetColor("_EmissionColor", eyeGlowColor * emissiveIntensity);
            eyeMat.SetFloat("_Metallic", 1f);
            eyeMat.SetFloat("_Smoothness", 1f);
            suitMaterials["Eyes"] = eyeMat;
        }
        
        private void BuildHelmetDetailed()
        {
            GameObject helmet = new GameObject("Helmet");
            helmet.transform.SetParent(suitRoot.transform);
            helmet.transform.localPosition = new Vector3(0, 1.6f, 0);
            
            // Main helmet shell (custom mesh would be better, using primitives for now)
            GameObject helmetMain = CreateDetailedMeshPrimitive("HelmetMain", PrimitiveType.Sphere);
            helmetMain.transform.SetParent(helmet.transform);
            helmetMain.transform.localPosition = Vector3.zero;
            helmetMain.transform.localScale = new Vector3(0.26f, 0.32f, 0.29f);
            ApplyMaterial(helmetMain, "Primary");
            
            // Faceplate - more angular design
            GameObject faceplate = CreateDetailedMeshPrimitive("Faceplate", PrimitiveType.Cube);
            faceplate.transform.SetParent(helmet.transform);
            faceplate.transform.localPosition = new Vector3(0, -0.05f, 0.12f);
            faceplate.transform.localScale = new Vector3(0.22f, 0.22f, 0.12f);
            faceplate.transform.localRotation = Quaternion.Euler(10, 0, 0);
            ApplyMaterial(faceplate, "Secondary");
            
            // Jaw piece
            GameObject jaw = CreateDetailedMeshPrimitive("Jaw", PrimitiveType.Cube);
            jaw.transform.SetParent(helmet.transform);
            jaw.transform.localPosition = new Vector3(0, -0.15f, 0.08f);
            jaw.transform.localScale = new Vector3(0.2f, 0.1f, 0.15f);
            ApplyMaterial(jaw, "Secondary");
            
            // Eye pieces with glow
            CreateEyePiece(helmet.transform, true);
            CreateEyePiece(helmet.transform, false);
            
            // Helmet details
            AddHelmetDetails(helmet.transform);
        }
        
        private void CreateEyePiece(Transform parent, bool isLeft)
        {
            float xOffset = isLeft ? -0.06f : 0.06f;
            
            // Eye socket
            GameObject eyeSocket = CreateDetailedMeshPrimitive($"{(isLeft ? "Left" : "Right")}EyeSocket", PrimitiveType.Cube);
            eyeSocket.transform.SetParent(parent);
            eyeSocket.transform.localPosition = new Vector3(xOffset, 0, 0.13f);
            eyeSocket.transform.localScale = new Vector3(0.05f, 0.03f, 0.02f);
            eyeSocket.transform.localRotation = Quaternion.Euler(0, 0, isLeft ? 5 : -5);
            ApplyMaterial(eyeSocket, "Tertiary");
            
            // Eye lens
            GameObject eyeLens = CreateDetailedMeshPrimitive($"{(isLeft ? "Left" : "Right")}Eye", PrimitiveType.Cube);
            eyeLens.transform.SetParent(eyeSocket.transform);
            eyeLens.transform.localPosition = new Vector3(0, 0, 0.5f);
            eyeLens.transform.localScale = new Vector3(0.8f, 0.8f, 0.5f);
            ApplyMaterial(eyeLens, "Eyes");
            
            // Add light
            Light eyeLight = eyeLens.AddComponent<Light>();
            eyeLight.type = LightType.Point;
            eyeLight.color = eyeGlowColor;
            eyeLight.intensity = 2f;
            eyeLight.range = 0.5f;
        }
        
        private void AddHelmetDetails(Transform helmetTransform)
        {
            // Mohawk-style detail on top
            GameObject mohawk = CreateDetailedMeshPrimitive("MohawkDetail", PrimitiveType.Cube);
            mohawk.transform.SetParent(helmetTransform);
            mohawk.transform.localPosition = new Vector3(0, 0.18f, 0);
            mohawk.transform.localScale = new Vector3(0.02f, 0.08f, 0.25f);
            ApplyMaterial(mohawk, "Secondary");
            
            // Side vents
            for (int i = 0; i < 3; i++)
            {
                float z = -0.05f + (i * 0.05f);
                CreateVentDetail(helmetTransform, new Vector3(0.13f, 0, z), "LeftVent" + i);
                CreateVentDetail(helmetTransform, new Vector3(-0.13f, 0, z), "RightVent" + i);
            }
            
            // Chin details
            GameObject chinDetail = CreateDetailedMeshPrimitive("ChinDetail", PrimitiveType.Cube);
            chinDetail.transform.SetParent(helmetTransform);
            chinDetail.transform.localPosition = new Vector3(0, -0.18f, 0.1f);
            chinDetail.transform.localScale = new Vector3(0.15f, 0.02f, 0.08f);
            ApplyMaterial(chinDetail, "Tertiary");
        }
        
        private void CreateVentDetail(Transform parent, Vector3 position, string name)
        {
            GameObject vent = CreateDetailedMeshPrimitive(name, PrimitiveType.Cube);
            vent.transform.SetParent(parent);
            vent.transform.localPosition = position;
            vent.transform.localScale = new Vector3(0.02f, 0.04f, 0.02f);
            ApplyMaterial(vent, "Tertiary");
        }
        
        private void BuildTorsoDetailed()
        {
            GameObject torso = new GameObject("Torso");
            torso.transform.SetParent(suitRoot.transform);
            torso.transform.localPosition = new Vector3(0, 0.9f, 0);
            
            // Chest main plate
            GameObject chestMain = CreateDetailedMeshPrimitive("ChestMain", PrimitiveType.Cube);
            chestMain.transform.SetParent(torso.transform);
            chestMain.transform.localPosition = Vector3.zero;
            chestMain.transform.localScale = new Vector3(0.5f, 0.6f, 0.3f);
            ApplyMaterial(chestMain, "Primary");
            
            // Chest center piece (gold)
            GameObject chestCenter = CreateDetailedMeshPrimitive("ChestCenter", PrimitiveType.Cube);
            chestCenter.transform.SetParent(torso.transform);
            chestCenter.transform.localPosition = new Vector3(0, 0, 0.16f);
            chestCenter.transform.localScale = new Vector3(0.3f, 0.5f, 0.05f);
            ApplyMaterial(chestCenter, "Secondary");
            
            // Arc Reactor housing
            GameObject reactorHousing = CreateDetailedMeshPrimitive("ReactorHousing", PrimitiveType.Cylinder);
            reactorHousing.transform.SetParent(torso.transform);
            reactorHousing.transform.localPosition = new Vector3(0, 0.05f, 0.17f);
            reactorHousing.transform.localRotation = Quaternion.Euler(90, 0, 0);
            reactorHousing.transform.localScale = new Vector3(0.18f, 0.03f, 0.18f);
            ApplyMaterial(reactorHousing, "Tertiary");
            
            // Arc Reactor core
            GameObject reactorCore = CreateDetailedMeshPrimitive("ArcReactor", PrimitiveType.Cylinder);
            reactorCore.transform.SetParent(reactorHousing.transform);
            reactorCore.transform.localPosition = new Vector3(0, 0.5f, 0);
            reactorCore.transform.localScale = new Vector3(0.8f, 0.5f, 0.8f);
            ApplyMaterial(reactorCore, "ArcReactor");
            
            // Arc Reactor light
            Light reactorLight = reactorCore.AddComponent<Light>();
            reactorLight.type = LightType.Point;
            reactorLight.color = arcReactorColor;
            reactorLight.intensity = 3f;
            reactorLight.range = 2f;
            
            // Add reactor details
            CreateArcReactorDetails(reactorHousing.transform);
            
            // Shoulder joints
            CreateShoulderJoint(torso.transform, true);
            CreateShoulderJoint(torso.transform, false);
            
            // Abdominal plates
            CreateAbdominalPlates(torso.transform);
            
            // Back details
            CreateBackDetails(torso.transform);
        }
        
        private void CreateArcReactorDetails(Transform reactorTransform)
        {
            // Create triangular segments around the reactor
            int segments = 8;
            for (int i = 0; i < segments; i++)
            {
                float angle = i * (360f / segments) * Mathf.Deg2Rad;
                float x = Mathf.Cos(angle) * 0.7f;
                float z = Mathf.Sin(angle) * 0.7f;
                
                GameObject segment = CreateDetailedMeshPrimitive($"ReactorSegment{i}", PrimitiveType.Cube);
                segment.transform.SetParent(reactorTransform);
                segment.transform.localPosition = new Vector3(x, 0.3f, z);
                segment.transform.localRotation = Quaternion.Euler(0, angle * Mathf.Rad2Deg, 0);
                segment.transform.localScale = new Vector3(0.15f, 0.3f, 0.05f);
                ApplyMaterial(segment, "Tertiary");
            }
        }
        
        private void CreateShoulderJoint(Transform parent, bool isLeft)
        {
            float xPos = isLeft ? -0.35f : 0.35f;
            
            GameObject shoulder = CreateDetailedMeshPrimitive($"{(isLeft ? "Left" : "Right")}Shoulder", PrimitiveType.Sphere);
            shoulder.transform.SetParent(parent);
            shoulder.transform.localPosition = new Vector3(xPos, 0.2f, 0);
            shoulder.transform.localScale = new Vector3(0.22f, 0.18f, 0.22f);
            ApplyMaterial(shoulder, "Secondary");
            
            // Shoulder plate
            GameObject shoulderPlate = CreateDetailedMeshPrimitive($"{(isLeft ? "Left" : "Right")}ShoulderPlate", PrimitiveType.Cube);
            shoulderPlate.transform.SetParent(shoulder.transform);
            shoulderPlate.transform.localPosition = new Vector3(isLeft ? -0.5f : 0.5f, 0.3f, 0);
            shoulderPlate.transform.localScale = new Vector3(0.4f, 0.3f, 0.8f);
            shoulderPlate.transform.localRotation = Quaternion.Euler(0, 0, isLeft ? 20 : -20);
            ApplyMaterial(shoulderPlate, "Primary");
        }
        
        private void CreateAbdominalPlates(Transform parent)
        {
            // Upper abs
            GameObject upperAbs = CreateDetailedMeshPrimitive("UpperAbs", PrimitiveType.Cube);
            upperAbs.transform.SetParent(parent);
            upperAbs.transform.localPosition = new Vector3(0, -0.2f, 0.1f);
            upperAbs.transform.localScale = new Vector3(0.35f, 0.15f, 0.25f);
            ApplyMaterial(upperAbs, "Secondary");
            
            // Lower abs
            GameObject lowerAbs = CreateDetailedMeshPrimitive("LowerAbs", PrimitiveType.Cube);
            lowerAbs.transform.SetParent(parent);
            lowerAbs.transform.localPosition = new Vector3(0, -0.35f, 0.08f);
            lowerAbs.transform.localScale = new Vector3(0.3f, 0.1f, 0.22f);
            ApplyMaterial(lowerAbs, "Primary");
            
            // Side plates
            CreateSidePlate(parent, true);
            CreateSidePlate(parent, false);
        }
        
        private void CreateSidePlate(Transform parent, bool isLeft)
        {
            float xPos = isLeft ? -0.22f : 0.22f;
            
            GameObject sidePlate = CreateDetailedMeshPrimitive($"{(isLeft ? "Left" : "Right")}SidePlate", PrimitiveType.Cube);
            sidePlate.transform.SetParent(parent);
            sidePlate.transform.localPosition = new Vector3(xPos, -0.1f, 0);
            sidePlate.transform.localScale = new Vector3(0.08f, 0.4f, 0.25f);
            sidePlate.transform.localRotation = Quaternion.Euler(0, 0, isLeft ? -10 : 10);
            ApplyMaterial(sidePlate, "Tertiary");
        }
        
        private void CreateBackDetails(Transform parent)
        {
            // Back thruster housings
            GameObject backPlate = CreateDetailedMeshPrimitive("BackPlate", PrimitiveType.Cube);
            backPlate.transform.SetParent(parent);
            backPlate.transform.localPosition = new Vector3(0, 0, -0.16f);
            backPlate.transform.localScale = new Vector3(0.4f, 0.5f, 0.05f);
            ApplyMaterial(backPlate, "Primary");
            
            // Thruster ports
            CreateThrusterPort(backPlate.transform, new Vector3(0.15f, 0.15f, -0.5f), "BackThruster1");
            CreateThrusterPort(backPlate.transform, new Vector3(-0.15f, 0.15f, -0.5f), "BackThruster2");
        }
        
        private void CreateThrusterPort(Transform parent, Vector3 position, string name)
        {
            GameObject port = CreateDetailedMeshPrimitive(name, PrimitiveType.Cylinder);
            port.transform.SetParent(parent);
            port.transform.localPosition = position;
            port.transform.localRotation = Quaternion.Euler(90, 0, 0);
            port.transform.localScale = new Vector3(0.3f, 0.5f, 0.3f);
            ApplyMaterial(port, "Tertiary");
            
            // Inner glow
            GameObject innerGlow = CreateDetailedMeshPrimitive(name + "_Glow", PrimitiveType.Cylinder);
            innerGlow.transform.SetParent(port.transform);
            innerGlow.transform.localPosition = Vector3.zero;
            innerGlow.transform.localScale = new Vector3(0.8f, 0.9f, 0.8f);
            ApplyMaterial(innerGlow, "ArcReactor");
        }
        
        private void BuildArmsDetailed()
        {
            BuildArmDetailed(true);
            BuildArmDetailed(false);
        }
        
        private void BuildArmDetailed(bool isLeft)
        {
            string side = isLeft ? "Left" : "Right";
            float xPos = isLeft ? -0.35f : 0.35f;
            
            GameObject arm = new GameObject($"{side}Arm");
            arm.transform.SetParent(suitRoot.transform);
            arm.transform.localPosition = new Vector3(xPos, 0.9f, 0);
            
            // Upper arm
            GameObject upperArm = CreateDetailedMeshPrimitive($"{side}UpperArm", PrimitiveType.Capsule);
            upperArm.transform.SetParent(arm.transform);
            upperArm.transform.localPosition = new Vector3(0, -0.2f, 0);
            upperArm.transform.localScale = new Vector3(0.13f, 0.2f, 0.13f);
            ApplyMaterial(upperArm, "Primary");
            
            // Bicep detail
            GameObject bicepDetail = CreateDetailedMeshPrimitive($"{side}BicepDetail", PrimitiveType.Cube);
            bicepDetail.transform.SetParent(upperArm.transform);
            bicepDetail.transform.localPosition = new Vector3(0, 0.3f, 0.5f);
            bicepDetail.transform.localScale = new Vector3(0.7f, 0.6f, 0.3f);
            ApplyMaterial(bicepDetail, "Secondary");
            
            // Elbow joint
            GameObject elbow = CreateDetailedMeshPrimitive($"{side}Elbow", PrimitiveType.Sphere);
            elbow.transform.SetParent(arm.transform);
            elbow.transform.localPosition = new Vector3(0, -0.4f, 0);
            elbow.transform.localScale = new Vector3(0.11f, 0.11f, 0.11f);
            ApplyMaterial(elbow, "Tertiary");
            
            // Forearm
            GameObject forearm = CreateDetailedMeshPrimitive($"{side}Forearm", PrimitiveType.Capsule);
            forearm.transform.SetParent(arm.transform);
            forearm.transform.localPosition = new Vector3(0, -0.6f, 0);
            forearm.transform.localScale = new Vector3(0.11f, 0.18f, 0.11f);
            ApplyMaterial(forearm, "Primary");
            
            // Forearm armor plates
            GameObject forearmPlate = CreateDetailedMeshPrimitive($"{side}ForearmPlate", PrimitiveType.Cube);
            forearmPlate.transform.SetParent(forearm.transform);
            forearmPlate.transform.localPosition = new Vector3(0, 0, 0.6f);
            forearmPlate.transform.localScale = new Vector3(0.8f, 0.9f, 0.3f);
            ApplyMaterial(forearmPlate, "Secondary");
            
            // Gauntlet
            BuildGauntletDetailed(arm.transform, isLeft);
        }
        
        private void BuildGauntletDetailed(Transform armTransform, bool isLeft)
        {
            string side = isLeft ? "Left" : "Right";
            
            // Hand base
            GameObject hand = CreateDetailedMeshPrimitive($"{side}Hand", PrimitiveType.Cube);
            hand.transform.SetParent(armTransform);
            hand.transform.localPosition = new Vector3(0, -0.8f, 0);
            hand.transform.localScale = new Vector3(0.09f, 0.14f, 0.07f);
            ApplyMaterial(hand, "Primary");
            
            // Knuckle guards
            GameObject knuckles = CreateDetailedMeshPrimitive($"{side}Knuckles", PrimitiveType.Cube);
            knuckles.transform.SetParent(hand.transform);
            knuckles.transform.localPosition = new Vector3(0, 0.3f, 0.5f);
            knuckles.transform.localScale = new Vector3(0.9f, 0.3f, 0.4f);
            ApplyMaterial(knuckles, "Secondary");
            
            // Repulsor housing
            GameObject repulsorHousing = CreateDetailedMeshPrimitive($"{side}RepulsorHousing", PrimitiveType.Cylinder);
            repulsorHousing.transform.SetParent(hand.transform);
            repulsorHousing.transform.localPosition = new Vector3(0, 0, 0.6f);
            repulsorHousing.transform.localRotation = Quaternion.Euler(90, 0, 0);
            repulsorHousing.transform.localScale = new Vector3(0.8f, 0.3f, 0.8f);
            ApplyMaterial(repulsorHousing, "Tertiary");
            
            // Repulsor emitter
            GameObject repulsor = CreateDetailedMeshPrimitive($"{side}Repulsor", PrimitiveType.Cylinder);
            repulsor.transform.SetParent(repulsorHousing.transform);
            repulsor.transform.localPosition = new Vector3(0, 0.5f, 0);
            repulsor.transform.localScale = new Vector3(0.7f, 0.5f, 0.7f);
            ApplyMaterial(repulsor, "ArcReactor");
            
            // Add light
            Light repulsorLight = repulsor.AddComponent<Light>();
            repulsorLight.type = LightType.Spot;
            repulsorLight.color = arcReactorColor;
            repulsorLight.intensity = 2f;
            repulsorLight.range = 5f;
            repulsorLight.spotAngle = 30f;
            
            // Fingers (simplified)
            for (int i = 0; i < 4; i++)
            {
                float fingerX = (i - 1.5f) * 0.2f;
                GameObject finger = CreateDetailedMeshPrimitive($"{side}Finger{i}", PrimitiveType.Capsule);
                finger.transform.SetParent(hand.transform);
                finger.transform.localPosition = new Vector3(fingerX, -0.6f, 0);
                finger.transform.localScale = new Vector3(0.15f, 0.4f, 0.15f);
                ApplyMaterial(finger, "Secondary");
            }
        }
        
        private void BuildLegsDetailed()
        {
            BuildLegDetailed(true);
            BuildLegDetailed(false);
        }
        
        private void BuildLegDetailed(bool isLeft)
        {
            string side = isLeft ? "Left" : "Right";
            float xPos = isLeft ? -0.15f : 0.15f;
            
            GameObject leg = new GameObject($"{side}Leg");
            leg.transform.SetParent(suitRoot.transform);
            leg.transform.localPosition = new Vector3(xPos, 0.5f, 0);
            
            // Hip joint
            GameObject hip = CreateDetailedMeshPrimitive($"{side}Hip", PrimitiveType.Sphere);
            hip.transform.SetParent(leg.transform);
            hip.transform.localPosition = Vector3.zero;
            hip.transform.localScale = new Vector3(0.18f, 0.15f, 0.18f);
            ApplyMaterial(hip, "Tertiary");
            
            // Thigh
            GameObject thigh = CreateDetailedMeshPrimitive($"{side}Thigh", PrimitiveType.Capsule);
            thigh.transform.SetParent(leg.transform);
            thigh.transform.localPosition = new Vector3(0, -0.3f, 0);
            thigh.transform.localScale = new Vector3(0.16f, 0.25f, 0.16f);
            ApplyMaterial(thigh, "Primary");
            
            // Thigh armor plate
            GameObject thighPlate = CreateDetailedMeshPrimitive($"{side}ThighPlate", PrimitiveType.Cube);
            thighPlate.transform.SetParent(thigh.transform);
            thighPlate.transform.localPosition = new Vector3(0, 0, 0.5f);
            thighPlate.transform.localScale = new Vector3(0.7f, 0.8f, 0.3f);
            ApplyMaterial(thighPlate, "Secondary");
            
            // Knee
            GameObject knee = CreateDetailedMeshPrimitive($"{side}Knee", PrimitiveType.Sphere);
            knee.transform.SetParent(leg.transform);
            knee.transform.localPosition = new Vector3(0, -0.6f, 0);
            knee.transform.localScale = new Vector3(0.13f, 0.13f, 0.13f);
            ApplyMaterial(knee, "Tertiary");
            
            // Knee cap
            GameObject kneeCap = CreateDetailedMeshPrimitive($"{side}KneeCap", PrimitiveType.Cube);
            kneeCap.transform.SetParent(knee.transform);
            kneeCap.transform.localPosition = new Vector3(0, 0, 0.5f);
            kneeCap.transform.localScale = new Vector3(0.8f, 0.8f, 0.4f);
            ApplyMaterial(kneeCap, "Secondary");
            
            // Shin
            GameObject shin = CreateDetailedMeshPrimitive($"{side}Shin", PrimitiveType.Capsule);
            shin.transform.SetParent(leg.transform);
            shin.transform.localPosition = new Vector3(0, -0.9f, 0);
            shin.transform.localScale = new Vector3(0.13f, 0.25f, 0.13f);
            ApplyMaterial(shin, "Primary");
            
            // Shin guard
            GameObject shinGuard = CreateDetailedMeshPrimitive($"{side}ShinGuard", PrimitiveType.Cube);
            shinGuard.transform.SetParent(shin.transform);
            shinGuard.transform.localPosition = new Vector3(0, 0, 0.5f);
            shinGuard.transform.localScale = new Vector3(0.7f, 0.9f, 0.3f);
            ApplyMaterial(shinGuard, "Secondary");
            
            // Boot
            BuildBootDetailed(leg.transform, isLeft);
        }
        
        private void BuildBootDetailed(Transform legTransform, bool isLeft)
        {
            string side = isLeft ? "Left" : "Right";
            
            // Ankle
            GameObject ankle = CreateDetailedMeshPrimitive($"{side}Ankle", PrimitiveType.Sphere);
            ankle.transform.SetParent(legTransform);
            ankle.transform.localPosition = new Vector3(0, -1.15f, 0);
            ankle.transform.localScale = new Vector3(0.11f, 0.08f, 0.11f);
            ApplyMaterial(ankle, "Tertiary");
            
            // Foot base
            GameObject foot = CreateDetailedMeshPrimitive($"{side}Foot", PrimitiveType.Cube);
            foot.transform.SetParent(legTransform);
            foot.transform.localPosition = new Vector3(0, -1.25f, 0.1f);
            foot.transform.localScale = new Vector3(0.13f, 0.08f, 0.28f);
            ApplyMaterial(foot, "Primary");
            
            // Toe cap
            GameObject toeCap = CreateDetailedMeshPrimitive($"{side}ToeCap", PrimitiveType.Cube);
            toeCap.transform.SetParent(foot.transform);
            toeCap.transform.localPosition = new Vector3(0, 0, 0.6f);
            toeCap.transform.localScale = new Vector3(0.9f, 1.2f, 0.3f);
            toeCap.transform.localRotation = Quaternion.Euler(-20, 0, 0);
            ApplyMaterial(toeCap, "Secondary");
            
            // Heel
            GameObject heel = CreateDetailedMeshPrimitive($"{side}Heel", PrimitiveType.Cube);
            heel.transform.SetParent(foot.transform);
            heel.transform.localPosition = new Vector3(0, -0.5f, -0.4f);
            heel.transform.localScale = new Vector3(0.8f, 0.5f, 0.4f);
            ApplyMaterial(heel, "Primary");
            
            // Boot thruster
            GameObject thrusterHousing = CreateDetailedMeshPrimitive($"{side}BootThruster", PrimitiveType.Cylinder);
            thrusterHousing.transform.SetParent(foot.transform);
            thrusterHousing.transform.localPosition = new Vector3(0, -0.8f, 0);
            thrusterHousing.transform.localScale = new Vector3(0.8f, 0.4f, 0.8f);
            ApplyMaterial(thrusterHousing, "Tertiary");
            
            // Thruster core
            GameObject thrusterCore = CreateDetailedMeshPrimitive($"{side}ThrusterCore", PrimitiveType.Cylinder);
            thrusterCore.transform.SetParent(thrusterHousing.transform);
            thrusterCore.transform.localPosition = new Vector3(0, -0.5f, 0);
            thrusterCore.transform.localScale = new Vector3(0.7f, 0.3f, 0.7f);
            ApplyMaterial(thrusterCore, "ArcReactor");
        }
        
        private void AddPanelLineDetails()
        {
            // This would add geometric panel lines to the suit
            // In a real implementation, this would modify the mesh or add decals
            Debug.Log("Panel lines would be added here with proper mesh editing or decals");
        }
        
        private void AddRivetDetails()
        {
            // This would add small rivet details
            // In production, these would be normal map details or small geometry
            Debug.Log("Rivet details would be added here");
        }
        
        private void ApplyBattleDamage()
        {
            // This would add scratches, dents, and wear
            // Would use damage textures and vertex deformation in production
            Debug.Log($"Battle damage at {damageAmount * 100}% intensity would be applied here");
        }
        
        private void GenerateLODs()
        {
            // This would create lower polygon versions for distance viewing
            // Unity's LOD Group component would be used in production
            Debug.Log("LODs would be generated here for optimized rendering");
        }
        
        private GameObject CreateDetailedMeshPrimitive(string name, PrimitiveType type)
        {
            GameObject obj = GameObject.CreatePrimitive(type);
            obj.name = name;
            
            // In production, this would create custom meshes with proper topology
            // For now, we're using primitives with modifications
            
            return obj;
        }
        
        private void ApplyMaterial(GameObject obj, string materialKey)
        {
            if (suitMaterials.ContainsKey(materialKey))
            {
                Renderer renderer = obj.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material = suitMaterials[materialKey];
                }
            }
        }
        
        [ContextMenu("Export Suit Design")]
        public void ExportSuitDesign()
        {
            // This would export the suit configuration for saving/sharing
            Debug.Log("Suit design configuration:");
            Debug.Log($"Preset: {currentPreset}");
            Debug.Log($"Primary Color: {primaryColor}");
            Debug.Log($"Secondary Color: {secondaryColor}");
            Debug.Log($"Metallic: {metallic}");
            Debug.Log($"Smoothness: {smoothness}");
        }
    }
}