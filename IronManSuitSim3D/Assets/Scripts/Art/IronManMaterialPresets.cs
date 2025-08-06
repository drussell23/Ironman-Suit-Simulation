using UnityEngine;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace IronManSim.Art
{
    /// <summary>
    /// Material preset system for Iron Man suit variations
    /// Allows quick switching between different suit designs
    /// </summary>
    [CreateAssetMenu(fileName = "IronManMaterialPreset", menuName = "IronMan/Material Preset")]
    public class IronManMaterialPreset : ScriptableObject
    {
        [Header("Preset Info")]
        public string presetName = "Mark III";
        public string description = "Classic red and gold armor";
        
        [Header("Base Colors")]
        public Color primaryColor = new Color(0.7f, 0.1f, 0.1f);
        public Color secondaryColor = new Color(0.9f, 0.75f, 0.1f);
        public Color tertiaryColor = new Color(0.3f, 0.3f, 0.3f);
        public Color detailColor = new Color(0.1f, 0.1f, 0.1f);
        
        [Header("Emissive Colors")]
        public Color arcReactorColor = new Color(0.3f, 0.8f, 1f);
        public Color eyeGlowColor = new Color(1f, 1f, 0.9f);
        public Color repulsorColor = new Color(0.4f, 0.8f, 1f);
        public float emissiveIntensity = 3f;
        
        [Header("Surface Properties")]
        [Range(0, 1)] public float primaryMetallic = 0.9f;
        [Range(0, 1)] public float primarySmoothness = 0.85f;
        [Range(0, 1)] public float secondaryMetallic = 0.95f;
        [Range(0, 1)] public float secondarySmoothness = 0.9f;
        
        [Header("Texture Settings")]
        public Texture2D baseColorMap;
        public Texture2D metallicMap;
        public Texture2D normalMap;
        public Texture2D emissionMap;
        public Texture2D occlusionMap;
        public Texture2D detailMask;
        
        [Header("Detail Properties")]
        [Range(0, 2)] public float normalStrength = 1f;
        [Range(0, 1)] public float occlusionStrength = 1f;
        public Vector2 textureScale = Vector2.one;
        
        [Header("Wear & Damage")]
        [Range(0, 1)] public float wearAmount = 0f;
        [Range(0, 1)] public float dirtAmount = 0f;
        [Range(0, 1)] public float battleDamage = 0f;
        public Texture2D damageTexture;
        public Texture2D dirtTexture;
    }
    
    /// <summary>
    /// Applies material presets to Iron Man suit models
    /// </summary>
    public class IronManMaterialManager : MonoBehaviour
    {
        [Header("Current Preset")]
        [SerializeField] private IronManMaterialPreset currentPreset;
        
        [Header("Material References")]
        [SerializeField] private Material primaryArmorMaterial;
        [SerializeField] private Material secondaryArmorMaterial;
        [SerializeField] private Material detailMaterial;
        [SerializeField] private Material emissiveMaterial;
        
        [Header("Preset Library")]
        [SerializeField] private List<IronManMaterialPreset> availablePresets = new List<IronManMaterialPreset>();
        
        [Header("Runtime Settings")]
        [SerializeField] private bool autoUpdateMaterials = true;
        [SerializeField] private float transitionDuration = 1f;
        
        private Dictionary<string, Material> materialCache = new Dictionary<string, Material>();
        
        #if UNITY_EDITOR
        [MenuItem("IronMan/Art/Create Material Preset Library")]
        public static void CreatePresetLibrary()
        {
            // Create default presets
            CreatePreset("Mark_I", "Cave-built prototype",
                new Color(0.4f, 0.4f, 0.4f), new Color(0.3f, 0.3f, 0.3f),
                0.7f, 0.4f, 0.3f, 0.2f);
            
            CreatePreset("Mark_II", "Silver prototype", 
                new Color(0.8f, 0.8f, 0.8f), new Color(0.7f, 0.7f, 0.7f),
                0.95f, 0.9f, 0f, 0f);
            
            CreatePreset("Mark_III", "Classic red and gold",
                new Color(0.7f, 0.1f, 0.1f), new Color(0.9f, 0.75f, 0.1f),
                0.9f, 0.85f, 0f, 0f);
            
            CreatePreset("Mark_VII", "Avengers suit",
                new Color(0.8f, 0.1f, 0.1f), new Color(0.95f, 0.8f, 0.1f),
                0.92f, 0.88f, 0.1f, 0.05f);
            
            CreatePreset("Mark_XLII", "Modular armor",
                new Color(0.75f, 0.15f, 0.1f), new Color(0.9f, 0.75f, 0.15f),
                0.9f, 0.85f, 0.15f, 0.1f);
            
            CreatePreset("Mark_L", "Infinity War nano-suit",
                new Color(0.8f, 0.1f, 0.05f), new Color(0.95f, 0.8f, 0.05f),
                0.95f, 0.92f, 0f, 0f);
            
            CreatePreset("Stealth", "Black stealth variant",
                new Color(0.1f, 0.1f, 0.1f), new Color(0.2f, 0.2f, 0.2f),
                0.8f, 0.6f, 0.05f, 0.1f);
            
            CreatePreset("War_Machine", "Heavy combat armor",
                new Color(0.3f, 0.3f, 0.3f), new Color(0.5f, 0.5f, 0.5f),
                0.85f, 0.7f, 0.3f, 0.2f);
            
            CreatePreset("Hulkbuster", "Heavy assault armor",
                new Color(0.8f, 0.1f, 0.1f), new Color(0.9f, 0.8f, 0.1f),
                0.9f, 0.8f, 0.2f, 0.15f);
            
            CreatePreset("Bleeding_Edge", "Advanced nano-armor",
                new Color(0.85f, 0.05f, 0.05f), new Color(1f, 0.85f, 0f),
                0.98f, 0.95f, 0f, 0f);
            
            AssetDatabase.Refresh();
            Debug.Log("Material preset library created in Assets/Materials/IronMan/Presets/");
        }
        
        private static void CreatePreset(string name, string description, 
            Color primary, Color secondary, 
            float metallic1, float smooth1, float wear, float dirt)
        {
            IronManMaterialPreset preset = ScriptableObject.CreateInstance<IronManMaterialPreset>();
            preset.presetName = name;
            preset.description = description;
            preset.primaryColor = primary;
            preset.secondaryColor = secondary;
            preset.primaryMetallic = metallic1;
            preset.primarySmoothness = smooth1;
            preset.secondaryMetallic = metallic1 + 0.05f;
            preset.secondarySmoothness = smooth1 + 0.05f;
            preset.wearAmount = wear;
            preset.dirtAmount = dirt;
            
            // Adjust emissive colors based on suit type
            if (name.Contains("Stealth"))
            {
                preset.arcReactorColor = new Color(0.1f, 0.3f, 0.5f);
                preset.emissiveIntensity = 1f;
            }
            else if (name.Contains("War_Machine"))
            {
                preset.arcReactorColor = new Color(1f, 0.5f, 0f);
                preset.eyeGlowColor = new Color(1f, 0.3f, 0f);
            }
            
            string path = "Assets/Materials/IronMan/Presets/";
            if (!System.IO.Directory.Exists(path))
            {
                System.IO.Directory.CreateDirectory(path);
            }
            
            AssetDatabase.CreateAsset(preset, path + name + "_Preset.asset");
        }
        #endif
        
        void Start()
        {
            if (currentPreset != null && autoUpdateMaterials)
            {
                ApplyPreset(currentPreset);
            }
        }
        
        [ContextMenu("Apply Current Preset")]
        public void ApplyCurrentPreset()
        {
            if (currentPreset != null)
            {
                ApplyPreset(currentPreset);
            }
        }
        
        public void ApplyPreset(IronManMaterialPreset preset)
        {
            if (preset == null) return;
            
            currentPreset = preset;
            
            // Create or update materials
            UpdatePrimaryArmorMaterial(preset);
            UpdateSecondaryArmorMaterial(preset);
            UpdateDetailMaterial(preset);
            UpdateEmissiveMaterial(preset);
            
            // Apply to all renderers in children
            ApplyMaterialsToRenderers();
            
            Debug.Log($"Applied material preset: {preset.presetName}");
        }
        
        private void UpdatePrimaryArmorMaterial(IronManMaterialPreset preset)
        {
            if (primaryArmorMaterial == null)
            {
                primaryArmorMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                primaryArmorMaterial.name = "IronMan_PrimaryArmor";
            }
            
            primaryArmorMaterial.color = preset.primaryColor;
            primaryArmorMaterial.SetFloat("_Metallic", preset.primaryMetallic);
            primaryArmorMaterial.SetFloat("_Smoothness", preset.primarySmoothness);
            
            // Apply textures if available
            if (preset.baseColorMap != null)
                primaryArmorMaterial.mainTexture = preset.baseColorMap;
            
            if (preset.metallicMap != null)
                primaryArmorMaterial.SetTexture("_MetallicGlossMap", preset.metallicMap);
            
            if (preset.normalMap != null)
            {
                primaryArmorMaterial.SetTexture("_BumpMap", preset.normalMap);
                primaryArmorMaterial.SetFloat("_BumpScale", preset.normalStrength);
                primaryArmorMaterial.EnableKeyword("_NORMALMAP");
            }
            
            if (preset.occlusionMap != null)
            {
                primaryArmorMaterial.SetTexture("_OcclusionMap", preset.occlusionMap);
                primaryArmorMaterial.SetFloat("_OcclusionStrength", preset.occlusionStrength);
            }
            
            // Apply wear and damage
            ApplyWearAndDamage(primaryArmorMaterial, preset);
        }
        
        private void UpdateSecondaryArmorMaterial(IronManMaterialPreset preset)
        {
            if (secondaryArmorMaterial == null)
            {
                secondaryArmorMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                secondaryArmorMaterial.name = "IronMan_SecondaryArmor";
            }
            
            secondaryArmorMaterial.color = preset.secondaryColor;
            secondaryArmorMaterial.SetFloat("_Metallic", preset.secondaryMetallic);
            secondaryArmorMaterial.SetFloat("_Smoothness", preset.secondarySmoothness);
            
            // Share textures with primary but use different color
            if (preset.baseColorMap != null)
            {
                // Tint the texture with secondary color
                secondaryArmorMaterial.mainTexture = preset.baseColorMap;
            }
            
            if (preset.metallicMap != null)
                secondaryArmorMaterial.SetTexture("_MetallicGlossMap", preset.metallicMap);
            
            if (preset.normalMap != null)
            {
                secondaryArmorMaterial.SetTexture("_BumpMap", preset.normalMap);
                secondaryArmorMaterial.SetFloat("_BumpScale", preset.normalStrength);
                secondaryArmorMaterial.EnableKeyword("_NORMALMAP");
            }
        }
        
        private void UpdateDetailMaterial(IronManMaterialPreset preset)
        {
            if (detailMaterial == null)
            {
                detailMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                detailMaterial.name = "IronMan_Details";
            }
            
            detailMaterial.color = preset.tertiaryColor;
            detailMaterial.SetFloat("_Metallic", 0.8f);
            detailMaterial.SetFloat("_Smoothness", 0.6f);
        }
        
        private void UpdateEmissiveMaterial(IronManMaterialPreset preset)
        {
            if (emissiveMaterial == null)
            {
                emissiveMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                emissiveMaterial.name = "IronMan_Emissive";
            }
            
            emissiveMaterial.color = preset.arcReactorColor;
            emissiveMaterial.EnableKeyword("_EMISSION");
            emissiveMaterial.SetColor("_EmissionColor", preset.arcReactorColor * preset.emissiveIntensity);
            emissiveMaterial.SetFloat("_Metallic", 1f);
            emissiveMaterial.SetFloat("_Smoothness", 1f);
            
            if (preset.emissionMap != null)
            {
                emissiveMaterial.SetTexture("_EmissionMap", preset.emissionMap);
            }
        }
        
        private void ApplyWearAndDamage(Material mat, IronManMaterialPreset preset)
        {
            if (preset.wearAmount > 0 || preset.battleDamage > 0)
            {
                // This would require a custom shader in production
                // For now, we'll adjust the colors to simulate wear
                Color wornColor = Color.Lerp(mat.color, new Color(0.6f, 0.6f, 0.6f), preset.wearAmount);
                mat.color = wornColor;
                
                float wornSmoothness = mat.GetFloat("_Smoothness") * (1f - preset.wearAmount * 0.5f);
                mat.SetFloat("_Smoothness", wornSmoothness);
            }
        }
        
        private void ApplyMaterialsToRenderers()
        {
            Renderer[] renderers = GetComponentsInChildren<Renderer>();
            
            foreach (var renderer in renderers)
            {
                string objName = renderer.gameObject.name.ToLower();
                
                // Assign materials based on object names
                if (objName.Contains("main") || objName.Contains("chest") || 
                    objName.Contains("thigh") || objName.Contains("shin") ||
                    objName.Contains("upper") || objName.Contains("helmet"))
                {
                    renderer.material = primaryArmorMaterial;
                }
                else if (objName.Contains("secondary") || objName.Contains("shoulder") ||
                         objName.Contains("faceplate") || objName.Contains("knuckle") ||
                         objName.Contains("knee"))
                {
                    renderer.material = secondaryArmorMaterial;
                }
                else if (objName.Contains("detail") || objName.Contains("vent") ||
                         objName.Contains("joint") || objName.Contains("elbow"))
                {
                    renderer.material = detailMaterial;
                }
                else if (objName.Contains("reactor") || objName.Contains("eye") ||
                         objName.Contains("repulsor") || objName.Contains("thruster") ||
                         objName.Contains("glow"))
                {
                    renderer.material = emissiveMaterial;
                }
            }
        }
        
        [ContextMenu("Cycle Through Presets")]
        public void CyclePresets()
        {
            if (availablePresets.Count == 0) return;
            
            int currentIndex = availablePresets.IndexOf(currentPreset);
            currentIndex = (currentIndex + 1) % availablePresets.Count;
            
            ApplyPreset(availablePresets[currentIndex]);
        }
        
        public void TransitionToPreset(IronManMaterialPreset newPreset, float duration)
        {
            StartCoroutine(TransitionCoroutine(currentPreset, newPreset, duration));
        }
        
        private System.Collections.IEnumerator TransitionCoroutine(
            IronManMaterialPreset from, IronManMaterialPreset to, float duration)
        {
            float elapsed = 0f;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float t = elapsed / duration;
                
                // Lerp between preset values
                Color primaryColor = Color.Lerp(from.primaryColor, to.primaryColor, t);
                Color secondaryColor = Color.Lerp(from.secondaryColor, to.secondaryColor, t);
                
                primaryArmorMaterial.color = primaryColor;
                secondaryArmorMaterial.color = secondaryColor;
                
                yield return null;
            }
            
            // Apply final preset
            ApplyPreset(to);
        }
    }
}