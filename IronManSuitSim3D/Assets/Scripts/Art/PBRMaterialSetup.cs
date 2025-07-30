using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using System.IO;
#endif

namespace IronManSim.Art
{
    /// <summary>
    /// Sets up PBR materials for Iron Man suit using generated textures
    /// Creates materials optimized for URP and HDRP
    /// </summary>
    public class PBRMaterialSetup : MonoBehaviour
    {
        [Header("Material Settings")]
        [SerializeField] private string materialName = "IronManSuit_PBR";
        [SerializeField] private Shader targetShader;
        [SerializeField] private string materialsPath = "Assets/Materials/IronMan/";
        [SerializeField] private string texturesPath = "Assets/Textures/IronMan/PBR/";
        
        [Header("Texture References")]
        [SerializeField] private Texture2D albedoTexture;
        [SerializeField] private Texture2D metallicTexture;
        [SerializeField] private Texture2D normalTexture;
        [SerializeField] private Texture2D heightTexture;
        [SerializeField] private Texture2D occlusionTexture;
        [SerializeField] private Texture2D maskTexture;
        
        [Header("Material Properties")]
        [SerializeField] private float normalStrength = 1f;
        [SerializeField] private float heightScale = 0.02f;
        [SerializeField] private float occlusionStrength = 1f;
        [SerializeField] private Vector2 tiling = Vector2.one;
        [SerializeField] private Vector2 offset = Vector2.zero;
        
        #if UNITY_EDITOR
        [MenuItem("IronMan/Art/Setup PBR Material")]
        public static void SetupPBRMaterialMenu()
        {
            GameObject tempObj = new GameObject("Material Setup");
            PBRMaterialSetup setup = tempObj.AddComponent<PBRMaterialSetup>();
            setup.CreatePBRMaterial();
            DestroyImmediate(tempObj);
        }
        
        [ContextMenu("Create PBR Material")]
        public void CreatePBRMaterial()
        {
            // Auto-detect shader if not set
            if (targetShader == null)
            {
                // Try to find URP Lit shader first
                targetShader = Shader.Find("Universal Render Pipeline/Lit");
                
                if (targetShader == null)
                {
                    // Fallback to standard
                    targetShader = Shader.Find("Standard");
                }
            }
            
            if (targetShader == null)
            {
                Debug.LogError("No suitable shader found!");
                return;
            }
            
            // Load textures if not assigned
            LoadTexturesIfNeeded();
            
            // Create material
            Material pbrMaterial = new Material(targetShader);
            pbrMaterial.name = materialName;
            
            // Configure based on shader type
            if (targetShader.name.Contains("Universal Render Pipeline"))
            {
                SetupURPMaterial(pbrMaterial);
            }
            else if (targetShader.name.Contains("HDRP"))
            {
                SetupHDRPMaterial(pbrMaterial);
            }
            else
            {
                SetupStandardMaterial(pbrMaterial);
            }
            
            // Save material
            if (!Directory.Exists(materialsPath))
            {
                Directory.CreateDirectory(materialsPath);
            }
            
            string materialPath = materialsPath + materialName + ".mat";
            AssetDatabase.CreateAsset(pbrMaterial, materialPath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            
            Debug.Log($"PBR Material created at: {materialPath}");
            
            // Select the created material
            Selection.activeObject = pbrMaterial;
        }
        
        private void LoadTexturesIfNeeded()
        {
            string prefix = "IronManSuit_PBR_";
            
            if (albedoTexture == null)
                albedoTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturesPath + prefix + "Albedo.png");
            
            if (metallicTexture == null)
                metallicTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturesPath + prefix + "Metallic.png");
            
            if (normalTexture == null)
                normalTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturesPath + prefix + "Normal.png");
            
            if (heightTexture == null)
                heightTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturesPath + prefix + "Height.png");
            
            if (occlusionTexture == null)
                occlusionTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturesPath + prefix + "AO.png");
            
            if (maskTexture == null)
                maskTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturesPath + prefix + "Mask.png");
        }
        
        private void SetupURPMaterial(Material mat)
        {
            // Base Map
            if (albedoTexture != null)
            {
                mat.SetTexture("_BaseMap", albedoTexture);
                mat.SetTexture("_MainTex", albedoTexture); // Legacy support
                mat.SetColor("_BaseColor", Color.white);
            }
            
            // Metallic and Smoothness
            if (maskTexture != null)
            {
                // URP uses Mask Map (RGBA: Metallic, Occlusion, Detail, Smoothness)
                mat.SetTexture("_MetallicGlossMap", maskTexture);
                mat.SetFloat("_Metallic", 1f);
                mat.SetFloat("_Smoothness", 1f);
            }
            else if (metallicTexture != null)
            {
                mat.SetTexture("_MetallicGlossMap", metallicTexture);
                mat.SetFloat("_Metallic", 1f);
            }
            
            // Normal Map
            if (normalTexture != null)
            {
                mat.EnableKeyword("_NORMALMAP");
                mat.SetTexture("_BumpMap", normalTexture);
                mat.SetFloat("_BumpScale", normalStrength);
            }
            
            // Height Map (Parallax)
            if (heightTexture != null)
            {
                mat.EnableKeyword("_PARALLAXMAP");
                mat.SetTexture("_ParallaxMap", heightTexture);
                mat.SetFloat("_Parallax", heightScale);
            }
            
            // Occlusion
            if (occlusionTexture != null)
            {
                mat.SetTexture("_OcclusionMap", occlusionTexture);
                mat.SetFloat("_OcclusionStrength", occlusionStrength);
            }
            
            // Tiling and Offset
            mat.SetTextureScale("_BaseMap", tiling);
            mat.SetTextureOffset("_BaseMap", offset);
            
            // Surface options
            mat.SetFloat("_Surface", 0); // 0 = Opaque
            mat.SetFloat("_WorkflowMode", 0); // 0 = Metallic
            
            // Enable GPU instancing
            mat.enableInstancing = true;
        }
        
        private void SetupHDRPMaterial(Material mat)
        {
            // Base Color
            if (albedoTexture != null)
            {
                mat.SetTexture("_BaseColorMap", albedoTexture);
                mat.SetColor("_BaseColor", Color.white);
            }
            
            // Mask Map (RGBA: Metallic, AO, Detail, Smoothness)
            if (maskTexture != null)
            {
                mat.SetTexture("_MaskMap", maskTexture);
                mat.SetFloat("_Metallic", 1f);
                mat.SetFloat("_Smoothness", 1f);
                mat.SetFloat("_AORemapMin", 0f);
                mat.SetFloat("_AORemapMax", 1f);
            }
            
            // Normal Map
            if (normalTexture != null)
            {
                mat.EnableKeyword("_NORMALMAP");
                mat.SetTexture("_NormalMap", normalTexture);
                mat.SetFloat("_NormalScale", normalStrength);
            }
            
            // Height Map
            if (heightTexture != null)
            {
                mat.EnableKeyword("_HEIGHTMAP");
                mat.SetTexture("_HeightMap", heightTexture);
                mat.SetFloat("_HeightAmplitude", heightScale);
                mat.SetFloat("_HeightCenter", 0.5f);
            }
            
            // Tiling and Offset
            mat.SetTextureScale("_BaseColorMap", tiling);
            mat.SetTextureOffset("_BaseColorMap", offset);
            
            // Enable features
            mat.SetFloat("_MaterialID", 1); // Metallic workflow
            mat.enableInstancing = true;
        }
        
        private void SetupStandardMaterial(Material mat)
        {
            // Albedo
            if (albedoTexture != null)
            {
                mat.SetTexture("_MainTex", albedoTexture);
                mat.SetColor("_Color", Color.white);
            }
            
            // Metallic and Smoothness
            if (metallicTexture != null)
            {
                mat.SetTexture("_MetallicGlossMap", metallicTexture);
                mat.SetFloat("_Metallic", 1f);
                mat.SetFloat("_GlossMapScale", 1f);
            }
            
            // Normal Map
            if (normalTexture != null)
            {
                mat.EnableKeyword("_NORMALMAP");
                mat.SetTexture("_BumpMap", normalTexture);
                mat.SetFloat("_BumpScale", normalStrength);
            }
            
            // Height Map (Parallax)
            if (heightTexture != null)
            {
                mat.EnableKeyword("_PARALLAXMAP");
                mat.SetTexture("_ParallaxMap", heightTexture);
                mat.SetFloat("_Parallax", heightScale);
            }
            
            // Occlusion
            if (occlusionTexture != null)
            {
                mat.SetTexture("_OcclusionMap", occlusionTexture);
                mat.SetFloat("_OcclusionStrength", occlusionStrength);
            }
            
            // Tiling and Offset
            mat.SetTextureScale("_MainTex", tiling);
            mat.SetTextureOffset("_MainTex", offset);
            
            mat.enableInstancing = true;
        }
        
        [ContextMenu("Create Material Variants")]
        public void CreateMaterialVariants()
        {
            // Load base material
            Material baseMaterial = AssetDatabase.LoadAssetAtPath<Material>(materialsPath + materialName + ".mat");
            if (baseMaterial == null)
            {
                Debug.LogError("Base material not found! Create it first.");
                return;
            }
            
            // Create battle-damaged variant
            Material damagedMaterial = new Material(baseMaterial);
            damagedMaterial.name = materialName + "_Damaged";
            
            // Increase roughness for damaged look
            if (damagedMaterial.HasProperty("_Smoothness"))
            {
                damagedMaterial.SetFloat("_Smoothness", 0.3f);
            }
            else if (damagedMaterial.HasProperty("_GlossMapScale"))
            {
                damagedMaterial.SetFloat("_GlossMapScale", 0.3f);
            }
            
            // Darken the base color slightly
            if (damagedMaterial.HasProperty("_BaseColor"))
            {
                damagedMaterial.SetColor("_BaseColor", new Color(0.8f, 0.8f, 0.8f));
            }
            else if (damagedMaterial.HasProperty("_Color"))
            {
                damagedMaterial.SetColor("_Color", new Color(0.8f, 0.8f, 0.8f));
            }
            
            AssetDatabase.CreateAsset(damagedMaterial, materialsPath + damagedMaterial.name + ".mat");
            
            // Create clean/polished variant
            Material polishedMaterial = new Material(baseMaterial);
            polishedMaterial.name = materialName + "_Polished";
            
            // Maximum smoothness for polished look
            if (polishedMaterial.HasProperty("_Smoothness"))
            {
                polishedMaterial.SetFloat("_Smoothness", 0.95f);
            }
            else if (polishedMaterial.HasProperty("_GlossMapScale"))
            {
                polishedMaterial.SetFloat("_GlossMapScale", 0.95f);
            }
            
            // Slightly brighter color
            if (polishedMaterial.HasProperty("_BaseColor"))
            {
                polishedMaterial.SetColor("_BaseColor", new Color(1.1f, 1.1f, 1.1f));
            }
            else if (polishedMaterial.HasProperty("_Color"))
            {
                polishedMaterial.SetColor("_Color", new Color(1.1f, 1.1f, 1.1f));
            }
            
            AssetDatabase.CreateAsset(polishedMaterial, materialsPath + polishedMaterial.name + ".mat");
            
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            
            Debug.Log("Material variants created!");
        }
        #endif
        
        /// <summary>
        /// Runtime method to apply PBR material to renderers
        /// </summary>
        public void ApplyMaterialToSuit(GameObject suitObject)
        {
            Material pbrMaterial = Resources.Load<Material>("Materials/IronMan/" + materialName);
            if (pbrMaterial == null)
            {
                Debug.LogError("PBR Material not found in Resources!");
                return;
            }
            
            Renderer[] renderers = suitObject.GetComponentsInChildren<Renderer>();
            foreach (var renderer in renderers)
            {
                // Skip emissive parts
                if (renderer.name.Contains("Reactor") || renderer.name.Contains("Eye") || 
                    renderer.name.Contains("Repulsor") || renderer.name.Contains("Glow"))
                {
                    continue;
                }
                
                renderer.material = pbrMaterial;
            }
            
            Debug.Log($"Applied PBR material to {renderers.Length} renderers");
        }
    }
}