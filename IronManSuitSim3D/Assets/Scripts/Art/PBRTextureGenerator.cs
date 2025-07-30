using UnityEngine;
using System.IO;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace IronManSim.Art
{
    /// <summary>
    /// Generates PBR textures for Iron Man suit with specific material properties
    /// Creates Albedo, Metallic, Roughness, Normal, Height, and AO maps
    /// </summary>
    public class PBRTextureGenerator : MonoBehaviour
    {
        [Header("Texture Resolution")]
        [SerializeField] private int textureWidth = 2048;
        [SerializeField] private int textureHeight = 2048;
        
        [Header("Material Properties")]
        [SerializeField] private Color baseRedMetallic = new Color(0.545f, 0.0f, 0.0f); // Dark red metal
        [SerializeField] private Color goldPlatingColor = new Color(0.831f, 0.686f, 0.216f); // Gold
        [SerializeField] private Color brushedAluminumColor = new Color(0.753f, 0.753f, 0.753f); // Aluminum
        
        [Header("Surface Details")]
        [SerializeField] private float brushedMetalAngle = 45f; // Angle of brushed texture
        [SerializeField] private float scratchDensity = 0.15f; // Amount of scratches
        [SerializeField] private float scratchDepth = 0.3f; // How deep scratches are
        [SerializeField] private float goldEdgeWidth = 0.05f; // Width of gold plating on edges
        
        [Header("PBR Values")]
        [SerializeField] private float redMetallic = 0.95f;
        [SerializeField] private float redRoughness = 0.25f;
        [SerializeField] private float goldMetallic = 0.98f;
        [SerializeField] private float goldRoughness = 0.15f;
        [SerializeField] private float aluminumMetallic = 0.92f;
        [SerializeField] private float aluminumRoughness = 0.35f;
        
        [Header("Save Settings")]
        [SerializeField] private string savePath = "Assets/Textures/IronMan/PBR/";
        [SerializeField] private string texturePrefix = "IronManSuit_PBR_";
        
        #if UNITY_EDITOR
        [MenuItem("IronMan/Art/Generate PBR Textures")]
        public static void GeneratePBRTexturesMenu()
        {
            GameObject tempObj = new GameObject("PBR Generator");
            PBRTextureGenerator generator = tempObj.AddComponent<PBRTextureGenerator>();
            generator.GenerateAllPBRTextures();
            DestroyImmediate(tempObj);
        }
        #endif
        
        [ContextMenu("Generate All PBR Textures")]
        public void GenerateAllPBRTextures()
        {
            #if UNITY_EDITOR
            // Ensure save directory exists
            if (!Directory.Exists(savePath))
            {
                Directory.CreateDirectory(savePath);
            }
            
            Debug.Log("Generating PBR textures...");
            
            // Generate base masks
            Texture2D edgeMask = GenerateEdgeMask();
            Texture2D panelMask = GeneratePanelMask();
            Texture2D scratchMask = GenerateScratchMask();
            Texture2D brushedMask = GenerateBrushedMetalMask();
            
            // Generate PBR maps
            Texture2D albedoMap = GenerateAlbedoMap(edgeMask, panelMask, scratchMask);
            Texture2D metallicMap = GenerateMetallicMap(edgeMask, panelMask);
            Texture2D roughnessMap = GenerateRoughnessMap(edgeMask, panelMask, scratchMask, brushedMask);
            Texture2D normalMap = GenerateNormalMap(scratchMask, brushedMask, panelMask);
            Texture2D heightMap = GenerateHeightMap(panelMask, scratchMask);
            Texture2D aoMap = GenerateAmbientOcclusionMap(panelMask, edgeMask);
            
            // Save all textures
            SaveTexture(albedoMap, "Albedo.png", false);
            SaveTexture(metallicMap, "Metallic.png", true);
            SaveTexture(roughnessMap, "Roughness.png", true);
            SaveTexture(normalMap, "Normal.png", true, true);
            SaveTexture(heightMap, "Height.png", true);
            SaveTexture(aoMap, "AO.png", true);
            
            // Create a combined Mask map (Red: Metallic, Green: AO, Blue: Detail, Alpha: Smoothness)
            Texture2D maskMap = GenerateMaskMap(metallicMap, aoMap, heightMap, roughnessMap);
            SaveTexture(maskMap, "Mask.png", true);
            
            AssetDatabase.Refresh();
            Debug.Log($"PBR textures generated and saved to {savePath}");
            
            // Clean up temporary masks
            DestroyImmediate(edgeMask);
            DestroyImmediate(panelMask);
            DestroyImmediate(scratchMask);
            DestroyImmediate(brushedMask);
            #endif
        }
        
        private Texture2D GenerateEdgeMask()
        {
            Texture2D mask = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float u = (float)x / textureWidth;
                    float v = (float)y / textureHeight;
                    
                    // Create edge detection based on UV coordinates
                    float edge = 0f;
                    
                    // Panel edges - vertical lines
                    if (Mathf.Repeat(u * 5f, 1f) < goldEdgeWidth || Mathf.Repeat(u * 5f, 1f) > (1f - goldEdgeWidth))
                        edge = 1f;
                    
                    // Panel edges - horizontal lines
                    if (Mathf.Repeat(v * 7f, 1f) < goldEdgeWidth || Mathf.Repeat(v * 7f, 1f) > (1f - goldEdgeWidth))
                        edge = 1f;
                    
                    // Major panel divisions
                    if (Mathf.Abs(u - 0.5f) < goldEdgeWidth * 0.5f) edge = 1f; // Center line
                    if (Mathf.Abs(v - 0.3f) < goldEdgeWidth * 0.5f) edge = 1f; // Chest line
                    if (Mathf.Abs(v - 0.7f) < goldEdgeWidth * 0.5f) edge = 1f; // Waist line
                    
                    mask.SetPixel(x, y, new Color(edge, edge, edge, 1f));
                }
            }
            
            mask.Apply();
            return mask;
        }
        
        private Texture2D GeneratePanelMask()
        {
            Texture2D mask = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float u = (float)x / textureWidth;
                    float v = (float)y / textureHeight;
                    
                    // Create panel pattern
                    float panel = 1f;
                    
                    // Recessed panel areas
                    if (Mathf.Repeat(u * 5f, 1f) > 0.2f && Mathf.Repeat(u * 5f, 1f) < 0.8f &&
                        Mathf.Repeat(v * 7f, 1f) > 0.2f && Mathf.Repeat(v * 7f, 1f) < 0.8f)
                    {
                        panel = 0.7f;
                    }
                    
                    // Add some variation
                    float noise = Mathf.PerlinNoise(u * 20f, v * 20f);
                    panel += noise * 0.05f;
                    
                    mask.SetPixel(x, y, new Color(panel, panel, panel, 1f));
                }
            }
            
            mask.Apply();
            return mask;
        }
        
        private Texture2D GenerateScratchMask()
        {
            Texture2D mask = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            Random.InitState(42); // Consistent randomness
            
            // Clear to black
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    mask.SetPixel(x, y, Color.black);
                }
            }
            
            // Add random scratches
            int numScratches = Mathf.RoundToInt(100 * scratchDensity);
            for (int i = 0; i < numScratches; i++)
            {
                // Random scratch parameters
                float startX = Random.Range(0f, 1f);
                float startY = Random.Range(0f, 1f);
                float angle = Random.Range(0f, Mathf.PI * 2f);
                float length = Random.Range(0.02f, 0.1f);
                float width = Random.Range(1f, 3f);
                float intensity = Random.Range(0.3f, 1f);
                
                // Draw scratch
                DrawScratch(mask, startX, startY, angle, length, width, intensity);
            }
            
            // Add some micro scratches
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float u = (float)x / textureWidth;
                    float v = (float)y / textureHeight;
                    
                    // Fine scratches using noise
                    float microScratches = Mathf.PerlinNoise(u * 200f, v * 200f);
                    if (microScratches > 0.8f)
                    {
                        Color current = mask.GetPixel(x, y);
                        current.r = Mathf.Max(current.r, (microScratches - 0.8f) * 2f * 0.3f);
                        mask.SetPixel(x, y, current);
                    }
                }
            }
            
            mask.Apply();
            return mask;
        }
        
        private void DrawScratch(Texture2D texture, float startU, float startV, float angle, float length, float width, float intensity)
        {
            int steps = Mathf.RoundToInt(length * textureWidth);
            float stepSize = length / steps;
            
            for (int i = 0; i < steps; i++)
            {
                float t = (float)i / steps;
                float u = startU + Mathf.Cos(angle) * stepSize * i;
                float v = startV + Mathf.Sin(angle) * stepSize * i;
                
                // Taper the scratch
                float taper = 1f - Mathf.Pow(t, 2f);
                
                // Draw with width
                for (int dy = -(int)width; dy <= (int)width; dy++)
                {
                    for (int dx = -(int)width; dx <= (int)width; dx++)
                    {
                        if (dx * dx + dy * dy <= width * width)
                        {
                            int x = Mathf.RoundToInt(u * textureWidth) + dx;
                            int y = Mathf.RoundToInt(v * textureHeight) + dy;
                            
                            if (x >= 0 && x < textureWidth && y >= 0 && y < textureHeight)
                            {
                                float dist = Mathf.Sqrt(dx * dx + dy * dy) / width;
                                float value = intensity * taper * (1f - dist);
                                
                                Color current = texture.GetPixel(x, y);
                                current.r = Mathf.Max(current.r, value);
                                texture.SetPixel(x, y, current);
                            }
                        }
                    }
                }
            }
        }
        
        private Texture2D GenerateBrushedMetalMask()
        {
            Texture2D mask = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            float angleRad = brushedMetalAngle * Mathf.Deg2Rad;
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float u = (float)x / textureWidth;
                    float v = (float)y / textureHeight;
                    
                    // Create directional noise for brushed effect
                    float brushed = 0f;
                    
                    // Sample along the brush direction
                    for (int i = -5; i <= 5; i++)
                    {
                        float offset = i * 0.001f;
                        float sampleU = u + Mathf.Cos(angleRad) * offset;
                        float sampleV = v + Mathf.Sin(angleRad) * offset;
                        
                        float noise = Mathf.PerlinNoise(sampleU * 500f, sampleV * 500f);
                        brushed += noise;
                    }
                    brushed /= 11f; // Average
                    
                    // Add fine detail
                    float detail = Mathf.PerlinNoise(u * 1000f * Mathf.Cos(angleRad), v * 1000f * Mathf.Sin(angleRad));
                    brushed = Mathf.Lerp(brushed, detail, 0.3f);
                    
                    mask.SetPixel(x, y, new Color(brushed, brushed, brushed, 1f));
                }
            }
            
            mask.Apply();
            return mask;
        }
        
        private Texture2D GenerateAlbedoMap(Texture2D edgeMask, Texture2D panelMask, Texture2D scratchMask)
        {
            Texture2D albedo = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float edge = edgeMask.GetPixel(x, y).r;
                    float panel = panelMask.GetPixel(x, y).r;
                    float scratch = scratchMask.GetPixel(x, y).r;
                    
                    // Base color
                    Color finalColor = baseRedMetallic;
                    
                    // Gold edges
                    if (edge > 0.5f)
                    {
                        finalColor = Color.Lerp(finalColor, goldPlatingColor, edge);
                    }
                    
                    // Panel variation
                    finalColor = Color.Lerp(finalColor, finalColor * 0.9f, (1f - panel) * 0.3f);
                    
                    // Scratches reveal aluminum underneath
                    if (scratch > 0.1f)
                    {
                        finalColor = Color.Lerp(finalColor, brushedAluminumColor, scratch * 0.7f);
                    }
                    
                    // Add subtle color variation
                    float variation = Mathf.PerlinNoise(x * 0.01f, y * 0.01f);
                    finalColor = Color.Lerp(finalColor, finalColor * 1.1f, variation * 0.1f);
                    
                    albedo.SetPixel(x, y, finalColor);
                }
            }
            
            albedo.Apply();
            return albedo;
        }
        
        private Texture2D GenerateMetallicMap(Texture2D edgeMask, Texture2D panelMask)
        {
            Texture2D metallic = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float edge = edgeMask.GetPixel(x, y).r;
                    float panel = panelMask.GetPixel(x, y).r;
                    
                    // Base metallic value
                    float metallicValue = redMetallic;
                    
                    // Gold edges are more metallic
                    if (edge > 0.5f)
                    {
                        metallicValue = Mathf.Lerp(metallicValue, goldMetallic, edge);
                    }
                    
                    // Slight variation based on panels
                    metallicValue *= Mathf.Lerp(0.95f, 1f, panel);
                    
                    Color metallicColor = new Color(metallicValue, metallicValue, metallicValue, 1f);
                    metallic.SetPixel(x, y, metallicColor);
                }
            }
            
            metallic.Apply();
            return metallic;
        }
        
        private Texture2D GenerateRoughnessMap(Texture2D edgeMask, Texture2D panelMask, Texture2D scratchMask, Texture2D brushedMask)
        {
            Texture2D roughness = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float edge = edgeMask.GetPixel(x, y).r;
                    float panel = panelMask.GetPixel(x, y).r;
                    float scratch = scratchMask.GetPixel(x, y).r;
                    float brushed = brushedMask.GetPixel(x, y).r;
                    
                    // Base roughness
                    float roughnessValue = redRoughness;
                    
                    // Gold is smoother
                    if (edge > 0.5f)
                    {
                        roughnessValue = Mathf.Lerp(roughnessValue, goldRoughness, edge);
                    }
                    
                    // Scratches increase roughness
                    roughnessValue = Mathf.Lerp(roughnessValue, aluminumRoughness * 1.5f, scratch);
                    
                    // Brushed metal effect
                    roughnessValue += (brushed - 0.5f) * 0.1f;
                    
                    // Panel centers are slightly smoother
                    roughnessValue *= Mathf.Lerp(1.1f, 0.9f, Mathf.Pow(panel, 2f));
                    
                    roughnessValue = Mathf.Clamp01(roughnessValue);
                    Color roughnessColor = new Color(roughnessValue, roughnessValue, roughnessValue, 1f);
                    roughness.SetPixel(x, y, roughnessColor);
                }
            }
            
            roughness.Apply();
            return roughness;
        }
        
        private Texture2D GenerateNormalMap(Texture2D scratchMask, Texture2D brushedMask, Texture2D panelMask)
        {
            Texture2D normal = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            
            // First, create a height field from our masks
            float[,] heights = new float[textureWidth, textureHeight];
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float scratch = scratchMask.GetPixel(x, y).r;
                    float brushed = brushedMask.GetPixel(x, y).r;
                    float panel = panelMask.GetPixel(x, y).r;
                    
                    // Base height
                    float height = 0.5f;
                    
                    // Scratches are indented
                    height -= scratch * scratchDepth * 0.1f;
                    
                    // Brushed texture
                    height += (brushed - 0.5f) * 0.02f;
                    
                    // Panel depth
                    height += (panel - 0.7f) * 0.05f;
                    
                    heights[x, y] = height;
                }
            }
            
            // Convert height to normal
            for (int y = 1; y < textureHeight - 1; y++)
            {
                for (int x = 1; x < textureWidth - 1; x++)
                {
                    float left = heights[x - 1, y];
                    float right = heights[x + 1, y];
                    float bottom = heights[x, y - 1];
                    float top = heights[x, y + 1];
                    
                    Vector3 normalVector = new Vector3(
                        (left - right) * 2f,
                        (bottom - top) * 2f,
                        1f
                    ).normalized;
                    
                    // Pack into color
                    Color normalColor = new Color(
                        normalVector.x * 0.5f + 0.5f,
                        normalVector.y * 0.5f + 0.5f,
                        normalVector.z * 0.5f + 0.5f,
                        1f
                    );
                    
                    normal.SetPixel(x, y, normalColor);
                }
            }
            
            normal.Apply();
            return normal;
        }
        
        private Texture2D GenerateHeightMap(Texture2D panelMask, Texture2D scratchMask)
        {
            Texture2D height = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float panel = panelMask.GetPixel(x, y).r;
                    float scratch = scratchMask.GetPixel(x, y).r;
                    
                    // Base height
                    float heightValue = 0.5f;
                    
                    // Panel variation
                    heightValue += (panel - 0.7f) * 0.2f;
                    
                    // Scratches
                    heightValue -= scratch * scratchDepth * 0.3f;
                    
                    heightValue = Mathf.Clamp01(heightValue);
                    Color heightColor = new Color(heightValue, heightValue, heightValue, 1f);
                    height.SetPixel(x, y, heightColor);
                }
            }
            
            height.Apply();
            return height;
        }
        
        private Texture2D GenerateAmbientOcclusionMap(Texture2D panelMask, Texture2D edgeMask)
        {
            Texture2D ao = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float panel = panelMask.GetPixel(x, y).r;
                    float edge = edgeMask.GetPixel(x, y).r;
                    
                    // Base AO
                    float aoValue = 1f;
                    
                    // Darken panel edges
                    if (edge > 0.1f)
                    {
                        aoValue *= 0.7f;
                    }
                    
                    // Darken recessed panels slightly
                    aoValue *= Mathf.Lerp(0.85f, 1f, panel);
                    
                    // Add some noise for realism
                    float noise = Mathf.PerlinNoise(x * 0.01f, y * 0.01f);
                    aoValue *= Mathf.Lerp(0.9f, 1f, noise);
                    
                    Color aoColor = new Color(aoValue, aoValue, aoValue, 1f);
                    ao.SetPixel(x, y, aoColor);
                }
            }
            
            ao.Apply();
            return ao;
        }
        
        private Texture2D GenerateMaskMap(Texture2D metallic, Texture2D ao, Texture2D detail, Texture2D roughness)
        {
            Texture2D mask = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float m = metallic.GetPixel(x, y).r;
                    float a = ao.GetPixel(x, y).r;
                    float d = detail.GetPixel(x, y).r;
                    float s = 1f - roughness.GetPixel(x, y).r; // Smoothness is inverse of roughness
                    
                    mask.SetPixel(x, y, new Color(m, a, d, s));
                }
            }
            
            mask.Apply();
            return mask;
        }
        
        #if UNITY_EDITOR
        private void SaveTexture(Texture2D texture, string filename, bool isLinear, bool isNormal = false)
        {
            byte[] bytes = texture.EncodeToPNG();
            string fullPath = savePath + texturePrefix + filename;
            File.WriteAllBytes(fullPath, bytes);
            
            AssetDatabase.ImportAsset(fullPath);
            TextureImporter importer = AssetImporter.GetAtPath(fullPath) as TextureImporter;
            
            if (importer != null)
            {
                importer.maxTextureSize = textureWidth;
                importer.textureCompression = TextureImporterCompression.Compressed;
                
                if (isNormal)
                {
                    importer.textureType = TextureImporterType.NormalMap;
                }
                else
                {
                    importer.textureType = TextureImporterType.Default;
                    importer.sRGBTexture = !isLinear;
                }
                
                AssetDatabase.ImportAsset(fullPath);
            }
            
            DestroyImmediate(texture);
        }
        #endif
    }
}