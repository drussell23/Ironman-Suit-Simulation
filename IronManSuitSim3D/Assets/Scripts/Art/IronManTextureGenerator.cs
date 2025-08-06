using UnityEngine;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
using System.IO;
#endif

namespace IronManSim.Art
{
    /// <summary>
    /// Generates procedural textures for the Iron Man suit
    /// Creates base color, metallic, normal, and emission maps
    /// </summary>
    public class IronManTextureGenerator : MonoBehaviour
    {
        [Header("Texture Settings")]
        [SerializeField] private int textureWidth = 2048;
        [SerializeField] private int textureHeight = 2048;
        [SerializeField] private string textureSavePath = "Assets/Textures/IronMan/";
        
        [Header("Base Colors")]
        [SerializeField] private Color primaryColor = new Color(0.7f, 0.1f, 0.1f);
        [SerializeField] private Color secondaryColor = new Color(0.9f, 0.75f, 0.1f);
        [SerializeField] private Color tertiaryColor = new Color(0.3f, 0.3f, 0.3f);
        [SerializeField] private Color emissiveColor = new Color(0.3f, 0.8f, 1f);
        
        [Header("Surface Properties")]
        [SerializeField] private float primaryMetallic = 0.9f;
        [SerializeField] private float primaryRoughness = 0.15f;
        [SerializeField] private float wearAmount = 0.2f;
        [SerializeField] private float dirtAmount = 0.1f;
        
        [Header("Detail Settings")]
        [SerializeField] private bool addPanelLines = true;
        [SerializeField] private float panelLineWidth = 2f;
        [SerializeField] private float panelLineDepth = 0.5f;
        [SerializeField] private bool addTechDetails = true;
        [SerializeField] private bool addWeatheringEffects = true;
        
        #if UNITY_EDITOR
        [MenuItem("IronMan/Art/Generate Suit Textures")]
        public static void GenerateTextures()
        {
            GameObject tempObj = new GameObject("TextureGenerator");
            IronManTextureGenerator generator = tempObj.AddComponent<IronManTextureGenerator>();
            generator.GenerateAllTextures();
            DestroyImmediate(tempObj);
        }
        #endif
        
        [ContextMenu("Generate All Textures")]
        public void GenerateAllTextures()
        {
            #if UNITY_EDITOR
            // Create save directory if it doesn't exist
            if (!Directory.Exists(textureSavePath))
            {
                Directory.CreateDirectory(textureSavePath);
            }
            
            // Generate each texture type
            Texture2D baseColorMap = GenerateBaseColorMap();
            Texture2D metallicMap = GenerateMetallicMap();
            Texture2D normalMap = GenerateNormalMap();
            Texture2D emissionMap = GenerateEmissionMap();
            Texture2D occlusionMap = GenerateOcclusionMap();
            
            // Save textures
            SaveTexture(baseColorMap, "IronMan_BaseColor.png");
            SaveTexture(metallicMap, "IronMan_Metallic.png");
            SaveTexture(normalMap, "IronMan_Normal.png");
            SaveTexture(emissionMap, "IronMan_Emission.png");
            SaveTexture(occlusionMap, "IronMan_Occlusion.png");
            
            // Create height map for detail
            if (addTechDetails)
            {
                Texture2D heightMap = GenerateHeightMap();
                SaveTexture(heightMap, "IronMan_Height.png");
            }
            
            AssetDatabase.Refresh();
            Debug.Log($"Textures generated and saved to {textureSavePath}");
            #endif
        }
        
        private Texture2D GenerateBaseColorMap()
        {
            Texture2D texture = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, true);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float u = (float)x / textureWidth;
                    float v = (float)y / textureHeight;
                    
                    Color pixelColor = GetBaseColorForUV(u, v);
                    
                    // Add weathering if enabled
                    if (addWeatheringEffects)
                    {
                        pixelColor = ApplyWeathering(pixelColor, u, v);
                    }
                    
                    texture.SetPixel(x, y, pixelColor);
                }
            }
            
            // Add panel lines
            if (addPanelLines)
            {
                AddPanelLinesToTexture(texture);
            }
            
            texture.Apply();
            return texture;
        }
        
        private Color GetBaseColorForUV(float u, float v)
        {
            // Define regions for different colored parts
            // This is a simplified UV layout - in production, use proper UV coordinates
            
            // Chest area (center)
            if (u > 0.4f && u < 0.6f && v > 0.6f && v < 0.8f)
            {
                // Arc reactor area
                if (Vector2.Distance(new Vector2(u, v), new Vector2(0.5f, 0.7f)) < 0.05f)
                {
                    return tertiaryColor;
                }
                return secondaryColor; // Gold chest piece
            }
            
            // Helmet area (top)
            if (v > 0.85f)
            {
                // Faceplate
                if (u > 0.35f && u < 0.65f && v > 0.9f)
                {
                    return secondaryColor;
                }
                return primaryColor;
            }
            
            // Arms (sides)
            if ((u < 0.2f || u > 0.8f) && v > 0.3f && v < 0.7f)
            {
                // Gauntlet areas
                if (v < 0.4f)
                {
                    return secondaryColor;
                }
                return primaryColor;
            }
            
            // Legs (bottom sides)
            if (v < 0.3f && (u < 0.3f || u > 0.7f))
            {
                return primaryColor;
            }
            
            // Default to primary color
            return primaryColor;
        }
        
        private Color ApplyWeathering(Color baseColor, float u, float v)
        {
            // Add wear and scratches
            float wear = Mathf.PerlinNoise(u * 50f, v * 50f);
            if (wear > (1f - wearAmount))
            {
                // Exposed metal underneath
                baseColor = Color.Lerp(baseColor, new Color(0.7f, 0.7f, 0.7f), 0.5f);
            }
            
            // Add dirt/grime
            float dirt = Mathf.PerlinNoise(u * 20f + 100f, v * 20f + 100f);
            if (dirt > (1f - dirtAmount))
            {
                baseColor = Color.Lerp(baseColor, new Color(0.2f, 0.2f, 0.2f), 0.3f);
            }
            
            return baseColor;
        }
        
        private void AddPanelLinesToTexture(Texture2D texture)
        {
            // Add vertical panel lines
            for (float x = 0.1f; x < 1f; x += 0.2f)
            {
                DrawLine(texture, x, 0f, x, 1f, tertiaryColor, (int)panelLineWidth);
            }
            
            // Add horizontal panel lines
            for (float y = 0.1f; y < 1f; y += 0.15f)
            {
                DrawLine(texture, 0f, y, 1f, y, tertiaryColor, (int)panelLineWidth);
            }
            
            // Add some diagonal details
            DrawLine(texture, 0.3f, 0.6f, 0.4f, 0.8f, tertiaryColor, (int)panelLineWidth);
            DrawLine(texture, 0.6f, 0.6f, 0.7f, 0.8f, tertiaryColor, (int)panelLineWidth);
        }
        
        private void DrawLine(Texture2D texture, float x0, float y0, float x1, float y1, Color color, int width)
        {
            int px0 = Mathf.RoundToInt(x0 * textureWidth);
            int py0 = Mathf.RoundToInt(y0 * textureHeight);
            int px1 = Mathf.RoundToInt(x1 * textureWidth);
            int py1 = Mathf.RoundToInt(y1 * textureHeight);
            
            // Bresenham's line algorithm with width
            int dx = Mathf.Abs(px1 - px0);
            int dy = Mathf.Abs(py1 - py0);
            int sx = px0 < px1 ? 1 : -1;
            int sy = py0 < py1 ? 1 : -1;
            int err = dx - dy;
            
            while (true)
            {
                // Draw a circle at each point for line width
                for (int w = -width/2; w <= width/2; w++)
                {
                    for (int h = -width/2; h <= width/2; h++)
                    {
                        int x = px0 + w;
                        int y = py0 + h;
                        if (x >= 0 && x < textureWidth && y >= 0 && y < textureHeight)
                        {
                            if (w*w + h*h <= (width/2)*(width/2))
                            {
                                texture.SetPixel(x, y, color);
                            }
                        }
                    }
                }
                
                if (px0 == px1 && py0 == py1) break;
                
                int e2 = 2 * err;
                if (e2 > -dy)
                {
                    err -= dy;
                    px0 += sx;
                }
                if (e2 < dx)
                {
                    err += dx;
                    py0 += sy;
                }
            }
        }
        
        private Texture2D GenerateMetallicMap()
        {
            Texture2D texture = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, true);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float u = (float)x / textureWidth;
                    float v = (float)y / textureHeight;
                    
                    // Metallic value based on region
                    float metallic = GetMetallicForUV(u, v);
                    
                    // Roughness (inverted smoothness) in alpha channel
                    float roughness = 1f - (primaryRoughness + Random.Range(-0.05f, 0.05f));
                    
                    // Add variation
                    float variation = Mathf.PerlinNoise(u * 30f, v * 30f) * 0.1f;
                    metallic = Mathf.Clamp01(metallic + variation);
                    
                    // Pack metallic in RGB, roughness in A
                    Color pixelColor = new Color(metallic, metallic, metallic, roughness);
                    texture.SetPixel(x, y, pixelColor);
                }
            }
            
            texture.Apply();
            return texture;
        }
        
        private float GetMetallicForUV(float u, float v)
        {
            // Different metallic values for different parts
            Color baseColor = GetBaseColorForUV(u, v);
            
            // Gold parts are highly metallic
            if (ColorDistance(baseColor, secondaryColor) < 0.1f)
            {
                return 0.95f;
            }
            // Red armor is also metallic but slightly less
            else if (ColorDistance(baseColor, primaryColor) < 0.1f)
            {
                return primaryMetallic;
            }
            // Dark parts are less metallic
            else
            {
                return 0.7f;
            }
        }
        
        private float ColorDistance(Color a, Color b)
        {
            return Vector3.Distance(new Vector3(a.r, a.g, a.b), new Vector3(b.r, b.g, b.b));
        }
        
        private Texture2D GenerateNormalMap()
        {
            Texture2D heightMap = GenerateHeightMap();
            Texture2D normalMap = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, true);
            
            // Convert height map to normal map
            for (int y = 1; y < textureHeight - 1; y++)
            {
                for (int x = 1; x < textureWidth - 1; x++)
                {
                    // Sample neighboring heights
                    float left = heightMap.GetPixel(x - 1, y).grayscale;
                    float right = heightMap.GetPixel(x + 1, y).grayscale;
                    float bottom = heightMap.GetPixel(x, y - 1).grayscale;
                    float top = heightMap.GetPixel(x, y + 1).grayscale;
                    
                    // Calculate normal
                    Vector3 normal = new Vector3(left - right, 2f, bottom - top).normalized;
                    
                    // Pack into color (0-1 range)
                    Color normalColor = new Color(
                        normal.x * 0.5f + 0.5f,
                        normal.y * 0.5f + 0.5f,
                        normal.z * 0.5f + 0.5f,
                        1f
                    );
                    
                    normalMap.SetPixel(x, y, normalColor);
                }
            }
            
            normalMap.Apply();
            return normalMap;
        }
        
        private Texture2D GenerateHeightMap()
        {
            Texture2D texture = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, true);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float u = (float)x / textureWidth;
                    float v = (float)y / textureHeight;
                    
                    float height = 0.5f; // Base height
                    
                    // Add panel depth
                    if (addPanelLines)
                    {
                        // Check if near a panel line
                        if (Mathf.Repeat(u * 5f, 1f) < 0.05f || Mathf.Repeat(v * 6.67f, 1f) < 0.05f)
                        {
                            height -= panelLineDepth * 0.5f;
                        }
                    }
                    
                    // Add tech details
                    if (addTechDetails)
                    {
                        // Rivets
                        float rivetPattern = Mathf.Sin(u * 50f) * Mathf.Sin(v * 50f);
                        if (rivetPattern > 0.9f)
                        {
                            height += 0.1f;
                        }
                        
                        // Raised armor sections
                        float armor = Mathf.PerlinNoise(u * 10f, v * 10f);
                        if (armor > 0.6f)
                        {
                            height += 0.05f;
                        }
                    }
                    
                    Color heightColor = new Color(height, height, height, 1f);
                    texture.SetPixel(x, y, heightColor);
                }
            }
            
            texture.Apply();
            return texture;
        }
        
        private Texture2D GenerateEmissionMap()
        {
            Texture2D texture = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, true);
            
            // Start with black (no emission)
            Color black = Color.black;
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    texture.SetPixel(x, y, black);
                }
            }
            
            // Add arc reactor glow (center chest)
            AddCircularGlow(texture, 0.5f, 0.7f, 0.05f, emissiveColor * 3f);
            
            // Add eye glows
            AddCircularGlow(texture, 0.45f, 0.92f, 0.02f, emissiveColor * 2f);
            AddCircularGlow(texture, 0.55f, 0.92f, 0.02f, emissiveColor * 2f);
            
            // Add repulsor glows (hands)
            AddCircularGlow(texture, 0.1f, 0.4f, 0.03f, emissiveColor);
            AddCircularGlow(texture, 0.9f, 0.4f, 0.03f, emissiveColor);
            
            // Add boot thruster glows
            AddRectangularGlow(texture, 0.2f, 0.05f, 0.08f, 0.03f, emissiveColor * 0.5f);
            AddRectangularGlow(texture, 0.8f, 0.05f, 0.08f, 0.03f, emissiveColor * 0.5f);
            
            texture.Apply();
            return texture;
        }
        
        private void AddCircularGlow(Texture2D texture, float centerU, float centerV, float radius, Color color)
        {
            int centerX = Mathf.RoundToInt(centerU * textureWidth);
            int centerY = Mathf.RoundToInt(centerV * textureHeight);
            int radiusPixels = Mathf.RoundToInt(radius * textureWidth);
            
            for (int y = -radiusPixels; y <= radiusPixels; y++)
            {
                for (int x = -radiusPixels; x <= radiusPixels; x++)
                {
                    int px = centerX + x;
                    int py = centerY + y;
                    
                    if (px >= 0 && px < textureWidth && py >= 0 && py < textureHeight)
                    {
                        float distance = Mathf.Sqrt(x * x + y * y);
                        if (distance <= radiusPixels)
                        {
                            float intensity = 1f - (distance / radiusPixels);
                            intensity = Mathf.Pow(intensity, 2f); // Falloff curve
                            Color glowColor = color * intensity;
                            texture.SetPixel(px, py, glowColor);
                        }
                    }
                }
            }
        }
        
        private void AddRectangularGlow(Texture2D texture, float centerU, float centerV, float width, float height, Color color)
        {
            int startX = Mathf.RoundToInt((centerU - width/2) * textureWidth);
            int endX = Mathf.RoundToInt((centerU + width/2) * textureWidth);
            int startY = Mathf.RoundToInt((centerV - height/2) * textureHeight);
            int endY = Mathf.RoundToInt((centerV + height/2) * textureHeight);
            
            for (int y = startY; y <= endY; y++)
            {
                for (int x = startX; x <= endX; x++)
                {
                    if (x >= 0 && x < textureWidth && y >= 0 && y < textureHeight)
                    {
                        texture.SetPixel(x, y, color);
                    }
                }
            }
        }
        
        private Texture2D GenerateOcclusionMap()
        {
            Texture2D texture = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, true);
            
            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    float u = (float)x / textureWidth;
                    float v = (float)y / textureHeight;
                    
                    float occlusion = 1f; // Start with no occlusion
                    
                    // Darken panel line areas
                    if (Mathf.Repeat(u * 5f, 1f) < 0.05f || Mathf.Repeat(v * 6.67f, 1f) < 0.05f)
                    {
                        occlusion *= 0.7f;
                    }
                    
                    // Darken crevices and joints
                    float creviceNoise = Mathf.PerlinNoise(u * 40f, v * 40f);
                    if (creviceNoise < 0.3f)
                    {
                        occlusion *= 0.8f;
                    }
                    
                    Color occlusionColor = new Color(occlusion, occlusion, occlusion, 1f);
                    texture.SetPixel(x, y, occlusionColor);
                }
            }
            
            texture.Apply();
            return texture;
        }
        
        #if UNITY_EDITOR
        private void SaveTexture(Texture2D texture, string filename)
        {
            byte[] bytes = texture.EncodeToPNG();
            string fullPath = textureSavePath + filename;
            File.WriteAllBytes(fullPath, bytes);
            
            // Import settings for the texture
            AssetDatabase.ImportAsset(fullPath);
            TextureImporter importer = AssetImporter.GetAtPath(fullPath) as TextureImporter;
            if (importer != null)
            {
                if (filename.Contains("Normal"))
                {
                    importer.textureType = TextureImporterType.NormalMap;
                }
                else if (filename.Contains("Metallic"))
                {
                    importer.sRGBTexture = false; // Linear for metallic maps
                }
                
                importer.maxTextureSize = textureWidth;
                importer.textureCompression = TextureImporterCompression.Compressed;
                AssetDatabase.ImportAsset(fullPath);
            }
        }
        #endif
    }
}