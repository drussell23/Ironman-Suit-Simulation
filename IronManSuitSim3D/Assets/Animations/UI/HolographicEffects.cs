using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;

namespace IronManSim.UI.Animations
{
    /// <summary>
    /// Creates and manages holographic UI effects for Iron Man interfaces
    /// </summary>
    public class HolographicEffects : MonoBehaviour
    {
        [Header("Hologram Settings")]
        [SerializeField] private Color hologramColor = new Color(0.2f, 0.8f, 1f, 0.8f);
        [SerializeField] private Color hologramEmissionColor = new Color(0.4f, 1f, 1f, 1f);
        [SerializeField] private float scanlineSpeed = 2f;
        [SerializeField] private float noiseIntensity = 0.02f;
        [SerializeField] private float flickerFrequency = 0.1f;
        
        [Header("Grid Effect")]
        [SerializeField] private bool enableGridEffect = true;
        [SerializeField] private int gridResolution = 50;
        [SerializeField] private float gridLineWidth = 1f;
        [SerializeField] private float gridPulseSpeed = 1f;
        
        [Header("Distortion")]
        [SerializeField] private bool enableDistortion = true;
        [SerializeField] private float distortionAmount = 0.01f;
        [SerializeField] private float distortionSpeed = 5f;
        
        [Header("Projection Effect")]
        [SerializeField] private GameObject projectionBeamPrefab;
        [SerializeField] private Transform projectorOrigin;
        [SerializeField] private float projectionBuildTime = 1f;
        
        [Header("Materials")]
        [SerializeField] private Material hologramMaterial;
        [SerializeField] private Material scanlineMaterial;
        [SerializeField] private Material gridMaterial;
        
        private UIAnimationController animController;
        private Dictionary<RectTransform, HologramInstance> activeHolograms = new Dictionary<RectTransform, HologramInstance>();
        
        private class HologramInstance
        {
            public Material instanceMaterial;
            public Coroutine effectCoroutine;
            public List<GameObject> effectObjects = new List<GameObject>();
            public float creationTime;
        }
        
        void Start()
        {
            animController = UIAnimationController.Instance;
            
            // Create material instances
            if (hologramMaterial != null)
            {
                hologramMaterial = new Material(hologramMaterial);
            }
        }
        
        #region Public Methods
        
        /// <summary>
        /// Apply holographic effect to a UI element
        /// </summary>
        public void ApplyHologramEffect(RectTransform target, bool animated = true)
        {
            if (activeHolograms.ContainsKey(target))
            {
                return;
            }
            
            HologramInstance instance = new HologramInstance
            {
                creationTime = Time.time
            };
            
            // Apply hologram material to all images
            ApplyHologramMaterial(target, instance);
            
            // Start effects
            if (animated)
            {
                instance.effectCoroutine = StartCoroutine(HologramEffectCoroutine(target, instance));
                StartCoroutine(BuildHologramAnimation(target));
            }
            
            // Add grid overlay
            if (enableGridEffect)
            {
                CreateGridOverlay(target, instance);
            }
            
            // Add scanlines
            CreateScanlineEffect(target, instance);
            
            activeHolograms[target] = instance;
        }
        
        /// <summary>
        /// Remove holographic effect from UI element
        /// </summary>
        public void RemoveHologramEffect(RectTransform target, bool animated = true)
        {
            if (!activeHolograms.ContainsKey(target))
            {
                return;
            }
            
            HologramInstance instance = activeHolograms[target];
            
            if (animated)
            {
                StartCoroutine(DismissHologramAnimation(target, instance));
            }
            else
            {
                CleanupHologram(target, instance);
            }
        }
        
        /// <summary>
        /// Create a holographic projection effect
        /// </summary>
        public void CreateHolographicProjection(Vector3 startPos, Vector3 endPos, RectTransform content)
        {
            StartCoroutine(ProjectHologram(startPos, endPos, content));
        }
        
        /// <summary>
        /// Apply glitch effect to hologram
        /// </summary>
        public void TriggerHologramGlitch(RectTransform target, float duration = 0.5f)
        {
            if (!activeHolograms.ContainsKey(target))
            {
                return;
            }
            
            StartCoroutine(HologramGlitchEffect(target, activeHolograms[target], duration));
        }
        
        #endregion
        
        #region Material Application
        
        private void ApplyHologramMaterial(RectTransform target, HologramInstance instance)
        {
            // Create instance material
            if (hologramMaterial != null)
            {
                instance.instanceMaterial = new Material(hologramMaterial);
                
                // Set material properties
                instance.instanceMaterial.SetColor("_BaseColor", hologramColor);
                instance.instanceMaterial.SetColor("_EmissionColor", hologramEmissionColor);
                instance.instanceMaterial.SetFloat("_NoiseIntensity", noiseIntensity);
            }
            
            // Apply to all Image components
            Image[] images = target.GetComponentsInChildren<Image>();
            foreach (var image in images)
            {
                if (instance.instanceMaterial != null)
                {
                    image.material = instance.instanceMaterial;
                }
                
                // Adjust color
                Color c = image.color;
                c = Color.Lerp(c, hologramColor, 0.8f);
                image.color = c;
            }
            
            // Apply to TextMeshPro components
            TextMeshProUGUI[] texts = target.GetComponentsInChildren<TextMeshProUGUI>();
            foreach (var text in texts)
            {
                if (instance.instanceMaterial != null)
                {
                    text.fontMaterial = instance.instanceMaterial;
                }
                
                text.color = hologramColor;
            }
        }
        
        #endregion
        
        #region Grid Effect
        
        private void CreateGridOverlay(RectTransform target, HologramInstance instance)
        {
            GameObject gridObject = new GameObject("HologramGrid");
            gridObject.transform.SetParent(target, false);
            
            RectTransform gridRect = gridObject.AddComponent<RectTransform>();
            gridRect.anchorMin = Vector2.zero;
            gridRect.anchorMax = Vector2.one;
            gridRect.sizeDelta = Vector2.zero;
            gridRect.anchoredPosition = Vector2.zero;
            
            // Create grid mesh
            MeshFilter meshFilter = gridObject.AddComponent<MeshFilter>();
            MeshRenderer meshRenderer = gridObject.AddComponent<MeshRenderer>();
            
            Mesh gridMesh = GenerateGridMesh(target.rect.width, target.rect.height);
            meshFilter.mesh = gridMesh;
            
            if (gridMaterial != null)
            {
                meshRenderer.material = new Material(gridMaterial);
                meshRenderer.material.SetColor("_Color", hologramColor);
            }
            
            instance.effectObjects.Add(gridObject);
            
            // Animate grid
            StartCoroutine(AnimateGrid(gridObject, instance));
        }
        
        private Mesh GenerateGridMesh(float width, float height)
        {
            Mesh mesh = new Mesh();
            List<Vector3> vertices = new List<Vector3>();
            List<int> indices = new List<int>();
            
            float cellWidth = width / gridResolution;
            float cellHeight = height / gridResolution;
            
            // Vertical lines
            for (int x = 0; x <= gridResolution; x++)
            {
                float xPos = -width / 2 + x * cellWidth;
                vertices.Add(new Vector3(xPos, -height / 2, 0));
                vertices.Add(new Vector3(xPos, height / 2, 0));
                
                int baseIndex = x * 2;
                indices.Add(baseIndex);
                indices.Add(baseIndex + 1);
            }
            
            // Horizontal lines
            int verticalLineCount = vertices.Count;
            for (int y = 0; y <= gridResolution; y++)
            {
                float yPos = -height / 2 + y * cellHeight;
                vertices.Add(new Vector3(-width / 2, yPos, 0));
                vertices.Add(new Vector3(width / 2, yPos, 0));
                
                int baseIndex = verticalLineCount + y * 2;
                indices.Add(baseIndex);
                indices.Add(baseIndex + 1);
            }
            
            mesh.vertices = vertices.ToArray();
            mesh.SetIndices(indices.ToArray(), MeshTopology.Lines, 0);
            mesh.RecalculateBounds();
            
            return mesh;
        }
        
        private IEnumerator AnimateGrid(GameObject gridObject, HologramInstance instance)
        {
            MeshRenderer renderer = gridObject.GetComponent<MeshRenderer>();
            if (renderer == null || renderer.material == null) yield break;
            
            while (gridObject != null && gridObject.activeInHierarchy)
            {
                float time = Time.time - instance.creationTime;
                
                // Pulse grid opacity
                float alpha = Mathf.Sin(time * gridPulseSpeed) * 0.2f + 0.8f;
                Color gridColor = hologramColor;
                gridColor.a *= alpha;
                renderer.material.SetColor("_Color", gridColor);
                
                // Animate grid offset
                float offset = time * 0.1f;
                renderer.material.SetTextureOffset("_MainTex", new Vector2(offset, offset));
                
                yield return null;
            }
        }
        
        #endregion
        
        #region Scanline Effect
        
        private void CreateScanlineEffect(RectTransform target, HologramInstance instance)
        {
            GameObject scanlineObject = new GameObject("HologramScanline");
            scanlineObject.transform.SetParent(target, false);
            
            RectTransform scanlineRect = scanlineObject.AddComponent<RectTransform>();
            scanlineRect.anchorMin = new Vector2(0, 1);
            scanlineRect.anchorMax = new Vector2(1, 1);
            scanlineRect.sizeDelta = new Vector2(0, 2);
            scanlineRect.pivot = new Vector2(0.5f, 0.5f);
            
            Image scanlineImage = scanlineObject.AddComponent<Image>();
            scanlineImage.color = hologramEmissionColor;
            
            if (scanlineMaterial != null)
            {
                scanlineImage.material = new Material(scanlineMaterial);
            }
            
            instance.effectObjects.Add(scanlineObject);
            
            // Animate scanline
            StartCoroutine(AnimateScanline(scanlineRect, target.rect.height));
        }
        
        private IEnumerator AnimateScanline(RectTransform scanline, float height)
        {
            while (scanline != null && scanline.gameObject.activeInHierarchy)
            {
                // Move scanline from top to bottom
                animController.AnimateFloat($"Scanline_{scanline.GetInstanceID()}",
                    0, -height, scanlineSpeed,
                    (y) => scanline.anchoredPosition = new Vector2(0, y),
                    () => scanline.anchoredPosition = Vector2.zero);
                
                yield return new WaitForSeconds(scanlineSpeed + 0.5f);
            }
        }
        
        #endregion
        
        #region Core Effects
        
        private IEnumerator HologramEffectCoroutine(RectTransform target, HologramInstance instance)
        {
            while (target != null && activeHolograms.ContainsKey(target))
            {
                float time = Time.time - instance.creationTime;
                
                // Update material properties
                if (instance.instanceMaterial != null)
                {
                    // Noise animation
                    instance.instanceMaterial.SetFloat("_NoiseOffset", time * 0.5f);
                    
                    // Distortion
                    if (enableDistortion)
                    {
                        float distortion = Mathf.Sin(time * distortionSpeed) * distortionAmount;
                        instance.instanceMaterial.SetFloat("_DistortionAmount", distortion);
                    }
                    
                    // Flicker effect
                    if (Random.Range(0f, 1f) < flickerFrequency * Time.deltaTime)
                    {
                        float flicker = Random.Range(0.8f, 1f);
                        instance.instanceMaterial.SetFloat("_Brightness", flicker);
                    }
                    else
                    {
                        instance.instanceMaterial.SetFloat("_Brightness", 1f);
                    }
                }
                
                yield return null;
            }
        }
        
        #endregion
        
        #region Build/Dismiss Animations
        
        private IEnumerator BuildHologramAnimation(RectTransform target)
        {
            CanvasGroup canvasGroup = target.GetComponent<CanvasGroup>();
            if (canvasGroup == null)
            {
                canvasGroup = target.gameObject.AddComponent<CanvasGroup>();
            }
            
            // Initial state
            canvasGroup.alpha = 0;
            target.localScale = new Vector3(1, 0.01f, 1);
            
            // Build effect
            float buildTime = projectionBuildTime;
            
            // Phase 1: Vertical expansion
            animController.AnimateVector3($"HoloBuild_Scale_{target.GetInstanceID()}",
                target.localScale, Vector3.one, buildTime * 0.3f,
                (scale) => target.localScale = scale);
            
            // Phase 2: Fade in with noise
            yield return new WaitForSeconds(buildTime * 0.1f);
            
            animController.AnimateFloat($"HoloBuild_Alpha_{target.GetInstanceID()}",
                0, 1, buildTime * 0.7f,
                (alpha) => 
                {
                    canvasGroup.alpha = alpha;
                    
                    // Add random flicker during build
                    if (Random.Range(0f, 1f) < 0.3f)
                    {
                        canvasGroup.alpha *= Random.Range(0.7f, 1f);
                    }
                });
            
            yield return new WaitForSeconds(buildTime);
            
            // Stabilize
            canvasGroup.alpha = 1;
            target.localScale = Vector3.one;
        }
        
        private IEnumerator DismissHologramAnimation(RectTransform target, HologramInstance instance)
        {
            CanvasGroup canvasGroup = target.GetComponent<CanvasGroup>();
            if (canvasGroup == null)
            {
                canvasGroup = target.gameObject.AddComponent<CanvasGroup>();
            }
            
            float dismissTime = 0.5f;
            
            // Glitch before dismiss
            yield return HologramGlitchEffect(target, instance, 0.2f);
            
            // Collapse effect
            animController.AnimateVector3($"HoloCollapse_{target.GetInstanceID()}",
                target.localScale, new Vector3(1, 0.01f, 1), dismissTime * 0.7f,
                (scale) => target.localScale = scale);
            
            animController.AnimateFloat($"HoloDismiss_Alpha_{target.GetInstanceID()}",
                canvasGroup.alpha, 0, dismissTime,
                (alpha) => canvasGroup.alpha = alpha);
            
            yield return new WaitForSeconds(dismissTime);
            
            CleanupHologram(target, instance);
        }
        
        #endregion
        
        #region Projection Effect
        
        private IEnumerator ProjectHologram(Vector3 startPos, Vector3 endPos, RectTransform content)
        {
            if (projectionBeamPrefab == null) yield break;
            
            // Create projection beam
            GameObject beam = Instantiate(projectionBeamPrefab, transform);
            LineRenderer lineRenderer = beam.GetComponent<LineRenderer>();
            
            if (lineRenderer == null)
            {
                lineRenderer = beam.AddComponent<LineRenderer>();
                lineRenderer.startWidth = 0.1f;
                lineRenderer.endWidth = 0.5f;
                lineRenderer.material = hologramMaterial;
                lineRenderer.startColor = hologramColor;
                lineRenderer.endColor = hologramColor;
            }
            
            // Animate beam extension
            float extendTime = 0.3f;
            float elapsed = 0;
            
            while (elapsed < extendTime)
            {
                elapsed += Time.deltaTime;
                float t = elapsed / extendTime;
                
                Vector3 currentEnd = Vector3.Lerp(startPos, endPos, t);
                lineRenderer.SetPosition(0, startPos);
                lineRenderer.SetPosition(1, currentEnd);
                
                yield return null;
            }
            
            // Build hologram at end position
            content.position = endPos;
            ApplyHologramEffect(content, true);
            
            // Fade out beam
            animController.AnimateFloat($"BeamFade_{beam.GetInstanceID()}",
                1, 0, 0.3f,
                (alpha) =>
                {
                    Color c = lineRenderer.startColor;
                    c.a = alpha;
                    lineRenderer.startColor = c;
                    lineRenderer.endColor = c;
                },
                () => Destroy(beam));
        }
        
        #endregion
        
        #region Glitch Effect
        
        private IEnumerator HologramGlitchEffect(RectTransform target, HologramInstance instance, float duration)
        {
            float elapsed = 0;
            Vector3 originalScale = target.localScale;
            Vector2 originalPosition = target.anchoredPosition;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                
                // Position glitch
                if (Random.Range(0f, 1f) < 0.3f)
                {
                    target.anchoredPosition = originalPosition + new Vector2(
                        Random.Range(-10f, 10f),
                        Random.Range(-10f, 10f)
                    );
                }
                else
                {
                    target.anchoredPosition = originalPosition;
                }
                
                // Scale glitch
                if (Random.Range(0f, 1f) < 0.2f)
                {
                    target.localScale = originalScale + new Vector3(
                        Random.Range(-0.1f, 0.1f),
                        Random.Range(-0.1f, 0.1f),
                        0
                    );
                }
                else
                {
                    target.localScale = originalScale;
                }
                
                // Material glitch
                if (instance.instanceMaterial != null)
                {
                    instance.instanceMaterial.SetFloat("_GlitchIntensity", Random.Range(0f, 1f));
                    instance.instanceMaterial.SetFloat("_ChromaticAberration", Random.Range(0f, 0.1f));
                }
                
                yield return new WaitForSeconds(0.05f);
            }
            
            // Reset
            target.localScale = originalScale;
            target.anchoredPosition = originalPosition;
            
            if (instance.instanceMaterial != null)
            {
                instance.instanceMaterial.SetFloat("_GlitchIntensity", 0);
                instance.instanceMaterial.SetFloat("_ChromaticAberration", 0);
            }
        }
        
        #endregion
        
        #region Cleanup
        
        private void CleanupHologram(RectTransform target, HologramInstance instance)
        {
            // Stop coroutines
            if (instance.effectCoroutine != null)
            {
                StopCoroutine(instance.effectCoroutine);
            }
            
            // Remove effect objects
            foreach (var obj in instance.effectObjects)
            {
                if (obj != null)
                {
                    Destroy(obj);
                }
            }
            
            // Reset materials
            Image[] images = target.GetComponentsInChildren<Image>();
            foreach (var image in images)
            {
                image.material = null;
            }
            
            TextMeshProUGUI[] texts = target.GetComponentsInChildren<TextMeshProUGUI>();
            foreach (var text in texts)
            {
                text.fontMaterial = null;
            }
            
            // Destroy instance material
            if (instance.instanceMaterial != null)
            {
                Destroy(instance.instanceMaterial);
            }
            
            activeHolograms.Remove(target);
        }
        
        void OnDestroy()
        {
            // Cleanup all active holograms
            foreach (var kvp in activeHolograms)
            {
                CleanupHologram(kvp.Key, kvp.Value);
            }
        }
        
        #endregion
    }
}