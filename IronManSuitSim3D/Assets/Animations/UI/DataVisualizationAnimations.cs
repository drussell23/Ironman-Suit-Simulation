using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;

namespace IronManSim.UI.Animations
{
    /// <summary>
    /// Animated data visualization components for Iron Man HUD
    /// </summary>
    public class DataVisualizationAnimations : MonoBehaviour
    {
        [System.Serializable]
        public class GraphData
        {
            public string label;
            public float value;
            public Color color = Color.cyan;
        }
        
        [System.Serializable]
        public class ChartConfig
        {
            public ChartType type = ChartType.Line;
            public int maxDataPoints = 50;
            public float updateInterval = 0.1f;
            public AnimationCurve animationCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
            public bool autoScale = true;
            public float minValue = 0f;
            public float maxValue = 100f;
        }
        
        public enum ChartType
        {
            Line,
            Bar,
            Radar,
            Pie,
            Scatter,
            Heatmap,
            WaveForm
        }
        
        [Header("Line Graph")]
        [SerializeField] private RectTransform lineGraphContainer;
        [SerializeField] private GameObject linePointPrefab;
        [SerializeField] private LineRenderer lineRenderer;
        [SerializeField] private ChartConfig lineChartConfig;
        private Queue<float> lineDataQueue = new Queue<float>();
        private List<GameObject> linePoints = new List<GameObject>();
        
        [Header("Bar Chart")]
        [SerializeField] private RectTransform barChartContainer;
        [SerializeField] private GameObject barPrefab;
        [SerializeField] private float barSpacing = 5f;
        [SerializeField] private ChartConfig barChartConfig;
        private List<RectTransform> bars = new List<RectTransform>();
        
        [Header("Radar Chart")]
        [SerializeField] private RectTransform radarChartContainer;
        [SerializeField] private int radarAxes = 6;
        [SerializeField] private float radarRadius = 100f;
        [SerializeField] private Material radarMaterial;
        private Mesh radarMesh;
        private MeshFilter radarMeshFilter;
        
        [Header("3D Holographic Graph")]
        [SerializeField] private Transform holographicGraphContainer;
        [SerializeField] private GameObject holographicPointPrefab;
        [SerializeField] private int holographicGridSize = 10;
        [SerializeField] private float holographicScale = 1f;
        private GameObject[,] holographicPoints;
        
        [Header("Waveform Display")]
        [SerializeField] private RectTransform waveformContainer;
        [SerializeField] private LineRenderer waveformLine;
        [SerializeField] private int waveformResolution = 128;
        [SerializeField] private float waveformAmplitude = 50f;
        private float[] waveformData;
        
        [Header("Data Stream")]
        [SerializeField] private RectTransform dataStreamContainer;
        [SerializeField] private TextMeshProUGUI dataStreamTextPrefab;
        [SerializeField] private float streamSpeed = 100f;
        [SerializeField] private int maxStreamItems = 20;
        private Queue<TextMeshProUGUI> dataStreamPool = new Queue<TextMeshProUGUI>();
        private List<DataStreamItem> activeStreamItems = new List<DataStreamItem>();
        
        [Header("Performance Metrics")]
        [SerializeField] private TextMeshProUGUI fpsDisplay;
        [SerializeField] private TextMeshProUGUI cpuDisplay;
        [SerializeField] private TextMeshProUGUI memoryDisplay;
        [SerializeField] private Image performanceBar;
        
        [Header("Visual Settings")]
        [SerializeField] private bool enableGlowEffect = true;
        [SerializeField] private float glowIntensity = 2f;
        [SerializeField] private bool enableGridAnimation = true;
        [SerializeField] private float gridAnimationSpeed = 1f;
        
        private UIAnimationController animController;
        private Dictionary<string, Coroutine> activeAnimations = new Dictionary<string, Coroutine>();
        
        private class DataStreamItem
        {
            public TextMeshProUGUI text;
            public float position;
            public float speed;
        }
        
        void Start()
        {
            animController = UIAnimationController.Instance;
            InitializeVisualizations();
        }
        
        #region Initialization
        
        private void InitializeVisualizations()
        {
            // Initialize line graph
            if (lineGraphContainer != null)
            {
                InitializeLineGraph();
            }
            
            // Initialize bar chart
            if (barChartContainer != null)
            {
                InitializeBarChart();
            }
            
            // Initialize radar chart
            if (radarChartContainer != null)
            {
                InitializeRadarChart();
            }
            
            // Initialize holographic graph
            if (holographicGraphContainer != null)
            {
                InitializeHolographicGraph();
            }
            
            // Initialize waveform
            if (waveformContainer != null && waveformLine != null)
            {
                InitializeWaveform();
            }
            
            // Initialize data stream
            if (dataStreamContainer != null && dataStreamTextPrefab != null)
            {
                InitializeDataStream();
            }
        }
        
        #endregion
        
        #region Line Graph
        
        private void InitializeLineGraph()
        {
            // Create line renderer if not assigned
            if (lineRenderer == null)
            {
                GameObject lineObj = new GameObject("LineRenderer");
                lineObj.transform.SetParent(lineGraphContainer, false);
                lineRenderer = lineObj.AddComponent<LineRenderer>();
                lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
                lineRenderer.startColor = Color.cyan;
                lineRenderer.endColor = Color.cyan;
                lineRenderer.startWidth = 2f;
                lineRenderer.endWidth = 2f;
            }
            
            // Start line graph animation
            StartAnimation("LineGraph", UpdateLineGraphCoroutine());
        }
        
        public void AddLineGraphData(float value)
        {
            lineDataQueue.Enqueue(value);
            
            // Limit queue size
            while (lineDataQueue.Count > lineChartConfig.maxDataPoints)
            {
                lineDataQueue.Dequeue();
            }
        }
        
        private IEnumerator UpdateLineGraphCoroutine()
        {
            while (true)
            {
                UpdateLineGraphVisual();
                yield return new WaitForSeconds(lineChartConfig.updateInterval);
            }
        }
        
        private void UpdateLineGraphVisual()
        {
            if (lineDataQueue.Count < 2) return;
            
            float[] data = lineDataQueue.ToArray();
            Vector3[] positions = new Vector3[data.Length];
            
            float width = lineGraphContainer.rect.width;
            float height = lineGraphContainer.rect.height;
            float xStep = width / (data.Length - 1);
            
            // Find min/max for scaling
            float minVal = lineChartConfig.minValue;
            float maxVal = lineChartConfig.maxValue;
            
            if (lineChartConfig.autoScale)
            {
                minVal = Mathf.Min(data);
                maxVal = Mathf.Max(data);
            }
            
            // Create line positions
            for (int i = 0; i < data.Length; i++)
            {
                float x = i * xStep - width / 2;
                float normalizedValue = Mathf.InverseLerp(minVal, maxVal, data[i]);
                float y = normalizedValue * height - height / 2;
                
                positions[i] = new Vector3(x, y, 0);
            }
            
            // Update line renderer
            lineRenderer.positionCount = positions.Length;
            lineRenderer.SetPositions(positions);
            
            // Animate newest point
            if (linePointPrefab != null && data.Length > 0)
            {
                AnimateDataPoint(positions[positions.Length - 1], data[data.Length - 1]);
            }
        }
        
        private void AnimateDataPoint(Vector3 position, float value)
        {
            // Create or reuse point
            GameObject point = null;
            if (linePoints.Count > 20) // Reuse old points
            {
                point = linePoints[0];
                linePoints.RemoveAt(0);
            }
            else
            {
                point = Instantiate(linePointPrefab, lineGraphContainer);
            }
            
            linePoints.Add(point);
            point.transform.localPosition = position;
            
            // Animate point
            RectTransform pointRect = point.GetComponent<RectTransform>();
            pointRect.localScale = Vector3.zero;
            
            animController.AnimateVector3($"DataPoint_{point.GetInstanceID()}",
                Vector3.zero, Vector3.one, 0.3f,
                (scale) => pointRect.localScale = scale,
                () =>
                {
                    // Fade out after a delay
                    StartCoroutine(FadeOutDataPoint(point));
                });
        }
        
        private IEnumerator FadeOutDataPoint(GameObject point)
        {
            yield return new WaitForSeconds(2f);
            
            CanvasGroup group = point.GetComponent<CanvasGroup>();
            if (group == null)
            {
                group = point.AddComponent<CanvasGroup>();
            }
            
            animController.FadeOut(group, 0.5f, () =>
            {
                point.SetActive(false);
            });
        }
        
        #endregion
        
        #region Bar Chart
        
        private void InitializeBarChart()
        {
            // Pre-create bars
            for (int i = 0; i < 10; i++)
            {
                CreateBar();
            }
        }
        
        private RectTransform CreateBar()
        {
            GameObject bar = barPrefab != null ? 
                Instantiate(barPrefab, barChartContainer) : 
                CreateDefaultBar();
            
            RectTransform barRect = bar.GetComponent<RectTransform>();
            bars.Add(barRect);
            
            return barRect;
        }
        
        private GameObject CreateDefaultBar()
        {
            GameObject bar = new GameObject("Bar");
            bar.transform.SetParent(barChartContainer, false);
            
            RectTransform rect = bar.AddComponent<RectTransform>();
            rect.sizeDelta = new Vector2(30, 100);
            
            Image image = bar.AddComponent<Image>();
            image.color = Color.cyan;
            
            return bar;
        }
        
        public void UpdateBarChart(List<GraphData> data)
        {
            StartCoroutine(AnimateBarChartUpdate(data));
        }
        
        private IEnumerator AnimateBarChartUpdate(List<GraphData> data)
        {
            // Ensure we have enough bars
            while (bars.Count < data.Count)
            {
                CreateBar();
            }
            
            float containerWidth = barChartContainer.rect.width;
            float barWidth = (containerWidth - barSpacing * (data.Count - 1)) / data.Count;
            
            for (int i = 0; i < data.Count; i++)
            {
                if (i >= bars.Count) break;
                
                RectTransform bar = bars[i];
                bar.gameObject.SetActive(true);
                
                // Position
                float xPos = -containerWidth / 2 + barWidth / 2 + i * (barWidth + barSpacing);
                bar.anchoredPosition = new Vector2(xPos, bar.anchoredPosition.y);
                
                // Animate height
                float targetHeight = (data[i].value / barChartConfig.maxValue) * barChartContainer.rect.height;
                
                animController.AnimateFloat($"BarHeight_{i}",
                    bar.sizeDelta.y, targetHeight, 0.5f,
                    (height) =>
                    {
                        bar.sizeDelta = new Vector2(barWidth, height);
                        bar.anchoredPosition = new Vector2(bar.anchoredPosition.x, -barChartContainer.rect.height / 2 + height / 2);
                    },
                    null,
                    barChartConfig.animationCurve);
                
                // Update color
                Image barImage = bar.GetComponent<Image>();
                if (barImage != null)
                {
                    animController.AnimateColor($"BarColor_{i}",
                        barImage.color, data[i].color, 0.3f,
                        (color) => barImage.color = color);
                }
                
                // Add label
                TextMeshProUGUI label = bar.GetComponentInChildren<TextMeshProUGUI>();
                if (label != null)
                {
                    label.text = data[i].label;
                }
            }
            
            // Hide unused bars
            for (int i = data.Count; i < bars.Count; i++)
            {
                bars[i].gameObject.SetActive(false);
            }
            
            yield return null;
        }
        
        #endregion
        
        #region Radar Chart
        
        private void InitializeRadarChart()
        {
            // Create mesh for radar chart
            GameObject radarObj = new GameObject("RadarMesh");
            radarObj.transform.SetParent(radarChartContainer, false);
            
            radarMeshFilter = radarObj.AddComponent<MeshFilter>();
            MeshRenderer renderer = radarObj.AddComponent<MeshRenderer>();
            
            if (radarMaterial != null)
            {
                renderer.material = radarMaterial;
            }
            else
            {
                renderer.material = new Material(Shader.Find("Sprites/Default"));
                renderer.material.color = new Color(0, 1, 1, 0.5f);
            }
            
            radarMesh = new Mesh();
            radarMeshFilter.mesh = radarMesh;
        }
        
        public void UpdateRadarChart(float[] values)
        {
            if (values.Length != radarAxes) return;
            
            StartCoroutine(AnimateRadarChartUpdate(values));
        }
        
        private IEnumerator AnimateRadarChartUpdate(float[] targetValues)
        {
            float[] currentValues = new float[radarAxes];
            
            // Get current values from mesh
            if (radarMesh != null && radarMesh.vertices.Length > 0)
            {
                Vector3[] vertices = radarMesh.vertices;
                for (int i = 0; i < radarAxes; i++)
                {
                    currentValues[i] = vertices[i].magnitude / radarRadius;
                }
            }
            
            float duration = 0.5f;
            float elapsed = 0;
            
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float t = elapsed / duration;
                
                // Interpolate values
                float[] interpolatedValues = new float[radarAxes];
                for (int i = 0; i < radarAxes; i++)
                {
                    interpolatedValues[i] = Mathf.Lerp(currentValues[i], targetValues[i], t);
                }
                
                UpdateRadarMesh(interpolatedValues);
                yield return null;
            }
            
            UpdateRadarMesh(targetValues);
        }
        
        private void UpdateRadarMesh(float[] values)
        {
            Vector3[] vertices = new Vector3[radarAxes + 1];
            int[] triangles = new int[radarAxes * 3];
            
            // Center vertex
            vertices[0] = Vector3.zero;
            
            // Outer vertices
            for (int i = 0; i < radarAxes; i++)
            {
                float angle = i * Mathf.PI * 2 / radarAxes;
                float distance = values[i] * radarRadius;
                
                vertices[i + 1] = new Vector3(
                    Mathf.Cos(angle) * distance,
                    Mathf.Sin(angle) * distance,
                    0
                );
                
                // Create triangles
                int triIndex = i * 3;
                triangles[triIndex] = 0;
                triangles[triIndex + 1] = i + 1;
                triangles[triIndex + 2] = (i + 1) % radarAxes + 1;
            }
            
            radarMesh.Clear();
            radarMesh.vertices = vertices;
            radarMesh.triangles = triangles;
            radarMesh.RecalculateNormals();
        }
        
        #endregion
        
        #region Holographic 3D Graph
        
        private void InitializeHolographicGraph()
        {
            holographicPoints = new GameObject[holographicGridSize, holographicGridSize];
            
            for (int x = 0; x < holographicGridSize; x++)
            {
                for (int z = 0; z < holographicGridSize; z++)
                {
                    GameObject point = holographicPointPrefab != null ?
                        Instantiate(holographicPointPrefab, holographicGraphContainer) :
                        CreateDefaultHolographicPoint();
                    
                    float xPos = (x - holographicGridSize / 2f) * holographicScale;
                    float zPos = (z - holographicGridSize / 2f) * holographicScale;
                    
                    point.transform.localPosition = new Vector3(xPos, 0, zPos);
                    holographicPoints[x, z] = point;
                }
            }
        }
        
        private GameObject CreateDefaultHolographicPoint()
        {
            GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            point.transform.localScale = Vector3.one * 0.1f;
            
            Renderer renderer = point.GetComponent<Renderer>();
            renderer.material = new Material(Shader.Find("Unlit/Color"));
            renderer.material.color = Color.cyan;
            
            return point;
        }
        
        public void UpdateHolographicGraph(float[,] heightData)
        {
            if (heightData.GetLength(0) != holographicGridSize ||
                heightData.GetLength(1) != holographicGridSize)
            {
                return;
            }
            
            StartCoroutine(AnimateHolographicUpdate(heightData));
        }
        
        private IEnumerator AnimateHolographicUpdate(float[,] targetHeights)
        {
            float duration = 0.5f;
            
            for (int x = 0; x < holographicGridSize; x++)
            {
                for (int z = 0; z < holographicGridSize; z++)
                {
                    GameObject point = holographicPoints[x, z];
                    Vector3 currentPos = point.transform.localPosition;
                    Vector3 targetPos = new Vector3(
                        currentPos.x,
                        targetHeights[x, z] * holographicScale,
                        currentPos.z
                    );
                    
                    int xCopy = x;
                    int zCopy = z;
                    
                    animController.AnimateVector3($"Holo_{x}_{z}",
                        currentPos, targetPos, duration,
                        (pos) => holographicPoints[xCopy, zCopy].transform.localPosition = pos);
                    
                    // Update color based on height
                    Renderer renderer = point.GetComponent<Renderer>();
                    if (renderer != null)
                    {
                        Color targetColor = Color.Lerp(Color.blue, Color.red, targetHeights[x, z]);
                        animController.AnimateColor($"HoloColor_{x}_{z}",
                            renderer.material.color, targetColor, duration,
                            (color) => renderer.material.color = color);
                    }
                }
                
                yield return new WaitForSeconds(0.01f); // Stagger animation
            }
        }
        
        #endregion
        
        #region Waveform
        
        private void InitializeWaveform()
        {
            waveformData = new float[waveformResolution];
            waveformLine.positionCount = waveformResolution;
            
            StartAnimation("Waveform", UpdateWaveformCoroutine());
        }
        
        public void UpdateWaveformData(float[] audioData)
        {
            if (audioData.Length > 0)
            {
                // Resample to match resolution
                for (int i = 0; i < waveformResolution; i++)
                {
                    int sourceIndex = (int)((float)i / waveformResolution * audioData.Length);
                    waveformData[i] = audioData[sourceIndex];
                }
            }
        }
        
        private IEnumerator UpdateWaveformCoroutine()
        {
            while (true)
            {
                UpdateWaveformVisual();
                yield return new WaitForSeconds(0.02f); // 50 FPS update
            }
        }
        
        private void UpdateWaveformVisual()
        {
            float width = waveformContainer.rect.width;
            float xStep = width / waveformResolution;
            
            for (int i = 0; i < waveformResolution; i++)
            {
                float x = i * xStep - width / 2;
                float y = waveformData[i] * waveformAmplitude;
                
                // Add some visual smoothing
                if (i > 0 && i < waveformResolution - 1)
                {
                    float smoothedY = (waveformData[i - 1] + waveformData[i] + waveformData[i + 1]) / 3f;
                    y = Mathf.Lerp(y, smoothedY * waveformAmplitude, 0.5f);
                }
                
                waveformLine.SetPosition(i, new Vector3(x, y, 0));
            }
            
            // Animate color based on amplitude
            float avgAmplitude = 0;
            foreach (float val in waveformData)
            {
                avgAmplitude += Mathf.Abs(val);
            }
            avgAmplitude /= waveformResolution;
            
            Color waveColor = Color.Lerp(Color.cyan, Color.yellow, avgAmplitude);
            waveformLine.startColor = waveColor;
            waveformLine.endColor = waveColor;
        }
        
        #endregion
        
        #region Data Stream
        
        private void InitializeDataStream()
        {
            // Create pool of text objects
            for (int i = 0; i < maxStreamItems; i++)
            {
                TextMeshProUGUI streamText = Instantiate(dataStreamTextPrefab, dataStreamContainer);
                streamText.gameObject.SetActive(false);
                dataStreamPool.Enqueue(streamText);
            }
            
            StartAnimation("DataStream", UpdateDataStreamCoroutine());
        }
        
        public void AddDataStreamEntry(string data, Color color)
        {
            if (dataStreamPool.Count == 0) return;
            
            TextMeshProUGUI streamText = dataStreamPool.Dequeue();
            streamText.gameObject.SetActive(true);
            streamText.text = data;
            streamText.color = color;
            
            RectTransform rect = streamText.GetComponent<RectTransform>();
            rect.anchoredPosition = new Vector2(dataStreamContainer.rect.width / 2 + 100, 
                Random.Range(-dataStreamContainer.rect.height / 2, dataStreamContainer.rect.height / 2));
            
            DataStreamItem item = new DataStreamItem
            {
                text = streamText,
                position = rect.anchoredPosition.x,
                speed = Random.Range(streamSpeed * 0.8f, streamSpeed * 1.2f)
            };
            
            activeStreamItems.Add(item);
            
            // Fade in
            CanvasGroup group = streamText.GetComponent<CanvasGroup>();
            if (group == null)
            {
                group = streamText.gameObject.AddComponent<CanvasGroup>();
            }
            group.alpha = 0;
            animController.FadeIn(group, 0.2f);
        }
        
        private IEnumerator UpdateDataStreamCoroutine()
        {
            while (true)
            {
                for (int i = activeStreamItems.Count - 1; i >= 0; i--)
                {
                    DataStreamItem item = activeStreamItems[i];
                    
                    // Move left
                    item.position -= item.speed * Time.deltaTime;
                    RectTransform rect = item.text.GetComponent<RectTransform>();
                    rect.anchoredPosition = new Vector2(item.position, rect.anchoredPosition.y);
                    
                    // Remove if off screen
                    if (item.position < -dataStreamContainer.rect.width / 2 - 100)
                    {
                        item.text.gameObject.SetActive(false);
                        dataStreamPool.Enqueue(item.text);
                        activeStreamItems.RemoveAt(i);
                    }
                }
                
                yield return null;
            }
        }
        
        #endregion
        
        #region Performance Metrics
        
        public void UpdatePerformanceMetrics(float fps, float cpu, float memory)
        {
            if (fpsDisplay != null)
            {
                fpsDisplay.text = $"FPS: {fps:F0}";
                fpsDisplay.color = fps > 30 ? Color.green : (fps > 20 ? Color.yellow : Color.red);
            }
            
            if (cpuDisplay != null)
            {
                cpuDisplay.text = $"CPU: {cpu:F1}%";
            }
            
            if (memoryDisplay != null)
            {
                memoryDisplay.text = $"MEM: {memory:F0} MB";
            }
            
            if (performanceBar != null)
            {
                float performance = fps / 60f; // Normalized to 60 FPS
                animController.AnimateFloat("PerformanceBar",
                    performanceBar.fillAmount, performance, 0.2f,
                    (fill) => performanceBar.fillAmount = fill);
                
                performanceBar.color = Color.Lerp(Color.red, Color.green, performance);
            }
        }
        
        #endregion
        
        #region Utility Methods
        
        private void StartAnimation(string name, IEnumerator animation)
        {
            if (activeAnimations.ContainsKey(name))
            {
                StopCoroutine(activeAnimations[name]);
            }
            
            activeAnimations[name] = StartCoroutine(animation);
        }
        
        void OnDestroy()
        {
            // Stop all animations
            foreach (var animation in activeAnimations.Values)
            {
                if (animation != null)
                {
                    StopCoroutine(animation);
                }
            }
        }
        
        #endregion
    }
}