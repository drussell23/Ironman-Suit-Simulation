using UnityEditor;
using UnityEngine;
using System.IO;
using System.Linq;

namespace IronManSim.Editor
{
    /// <summary>
    /// Automated WebGL build configuration for Iron Man Experience
    /// </summary>
    public class WebGLBuilder : EditorWindow
    {
        private static BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions();
        private static string buildPath = "Build/WebGL";
        
        [MenuItem("IronMan/Build/WebGL Build")]
        public static void ShowWindow()
        {
            GetWindow<WebGLBuilder>("WebGL Builder");
        }
        
        void OnGUI()
        {
            GUILayout.Label("Iron Man Experience - WebGL Builder", EditorStyles.boldLabel);
            
            EditorGUILayout.Space();
            
            buildPath = EditorGUILayout.TextField("Build Path:", buildPath);
            
            EditorGUILayout.Space();
            
            if (GUILayout.Button("Configure WebGL Settings"))
            {
                ConfigureWebGLSettings();
            }
            
            if (GUILayout.Button("Build for WebGL"))
            {
                BuildProject();
            }
            
            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(
                "This will build the Iron Man Experience for web browsers. " +
                "Make sure to test in Chrome or Firefox for best results.", 
                MessageType.Info
            );
        }
        
        public static void ConfigureWebGLSettings()
        {
            Debug.Log("Configuring WebGL settings for Iron Man Experience...");
            
            // Player settings
            PlayerSettings.WebGL.linkerTarget = WebGLLinkerTarget.Wasm;
            PlayerSettings.WebGL.memorySize = 512;
            PlayerSettings.WebGL.exceptionSupport = WebGLExceptionSupport.None;
            PlayerSettings.WebGL.compressionFormat = WebGLCompressionFormat.Gzip;
            
            // Enable WebAssembly streaming
            PlayerSettings.WebGL.wasmStreaming = true;
            PlayerSettings.WebGL.dataCaching = true;
            
            // Graphics settings
            PlayerSettings.SetGraphicsAPIs(BuildTarget.WebGL, new[] { 
                UnityEngine.Rendering.GraphicsDeviceType.OpenGLES3,
                UnityEngine.Rendering.GraphicsDeviceType.OpenGLES2 
            });
            
            // Quality settings for web
            QualitySettings.pixelLightCount = 2;
            QualitySettings.shadows = ShadowQuality.HardOnly;
            QualitySettings.shadowResolution = ShadowResolution.Medium;
            QualitySettings.antiAliasing = 2;
            
            // Company and product info
            PlayerSettings.companyName = "Stark Industries";
            PlayerSettings.productName = "Iron Man Suit Experience";
            PlayerSettings.applicationIdentifier = "com.starkindustries.ironmanexperience";
            
            // Splash screen
            PlayerSettings.SplashScreen.show = true;
            PlayerSettings.SplashScreen.showUnityLogo = false;
            
            Debug.Log("WebGL settings configured successfully!");
        }
        
        [MenuItem("IronMan/Build/Quick WebGL Build")]
        public static void QuickBuild()
        {
            ConfigureWebGLSettings();
            BuildProject();
        }
        
        public static void BuildProject()
        {
            // Get scenes to build
            string[] scenes = GetScenesToBuild();
            
            if (scenes.Length == 0)
            {
                Debug.LogError("No scenes found in build settings!");
                return;
            }
            
            // Setup build options
            buildPlayerOptions.scenes = scenes;
            buildPlayerOptions.locationPathName = GetBuildPath();
            buildPlayerOptions.target = BuildTarget.WebGL;
            buildPlayerOptions.options = BuildOptions.None;
            
            // Add development build option if in editor
            if (EditorUserBuildSettings.development)
            {
                buildPlayerOptions.options |= BuildOptions.Development;
            }
            
            Debug.Log($"Starting WebGL build to: {buildPlayerOptions.locationPathName}");
            
            // Perform build
            var report = BuildPipeline.BuildPlayer(buildPlayerOptions);
            
            if (report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded)
            {
                Debug.Log($"Build succeeded! Total time: {report.summary.totalTime}");
                
                // Create index.html with custom template
                CreateCustomIndexHTML(buildPlayerOptions.locationPathName);
                
                // Open build folder
                EditorUtility.RevealInFinder(buildPlayerOptions.locationPathName);
            }
            else
            {
                Debug.LogError($"Build failed with {report.summary.totalErrors} errors");
            }
        }
        
        private static string[] GetScenesToBuild()
        {
            return EditorBuildSettings.scenes
                .Where(scene => scene.enabled)
                .Select(scene => scene.path)
                .ToArray();
        }
        
        private static string GetBuildPath()
        {
            string projectPath = Directory.GetParent(Application.dataPath).FullName;
            return Path.Combine(projectPath, "..", "frontend", "IronManExperience", "Build");
        }
        
        private static void CreateCustomIndexHTML(string buildPath)
        {
            string indexPath = Path.Combine(buildPath, "index.html");
            
            if (File.Exists(indexPath))
            {
                // Read default template
                string html = File.ReadAllText(indexPath);
                
                // Customize HTML
                html = html.Replace("<title>Unity WebGL Player", "<title>Iron Man Suit Experience");
                html = html.Replace("</head>", @"
    <style>
        body { 
            background-color: #000; 
            margin: 0; 
            padding: 0;
            overflow: hidden;
        }
        #unity-container { 
            position: absolute;
            width: 100%;
            height: 100%;
        }
        #unity-canvas {
            width: 100%;
            height: 100%;
            display: block;
        }
        #unity-loading-bar {
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        }
        #unity-progress-bar-full {
            background: #00a8ff;
        }
        .loading-text {
            position: absolute;
            top: 60%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #00a8ff;
            font-family: Arial, sans-serif;
            font-size: 24px;
            text-align: center;
        }
    </style>
</head>");
                
                // Add loading text
                html = html.Replace("<div id=\"unity-loading-bar\">", @"
<div class='loading-text'>
    JARVIS SYSTEM INITIALIZING<br>
    <span style='font-size: 14px;'>Stark Industries</span>
</div>
<div id='unity-loading-bar'>");
                
                // Write customized HTML
                File.WriteAllText(indexPath, html);
                Debug.Log("Custom index.html created");
            }
        }
        
        [MenuItem("IronMan/Build/Open Build Folder")]
        public static void OpenBuildFolder()
        {
            string path = GetBuildPath();
            if (Directory.Exists(path))
            {
                EditorUtility.RevealInFinder(path);
            }
            else
            {
                Debug.LogWarning("Build folder doesn't exist yet. Run a build first.");
            }
        }
    }
}