using UnityEngine;
using UnityEditor;
using System.IO;

namespace IronManSim.Editor
{
    /// <summary>
    /// Creates the complete folder structure for the Iron Man Suit Simulation project
    /// Organized to complement backend and systems programming components
    /// </summary>
    public class CreateFolderStructure : EditorWindow
    {
        private static string[] folderStructure = new string[]
        {
            // Core Asset Folders
            "Assets/Models",
            "Assets/Models/Suit",
            "Assets/Models/Suit/Armor",
            "Assets/Models/Suit/Weapons",
            "Assets/Models/Suit/Thrusters",
            "Assets/Models/Environment",
            "Assets/Models/Props",
            "Assets/Models/Enemies",
            
            // Textures organized by type
            "Assets/Textures",
            "Assets/Textures/IronMan",
            "Assets/Textures/IronMan/PBR",
            "Assets/Textures/IronMan/Emissive",
            "Assets/Textures/IronMan/Normal",
            "Assets/Textures/IronMan/Masks",
            "Assets/Textures/Environment",
            "Assets/Textures/UI",
            "Assets/Textures/Effects",
            
            // Materials
            "Assets/Materials",
            "Assets/Materials/IronMan",
            "Assets/Materials/IronMan/Armor",
            "Assets/Materials/IronMan/Energy",
            "Assets/Materials/IronMan/Weapons",
            "Assets/Materials/Environment",
            "Assets/Materials/Effects",
            "Assets/Materials/UI",
            
            // Prefabs
            "Assets/Prefabs",
            "Assets/Prefabs/Suit",
            "Assets/Prefabs/Suit/Variants",
            "Assets/Prefabs/Weapons",
            "Assets/Prefabs/Effects",
            "Assets/Prefabs/Environment",
            "Assets/Prefabs/UI",
            "Assets/Prefabs/Systems",
            
            // Scripts - Organized to match backend structure
            "Assets/Scripts",
            "Assets/Scripts/Core",
            "Assets/Scripts/Suit",
            "Assets/Scripts/Art",
            "Assets/Scripts/Controllers",
            "Assets/Scripts/Controllers/Flight",
            "Assets/Scripts/Controllers/Combat",
            "Assets/Scripts/Controllers/Input",
            
            // Backend Integration Scripts
            "Assets/Scripts/Integration",
            "Assets/Scripts/Integration/Aerodynamics",
            "Assets/Scripts/Integration/AdaptiveAI",
            "Assets/Scripts/Integration/JARVIS",
            "Assets/Scripts/Integration/Sensors",
            "Assets/Scripts/Integration/Weapons",
            "Assets/Scripts/Integration/Networking",
            
            // Systems Programming Interface
            "Assets/Scripts/SystemsInterface",
            "Assets/Scripts/SystemsInterface/Hardware",
            "Assets/Scripts/SystemsInterface/RealTime",
            "Assets/Scripts/SystemsInterface/Embedded",
            
            // UI System
            "Assets/Scripts/UI",
            "Assets/Scripts/UI/HUD",
            "Assets/Scripts/UI/Menus",
            "Assets/Scripts/UI/Diagnostics",
            
            // Managers
            "Assets/Scripts/Managers",
            "Assets/Scripts/Managers/Systems",
            "Assets/Scripts/Managers/Combat",
            "Assets/Scripts/Managers/Environment",
            
            // Utilities
            "Assets/Scripts/Utilities",
            "Assets/Scripts/Utilities/Extensions",
            "Assets/Scripts/Utilities/Helpers",
            "Assets/Scripts/Utilities/Debug",
            
            // Visual Effects
            "Assets/VFX",
            "Assets/VFX/Particles",
            "Assets/VFX/Particles/Thrusters",
            "Assets/VFX/Particles/Weapons",
            "Assets/VFX/Particles/Impacts",
            "Assets/VFX/Particles/Energy",
            "Assets/VFX/Shaders",
            "Assets/VFX/PostProcessing",
            
            // Audio
            "Assets/Audio",
            "Assets/Audio/SFX",
            "Assets/Audio/SFX/Suit",
            "Assets/Audio/SFX/Weapons",
            "Assets/Audio/SFX/Environment",
            "Assets/Audio/Music",
            "Assets/Audio/Voice",
            "Assets/Audio/Voice/JARVIS",
            "Assets/Audio/Voice/Alerts",
            
            // Animations
            "Assets/Animations",
            "Assets/Animations/Suit",
            "Assets/Animations/Suit/Flight",
            "Assets/Animations/Suit/Combat",
            "Assets/Animations/UI",
            "Assets/Animations/Environment",
            
            // UI Assets
            "Assets/UI",
            "Assets/UI/HUD",
            "Assets/UI/Menus",
            "Assets/UI/Icons",
            "Assets/UI/Fonts",
            "Assets/UI/Sprites",
            
            // Configuration and Data
            "Assets/Configuration",
            "Assets/Configuration/SuitConfigs",
            "Assets/Configuration/WeaponConfigs",
            "Assets/Configuration/AIConfigs",
            "Assets/Configuration/PhysicsConfigs",
            
            // Resources for runtime loading
            "Assets/Resources",
            "Assets/Resources/Prefabs",
            "Assets/Resources/Materials",
            "Assets/Resources/Configs",
            "Assets/Resources/Audio",
            
            // Streaming Assets for large files
            "Assets/StreamingAssets",
            "Assets/StreamingAssets/Videos",
            "Assets/StreamingAssets/Data",
            
            // Documentation
            "Assets/Documentation",
            "Assets/Documentation/API",
            "Assets/Documentation/Guides",
            "Assets/Documentation/Integration",
            
            // Third Party Assets
            "Assets/ThirdParty",
            "Assets/ThirdParty/Plugins",
            "Assets/ThirdParty/Tools",
            
            // Editor Tools
            "Assets/Scripts/Editor",
            "Assets/Scripts/Editor/Tools",
            "Assets/Scripts/Editor/Inspectors",
            "Assets/Scripts/Editor/Windows",
            
            // Testing
            "Assets/Scripts/Tests",
            "Assets/Scripts/Tests/EditMode",
            "Assets/Scripts/Tests/PlayMode",
            "Assets/Scripts/Tests/Integration"
        };
        
        [MenuItem("IronMan/Project Setup/Create Folder Structure")]
        public static void ShowWindow()
        {
            GetWindow<CreateFolderStructure>("Create Folders");
        }
        
        void OnGUI()
        {
            GUILayout.Label("Iron Man Suit Simulation Folder Structure", EditorStyles.boldLabel);
            GUILayout.Space(10);
            
            EditorGUILayout.HelpBox(
                "This will create a complete folder structure for the Iron Man project that:\n" +
                "• Integrates with backend systems\n" +
                "• Supports hardware interfaces\n" +
                "• Organizes all Unity assets\n" +
                "• Provides clear separation of concerns",
                MessageType.Info);
            
            GUILayout.Space(10);
            
            if (GUILayout.Button("Create All Folders", GUILayout.Height(30)))
            {
                CreateFolders();
            }
            
            GUILayout.Space(10);
            
            if (GUILayout.Button("Create README Files", GUILayout.Height(25)))
            {
                CreateReadmeFiles();
            }
        }
        
        private void CreateFolders()
        {
            int created = 0;
            int existing = 0;
            
            foreach (string folderPath in folderStructure)
            {
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                    created++;
                }
                else
                {
                    existing++;
                }
            }
            
            // Create .gitkeep files in empty directories
            foreach (string folderPath in folderStructure)
            {
                string gitkeepPath = Path.Combine(folderPath, ".gitkeep");
                if (!File.Exists(gitkeepPath))
                {
                    File.WriteAllText(gitkeepPath, "# This file ensures Git tracks this empty directory\n");
                }
            }
            
            AssetDatabase.Refresh();
            
            EditorUtility.DisplayDialog("Folder Structure Created",
                $"Created {created} new folders.\n{existing} folders already existed.",
                "OK");
        }
        
        private void CreateReadmeFiles()
        {
            // Scripts/Integration README
            CreateReadme("Assets/Scripts/Integration/README.md",
                "# Backend Integration Scripts\n\n" +
                "This folder contains Unity-side integration scripts for backend systems:\n\n" +
                "- **Aerodynamics**: Interfaces with backend/aerodynamics physics calculations\n" +
                "- **AdaptiveAI**: Connects to backend/adaptive_ai for intelligent behaviors\n" +
                "- **JARVIS**: Voice command integration with backend/jarvis\n" +
                "- **Sensors**: Real-time sensor data from backend/sensors\n" +
                "- **Weapons**: Weapon system coordination with backend/weapons\n" +
                "- **Networking**: Communication protocols for backend API\n");
            
            // Scripts/SystemsInterface README
            CreateReadme("Assets/Scripts/SystemsInterface/README.md",
                "# Systems Programming Interface\n\n" +
                "Unity interfaces for low-level systems:\n\n" +
                "- **Hardware**: Direct hardware communication interfaces\n" +
                "- **RealTime**: Real-time control system interfaces\n" +
                "- **Embedded**: Embedded system communication protocols\n\n" +
                "These scripts bridge Unity with systems_programming C code.");
            
            // Configuration README
            CreateReadme("Assets/Configuration/README.md",
                "# Configuration Files\n\n" +
                "Stores configuration data that mirrors backend configs:\n\n" +
                "- **SuitConfigs**: Suit parameters (matches backend/aerodynamics/config)\n" +
                "- **WeaponConfigs**: Weapon settings (matches backend/weapons configs)\n" +
                "- **AIConfigs**: AI behavior parameters (matches backend/adaptive_ai)\n" +
                "- **PhysicsConfigs**: Physics settings for Unity simulation\n");
            
            // Models README
            CreateReadme("Assets/Models/README.md",
                "# 3D Models\n\n" +
                "- **Suit**: Iron Man suit components and variants\n" +
                "- **Environment**: Test environments and obstacles\n" +
                "- **Props**: Interactive objects\n" +
                "- **Enemies**: Enemy models for combat testing\n");
            
            // VFX README
            CreateReadme("Assets/VFX/README.md",
                "# Visual Effects\n\n" +
                "- **Particles**: Thruster flames, weapon effects, impacts\n" +
                "- **Shaders**: Custom shaders for energy effects, holograms\n" +
                "- **PostProcessing**: Screen effects for HUD, damage states\n");
            
            AssetDatabase.Refresh();
            
            EditorUtility.DisplayDialog("README Files Created",
                "Created README files in key directories.",
                "OK");
        }
        
        private void CreateReadme(string path, string content)
        {
            string directory = Path.GetDirectoryName(path);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }
            
            File.WriteAllText(path, content);
        }
    }
}