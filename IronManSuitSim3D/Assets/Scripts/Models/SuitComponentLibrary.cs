using UnityEngine;
using System.Collections.Generic;

namespace IronManSim.Models
{
    /// <summary>
    /// Library of prefab suit components and detailed parts
    /// </summary>
    public class SuitComponentLibrary : MonoBehaviour
    {
        [System.Serializable]
        public class SuitComponent
        {
            public string componentName;
            public ComponentType type;
            public GameObject prefab;
            public Material[] materials;
            public Vector3 attachmentOffset;
            public Vector3 attachmentRotation;
            public bool isWeapon;
            public bool hasLight;
            public bool hasParticles;
        }
        
        public enum ComponentType
        {
            // Armor pieces
            HelmetVariant,
            ChestPlate,
            BackPlate,
            ShoulderPad,
            ArmGuard,
            Gauntlet,
            ThighArmor,
            ShinGuard,
            Boot,
            
            // Weapons
            Repulsor,
            UniBeam,
            MissilePod,
            LaserArray,
            ShoulderGun,
            
            // Utility
            Thruster,
            Stabilizer,
            Sensor,
            Antenna,
            Light,
            
            // Details
            Panel,
            Vent,
            Joint,
            Servo,
            Cable
        }
        
        [Header("Component Library")]
        [SerializeField] private List<SuitComponent> components = new List<SuitComponent>();
        [SerializeField] private Dictionary<ComponentType, List<SuitComponent>> componentsByType;
        
        [Header("Procedural Generation")]
        [SerializeField] private bool generateProceduralComponents = true;
        [SerializeField] private int detailLevel = 3;
        
        private MeshBuilder meshBuilder;
        
        void Awake()
        {
            meshBuilder = new MeshBuilder();
            InitializeLibrary();
        }
        
        #region Library Initialization
        
        private void InitializeLibrary()
        {
            componentsByType = new Dictionary<ComponentType, List<SuitComponent>>();
            
            // Organize components by type
            foreach (var component in components)
            {
                if (!componentsByType.ContainsKey(component.type))
                {
                    componentsByType[component.type] = new List<SuitComponent>();
                }
                componentsByType[component.type].Add(component);
            }
            
            // Generate procedural components if needed
            if (generateProceduralComponents)
            {
                GenerateProceduralComponents();
            }
        }
        
        private void GenerateProceduralComponents()
        {
            // Generate various detail components
            GenerateRepulsorVariants();
            GenerateThrusterVariants();
            GeneratePanelVariants();
            GenerateVentVariants();
            GenerateWeaponComponents();
            GenerateDetailComponents();
        }
        
        #endregion
        
        #region Component Generation
        
        private void GenerateRepulsorVariants()
        {
            // Standard Repulsor
            CreateRepulsorComponent("Standard Repulsor", 0.03f, 16, false);
            
            // Heavy Repulsor
            CreateRepulsorComponent("Heavy Repulsor", 0.04f, 24, true);
            
            // Micro Repulsor
            CreateRepulsorComponent("Micro Repulsor", 0.015f, 12, false);
        }
        
        private void CreateRepulsorComponent(string name, float radius, int segments, bool enhanced)
        {
            GameObject repulsor = new GameObject(name);
            repulsor.transform.SetParent(transform);
            
            // Create mesh
            MeshFilter filter = repulsor.AddComponent<MeshFilter>();
            MeshRenderer renderer = repulsor.AddComponent<MeshRenderer>();
            
            filter.mesh = GenerateRepulsorMesh(radius, segments, enhanced);
            
            // Create component
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.Repulsor,
                prefab = repulsor,
                isWeapon = true,
                hasLight = true,
                hasParticles = true
            };
            
            // Add light
            Light repulsorLight = repulsor.AddComponent<Light>();
            repulsorLight.type = LightType.Spot;
            repulsorLight.color = new Color(0.2f, 0.8f, 1f);
            repulsorLight.intensity = enhanced ? 5f : 3f;
            repulsorLight.range = enhanced ? 3f : 2f;
            repulsorLight.spotAngle = 30f;
            repulsorLight.enabled = false;
            
            // Add particle system
            AddRepulsorParticles(repulsor, enhanced);
            
            components.Add(component);
            repulsor.SetActive(false);
        }
        
        private Mesh GenerateRepulsorMesh(float radius, int segments, bool enhanced)
        {
            meshBuilder.Clear();
            
            // Center
            meshBuilder.AddVertex(Vector3.zero, Vector3.forward);
            
            // Inner ring
            float innerRadius = radius * 0.6f;
            for (int i = 0; i <= segments; i++)
            {
                float angle = i * 2 * Mathf.PI / segments;
                Vector3 pos = new Vector3(
                    Mathf.Cos(angle) * innerRadius,
                    Mathf.Sin(angle) * innerRadius,
                    0
                );
                meshBuilder.AddVertex(pos, Vector3.forward);
            }
            
            // Outer ring
            for (int i = 0; i <= segments; i++)
            {
                float angle = i * 2 * Mathf.PI / segments;
                Vector3 pos = new Vector3(
                    Mathf.Cos(angle) * radius,
                    Mathf.Sin(angle) * radius,
                    enhanced ? -0.005f : 0
                );
                meshBuilder.AddVertex(pos, Vector3.forward);
            }
            
            // Create triangles
            // Center to inner ring
            for (int i = 0; i < segments; i++)
            {
                meshBuilder.AddTriangle(0, i + 1, i + 2);
            }
            
            // Inner ring to outer ring
            for (int i = 0; i < segments; i++)
            {
                int inner1 = i + 1;
                int inner2 = i + 2;
                int outer1 = i + segments + 2;
                int outer2 = i + segments + 3;
                
                meshBuilder.AddTriangle(inner1, outer1, outer2);
                meshBuilder.AddTriangle(inner1, outer2, inner2);
            }
            
            return meshBuilder.CreateMesh();
        }
        
        private void AddRepulsorParticles(GameObject repulsor, bool enhanced)
        {
            ParticleSystem particles = repulsor.AddComponent<ParticleSystem>();
            var main = particles.main;
            main.startLifetime = enhanced ? 0.5f : 0.3f;
            main.startSpeed = enhanced ? 10f : 5f;
            main.startSize = enhanced ? 0.1f : 0.05f;
            main.startColor = new Color(0.2f, 0.8f, 1f);
            main.maxParticles = enhanced ? 200 : 100;
            
            var emission = particles.emission;
            emission.enabled = false;
            emission.rateOverTime = enhanced ? 100 : 50;
            
            var shape = particles.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 15f;
            shape.radius = 0.01f;
            
            var velocityOverLifetime = particles.velocityOverLifetime;
            velocityOverLifetime.enabled = true;
            velocityOverLifetime.space = ParticleSystemSimulationSpace.Local;
            velocityOverLifetime.z = new ParticleSystem.MinMaxCurve(enhanced ? 20f : 10f);
            
            // Add glow
            var renderer = particles.GetComponent<ParticleSystemRenderer>();
            renderer.material = new Material(Shader.Find("Sprites/Default"));
            renderer.material.SetColor("_Color", new Color(0.2f, 0.8f, 1f));
        }
        
        #endregion
        
        #region Thruster Variants
        
        private void GenerateThrusterVariants()
        {
            // Main Thruster
            CreateThrusterComponent("Main Thruster", 0.06f, 0.1f, 3f);
            
            // Secondary Thruster
            CreateThrusterComponent("Secondary Thruster", 0.04f, 0.08f, 2f);
            
            // Maneuvering Thruster
            CreateThrusterComponent("Maneuvering Thruster", 0.02f, 0.04f, 1f);
        }
        
        private void CreateThrusterComponent(string name, float radius, float length, float thrust)
        {
            GameObject thruster = new GameObject(name);
            thruster.transform.SetParent(transform);
            
            // Create mesh
            MeshFilter filter = thruster.AddComponent<MeshFilter>();
            MeshRenderer renderer = thruster.AddComponent<MeshRenderer>();
            
            filter.mesh = GenerateThrusterMesh(radius, length);
            
            // Create component
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.Thruster,
                prefab = thruster,
                hasLight = true,
                hasParticles = true
            };
            
            // Add thruster particles
            AddThrusterParticles(thruster, thrust);
            
            components.Add(component);
            thruster.SetActive(false);
        }
        
        private Mesh GenerateThrusterMesh(float radius, float length)
        {
            meshBuilder.Clear();
            
            int segments = 16;
            
            // Create nozzle shape
            for (int i = 0; i <= segments; i++)
            {
                float angle = i * 2 * Mathf.PI / segments;
                
                // Top ring (wider)
                Vector3 topPos = new Vector3(
                    Mathf.Cos(angle) * radius,
                    0,
                    Mathf.Sin(angle) * radius
                );
                meshBuilder.AddVertex(topPos, new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle)));
                
                // Bottom ring (narrower)
                Vector3 bottomPos = new Vector3(
                    Mathf.Cos(angle) * radius * 0.7f,
                    -length,
                    Mathf.Sin(angle) * radius * 0.7f
                );
                meshBuilder.AddVertex(bottomPos, new Vector3(Mathf.Cos(angle), -0.5f, Mathf.Sin(angle)).normalized);
            }
            
            // Create triangles
            for (int i = 0; i < segments; i++)
            {
                int top1 = i * 2;
                int bottom1 = i * 2 + 1;
                int top2 = (i + 1) * 2;
                int bottom2 = (i + 1) * 2 + 1;
                
                meshBuilder.AddTriangle(top1, bottom1, bottom2);
                meshBuilder.AddTriangle(top1, bottom2, top2);
            }
            
            return meshBuilder.CreateMesh();
        }
        
        private void AddThrusterParticles(GameObject thruster, float thrust)
        {
            ParticleSystem particles = thruster.AddComponent<ParticleSystem>();
            var main = particles.main;
            main.startLifetime = 1f;
            main.startSpeed = thrust * 5f;
            main.startSize = 0.2f;
            main.startColor = new Gradient
            {
                colorKeys = new GradientColorKey[]
                {
                    new GradientColorKey(Color.white, 0f),
                    new GradientColorKey(new Color(1f, 0.8f, 0.2f), 0.3f),
                    new GradientColorKey(new Color(1f, 0.3f, 0f), 0.7f),
                    new GradientColorKey(new Color(0.5f, 0f, 0f), 1f)
                }
            };
            
            var emission = particles.emission;
            emission.enabled = false;
            emission.rateOverTime = thrust * 30f;
            
            var shape = particles.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 20f;
            shape.radius = 0.01f;
            
            // Add light
            Light thrusterLight = thruster.AddComponent<Light>();
            thrusterLight.type = LightType.Point;
            thrusterLight.color = new Color(1f, 0.8f, 0.2f);
            thrusterLight.intensity = thrust;
            thrusterLight.range = thrust;
            thrusterLight.enabled = false;
        }
        
        #endregion
        
        #region Panel and Vent Variants
        
        private void GeneratePanelVariants()
        {
            // Armor Panel
            CreatePanelComponent("Armor Panel", 0.1f, 0.15f, 0.01f, true);
            
            // Access Panel
            CreatePanelComponent("Access Panel", 0.05f, 0.05f, 0.005f, false);
            
            // Decorative Panel
            CreatePanelComponent("Decorative Panel", 0.08f, 0.12f, 0.008f, true);
        }
        
        private void GenerateVentVariants()
        {
            // Heat Vent
            CreateVentComponent("Heat Vent", 0.04f, 0.08f, 6);
            
            // Cooling Vent
            CreateVentComponent("Cooling Vent", 0.03f, 0.06f, 4);
            
            // Exhaust Vent
            CreateVentComponent("Exhaust Vent", 0.05f, 0.1f, 8);
        }
        
        private void CreatePanelComponent(string name, float width, float height, float depth, bool beveled)
        {
            GameObject panel = new GameObject(name);
            panel.transform.SetParent(transform);
            
            MeshFilter filter = panel.AddComponent<MeshFilter>();
            MeshRenderer renderer = panel.AddComponent<MeshRenderer>();
            
            filter.mesh = GeneratePanelMesh(width, height, depth, beveled);
            
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.Panel,
                prefab = panel
            };
            
            components.Add(component);
            panel.SetActive(false);
        }
        
        private Mesh GeneratePanelMesh(float width, float height, float depth, bool beveled)
        {
            meshBuilder.Clear();
            
            float bevel = beveled ? 0.9f : 1f;
            
            // Front face
            meshBuilder.AddVertex(new Vector3(-width/2 * bevel, height/2 * bevel, depth), Vector3.forward);
            meshBuilder.AddVertex(new Vector3(width/2 * bevel, height/2 * bevel, depth), Vector3.forward);
            meshBuilder.AddVertex(new Vector3(width/2 * bevel, -height/2 * bevel, depth), Vector3.forward);
            meshBuilder.AddVertex(new Vector3(-width/2 * bevel, -height/2 * bevel, depth), Vector3.forward);
            
            // Back face
            meshBuilder.AddVertex(new Vector3(-width/2, height/2, 0), Vector3.back);
            meshBuilder.AddVertex(new Vector3(width/2, height/2, 0), Vector3.back);
            meshBuilder.AddVertex(new Vector3(width/2, -height/2, 0), Vector3.back);
            meshBuilder.AddVertex(new Vector3(-width/2, -height/2, 0), Vector3.back);
            
            // Front face
            meshBuilder.AddTriangle(0, 1, 2);
            meshBuilder.AddTriangle(0, 2, 3);
            
            // Back face
            meshBuilder.AddTriangle(4, 6, 5);
            meshBuilder.AddTriangle(4, 7, 6);
            
            // Sides
            if (beveled)
            {
                // Top
                meshBuilder.AddTriangle(4, 5, 1);
                meshBuilder.AddTriangle(4, 1, 0);
                // Right
                meshBuilder.AddTriangle(5, 6, 2);
                meshBuilder.AddTriangle(5, 2, 1);
                // Bottom
                meshBuilder.AddTriangle(6, 7, 3);
                meshBuilder.AddTriangle(6, 3, 2);
                // Left
                meshBuilder.AddTriangle(7, 4, 0);
                meshBuilder.AddTriangle(7, 0, 3);
            }
            
            return meshBuilder.CreateMesh();
        }
        
        private void CreateVentComponent(string name, float width, float height, int slats)
        {
            GameObject vent = new GameObject(name);
            vent.transform.SetParent(transform);
            
            MeshFilter filter = vent.AddComponent<MeshFilter>();
            MeshRenderer renderer = vent.AddComponent<MeshRenderer>();
            
            filter.mesh = GenerateVentMesh(width, height, slats);
            
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.Vent,
                prefab = vent
            };
            
            components.Add(component);
            vent.SetActive(false);
        }
        
        private Mesh GenerateVentMesh(float width, float height, int slats)
        {
            meshBuilder.Clear();
            
            float slatHeight = height / (slats + 1);
            float slatDepth = 0.01f;
            
            for (int i = 0; i < slats; i++)
            {
                float y = -height/2 + slatHeight * (i + 1);
                
                // Each slat
                int baseIndex = i * 4;
                
                meshBuilder.AddVertex(new Vector3(-width/2, y + slatHeight * 0.3f, 0), Vector3.forward);
                meshBuilder.AddVertex(new Vector3(width/2, y + slatHeight * 0.3f, 0), Vector3.forward);
                meshBuilder.AddVertex(new Vector3(width/2, y - slatHeight * 0.3f, slatDepth), Vector3.forward);
                meshBuilder.AddVertex(new Vector3(-width/2, y - slatHeight * 0.3f, slatDepth), Vector3.forward);
                
                meshBuilder.AddTriangle(baseIndex, baseIndex + 1, baseIndex + 2);
                meshBuilder.AddTriangle(baseIndex, baseIndex + 2, baseIndex + 3);
            }
            
            return meshBuilder.CreateMesh();
        }
        
        #endregion
        
        #region Weapon Components
        
        private void GenerateWeaponComponents()
        {
            // Missile Pod
            CreateMissilePodComponent("Shoulder Missile Pod", 4, 0.02f);
            
            // Laser Array
            CreateLaserArrayComponent("Wrist Laser", 3, 0.01f);
            
            // Mini Gun
            CreateMiniGunComponent("Shoulder Minigun", 6, 0.15f);
        }
        
        private void CreateMissilePodComponent(string name, int tubes, float tubeRadius)
        {
            GameObject pod = new GameObject(name);
            pod.transform.SetParent(transform);
            
            // Create housing
            GameObject housing = GameObject.CreatePrimitive(PrimitiveType.Cube);
            housing.transform.SetParent(pod.transform);
            housing.transform.localScale = new Vector3(0.1f, 0.05f, 0.15f);
            
            // Create missile tubes
            for (int i = 0; i < tubes; i++)
            {
                GameObject tube = new GameObject($"Tube_{i}");
                tube.transform.SetParent(pod.transform);
                
                float x = (i % 2) * 0.04f - 0.02f;
                float y = (i / 2) * 0.03f - 0.015f;
                tube.transform.localPosition = new Vector3(x, y, 0.08f);
                
                MeshFilter filter = tube.AddComponent<MeshFilter>();
                filter.mesh = GenerateTubeMesh(tubeRadius, 0.1f);
                tube.AddComponent<MeshRenderer>();
            }
            
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.MissilePod,
                prefab = pod,
                isWeapon = true
            };
            
            components.Add(component);
            pod.SetActive(false);
        }
        
        private void CreateLaserArrayComponent(string name, int barrels, float barrelRadius)
        {
            GameObject laser = new GameObject(name);
            laser.transform.SetParent(transform);
            
            // Create laser barrels
            for (int i = 0; i < barrels; i++)
            {
                GameObject barrel = new GameObject($"Barrel_{i}");
                barrel.transform.SetParent(laser.transform);
                
                float angle = i * 120f; // For 3 barrels
                float x = Mathf.Cos(angle * Mathf.Deg2Rad) * 0.02f;
                float y = Mathf.Sin(angle * Mathf.Deg2Rad) * 0.02f;
                barrel.transform.localPosition = new Vector3(x, y, 0);
                
                MeshFilter filter = barrel.AddComponent<MeshFilter>();
                filter.mesh = GenerateTubeMesh(barrelRadius, 0.05f);
                barrel.AddComponent<MeshRenderer>();
                
                // Add laser light
                Light laserLight = barrel.AddComponent<Light>();
                laserLight.type = LightType.Spot;
                laserLight.color = Color.red;
                laserLight.intensity = 10f;
                laserLight.range = 50f;
                laserLight.spotAngle = 1f;
                laserLight.enabled = false;
            }
            
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.LaserArray,
                prefab = laser,
                isWeapon = true,
                hasLight = true
            };
            
            components.Add(component);
            laser.SetActive(false);
        }
        
        private void CreateMiniGunComponent(string name, int barrels, float length)
        {
            GameObject minigun = new GameObject(name);
            minigun.transform.SetParent(transform);
            
            // Create rotating barrel assembly
            GameObject barrelAssembly = new GameObject("BarrelAssembly");
            barrelAssembly.transform.SetParent(minigun.transform);
            
            for (int i = 0; i < barrels; i++)
            {
                float angle = i * 360f / barrels;
                GameObject barrel = new GameObject($"Barrel_{i}");
                barrel.transform.SetParent(barrelAssembly.transform);
                
                float x = Mathf.Cos(angle * Mathf.Deg2Rad) * 0.03f;
                float y = Mathf.Sin(angle * Mathf.Deg2Rad) * 0.03f;
                barrel.transform.localPosition = new Vector3(x, y, 0);
                
                MeshFilter filter = barrel.AddComponent<MeshFilter>();
                filter.mesh = GenerateTubeMesh(0.008f, length);
                barrel.AddComponent<MeshRenderer>();
            }
            
            // Add rotation script
            barrelAssembly.AddComponent<RotateComponent>().rotationSpeed = new Vector3(0, 0, 1000f);
            
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.ShoulderGun,
                prefab = minigun,
                isWeapon = true
            };
            
            components.Add(component);
            minigun.SetActive(false);
        }
        
        private Mesh GenerateTubeMesh(float radius, float length)
        {
            meshBuilder.Clear();
            
            int segments = 12;
            
            // Generate cylinder
            for (int i = 0; i <= segments; i++)
            {
                float angle = i * 2 * Mathf.PI / segments;
                
                // Front ring
                Vector3 frontPos = new Vector3(
                    Mathf.Cos(angle) * radius,
                    Mathf.Sin(angle) * radius,
                    0
                );
                meshBuilder.AddVertex(frontPos, new Vector3(Mathf.Cos(angle), Mathf.Sin(angle), 0));
                
                // Back ring
                Vector3 backPos = new Vector3(
                    Mathf.Cos(angle) * radius,
                    Mathf.Sin(angle) * radius,
                    length
                );
                meshBuilder.AddVertex(backPos, new Vector3(Mathf.Cos(angle), Mathf.Sin(angle), 0));
            }
            
            // Create triangles
            for (int i = 0; i < segments; i++)
            {
                int front1 = i * 2;
                int back1 = i * 2 + 1;
                int front2 = (i + 1) * 2;
                int back2 = (i + 1) * 2 + 1;
                
                meshBuilder.AddTriangle(front1, back1, back2);
                meshBuilder.AddTriangle(front1, back2, front2);
            }
            
            return meshBuilder.CreateMesh();
        }
        
        #endregion
        
        #region Detail Components
        
        private void GenerateDetailComponents()
        {
            // Servo Joint
            CreateServoComponent("Servo Joint", 0.03f);
            
            // Cable
            CreateCableComponent("Power Cable", 0.005f, 0.2f);
            
            // Sensor
            CreateSensorComponent("Proximity Sensor", 0.02f);
            
            // Antenna
            CreateAntennaComponent("Comm Antenna", 0.003f, 0.1f);
        }
        
        private void CreateServoComponent(string name, float size)
        {
            GameObject servo = new GameObject(name);
            servo.transform.SetParent(transform);
            
            // Main body
            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            body.transform.SetParent(servo.transform);
            body.transform.localScale = new Vector3(size, size * 0.5f, size);
            
            // Rotating disc
            GameObject disc = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            disc.transform.SetParent(servo.transform);
            disc.transform.localPosition = new Vector3(0, size * 0.5f, 0);
            disc.transform.localScale = new Vector3(size * 0.8f, size * 0.1f, size * 0.8f);
            
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.Servo,
                prefab = servo
            };
            
            components.Add(component);
            servo.SetActive(false);
        }
        
        private void CreateCableComponent(string name, float thickness, float length)
        {
            GameObject cable = new GameObject(name);
            cable.transform.SetParent(transform);
            
            LineRenderer line = cable.AddComponent<LineRenderer>();
            line.startWidth = thickness;
            line.endWidth = thickness;
            line.positionCount = 10;
            
            // Create curved cable path
            for (int i = 0; i < 10; i++)
            {
                float t = i / 9f;
                float x = t * length;
                float y = Mathf.Sin(t * Mathf.PI) * 0.02f;
                line.SetPosition(i, new Vector3(x - length/2, y, 0));
            }
            
            line.material = new Material(Shader.Find("Sprites/Default"));
            line.material.color = Color.black;
            
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.Cable,
                prefab = cable
            };
            
            components.Add(component);
            cable.SetActive(false);
        }
        
        private void CreateSensorComponent(string name, float size)
        {
            GameObject sensor = new GameObject(name);
            sensor.transform.SetParent(transform);
            
            // Sensor dome
            GameObject dome = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            dome.transform.SetParent(sensor.transform);
            dome.transform.localScale = Vector3.one * size;
            
            // Make it look like a sensor
            MeshRenderer renderer = dome.GetComponent<MeshRenderer>();
            renderer.material = new Material(Shader.Find("Standard"));
            renderer.material.color = new Color(0.2f, 0.2f, 0.2f, 0.5f);
            renderer.material.SetFloat("_Metallic", 0.9f);
            renderer.material.SetFloat("_Glossiness", 0.9f);
            
            // Add blinking light
            GameObject light = new GameObject("SensorLight");
            light.transform.SetParent(sensor.transform);
            light.transform.localPosition = new Vector3(0, size * 0.5f, 0);
            
            Light sensorLight = light.AddComponent<Light>();
            sensorLight.type = LightType.Point;
            sensorLight.color = Color.green;
            sensorLight.intensity = 0.5f;
            sensorLight.range = 0.1f;
            
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.Sensor,
                prefab = sensor,
                hasLight = true
            };
            
            components.Add(component);
            sensor.SetActive(false);
        }
        
        private void CreateAntennaComponent(string name, float thickness, float height)
        {
            GameObject antenna = new GameObject(name);
            antenna.transform.SetParent(transform);
            
            // Base
            GameObject baseObj = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            baseObj.transform.SetParent(antenna.transform);
            baseObj.transform.localScale = new Vector3(thickness * 3, thickness, thickness * 3);
            
            // Antenna rod
            GameObject rod = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            rod.transform.SetParent(antenna.transform);
            rod.transform.localPosition = new Vector3(0, height/2, 0);
            rod.transform.localScale = new Vector3(thickness, height/2, thickness);
            
            // Tip
            GameObject tip = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            tip.transform.SetParent(antenna.transform);
            tip.transform.localPosition = new Vector3(0, height, 0);
            tip.transform.localScale = Vector3.one * thickness * 2;
            
            SuitComponent component = new SuitComponent
            {
                componentName = name,
                type = ComponentType.Antenna,
                prefab = antenna
            };
            
            components.Add(component);
            antenna.SetActive(false);
        }
        
        #endregion
        
        #region Public API
        
        public GameObject GetComponent(string componentName)
        {
            SuitComponent component = components.Find(c => c.componentName == componentName);
            if (component != null && component.prefab != null)
            {
                GameObject instance = Instantiate(component.prefab);
                instance.SetActive(true);
                return instance;
            }
            return null;
        }
        
        public List<GameObject> GetComponentsByType(ComponentType type)
        {
            List<GameObject> instances = new List<GameObject>();
            
            if (componentsByType.ContainsKey(type))
            {
                foreach (var component in componentsByType[type])
                {
                    if (component.prefab != null)
                    {
                        GameObject instance = Instantiate(component.prefab);
                        instance.SetActive(true);
                        instances.Add(instance);
                    }
                }
            }
            
            return instances;
        }
        
        public void AttachComponent(GameObject component, Transform attachPoint, Vector3 offset, Vector3 rotation)
        {
            component.transform.SetParent(attachPoint);
            component.transform.localPosition = offset;
            component.transform.localRotation = Quaternion.Euler(rotation);
        }
        
        #endregion
    }
}