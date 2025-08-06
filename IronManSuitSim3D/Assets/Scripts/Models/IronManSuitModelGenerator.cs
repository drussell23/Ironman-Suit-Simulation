using UnityEngine;
using System.Collections.Generic;

namespace IronManSim.Models
{
    /// <summary>
    /// Procedurally generates a detailed Iron Man suit model with all components
    /// </summary>
    public class IronManSuitModelGenerator : MonoBehaviour
    {
        [System.Serializable]
        public class SuitConfiguration
        {
            [Header("Model Settings")]
            public float modelScale = 1.8f; // Human scale
            public int meshResolution = 32;
            public bool generateColliders = true;
            public bool generateRigidBodies = true;
            
            [Header("Suit Variant")]
            public SuitModel suitModel = SuitModel.Mark85;
            public Color primaryColor = new Color(0.8f, 0.1f, 0.1f); // Classic red
            public Color secondaryColor = new Color(0.9f, 0.75f, 0.1f); // Gold
            public Color accentColor = new Color(0.2f, 0.8f, 1f); // Arc reactor blue
            
            [Header("Detail Level")]
            public bool generatePanelLines = true;
            public bool generateRivets = true;
            public bool generateServos = true;
            public bool generateInternalComponents = false;
        }
        
        public enum SuitModel
        {
            Mark1,      // Cave-built prototype
            Mark3,      // Classic red and gold
            Mark7,      // Avengers suit
            Mark42,     // Modular suit
            Mark44,     // Hulkbuster
            Mark50,     // Nanoparticle suit
            Mark85,     // Endgame suit
            WarMachine, // Heavy armor variant
            Rescue      // Pepper's suit
        }
        
        [Header("Configuration")]
        [SerializeField] private SuitConfiguration config = new SuitConfiguration();
        
        [Header("Materials")]
        [SerializeField] private Material primaryMaterial;
        [SerializeField] private Material secondaryMaterial;
        [SerializeField] private Material emissiveMaterial;
        [SerializeField] private Material glassMaterial;
        
        [Header("Generated Components")]
        [SerializeField] private Transform suitRoot;
        [SerializeField] private Dictionary<string, Transform> bodyParts = new Dictionary<string, Transform>();
        
        private MeshBuilder meshBuilder;
        
        void Start()
        {
            meshBuilder = new MeshBuilder();
            GenerateSuit();
        }
        
        #region Public Methods
        
        [ContextMenu("Generate Suit")]
        public void GenerateSuit()
        {
            ClearExistingSuit();
            CreateSuitHierarchy();
            
            // Generate each body part
            GenerateHelmet();
            GenerateTorso();
            GenerateArms();
            GenerateLegs();
            
            // Add details based on configuration
            if (config.generatePanelLines)
            {
                AddPanelLines();
            }
            
            if (config.generateServos)
            {
                AddServos();
            }
            
            // Apply materials
            ApplyMaterials();
            
            // Setup physics if needed
            if (config.generateColliders)
            {
                SetupColliders();
            }
            
            if (config.generateRigidBodies)
            {
                SetupRigidBodies();
            }
        }
        
        [ContextMenu("Clear Suit")]
        public void ClearExistingSuit()
        {
            if (suitRoot != null)
            {
                DestroyImmediate(suitRoot.gameObject);
            }
            bodyParts.Clear();
        }
        
        #endregion
        
        #region Suit Generation
        
        private void CreateSuitHierarchy()
        {
            // Create root
            GameObject rootObj = new GameObject($"IronManSuit_{config.suitModel}");
            rootObj.transform.SetParent(transform);
            rootObj.transform.localPosition = Vector3.zero;
            rootObj.transform.localScale = Vector3.one * config.modelScale;
            suitRoot = rootObj.transform;
            
            // Create body part hierarchy
            CreateBodyPart("Head", suitRoot);
            CreateBodyPart("Torso", suitRoot);
            CreateBodyPart("LeftArm", suitRoot);
            CreateBodyPart("RightArm", suitRoot);
            CreateBodyPart("LeftLeg", suitRoot);
            CreateBodyPart("RightLeg", suitRoot);
        }
        
        private Transform CreateBodyPart(string name, Transform parent)
        {
            GameObject part = new GameObject(name);
            part.transform.SetParent(parent);
            part.transform.localPosition = Vector3.zero;
            bodyParts[name] = part.transform;
            return part.transform;
        }
        
        #endregion
        
        #region Helmet Generation
        
        private void GenerateHelmet()
        {
            Transform head = bodyParts["Head"];
            
            // Main helmet shape
            GameObject helmet = CreateMeshObject("Helmet", head);
            Mesh helmetMesh = GenerateHelmetMesh();
            helmet.GetComponent<MeshFilter>().mesh = helmetMesh;
            
            // Faceplate
            GameObject faceplate = CreateMeshObject("Faceplate", head);
            Mesh faceplateMesh = GenerateFaceplateMesh();
            faceplate.GetComponent<MeshFilter>().mesh = faceplateMesh;
            faceplate.GetComponent<MeshRenderer>().material = secondaryMaterial;
            
            // Eyes
            GameObject leftEye = CreateEye("LeftEye", head, new Vector3(-0.03f, 0.02f, 0.08f));
            GameObject rightEye = CreateEye("RightEye", head, new Vector3(0.03f, 0.02f, 0.08f));
            
            // Add detail components
            if (config.suitModel == SuitModel.Mark85 || config.suitModel == SuitModel.Mark50)
            {
                AddNanoParticleDetails(helmet);
            }
        }
        
        private Mesh GenerateHelmetMesh()
        {
            meshBuilder.Clear();
            
            // Create helmet using sphere with modifications
            int segments = config.meshResolution;
            float radius = 0.12f;
            
            // Generate modified sphere for helmet shape
            for (int lat = 0; lat <= segments; lat++)
            {
                float theta = lat * Mathf.PI / segments;
                float sinTheta = Mathf.Sin(theta);
                float cosTheta = Mathf.Cos(theta);
                
                for (int lon = 0; lon <= segments; lon++)
                {
                    float phi = lon * 2 * Mathf.PI / segments;
                    float sinPhi = Mathf.Sin(phi);
                    float cosPhi = Mathf.Cos(phi);
                    
                    // Modify sphere shape for helmet
                    float x = radius * sinTheta * cosPhi;
                    float y = radius * cosTheta;
                    float z = radius * sinTheta * sinPhi;
                    
                    // Flatten bottom
                    if (y < -0.05f)
                    {
                        y = -0.05f;
                    }
                    
                    // Elongate front slightly
                    if (z > 0)
                    {
                        z *= 1.1f;
                    }
                    
                    Vector3 position = new Vector3(x, y + 0.15f, z);
                    Vector3 normal = position.normalized;
                    
                    meshBuilder.AddVertex(position, normal);
                }
            }
            
            // Generate triangles
            for (int lat = 0; lat < segments; lat++)
            {
                for (int lon = 0; lon < segments; lon++)
                {
                    int current = lat * (segments + 1) + lon;
                    int next = current + segments + 1;
                    
                    meshBuilder.AddTriangle(current, next, current + 1);
                    meshBuilder.AddTriangle(current + 1, next, next + 1);
                }
            }
            
            return meshBuilder.CreateMesh();
        }
        
        private Mesh GenerateFaceplateMesh()
        {
            meshBuilder.Clear();
            
            // Create faceplate shape
            float width = 0.08f;
            float height = 0.1f;
            float depth = 0.02f;
            
            // Front face vertices
            Vector3[] vertices = new Vector3[]
            {
                // Front face - angular shape
                new Vector3(-width * 0.7f, height, depth),      // Top left
                new Vector3(width * 0.7f, height, depth),       // Top right
                new Vector3(width, height * 0.3f, depth),       // Mid right
                new Vector3(width * 0.8f, -height * 0.5f, depth), // Bottom right
                new Vector3(0, -height, depth),                 // Bottom center
                new Vector3(-width * 0.8f, -height * 0.5f, depth), // Bottom left
                new Vector3(-width, height * 0.3f, depth),      // Mid left
            };
            
            // Add vertices with proper normals
            foreach (var vertex in vertices)
            {
                meshBuilder.AddVertex(vertex + new Vector3(0, 0.02f, 0.08f), Vector3.forward);
            }
            
            // Create triangles for front face
            meshBuilder.AddTriangle(0, 1, 2);
            meshBuilder.AddTriangle(0, 2, 6);
            meshBuilder.AddTriangle(6, 2, 3);
            meshBuilder.AddTriangle(6, 3, 5);
            meshBuilder.AddTriangle(5, 3, 4);
            
            return meshBuilder.CreateMesh();
        }
        
        private GameObject CreateEye(string name, Transform parent, Vector3 position)
        {
            GameObject eye = CreateMeshObject(name, parent);
            eye.transform.localPosition = position;
            
            // Create eye mesh (elongated hexagon)
            meshBuilder.Clear();
            
            float width = 0.025f;
            float height = 0.015f;
            
            Vector3[] eyeVerts = new Vector3[]
            {
                new Vector3(-width, 0, 0),
                new Vector3(-width * 0.5f, height, 0),
                new Vector3(width * 0.5f, height, 0),
                new Vector3(width, 0, 0),
                new Vector3(width * 0.5f, -height, 0),
                new Vector3(-width * 0.5f, -height, 0),
            };
            
            foreach (var vert in eyeVerts)
            {
                meshBuilder.AddVertex(vert, Vector3.forward);
            }
            
            // Create triangles
            for (int i = 1; i < 5; i++)
            {
                meshBuilder.AddTriangle(0, i, i + 1);
            }
            meshBuilder.AddTriangle(0, 5, 1);
            
            Mesh eyeMesh = meshBuilder.CreateMesh();
            eye.GetComponent<MeshFilter>().mesh = eyeMesh;
            eye.GetComponent<MeshRenderer>().material = emissiveMaterial;
            
            // Add glow effect
            Light eyeLight = eye.AddComponent<Light>();
            eyeLight.type = LightType.Point;
            eyeLight.color = config.accentColor;
            eyeLight.intensity = 2f;
            eyeLight.range = 0.2f;
            
            return eye;
        }
        
        #endregion
        
        #region Torso Generation
        
        private void GenerateTorso()
        {
            Transform torso = bodyParts["Torso"];
            
            // Chest piece
            GameObject chest = CreateMeshObject("ChestPlate", torso);
            Mesh chestMesh = GenerateChestMesh();
            chest.GetComponent<MeshFilter>().mesh = chestMesh;
            
            // Arc Reactor
            GameObject arcReactor = CreateArcReactor(torso);
            
            // Abdomen
            GameObject abdomen = CreateMeshObject("Abdomen", torso);
            Mesh abdomenMesh = GenerateAbdomenMesh();
            abdomen.GetComponent<MeshFilter>().mesh = abdomenMesh;
            
            // Back piece
            GameObject back = CreateMeshObject("BackPlate", torso);
            Mesh backMesh = GenerateBackMesh();
            back.GetComponent<MeshFilter>().mesh = backMesh;
            
            // Add shoulder mounts
            CreateShoulderMount("LeftShoulder", torso, new Vector3(-0.15f, 0.1f, 0));
            CreateShoulderMount("RightShoulder", torso, new Vector3(0.15f, 0.1f, 0));
        }
        
        private Mesh GenerateChestMesh()
        {
            meshBuilder.Clear();
            
            // Create chest armor shape
            float width = 0.25f;
            float height = 0.3f;
            float depth = 0.15f;
            
            // Define chest plate vertices
            int segments = 16;
            
            // Front surface
            for (int y = 0; y <= segments; y++)
            {
                float yPos = -height / 2 + (height * y / segments);
                float widthAtY = width * (1f - Mathf.Abs(yPos) / height * 0.3f); // Taper
                
                for (int x = 0; x <= segments; x++)
                {
                    float xPos = -widthAtY + (widthAtY * 2 * x / segments);
                    float zPos = depth * (1f - (xPos * xPos) / (widthAtY * widthAtY) * 0.3f); // Curved front
                    
                    Vector3 position = new Vector3(xPos, yPos, zPos);
                    Vector3 normal = new Vector3(xPos * 0.3f, 0, 1).normalized;
                    
                    meshBuilder.AddVertex(position, normal);
                }
            }
            
            // Generate triangles
            for (int y = 0; y < segments; y++)
            {
                for (int x = 0; x < segments; x++)
                {
                    int current = y * (segments + 1) + x;
                    int next = current + segments + 1;
                    
                    meshBuilder.AddTriangle(current, next, current + 1);
                    meshBuilder.AddTriangle(current + 1, next, next + 1);
                }
            }
            
            return meshBuilder.CreateMesh();
        }
        
        private GameObject CreateArcReactor(Transform parent)
        {
            GameObject reactor = CreateMeshObject("ArcReactor", parent);
            reactor.transform.localPosition = new Vector3(0, 0, 0.16f);
            
            // Create reactor mesh (circular with rings)
            meshBuilder.Clear();
            
            float radius = 0.04f;
            int segments = 32;
            
            // Center
            meshBuilder.AddVertex(Vector3.zero, Vector3.forward);
            
            // Create rings
            for (int ring = 0; ring < 3; ring++)
            {
                float ringRadius = radius * (ring + 1) / 3f;
                
                for (int i = 0; i <= segments; i++)
                {
                    float angle = i * 2 * Mathf.PI / segments;
                    Vector3 position = new Vector3(
                        Mathf.Cos(angle) * ringRadius,
                        Mathf.Sin(angle) * ringRadius,
                        0
                    );
                    
                    meshBuilder.AddVertex(position, Vector3.forward);
                }
            }
            
            // Create triangles for center
            for (int i = 0; i < segments; i++)
            {
                meshBuilder.AddTriangle(0, i + 1, i + 2);
            }
            
            Mesh reactorMesh = meshBuilder.CreateMesh();
            reactor.GetComponent<MeshFilter>().mesh = reactorMesh;
            reactor.GetComponent<MeshRenderer>().material = emissiveMaterial;
            
            // Add glow
            Light reactorLight = reactor.AddComponent<Light>();
            reactorLight.type = LightType.Point;
            reactorLight.color = config.accentColor;
            reactorLight.intensity = 5f;
            reactorLight.range = 0.5f;
            
            // Add rotating component
            reactor.AddComponent<RotateComponent>().rotationSpeed = new Vector3(0, 0, 30f);
            
            return reactor;
        }
        
        #endregion
        
        #region Arm Generation
        
        private void GenerateArms()
        {
            GenerateArm("LeftArm", new Vector3(-0.2f, 0, 0), false);
            GenerateArm("RightArm", new Vector3(0.2f, 0, 0), true);
        }
        
        private void GenerateArm(string armName, Vector3 offset, bool isRight)
        {
            Transform arm = bodyParts[armName];
            arm.localPosition = offset;
            
            // Upper arm
            GameObject upperArm = CreateMeshObject("UpperArm", arm);
            Mesh upperArmMesh = GenerateLimbSegment(0.06f, 0.05f, 0.25f);
            upperArm.GetComponent<MeshFilter>().mesh = upperArmMesh;
            
            // Elbow joint
            GameObject elbow = CreateMeshObject("Elbow", arm);
            elbow.transform.localPosition = new Vector3(0, -0.25f, 0);
            Mesh elbowMesh = GenerateJointMesh(0.05f);
            elbow.GetComponent<MeshFilter>().mesh = elbowMesh;
            elbow.GetComponent<MeshRenderer>().material = secondaryMaterial;
            
            // Forearm
            GameObject forearm = CreateMeshObject("Forearm", arm);
            forearm.transform.localPosition = new Vector3(0, -0.25f, 0);
            Mesh forearmMesh = GenerateLimbSegment(0.05f, 0.04f, 0.22f);
            forearm.GetComponent<MeshFilter>().mesh = forearmMesh;
            
            // Hand/Gauntlet
            GameObject hand = CreateMeshObject("Hand", arm);
            hand.transform.localPosition = new Vector3(0, -0.47f, 0);
            Mesh handMesh = GenerateHandMesh(isRight);
            hand.GetComponent<MeshFilter>().mesh = handMesh;
            
            // Repulsor
            GameObject repulsor = CreateRepulsor(hand.transform, new Vector3(0, -0.05f, 0.02f));
        }
        
        private Mesh GenerateHandMesh(bool isRight)
        {
            meshBuilder.Clear();
            
            // Simplified gauntlet shape
            float width = 0.08f;
            float height = 0.12f;
            float depth = 0.05f;
            
            // Palm section
            Vector3[] palmVerts = new Vector3[]
            {
                // Top
                new Vector3(-width/2, 0, -depth/2),
                new Vector3(width/2, 0, -depth/2),
                new Vector3(width/2, 0, depth/2),
                new Vector3(-width/2, 0, depth/2),
                // Bottom
                new Vector3(-width/2, -height, -depth/2),
                new Vector3(width/2, -height, -depth/2),
                new Vector3(width/2, -height, depth/2),
                new Vector3(-width/2, -height, depth/2),
            };
            
            foreach (var vert in palmVerts)
            {
                meshBuilder.AddVertex(vert, Vector3.forward);
            }
            
            // Create box triangles
            // Front
            meshBuilder.AddTriangle(0, 4, 5);
            meshBuilder.AddTriangle(0, 5, 1);
            // Back
            meshBuilder.AddTriangle(2, 6, 7);
            meshBuilder.AddTriangle(2, 7, 3);
            // Left
            meshBuilder.AddTriangle(3, 7, 4);
            meshBuilder.AddTriangle(3, 4, 0);
            // Right
            meshBuilder.AddTriangle(1, 5, 6);
            meshBuilder.AddTriangle(1, 6, 2);
            // Top
            meshBuilder.AddTriangle(0, 1, 2);
            meshBuilder.AddTriangle(0, 2, 3);
            // Bottom
            meshBuilder.AddTriangle(4, 6, 5);
            meshBuilder.AddTriangle(4, 7, 6);
            
            return meshBuilder.CreateMesh();
        }
        
        private GameObject CreateRepulsor(Transform parent, Vector3 position)
        {
            GameObject repulsor = CreateMeshObject("Repulsor", parent);
            repulsor.transform.localPosition = position;
            
            // Create repulsor mesh (circular)
            meshBuilder.Clear();
            
            float radius = 0.025f;
            int segments = 16;
            
            // Center
            meshBuilder.AddVertex(Vector3.zero, Vector3.forward);
            
            // Outer ring
            for (int i = 0; i <= segments; i++)
            {
                float angle = i * 2 * Mathf.PI / segments;
                Vector3 pos = new Vector3(
                    Mathf.Cos(angle) * radius,
                    Mathf.Sin(angle) * radius,
                    0
                );
                meshBuilder.AddVertex(pos, Vector3.forward);
            }
            
            // Create triangles
            for (int i = 0; i < segments; i++)
            {
                meshBuilder.AddTriangle(0, i + 1, i + 2);
            }
            
            Mesh repulsorMesh = meshBuilder.CreateMesh();
            repulsor.GetComponent<MeshFilter>().mesh = repulsorMesh;
            repulsor.GetComponent<MeshRenderer>().material = emissiveMaterial;
            
            // Add light
            Light repulsorLight = repulsor.AddComponent<Light>();
            repulsorLight.type = LightType.Spot;
            repulsorLight.color = config.accentColor;
            repulsorLight.intensity = 3f;
            repulsorLight.range = 2f;
            repulsorLight.spotAngle = 30f;
            
            return repulsor;
        }
        
        #endregion
        
        #region Leg Generation
        
        private void GenerateLegs()
        {
            GenerateLeg("LeftLeg", new Vector3(-0.1f, -0.5f, 0));
            GenerateLeg("RightLeg", new Vector3(0.1f, -0.5f, 0));
        }
        
        private void GenerateLeg(string legName, Vector3 offset)
        {
            Transform leg = bodyParts[legName];
            leg.localPosition = offset;
            
            // Thigh
            GameObject thigh = CreateMeshObject("Thigh", leg);
            Mesh thighMesh = GenerateLimbSegment(0.08f, 0.07f, 0.35f);
            thigh.GetComponent<MeshFilter>().mesh = thighMesh;
            
            // Knee joint
            GameObject knee = CreateMeshObject("Knee", leg);
            knee.transform.localPosition = new Vector3(0, -0.35f, 0);
            Mesh kneeMesh = GenerateJointMesh(0.07f);
            knee.GetComponent<MeshFilter>().mesh = kneeMesh;
            knee.GetComponent<MeshRenderer>().material = secondaryMaterial;
            
            // Shin
            GameObject shin = CreateMeshObject("Shin", leg);
            shin.transform.localPosition = new Vector3(0, -0.35f, 0);
            Mesh shinMesh = GenerateLimbSegment(0.07f, 0.06f, 0.35f);
            shin.GetComponent<MeshFilter>().mesh = shinMesh;
            
            // Boot
            GameObject boot = CreateMeshObject("Boot", leg);
            boot.transform.localPosition = new Vector3(0, -0.7f, 0);
            Mesh bootMesh = GenerateBootMesh();
            boot.GetComponent<MeshFilter>().mesh = bootMesh;
            
            // Boot thruster
            GameObject thruster = CreateMeshObject("BootThruster", boot.transform);
            thruster.transform.localPosition = new Vector3(0, -0.08f, 0);
            Mesh thrusterMesh = GenerateThrusterMesh();
            thruster.GetComponent<MeshFilter>().mesh = thrusterMesh;
            thruster.GetComponent<MeshRenderer>().material = secondaryMaterial;
        }
        
        private Mesh GenerateBootMesh()
        {
            meshBuilder.Clear();
            
            // Boot shape
            float width = 0.1f;
            float height = 0.1f;
            float length = 0.2f;
            
            // Create boot box with extended front
            Vector3[] bootVerts = new Vector3[]
            {
                // Top
                new Vector3(-width/2, 0, -length/3),
                new Vector3(width/2, 0, -length/3),
                new Vector3(width/2, 0, length*2/3),
                new Vector3(-width/2, 0, length*2/3),
                // Bottom
                new Vector3(-width/2, -height, -length/3),
                new Vector3(width/2, -height, -length/3),
                new Vector3(width/2, -height, length*2/3),
                new Vector3(-width/2, -height, length*2/3),
            };
            
            foreach (var vert in bootVerts)
            {
                meshBuilder.AddVertex(vert, Vector3.forward);
            }
            
            // Create box triangles
            CreateBoxTriangles();
            
            return meshBuilder.CreateMesh();
        }
        
        #endregion
        
        #region Helper Methods
        
        private Mesh GenerateLimbSegment(float topRadius, float bottomRadius, float height)
        {
            meshBuilder.Clear();
            
            int segments = 16;
            
            // Generate cylinder with taper
            for (int y = 0; y <= segments; y++)
            {
                float t = (float)y / segments;
                float radius = Mathf.Lerp(topRadius, bottomRadius, t);
                float yPos = -height * t;
                
                for (int x = 0; x <= segments; x++)
                {
                    float angle = x * 2 * Mathf.PI / segments;
                    
                    Vector3 position = new Vector3(
                        Mathf.Cos(angle) * radius,
                        yPos,
                        Mathf.Sin(angle) * radius
                    );
                    
                    Vector3 normal = new Vector3(
                        Mathf.Cos(angle),
                        0,
                        Mathf.Sin(angle)
                    ).normalized;
                    
                    meshBuilder.AddVertex(position, normal);
                }
            }
            
            // Generate triangles
            for (int y = 0; y < segments; y++)
            {
                for (int x = 0; x < segments; x++)
                {
                    int current = y * (segments + 1) + x;
                    int next = current + segments + 1;
                    
                    meshBuilder.AddTriangle(current, next, current + 1);
                    meshBuilder.AddTriangle(current + 1, next, next + 1);
                }
            }
            
            return meshBuilder.CreateMesh();
        }
        
        private Mesh GenerateJointMesh(float radius)
        {
            meshBuilder.Clear();
            
            // Create sphere for joint
            int segments = 16;
            
            for (int lat = 0; lat <= segments; lat++)
            {
                float theta = lat * Mathf.PI / segments;
                float sinTheta = Mathf.Sin(theta);
                float cosTheta = Mathf.Cos(theta);
                
                for (int lon = 0; lon <= segments; lon++)
                {
                    float phi = lon * 2 * Mathf.PI / segments;
                    float sinPhi = Mathf.Sin(phi);
                    float cosPhi = Mathf.Cos(phi);
                    
                    Vector3 position = new Vector3(
                        radius * sinTheta * cosPhi,
                        radius * cosTheta,
                        radius * sinTheta * sinPhi
                    );
                    
                    meshBuilder.AddVertex(position, position.normalized);
                }
            }
            
            // Generate triangles
            for (int lat = 0; lat < segments; lat++)
            {
                for (int lon = 0; lon < segments; lon++)
                {
                    int current = lat * (segments + 1) + lon;
                    int next = current + segments + 1;
                    
                    meshBuilder.AddTriangle(current, next, current + 1);
                    meshBuilder.AddTriangle(current + 1, next, next + 1);
                }
            }
            
            return meshBuilder.CreateMesh();
        }
        
        private Mesh GenerateThrusterMesh()
        {
            meshBuilder.Clear();
            
            // Create cone for thruster
            float radius = 0.04f;
            float height = 0.05f;
            int segments = 12;
            
            // Tip
            meshBuilder.AddVertex(new Vector3(0, -height, 0), Vector3.down);
            
            // Base ring
            for (int i = 0; i <= segments; i++)
            {
                float angle = i * 2 * Mathf.PI / segments;
                Vector3 position = new Vector3(
                    Mathf.Cos(angle) * radius,
                    0,
                    Mathf.Sin(angle) * radius
                );
                meshBuilder.AddVertex(position, Vector3.up);
            }
            
            // Create triangles
            for (int i = 0; i < segments; i++)
            {
                meshBuilder.AddTriangle(0, i + 2, i + 1);
            }
            
            return meshBuilder.CreateMesh();
        }
        
        private void CreateBoxTriangles()
        {
            // Front
            meshBuilder.AddTriangle(0, 4, 5);
            meshBuilder.AddTriangle(0, 5, 1);
            // Back
            meshBuilder.AddTriangle(2, 6, 7);
            meshBuilder.AddTriangle(2, 7, 3);
            // Left
            meshBuilder.AddTriangle(3, 7, 4);
            meshBuilder.AddTriangle(3, 4, 0);
            // Right
            meshBuilder.AddTriangle(1, 5, 6);
            meshBuilder.AddTriangle(1, 6, 2);
            // Top
            meshBuilder.AddTriangle(0, 1, 2);
            meshBuilder.AddTriangle(0, 2, 3);
            // Bottom
            meshBuilder.AddTriangle(4, 6, 5);
            meshBuilder.AddTriangle(4, 7, 6);
        }
        
        private GameObject CreateMeshObject(string name, Transform parent)
        {
            GameObject obj = new GameObject(name);
            obj.transform.SetParent(parent);
            obj.transform.localPosition = Vector3.zero;
            obj.AddComponent<MeshFilter>();
            MeshRenderer renderer = obj.AddComponent<MeshRenderer>();
            renderer.material = primaryMaterial;
            return obj;
        }
        
        private void CreateShoulderMount(string name, Transform parent, Vector3 position)
        {
            GameObject shoulder = CreateMeshObject(name, parent);
            shoulder.transform.localPosition = position;
            
            Mesh shoulderMesh = GenerateJointMesh(0.08f);
            shoulder.GetComponent<MeshFilter>().mesh = shoulderMesh;
            shoulder.GetComponent<MeshRenderer>().material = secondaryMaterial;
        }
        
        private Mesh GenerateBackMesh()
        {
            // Similar to chest but flatter
            return GenerateChestMesh(); // Simplified for now
        }
        
        private Mesh GenerateAbdomenMesh()
        {
            // Segmented armor plates
            return GenerateLimbSegment(0.2f, 0.15f, 0.2f);
        }
        
        #endregion
        
        #region Details
        
        private void AddPanelLines()
        {
            // Add decorative panel lines to the suit
            // This would add edge loops and detail geometry
        }
        
        private void AddServos()
        {
            // Add mechanical servo details at joints
        }
        
        private void AddNanoParticleDetails(GameObject component)
        {
            // Add nano-particle effect details for Mark 50/85
        }
        
        #endregion
        
        #region Materials and Physics
        
        private void ApplyMaterials()
        {
            // Create materials if not assigned
            if (primaryMaterial == null)
            {
                primaryMaterial = new Material(Shader.Find("Standard"));
                primaryMaterial.color = config.primaryColor;
                primaryMaterial.metallic = 0.8f;
                primaryMaterial.smoothness = 0.9f;
            }
            
            if (secondaryMaterial == null)
            {
                secondaryMaterial = new Material(Shader.Find("Standard"));
                secondaryMaterial.color = config.secondaryColor;
                secondaryMaterial.metallic = 0.9f;
                secondaryMaterial.smoothness = 0.95f;
            }
            
            if (emissiveMaterial == null)
            {
                emissiveMaterial = new Material(Shader.Find("Standard"));
                emissiveMaterial.color = config.accentColor;
                emissiveMaterial.SetColor("_EmissionColor", config.accentColor * 5f);
                emissiveMaterial.EnableKeyword("_EMISSION");
            }
            
            if (glassMaterial == null)
            {
                glassMaterial = new Material(Shader.Find("Standard"));
                glassMaterial.color = new Color(0.2f, 0.2f, 0.2f, 0.3f);
                glassMaterial.SetFloat("_Mode", 3); // Transparent
                glassMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                glassMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                glassMaterial.SetInt("_ZWrite", 0);
                glassMaterial.DisableKeyword("_ALPHATEST_ON");
                glassMaterial.EnableKeyword("_ALPHABLEND_ON");
                glassMaterial.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                glassMaterial.renderQueue = 3000;
            }
        }
        
        private void SetupColliders()
        {
            // Add colliders to each body part
            foreach (var part in bodyParts.Values)
            {
                AddCollidersToHierarchy(part);
            }
        }
        
        private void AddCollidersToHierarchy(Transform root)
        {
            MeshFilter[] meshFilters = root.GetComponentsInChildren<MeshFilter>();
            
            foreach (var filter in meshFilters)
            {
                if (filter.mesh != null)
                {
                    MeshCollider collider = filter.gameObject.AddComponent<MeshCollider>();
                    collider.convex = true;
                }
            }
        }
        
        private void SetupRigidBodies()
        {
            // Add rigidbody to suit root
            Rigidbody rb = suitRoot.gameObject.AddComponent<Rigidbody>();
            rb.mass = 100f; // Heavy armor
            rb.linearDamping = 0.5f;
            rb.angularDamping = 2f;
            rb.useGravity = true;
            
            // Add configurable joints for articulation
            SetupArticulation();
        }
        
        private void SetupArticulation()
        {
            // Setup joints for movement
            // This would add ConfigurableJoints between body parts
        }
        
        #endregion
    }
    
    /// <summary>
    /// Helper class for building meshes
    /// </summary>
    public class MeshBuilder
    {
        private List<Vector3> vertices = new List<Vector3>();
        private List<Vector3> normals = new List<Vector3>();
        private List<Vector2> uvs = new List<Vector2>();
        private List<int> triangles = new List<int>();
        
        public void Clear()
        {
            vertices.Clear();
            normals.Clear();
            uvs.Clear();
            triangles.Clear();
        }
        
        public void AddVertex(Vector3 position, Vector3 normal)
        {
            vertices.Add(position);
            normals.Add(normal);
            uvs.Add(new Vector2(position.x, position.y)); // Simple UV mapping
        }
        
        public void AddTriangle(int a, int b, int c)
        {
            triangles.Add(a);
            triangles.Add(b);
            triangles.Add(c);
        }
        
        public Mesh CreateMesh()
        {
            Mesh mesh = new Mesh();
            mesh.name = "GeneratedMesh";
            mesh.vertices = vertices.ToArray();
            mesh.normals = normals.ToArray();
            mesh.uv = uvs.ToArray();
            mesh.triangles = triangles.ToArray();
            
            if (normals.Count == 0)
            {
                mesh.RecalculateNormals();
            }
            
            mesh.RecalculateBounds();
            mesh.RecalculateTangents();
            
            return mesh;
        }
    }
    
    /// <summary>
    /// Simple rotation component
    /// </summary>
    public class RotateComponent : MonoBehaviour
    {
        public Vector3 rotationSpeed = Vector3.zero;
        
        void Update()
        {
            transform.Rotate(rotationSpeed * Time.deltaTime);
        }
    }
}