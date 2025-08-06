using UnityEngine;
using System.Collections.Generic;

namespace IronManSim.Models
{
    /// <summary>
    /// Sets up a rigged character controller for the Iron Man suit with IK support
    /// </summary>
    public class IronManSuitRig : MonoBehaviour
    {
        [System.Serializable]
        public class BoneInfo
        {
            public string name;
            public Transform bone;
            public Vector3 localPosition;
            public Vector3 localRotation;
            public float length;
        }
        
        [System.Serializable]
        public class IKChain
        {
            public Transform root;
            public Transform mid;
            public Transform end;
            public Transform target;
            public Transform poleTarget;
            [Range(0, 1)] public float weight = 1f;
            public bool usePoleTarget = true;
        }
        
        [Header("Skeleton Setup")]
        [SerializeField] private Transform rootBone;
        [SerializeField] private List<BoneInfo> bones = new List<BoneInfo>();
        
        [Header("IK Chains")]
        [SerializeField] private IKChain leftArmIK;
        [SerializeField] private IKChain rightArmIK;
        [SerializeField] private IKChain leftLegIK;
        [SerializeField] private IKChain rightLegIK;
        
        [Header("Animation Settings")]
        [SerializeField] private bool useIK = true;
        [SerializeField] private bool useFKBlending = true;
        [SerializeField] private float ikSmoothTime = 0.1f;
        
        [Header("Suit Parts Mapping")]
        [SerializeField] private Dictionary<string, Transform> suitParts = new Dictionary<string, Transform>();
        
        [Header("Humanoid Avatar")]
        [SerializeField] private Avatar humanoidAvatar;
        [SerializeField] private bool createHumanoidAvatar = true;
        
        private Animator animator;
        private Dictionary<string, Quaternion> defaultRotations = new Dictionary<string, Quaternion>();
        
        void Start()
        {
            SetupRig();
        }
        
        void LateUpdate()
        {
            if (useIK)
            {
                SolveIK();
            }
        }
        
        #region Rig Setup
        
        [ContextMenu("Setup Rig")]
        public void SetupRig()
        {
            CreateSkeleton();
            MapSuitPartsToSkeleton();
            SetupAnimator();
            CacheDefaultPoses();
        }
        
        private void CreateSkeleton()
        {
            if (rootBone == null)
            {
                GameObject root = new GameObject("Root");
                root.transform.SetParent(transform);
                root.transform.localPosition = Vector3.zero;
                rootBone = root.transform;
            }
            
            // Create humanoid skeleton hierarchy
            CreateSpine();
            CreateArms();
            CreateLegs();
            CreateHead();
        }
        
        private void CreateSpine()
        {
            Transform hips = CreateBone("Hips", rootBone, Vector3.zero);
            Transform spine = CreateBone("Spine", hips, new Vector3(0, 0.1f, 0));
            Transform spine1 = CreateBone("Spine1", spine, new Vector3(0, 0.15f, 0));
            Transform spine2 = CreateBone("Spine2", spine1, new Vector3(0, 0.15f, 0));
            Transform chest = CreateBone("Chest", spine2, new Vector3(0, 0.15f, 0));
            
            // Neck and head connection point
            Transform neck = CreateBone("Neck", chest, new Vector3(0, 0.15f, 0));
        }
        
        private void CreateArms()
        {
            Transform chest = transform.Find("Root/Hips/Spine/Spine1/Spine2/Chest");
            
            // Left arm
            Transform leftShoulder = CreateBone("LeftShoulder", chest, new Vector3(-0.08f, 0.1f, 0));
            Transform leftUpperArm = CreateBone("LeftUpperArm", leftShoulder, new Vector3(-0.12f, 0, 0));
            Transform leftForearm = CreateBone("LeftForearm", leftUpperArm, new Vector3(-0.25f, 0, 0));
            Transform leftHand = CreateBone("LeftHand", leftForearm, new Vector3(-0.22f, 0, 0));
            
            // Left fingers
            CreateFingers(leftHand, "Left");
            
            // Right arm
            Transform rightShoulder = CreateBone("RightShoulder", chest, new Vector3(0.08f, 0.1f, 0));
            Transform rightUpperArm = CreateBone("RightUpperArm", rightShoulder, new Vector3(0.12f, 0, 0));
            Transform rightForearm = CreateBone("RightForearm", rightUpperArm, new Vector3(0.25f, 0, 0));
            Transform rightHand = CreateBone("RightHand", rightForearm, new Vector3(0.22f, 0, 0));
            
            // Right fingers
            CreateFingers(rightHand, "Right");
            
            // Setup IK chains
            leftArmIK = new IKChain
            {
                root = leftUpperArm,
                mid = leftForearm,
                end = leftHand
            };
            
            rightArmIK = new IKChain
            {
                root = rightUpperArm,
                mid = rightForearm,
                end = rightHand
            };
        }
        
        private void CreateLegs()
        {
            Transform hips = transform.Find("Root/Hips");
            
            // Left leg
            Transform leftUpLeg = CreateBone("LeftUpLeg", hips, new Vector3(-0.1f, -0.05f, 0));
            Transform leftLeg = CreateBone("LeftLeg", leftUpLeg, new Vector3(0, -0.35f, 0));
            Transform leftFoot = CreateBone("LeftFoot", leftLeg, new Vector3(0, -0.35f, 0));
            Transform leftToes = CreateBone("LeftToes", leftFoot, new Vector3(0, -0.05f, 0.1f));
            
            // Right leg
            Transform rightUpLeg = CreateBone("RightUpLeg", hips, new Vector3(0.1f, -0.05f, 0));
            Transform rightLeg = CreateBone("RightLeg", rightUpLeg, new Vector3(0, -0.35f, 0));
            Transform rightFoot = CreateBone("RightFoot", rightLeg, new Vector3(0, -0.35f, 0));
            Transform rightToes = CreateBone("RightToes", rightFoot, new Vector3(0, -0.05f, 0.1f));
            
            // Setup IK chains
            leftLegIK = new IKChain
            {
                root = leftUpLeg,
                mid = leftLeg,
                end = leftFoot
            };
            
            rightLegIK = new IKChain
            {
                root = rightUpLeg,
                mid = rightLeg,
                end = rightFoot
            };
        }
        
        private void CreateHead()
        {
            Transform neck = transform.Find("Root/Hips/Spine/Spine1/Spine2/Chest/Neck");
            Transform head = CreateBone("Head", neck, new Vector3(0, 0.1f, 0));
            
            // Eye bones for look-at
            Transform leftEye = CreateBone("LeftEye", head, new Vector3(-0.03f, 0.05f, 0.08f));
            Transform rightEye = CreateBone("RightEye", head, new Vector3(0.03f, 0.05f, 0.08f));
        }
        
        private void CreateFingers(Transform hand, string side)
        {
            // Simplified finger bones
            string[] fingers = { "Thumb", "Index", "Middle", "Ring", "Pinky" };
            float[] xOffsets = { -0.02f, -0.03f, -0.01f, 0.01f, 0.03f };
            
            for (int i = 0; i < fingers.Length; i++)
            {
                float xOffset = side == "Left" ? xOffsets[i] : -xOffsets[i];
                Vector3 fingerBase = new Vector3(xOffset, -0.02f, 0.03f);
                
                Transform proximal = CreateBone($"{side}{fingers[i]}Proximal", hand, fingerBase);
                Transform intermediate = CreateBone($"{side}{fingers[i]}Intermediate", proximal, new Vector3(0, -0.02f, 0));
                Transform distal = CreateBone($"{side}{fingers[i]}Distal", intermediate, new Vector3(0, -0.015f, 0));
            }
        }
        
        private Transform CreateBone(string name, Transform parent, Vector3 localPosition)
        {
            GameObject bone = new GameObject(name);
            bone.transform.SetParent(parent);
            bone.transform.localPosition = localPosition;
            bone.transform.localRotation = Quaternion.identity;
            
            // Add visual gizmo
            BoneRenderer boneRenderer = bone.AddComponent<BoneRenderer>();
            
            // Store bone info
            BoneInfo info = new BoneInfo
            {
                name = name,
                bone = bone.transform,
                localPosition = localPosition,
                localRotation = Vector3.zero,
                length = localPosition.magnitude
            };
            bones.Add(info);
            
            return bone.transform;
        }
        
        #endregion
        
        #region Suit Mapping
        
        private void MapSuitPartsToSkeleton()
        {
            // Find suit parts and map them to bones
            IronManSuitModelGenerator generator = GetComponent<IronManSuitModelGenerator>();
            if (generator == null) return;
            
            // Map major parts
            MapSuitPart("Head", "Helmet");
            MapSuitPart("Chest", "ChestPlate");
            MapSuitPart("LeftUpperArm", "LeftArm/UpperArm");
            MapSuitPart("LeftForearm", "LeftArm/Forearm");
            MapSuitPart("LeftHand", "LeftArm/Hand");
            MapSuitPart("RightUpperArm", "RightArm/UpperArm");
            MapSuitPart("RightForearm", "RightArm/Forearm");
            MapSuitPart("RightHand", "RightArm/Hand");
            MapSuitPart("LeftUpLeg", "LeftLeg/Thigh");
            MapSuitPart("LeftLeg", "LeftLeg/Shin");
            MapSuitPart("LeftFoot", "LeftLeg/Boot");
            MapSuitPart("RightUpLeg", "RightLeg/Thigh");
            MapSuitPart("RightLeg", "RightLeg/Shin");
            MapSuitPart("RightFoot", "RightLeg/Boot");
        }
        
        private void MapSuitPart(string boneName, string suitPartPath)
        {
            Transform bone = transform.Find($"Root/{GetBonePath(boneName)}");
            Transform suitPart = transform.Find($"IronManSuit_Mark85/{suitPartPath}");
            
            if (bone != null && suitPart != null)
            {
                // Parent suit part to bone
                suitPart.SetParent(bone);
                suitPart.localPosition = Vector3.zero;
                suitPart.localRotation = Quaternion.identity;
                
                suitParts[boneName] = suitPart;
            }
        }
        
        private string GetBonePath(string boneName)
        {
            // Convert bone name to path
            Dictionary<string, string> paths = new Dictionary<string, string>
            {
                { "Head", "Hips/Spine/Spine1/Spine2/Chest/Neck/Head" },
                { "Chest", "Hips/Spine/Spine1/Spine2/Chest" },
                { "LeftUpperArm", "Hips/Spine/Spine1/Spine2/Chest/LeftShoulder/LeftUpperArm" },
                { "LeftForearm", "Hips/Spine/Spine1/Spine2/Chest/LeftShoulder/LeftUpperArm/LeftForearm" },
                { "LeftHand", "Hips/Spine/Spine1/Spine2/Chest/LeftShoulder/LeftUpperArm/LeftForearm/LeftHand" },
                { "RightUpperArm", "Hips/Spine/Spine1/Spine2/Chest/RightShoulder/RightUpperArm" },
                { "RightForearm", "Hips/Spine/Spine1/Spine2/Chest/RightShoulder/RightUpperArm/RightForearm" },
                { "RightHand", "Hips/Spine/Spine1/Spine2/Chest/RightShoulder/RightUpperArm/RightForearm/RightHand" },
                { "LeftUpLeg", "Hips/LeftUpLeg" },
                { "LeftLeg", "Hips/LeftUpLeg/LeftLeg" },
                { "LeftFoot", "Hips/LeftUpLeg/LeftLeg/LeftFoot" },
                { "RightUpLeg", "Hips/RightUpLeg" },
                { "RightLeg", "Hips/RightUpLeg/RightLeg" },
                { "RightFoot", "Hips/RightUpLeg/RightLeg/RightFoot" }
            };
            
            return paths.ContainsKey(boneName) ? paths[boneName] : boneName;
        }
        
        #endregion
        
        #region Animation Setup
        
        private void SetupAnimator()
        {
            animator = GetComponent<Animator>();
            if (animator == null)
            {
                animator = gameObject.AddComponent<Animator>();
            }
            
            if (createHumanoidAvatar && humanoidAvatar == null)
            {
                CreateHumanoidMapping();
            }
            
            if (humanoidAvatar != null)
            {
                animator.avatar = humanoidAvatar;
            }
        }
        
        private void CreateHumanoidMapping()
        {
            // This would create a humanoid avatar mapping
            // In practice, you'd use Unity's AvatarBuilder API
            // For now, we'll use a generic setup
        }
        
        private void CacheDefaultPoses()
        {
            foreach (var bone in bones)
            {
                if (bone.bone != null)
                {
                    defaultRotations[bone.name] = bone.bone.localRotation;
                }
            }
        }
        
        #endregion
        
        #region IK Solving
        
        private void SolveIK()
        {
            if (leftArmIK.target != null && leftArmIK.weight > 0)
            {
                SolveIKChain(leftArmIK);
            }
            
            if (rightArmIK.target != null && rightArmIK.weight > 0)
            {
                SolveIKChain(rightArmIK);
            }
            
            if (leftLegIK.target != null && leftLegIK.weight > 0)
            {
                SolveIKChain(leftLegIK);
            }
            
            if (rightLegIK.target != null && rightLegIK.weight > 0)
            {
                SolveIKChain(rightLegIK);
            }
        }
        
        private void SolveIKChain(IKChain chain)
        {
            if (chain.root == null || chain.mid == null || chain.end == null || chain.target == null)
                return;
            
            // Two-bone IK solver
            Vector3 rootPos = chain.root.position;
            Vector3 targetPos = chain.target.position;
            
            // Calculate bone lengths
            float upperLength = Vector3.Distance(chain.root.position, chain.mid.position);
            float lowerLength = Vector3.Distance(chain.mid.position, chain.end.position);
            float targetDistance = Vector3.Distance(rootPos, targetPos);
            
            // Clamp target distance
            targetDistance = Mathf.Clamp(targetDistance, 0.01f, upperLength + lowerLength - 0.01f);
            
            // Calculate elbow/knee angle using law of cosines
            float a = upperLength;
            float b = lowerLength;
            float c = targetDistance;
            
            float angleA = Mathf.Acos((b * b + c * c - a * a) / (2 * b * c)) * Mathf.Rad2Deg;
            float angleB = Mathf.Acos((a * a + c * c - b * b) / (2 * a * c)) * Mathf.Rad2Deg;
            
            // Apply rotations
            Vector3 targetDir = (targetPos - rootPos).normalized;
            chain.root.rotation = Quaternion.LookRotation(targetDir) * Quaternion.Euler(angleB, 0, 0);
            
            if (chain.usePoleTarget && chain.poleTarget != null)
            {
                // Use pole target for elbow/knee direction
                Vector3 poleDir = (chain.poleTarget.position - chain.mid.position).normalized;
                Vector3 limbDir = (chain.end.position - chain.root.position).normalized;
                Vector3 bendNormal = Vector3.Cross(limbDir, poleDir);
                
                chain.root.rotation = Quaternion.LookRotation(targetDir, bendNormal) * Quaternion.Euler(angleB, 0, 0);
            }
            
            // Point mid bone at end
            chain.mid.rotation = Quaternion.LookRotation(targetPos - chain.mid.position);
            
            // Match end rotation to target
            chain.end.rotation = chain.target.rotation;
            
            // Apply weight
            if (chain.weight < 1f)
            {
                // Blend with FK
                chain.root.localRotation = Quaternion.Slerp(defaultRotations[chain.root.name], chain.root.localRotation, chain.weight);
                chain.mid.localRotation = Quaternion.Slerp(defaultRotations[chain.mid.name], chain.mid.localRotation, chain.weight);
                chain.end.localRotation = Quaternion.Slerp(defaultRotations[chain.end.name], chain.end.localRotation, chain.weight);
            }
        }
        
        #endregion
        
        #region Public Methods
        
        public void SetIKTarget(string chainName, Transform target)
        {
            switch (chainName)
            {
                case "LeftArm":
                    leftArmIK.target = target;
                    break;
                case "RightArm":
                    rightArmIK.target = target;
                    break;
                case "LeftLeg":
                    leftLegIK.target = target;
                    break;
                case "RightLeg":
                    rightLegIK.target = target;
                    break;
            }
        }
        
        public void SetIKWeight(string chainName, float weight)
        {
            weight = Mathf.Clamp01(weight);
            
            switch (chainName)
            {
                case "LeftArm":
                    leftArmIK.weight = weight;
                    break;
                case "RightArm":
                    rightArmIK.weight = weight;
                    break;
                case "LeftLeg":
                    leftLegIK.weight = weight;
                    break;
                case "RightLeg":
                    rightLegIK.weight = weight;
                    break;
            }
        }
        
        public Transform GetBone(string boneName)
        {
            BoneInfo bone = bones.Find(b => b.name == boneName);
            return bone?.bone;
        }
        
        #endregion
        
        #region Gizmos
        
        void OnDrawGizmos()
        {
            if (!Application.isPlaying) return;
            
            // Draw skeleton
            Gizmos.color = Color.yellow;
            foreach (var bone in bones)
            {
                if (bone.bone != null && bone.bone.parent != null)
                {
                    Gizmos.DrawLine(bone.bone.position, bone.bone.parent.position);
                    Gizmos.DrawWireSphere(bone.bone.position, 0.01f);
                }
            }
            
            // Draw IK targets
            DrawIKChainGizmo(leftArmIK, Color.red);
            DrawIKChainGizmo(rightArmIK, Color.blue);
            DrawIKChainGizmo(leftLegIK, Color.green);
            DrawIKChainGizmo(rightLegIK, Color.cyan);
        }
        
        private void DrawIKChainGizmo(IKChain chain, Color color)
        {
            if (chain.target == null) return;
            
            Gizmos.color = color;
            Gizmos.DrawWireSphere(chain.target.position, 0.05f);
            
            if (chain.poleTarget != null)
            {
                Gizmos.color = color * 0.5f;
                Gizmos.DrawWireCube(chain.poleTarget.position, Vector3.one * 0.03f);
                Gizmos.DrawLine(chain.mid.position, chain.poleTarget.position);
            }
        }
        
        #endregion
    }
    
    /// <summary>
    /// Visual bone renderer for debugging
    /// </summary>
    public class BoneRenderer : MonoBehaviour
    {
        void OnDrawGizmos()
        {
            if (transform.parent != null)
            {
                Gizmos.color = Color.white;
                Gizmos.DrawLine(transform.position, transform.parent.position);
                
                // Draw joint
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(transform.position, 0.01f);
            }
        }
    }
}