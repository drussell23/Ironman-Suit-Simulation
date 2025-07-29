using UnityEngine;
using System;
using System.Collections.Generic;

namespace IronManSim.Aerodynamics.Turbulence
{
    /// <summary>
    /// Implements a simplified k-epsilon turbulence model for Unity.
    /// Calculates turbulent kinetic energy and dissipation rate to model turbulent flow effects.
    /// </summary>
    public class KEpsilonModel : MonoBehaviour
    {
        [Header("Model Configuration")]
        [SerializeField] private bool enableTurbulenceModel = true;
        [SerializeField] private int gridResolution = 10; // Grid cells per dimension
        [SerializeField] private float domainSize = 20f; // meters
        [SerializeField] private float timeStep = 0.02f; // seconds
        
        [Header("Model Constants")]
        [SerializeField] private float cMu = 0.09f; // Turbulent viscosity constant
        [SerializeField] private float c1Epsilon = 1.44f; // Epsilon equation constant
        [SerializeField] private float c2Epsilon = 1.92f; // Epsilon equation constant
        [SerializeField] private float sigmaK = 1.0f; // Turbulent Prandtl number for k
        [SerializeField] private float sigmaEpsilon = 1.3f; // Turbulent Prandtl number for epsilon
        
        [Header("Initial Conditions")]
        [SerializeField] private float initialTurbulenceIntensity = 0.05f; // 5%
        [SerializeField] private float initialLengthScale = 1f; // meters
        [SerializeField] private float molecularViscosity = 1.5e-5f; // m²/s (air at 20°C)
        
        [Header("Boundary Conditions")]
        [SerializeField] private float wallRoughness = 0.001f; // meters
        [SerializeField] private float inletTurbulenceIntensity = 0.1f;
        [SerializeField] private float inletVelocity = 10f; // m/s
        
        [Header("Current State")]
        [SerializeField] private float averageTurbulentKineticEnergy;
        [SerializeField] private float averageDissipationRate;
        [SerializeField] private float averageTurbulentViscosity;
        [SerializeField] private float maxTurbulenceIntensity;
        
        // Grid data structures
        private float[,,] kField; // Turbulent kinetic energy
        private float[,,] epsilonField; // Dissipation rate
        private float[,,] nutField; // Turbulent viscosity
        private Vector3[,,] velocityField; // Velocity field
        private float[,,] productionField; // Production of turbulence
        
        // Components
        private Rigidbody rb;
        private WindInteraction wind;
        private AerodynamicForces aeroForces;
        
        // Grid properties
        private float cellSize;
        private Vector3 gridOrigin;
        
        // Properties
        public float AverageTurbulentKineticEnergy => averageTurbulentKineticEnergy;
        public float AverageTurbulentViscosity => averageTurbulentViscosity;
        public float MaxTurbulenceIntensity => maxTurbulenceIntensity;
        
        void Start()
        {
            rb = GetComponent<Rigidbody>();
            wind = GetComponent<WindInteraction>();
            aeroForces = GetComponent<AerodynamicForces>();
            
            InitializeGrid();
            InitializeFields();
        }
        
        void InitializeGrid()
        {
            cellSize = domainSize / gridResolution;
            gridOrigin = transform.position - Vector3.one * (domainSize / 2f);
            
            // Allocate fields
            kField = new float[gridResolution, gridResolution, gridResolution];
            epsilonField = new float[gridResolution, gridResolution, gridResolution];
            nutField = new float[gridResolution, gridResolution, gridResolution];
            velocityField = new Vector3[gridResolution, gridResolution, gridResolution];
            productionField = new float[gridResolution, gridResolution, gridResolution];
        }
        
        void InitializeFields()
        {
            float u0 = inletVelocity;
            float k0 = 1.5f * Mathf.Pow(u0 * initialTurbulenceIntensity, 2);
            float epsilon0 = Mathf.Pow(cMu, 0.75f) * Mathf.Pow(k0, 1.5f) / initialLengthScale;
            
            for (int i = 0; i < gridResolution; i++)
            {
                for (int j = 0; j < gridResolution; j++)
                {
                    for (int k = 0; k < gridResolution; k++)
                    {
                        kField[i, j, k] = k0;
                        epsilonField[i, j, k] = epsilon0;
                        nutField[i, j, k] = cMu * k0 * k0 / epsilon0;
                        velocityField[i, j, k] = Vector3.forward * u0;
                    }
                }
            }
        }
        
        void FixedUpdate()
        {
            if (!enableTurbulenceModel) return;
            
            UpdateVelocityField();
            CalculateProduction();
            SolveKEquation();
            SolveEpsilonEquation();
            UpdateTurbulentViscosity();
            CalculateStatistics();
            ApplyTurbulenceEffects();
        }
        
        void UpdateVelocityField()
        {
            // Update velocity field based on object motion and wind
            Vector3 objectVelocity = rb.linearVelocity;
            Vector3 windVelocity = wind != null ? wind.CurrentWind : Vector3.zero;
            
            for (int i = 0; i < gridResolution; i++)
            {
                for (int j = 0; j < gridResolution; j++)
                {
                    for (int k = 0; k < gridResolution; k++)
                    {
                        Vector3 cellPos = GetCellWorldPosition(i, j, k);
                        Vector3 relativePos = cellPos - transform.position;
                        
                        // Base flow
                        Vector3 baseFlow = windVelocity - objectVelocity;
                        
                        // Add rotation effects
                        Vector3 rotationalVelocity = Vector3.Cross(rb.angularVelocity, relativePos);
                        
                        // Wake effects (simplified)
                        float wakeFactor = 1f;
                        if (Vector3.Dot(relativePos, objectVelocity) < 0) // Behind object
                        {
                            wakeFactor = 0.5f + 0.5f * Mathf.Exp(-relativePos.magnitude / 5f);
                        }
                        
                        velocityField[i, j, k] = (baseFlow + rotationalVelocity) * wakeFactor;
                    }
                }
            }
        }
        
        void CalculateProduction()
        {
            // Calculate production of turbulent kinetic energy
            for (int i = 1; i < gridResolution - 1; i++)
            {
                for (int j = 1; j < gridResolution - 1; j++)
                {
                    for (int k = 1; k < gridResolution - 1; k++)
                    {
                        // Calculate velocity gradients
                        Vector3 dudx = (velocityField[i + 1, j, k] - velocityField[i - 1, j, k]) / (2f * cellSize);
                        Vector3 dudy = (velocityField[i, j + 1, k] - velocityField[i, j - 1, k]) / (2f * cellSize);
                        Vector3 dudz = (velocityField[i, j, k + 1] - velocityField[i, j, k - 1]) / (2f * cellSize);
                        
                        // Mean strain rate tensor magnitude
                        float s11 = dudx.x;
                        float s22 = dudy.y;
                        float s33 = dudz.z;
                        float s12 = 0.5f * (dudx.y + dudy.x);
                        float s13 = 0.5f * (dudx.z + dudz.x);
                        float s23 = 0.5f * (dudy.z + dudz.y);
                        
                        float strainRateMagnitude = Mathf.Sqrt(
                            2f * (s11 * s11 + s22 * s22 + s33 * s33) +
                            4f * (s12 * s12 + s13 * s13 + s23 * s23)
                        );
                        
                        // Production: P = nut * S²
                        productionField[i, j, k] = nutField[i, j, k] * strainRateMagnitude * strainRateMagnitude;
                    }
                }
            }
        }
        
        void SolveKEquation()
        {
            // Solve turbulent kinetic energy equation
            float[,,] newK = new float[gridResolution, gridResolution, gridResolution];
            
            for (int i = 1; i < gridResolution - 1; i++)
            {
                for (int j = 1; j < gridResolution - 1; j++)
                {
                    for (int k = 1; k < gridResolution - 1; k++)
                    {
                        float k_current = kField[i, j, k];
                        float epsilon_current = epsilonField[i, j, k];
                        float nut_current = nutField[i, j, k];
                        float production = productionField[i, j, k];
                        
                        // Diffusion term (simplified)
                        float diffusion = CalculateDiffusion(kField, i, j, k, molecularViscosity + nut_current / sigmaK);
                        
                        // k equation: dk/dt = P - epsilon + diffusion
                        float dkdt = production - epsilon_current + diffusion;
                        
                        // Update k
                        newK[i, j, k] = Mathf.Max(0, k_current + dkdt * timeStep);
                    }
                }
            }
            
            // Apply boundary conditions
            ApplyBoundaryConditions(newK, true);
            
            // Update field
            kField = newK;
        }
        
        void SolveEpsilonEquation()
        {
            // Solve dissipation rate equation
            float[,,] newEpsilon = new float[gridResolution, gridResolution, gridResolution];
            
            for (int i = 1; i < gridResolution - 1; i++)
            {
                for (int j = 1; j < gridResolution - 1; j++)
                {
                    for (int k = 1; k < gridResolution - 1; k++)
                    {
                        float k_current = kField[i, j, k];
                        float epsilon_current = epsilonField[i, j, k];
                        float nut_current = nutField[i, j, k];
                        float production = productionField[i, j, k];
                        
                        if (k_current > 1e-10f)
                        {
                            // Diffusion term
                            float diffusion = CalculateDiffusion(epsilonField, i, j, k, molecularViscosity + nut_current / sigmaEpsilon);
                            
                            // Epsilon equation: depsilon/dt = (C1*P - C2*epsilon)*epsilon/k + diffusion
                            float depsilondt = (c1Epsilon * production - c2Epsilon * epsilon_current) * epsilon_current / k_current + diffusion;
                            
                            // Update epsilon
                            newEpsilon[i, j, k] = Mathf.Max(1e-10f, epsilon_current + depsilondt * timeStep);
                        }
                        else
                        {
                            newEpsilon[i, j, k] = 1e-10f;
                        }
                    }
                }
            }
            
            // Apply boundary conditions
            ApplyBoundaryConditions(newEpsilon, false);
            
            // Update field
            epsilonField = newEpsilon;
        }
        
        float CalculateDiffusion(float[,,] field, int i, int j, int k, float diffusivity)
        {
            // Calculate Laplacian using finite differences
            float laplacian = 0;
            
            // X direction
            laplacian += (field[i + 1, j, k] - 2f * field[i, j, k] + field[i - 1, j, k]) / (cellSize * cellSize);
            
            // Y direction
            laplacian += (field[i, j + 1, k] - 2f * field[i, j, k] + field[i, j - 1, k]) / (cellSize * cellSize);
            
            // Z direction
            laplacian += (field[i, j, k + 1] - 2f * field[i, j, k] + field[i, j, k - 1]) / (cellSize * cellSize);
            
            return diffusivity * laplacian;
        }
        
        void UpdateTurbulentViscosity()
        {
            // Update turbulent viscosity field
            for (int i = 0; i < gridResolution; i++)
            {
                for (int j = 0; j < gridResolution; j++)
                {
                    for (int k = 0; k < gridResolution; k++)
                    {
                        float k_current = kField[i, j, k];
                        float epsilon_current = epsilonField[i, j, k];
                        
                        if (epsilon_current > 1e-10f)
                        {
                            nutField[i, j, k] = cMu * k_current * k_current / epsilon_current;
                        }
                        else
                        {
                            nutField[i, j, k] = 0;
                        }
                    }
                }
            }
        }
        
        void ApplyBoundaryConditions(float[,,] field, bool isKField)
        {
            // Wall boundary conditions
            for (int i = 0; i < gridResolution; i++)
            {
                for (int j = 0; j < gridResolution; j++)
                {
                    // Bottom wall
                    if (isKField)
                    {
                        field[i, 0, j] = 0; // No-slip condition for k
                    }
                    else
                    {
                        // Wall function for epsilon
                        float yPlus = wallRoughness * Mathf.Sqrt(kField[i, 1, j]) / molecularViscosity;
                        field[i, 0, j] = Mathf.Pow(cMu, 0.75f) * Mathf.Pow(kField[i, 1, j], 1.5f) / (0.41f * cellSize);
                    }
                }
            }
            
            // Inlet conditions
            float k_inlet = 1.5f * Mathf.Pow(inletVelocity * inletTurbulenceIntensity, 2);
            float epsilon_inlet = Mathf.Pow(cMu, 0.75f) * Mathf.Pow(k_inlet, 1.5f) / initialLengthScale;
            
            for (int j = 0; j < gridResolution; j++)
            {
                for (int k = 0; k < gridResolution; k++)
                {
                    field[0, j, k] = isKField ? k_inlet : epsilon_inlet;
                }
            }
        }
        
        void CalculateStatistics()
        {
            float sumK = 0;
            float sumEpsilon = 0;
            float sumNut = 0;
            float maxK = 0;
            int count = 0;
            
            for (int i = 0; i < gridResolution; i++)
            {
                for (int j = 0; j < gridResolution; j++)
                {
                    for (int k = 0; k < gridResolution; k++)
                    {
                        sumK += kField[i, j, k];
                        sumEpsilon += epsilonField[i, j, k];
                        sumNut += nutField[i, j, k];
                        maxK = Mathf.Max(maxK, kField[i, j, k]);
                        count++;
                    }
                }
            }
            
            averageTurbulentKineticEnergy = sumK / count;
            averageDissipationRate = sumEpsilon / count;
            averageTurbulentViscosity = sumNut / count;
            
            // Calculate max turbulence intensity
            float referenceVelocity = rb.linearVelocity.magnitude + 0.1f;
            maxTurbulenceIntensity = Mathf.Sqrt(2f * maxK / 3f) / referenceVelocity;
        }
        
        void ApplyTurbulenceEffects()
        {
            // Apply turbulence-induced forces and moments to the object
            if (aeroForces == null) return;
            
            // Sample turbulence at object location
            Vector3 objectGridPos = WorldToGridPosition(transform.position);
            int i = Mathf.Clamp(Mathf.RoundToInt(objectGridPos.x), 0, gridResolution - 1);
            int j = Mathf.Clamp(Mathf.RoundToInt(objectGridPos.y), 0, gridResolution - 1);
            int k = Mathf.Clamp(Mathf.RoundToInt(objectGridPos.z), 0, gridResolution - 1);
            
            float localK = kField[i, j, k];
            float turbulenceVelocity = Mathf.Sqrt(2f * localK / 3f);
            
            // Add turbulent fluctuations
            Vector3 turbulentForce = UnityEngine.Random.insideUnitSphere * turbulenceVelocity * aeroForces.DynamicPressure * 0.1f;
            rb.AddForce(turbulentForce);
            
            // Add turbulent moments
            Vector3 turbulentMoment = UnityEngine.Random.insideUnitSphere * turbulenceVelocity * 0.05f;
            rb.AddTorque(turbulentMoment);
        }
        
        Vector3 GetCellWorldPosition(int i, int j, int k)
        {
            return gridOrigin + new Vector3(i, j, k) * cellSize + Vector3.one * (cellSize / 2f);
        }
        
        Vector3 WorldToGridPosition(Vector3 worldPos)
        {
            Vector3 localPos = worldPos - gridOrigin;
            return localPos / cellSize;
        }
        
        // Public methods
        public float GetTurbulenceIntensityAt(Vector3 worldPosition)
        {
            Vector3 gridPos = WorldToGridPosition(worldPosition);
            
            if (gridPos.x >= 0 && gridPos.x < gridResolution &&
                gridPos.y >= 0 && gridPos.y < gridResolution &&
                gridPos.z >= 0 && gridPos.z < gridResolution)
            {
                int i = Mathf.FloorToInt(gridPos.x);
                int j = Mathf.FloorToInt(gridPos.y);
                int k = Mathf.FloorToInt(gridPos.z);
                
                float k_value = kField[i, j, k];
                float referenceVelocity = velocityField[i, j, k].magnitude + 0.1f;
                
                return Mathf.Sqrt(2f * k_value / 3f) / referenceVelocity;
            }
            
            return 0;
        }
        
        public float GetTurbulentViscosityAt(Vector3 worldPosition)
        {
            Vector3 gridPos = WorldToGridPosition(worldPosition);
            
            if (gridPos.x >= 0 && gridPos.x < gridResolution &&
                gridPos.y >= 0 && gridPos.y < gridResolution &&
                gridPos.z >= 0 && gridPos.z < gridResolution)
            {
                int i = Mathf.FloorToInt(gridPos.x);
                int j = Mathf.FloorToInt(gridPos.y);
                int k = Mathf.FloorToInt(gridPos.z);
                
                return nutField[i, j, k];
            }
            
            return 0;
        }
        
        void OnDrawGizmosSelected()
        {
            if (!Application.isPlaying || !enableTurbulenceModel) return;
            
            // Visualize turbulence intensity
            for (int i = 0; i < gridResolution; i += 2)
            {
                for (int j = 0; j < gridResolution; j += 2)
                {
                    for (int k = 0; k < gridResolution; k += 2)
                    {
                        Vector3 pos = GetCellWorldPosition(i, j, k);
                        float intensity = GetTurbulenceIntensityAt(pos);
                        
                        if (intensity > 0.01f)
                        {
                            Gizmos.color = Color.Lerp(Color.blue, Color.red, intensity * 10f);
                            Gizmos.DrawWireCube(pos, Vector3.one * cellSize * 0.8f);
                        }
                    }
                }
            }
        }
    }
}