using UnityEngine;
using System;

namespace IronManSim.Aerodynamics
{
    /// <summary>
    /// Simulates wind effects including gusts, turbulence, and steady wind.
    /// Applies realistic wind forces and moments to the Iron Man suit.
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    public class WindInteraction : MonoBehaviour
    {
        [Header("Wind Configuration")]
        [SerializeField] private bool windEnabled = true;
        [SerializeField] private WindProfile windProfile = WindProfile.Moderate;
        
        [Header("Steady Wind")]
        [SerializeField] private Vector3 baseWindDirection = Vector3.forward;
        [SerializeField] private float baseWindSpeed = 10f; // m/s
        [SerializeField] private AnimationCurve windSpeedByAltitude; // Multiplier vs altitude
        
        [Header("Gusts")]
        [SerializeField] private bool gustsEnabled = true;
        [SerializeField] private float gustFrequency = 0.3f; // Gusts per second
        [SerializeField] private float gustIntensity = 5f; // m/s
        [SerializeField] private float gustDuration = 2f; // seconds
        [SerializeField] private float gustSpatialScale = 50f; // meters
        
        [Header("Turbulence")]
        [SerializeField] private bool turbulenceEnabled = true;
        [SerializeField] private float turbulenceIntensity = 2f;
        [SerializeField] private float turbulenceScale = 10f;
        [SerializeField] private float turbulenceSpeed = 1f;
        
        [Header("Shear Effects")]
        [SerializeField] private bool windShearEnabled = true;
        [SerializeField] private float shearLayerHeight = 100f; // meters
        [SerializeField] private float shearStrength = 0.5f; // Wind change per meter
        
        [Header("Wake Turbulence")]
        [SerializeField] private bool wakeEffectsEnabled = true;
        [SerializeField] private float wakeDecayRate = 0.1f;
        [SerializeField] private float wakeStrength = 10f;
        
        [Header("Current State")]
        [SerializeField] private Vector3 currentWindVector;
        [SerializeField] private Vector3 gustVector;
        [SerializeField] private Vector3 turbulenceVector;
        [SerializeField] private float currentGustTime;
        [SerializeField] private bool inGust;
        
        // Components
        private Rigidbody rb;
        private AtmosphericDensity atmosphere;
        private AerodynamicForces aeroForces;
        
        // Noise generators
        private float turbulenceOffset;
        private Vector3 gustDirection;
        private float nextGustTime;
        
        // Wake vortices (simplified)
        private struct WakeVortex
        {
            public Vector3 position;
            public Vector3 axis;
            public float strength;
            public float age;
        }
        private WakeVortex[] wakeVortices = new WakeVortex[10];
        private int wakeIndex = 0;
        
        public enum WindProfile
        {
            Calm,
            Light,
            Moderate,
            Strong,
            Severe,
            Hurricane
        }
        
        // Properties
        public Vector3 CurrentWind => currentWindVector;
        public float WindSpeed => currentWindVector.magnitude;
        public bool InTurbulence => turbulenceVector.magnitude > turbulenceIntensity * 0.5f;
        
        void Start()
        {
            rb = GetComponent<Rigidbody>();
            atmosphere = GetComponent<AtmosphericDensity>();
            aeroForces = GetComponent<AerodynamicForces>();
            
            InitializeWindCurve();
            ApplyWindProfile();
            
            turbulenceOffset = UnityEngine.Random.Range(0f, 100f);
            nextGustTime = Time.time + UnityEngine.Random.Range(1f, 5f);
        }
        
        void FixedUpdate()
        {
            if (!windEnabled) return;
            
            UpdateWindComponents();
            ApplyWindForces();
            UpdateWakeVortices();
        }
        
        void InitializeWindCurve()
        {
            if (windSpeedByAltitude == null || windSpeedByAltitude.length == 0)
            {
                windSpeedByAltitude = new AnimationCurve();
                windSpeedByAltitude.AddKey(0f, 0.3f);     // Ground effect reduction
                windSpeedByAltitude.AddKey(10f, 0.5f);    // Surface layer
                windSpeedByAltitude.AddKey(100f, 1.0f);   // Boundary layer
                windSpeedByAltitude.AddKey(1000f, 1.5f);  // Free atmosphere
                windSpeedByAltitude.AddKey(10000f, 2.0f); // Jet stream level
            }
        }
        
        void ApplyWindProfile()
        {
            switch (windProfile)
            {
                case WindProfile.Calm:
                    baseWindSpeed = 2f;
                    gustIntensity = 1f;
                    turbulenceIntensity = 0.5f;
                    break;
                case WindProfile.Light:
                    baseWindSpeed = 5f;
                    gustIntensity = 2f;
                    turbulenceIntensity = 1f;
                    break;
                case WindProfile.Moderate:
                    baseWindSpeed = 10f;
                    gustIntensity = 5f;
                    turbulenceIntensity = 2f;
                    break;
                case WindProfile.Strong:
                    baseWindSpeed = 20f;
                    gustIntensity = 10f;
                    turbulenceIntensity = 4f;
                    break;
                case WindProfile.Severe:
                    baseWindSpeed = 30f;
                    gustIntensity = 15f;
                    turbulenceIntensity = 6f;
                    break;
                case WindProfile.Hurricane:
                    baseWindSpeed = 50f;
                    gustIntensity = 25f;
                    turbulenceIntensity = 10f;
                    break;
            }
        }
        
        void UpdateWindComponents()
        {
            float altitude = transform.position.y;
            float altitudeMultiplier = windSpeedByAltitude.Evaluate(altitude);
            
            // Base wind with altitude variation
            Vector3 baseWind = baseWindDirection.normalized * baseWindSpeed * altitudeMultiplier;
            
            // Wind shear
            if (windShearEnabled && altitude < shearLayerHeight)
            {
                float shearFactor = (shearLayerHeight - altitude) / shearLayerHeight;
                Vector3 shearWind = Vector3.Cross(Vector3.up, baseWind) * shearStrength * shearFactor;
                baseWind += shearWind;
            }
            
            // Gusts
            UpdateGusts();
            
            // Turbulence
            UpdateTurbulence();
            
            // Combine all components
            currentWindVector = baseWind + gustVector + turbulenceVector;
        }
        
        void UpdateGusts()
        {
            if (!gustsEnabled)
            {
                gustVector = Vector3.zero;
                return;
            }
            
            // Check if we should start a new gust
            if (!inGust && Time.time > nextGustTime)
            {
                StartGust();
            }
            
            // Update current gust
            if (inGust)
            {
                currentGustTime += Time.fixedDeltaTime;
                
                // Gust envelope (sine wave)
                float gustPhase = currentGustTime / gustDuration;
                if (gustPhase >= 1f)
                {
                    EndGust();
                }
                else
                {
                    float envelope = Mathf.Sin(gustPhase * Mathf.PI);
                    
                    // Spatial variation
                    float spatialNoise = Perlin3D(
                        transform.position / gustSpatialScale + Vector3.one * currentGustTime
                    );
                    
                    gustVector = gustDirection * gustIntensity * envelope * (0.5f + 0.5f * spatialNoise);
                }
            }
        }
        
        void StartGust()
        {
            inGust = true;
            currentGustTime = 0f;
            
            // Random gust direction (mostly horizontal)
            float angle = UnityEngine.Random.Range(0f, 360f) * Mathf.Deg2Rad;
            float verticalComponent = UnityEngine.Random.Range(-0.3f, 0.3f);
            gustDirection = new Vector3(
                Mathf.Cos(angle),
                verticalComponent,
                Mathf.Sin(angle)
            ).normalized;
        }
        
        void EndGust()
        {
            inGust = false;
            gustVector = Vector3.zero;
            nextGustTime = Time.time + UnityEngine.Random.Range(1f / gustFrequency, 3f / gustFrequency);
        }
        
        void UpdateTurbulence()
        {
            if (!turbulenceEnabled)
            {
                turbulenceVector = Vector3.zero;
                return;
            }
            
            // 3D Perlin noise for turbulence
            Vector3 noiseInput = transform.position / turbulenceScale + Vector3.one * (Time.time * turbulenceSpeed + turbulenceOffset);
            
            turbulenceVector = new Vector3(
                Perlin3D(noiseInput) - 0.5f,
                Perlin3D(noiseInput + Vector3.one * 100f) - 0.5f,
                Perlin3D(noiseInput + Vector3.one * 200f) - 0.5f
            ) * turbulenceIntensity * 2f;
            
            // Reduce turbulence near ground
            float groundEffect = Mathf.Clamp01(transform.position.y / 50f);
            turbulenceVector *= groundEffect;
        }
        
        void ApplyWindForces()
        {
            // Get relative wind
            Vector3 relativeWind = currentWindVector - rb.linearVelocity;
            
            // Only apply if there's significant relative wind
            if (relativeWind.magnitude < 0.1f) return;
            
            // Calculate dynamic pressure from wind
            float density = atmosphere != null ? atmosphere.Density : 1.225f;
            float windPressure = 0.5f * density * relativeWind.sqrMagnitude;
            
            // Estimate exposed area (simplified)
            float exposedArea = EstimateExposedArea(relativeWind.normalized);
            
            // Wind force
            Vector3 windForce = relativeWind.normalized * windPressure * exposedArea * 0.5f; // Drag coefficient
            rb.AddForce(windForce);
            
            // Wind moments (causes rotation)
            Vector3 windMoment = CalculateWindMoment(relativeWind, windPressure);
            rb.AddTorque(windMoment);
        }
        
        float EstimateExposedArea(Vector3 windDirection)
        {
            // Simplified area calculation based on suit orientation
            float frontalArea = 2.0f; // m² - suit frontal area
            float sideArea = 1.5f;    // m² - suit side area
            float topArea = 0.8f;     // m² - suit top area
            
            // Dot products to determine exposure
            float frontExposure = Mathf.Abs(Vector3.Dot(windDirection, transform.forward));
            float sideExposure = Mathf.Abs(Vector3.Dot(windDirection, transform.right));
            float topExposure = Mathf.Abs(Vector3.Dot(windDirection, transform.up));
            
            return frontalArea * frontExposure + sideArea * sideExposure + topArea * topExposure;
        }
        
        Vector3 CalculateWindMoment(Vector3 relativeWind, float pressure)
        {
            // Wind can cause rotation, especially if center of pressure is offset from center of mass
            Vector3 centerOfPressureOffset = new Vector3(0, 0.2f, 0); // Slightly above COM
            
            // Torque = r × F
            Vector3 windForce = relativeWind.normalized * pressure * 0.5f;
            Vector3 moment = Vector3.Cross(centerOfPressureOffset, windForce);
            
            // Add some turbulence-induced rotation
            if (turbulenceEnabled)
            {
                moment += turbulenceVector * 0.1f;
            }
            
            return moment;
        }
        
        void UpdateWakeVortices()
        {
            if (!wakeEffectsEnabled) return;
            
            // Age and decay existing vortices
            for (int i = 0; i < wakeVortices.Length; i++)
            {
                if (wakeVortices[i].strength > 0)
                {
                    wakeVortices[i].age += Time.fixedDeltaTime;
                    wakeVortices[i].strength *= (1f - wakeDecayRate * Time.fixedDeltaTime);
                    
                    // Apply vortex effect if close enough
                    Vector3 toVortex = wakeVortices[i].position - transform.position;
                    float distance = toVortex.magnitude;
                    
                    if (distance < 20f && distance > 1f)
                    {
                        // Biot-Savart law simplified
                        Vector3 velocity = Vector3.Cross(wakeVortices[i].axis, toVortex) / 
                            (distance * distance) * wakeVortices[i].strength;
                        currentWindVector += velocity;
                    }
                }
            }
            
            // Create new wake vortex when moving fast
            if (rb.linearVelocity.magnitude > 50f && UnityEngine.Random.Range(0f, 1f) < 0.1f)
            {
                CreateWakeVortex();
            }
        }
        
        void CreateWakeVortex()
        {
            wakeVortices[wakeIndex] = new WakeVortex
            {
                position = transform.position - rb.linearVelocity.normalized * 5f,
                axis = transform.right * UnityEngine.Random.Range(-1f, 1f),
                strength = wakeStrength * rb.linearVelocity.magnitude / 100f,
                age = 0f
            };
            
            wakeIndex = (wakeIndex + 1) % wakeVortices.Length;
        }
        
        // Utility functions
        float Perlin3D(Vector3 pos)
        {
            float xy = Mathf.PerlinNoise(pos.x, pos.y);
            float xz = Mathf.PerlinNoise(pos.x, pos.z);
            float yz = Mathf.PerlinNoise(pos.y, pos.z);
            return (xy + xz + yz) / 3f;
        }
        
        // Public methods
        public void SetWindProfile(WindProfile profile)
        {
            windProfile = profile;
            ApplyWindProfile();
        }
        
        public void SetBaseWind(Vector3 direction, float speed)
        {
            baseWindDirection = direction.normalized;
            baseWindSpeed = speed;
        }
        
        public void TriggerGust(Vector3 direction, float intensity)
        {
            if (!inGust)
            {
                StartGust();
                gustDirection = direction.normalized;
                gustIntensity = intensity;
            }
        }
        
        public Vector3 GetWindAt(Vector3 position)
        {
            // Calculate wind at any position (useful for particle effects)
            Vector3 wind = baseWindDirection * baseWindSpeed;
            
            // Add turbulence at position
            if (turbulenceEnabled)
            {
                Vector3 noiseInput = position / turbulenceScale + Vector3.one * Time.time * turbulenceSpeed;
                wind += new Vector3(
                    Perlin3D(noiseInput) - 0.5f,
                    Perlin3D(noiseInput + Vector3.one * 100f) - 0.5f,
                    Perlin3D(noiseInput + Vector3.one * 200f) - 0.5f
                ) * turbulenceIntensity * 2f;
            }
            
            return wind;
        }
        
        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !windEnabled) return;
            
            // Draw wind vector
            Gizmos.color = Color.cyan;
            Gizmos.DrawRay(transform.position, currentWindVector * 0.1f);
            
            // Draw gust separately
            if (inGust)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawRay(transform.position + Vector3.up, gustVector * 0.1f);
            }
            
            // Draw wake vortices
            if (wakeEffectsEnabled)
            {
                Gizmos.color = Color.magenta;
                foreach (var vortex in wakeVortices)
                {
                    if (vortex.strength > 0.1f)
                    {
                        Gizmos.DrawWireSphere(vortex.position, vortex.strength);
                    }
                }
            }
        }
    }
}