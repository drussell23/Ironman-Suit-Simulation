using UnityEngine;
using System;

namespace IronManSim.Aerodynamics.SensorEmulation
{
    /// <summary>
    /// Emulates a pitot tube sensor for airspeed measurement.
    /// Includes realistic errors, calibration effects, and environmental factors.
    /// </summary>
    public class PitotEmulator : MonoBehaviour
    {
        [Header("Pitot Configuration")]
        [SerializeField] private float updateRate = 50f; // Hz
        [SerializeField] private Vector3 pitotPosition = new Vector3(0, 0, 1f); // Position relative to suit
        [SerializeField] private Vector3 pitotDirection = Vector3.forward; // Probe direction
        [SerializeField] private float probeDiameter = 0.01f; // meters
        
        [Header("Sensor Characteristics")]
        [SerializeField] private bool addNoise = true;
        [SerializeField] private float pressureNoise = 1f; // Pa
        [SerializeField] private float calibrationError = 0.02f; // 2% error
        [SerializeField] private float responseTime = 0.1f; // seconds
        [SerializeField] private float minDetectableSpeed = 2f; // m/s
        
        [Header("Environmental Effects")]
        [SerializeField] private bool iceAccumulation = false;
        [SerializeField] private float iceBuildup = 0f; // 0-1
        [SerializeField] private bool waterIngestion = false;
        [SerializeField] private float blockageLevel = 0f; // 0-1
        
        [Header("Angle of Attack Effects")]
        [SerializeField] private AnimationCurve aoaErrorCurve; // Error vs AoA
        [SerializeField] private float maxAoA = 30f; // degrees
        
        [Header("Current Readings")]
        [SerializeField] private float indicatedAirspeed; // m/s
        [SerializeField] private float calibratedAirspeed; // m/s
        [SerializeField] private float trueAirspeed; // m/s
        [SerializeField] private float dynamicPressure; // Pa
        [SerializeField] private float staticPressure; // Pa
        [SerializeField] private float totalPressure; // Pa
        [SerializeField] private float machNumber;
        
        // Components
        private Rigidbody rb;
        private AtmosphericDensity atmosphere;
        private WindInteraction wind;
        private AerodynamicForces aeroForces;
        
        // Internal state
        private float lastUpdateTime;
        private float updateInterval;
        private float filteredSpeed;
        private System.Random random;
        
        // Public properties
        public float IndicatedAirspeed => indicatedAirspeed;
        public float CalibratedAirspeed => calibratedAirspeed;
        public float TrueAirspeed => trueAirspeed;
        public float DynamicPressure => dynamicPressure;
        public float MachNumber => machNumber;
        public bool IsOperational => !iceAccumulation && blockageLevel < 0.5f;
        
        // Data structure for pitot output
        [Serializable]
        public struct PitotData
        {
            public float indicatedAirspeed;
            public float calibratedAirspeed;
            public float trueAirspeed;
            public float dynamicPressure;
            public float staticPressure;
            public float machNumber;
            public bool operational;
            public float timestamp;
        }
        
        void Start()
        {
            rb = GetComponent<Rigidbody>();
            atmosphere = GetComponent<AtmosphericDensity>();
            wind = GetComponent<WindInteraction>();
            aeroForces = GetComponent<AerodynamicForces>();
            
            updateInterval = 1f / updateRate;
            lastUpdateTime = Time.time;
            random = new System.Random();
            
            InitializeAoACurve();
        }
        
        void InitializeAoACurve()
        {
            if (aoaErrorCurve == null || aoaErrorCurve.length == 0)
            {
                aoaErrorCurve = new AnimationCurve();
                aoaErrorCurve.AddKey(0f, 1.0f);   // No error at 0° AoA
                aoaErrorCurve.AddKey(10f, 0.98f); // 2% error at 10°
                aoaErrorCurve.AddKey(20f, 0.95f); // 5% error at 20°
                aoaErrorCurve.AddKey(30f, 0.90f); // 10% error at 30°
                aoaErrorCurve.AddKey(45f, 0.80f); // 20% error at 45°
            }
        }
        
        void Update()
        {
            // Update at specified rate
            if (Time.time - lastUpdateTime >= updateInterval)
            {
                UpdatePitotReadings();
                lastUpdateTime = Time.time;
            }
            
            // Update environmental effects
            UpdateEnvironmentalEffects();
        }
        
        void UpdatePitotReadings()
        {
            // Get true airspeed vector
            Vector3 velocity = rb.linearVelocity;
            if (wind != null)
            {
                velocity -= wind.CurrentWind;
            }
            
            // Calculate relative velocity at pitot position
            Vector3 worldPitotPos = transform.TransformPoint(pitotPosition);
            Vector3 rotationalVelocity = Vector3.Cross(rb.angularVelocity, worldPitotPos - transform.position);
            Vector3 totalVelocity = velocity + rotationalVelocity;
            
            // True airspeed
            trueAirspeed = totalVelocity.magnitude;
            
            // Get atmospheric conditions
            float density = atmosphere != null ? atmosphere.Density : 1.225f;
            float pressure = atmosphere != null ? atmosphere.Pressure : 101325f;
            float temperature = atmosphere != null ? atmosphere.Temperature : 288.15f;
            float speedOfSound = atmosphere != null ? atmosphere.SpeedOfSound : 340.29f;
            
            // Calculate pressures
            staticPressure = pressure;
            
            if (trueAirspeed > minDetectableSpeed && IsOperational)
            {
                // Dynamic pressure: q = 0.5 * Á * V²
                dynamicPressure = 0.5f * density * trueAirspeed * trueAirspeed;
                
                // Apply angle of attack correction
                float aoaCorrection = CalculateAoACorrection();
                dynamicPressure *= aoaCorrection;
                
                // Apply blockage effects
                if (blockageLevel > 0)
                {
                    dynamicPressure *= (1f - blockageLevel * 0.8f);
                }
                
                // Total pressure
                totalPressure = staticPressure + dynamicPressure;
                
                // Calculate indicated airspeed (IAS)
                // Using Bernoulli's equation at sea level conditions
                float seaLevelDensity = 1.225f;
                indicatedAirspeed = Mathf.Sqrt(2f * dynamicPressure / seaLevelDensity);
                
                // Apply calibration error
                indicatedAirspeed *= (1f + GetCalibrationError());
                
                // Add noise
                if (addNoise)
                {
                    float noise = GetGaussian() * pressureNoise / seaLevelDensity;
                    indicatedAirspeed += noise;
                }
                
                // Apply response lag
                filteredSpeed = Mathf.Lerp(filteredSpeed, indicatedAirspeed, Time.deltaTime / responseTime);
                indicatedAirspeed = filteredSpeed;
                
                // Calculate calibrated airspeed (CAS)
                // Correct for compressibility at high speeds
                calibratedAirspeed = CalculateCAS(indicatedAirspeed, pressure);
                
                // Mach number
                machNumber = trueAirspeed / speedOfSound;
            }
            else
            {
                // Below minimum detectable speed or blocked
                dynamicPressure = 0;
                totalPressure = staticPressure;
                indicatedAirspeed = 0;
                calibratedAirspeed = 0;
                machNumber = 0;
                filteredSpeed = 0;
            }
        }
        
        float CalculateAoACorrection()
        {
            if (aeroForces == null) return 1f;
            
            float aoa = Mathf.Abs(aeroForces.AngleOfAttack);
            
            // Pitot tube alignment with flow
            Vector3 flowDirection = rb.linearVelocity.normalized;
            Vector3 worldPitotDir = transform.TransformDirection(pitotDirection);
            float alignment = Vector3.Dot(flowDirection, worldPitotDir);
            
            // Use curve for AoA error
            float curveError = aoaErrorCurve.Evaluate(aoa);
            
            // Combine alignment and AoA effects
            return Mathf.Max(0.1f, alignment * curveError);
        }
        
        float GetCalibrationError()
        {
            // Systematic calibration error
            float baseError = calibrationError;
            
            // Temperature effects
            if (atmosphere != null)
            {
                float tempDeviation = (atmosphere.Temperature - 288.15f) / 288.15f;
                baseError += tempDeviation * 0.01f; // 1% per 100% temp change
            }
            
            // Add some drift over time
            float drift = Mathf.Sin(Time.time * 0.01f) * 0.005f; // 0.5% drift
            
            return baseError + drift;
        }
        
        float CalculateCAS(float ias, float staticPressure)
        {
            // Compressibility correction for high-speed flight
            float seaLevelPressure = 101325f;
            float pressureRatio = staticPressure / seaLevelPressure;
            
            // Simplified compressibility correction
            float compressibilityFactor = 1f;
            if (machNumber > 0.3f)
            {
                compressibilityFactor = 1f + 0.125f * machNumber * machNumber;
            }
            
            return ias * Mathf.Sqrt(pressureRatio) * compressibilityFactor;
        }
        
        void UpdateEnvironmentalEffects()
        {
            // Ice accumulation in cold, moist conditions
            if (atmosphere != null && atmosphere.Temperature < 273.15f) // Below freezing
            {
                float moistureLevel = 0.5f; // Simplified - would need humidity data
                if (moistureLevel > 0.3f && trueAirspeed > 50f)
                {
                    iceBuildup += Time.deltaTime * 0.1f;
                    iceBuildup = Mathf.Clamp01(iceBuildup);
                    
                    if (iceBuildup > 0.5f)
                    {
                        iceAccumulation = true;
                    }
                }
            }
            else if (atmosphere != null && atmosphere.Temperature > 283.15f) // Above 10°C
            {
                // Ice melts
                iceBuildup -= Time.deltaTime * 0.2f;
                iceBuildup = Mathf.Max(0, iceBuildup);
                
                if (iceBuildup < 0.1f)
                {
                    iceAccumulation = false;
                }
            }
            
            // Water ingestion during rain (simplified)
            if (wind != null && wind.WindSpeed > 20f) // High wind = possible rain
            {
                waterIngestion = UnityEngine.Random.value < 0.1f;
                if (waterIngestion)
                {
                    blockageLevel = Mathf.Min(blockageLevel + Time.deltaTime * 0.5f, 0.8f);
                }
            }
            else
            {
                waterIngestion = false;
                blockageLevel = Mathf.Max(0, blockageLevel - Time.deltaTime * 0.3f);
            }
        }
        
        float GetGaussian()
        {
            // Box-Muller transform
            float u1 = 1f - (float)random.NextDouble();
            float u2 = 1f - (float)random.NextDouble();
            return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Sin(2f * Mathf.PI * u2);
        }
        
        // Public methods
        public PitotData GetPitotData()
        {
            return new PitotData
            {
                indicatedAirspeed = indicatedAirspeed,
                calibratedAirspeed = calibratedAirspeed,
                trueAirspeed = trueAirspeed,
                dynamicPressure = dynamicPressure,
                staticPressure = staticPressure,
                machNumber = machNumber,
                operational = IsOperational,
                timestamp = Time.time
            };
        }
        
        public void ActivateHeater()
        {
            // Pitot heater to prevent ice
            if (iceBuildup > 0)
            {
                iceBuildup -= Time.deltaTime * 0.5f;
                iceBuildup = Mathf.Max(0, iceBuildup);
            }
        }
        
        public void ClearBlockage()
        {
            // Manual blockage clearing
            blockageLevel = 0;
            waterIngestion = false;
        }
        
        public void Calibrate(float knownSpeed)
        {
            // In-flight calibration
            if (trueAirspeed > minDetectableSpeed)
            {
                float error = (indicatedAirspeed - knownSpeed) / knownSpeed;
                calibrationError = -error;
                Debug.Log($"Pitot calibrated with {error*100:F1}% correction");
            }
        }
        
        void OnDrawGizmosSelected()
        {
            // Draw pitot tube position and direction
            Vector3 worldPos = transform.TransformPoint(pitotPosition);
            Vector3 worldDir = transform.TransformDirection(pitotDirection);
            
            Gizmos.color = IsOperational ? Color.green : Color.red;
            Gizmos.DrawWireSphere(worldPos, probeDiameter * 10f);
            Gizmos.DrawRay(worldPos, worldDir * 0.5f);
            
            // Draw airflow
            if (Application.isPlaying && rb != null)
            {
                Gizmos.color = Color.cyan;
                Gizmos.DrawRay(worldPos, -rb.linearVelocity.normalized);
            }
        }
    }
}