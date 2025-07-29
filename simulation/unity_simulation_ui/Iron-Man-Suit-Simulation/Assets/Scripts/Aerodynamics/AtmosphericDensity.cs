using UnityEngine;
using System;

namespace IronManSim.Aerodynamics
{
    /// <summary>
    /// Calculates atmospheric properties based on altitude using the International Standard Atmosphere (ISA) model.
    /// Provides density, pressure, temperature, and speed of sound.
    /// </summary>
    public class AtmosphericDensity : MonoBehaviour
    {
        [Header("Atmosphere Model")]
        [SerializeField] private bool useSimplifiedModel = false;
        [SerializeField] private float seaLevelDensity = 1.225f; // kg/m³
        [SerializeField] private float seaLevelPressure = 101325f; // Pa
        [SerializeField] private float seaLevelTemperature = 288.15f; // K (15°C)
        
        [Header("Current Conditions")]
        [SerializeField] private float currentAltitude;
        [SerializeField] private float currentDensity;
        [SerializeField] private float currentPressure;
        [SerializeField] private float currentTemperature;
        [SerializeField] private float currentSpeedOfSound;
        [SerializeField] private float currentMachNumber;
        
        [Header("Weather Effects")]
        [SerializeField] private float temperatureOffset = 0f; // °C deviation from ISA
        [SerializeField] private float pressureOffset = 0f; // Pa deviation from ISA
        [SerializeField] private float humidity = 0.5f; // 0-1 relative humidity
        
        // Constants
        private const float GRAVITY = 9.80665f; // m/s²
        private const float GAS_CONSTANT = 287.053f; // J/(kg·K)
        private const float GAMMA = 1.4f; // Specific heat ratio for air
        private const float TROPOPAUSE_ALTITUDE = 11000f; // m
        private const float STRATOSPHERE_ALTITUDE = 20000f; // m
        private const float TEMPERATURE_LAPSE_RATE = -0.0065f; // K/m in troposphere
        
        // Properties
        public float Density => currentDensity;
        public float Pressure => currentPressure;
        public float Temperature => currentTemperature;
        public float SpeedOfSound => currentSpeedOfSound;
        public float MachNumber => currentMachNumber;
        public float Altitude => currentAltitude;
        
        void Update()
        {
            UpdateAtmosphericProperties();
        }
        
        void UpdateAtmosphericProperties()
        {
            currentAltitude = transform.position.y;
            
            if (useSimplifiedModel)
            {
                CalculateSimplifiedAtmosphere();
            }
            else
            {
                CalculateISAAtmosphere();
            }
            
            // Apply weather effects
            ApplyWeatherEffects();
            
            // Calculate derived properties
            currentSpeedOfSound = CalculateSpeedOfSound(currentTemperature);
            
            // Calculate Mach number if we have a rigidbody
            var rb = GetComponent<Rigidbody>();
            if (rb != null)
            {
                currentMachNumber = rb.velocity.magnitude / currentSpeedOfSound;
            }
        }
        
        void CalculateSimplifiedAtmosphere()
        {
            // Simple exponential decay model
            float scaleHeight = 8000f; // m
            currentDensity = seaLevelDensity * Mathf.Exp(-currentAltitude / scaleHeight);
            currentPressure = seaLevelPressure * Mathf.Exp(-currentAltitude / scaleHeight);
            currentTemperature = seaLevelTemperature - 2f * (currentAltitude / 1000f); // Simple linear decrease
        }
        
        void CalculateISAAtmosphere()
        {
            // International Standard Atmosphere calculations
            if (currentAltitude < 0)
            {
                // Below sea level
                currentAltitude = 0;
            }
            
            if (currentAltitude <= TROPOPAUSE_ALTITUDE)
            {
                // Troposphere (0-11km)
                CalculateTroposphere();
            }
            else if (currentAltitude <= STRATOSPHERE_ALTITUDE)
            {
                // Lower Stratosphere (11-20km)
                CalculateLowerStratosphere();
            }
            else
            {
                // Upper atmosphere (>20km)
                CalculateUpperAtmosphere();
            }
        }
        
        void CalculateTroposphere()
        {
            // Temperature decreases linearly with altitude
            currentTemperature = seaLevelTemperature + TEMPERATURE_LAPSE_RATE * currentAltitude;
            
            // Pressure using barometric formula
            float tempRatio = currentTemperature / seaLevelTemperature;
            float exponent = -GRAVITY / (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE);
            currentPressure = seaLevelPressure * Mathf.Pow(tempRatio, exponent);
            
            // Density from ideal gas law
            currentDensity = currentPressure / (GAS_CONSTANT * currentTemperature);
        }
        
        void CalculateLowerStratosphere()
        {
            // Temperature is constant in lower stratosphere
            currentTemperature = 216.65f; // K (-56.5°C)
            
            // First calculate conditions at tropopause
            float tropoPressure = seaLevelPressure * Mathf.Pow(216.65f / seaLevelTemperature, 
                -GRAVITY / (GAS_CONSTANT * TEMPERATURE_LAPSE_RATE));
            
            // Then exponential decay from tropopause
            float heightAboveTropo = currentAltitude - TROPOPAUSE_ALTITUDE;
            float exponent = -GRAVITY * heightAboveTropo / (GAS_CONSTANT * currentTemperature);
            currentPressure = tropoPressure * Mathf.Exp(exponent);
            
            // Density from ideal gas law
            currentDensity = currentPressure / (GAS_CONSTANT * currentTemperature);
        }
        
        void CalculateUpperAtmosphere()
        {
            // Simplified model for upper atmosphere
            // Temperature increases slightly
            currentTemperature = 216.65f + 0.001f * (currentAltitude - STRATOSPHERE_ALTITUDE);
            
            // Very low pressure and density
            float heightFactor = currentAltitude / STRATOSPHERE_ALTITUDE;
            currentPressure = 5474.89f * Mathf.Exp(-heightFactor * 2f); // Rough approximation
            currentDensity = currentPressure / (GAS_CONSTANT * currentTemperature);
        }
        
        void ApplyWeatherEffects()
        {
            // Temperature deviation
            currentTemperature += temperatureOffset + 273.15f; // Convert °C offset to K
            
            // Pressure deviation
            currentPressure += pressureOffset;
            
            // Humidity effects on density (simplified)
            // Water vapor is less dense than dry air
            float vaporPressure = CalculateVaporPressure(currentTemperature) * humidity;
            float dryPressure = currentPressure - vaporPressure;
            
            // Adjusted density accounting for humidity
            float Rd = 287.058f; // Specific gas constant for dry air
            float Rv = 461.495f; // Specific gas constant for water vapor
            
            currentDensity = (dryPressure / (Rd * currentTemperature)) + 
                           (vaporPressure / (Rv * currentTemperature));
        }
        
        float CalculateVaporPressure(float temperature)
        {
            // Magnus formula for saturation vapor pressure
            float tempC = temperature - 273.15f;
            return 610.78f * Mathf.Exp(17.2694f * tempC / (tempC + 238.3f));
        }
        
        float CalculateSpeedOfSound(float temperature)
        {
            // Speed of sound: a = sqrt(³ * R * T)
            return Mathf.Sqrt(GAMMA * GAS_CONSTANT * temperature);
        }
        
        // Public methods for other components
        public float GetDensity(float altitude)
        {
            // Quick calculation for any altitude
            if (Mathf.Abs(altitude - currentAltitude) < 1f)
            {
                return currentDensity;
            }
            
            // Otherwise calculate for specific altitude
            float temp = currentAltitude;
            currentAltitude = altitude;
            CalculateISAAtmosphere();
            float density = currentDensity;
            currentAltitude = temp;
            return density;
        }
        
        public float GetPressure(float altitude)
        {
            if (Mathf.Abs(altitude - currentAltitude) < 1f)
            {
                return currentPressure;
            }
            
            float temp = currentAltitude;
            currentAltitude = altitude;
            CalculateISAAtmosphere();
            float pressure = currentPressure;
            currentAltitude = temp;
            return pressure;
        }
        
        public Vector3 GetWindVector()
        {
            // Placeholder for wind calculation
            // In full implementation, would integrate with WindInteraction component
            return Vector3.zero;
        }
        
        public float GetDensityRatio()
        {
            // Ratio compared to sea level
            return currentDensity / seaLevelDensity;
        }
        
        public float GetPressureRatio()
        {
            // Ratio compared to sea level
            return currentPressure / seaLevelPressure;
        }
        
        void OnValidate()
        {
            // Clamp values to reasonable ranges
            humidity = Mathf.Clamp01(humidity);
            temperatureOffset = Mathf.Clamp(temperatureOffset, -50f, 50f);
            pressureOffset = Mathf.Clamp(pressureOffset, -10000f, 10000f);
        }
    }
}