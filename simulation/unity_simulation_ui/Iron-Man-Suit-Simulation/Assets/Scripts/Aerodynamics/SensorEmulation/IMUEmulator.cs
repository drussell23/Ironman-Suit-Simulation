using UnityEngine;
using System;

namespace IronManSim.Aerodynamics.SensorEmulation
{
    /// <summary>
    /// Emulates an Inertial Measurement Unit (IMU) with realistic noise and error characteristics.
    /// Provides accelerometer, gyroscope, and magnetometer data.
    /// </summary>
    public class IMUEmulator : MonoBehaviour
    {
        [Header("IMU Configuration")]
        [SerializeField] private float updateRate = 100f; // Hz
        [SerializeField] private bool addNoise = true;
        [SerializeField] private bool addBias = true;
        [SerializeField] private bool addDrift = true;
        
        [Header("Accelerometer Settings")]
        [SerializeField] private float accelNoiseDensity = 0.01f; // m/s²/Hz
        [SerializeField] private float accelBias = 0.05f; // m/s²
        [SerializeField] private float accelScaleFactor = 1.0f;
        [SerializeField] private float accelRange = 16f; // ±g
        
        [Header("Gyroscope Settings")]
        [SerializeField] private float gyroNoiseDensity = 0.01f; // rad/s/Hz
        [SerializeField] private float gyroBias = 0.002f; // rad/s
        [SerializeField] private float gyroDriftRate = 0.0001f; // rad/s²
        [SerializeField] private float gyroRange = 2000f; // deg/s
        
        [Header("Magnetometer Settings")]
        [SerializeField] private float magNoiseDensity = 0.1f; // ¼T/Hz
        [SerializeField] private float magBias = 5f; // ¼T
        [SerializeField] private Vector3 magneticNorth = new Vector3(0, 0, 1);
        [SerializeField] private float magneticFieldStrength = 50f; // ¼T
        
        [Header("Temperature Effects")]
        [SerializeField] private bool temperatureEffects = true;
        [SerializeField] private float currentTemperature = 20f; // °C
        [SerializeField] private float temperatureCoefficient = 0.001f; // per °C
        
        [Header("Current Readings")]
        [SerializeField] private Vector3 acceleration;
        [SerializeField] private Vector3 angularVelocity;
        [SerializeField] private Vector3 magneticField;
        [SerializeField] private float temperature;
        
        // Components
        private Rigidbody rb;
        private float lastUpdateTime;
        private float updateInterval;
        
        // Bias tracking
        private Vector3 accelBiasVector;
        private Vector3 gyroBiasVector;
        private Vector3 gyroDrift;
        private Vector3 magBiasVector;
        
        // Previous values for derivative calculation
        private Vector3 previousVelocity;
        private Vector3 previousAngularVelocity;
        
        // Random number generators
        private System.Random random;
        
        // Public properties
        public Vector3 Acceleration => acceleration;
        public Vector3 AngularVelocity => angularVelocity;
        public Vector3 MagneticField => magneticField;
        public float Temperature => temperature;
        public float UpdateRate => updateRate;
        
        // Data structure for IMU output
        [Serializable]
        public struct IMUData
        {
            public Vector3 acceleration;
            public Vector3 angularVelocity;
            public Vector3 magneticField;
            public float temperature;
            public float timestamp;
        }
        
        void Start()
        {
            rb = GetComponent<Rigidbody>();
            if (rb == null)
            {
                Debug.LogError("IMUEmulator requires a Rigidbody component!");
                enabled = false;
                return;
            }
            
            updateInterval = 1f / updateRate;
            lastUpdateTime = Time.time;
            random = new System.Random();
            
            InitializeBiases();
        }
        
        void InitializeBiases()
        {
            // Initialize random biases
            if (addBias)
            {
                accelBiasVector = new Vector3(
                    GetGaussian() * accelBias,
                    GetGaussian() * accelBias,
                    GetGaussian() * accelBias
                );
                
                gyroBiasVector = new Vector3(
                    GetGaussian() * gyroBias,
                    GetGaussian() * gyroBias,
                    GetGaussian() * gyroBias
                );
                
                magBiasVector = new Vector3(
                    GetGaussian() * magBias,
                    GetGaussian() * magBias,
                    GetGaussian() * magBias
                );
            }
            
            gyroDrift = Vector3.zero;
        }
        
        void Update()
        {
            // Update at specified rate
            if (Time.time - lastUpdateTime >= updateInterval)
            {
                UpdateIMUReadings();
                lastUpdateTime = Time.time;
            }
            
            // Update temperature simulation
            if (temperatureEffects)
            {
                UpdateTemperature();
            }
        }
        
        void UpdateIMUReadings()
        {
            // Calculate true values
            Vector3 trueAccel = CalculateTrueAcceleration();
            Vector3 trueGyro = rb.angularVelocity;
            Vector3 trueMag = CalculateTrueMagneticField();
            
            // Apply sensor characteristics
            acceleration = ProcessAccelerometer(trueAccel);
            angularVelocity = ProcessGyroscope(trueGyro);
            magneticField = ProcessMagnetometer(trueMag);
            
            // Update drift
            if (addDrift)
            {
                gyroDrift += new Vector3(
                    GetGaussian() * gyroDriftRate * updateInterval,
                    GetGaussian() * gyroDriftRate * updateInterval,
                    GetGaussian() * gyroDriftRate * updateInterval
                );
            }
        }
        
        Vector3 CalculateTrueAcceleration()
        {
            // Calculate acceleration from velocity change
            Vector3 currentVelocity = rb.linearVelocity;
            Vector3 linearAccel = (currentVelocity - previousVelocity) / updateInterval;
            previousVelocity = currentVelocity;
            
            // Add gravity (in body frame)
            Vector3 gravity = transform.InverseTransformDirection(Physics.gravity);
            
            // Add centripetal acceleration
            Vector3 centripetal = Vector3.Cross(rb.angularVelocity, 
                Vector3.Cross(rb.angularVelocity, rb.centerOfMass));
            
            // Total acceleration in body frame
            return transform.InverseTransformDirection(linearAccel) - gravity + centripetal;
        }
        
        Vector3 CalculateTrueMagneticField()
        {
            // Transform magnetic north to body frame
            Vector3 bodyMag = transform.InverseTransformDirection(magneticNorth) * magneticFieldStrength;
            
            // Add hard iron distortion (from suit electronics)
            Vector3 hardIron = new Vector3(10f, -5f, 2f);
            
            // Add soft iron distortion (simplified)
            Matrix4x4 softIron = Matrix4x4.Scale(new Vector3(1.1f, 0.9f, 1.05f));
            bodyMag = softIron.MultiplyVector(bodyMag);
            
            return bodyMag + hardIron;
        }
        
        Vector3 ProcessAccelerometer(Vector3 trueValue)
        {
            Vector3 measured = trueValue;
            
            // Apply scale factor and misalignment
            measured *= accelScaleFactor;
            
            // Add bias
            if (addBias)
            {
                measured += accelBiasVector;
            }
            
            // Add temperature effects
            if (temperatureEffects)
            {
                float tempError = (temperature - 20f) * temperatureCoefficient;
                measured *= (1f + tempError);
            }
            
            // Add noise
            if (addNoise)
            {
                float noiseMagnitude = accelNoiseDensity * Mathf.Sqrt(updateRate);
                measured += new Vector3(
                    GetGaussian() * noiseMagnitude,
                    GetGaussian() * noiseMagnitude,
                    GetGaussian() * noiseMagnitude
                );
            }
            
            // Apply range limits
            measured = ClampToRange(measured, accelRange * 9.81f);
            
            return measured;
        }
        
        Vector3 ProcessGyroscope(Vector3 trueValue)
        {
            Vector3 measured = trueValue;
            
            // Add bias and drift
            if (addBias)
            {
                measured += gyroBiasVector + gyroDrift;
            }
            
            // Add temperature effects
            if (temperatureEffects)
            {
                float tempError = (temperature - 20f) * temperatureCoefficient * 2f;
                measured *= (1f + tempError);
            }
            
            // Add noise
            if (addNoise)
            {
                float noiseMagnitude = gyroNoiseDensity * Mathf.Sqrt(updateRate);
                measured += new Vector3(
                    GetGaussian() * noiseMagnitude,
                    GetGaussian() * noiseMagnitude,
                    GetGaussian() * noiseMagnitude
                );
            }
            
            // Apply range limits
            float rangeRad = gyroRange * Mathf.Deg2Rad;
            measured = ClampToRange(measured, rangeRad);
            
            return measured;
        }
        
        Vector3 ProcessMagnetometer(Vector3 trueValue)
        {
            Vector3 measured = trueValue;
            
            // Add bias
            if (addBias)
            {
                measured += magBiasVector;
            }
            
            // Add noise
            if (addNoise)
            {
                float noiseMagnitude = magNoiseDensity * Mathf.Sqrt(updateRate);
                measured += new Vector3(
                    GetGaussian() * noiseMagnitude,
                    GetGaussian() * noiseMagnitude,
                    GetGaussian() * noiseMagnitude
                );
            }
            
            // Add interference from electronics
            float interference = Mathf.Sin(Time.time * 50f) * 2f; // 50Hz interference
            measured += Vector3.one * interference;
            
            return measured;
        }
        
        void UpdateTemperature()
        {
            // Simple temperature model based on altitude and activity
            float altitude = transform.position.y;
            float altitudeTemp = 20f - (altitude / 1000f) * 6.5f; // Standard lapse rate
            
            // Heat from electronics and thrusters
            float heatGeneration = rb.linearVelocity.magnitude * 0.1f;
            
            // Temperature changes slowly
            temperature = Mathf.Lerp(temperature, altitudeTemp + heatGeneration, Time.deltaTime * 0.1f);
            
            // Add some noise
            temperature += GetGaussian() * 0.1f;
        }
        
        Vector3 ClampToRange(Vector3 value, float range)
        {
            return new Vector3(
                Mathf.Clamp(value.x, -range, range),
                Mathf.Clamp(value.y, -range, range),
                Mathf.Clamp(value.z, -range, range)
            );
        }
        
        float GetGaussian()
        {
            // Box-Muller transform for Gaussian distribution
            float u1 = 1f - (float)random.NextDouble();
            float u2 = 1f - (float)random.NextDouble();
            return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Sin(2f * Mathf.PI * u2);
        }
        
        // Public methods for accessing IMU data
        public IMUData GetIMUData()
        {
            return new IMUData
            {
                acceleration = acceleration,
                angularVelocity = angularVelocity,
                magneticField = magneticField,
                temperature = temperature,
                timestamp = Time.time
            };
        }
        
        public void Calibrate()
        {
            // Simple calibration routine
            if (rb.linearVelocity.magnitude < 0.1f && rb.angularVelocity.magnitude < 0.01f)
            {
                // Stationary calibration
                accelBiasVector = -acceleration + transform.InverseTransformDirection(-Physics.gravity);
                gyroBiasVector = -angularVelocity;
                gyroDrift = Vector3.zero;
                
                Debug.Log("IMU Calibrated!");
            }
            else
            {
                Debug.LogWarning("IMU must be stationary for calibration!");
            }
        }
        
        public void SetUpdateRate(float hz)
        {
            updateRate = Mathf.Clamp(hz, 1f, 1000f);
            updateInterval = 1f / updateRate;
        }
        
        public void ResetDrift()
        {
            gyroDrift = Vector3.zero;
        }
        
        void OnDrawGizmosSelected()
        {
            if (!Application.isPlaying) return;
            
            // Draw acceleration vector
            Gizmos.color = Color.red;
            Gizmos.DrawRay(transform.position, transform.TransformDirection(acceleration) * 0.1f);
            
            // Draw magnetic field vector
            Gizmos.color = Color.blue;
            Gizmos.DrawRay(transform.position, transform.TransformDirection(magneticField.normalized) * 2f);
            
            // Draw angular velocity
            Gizmos.color = Color.green;
            Gizmos.DrawRay(transform.position, transform.TransformDirection(angularVelocity) * 0.5f);
        }
    }
}