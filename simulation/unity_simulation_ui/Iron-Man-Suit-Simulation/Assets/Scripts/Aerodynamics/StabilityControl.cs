using UnityEngine;
using System;

namespace IronManSim.Aerodynamics
{
    /// <summary>
    /// Provides automatic stability augmentation and flight envelope protection.
    /// Implements PID controllers for attitude stabilization and prevents dangerous flight conditions.
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    public class StabilityControl : MonoBehaviour
    {
        [Header("Stability Modes")]
        [SerializeField] private bool stabilityAssistEnabled = true;
        [SerializeField] private bool autoLevel = true;
        [SerializeField] private bool stallProtection = true;
        [SerializeField] private bool spinRecovery = true;
        [SerializeField] private bool altitudeHold = false;
        [SerializeField] private bool velocityHold = false;
        
        [Header("PID Controllers")]
        [SerializeField] private PIDController pitchController = new PIDController(2f, 0.1f, 0.5f);
        [SerializeField] private PIDController rollController = new PIDController(3f, 0.1f, 0.3f);
        [SerializeField] private PIDController yawController = new PIDController(1f, 0.05f, 0.2f);
        [SerializeField] private PIDController altitudeController = new PIDController(1f, 0.02f, 0.3f);
        [SerializeField] private PIDController velocityController = new PIDController(0.5f, 0.01f, 0.1f);
        
        [Header("Flight Envelope")]
        [SerializeField] private float maxAngleOfAttack = 25f; // degrees
        [SerializeField] private float maxBankAngle = 60f; // degrees
        [SerializeField] private float maxPitchAngle = 45f; // degrees
        [SerializeField] private float maxRotationRate = 180f; // degrees/second
        [SerializeField] private float minAltitude = 10f; // meters
        [SerializeField] private float maxAltitude = 10000f; // meters
        [SerializeField] private float maxSpeed = 300f; // m/s
        
        [Header("Response Tuning")]
        [SerializeField] private float stabilityAuthority = 0.5f; // 0-1, how much control stability has
        [SerializeField] private float recoveryAggressiveness = 2f;
        [SerializeField] private float dampingFactor = 0.8f;
        
        [Header("Current State")]
        [SerializeField] private Vector3 currentAngles;
        [SerializeField] private Vector3 targetAngles;
        [SerializeField] private Vector3 angularRates;
        [SerializeField] private float currentAltitude;
        [SerializeField] private float targetAltitude;
        [SerializeField] private float currentSpeed;
        [SerializeField] private float targetSpeed;
        [SerializeField] private bool inDangerousState;
        
        // Components
        private Rigidbody rb;
        private AerodynamicForces aeroForces;
        private AtmosphericDensity atmosphere;
        
        // Control outputs
        private Vector3 stabilityTorque;
        private float thrustCommand;
        
        // Properties
        public bool IsStabilityEnabled => stabilityAssistEnabled;
        public Vector3 StabilityTorque => stabilityTorque;
        public float ThrustCommand => thrustCommand;
        public bool InDangerousState => inDangerousState;
        
        [Serializable]
        public class PIDController
        {
            public float kP = 1f;
            public float kI = 0.1f;
            public float kD = 0.1f;
            public float integralMax = 10f;
            
            private float integral;
            private float lastError;
            
            public PIDController(float p, float i, float d)
            {
                kP = p;
                kI = i;
                kD = d;
            }
            
            public float Calculate(float error, float deltaTime)
            {
                // Proportional
                float proportional = kP * error;
                
                // Integral
                integral += error * deltaTime;
                integral = Mathf.Clamp(integral, -integralMax, integralMax);
                float integralTerm = kI * integral;
                
                // Derivative
                float derivative = (error - lastError) / deltaTime;
                float derivativeTerm = kD * derivative;
                
                lastError = error;
                
                return proportional + integralTerm + derivativeTerm;
            }
            
            public void Reset()
            {
                integral = 0;
                lastError = 0;
            }
        }
        
        void Start()
        {
            rb = GetComponent<Rigidbody>();
            aeroForces = GetComponent<AerodynamicForces>();
            atmosphere = GetComponent<AtmosphericDensity>();
        }
        
        void Update()
        {
            UpdateFlightState();
            CheckDangerousStates();
        }
        
        void FixedUpdate()
        {
            if (stabilityAssistEnabled)
            {
                CalculateStabilityControl();
                ApplyStabilityControl();
            }
        }
        
        void UpdateFlightState()
        {
            // Current angles
            currentAngles = transform.rotation.eulerAngles;
            NormalizeAngles(ref currentAngles);
            
            // Angular rates
            angularRates = rb.angularVelocity * Mathf.Rad2Deg;
            
            // Altitude and speed
            currentAltitude = transform.position.y;
            currentSpeed = rb.linearVelocity.magnitude;
            
            // Update target angles based on mode
            if (autoLevel && rb.linearVelocity.magnitude > 5f)
            {
                // Auto-level when moving
                targetAngles.x = 0; // Level pitch
                targetAngles.z = 0; // Level roll
                // Keep current yaw
                targetAngles.y = currentAngles.y;
            }
        }
        
        void CheckDangerousStates()
        {
            inDangerousState = false;
            
            // Check stall condition
            if (stallProtection && aeroForces != null)
            {
                float aoa = aeroForces.AngleOfAttack;
                if (Mathf.Abs(aoa) > maxAngleOfAttack)
                {
                    inDangerousState = true;
                    InitiateStallRecovery();
                }
            }
            
            // Check spin condition
            if (spinRecovery)
            {
                float spinRate = Mathf.Abs(angularRates.y);
                if (spinRate > maxRotationRate)
                {
                    inDangerousState = true;
                    InitiateSpinRecovery();
                }
            }
            
            // Check altitude limits
            if (currentAltitude < minAltitude)
            {
                inDangerousState = true;
                InitiateTerrainAvoidance();
            }
            
            // Check excessive angles
            if (Mathf.Abs(currentAngles.x) > maxPitchAngle || 
                Mathf.Abs(currentAngles.z) > maxBankAngle)
            {
                inDangerousState = true;
            }
        }
        
        void CalculateStabilityControl()
        {
            stabilityTorque = Vector3.zero;
            thrustCommand = 0;
            
            // Attitude control
            if (autoLevel || inDangerousState)
            {
                // Pitch control
                float pitchError = targetAngles.x - currentAngles.x;
                NormalizeAngle(ref pitchError);
                float pitchCommand = pitchController.Calculate(pitchError, Time.fixedDeltaTime);
                
                // Roll control
                float rollError = targetAngles.z - currentAngles.z;
                NormalizeAngle(ref rollError);
                float rollCommand = rollController.Calculate(rollError, Time.fixedDeltaTime);
                
                // Yaw control (with damping)
                float yawError = -angularRates.y * dampingFactor;
                float yawCommand = yawController.Calculate(yawError, Time.fixedDeltaTime);
                
                // Combine commands
                stabilityTorque = new Vector3(pitchCommand, yawCommand, -rollCommand);
                
                // Apply authority limit
                stabilityTorque *= stabilityAuthority;
                
                // Increase authority in dangerous states
                if (inDangerousState)
                {
                    stabilityTorque *= recoveryAggressiveness;
                }
            }
            
            // Altitude hold
            if (altitudeHold && !inDangerousState)
            {
                float altError = targetAltitude - currentAltitude;
                thrustCommand = altitudeController.Calculate(altError, Time.fixedDeltaTime);
                thrustCommand = Mathf.Clamp01(thrustCommand);
            }
            
            // Velocity hold
            if (velocityHold && !inDangerousState)
            {
                float speedError = targetSpeed - currentSpeed;
                float speedCommand = velocityController.Calculate(speedError, Time.fixedDeltaTime);
                
                // Convert to pitch adjustment
                targetAngles.x = Mathf.Clamp(speedCommand * 10f, -20f, 20f);
            }
        }
        
        void ApplyStabilityControl()
        {
            // Apply torque for attitude control
            if (stabilityTorque.magnitude > 0.01f)
            {
                rb.AddRelativeTorque(stabilityTorque);
            }
            
            // Apply thrust for altitude control
            if (thrustCommand > 0)
            {
                Vector3 thrust = transform.up * thrustCommand * 10000f; // Scale appropriately
                rb.AddForce(thrust);
            }
            
            // Apply damping to prevent oscillations
            ApplyAngularDamping();
        }
        
        void ApplyAngularDamping()
        {
            // Additional damping based on angular velocity
            Vector3 damping = -rb.angularVelocity * dampingFactor * 0.5f;
            rb.AddTorque(damping);
        }
        
        void InitiateStallRecovery()
        {
            // Push nose down
            targetAngles.x = -10f;
            
            // Level wings
            targetAngles.z = 0;
            
            // Add thrust
            thrustCommand = 1f;
        }
        
        void InitiateSpinRecovery()
        {
            // Opposite rudder
            float spinDirection = Mathf.Sign(angularRates.y);
            stabilityTorque.y = -spinDirection * recoveryAggressiveness * 100f;
            
            // Neutral ailerons (level wings)
            targetAngles.z = 0;
            
            // Forward stick (reduce angle of attack)
            targetAngles.x = -5f;
        }
        
        void InitiateTerrainAvoidance()
        {
            // Pull up aggressively
            targetAngles.x = 20f;
            
            // Full thrust
            thrustCommand = 1f;
            
            // Level wings for maximum lift
            targetAngles.z = 0;
        }
        
        // Public control methods
        public void SetAltitudeHold(float altitude)
        {
            altitudeHold = true;
            targetAltitude = Mathf.Clamp(altitude, minAltitude, maxAltitude);
        }
        
        public void SetVelocityHold(float speed)
        {
            velocityHold = true;
            targetSpeed = Mathf.Clamp(speed, 0, maxSpeed);
        }
        
        public void SetAutoLevel(bool enabled)
        {
            autoLevel = enabled;
            if (enabled)
            {
                targetAngles = Vector3.zero;
                targetAngles.y = currentAngles.y; // Keep heading
            }
        }
        
        public void DisableAllHolds()
        {
            altitudeHold = false;
            velocityHold = false;
            autoLevel = false;
        }
        
        public void ResetControllers()
        {
            pitchController.Reset();
            rollController.Reset();
            yawController.Reset();
            altitudeController.Reset();
            velocityController.Reset();
        }
        
        // Utility methods
        void NormalizeAngles(ref Vector3 angles)
        {
            // Convert from Unity's 0-360 to -180 to 180
            if (angles.x > 180) angles.x -= 360;
            if (angles.y > 180) angles.y -= 360;
            if (angles.z > 180) angles.z -= 360;
        }
        
        void NormalizeAngle(ref float angle)
        {
            while (angle > 180) angle -= 360;
            while (angle < -180) angle += 360;
        }
        
        void OnDrawGizmos()
        {
            if (!Application.isPlaying) return;
            
            // Draw stability state
            Gizmos.color = inDangerousState ? Color.red : Color.green;
            Gizmos.DrawWireSphere(transform.position, 2f);
            
            // Draw target direction
            if (autoLevel)
            {
                Gizmos.color = Color.cyan;
                Vector3 targetDir = Quaternion.Euler(targetAngles) * Vector3.forward;
                Gizmos.DrawRay(transform.position, targetDir * 5f);
            }
        }
    }
}