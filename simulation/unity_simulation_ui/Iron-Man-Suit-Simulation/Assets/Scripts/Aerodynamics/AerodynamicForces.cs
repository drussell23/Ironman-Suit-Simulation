using UnityEngine;
using System;

namespace IronManSim.Aerodynamics
{
    /// <summary>
    /// Calculates and applies aerodynamic forces (lift, drag, side force) to the Iron Man suit.
    /// Based on real aerodynamic principles with suit-specific modifications.
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    public class AerodynamicForces : MonoBehaviour
    {
        [Header("Aerodynamic Properties")]
        [SerializeField] private float referenceArea = 2.0f; // m² - Suit frontal area
        [SerializeField] private float aspectRatio = 1.5f; // Wing-like surfaces aspect ratio
        
        [Header("Coefficients")]
        [SerializeField] private AnimationCurve liftCurve; // CL vs angle of attack
        [SerializeField] private AnimationCurve dragCurve; // CD vs angle of attack
        [SerializeField] private float CL0 = 0.1f; // Base lift coefficient
        [SerializeField] private float CLalpha = 5.0f; // Lift curve slope (per radian)
        [SerializeField] private float CD0 = 0.02f; // Parasitic drag coefficient
        [SerializeField] private float CDi = 0.1f; // Induced drag factor
        [SerializeField] private float stallAngle = 25f; // Degrees
        
        [Header("Control Surfaces")]
        [SerializeField] private float elevatorEffectiveness = 0.5f;
        [SerializeField] private float aileronEffectiveness = 0.3f;
        [SerializeField] private float rudderEffectiveness = 0.2f;
        
        [Header("Current State")]
        [SerializeField] private float currentAngleOfAttack;
        [SerializeField] private float currentSideslipAngle;
        [SerializeField] private Vector3 currentLift;
        [SerializeField] private Vector3 currentDrag;
        [SerializeField] private float dynamicPressure;
        
        // Components
        private Rigidbody rb;
        private AtmosphericDensity atmosphere;
        
        // Properties
        public float AngleOfAttack => currentAngleOfAttack;
        public float DynamicPressure => dynamicPressure;
        public Vector3 Lift => currentLift;
        public Vector3 Drag => currentDrag;
        public float LiftMagnitude => currentLift.magnitude;
        public float DragMagnitude => currentDrag.magnitude;
        
        void Start()
        {
            rb = GetComponent<Rigidbody>();
            atmosphere = GetComponent<AtmosphericDensity>();
            
            // Initialize curves if not set
            if (liftCurve == null || liftCurve.length == 0)
            {
                InitializeDefaultCurves();
            }
        }
        
        void FixedUpdate()
        {
            CalculateAerodynamicForces();
            ApplyAerodynamicMoments();
        }
        
        void CalculateAerodynamicForces()
        {
            // Get atmospheric conditions
            float altitude = transform.position.y;
            float density = atmosphere != null ? atmosphere.GetDensity(altitude) : 1.225f;
            
            // Calculate relative wind
            Vector3 velocity = rb.velocity;
            float speed = velocity.magnitude;
            
            // Avoid division by zero
            if (speed < 0.1f)
            {
                currentLift = Vector3.zero;
                currentDrag = Vector3.zero;
                dynamicPressure = 0;
                return;
            }
            
            // Dynamic pressure: q = 0.5 * Á * V²
            dynamicPressure = 0.5f * density * speed * speed;
            
            // Calculate angles
            Vector3 localVelocity = transform.InverseTransformDirection(velocity);
            currentAngleOfAttack = Mathf.Atan2(-localVelocity.y, localVelocity.z) * Mathf.Rad2Deg;
            currentSideslipAngle = Mathf.Atan2(localVelocity.x, localVelocity.z) * Mathf.Rad2Deg;
            
            // Get coefficients
            float CL = CalculateLiftCoefficient(currentAngleOfAttack);
            float CD = CalculateDragCoefficient(currentAngleOfAttack, CL);
            float CY = CalculateSideForceCoefficient(currentSideslipAngle);
            
            // Calculate forces
            float liftMagnitude = dynamicPressure * referenceArea * CL;
            float dragMagnitude = dynamicPressure * referenceArea * CD;
            float sideForceMagnitude = dynamicPressure * referenceArea * CY;
            
            // Get force directions in world space
            Vector3 velocityNormalized = velocity.normalized;
            Vector3 liftDirection = Vector3.Cross(Vector3.Cross(velocityNormalized, transform.up), velocityNormalized).normalized;
            Vector3 dragDirection = -velocityNormalized;
            Vector3 sideDirection = Vector3.Cross(transform.up, velocityNormalized).normalized;
            
            // Apply forces
            currentLift = liftDirection * liftMagnitude;
            currentDrag = dragDirection * dragMagnitude;
            Vector3 sideForce = sideDirection * sideForceMagnitude;
            
            // Apply to rigidbody
            rb.AddForce(currentLift + currentDrag + sideForce);
        }
        
        float CalculateLiftCoefficient(float angleOfAttack)
        {
            // Use curve if available
            if (liftCurve != null && liftCurve.length > 0)
            {
                return liftCurve.Evaluate(angleOfAttack);
            }
            
            // Otherwise use linear model with stall
            float alpha = angleOfAttack * Mathf.Deg2Rad;
            float CL = CL0 + CLalpha * alpha;
            
            // Simple stall model
            if (Mathf.Abs(angleOfAttack) > stallAngle)
            {
                float stallFactor = 1f - Mathf.Clamp01((Mathf.Abs(angleOfAttack) - stallAngle) / 10f);
                CL *= stallFactor;
            }
            
            return CL;
        }
        
        float CalculateDragCoefficient(float angleOfAttack, float CL)
        {
            // Use curve if available
            if (dragCurve != null && dragCurve.length > 0)
            {
                return dragCurve.Evaluate(angleOfAttack);
            }
            
            // Parabolic drag polar: CD = CD0 + CDi * CL²
            float inducedDrag = CDi * CL * CL / (Mathf.PI * aspectRatio);
            
            // Add form drag increase at high angles
            float alpha = Mathf.Abs(angleOfAttack);
            float formDragFactor = 1f + 0.1f * Mathf.Pow(alpha / 30f, 2);
            
            return (CD0 + inducedDrag) * formDragFactor;
        }
        
        float CalculateSideForceCoefficient(float sideslipAngle)
        {
            // Simple linear model for side force
            return -0.05f * sideslipAngle * Mathf.Deg2Rad;
        }
        
        void ApplyAerodynamicMoments()
        {
            // Calculate control surface deflections (would come from input)
            float elevator = Input.GetAxis("Vertical");
            float aileron = Input.GetAxis("Horizontal");
            float rudder = Input.GetAxis("Yaw");
            
            // Calculate moments
            float pitchMoment = elevator * elevatorEffectiveness * dynamicPressure * referenceArea;
            float rollMoment = aileron * aileronEffectiveness * dynamicPressure * referenceArea;
            float yawMoment = rudder * rudderEffectiveness * dynamicPressure * referenceArea;
            
            // Apply torques
            rb.AddRelativeTorque(new Vector3(pitchMoment, yawMoment, -rollMoment));
            
            // Add stability augmentation
            ApplyStabilityAugmentation();
        }
        
        void ApplyStabilityAugmentation()
        {
            // Damping to prevent oscillations
            Vector3 angularVelocity = rb.angularVelocity;
            float dampingFactor = 0.5f * dynamicPressure * referenceArea;
            
            Vector3 dampingTorque = -angularVelocity * dampingFactor * 0.1f;
            rb.AddTorque(dampingTorque);
        }
        
        void InitializeDefaultCurves()
        {
            // Create default lift curve
            liftCurve = new AnimationCurve();
            liftCurve.AddKey(-180f, 0f);
            liftCurve.AddKey(-90f, -1f);
            liftCurve.AddKey(-stallAngle, -1.2f);
            liftCurve.AddKey(0f, CL0);
            liftCurve.AddKey(stallAngle, 1.2f);
            liftCurve.AddKey(90f, 1f);
            liftCurve.AddKey(180f, 0f);
            
            // Create default drag curve
            dragCurve = new AnimationCurve();
            dragCurve.AddKey(-180f, 1.5f);
            dragCurve.AddKey(-90f, 1.2f);
            dragCurve.AddKey(0f, CD0);
            dragCurve.AddKey(90f, 1.2f);
            dragCurve.AddKey(180f, 1.5f);
        }
        
        // Public methods for external control
        public void SetControlSurfaces(float elevator, float aileron, float rudder)
        {
            // Store for use in moment calculations
            // In a full implementation, these would affect the coefficients
        }
        
        public float GetLiftToDragRatio()
        {
            return DragMagnitude > 0 ? LiftMagnitude / DragMagnitude : 0;
        }
        
        void OnDrawGizmos()
        {
            if (!Application.isPlaying) return;
            
            // Draw force vectors
            Gizmos.color = Color.cyan;
            Gizmos.DrawRay(transform.position, currentLift * 0.001f);
            
            Gizmos.color = Color.red;
            Gizmos.DrawRay(transform.position, currentDrag * 0.001f);
            
            // Draw velocity vector
            if (rb != null)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawRay(transform.position, rb.velocity * 0.1f);
            }
        }
    }
}