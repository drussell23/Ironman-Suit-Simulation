using UnityEngine;
using UnityEngine.InputSystem;
using System;

namespace IronManSim.Suit
{
    /// <summary>
    /// Main controller for the Iron Man suit simulation
    /// Handles movement, flight, repulsors, and suit state management
    /// </summary>
    public class IronManSuitController : MonoBehaviour
    {
        [Header("Movement Settings")]
        [SerializeField] private float walkSpeed = 5f;
        [SerializeField] private float runSpeed = 10f;
        [SerializeField] private float flightSpeed = 50f;
        [SerializeField] private float maxFlightSpeed = 200f;
        [SerializeField] private float rotationSpeed = 180f;
        
        [Header("Flight Physics")]
        [SerializeField] private float thrustPower = 100f;
        [SerializeField] private float hoverHeight = 2f;
        [SerializeField] private float stabilizationForce = 10f;
        [SerializeField] private AnimationCurve thrustCurve = AnimationCurve.Linear(0, 0, 1, 1);
        
        [Header("Repulsor Settings")]
        [SerializeField] private Transform leftHandRepulsor;
        [SerializeField] private Transform rightHandRepulsor;
        [SerializeField] private Transform chestRepulsor;
        [SerializeField] private float repulsorForce = 500f;
        [SerializeField] private float repulsorRange = 50f;
        
        [Header("Suit Components")]
        [SerializeField] private GameObject helmetHUD;
        [SerializeField] private Light[] suitLights;
        [SerializeField] private ParticleSystem[] thrusterEffects;
        [SerializeField] private AudioSource thrusterAudio;
        
        // State management
        private bool isFlying = false;
        private bool isHovering = false;
        private bool suitPoweredOn = false;
        private float currentPower = 100f;
        private float maxPower = 100f;
        
        // Physics components
        private Rigidbody rb;
        private CapsuleCollider capsuleCollider;
        
        // Input handling
        private Vector2 moveInput;
        private Vector2 lookInput;
        private float verticalInput;
        private bool thrustInput;
        
        // Flight control
        private float currentThrust = 0f;
        private Vector3 flightVelocity = Vector3.zero;
        private Quaternion targetRotation;
        
        void Awake()
        {
            rb = GetComponent<Rigidbody>();
            if (rb == null)
            {
                rb = gameObject.AddComponent<Rigidbody>();
            }
            
            capsuleCollider = GetComponent<CapsuleCollider>();
            if (capsuleCollider == null)
            {
                capsuleCollider = gameObject.AddComponent<CapsuleCollider>();
                capsuleCollider.height = 2f;
                capsuleCollider.radius = 0.5f;
            }
            
            // Configure rigidbody for suit physics
            rb.mass = 150f; // Suit + human weight
            rb.drag = 0.5f;
            rb.angularDrag = 2f;
            rb.useGravity = true;
            rb.constraints = RigidbodyConstraints.FreezeRotation;
        }
        
        void Start()
        {
            targetRotation = transform.rotation;
            PowerOnSuit();
        }
        
        void Update()
        {
            if (!suitPoweredOn) return;
            
            HandleRotation();
            UpdateHUD();
            UpdatePowerConsumption();
            
            // Check for state transitions
            if (Input.GetKeyDown(KeyCode.Space) && !isFlying)
            {
                StartFlying();
            }
            else if (Input.GetKeyDown(KeyCode.LeftControl) && isFlying)
            {
                Land();
            }
        }
        
        void FixedUpdate()
        {
            if (!suitPoweredOn) return;
            
            if (isFlying)
            {
                HandleFlightPhysics();
            }
            else
            {
                HandleGroundMovement();
            }
            
            StabilizeSuit();
        }
        
        #region Movement & Flight
        
        private void HandleGroundMovement()
        {
            Vector3 movement = new Vector3(moveInput.x, 0, moveInput.y);
            movement = transform.TransformDirection(movement);
            
            float speed = Input.GetKey(KeyCode.LeftShift) ? runSpeed : walkSpeed;
            rb.MovePosition(rb.position + movement * speed * Time.fixedDeltaTime);
        }
        
        private void HandleFlightPhysics()
        {
            // Calculate thrust based on input
            if (thrustInput)
            {
                currentThrust = Mathf.Lerp(currentThrust, 1f, Time.fixedDeltaTime * 2f);
            }
            else
            {
                currentThrust = Mathf.Lerp(currentThrust, 0.3f, Time.fixedDeltaTime * 2f); // Hover thrust
            }
            
            // Apply thrust force
            float thrustMultiplier = thrustCurve.Evaluate(currentThrust);
            Vector3 thrustForce = transform.up * thrustPower * thrustMultiplier;
            
            // Add directional movement
            Vector3 moveDirection = new Vector3(moveInput.x, verticalInput, moveInput.y);
            moveDirection = transform.TransformDirection(moveDirection);
            
            rb.AddForce(thrustForce + moveDirection * flightSpeed, ForceMode.Force);
            
            // Limit velocity
            if (rb.velocity.magnitude > maxFlightSpeed)
            {
                rb.velocity = rb.velocity.normalized * maxFlightSpeed;
            }
            
            // Update thruster effects
            UpdateThrusterEffects(currentThrust);
        }
        
        private void StabilizeSuit()
        {
            if (!isFlying) return;
            
            // Auto-stabilization when hovering
            if (isHovering && moveInput.magnitude < 0.1f)
            {
                // Stabilize rotation
                Quaternion uprightRotation = Quaternion.Euler(0, transform.eulerAngles.y, 0);
                transform.rotation = Quaternion.Slerp(transform.rotation, uprightRotation, 
                    Time.fixedDeltaTime * stabilizationForce);
                
                // Maintain hover height
                RaycastHit hit;
                if (Physics.Raycast(transform.position, Vector3.down, out hit, hoverHeight * 2f))
                {
                    float heightError = hoverHeight - hit.distance;
                    rb.AddForce(Vector3.up * heightError * stabilizationForce, ForceMode.Force);
                }
            }
        }
        
        private void HandleRotation()
        {
            // Mouse/stick look
            float yaw = lookInput.x * rotationSpeed * Time.deltaTime;
            float pitch = -lookInput.y * rotationSpeed * Time.deltaTime;
            
            if (isFlying)
            {
                // Free rotation in flight
                transform.Rotate(pitch, yaw, 0, Space.Self);
            }
            else
            {
                // Yaw only on ground
                transform.Rotate(0, yaw, 0, Space.World);
            }
        }
        
        #endregion
        
        #region Suit Systems
        
        private void PowerOnSuit()
        {
            suitPoweredOn = true;
            
            // Enable HUD
            if (helmetHUD != null)
                helmetHUD.SetActive(true);
            
            // Enable suit lights
            foreach (var light in suitLights)
            {
                if (light != null)
                    light.enabled = true;
            }
            
            // Play power-on sound
            // TODO: Add power-on audio
        }
        
        private void StartFlying()
        {
            isFlying = true;
            rb.useGravity = false;
            rb.constraints = RigidbodyConstraints.None;
            
            // Initial upward boost
            rb.AddForce(Vector3.up * thrustPower * 2f, ForceMode.Impulse);
            
            // Enable thruster effects
            foreach (var effect in thrusterEffects)
            {
                if (effect != null)
                    effect.Play();
            }
            
            if (thrusterAudio != null)
                thrusterAudio.Play();
        }
        
        private void Land()
        {
            isFlying = false;
            rb.useGravity = true;
            rb.constraints = RigidbodyConstraints.FreezeRotation;
            currentThrust = 0f;
            
            // Disable thruster effects
            foreach (var effect in thrusterEffects)
            {
                if (effect != null)
                    effect.Stop();
            }
            
            if (thrusterAudio != null)
                thrusterAudio.Stop();
        }
        
        private void UpdateThrusterEffects(float intensity)
        {
            foreach (var effect in thrusterEffects)
            {
                if (effect != null)
                {
                    var emission = effect.emission;
                    emission.rateOverTime = Mathf.Lerp(10f, 100f, intensity);
                    
                    var main = effect.main;
                    main.startSpeed = Mathf.Lerp(5f, 20f, intensity);
                }
            }
            
            if (thrusterAudio != null)
            {
                thrusterAudio.volume = Mathf.Lerp(0.3f, 1f, intensity);
                thrusterAudio.pitch = Mathf.Lerp(0.8f, 1.2f, intensity);
            }
        }
        
        private void UpdatePowerConsumption()
        {
            float consumption = 0.1f; // Base consumption
            
            if (isFlying)
                consumption += 0.5f;
            
            if (thrustInput)
                consumption += 1f;
            
            currentPower = Mathf.Max(0, currentPower - consumption * Time.deltaTime);
            
            if (currentPower <= 0)
            {
                // Emergency landing
                if (isFlying)
                    Land();
                
                suitPoweredOn = false;
            }
        }
        
        private void UpdateHUD()
        {
            // TODO: Update HUD elements with suit status
            // This would interface with a separate HUD controller
        }
        
        #endregion
        
        #region Weapons
        
        public void FireRepulsor(bool leftHand)
        {
            Transform repulsor = leftHand ? leftHandRepulsor : rightHandRepulsor;
            if (repulsor == null) return;
            
            // Create repulsor beam effect
            RaycastHit hit;
            Vector3 direction = repulsor.forward;
            
            if (Physics.Raycast(repulsor.position, direction, out hit, repulsorRange))
            {
                // Apply force to hit object
                Rigidbody targetRb = hit.collider.GetComponent<Rigidbody>();
                if (targetRb != null)
                {
                    targetRb.AddForceAtPosition(direction * repulsorForce, hit.point, ForceMode.Impulse);
                }
                
                // TODO: Add visual effect from repulsor to hit point
            }
            
            // Consume power
            currentPower = Mathf.Max(0, currentPower - 2f);
        }
        
        public void FireUnibeam()
        {
            if (chestRepulsor == null || currentPower < 20f) return;
            
            // TODO: Implement powerful chest beam
            currentPower -= 20f;
        }
        
        #endregion
        
        #region Input Handling
        
        public void OnMove(InputValue value)
        {
            moveInput = value.Get<Vector2>();
        }
        
        public void OnLook(InputValue value)
        {
            lookInput = value.Get<Vector2>();
        }
        
        public void OnThrust(InputValue value)
        {
            thrustInput = value.isPressed;
        }
        
        public void OnVertical(InputValue value)
        {
            verticalInput = value.Get<float>();
        }
        
        public void OnFireLeft(InputValue value)
        {
            if (value.isPressed)
                FireRepulsor(true);
        }
        
        public void OnFireRight(InputValue value)
        {
            if (value.isPressed)
                FireRepulsor(false);
        }
        
        #endregion
        
        #region Public Methods
        
        public float GetPowerLevel()
        {
            return currentPower / maxPower;
        }
        
        public bool IsFlying()
        {
            return isFlying;
        }
        
        public Vector3 GetVelocity()
        {
            return rb.velocity;
        }
        
        public float GetAltitude()
        {
            RaycastHit hit;
            if (Physics.Raycast(transform.position, Vector3.down, out hit, 1000f))
            {
                return hit.distance;
            }
            return transform.position.y;
        }
        
        #endregion
    }
}