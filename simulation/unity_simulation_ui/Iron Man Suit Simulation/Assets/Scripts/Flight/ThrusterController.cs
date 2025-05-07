using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class ThrusterController : MonoBehaviour
{
    [Tooltip("Upward acceleration in m/s² applied per input")]
    public float thrustForce = 30f;  
    [Tooltip("Maximum total speed in m/s")]
    public float maxSpeed    = 20f;  

    Rigidbody rb;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        // Read vertical input (W/S or Up/Down arrows)
        float input = Input.GetAxis("Vertical");
        if (input > 0f)
        {
            // Apply thrust upward relative to the capsule's local up
            Vector3 force = transform.up * input * thrustForce;
            rb.AddForce(force, ForceMode.Acceleration);
        }

        // Clamp maximum speed using the new linearVelocity API
        Vector3 lv = rb.linearVelocity;
        if (lv.magnitude > maxSpeed)
            rb.linearVelocity = lv.normalized * maxSpeed;
    }
}
