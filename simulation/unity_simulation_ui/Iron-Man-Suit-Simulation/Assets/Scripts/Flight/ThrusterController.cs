using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class ThrusterController : MonoBehaviour
{
    [Header("Thrust Settings")]
    public float thrustForce = 30f;
    public float maxSpeed    = 20f;

    [Header("Aerodynamic Settings")]
    public float Area       = 0.5f;   // reference area (m²)
    public float DragCoeff = 0.03f;   // drag coefficient

    private Rigidbody rb;

    // Expose current speed so FlightManager can read it
    public float CurrentSpeed => rb.linearVelocity.magnitude;


    void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        float input = Input.GetAxis("Vertical");
        if (input > 0f)
        {
            // Apply thrust as acceleration (mass‐independent)
            rb.AddForce(transform.up * input * thrustForce, ForceMode.Acceleration);
        }

        // Clamp top speed
        if (rb.linearVelocity.magnitude > maxSpeed)
            rb.linearVelocity = rb.linearVelocity.normalized * maxSpeed;
    }

    /// <summary>
    /// Called by FlightManager to apply an external drag force.
    /// </summary>
    public void ApplyDrag(float drag)
    {
        // drag is a scalar magnitude; apply opposite to current velocity
        if (rb.linearVelocity.sqrMagnitude > 0.001f)
            rb.AddForce(-rb.linearVelocity.normalized * drag, ForceMode.Force);
    }
}
