using UnityEngine;
using TMPro;                // for TextMeshProUGUI
// using UnityEngine.UI;    // uncomment if you end up using UI.Text instead

public class TelemetryUI : MonoBehaviour 
{
    [Tooltip("Suit's Rigidbody for velocity")]
    public Rigidbody rb;
    [Tooltip("Text element for speed display")]
    public TextMeshProUGUI speedText;
    [Tooltip("Text element for altitude display")]
    public TextMeshProUGUI altitudeText;

    void Update()
    {
        // Use the new linearVelocity property
        Vector3 linVel = rb.linearVelocity;  
        float speed = linVel.magnitude;
        speedText.text = $"Speed: {speed:0.0} m/s";

        // Altitude stays the same
        float altitude = transform.position.y;
        altitudeText.text = $"Altitude: {altitude:0.0} m";
    }
}
