using System.Collections;
using UnityEngine;

public class FlightManager : MonoBehaviour
{
    public ThrusterController thruster;
    AerodynamicsService aeroService;

    void Start()
    {
        aeroService = gameObject.AddComponent<AerodynamicsService>();
    }

    IEnumerator FixedUpdate()
    {
        float v = thruster.CurrentSpeed;
        yield return aeroService.GetDrag(v, thruster.Area, thruster.DragCoeff, drag => thruster.ApplyDrag(drag));
    }
}