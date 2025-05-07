using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform target;          // what to follow
    public Vector3 offset = new(0, 2, -5);
    public float smoothTime = 0.3f;

    private Vector3 veclocity = Vector3.zero;

    void LateUpdate()
    {
        if (target == null) 
            return;

        Vector3 desiredPos = target.position + offset;
        transform.position = Vector3.SmoothDamp(
            transform.position,
            desiredPos,
            ref veclocity,
            smoothTime
        );
        transform.LookAt(target);
    }
}
