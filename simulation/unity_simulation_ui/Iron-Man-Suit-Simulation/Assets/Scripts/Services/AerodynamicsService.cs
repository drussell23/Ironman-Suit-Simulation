using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class AerodynamicsService : MonoBehaviour
{
    const string URL = "http://localhost:3001/api/aero/drag";

    public IEnumerator GetDrag(float velocity, float area, float dragCoeff, System.Action<float> onResult)
    {
        var payload = JsonUtility.ToJson(new { velocity, area, dragCoeff });
        using var req = new UnityWebRequest(URL, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(payload);
        req.uploadHandler = new UploadHandlerRaw(bodyRaw);
        req.downloadHandler = new DownloadHandlerBuffer();
        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError(req.error);
        }
        else
        {
            var resp = JsonUtility.FromJson<DragResponse>(req.downloadHandler.text);
            onResult(resp.drag);
        }
    }

    [System.Serializable]
    class DragResponse { public float drag; }
}