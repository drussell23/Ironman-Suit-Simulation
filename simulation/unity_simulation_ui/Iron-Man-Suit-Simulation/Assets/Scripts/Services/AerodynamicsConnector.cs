using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System.Text;
using System;

namespace IronManSim.Services
{
    /// <summary>
    /// Enhanced connector that integrates Unity aerodynamics components with Python backend.
    /// Synchronizes all physics calculations between Unity and backend simulation.
    /// </summary>
    public class AerodynamicsConnector : MonoBehaviour
    {
        [Header("Connection Settings")]
        [SerializeField] private string backendUrl = "http://localhost:8001";
        [SerializeField] private float syncInterval = 0.02f; // 50Hz
        [SerializeField] private bool useBidirectionalSync = true;
        [SerializeField] private bool preferBackendPhysics = false;
        
        [Header("Sync Options")]
        [SerializeField] private bool syncAerodynamics = true;
        [SerializeField] private bool syncWind = true;
        [SerializeField] private bool syncSensors = true;
        [SerializeField] private bool syncTurbulence = true;
        [SerializeField] private bool syncControl = true;
        
        [Header("Components")]
        [SerializeField] private AerodynamicForces aeroForces;
        [SerializeField] private AtmosphericDensity atmosphere;
        [SerializeField] private StabilityControl stability;
        [SerializeField] private WindInteraction wind;
        [SerializeField] private SensorEmulation.IMUEmulator imu;
        [SerializeField] private SensorEmulation.PitotEmulator pitot;
        [SerializeField] private Turbulence.KEpsilonModel kEpsilon;
        [SerializeField] private Turbulence.SmagorinskyModel smagorinsky;
        
        [Header("Status")]
        [SerializeField] private bool isConnected;
        [SerializeField] private float lastSyncTime;
        [SerializeField] private int syncCount;
        [SerializeField] private float latency;
        
        // Data structures matching backend
        [Serializable]
        public class AerodynamicState
        {
            public Vector3Data position;
            public Vector3Data velocity;
            public Vector3Data acceleration;
            public Vector3Data rotation;
            public Vector3Data angular_velocity;
            
            public float angle_of_attack;
            public float sideslip_angle;
            public float dynamic_pressure;
            public float mach_number;
            public float reynolds_number;
            
            public Vector3Data lift;
            public Vector3Data drag;
            public Vector3Data side_force;
            public Vector3Data thrust;
            public Vector3Data total_force;
            public Vector3Data total_moment;
            
            public float altitude;
            public float air_density;
            public float air_pressure;
            public float air_temperature;
            public float speed_of_sound;
            
            public float cl;
            public float cd;
            public float cy;
            
            public float timestamp;
        }
        
        [Serializable]
        public class Vector3Data
        {
            public float x, y, z;
            
            public Vector3Data(Vector3 v)
            {
                x = v.x; y = v.y; z = v.z;
            }
            
            public Vector3 ToVector3() => new Vector3(x, y, z);
        }
        
        [Serializable]
        public class WindState
        {
            public Vector3Data steady_wind;
            public Vector3Data gust_vector;
            public Vector3Data turbulence_vector;
            public float turbulence_intensity;
            public float wind_shear_gradient;
            public bool is_in_wake;
        }
        
        [Serializable]
        public class SensorData
        {
            public Vector3Data imu_acceleration;
            public Vector3Data imu_angular_velocity;
            public Vector3Data imu_magnetic_field;
            public float imu_temperature;
            
            public float indicated_airspeed;
            public float calibrated_airspeed;
            public float true_airspeed;
            public float pitot_pressure;
            public float static_pressure;
            
            public Vector3Data gps_position;
            public Vector3Data gps_velocity;
            public float gps_accuracy;
        }
        
        [Serializable]
        public class ControlInputs
        {
            public float thrust_command;
            public Vector3Data control_surfaces;
            
            public bool stability_assist_enabled;
            public float? altitude_hold;
            public float? velocity_hold;
            public float? heading_hold;
            
            public float flaps_position;
            public float airbrake_position;
            public bool landing_gear_deployed;
        }
        
        private Rigidbody rb;
        private Coroutine syncCoroutine;
        
        void Start()
        {
            rb = GetComponent<Rigidbody>();
            
            // Auto-find components if not assigned
            if (!aeroForces) aeroForces = GetComponent<AerodynamicForces>();
            if (!atmosphere) atmosphere = GetComponent<AtmosphericDensity>();
            if (!stability) stability = GetComponent<StabilityControl>();
            if (!wind) wind = GetComponent<WindInteraction>();
            if (!imu) imu = GetComponent<SensorEmulation.IMUEmulator>();
            if (!pitot) pitot = GetComponent<SensorEmulation.PitotEmulator>();
            if (!kEpsilon) kEpsilon = GetComponent<Turbulence.KEpsilonModel>();
            if (!smagorinsky) smagorinsky = GetComponent<Turbulence.SmagorinskyModel>();
            
            // Start synchronization
            syncCoroutine = StartCoroutine(SynchronizationLoop());
        }
        
        IEnumerator SynchronizationLoop()
        {
            while (true)
            {
                float startTime = Time.realtimeSinceStartup;
                
                // Sync all enabled systems
                if (syncAerodynamics) yield return StartCoroutine(SyncAerodynamics());
                if (syncWind) yield return StartCoroutine(SyncWind());
                if (syncSensors) yield return StartCoroutine(SyncSensors());
                if (syncTurbulence) yield return StartCoroutine(SyncTurbulence());
                if (syncControl) yield return StartCoroutine(SyncControl());
                
                // Calculate latency
                latency = Time.realtimeSinceStartup - startTime;
                lastSyncTime = Time.time;
                syncCount++;
                
                yield return new WaitForSeconds(syncInterval);
            }
        }
        
        IEnumerator SyncAerodynamics()
        {
            if (useBidirectionalSync && !preferBackendPhysics)
            {
                // Send Unity calculations to backend
                var state = CreateAerodynamicState();
                yield return StartCoroutine(PostData("/api/aerodynamics/state", state));
            }
            else
            {
                // Get state from backend
                yield return StartCoroutine(GetData<AerodynamicState>("/api/aerodynamics/state", 
                    (state) => ApplyAerodynamicState(state)));
            }
        }
        
        AerodynamicState CreateAerodynamicState()
        {
            var state = new AerodynamicState
            {
                position = new Vector3Data(transform.position),
                velocity = new Vector3Data(rb.linearVelocity),
                acceleration = new Vector3Data(rb.linearVelocity / Time.fixedDeltaTime), // Simplified
                rotation = new Vector3Data(transform.eulerAngles),
                angular_velocity = new Vector3Data(rb.angularVelocity * Mathf.Rad2Deg),
                
                altitude = transform.position.y,
                timestamp = Time.time
            };
            
            // Get data from components
            if (aeroForces)
            {
                state.angle_of_attack = aeroForces.AngleOfAttack;
                state.dynamic_pressure = aeroForces.DynamicPressure;
                state.lift = new Vector3Data(aeroForces.Lift);
                state.drag = new Vector3Data(aeroForces.Drag);
            }
            
            if (atmosphere)
            {
                state.air_density = atmosphere.Density;
                state.air_pressure = atmosphere.Pressure;
                state.air_temperature = atmosphere.Temperature;
                state.speed_of_sound = atmosphere.SpeedOfSound;
                state.mach_number = atmosphere.MachNumber;
            }
            
            return state;
        }
        
        void ApplyAerodynamicState(AerodynamicState state)
        {
            if (state == null) return;
            
            // Apply position and velocity from backend
            if (preferBackendPhysics)
            {
                transform.position = state.position.ToVector3();
                rb.linearVelocity = state.velocity.ToVector3();
                transform.eulerAngles = state.rotation.ToVector3();
                rb.angularVelocity = state.angular_velocity.ToVector3() * Mathf.Deg2Rad;
            }
            
            // Update component data
            isConnected = true;
        }
        
        IEnumerator SyncWind()
        {
            if (wind == null) yield break;
            
            if (useBidirectionalSync)
            {
                // Send Unity wind data to backend
                var windState = new WindState
                {
                    steady_wind = new Vector3Data(wind.CurrentWind),
                    turbulence_intensity = wind.InTurbulence ? 0.1f : 0.01f,
                    wind_shear_gradient = 0.1f,
                    is_in_wake = false
                };
                
                yield return StartCoroutine(PostData("/api/wind/state", windState));
            }
            else
            {
                // Get wind from backend
                yield return StartCoroutine(GetData<WindState>("/api/wind/state",
                    (state) => {
                        if (state != null && wind != null)
                        {
                            wind.SetBaseWind(state.steady_wind.ToVector3(), state.steady_wind.ToVector3().magnitude);
                        }
                    }));
            }
        }
        
        IEnumerator SyncSensors()
        {
            if (!imu && !pitot) yield break;
            
            // Always get sensor data from backend (backend is authoritative for sensors)
            yield return StartCoroutine(GetData<SensorData>("/api/sensors/data",
                (data) => {
                    if (data == null) return;
                    
                    // Update IMU display (if you have UI for it)
                    // The actual IMU component generates its own noisy data
                    
                    // Could use this to validate sensor readings
                    if (pitot && Mathf.Abs(pitot.TrueAirspeed - data.true_airspeed) > 10f)
                    {
                        Debug.LogWarning($"Pitot mismatch: Unity={pitot.TrueAirspeed}, Backend={data.true_airspeed}");
                    }
                }));
        }
        
        IEnumerator SyncTurbulence()
        {
            // Turbulence models are computationally expensive, so we only sync key metrics
            if (!kEpsilon && !smagorinsky) yield break;
            
            yield return StartCoroutine(GetData<TurbulenceData>("/api/turbulence/data",
                (data) => {
                    if (data == null) return;
                    
                    // Could use backend turbulence data to adjust Unity models
                    // For now, just log differences for validation
                    if (kEpsilon && data.model_type == "k-epsilon")
                    {
                        float unityTKE = kEpsilon.AverageTurbulentKineticEnergy;
                        float backendTKE = data.turbulent_kinetic_energy ?? 0;
                        
                        if (Mathf.Abs(unityTKE - backendTKE) > 0.1f)
                        {
                            Debug.Log($"TKE: Unity={unityTKE:F3}, Backend={backendTKE:F3}");
                        }
                    }
                }));
        }
        
        IEnumerator SyncControl()
        {
            if (!stability) yield break;
            
            // Send control inputs to backend
            var controls = new ControlInputs
            {
                thrust_command = Input.GetAxis("Thrust"),
                control_surfaces = new Vector3Data(new Vector3(
                    Input.GetAxis("Pitch"),
                    Input.GetAxis("Yaw"),
                    Input.GetAxis("Roll")
                )),
                stability_assist_enabled = stability.IsStabilityEnabled,
                flaps_position = 0, // Add UI control for this
                airbrake_position = 0,
                landing_gear_deployed = false
            };
            
            yield return StartCoroutine(PostData("/api/control/input", controls));
        }
        
        // Data structures for turbulence
        [Serializable]
        public class TurbulenceData
        {
            public string model_type;
            public int grid_resolution;
            public float? turbulent_kinetic_energy;
            public float? dissipation_rate;
            public float? turbulent_viscosity;
            public float? subgrid_viscosity;
            public float? resolved_tke;
            public float? max_vorticity;
        }
        
        // HTTP helpers
        IEnumerator GetData<T>(string endpoint, System.Action<T> callback)
        {
            using (UnityWebRequest request = UnityWebRequest.Get(backendUrl + endpoint))
            {
                yield return request.SendWebRequest();
                
                if (request.result == UnityWebRequest.Result.Success)
                {
                    try
                    {
                        T data = JsonUtility.FromJson<T>(request.downloadHandler.text);
                        callback(data);
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"JSON parse error: {e.Message}");
                    }
                }
                else
                {
                    Debug.LogWarning($"GET {endpoint} failed: {request.error}");
                    isConnected = false;
                }
            }
        }
        
        IEnumerator PostData<T>(string endpoint, T data)
        {
            string json = JsonUtility.ToJson(data);
            
            using (UnityWebRequest request = new UnityWebRequest(backendUrl + endpoint, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                
                yield return request.SendWebRequest();
                
                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError($"POST {endpoint} failed: {request.error}");
                    isConnected = false;
                }
            }
        }
        
        // Public methods
        public void SetPhysicsAuthority(bool useBackend)
        {
            preferBackendPhysics = useBackend;
        }
        
        public void RequestSimulationStep()
        {
            StartCoroutine(PostData("/api/simulation/step", new { dt = Time.fixedDeltaTime }));
        }
        
        void OnDestroy()
        {
            if (syncCoroutine != null)
            {
                StopCoroutine(syncCoroutine);
            }
        }
        
        void OnDrawGizmos()
        {
            if (!isConnected) return;
            
            // Draw connection status
            Gizmos.color = isConnected ? Color.green : Color.red;
            Gizmos.DrawWireSphere(transform.position + Vector3.up * 3f, 0.5f);
        }
    }
}