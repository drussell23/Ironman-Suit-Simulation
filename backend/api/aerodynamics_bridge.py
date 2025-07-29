#!/usr/bin/env python3
"""
Aerodynamics Bridge API - Connects Unity aerodynamics components to Python backend.
Provides comprehensive data exchange for advanced flight simulation.
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import asyncio
import logging
from datetime import datetime

# Import backend aerodynamics modules
from backend.aerodynamics.flight_models.flight_dynamics import FlightDynamics
from backend.aerodynamics.environmental_effects.atmospheric_density import density_at_altitude
from backend.aerodynamics.environmental_effects.wind import calculate_wind_forces
from backend.aerodynamics.sensor_emulation.imu import IMUSensor
from backend.aerodynamics.sensor_emulation.pitot import PitotSensor
from backend.aerodynamics.control.flight_controllers import PIDController

# Try to import C++ physics plugin
try:
    from backend.aerodynamics.physics_plugin.python.bindings import (
        cfd_compute_forces,
        smagorinsky_model_step,
        kepsilon_model_step
    )
    PHYSICS_PLUGIN_AVAILABLE = True
except ImportError:
    PHYSICS_PLUGIN_AVAILABLE = False
    logging.warning("C++ physics plugin not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Iron Man Aerodynamics Bridge", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models for Unity integration
class Vector3(BaseModel):
    x: float
    y: float
    z: float

class AerodynamicState(BaseModel):
    """Complete aerodynamic state for Unity"""
    # Position and motion
    position: Vector3
    velocity: Vector3
    acceleration: Vector3
    rotation: Vector3  # Euler angles in degrees
    angular_velocity: Vector3
    
    # Aerodynamic properties
    angle_of_attack: float
    sideslip_angle: float
    dynamic_pressure: float
    mach_number: float
    reynolds_number: float
    
    # Forces and moments
    lift: Vector3
    drag: Vector3
    side_force: Vector3
    thrust: Vector3
    total_force: Vector3
    total_moment: Vector3
    
    # Atmospheric conditions
    altitude: float
    air_density: float
    air_pressure: float
    air_temperature: float
    speed_of_sound: float
    
    # Coefficients
    cl: float  # Lift coefficient
    cd: float  # Drag coefficient
    cy: float  # Side force coefficient
    
    timestamp: float

class WindState(BaseModel):
    """Wind and turbulence state"""
    steady_wind: Vector3
    gust_vector: Vector3
    turbulence_vector: Vector3
    turbulence_intensity: float
    wind_shear_gradient: float
    is_in_wake: bool

class SensorData(BaseModel):
    """Sensor readings for Unity"""
    # IMU data
    imu_acceleration: Vector3
    imu_angular_velocity: Vector3
    imu_magnetic_field: Vector3
    imu_temperature: float
    
    # Pitot data
    indicated_airspeed: float
    calibrated_airspeed: float
    true_airspeed: float
    pitot_pressure: float
    static_pressure: float
    
    # GPS data (simplified)
    gps_position: Vector3
    gps_velocity: Vector3
    gps_accuracy: float

class ControlInputs(BaseModel):
    """Control inputs from Unity"""
    # Direct controls
    thrust_command: float  # 0-1
    control_surfaces: Vector3  # elevator, aileron, rudder (-1 to 1)
    
    # Autopilot settings
    stability_assist_enabled: bool
    altitude_hold: Optional[float] = None
    velocity_hold: Optional[float] = None
    heading_hold: Optional[float] = None
    
    # Advanced controls
    flaps_position: float  # 0-1
    airbrake_position: float  # 0-1
    landing_gear_deployed: bool

class TurbulenceData(BaseModel):
    """Turbulence model data"""
    model_type: str  # "k-epsilon" or "smagorinsky"
    grid_resolution: int
    
    # k-epsilon specific
    turbulent_kinetic_energy: Optional[float] = None
    dissipation_rate: Optional[float] = None
    turbulent_viscosity: Optional[float] = None
    
    # LES specific
    subgrid_viscosity: Optional[float] = None
    resolved_tke: Optional[float] = None
    max_vorticity: Optional[float] = None
    
    # Grid data (simplified - full grid would be too large)
    sample_points: List[Dict[str, any]]  # Position and turbulence values

# Global simulation state
simulation_state = {
    "aerodynamic": None,
    "wind": None,
    "sensors": None,
    "control": None,
    "turbulence": None,
    "flight_dynamics": None,
    "imu_sensor": None,
    "pitot_sensor": None,
    "pid_controllers": {},
}

# Initialize components
def initialize_simulation():
    """Initialize all simulation components"""
    # Flight dynamics model
    simulation_state["flight_dynamics"] = FlightDynamics(
        mass=100.0,  # kg
        wing_area=2.0,  # m²
        Cl0=0.1,
        Cld_alpha=5.0,
        Cd0=0.02,
        k=0.05
    )
    
    # Sensors
    simulation_state["imu_sensor"] = IMUSensor()
    simulation_state["pitot_sensor"] = PitotSensor()
    
    # PID controllers
    simulation_state["pid_controllers"] = {
        "pitch": PIDController(kp=2.0, ki=0.1, kd=0.5),
        "roll": PIDController(kp=3.0, ki=0.1, kd=0.3),
        "yaw": PIDController(kp=1.0, ki=0.05, kd=0.2),
        "altitude": PIDController(kp=1.0, ki=0.02, kd=0.3),
    }
    
    # Initial state
    simulation_state["aerodynamic"] = AerodynamicState(
        position=Vector3(x=0, y=100, z=0),
        velocity=Vector3(x=0, y=0, z=0),
        acceleration=Vector3(x=0, y=0, z=0),
        rotation=Vector3(x=0, y=0, z=0),
        angular_velocity=Vector3(x=0, y=0, z=0),
        angle_of_attack=0,
        sideslip_angle=0,
        dynamic_pressure=0,
        mach_number=0,
        reynolds_number=0,
        lift=Vector3(x=0, y=0, z=0),
        drag=Vector3(x=0, y=0, z=0),
        side_force=Vector3(x=0, y=0, z=0),
        thrust=Vector3(x=0, y=0, z=0),
        total_force=Vector3(x=0, y=0, z=0),
        total_moment=Vector3(x=0, y=0, z=0),
        altitude=100,
        air_density=1.225,
        air_pressure=101325,
        air_temperature=288.15,
        speed_of_sound=340.29,
        cl=0,
        cd=0,
        cy=0,
        timestamp=0
    )

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize simulation on startup"""
    initialize_simulation()
    logger.info("Aerodynamics Bridge API started")

@app.get("/")
async def root():
    """API information"""
    return {
        "title": "Iron Man Aerodynamics Bridge",
        "version": "2.0.0",
        "physics_plugin": PHYSICS_PLUGIN_AVAILABLE,
        "endpoints": {
            "aerodynamics": "/api/aerodynamics/state",
            "wind": "/api/wind/state",
            "sensors": "/api/sensors/data",
            "control": "/api/control/input",
            "turbulence": "/api/turbulence/data",
        }
    }

@app.get("/api/aerodynamics/state", response_model=AerodynamicState)
async def get_aerodynamic_state():
    """Get complete aerodynamic state for Unity"""
    if simulation_state["aerodynamic"] is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    # Update calculations if physics plugin available
    if PHYSICS_PLUGIN_AVAILABLE and simulation_state["flight_dynamics"]:
        try:
            # Get current state
            state = simulation_state["aerodynamic"]
            
            # Prepare data for C++ plugin
            velocity_array = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
            position_array = np.array([state.position.x, state.position.y, state.position.z])
            
            # Call C++ CFD solver
            forces = cfd_compute_forces(
                velocity_array,
                state.angle_of_attack,
                state.air_density,
                state.air_temperature
            )
            
            # Update forces
            state.lift = Vector3(x=forces[0], y=forces[1], z=forces[2])
            state.drag = Vector3(x=forces[3], y=forces[4], z=forces[5])
            
        except Exception as e:
            logger.error(f"Physics plugin error: {e}")
    
    return simulation_state["aerodynamic"]

@app.post("/api/aerodynamics/state")
async def update_aerodynamic_state(state: AerodynamicState):
    """Update aerodynamic state from Unity calculations"""
    simulation_state["aerodynamic"] = state
    
    # Update sensor readings based on new state
    if simulation_state["imu_sensor"]:
        simulation_state["imu_sensor"].update(
            acceleration=[state.acceleration.x, state.acceleration.y, state.acceleration.z],
            angular_velocity=[state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z]
        )
    
    return {"status": "ok", "timestamp": state.timestamp}

@app.get("/api/wind/state", response_model=WindState)
async def get_wind_state():
    """Get wind and turbulence state"""
    if simulation_state["wind"] is None:
        # Generate default wind state
        simulation_state["wind"] = WindState(
            steady_wind=Vector3(x=5, y=0, z=0),
            gust_vector=Vector3(x=0, y=0, z=0),
            turbulence_vector=Vector3(x=0, y=0, z=0),
            turbulence_intensity=0.05,
            wind_shear_gradient=0.1,
            is_in_wake=False
        )
    
    return simulation_state["wind"]

@app.post("/api/wind/state")
async def update_wind_state(wind: WindState):
    """Update wind state from Unity"""
    simulation_state["wind"] = wind
    return {"status": "ok"}

@app.get("/api/sensors/data", response_model=SensorData)
async def get_sensor_data():
    """Get all sensor readings"""
    # Generate sensor data based on current state
    state = simulation_state["aerodynamic"]
    
    if simulation_state["sensors"] is None:
        simulation_state["sensors"] = SensorData(
            imu_acceleration=state.acceleration,
            imu_angular_velocity=state.angular_velocity,
            imu_magnetic_field=Vector3(x=0, y=0, z=50),  # μT
            imu_temperature=20.0,
            indicated_airspeed=0,
            calibrated_airspeed=0,
            true_airspeed=np.linalg.norm([state.velocity.x, state.velocity.y, state.velocity.z]),
            pitot_pressure=state.dynamic_pressure,
            static_pressure=state.air_pressure,
            gps_position=state.position,
            gps_velocity=state.velocity,
            gps_accuracy=1.0
        )
    
    return simulation_state["sensors"]

@app.post("/api/control/input")
async def update_control_inputs(control: ControlInputs):
    """Receive control inputs from Unity"""
    simulation_state["control"] = control
    
    # Apply control inputs to simulation
    if control.stability_assist_enabled:
        # Use PID controllers
        if control.altitude_hold is not None:
            pid = simulation_state["pid_controllers"]["altitude"]
            current_alt = simulation_state["aerodynamic"].altitude
            thrust_correction = pid.update(control.altitude_hold - current_alt)
            control.thrust_command = np.clip(control.thrust_command + thrust_correction, 0, 1)
    
    return {"status": "ok", "controls_applied": True}

@app.get("/api/turbulence/data", response_model=TurbulenceData)
async def get_turbulence_data():
    """Get turbulence model data"""
    if simulation_state["turbulence"] is None:
        # Default turbulence data
        simulation_state["turbulence"] = TurbulenceData(
            model_type="k-epsilon",
            grid_resolution=10,
            turbulent_kinetic_energy=0.1,
            dissipation_rate=0.01,
            turbulent_viscosity=1e-5,
            sample_points=[]
        )
    
    # If physics plugin available, compute turbulence
    if PHYSICS_PLUGIN_AVAILABLE:
        try:
            state = simulation_state["aerodynamic"]
            velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
            
            if simulation_state["turbulence"].model_type == "k-epsilon":
                k, epsilon, nut = kepsilon_model_step(
                    velocity,
                    simulation_state["turbulence"].grid_resolution,
                    0.02  # timestep
                )
                simulation_state["turbulence"].turbulent_kinetic_energy = float(np.mean(k))
                simulation_state["turbulence"].dissipation_rate = float(np.mean(epsilon))
                simulation_state["turbulence"].turbulent_viscosity = float(np.mean(nut))
                
        except Exception as e:
            logger.error(f"Turbulence calculation error: {e}")
    
    return simulation_state["turbulence"]

@app.post("/api/simulation/step")
async def simulation_step(dt: float = 0.02):
    """Advance simulation by one timestep"""
    if simulation_state["flight_dynamics"] and simulation_state["aerodynamic"]:
        # Get current state
        state = simulation_state["aerodynamic"]
        control = simulation_state["control"] or ControlInputs(
            thrust_command=0,
            control_surfaces=Vector3(x=0, y=0, z=0),
            stability_assist_enabled=False,
            flaps_position=0,
            airbrake_position=0,
            landing_gear_deployed=False
        )
        
        # Convert to numpy arrays
        position = np.array([state.position.x, state.position.y, state.position.z])
        velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
        
        # Calculate forces
        dynamics = simulation_state["flight_dynamics"]
        
        # Get atmospheric properties
        altitude = position[1]
        density = density_at_altitude(altitude)
        
        # Calculate aerodynamic forces
        speed = np.linalg.norm(velocity)
        if speed > 0.1:
            # Angle of attack (simplified)
            alpha = np.arctan2(-velocity[1], velocity[2]) if velocity[2] != 0 else 0
            
            # Get coefficients
            cl, cd = dynamics.aerodynamic_coeffs(alpha)
            
            # Dynamic pressure
            q = 0.5 * density * speed * speed
            
            # Forces
            lift_mag = q * dynamics.S * cl
            drag_mag = q * dynamics.S * cd
            
            # Force vectors (simplified)
            drag_dir = -velocity / speed
            lift_dir = np.cross(np.cross(drag_dir, np.array([0, 1, 0])), drag_dir)
            lift_dir = lift_dir / (np.linalg.norm(lift_dir) + 1e-6)
            
            lift_force = lift_dir * lift_mag
            drag_force = drag_dir * drag_mag
        else:
            lift_force = np.zeros(3)
            drag_force = np.zeros(3)
        
        # Thrust force (simplified - along forward direction)
        thrust_mag = control.thrust_command * 10000  # N
        thrust_force = np.array([0, 0, thrust_mag])
        
        # Total forces
        weight = np.array([0, -dynamics.mass * dynamics.g, 0])
        total_force = lift_force + drag_force + thrust_force + weight
        
        # Update acceleration
        acceleration = total_force / dynamics.mass
        
        # Integrate motion (simple Euler for now)
        velocity += acceleration * dt
        position += velocity * dt
        
        # Update state
        state.position = Vector3(x=position[0], y=position[1], z=position[2])
        state.velocity = Vector3(x=velocity[0], y=velocity[1], z=velocity[2])
        state.acceleration = Vector3(x=acceleration[0], y=acceleration[1], z=acceleration[2])
        state.altitude = position[1]
        state.air_density = density
        state.dynamic_pressure = q if speed > 0.1 else 0
        state.angle_of_attack = alpha if speed > 0.1 else 0
        state.cl = cl if speed > 0.1 else 0
        state.cd = cd if speed > 0.1 else 0
        state.lift = Vector3(x=lift_force[0], y=lift_force[1], z=lift_force[2])
        state.drag = Vector3(x=drag_force[0], y=drag_force[1], z=drag_force[2])
        state.thrust = Vector3(x=thrust_force[0], y=thrust_force[1], z=thrust_force[2])
        state.total_force = Vector3(x=total_force[0], y=total_force[1], z=total_force[2])
        state.timestamp = datetime.now().timestamp()
        
        simulation_state["aerodynamic"] = state
        
        return {"status": "ok", "time_advanced": dt}
    
    raise HTTPException(status_code=503, detail="Simulation not ready")

# WebSocket for real-time updates
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Stream telemetry data to Unity in real-time"""
    await websocket.accept()
    try:
        while True:
            # Send current state
            data = {
                "type": "telemetry_update",
                "aerodynamics": simulation_state["aerodynamic"].dict() if simulation_state["aerodynamic"] else None,
                "wind": simulation_state["wind"].dict() if simulation_state["wind"] else None,
                "sensors": simulation_state["sensors"].dict() if simulation_state["sensors"] else None,
                "turbulence": simulation_state["turbulence"].dict() if simulation_state["turbulence"] else None,
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.02)  # 50Hz update rate
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port from unity_bridge