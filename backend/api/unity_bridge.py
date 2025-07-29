#!/usr/bin/env python3
"""
Unity Bridge API - Connects Unity visualization to Python simulation backend.
Provides REST endpoints for real-time data exchange.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import asyncio
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Iron Man Unity Bridge", version="1.0.0")

# Enable CORS for Unity WebGL builds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class Vector3(BaseModel):
    x: float
    y: float
    z: float

class FlightState(BaseModel):
    position: List[float]  # [x, y, z]
    velocity: List[float]  # [vx, vy, vz]
    rotation: List[float]  # [roll, pitch, yaw] in degrees
    altitude: float
    speed: float
    thrust: float
    energy: float = 100.0
    health: float = 100.0
    timestamp: float

class ControlCommand(BaseModel):
    command: str
    value: float
    timestamp: Optional[float] = None

class ThrusterState(BaseModel):
    main: float = 0.0
    left_hand: float = 0.0
    right_hand: float = 0.0
    left_foot: float = 0.0
    right_foot: float = 0.0

# Global state (in production, use proper state management)
simulation_state = {
    "flight": FlightState(
        position=[0.0, 100.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 0.0],
        altitude=100.0,
        speed=0.0,
        thrust=0.0,
        energy=100.0,
        health=100.0,
        timestamp=time.time()
    ),
    "thrusters": ThrusterState(),
    "autopilot": False,
    "target_altitude": 150.0
}

# Background simulation task
async def physics_simulation():
    """Run physics simulation in background."""
    dt = 0.02  # 50 Hz
    gravity = 9.81
    mass = 100.0  # kg
    
    while True:
        try:
            state = simulation_state["flight"]
            thrusters = simulation_state["thrusters"]
            
            # Simple physics update
            pos = state.position.copy()
            vel = state.velocity.copy()
            
            # Calculate total thrust
            total_thrust = (
                thrusters.main + 
                thrusters.left_hand + thrusters.right_hand +
                thrusters.left_foot + thrusters.right_foot
            )
            
            # Apply forces
            force_y = total_thrust - (mass * gravity)
            acc_y = force_y / mass
            
            # Update velocity and position
            vel[1] += acc_y * dt
            pos[0] += vel[0] * dt
            pos[1] += vel[1] * dt
            pos[2] += vel[2] * dt
            
            # Apply drag
            drag = 0.1
            vel[0] *= (1 - drag * dt)
            vel[1] *= (1 - drag * dt)
            vel[2] *= (1 - drag * dt)
            
            # Update state
            simulation_state["flight"] = FlightState(
                position=pos,
                velocity=vel,
                rotation=state.rotation,
                altitude=pos[1],
                speed=float(np.linalg.norm(vel)),
                thrust=total_thrust,
                energy=max(0, state.energy - total_thrust * dt * 0.01),
                health=state.health,
                timestamp=time.time()
            )
            
            # Simple autopilot
            if simulation_state["autopilot"]:
                error = simulation_state["target_altitude"] - pos[1]
                thrust_cmd = max(0, min(2000, 1000 + error * 10))
                simulation_state["thrusters"].main = thrust_cmd
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        
        await asyncio.sleep(dt)

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup."""
    asyncio.create_task(physics_simulation())
    logger.info("Unity Bridge API started")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Iron Man Unity Bridge API",
        "status": "online",
        "endpoints": [
            "/api/flight/state",
            "/api/control",
            "/api/thrusters",
            "/api/autopilot"
        ]
    }

@app.get("/api/flight/state", response_model=FlightState)
async def get_flight_state():
    """Get current flight state for Unity."""
    return simulation_state["flight"]

@app.post("/api/control")
async def send_control_command(command: ControlCommand):
    """Receive control commands from Unity."""
    logger.info(f"Control command: {command.command} = {command.value}")
    
    if command.command == "thrust":
        simulation_state["thrusters"].main = command.value
    elif command.command == "thrust_left":
        simulation_state["thrusters"].left_hand = command.value
    elif command.command == "thrust_right":
        simulation_state["thrusters"].right_hand = command.value
    elif command.command == "roll":
        simulation_state["flight"].rotation[0] = command.value
    elif command.command == "pitch":
        simulation_state["flight"].rotation[1] = command.value
    elif command.command == "yaw":
        simulation_state["flight"].rotation[2] = command.value
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {command.command}")
    
    return {"status": "ok", "command": command.command, "value": command.value}

@app.get("/api/thrusters", response_model=ThrusterState)
async def get_thruster_state():
    """Get current thruster states."""
    return simulation_state["thrusters"]

@app.post("/api/thrusters")
async def set_thruster_state(thrusters: ThrusterState):
    """Set all thruster values at once."""
    simulation_state["thrusters"] = thrusters
    return {"status": "ok", "thrusters": thrusters}

@app.post("/api/autopilot/{enabled}")
async def set_autopilot(enabled: bool, target_altitude: Optional[float] = None):
    """Enable/disable autopilot."""
    simulation_state["autopilot"] = enabled
    if target_altitude is not None:
        simulation_state["target_altitude"] = target_altitude
    
    return {
        "status": "ok",
        "autopilot": enabled,
        "target_altitude": simulation_state["target_altitude"]
    }

@app.get("/api/telemetry")
async def get_telemetry():
    """Get detailed telemetry data."""
    state = simulation_state["flight"]
    return {
        "position": {"x": state.position[0], "y": state.position[1], "z": state.position[2]},
        "velocity": {"x": state.velocity[0], "y": state.velocity[1], "z": state.velocity[2]},
        "rotation": {"roll": state.rotation[0], "pitch": state.rotation[1], "yaw": state.rotation[2]},
        "altitude": state.altitude,
        "speed": state.speed,
        "vertical_speed": state.velocity[1],
        "thrust": {
            "total": state.thrust,
            "main": simulation_state["thrusters"].main,
            "left_hand": simulation_state["thrusters"].left_hand,
            "right_hand": simulation_state["thrusters"].right_hand,
            "left_foot": simulation_state["thrusters"].left_foot,
            "right_foot": simulation_state["thrusters"].right_foot
        },
        "energy": state.energy,
        "health": state.health,
        "autopilot": {
            "enabled": simulation_state["autopilot"],
            "target_altitude": simulation_state["target_altitude"]
        },
        "timestamp": state.timestamp
    }

@app.post("/api/reset")
async def reset_simulation():
    """Reset simulation to initial state."""
    simulation_state["flight"] = FlightState(
        position=[0.0, 100.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 0.0],
        altitude=100.0,
        speed=0.0,
        thrust=0.0,
        energy=100.0,
        health=100.0,
        timestamp=time.time()
    )
    simulation_state["thrusters"] = ThrusterState()
    simulation_state["autopilot"] = False
    
    return {"status": "ok", "message": "Simulation reset"}

# WebSocket support for real-time updates (optional)
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time telemetry streaming."""
    await websocket.accept()
    try:
        while True:
            # Send telemetry every 100ms
            await websocket.send_json({
                "type": "telemetry",
                "data": await get_telemetry()
            })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info("Unity client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)