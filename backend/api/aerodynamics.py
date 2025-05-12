# backend/api/aerodynamics.py

import numpy as np
from flask import Blueprint, request, jsonify

# Ensure the project root is on PYTHONPATH so these resolve:
from backend.aerodynamics.flight_models.thrust_and_propulsion import compute_thrust
from backend.aerodynamics.flight_models.flight_dynamics import FlightDynamics

# Blueprint for all aerodynamics endpoints.
aero_bp = Blueprint("aerodynamics", __name__, url_prefix="/api")

# Default suit parameters if none are provided
DEFAULT_PARAMS = {
    "mass": 80.0,  # kg
    "wing_area": 0.5,  # m²
    "Cl0": 0.2,  # zero-AoA lift coefficient
    "Cld_alpha": 5.7,  # lift slope per rad
    "Cd0": 0.03,  # zero-lift drag coefficient
    "k": 0.1,  # induced drag factor
}


@aero_bp.route("/thrust", methods=["POST"])
def thrust_endpoint():
    """
    POST /api/thrust
    Request JSON: { "mass": float, "accel": float }
    Response JSON: { "thrust": float }
    """
    data = request.get_json(force=True)
    mass = data.get("mass", DEFAULT_PARAMS["mass"])
    accel = data.get("accel", 0.0)
    thrust = compute_thrust(mass, accel)

    return jsonify({"thrust": thrust})


aero_bp.route("/drag", methods=["POST"])
def drag_endpoint():
    """
    POST /api/drag
    Request JSON: {
      "velocity": [vx, vy, vz],
      "alpha": float,
      "altitude": float,
      // optionally override defaults:
      "mass", "wing_area", "Cl0", "Cld_alpha", "Cd0", "k"
    }
    Response JSON: { "lift": [Lx,Ly,Lz], "drag": [Dx,Dy,Dz] }
    """
    data = request.get_json(force=True)
    
    # Unpack inputs.
    velocity = np.array(data.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    alpha = float(data.get("alpha", 0.0))
    altitude = float(data.get("altitude", 0.0))
    
    # Override defaults if provided.
    params = {
        key: float(data.get(key, DEFAULT_PARAMS[key]))
        for key in DEFAULT_PARAMS
    }
    
    # Instantiate flight_dynamics with those parameters.
    fd = FlightDynamics(
        mass      = params["mass"],
        wing_area = params["wing_area"],
        Cl0       = params["Cl0"],
        Cld_alpha = params["Cld_alpha"],
        Cd0       = params["Cd0"],
        k         = params["k"]
    )
    
    lift, drag = fd.aerodynamics_forces(velocity, alpha, altitude)
    
    return jsonify({
        "lift": lift.tolist(),
        "drag": drag.tolist()
    })