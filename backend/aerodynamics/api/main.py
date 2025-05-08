# main.py
from flask import Flask, request, jsonify
from flight_models.thrust_and_propulsion import compute_thrust
app = Flask(__name__)

@app.post('/api/thrust')
def thrust_endpoint():
    data = request.json
    t = compute_thrust(data['mass'], data['accel'])
    return jsonify({'thrust': t})
