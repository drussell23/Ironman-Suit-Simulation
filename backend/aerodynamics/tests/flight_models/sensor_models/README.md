# Sensor Models Tests

This directory contains unit tests for the `sensor_models` package (IMU and Pitot) under `backend/aerodynamics/flight_models/sensor_models`.

## Files

- `test_imu.py`  
  Validates IMUSensor measurement methods (accelerometer & gyroscope), covering noise, bias, scale, misalignment, and error handling.

- `test_pitot.py`  
  Validates PitotSensor measurement methods (static, total, dynamic pressure), IAS/TAS computations, compressibility, and invalid input errors.

## Prerequisites

- Python 3.8+ with `numpy` and `pytest` installed
- Ensure the project root is on `PYTHONPATH`:
  ```bash
  export PYTHONPATH="$(pwd)/backend:$PYTHONPATH"
  ```

## Running Tests

From the project root, run:
```bash
pytest backend/aerodynamics/tests/flight_models/sensor_models
```

To run a single test file:
```bash
pytest backend/aerodynamics/tests/flight_models/sensor_models/test_imu.py
```  
or
```bash
pytest backend/aerodynamics/tests/flight_models/sensor_models/test_pitot.py
```

## Guidelines

- All tests use `pytest` and rely on deterministic behavior by disabling noise or setting known seeds.
- Review test failures to identify issues in sensor noise, bias, or measurement logic.
- Add new tests when adding features or edge cases to `IMUSensor` or `PitotSensor`.
- Maintain test coverage for boundary conditions (e.g., zero velocity, negative inputs).

---
Happy testing! 🎯
