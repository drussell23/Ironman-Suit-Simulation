# Sensor Models

The `sensor_models` package provides realistic sensor simulations for the Iron Man suit’s flight dynamics:

## IMUSensor
Simulates a 3-axis **accelerometer** and **gyroscope** with configurable:
- **Noise** (`noise_std_accel`, `noise_std_gyro`)
- **Bias** (`bias_accel`, `bias_gyro`)
- **Scale factors** (`scale_accel`, `scale_gyro`)
- **Misalignment matrix** (`misalignment`)

### Key Methods
- `measure_acceleration(true_accel: np.ndarray) -> np.ndarray`  
  Returns noisy, biased, scaled, and misaligned acceleration.
- `measure_angular_rate(true_gyro: np.ndarray) -> np.ndarray`  
  Returns noisy, biased, scaled, and misaligned angular rates.

## PitotSensor
Simulates a pitot tube measuring **static**, **total**, and **dynamic** pressures to infer airspeed.
Configurable:
- **Noise** (`noise_std_static`, `noise_std_total`)
- **Bias** (`bias_static`, `bias_total`)
- **Optional compressibility correction** in `true_airspeed`

### Key Methods
- `measure_static_pressure(altitude: float) -> float`  
  Simulates static pressure (Pa) at altitude.
- `measure_total_pressure(velocity: np.ndarray, altitude: float) -> float`  
  Simulates total pressure (Pa) from dynamic pressure + static.
- `measure_dynamic_pressure(velocity: np.ndarray, altitude: float) -> float`  
  Differential pressure (total - static).
- `indicated_airspeed(q: float) -> float`  
  IAS using sea-level density.
- `true_airspeed(q: float, altitude: float, compressibility: bool=False) -> float`  
  TAS corrected for local density and optional compressibility.

## Usage Example
```python
from backend.aerodynamics.flight_models.sensor_models import IMUSensor, PitotSensor

# Create sensors
e_imu = IMUSensor(noise_std_accel=0.01, bias_gyro=[0.0,0.0,0.0])
pitot = PitotSensor(noise_std_static=0.5, noise_std_total=0.5)

# Simulate measurements
accel_meas = e_imu.measure_acceleration(np.array([0.0,9.81,0.0]))
gyro_meas = e_imu.measure_angular_rate(np.array([0.0,0.0,0.1]))
static_p = pitot.measure_static_pressure(1000)
total_p  = pitot.measure_total_pressure([30.0,0.0,0.0], 1000)
q_dyn     = pitot.measure_dynamic_pressure([30.0,0.0,0.0], 1000)
ias       = pitot.indicated_airspeed(q_dyn)
tas       = pitot.true_airspeed(q_dyn, 1000, compressibility=True)
```

## Logging
All sensor methods emit `DEBUG`-level logs via Python’s `logging` module. Configure your logger to capture sensor diagnostics:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
