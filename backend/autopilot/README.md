# Autopilot

The `autopilot` package provides high-level guidance and control modules for the Iron Man suit simulation.

## Modules

### WaypointPlanner
- Manages a sequence of 3D waypoints with methods to:
  - Add, insert, clear and reset waypoints
  - Compute distance and bearing to the current waypoint
  - Automatically advance when within an acceptance radius
  - Generate loiter (hold) patterns
  - Calculate cross-track error
  - Produce pure-pursuit velocity vectors

### NeuralController
- A feed-forward PyTorch neural network autopilot:
  - Configurable layer sizes, layer normalization, dropout
  - `predict` for inference on state vectors
  - `save`/`load` for model persistence

#### ControllerDataset & ControllerTrainer
- `ControllerDataset`: wrap numpy data for supervised training
- `ControllerTrainer`: handles training loops, validation, optimizer (Adam), LR scheduler, and logging

## Usage Example
```python
from backend.autopilot import WaypointPlanner, NeuralController, ControllerDataset, ControllerTrainer
import numpy as np

# Waypoint planning
wps = [[0,0,10], [50,0,10], [50,50,10]]
planner = WaypointPlanner(wps, acceptance_radius=2.0)
pos = [0,0,0]
vel_vec = planner.get_desired_velocity(pos, speed=5.0)

# Neural controller inference
state = np.array([*pos, *vel_vec])  # example state vector
model = NeuralController(input_size=len(state), output_size=3)
actions = model.predict(state)

# Training a controller
X_train, y_train = np.random.rand(100, len(state)), np.random.rand(100,3)
dataset = ControllerDataset(X_train, y_train)
trainer = ControllerTrainer(model)
trainer.fit(dataset, epochs=10, batch_size=16)
```  

## Logging
All classes use Python’s `logging` at `DEBUG` and `INFO` levels. To enable:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
