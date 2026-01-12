# Physical Model Documentation

This document describes the physical model layer that interfaces between the neural substrate (Charly) and simulated physical worlds.

## Overview

The physical model layer provides:
- Abstract interface (`PhysicalModel`) defining sensor/actuator contracts
- Concrete implementations (e.g., `Linear` for 1D world)
- State history tracking and visualization

## Abstract Interface (`PhysicalModel`)

Located in `cl_01/physical_model.py`

### Required Methods

```python
set(actuators: Dict[str, float]) -> None
```
Apply actuator values to advance the simulation one step. Values are clamped to [0, 1].

```python
get(sensors: Union[List[str], Set[str]]) -> Dict[str, float]
```
Read current sensor values. Returns normalized values in [0, 1].

```python
get_names() -> Tuple[List[str], List[str]]
```
Returns `(input_names, output_names)` for compatibility verification.

```python
reset() -> None
```
Reset model to initial state and clear history.

```python
_get_state_snapshot() -> Dict
```
Capture current state for history recording.

```python
_create_raw_visualization(num_iterations: int) -> Image.Image
```
Generate visualization from recorded history.

### Template Methods (Inherited)

- `save_state()` - Records state snapshot to history
- `visualize(width, height, iterations)` - Creates scaled visualization
- `save_visualization(filename, ...)` - Saves visualization to PNG
- `get_history()` / `clear_history()` - History management
- `set_logger(log_func)` - Inject custom logging

---

## Linear Model (1D World)

Located in `cl_01/model_linear.py`

### World Description

A 1D environment where:
- **Agent**: Blue dot with light sensor ("eye") and two flagella actuators
- **Lamp**: Red dot moving sinusoidally as a light source
- **World size**: 1920 pixels (configurable)

### Constants

```python
WORLD_SIZE = 1920           # Default world size in pixels
FLAGELLA_STRENGTH = 5.0     # Maximum movement per iteration
LIGHT_FALLOFF = 200.0       # Reference distance for light falloff
MAX_ILLUMINATION = 1.0      # Maximum light intensity
```

### Initialization

```python
Linear(world_size=1920, lamp_amplitude=None, lamp_frequency=0.01, lamp_center=None)
```

- `lamp_amplitude`: Default = world_size/3 (≈640 pixels)
- `lamp_frequency`: Default = 0.01 (period = 100 iterations)
- `lamp_center`: Default = world_size/2 (960 pixels)

Initial agent position: center of world

---

## Mathematical Models

### Light Intensity

**Inverse-square falloff with offset:**

```
I = I_max / (1 + (d / d₀)²)
```

Where:
- `I_max = 1.0` (maximum illumination)
- `d` = distance between agent and lamp (pixels)
- `d₀ = 200.0` (LIGHT_FALLOFF reference distance)

**Behavior:**
| Distance | Intensity |
|----------|-----------|
| 0 | 1.0 |
| 200 | 0.5 |
| 400 | 0.2 |
| 1000 | 0.04 |

The "+1" offset prevents singularity at d=0 and ensures smooth falloff.

### Lamp Movement

**Sinusoidal oscillation:**

```
position = center + amplitude × sin(2π × frequency × iteration)
```

With defaults:
- `center = 960`
- `amplitude = 640`
- `frequency = 0.01`
- Period = 1/0.01 = 100 iterations
- Range: [320, 1600] pixels

Position clamped to [0, world_size-1].

### Agent Movement

```
movement = (left_activation - right_activation) × FLAGELLA_STRENGTH
new_position = clamp(current_position + movement, 0, world_size-1)
```

**Key semantics:**
- Left flagella activation pushes agent **RIGHT** (+)
- Right flagella activation pushes agent **LEFT** (-)
- Simultaneous activation: partial cancellation
- Maximum movement: ±5 pixels per step

---

## Sensor Interface

| Sensor | Range | Description |
|--------|-------|-------------|
| `eye` | [0, 1] | Light intensity at agent position |
| `agent_pos` | [0, 1] | Normalized position (agent_pos / world_size) |
| `lamp_pos` | [0, 1] | Normalized position (lamp_pos / world_size) |

### Sensor-to-Neuron Mapping

Configured in YAML `inputs` section:

```yaml
inputs:
  - signal: "eye"
    min_val: 0.0
    max_val: 0.3
    neurons:
      - idx: 0
        eq: -25
        name: "eye_dark"
  - signal: "eye"
    min_val: 0.3
    max_val: 0.7
    neurons:
      - idx: 1
        eq: -50
        name: "eye_medium"
  - signal: "eye"
    min_val: 0.7
    max_val: 1.0
    neurons:
      - idx: 2
        eq: -100
        name: "eye_bright"
```

A neuron activates when the sensor value falls within its configured range. Multiple ranges enable multi-way sensory encoding.

---

## Actuator Interface

### Population Coding

Actuator activation = fraction of active neurons in the assigned group:

```python
activation = active_neuron_count / total_neurons_in_group
```

With 10 neurons per actuator, each active neuron contributes 10% activation.

### Configuration

```yaml
actuators:
  left:
    - 180
    - 181
    - ...
    - 189    # 10 neurons
  right:
    - 190
    - 191
    - ...
    - 199    # 10 neurons
```

---

## Data Flow

### Per-Iteration Sequence

```
Step N:
1. physical_model.set(actuators_from_step_N-1)
   ├─ Apply movement
   ├─ Update lamp position
   ├─ Recalculate illumination
   └─ save_state()

2. sensor_values = physical_model.get(['eye', ...])

3. Neural computation (Charly.day_step)
   ├─ Activate receptor neurons based on sensor ranges
   ├─ Process neural network
   └─ Calculate new actuator outputs

4. Store actuator outputs for Step N+1
```

**Important**: There is a 1-step delay between sensing and action, mimicking neural processing latency.

### History Record Structure

Each iteration saves:
```python
{
    'iteration': N,
    'esp': sum_positive_active_EQs,
    'esn': sum_negative_active_EQs,
    'inputs': {'eye': 0.6, 'agent_pos': 0.5, 'lamp_pos': 0.8},
    'outputs': {'left': 0.1, 'right': 0.2},
    'stars': {50: (True, 25), 51: (False, -20)},
    'neurons': [neuron_states...]
}
```

---

## Configuration Reference

### World Parameters

```yaml
WORLD_SIZE: 1920            # Size of 1D world
COLOR_AGENT: [0, 0, 255]    # Blue
COLOR_LAMP: [255, 0, 0]     # Red
COLOR_BG: [0, 0, 0]         # Black
```

### Lamp Parameters (constructor)

- `lamp_amplitude`: Oscillation magnitude (default: world_size/3)
- `lamp_frequency`: Oscillation rate (default: 0.01)
- `lamp_center`: Center of oscillation (default: world_size/2)

---

## Visualization

Output: PNG image (world_size × num_iterations)

- Each row = one iteration
- Each column = one world position
- Red pixel = lamp position
- Blue pixel = agent position (brightness reflects illumination)
  - Bright blue = high light intensity
  - Dark blue = low light intensity
  - Minimum brightness 50 ensures visibility

---

## Extending with New Models

To create a new physical model:

```python
from physical_model import PhysicalModel
from typing import Dict, List, Set, Tuple, Union
from PIL import Image

class MyModel(PhysicalModel):
    def __init__(self, ...):
        super().__init__()
        # Initialize state

    def set(self, actuators: Dict[str, float]) -> None:
        # Apply actuators, update state
        self.iteration += 1
        self.save_state()

    def get(self, sensors: Union[List[str], Set[str]]) -> Dict[str, float]:
        # Return requested sensor values (normalized to [0,1])
        return {name: value for name in sensors}

    def get_names(self) -> Tuple[List[str], List[str]]:
        return (['sensor1', 'sensor2'], ['actuator1', 'actuator2'])

    def reset(self) -> None:
        self.iteration = 0
        self.history = []
        # Reset other state

    def _get_state_snapshot(self) -> Dict:
        return {'var1': self.var1, 'var2': self.var2}

    def _create_raw_visualization(self, num_iterations: int) -> Image.Image:
        # Create PIL image from self.history
        pass
```

---

## Value Ranges Summary

| Component | Type | Range | Notes |
|-----------|------|-------|-------|
| Agent position | float | [0, world_size-1] | Pixels |
| Lamp position | float | [0, world_size-1] | Pixels |
| Light intensity | float | [0, 1.0] | Asymptotic max |
| Sensor values | float | [0, 1.0] | Normalized |
| Actuator activation | float | [0, 1.0] | Population ratio |
| Neuron activation | bool | True/False | Binary |
| EQ (Emotional Quantum) | int | [-255, 255] | Genetic bias |
