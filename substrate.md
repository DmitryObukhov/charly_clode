# Neural Substrate Documentation

This document describes the composition and functioning of the Charly neural substrate.

## Overview

The `Charly` class implements a neuromorphic cellular automaton that:
- Maintains a population of neurons with synaptic connections
- Interfaces with physical models via sensors and actuators
- Executes day/night cycles for simulation and learning

Located in `cl_01/charly.py`

---

## Substrate Composition

### Core Components

```python
class Charly:
    # Neural arrays (double-buffered)
    current: List[Neuron]       # Current state
    next: List[Neuron]          # Next state (computed, not yet committed)

    # Synaptic network
    connectome: Connectome      # All synaptic connections

    # Input/output mappings
    receptors: Dict[str, List[Dict]]    # Sensor → neuron mappings
    actuators: Dict[str, List[int]]     # Neuron → actuator mappings
    innates: Dict[str, Dict]            # Pre-wired generators
    named: Dict[str, int]               # Named neuron aliases

    # Special neurons
    stars: List[int]            # High-EQ neurons for learning

    # Execution state
    iteration: int              # Current step within day
    day_index: int              # Current day number
    day_history: List[Dict]     # Records of all day steps
```

### Neuron Regions

| Region | Indices | Count | Purpose |
|--------|---------|-------|---------|
| Head | 0 to HEAD_COUNT-1 | 200 | Sensory input and innate generators |
| Body | HEAD_COUNT to NEURON_COUNT-1 | 1720 | Recurrent processing network |

**Total neurons**: 1920 (default)

---

## Connectome

The `Connectome` class manages all synaptic connections.

### Structure

```python
class Connectome:
    synapses: List[Synapse]                    # All synapses
    _by_src: Dict[int, List[Synapse]]          # Index by source
    _by_dst: Dict[int, List[Synapse]]          # Index by destination
```

### Methods

```python
add(src: int, dst: int, weight: float)         # Add synapse
get_inputs(dst: int) -> List[Synapse]          # Get incoming synapses
get_outputs(src: int) -> List[Synapse]         # Get outgoing synapses
remove_inputs(dst: int)                        # Remove all inputs to neuron
clear()                                        # Remove all synapses
reindex()                                      # Rebuild indices
```

### Connection Generation

Body neurons (HEAD_COUNT onwards) receive random connections using **banded probability distributions**. Connections are categorized into three distance bands (short/mid/long) with configurable allocation percentages.

#### Distance Bands

The link length range is divided into three bands:

```
|<---- short --->|<-------- mid -------->|<---- long ---->|
0           short_max                long_min      LINK_LENGTH_MAX
```

| Parameter | Default | Formula |
|-----------|---------|---------|
| LINK_LEN_SHORT_RANGE | 10 | short_max = LINK_LENGTH_MAX × 10% |
| LINK_LEN_LONG_RANGE | 10 | long_min = LINK_LENGTH_MAX × 90% |

#### Distribution Configuration

Each neuron segment can have a different distribution:

```yaml
# config.yaml
LINK_DISTRIBUTION_DEFAULT: [80, 15, 5]  # [short%, mid%, long%]

connectome_distribution:
  - start: 200
    end: 500
    distribution: [80, 15, 5]   # Local connectivity (mostly short)
  - start: 500
    end: 800
    distribution: [5, 15, 80]   # Distant connectivity (mostly long)
  # Creates "zebra" pattern: alternating local and distant bands
```

#### Algorithm

```python
for dst_idx in range(HEAD_COUNT, NEURON_COUNT):
    n_links = rng.randint(LINKS_PER_NEURON_MIN, LINKS_PER_NEURON_MAX)

    # Get distribution for this neuron's segment
    dist = get_distribution(dst_idx)  # e.g., [80, 15, 5]
    short_pct, mid_pct, long_pct = dist[0]/100, dist[1]/100, dist[2]/100

    # Categorize candidates by distance band
    short_candidates = [src for distance <= short_max]
    mid_candidates = [src for short_max < distance < long_min]
    long_candidates = [src for distance >= long_min]

    # Calculate target count for each band
    n_short = int(n_links * short_pct)
    n_long = int(n_links * long_pct)
    n_mid = n_links - n_short - n_long

    # Sample from each band
    selected = sample(short_candidates, n_short)
             + sample(mid_candidates, n_mid)
             + sample(long_candidates, n_long)

    # Assign normalized weights
    weights = normalize_to(DEFAULT_CUMULATIVE_INPUT)

    for src, weight in zip(selected, weights):
        connectome.add(src, dst, weight)
```

#### Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| LINKS_PER_NEURON_MIN | 10 | Minimum synapses per neuron |
| LINKS_PER_NEURON_MAX | 200 | Maximum synapses per neuron |
| LINK_LENGTH_MAX | 200 | Maximum index distance for connections |
| LINK_LEN_SHORT_RANGE | 10 | Percentage of range for "short" band |
| LINK_LEN_LONG_RANGE | 10 | Percentage of range for "long" band |
| LINK_DISTRIBUTION_DEFAULT | [80, 15, 5] | Default [short%, mid%, long%] |
| DEFAULT_CUMULATIVE_INPUT | 1000 | Total weight per neuron |

---

## Initialization Sequence

### 1. `_init_substrate()`

Creates all neurons and calls sub-initialization methods:

```python
def _init_substrate(self) -> None:
    # Create neurons with random EQ
    for i in range(NEURON_COUNT):
        neuron = Neuron(
            name=f"N{i}",
            eq=rng.randint(DEFAULT_MIN_EQ, DEFAULT_MAX_EQ)
        )
        self.current.append(neuron)
        self.next.append(neuron.clone())

    self._build_connectome()
    self._configure_receptors()
    self._configure_actuators()
    self._configure_innates()
    self._configure_named()
    self._identify_stars()
```

### 2. `_build_connectome()`

Generates random synaptic connections for body neurons (see Connectome section).

### 3. `_configure_receptors()`

Maps physical model sensors to head neurons using population coding:

```yaml
# config.yaml
inputs:
  - signal: "eye"
    type: "population"
    neuron_start: 0
    neuron_count: 50
    gamma: 0.4
    eq_gradient:
      low: -20
      high: 20
```

**Initialization process:**
1. Assigns neurons from `neuron_start` to `neuron_start + neuron_count - 1`
2. Sets each neuron's EQ via linear interpolation between `eq_gradient.low` and `eq_gradient.high`
3. Names neurons as `{signal}_{index}` (e.g., "eye_0", "eye_1", ..., "eye_49")

**Result structure:**
```python
self.receptors = {
    "eye": {
        'type': 'population',
        'neuron_start': 0,
        'neuron_count': 50,
        'gamma': 0.4,
        'eq_gradient': {'low': -20, 'high': 20}
    }
}
```

### 4. `_configure_actuators()`

Maps neuron groups to physical model actuators. Supports two formats:

#### New Format (Recommended)

List of actuator definitions with optional wired inputs:

```yaml
# config.yaml
actuators:
  - name: move_up
    start: 4000          # First neuron index
    count: 400           # Number of neurons
    default_eq: 5        # EQ assigned to all neurons
    color: [0, 255, 0]   # Visualization color
    inputs:              # Optional: wire direct synaptic connections
      start: 700         # Source neuron range start
      count: 300         # Number of source neurons
      weight: 100        # Weight per connection

  - name: move_down
    start: 4500
    count: 400
    default_eq: 5
    color: [255, 0, 0]
    # No inputs: actuator receives connections only from connectome
```

**Wired Inputs**: When `inputs` is specified, direct synaptic connections are added from the source range to all actuator neurons. This bypasses the random connectome for targeted control (e.g., connecting pleasure/pain signals directly to motor neurons).

**Result structure:**
```python
self.actuators = {
    "move_up": {
        'indices': [4000, 4001, ..., 4399],
        'color': (0, 255, 0),
        'start': 4000,
        'count': 400
    },
    ...
}
```

#### Legacy Format

Simple dict mapping names to neuron index lists:

```yaml
# config.yaml (legacy)
actuators:
  left: [180, 181, 182, 183, 184, 185, 186, 187, 188, 189]
  right: [190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
```

### 5. `_configure_innates()`

Sets up pre-wired generator neurons:

```yaml
# config.yaml
innates:
  hunger_generator:
    idx: 50
    eq: 25
    charge_cycle: 100      # Fires every 100 iterations
    inputs: []
  satisfaction_generator:
    idx: 51
    eq: -20
    charge_cycle: 0        # No cyclic firing
    inputs:
      - [2, 500]           # Connected to neuron 2 with weight 500
```

**Types:**
- **Generators**: `charge_cycle > 0` - fire periodically
- **Feedforward**: `charge_cycle = 0` with inputs - respond to specific neurons

### 6. `_configure_named()`

Creates aliases for monitoring:

```yaml
# config.yaml
named:
  motor_left_main: 180
  motor_right_main: 190
```

### 7. `_identify_stars()`

Finds neurons with high emotional quantum:

```python
def _identify_stars(self) -> None:
    self.stars = [idx for idx, n in enumerate(self.current)
                  if abs(n.eq) >= STAR_LEVEL]
```

Stars are used during night phase learning.

### 8. `_configure_saurons_finger()`

Configures dynamic substrate modification using formula-based "finger presses":

```yaml
# config.yaml
saurons_finger:
  - name: suppression_wave
    x: 500                    # Center neuron index
    r: 100                    # Radius (affects neurons x-r to x+r)
    field: elastic_trigger    # Field to modify
    shape: "exp(-((DIST/R)**2))"  # Gaussian falloff
    formula: "PREV + SHAPE * TEMPORAL * 100"
    rows:
      - start: 0              # Active from iteration 0
        end: 500              # Until iteration 500
        formula: "1.0"        # Full strength
      - start: 501
        end: 1000
        formula: "1.0 - (ROW - 501) / 500"  # Linear fade out
```

See **Sauron's Finger** section below for full documentation.

---

## Sauron's Finger

A subsystem for formula-based dynamic modification of neuron fields during simulation. Named for its ability to "press" on the neural substrate with configurable spatial and temporal patterns.

### Configuration

Each finger press defines:

| Field | Purpose |
|-------|---------|
| `name` | Identifier for logging |
| `x` | Center neuron index |
| `r` | Radius of effect |
| `field` | Neuron field to modify (see table below) |
| `shape` | Formula for spatial falloff |
| `formula` | Formula for new field value |
| `rows` | List of temporal windows |

### Supported Field Types

| Field | Effect | Use Case |
|-------|--------|----------|
| `elastic_trigger` | Modifies activation threshold | Make neurons easier/harder to fire |
| `elastic_recharge` | Modifies recharge rate | Speed up/slow down recovery |
| `charge` | Directly injects/drains charge | Force activation or silence neurons |
| `cumulative_signal` | Adds artificial input signal | Simulate external input |
| `eq` | Changes emotional quantum | Turn neutral zone into pleasure/pain |
| `fatigue` | Adds/removes fatigue | Create hyperactive or exhausted zones |

### Finger Types Summary

**1. Trigger Finger** (`elastic_trigger`)
- Lowers threshold → neurons fire more easily
- Raises threshold → neurons become less responsive
- Affects wave propagation speed and spread

**2. EQ Finger** (`eq`)
- Increases EQ → zone contributes positive emotion (ESP)
- Decreases EQ → zone contributes negative emotion (ESN)
- Affects learning during night phase (star neurons)

**3. Fatigue Finger** (`fatigue`)
- Reduces fatigue → neurons can fire rapidly (hyperactive)
- Increases fatigue → neurons become exhausted (silenced)
- Creates local "anesthesia" or "stimulation" effects

**4. Charge Finger** (`charge`)
- Injects charge → forces neurons toward firing threshold
- Drains charge → prevents neurons from firing
- Most direct intervention, immediate effect

### Formula Variables (Macros)

Formulas can use these variables:

| Variable | Description |
|----------|-------------|
| `IDX` | Current neuron index |
| `X` | Center of finger press |
| `R` | Radius of finger press |
| `DIST` | Distance from center: \|IDX - X\| |
| `PREV` | Current value of the target field |
| `SHAPE` | Evaluated shape formula result |
| `TEMPORAL` | Evaluated row formula result (0 if no row active) |
| `ROW` | Current iteration (for row formulas) |

### Available Math Functions

```python
exp, sin, cos, sqrt, abs, min, max, pi, e
```

### Example: Gaussian Suppression

```yaml
saurons_finger:
  - name: center_suppression
    x: 960                    # Center of substrate
    r: 200
    field: elastic_trigger
    shape: "exp(-((DIST/R)**2) * 2)"  # Gaussian with width control
    formula: "PREV + SHAPE * TEMPORAL * 50"
    rows:
      - start: 0
        end: 1080
        formula: "sin(ROW / 100 * pi)"  # Sinusoidal pulse
```

### Temporal Windows (Rows)

Each row defines when the finger is active:

```yaml
rows:
  - start: 0      # First active iteration
    end: 500      # Last active iteration
    formula: "1.0"  # Temporal multiplier (evaluated with ROW variable)
```

Multiple rows can create complex temporal patterns:
- Only one row is active at a time (first matching)
- If no row matches current iteration, `TEMPORAL = 0` and finger has no effect

### Interactive Mode (Arc Visualization)

When running with `--arc` flag, fingers can be applied interactively:

```
python main.py --arc
```

**Controls:**
| Key | Action |
|-----|--------|
| `1` | Select Trigger Finger (elastic_trigger) |
| `2` | Select EQ Finger (emotional quantum) |
| `3` | Select Fatigue Finger |
| `4` | Select Charge Finger |
| LMB | Apply positive effect (lower threshold, +EQ, -fatigue, +charge) |
| RMB | Apply negative effect (raise threshold, -EQ, +fatigue, -charge) |
| `q` | Quit simulation |

**Visual Indicators:**
- Yellow: Trigger Finger
- Green: EQ Finger
- Magenta: Fatigue Finger
- Cyan: Charge Finger

### Application Order

Finger presses are applied during `day_step()` after input processing but before body neuron processing:

```
1. Process inputs (activate receptors)
2. Process innates (fire generators)
3. Apply Sauron's Finger presses  ← HERE
4. Copy head activations to next array
5. Process body neurons
6. Calculate emotional state
```

---

## Day Phase Execution

### `run_day()` Method

Executes a complete day:

```python
def run_day(self) -> List[Dict]:
    self.day_history = []
    actuator_values = None

    for step in range(DAY_STEPS):
        result = self.day_step(actuator_values)
        actuator_values = result['outputs']

    return self.day_history
```

**DAY_STEPS**: 1080 iterations per day (default)

### `day_step()` Algorithm

Single iteration of day simulation:

```
1. Apply actuators to physical model
   └─ physical_model.set(actuator_values)

2. Read sensors from physical model
   └─ sensor_values = physical_model.get(sensor_names)

3. Copy current → next, clear activations
   └─ next[i] = current[i].clone(); next[i].active = False

4. Process head neurons
   ├─ Clear head activations
   ├─ _process_inputs()   ← Activate receptors based on sensors
   └─ _process_innates()  ← Fire generator neurons

5. Collect star values (before body processing)

6. Process body neurons
   └─ for idx in range(HEAD_COUNT, NEURON_COUNT):
          _process_neuron(idx)

7. Calculate emotional state
   └─ esp, esn = _calculate_emotional_state()

8. Commit: current = next

9. Calculate actuator outputs
   └─ new_actuators = _calculate_actuator_outputs()

10. Save to history and return
```

### Input Processing: `_process_inputs()`

Activates receptor neurons using population coding with Stevens' power law:

```python
def _process_inputs(self) -> None:
    sensor_values = self.physical_model.get(sensor_names)

    for signal, cfg in self.receptors.items():
        value = sensor_values.get(signal, 0.0)

        if cfg['type'] == 'population':
            # Stevens' power law: perceived = physical^gamma
            gamma = cfg.get('gamma', 0.4)
            perceived = value ** gamma

            # Calculate number of active neurons
            n_active = round(cfg['neuron_count'] * perceived)

            # Activate neurons 0 through n_active-1 (thermometer code)
            start = cfg['neuron_start']
            for i in range(n_active):
                self.current[start + i].active = True
```

**Thermometer Code**: Neurons activate from index 0 upward. At low luminosity, only the first few neurons fire. At high luminosity, most or all neurons in the population fire. This creates a graded representation where:
- Darkness → sparse activity (few neurons, negative-EQ dominated)
- Brightness → dense activity (many neurons, positive-EQ dominated)

### Innate Processing: `_process_innates()`

Fires generator neurons:

```python
def _process_innates(self) -> None:
    for name, cfg in self.innates.items():
        idx = cfg['idx']
        charge_cycle = cfg.get('charge_cycle', 0)

        if charge_cycle > 0 and self.iteration % charge_cycle == 0:
            self.current[idx].active = True
```

### Actuator Output: `_calculate_actuator_outputs()`

Population coding - fraction of active neurons:

```python
def _calculate_actuator_outputs(self) -> Dict[str, float]:
    outputs = {}
    for name, indices in self.actuators.items():
        active_count = sum(1 for idx in indices if self.current[idx].active)
        outputs[name] = active_count / len(indices)
    return outputs
```

**Output range**: [0.0, 1.0]

---

## Emotional State

### Calculation

```python
def _calculate_emotional_state(self) -> Tuple[int, int]:
    esp = 0  # Emotional State Positive
    esn = 0  # Emotional State Negative

    for neuron in self.next:
        if neuron.active:
            if neuron.eq > 0:
                esp += neuron.eq
            elif neuron.eq < 0:
                esn += neuron.eq

    return esp, esn
```

| Value | Meaning |
|-------|---------|
| ESP | Sum of EQ from active positive-EQ neurons |
| ESN | Sum of EQ from active negative-EQ neurons (negative value) |

### Interpretation

- **High ESP, low |ESN|**: Positive emotional state (approach)
- **Low ESP, high |ESN|**: Negative emotional state (avoidance)
- **Combined**: `ESP + ESN` gives overall valence

---

## Night Phase: Learning

### `run_night()` Method

Simple Hebbian-style learning on star neurons:

```python
def run_night(self) -> None:
    for step_record in self.day_history:
        stars = step_record.get('stars', {})

        for star_idx, (was_active, eq) in stars.items():
            if was_active:
                inputs = self.connectome.get_inputs(star_idx)

                for synapse in inputs:
                    if eq > 0:
                        synapse.weight *= 1.01    # Strengthen
                    elif eq < 0:
                        synapse.weight *= 0.99    # Weaken

    self.connectome.reindex()
    self.day_index += 1
```

### Learning Rule

| Star EQ | Learning Effect |
|---------|-----------------|
| Positive | Strengthen incoming synapses (×1.01) |
| Negative | Weaken incoming synapses (×0.99) |

This reinforces pathways leading to positive affective states and inhibits those leading to negative states.

---

## History Record Structure

Each `day_step()` saves:

```python
{
    'iteration': int,           # Step number
    'esp': int,                 # Emotional State Positive
    'esn': int,                 # Emotional State Negative
    'inputs': {                 # Sensor values
        'eye': 0.6,
        'agent_pos': 0.5,
        ...
    },
    'outputs': {                # Actuator values
        'left': 0.3,
        'right': 0.1,
        ...
    },
    'stars': {                  # Star neuron states
        50: (True, 25),         # (was_active, eq)
        51: (False, -20),
        ...
    },
    'neurons': [Neuron, ...]    # Full neuron state array
}
```

---

## Configuration Reference

### Substrate Parameters

```yaml
NEURON_COUNT: 1920          # Total neurons
HEAD_COUNT: 200             # Sensory/innate region size
SEED: 42                    # Random seed for reproducibility
```

### Neuron Parameters

```yaml
CHARGE_MIN: 100             # Minimum charge to fire
RECHARGE_RATE: 10           # Base charge per iteration
NORMAL_TRIGGER: 500         # Activation threshold
DEFAULT_CUMULATIVE_INPUT: 1000  # Total weight per neuron
```

### Elastic Parameters

```yaml
ELASTIC_TRIGGER_PROPAGATION: 0.75
ELASTIC_TRIGGER_DEGRADATION: 1
ELASTIC_RECHARGE_PROPAGATION: 0.75
ELASTIC_RECHARGE_DEGRADATION: 1
```

### EQ Parameters

```yaml
EQ_MAX: 255                 # Maximum absolute EQ
DEFAULT_MIN_EQ: -3          # Initialization min
DEFAULT_MAX_EQ: 3           # Initialization max
STAR_LEVEL: 20              # Threshold for star classification
```

### Connectome Parameters

```yaml
LINKS_PER_NEURON_MIN: 10
LINKS_PER_NEURON_MAX: 200
LINK_LENGTH_MAX: 200
```

### Timing

```yaml
DAY_STEPS: 1080             # Iterations per day
HISTORY_MAXLEN: 256         # Neuron history length
```

---

## Data Flow Diagram

```
                      PHYSICAL MODEL
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         │
         ┌─────────┐                    │
         │ Sensors │                    │
         └────┬────┘                    │
              │                         │
              ▼                         │
    ┌─────────────────┐                 │
    │   RECEPTORS     │                 │
    │ (value ranges)  │                 │
    └────────┬────────┘                 │
              │                         │
              ▼                         │
    ┌─────────────────┐                 │
    │   HEAD NEURONS  │◄────────────────┤
    │   (0-199)       │    INNATES      │
    └────────┬────────┘    (generators) │
              │                         │
              ▼                         │
    ┌─────────────────┐                 │
    │   CONNECTOME    │                 │
    │   (synapses)    │                 │
    └────────┬────────┘                 │
              │                         │
              ▼                         │
    ┌─────────────────┐                 │
    │   BODY NEURONS  │                 │
    │   (200-1919)    │                 │
    └────────┬────────┘                 │
              │                         │
              ▼                         │
    ┌─────────────────┐                 │
    │   ACTUATORS     │                 │
    │ (population     │                 │
    │  coding)        │                 │
    └────────┬────────┘                 │
              │                         │
              ▼                         │
         ┌─────────┐                    │
         │ Control │────────────────────┘
         └─────────┘
```

---

## Complete Simulation Loop

```python
# Setup
config = yaml.safe_load(open('config.yaml'))
physical_model = Linear(config)
charly = Charly(config, physical_model)

# Run simulation
for day in range(num_days):
    results = charly.run_day()      # 1080 iterations
    charly.run_night()              # Learn from day
    charly.save_visualization(...)  # Generate output
```

---

## Visualization

### Day History: `visualize()`

Renders neural activity over time as a 2D image (neurons × iterations).

### Connectome Structure: `faceshot()`

```python
faceshot(w: int, h: int, filename: str = None) -> PIL.Image
```

Visualizes the connectome structure on a circular diagram.

**Layout:**
- Neurons arranged evenly around a circle
- Connections drawn as lines between neurons

**Neuron Colors (based on EQ):**
| EQ Value | Color | Brightness |
|----------|-------|------------|
| Positive | Green | Proportional to \|EQ\| |
| Negative | Red | Proportional to \|EQ\| |
| Zero | Gray | Dim |

**Connection Colors (based on weight):**
| Weight | Color | Brightness |
|--------|-------|------------|
| Positive | Green | Proportional to \|weight\| / max_weight |
| Negative | Red | Proportional to \|weight\| / max_weight |

**Parameters:**
- `w`: Output image width in pixels
- `h`: Output image height in pixels
- `filename`: Optional path to save PNG (if None, only returns Image)

**Returns:** PIL.Image object

**Usage:**
```python
# Get image only
img = charly.faceshot(1920, 1080)

# Save to file
img = charly.faceshot(1920, 1080, "connectome.png")
```
