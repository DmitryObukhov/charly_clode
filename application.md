# Application Documentation

This document describes the main CLI application (`main.py`) that orchestrates the Charly simulation.

## Overview

The test application:
- Reads configuration and creates a Charly instance
- Creates the physical model
- Executes day/night simulation cycles
- Saves history and visualizations
- Generates video output

Located in `cl_01/main.py`

---

## CLI Interface

```bash
python main.py [options]
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--config` | `-c` | `config.yaml` | Configuration file path |
| `--days` | `-d` | `1` | Number of days to simulate |
| `--output` | `-o` | `output` | Output directory for results |
| `--fps` | | `30` | Video frames per second |
| `--no-video` | | `False` | Skip video generation |
| `--model-path` | | | Physics model file path |
| `--log-level` | | `INFO` | Logging verbosity |

### Examples

```bash
# Basic simulation
python main.py --days 5

# Custom config and output
python main.py --config custom.yaml --output results --days 10

# High frame rate video
python main.py --days 3 --fps 60

# Skip video generation
python main.py --days 5 --no-video
```

---

## Execution Flow

### 1. Initialization

```python
# Load configuration
config = yaml.safe_load(open(config_path))

# Create physical model
physical_model = Linear(world_size=config.get('WORLD_SIZE', 1920))

# Create Charly instance
charly = Charly(config, physical_model)
```

### 2. Simulation Loop

For each day:

```python
for day in range(num_days):
    # Run day phase (DAY_STEPS iterations)
    results = charly.run_day()

    # Run night phase (learning)
    charly.run_night()

    # Save visualization
    filename = f"output/day_{day:04d}.png"
    charly.save_visualization(filename, width=1920, height=1080)
```

### 3. Output Generation

**Per-day outputs:**
- `output/day_0000.png` - Neural activity visualization
- `output/day_0001.png` - ...
- Day history logs

**Final outputs:**
- `output/simulation.mp4` - Compiled video from all day images

---

## Output Files

### Visualization PNG

Each day produces a PNG visualization:
- **Width**: NEURON_COUNT pixels (1920)
- **Height**: DAY_STEPS pixels (1080)
- **Content**: Neural activity heatmap
  - X-axis: Neuron index
  - Y-axis: Iteration (time)
  - Color: EQ-based (green=positive, red=negative)
  - Brightness: |EQ| magnitude

### Video Compilation

After simulation completes:
1. Collect all `day_*.png` images
2. Encode to MP4 using OpenCV (preferred) or MoviePy
3. Frame rate from `--fps` argument
4. Output: `output/simulation.mp4`

---

## Progress Output

The application prints progress to stdout:

```
=== Day 1 ===
Starting day 0...
  Step 100/1080: ESP=125, ESN=-45
  Step 200/1080: ESP=130, ESN=-50
  ...
Visualization saved to output/day_0000.png

=== Day 2 ===
...

Simulation complete!
Generating video...
Video saved to output/simulation.mp4
```

---

## Configuration Loading

The application reads `config.yaml` containing:

```yaml
# Simulation parameters
SEED: 42
NEURON_COUNT: 1920
HEAD_COUNT: 200
DAY_STEPS: 1080

# Physical model
WORLD_SIZE: 1920

# Neural substrate parameters
CHARGE_MIN: 100
RECHARGE_RATE: 10
NORMAL_TRIGGER: 500

# Input/output mappings
inputs:
  - signal: "eye"
    min_val: 0.0
    max_val: 0.3
    neurons:
      - idx: 0
        eq: -25

actuators:
  left: [180, 181, ..., 189]
  right: [190, 191, ..., 199]

innates:
  hunger_generator:
    idx: 50
    eq: 25
    charge_cycle: 100

named:
  motor_left_main: 180
```

---

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| FileNotFoundError | Missing config file | Check `--config` path |
| KeyError | Missing config parameter | Add required field to YAML |
| ImportError | Missing dependency | Install: `pip install -r requirements.txt` |
| PermissionError | Cannot write output | Check output directory permissions |

---

## Dependencies

Required for full functionality:

```
PyYAML          # Configuration parsing
Pillow          # Image generation
numpy           # Numerical operations
opencv-python   # Video encoding (preferred)
moviepy         # Video encoding (fallback)
```

---

## Module Testing

The main.py can also be used for quick tests:

```bash
# Test with minimal config
python main.py --days 1 --config config_small.yaml

# Test without video
python main.py --days 1 --no-video
```

Each submodule also has standalone tests:

```bash
python charly.py          # Test neural substrate
python model_linear.py    # Test physical model
python physical_model.py  # Test base class
```
