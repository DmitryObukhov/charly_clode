# Devlog: 2026-01-16 - Physical Model v2: Orgasm/Terror System

## Session Summary

Complete redesign of the physical model to implement pleasure/pain feedback loop for agent tracking a moving lamp.

## New Physical Model

### Concept
- Agent must track a lamp moving in 1D space
- **Orgasm**: Reward when agent matches lamp position (within tolerance)
- **Terror**: Punishment when agent deviates above or below lamp

### Lamp Movement
- Configurable modes: `sine`, `linear`, `file`
- Default: sine wave with 40% amplitude, 4000 step period
- File mode: load positions from CSV (e.g., stock prices)

### Sensory Signals
| Signal | Condition | Range |
|--------|-----------|-------|
| `orgasm` | Within tolerance | 0-1 (1 = perfect match) |
| `terror_up` | Agent above lamp | 0-1 (sigmoid curve) |
| `terror_down` | Agent below lamp | 0-1 (sigmoid curve) |

### Response Curve Control (Smoothness)
New parameter controlling signal response shape:
- **0%** = Binary step (instant 0→1 transition)
- **50%** = Sigmoid (slow start, sharp rise, slow finish) - default
- **100%** = Linear (uniform growth)

Formula: Blend of sigmoid with variable steepness and linear interpolation.

## Neuron Layout (5000 total)

### Head (0-999) - Sensory Input
| Range | Function | EQ |
|-------|----------|-----|
| 0-99 | Reserved | - |
| 100-399 | Terror Up | -20 to -600 |
| 400-699 | Terror Down | -20 to -600 |
| 700-999 | Orgasm | +50 to +800 |

### Body (1000-3999) - Integration
Standard connectome with mixed local/distant connections.

### Tail (4000-4999) - Motor Output
| Range | Function | EQ |
|-------|----------|-----|
| 4000-4399 | Move Up actuator | +30 |
| 4500-4899 | Move Down actuator | +30 |

## New Config Parameters

```yaml
# Lamp Movement
LAMP_MODE: "sine"
LAMP_AMPLITUDE: 0.4
LAMP_PERIOD: 4000
LAMP_CENTER: 0.5

# Agent
AGENT_SPEED: 0.001
AGENT_START: 0.5

# Sensory
ORGASM_TOLERANCE: 0.10
TERROR_RANGE: 0.5
TERROR_SMOOTHNESS: 50
ORGASM_SMOOTHNESS: 50
```

## Files Modified

- `config/config.yaml` - Complete restructure for new model
- `src/model_linear.py` - New physical model with orgasm/terror signals
- `src/main.py` - Pass new config parameters to model

## Test Results

Dynamics working correctly:
- Lamp oscillates 0.1 → 0.9 over 4000 steps
- Orgasm = 1.0 when agent matches lamp
- Terror increases with sigmoid curve as agent deviates
- Smoothness parameter controls curve shape
