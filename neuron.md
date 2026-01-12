# Neuron Structure and Processing

This document describes the neuron data structure and the `_process_neuron` algorithm.

## Neuron Dataclass

Located in `cl_01/charly.py`

```python
@dataclass
class Neuron:
    name: str = ""
    active: bool = False
    eq: int = 0
    cumulative_signal: float = 0.0
    elastic_trigger: float = 0.0
    charge: float = 0.0
    elastic_recharge: float = 0.0
    charge_cycle: int = 0
    history: str = ""
```

## Field Descriptions

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `name` | str | "" | Logical identifier (e.g., "N42", "eye_bright") |
| `active` | bool | False | Current firing state (binary: on/off) |
| `eq` | int | 0 | Emotional Quantum: affective tone [-EQ_MAX, +EQ_MAX] |
| `cumulative_signal` | float | 0.0 | Sum of weighted inputs from active synapses |
| `elastic_trigger` | float | 0.0 | Elastic adjustment to firing threshold |
| `charge` | float | 0.0 | Current charge level for threshold activation |
| `elastic_recharge` | float | 0.0 | Elastic adjustment to recharge rate |
| `charge_cycle` | int | 0 | Period for cyclic discharge (0 = disabled) |
| `history` | str | "" | Rolling string of '0'/'1' tracking recent activations |

## Neuron Initialization

```python
neuron = Neuron(
    name=f"N{i}",
    active=False,
    eq=rng.randint(DEFAULT_MIN_EQ, DEFAULT_MAX_EQ),  # Typically [-3, +3]
    cumulative_signal=0.0,
    elastic_trigger=0.0,
    charge=0.0,
    elastic_recharge=0.0,
    charge_cycle=0,
    history=""
)
```

---

## Synapse Structure

```python
@dataclass
class Synapse:
    src: int        # Source neuron index
    dst: int        # Destination neuron index
    weight: float   # Synaptic weight (typically positive)
```

**Weight Distribution:**
- Weights are normalized so total input per neuron = DEFAULT_CUMULATIVE_INPUT (1000)
- Formula: `weight_i = (random_i / sum_random) × DEFAULT_CUMULATIVE_INPUT`
- Only active source neurons contribute to cumulative_signal

---

## Processing Algorithm: `_process_neuron(idx)`

### Step 1: Base Copy
```python
next_neuron.eq = current_neuron.eq
next_neuron.name = current_neuron.name
next_neuron.charge_cycle = current_neuron.charge_cycle
```

### Step 2: Calculate Cumulative Signal
```python
inputs = self.connectome.get_inputs(idx)
cumulative_signal = 0.0
has_active_inputs = False

for synapse in inputs:
    src_neuron = self.current[synapse.src]
    if src_neuron.active:
        cumulative_signal += synapse.weight
        has_active_inputs = True

next_neuron.cumulative_signal = cumulative_signal
```

**Formula:**
```
cumulative_signal = Σ (weight_i × active_i)
```
where `active_i` is 1 if source neuron is active, 0 otherwise.

### Step 3: Update Elastic Parameters

**With active inputs (propagation):**
```python
next_neuron.elastic_trigger = current_neuron.elastic_trigger * ELASTIC_TRIGGER_PROPAGATION
next_neuron.elastic_recharge = current_neuron.elastic_recharge * ELASTIC_RECHARGE_PROPAGATION
```

**Without active inputs (degradation):**
```python
next_neuron.elastic_trigger = max(0, current_neuron.elastic_trigger - ELASTIC_TRIGGER_DEGRADATION)
next_neuron.elastic_recharge = max(0, current_neuron.elastic_recharge - ELASTIC_RECHARGE_DEGRADATION)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| ELASTIC_TRIGGER_PROPAGATION | 0.75 | Elastic trigger preserved at 75% with active input |
| ELASTIC_TRIGGER_DEGRADATION | 1 | Decay by 1 per iteration without input |
| ELASTIC_RECHARGE_PROPAGATION | 0.75 | Elastic recharge preserved at 75% |
| ELASTIC_RECHARGE_DEGRADATION | 1 | Decay by 1 per iteration |

### Step 4: Accumulate Charge
```python
next_neuron.charge = current_neuron.charge + RECHARGE_RATE + next_neuron.elastic_recharge
```

**Formula:**
```
charge_new = charge_old + RECHARGE_RATE + elastic_recharge
```

### Step 5: Activation Decision

Three-stage check in priority order:

**5a. Charge Minimum Check:**
```python
if next_neuron.charge < CHARGE_MIN:
    activated = False
```
Neuron cannot fire if charge is below minimum threshold.

**5b. Cyclic Discharge:**
```python
elif current_neuron.charge_cycle > 0 and iteration % current_neuron.charge_cycle == 0:
    activated = True
    next_neuron.charge = 0
```
Generator neurons fire periodically regardless of input.

**5c. Threshold Crossing:**
```python
elif cumulative_signal > (NORMAL_TRIGGER + next_neuron.elastic_trigger):
    activated = True
    next_neuron.charge = 0
```
Standard activation when cumulative input exceeds adjusted threshold.

**Effective threshold:**
```
threshold = NORMAL_TRIGGER + elastic_trigger
```
Elastic trigger can lower the effective threshold (when negative) or raise it (when positive).

### Step 6: Update History
```python
history = current_neuron.history + ('1' if activated else '0')
if len(history) > HISTORY_MAXLEN:
    history = history[-HISTORY_MAXLEN:]
next_neuron.history = history
```

Rolling window of activation states (default: 256 characters).

---

## Processing Summary

| Step | Operation | Formula/Condition |
|------|-----------|-------------------|
| 1 | Base Copy | Preserve EQ, name, charge_cycle |
| 2 | Signal Calc | `cumulative_signal = Σ(weight × active)` |
| 3 | Elastic Update | Propagate (×0.75) or decay (-1) |
| 4 | Charge Accum | `charge += RECHARGE_RATE + elastic_recharge` |
| 5a | Charge Check | `charge < CHARGE_MIN` → inactive |
| 5b | Cyclic Fire | `iteration % charge_cycle == 0` → active |
| 5c | Threshold | `signal > NORMAL_TRIGGER + elastic_trigger` → active |
| 6 | History | Append '1' or '0', trim to HISTORY_MAXLEN |

---

## Key Constants

### Charge Dynamics

| Parameter | Default | Purpose |
|-----------|---------|---------|
| CHARGE_MIN | 100 | Minimum charge required for firing |
| RECHARGE_RATE | 10 | Base charge accumulation per iteration |
| NORMAL_TRIGGER | 500 | Threshold for cumulative signal activation |
| DEFAULT_CUMULATIVE_INPUT | 1000 | Total synapse weight per neuron |

### Emotional Quantum

| Parameter | Default | Purpose |
|-----------|---------|---------|
| EQ_MAX | 255 | Maximum absolute EQ magnitude |
| DEFAULT_MIN_EQ | -3 | Random initialization minimum |
| DEFAULT_MAX_EQ | 3 | Random initialization maximum |
| STAR_LEVEL | 20 | Minimum |EQ| for star classification |

### History

| Parameter | Default | Purpose |
|-----------|---------|---------|
| HISTORY_MAXLEN | 256 | Maximum activation history length |

---

## Activation Flow Diagram

```
                    ┌─────────────────┐
                    │ Synapse Inputs  │
                    │ (active sources)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ cumulative_     │
                    │ signal = Σw     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ charge <        │ │ charge_cycle >0 │ │ signal >        │
│ CHARGE_MIN?     │ │ && iter % cycle │ │ threshold?      │
│                 │ │ == 0?           │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │ YES               │ YES               │ YES
         ▼                   ▼                   ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │ INACTIVE│         │ ACTIVE  │         │ ACTIVE  │
    │         │         │ charge=0│         │ charge=0│
    └─────────┘         └─────────┘         └─────────┘
```

---

## Elastic Mechanism

The elastic system creates adaptive thresholds:

1. **Elastic Trigger**: Modifies activation threshold
   - Positive: raises threshold (harder to fire)
   - Negative: lowers threshold (easier to fire)
   - Decays without active input

2. **Elastic Recharge**: Modifies charge accumulation
   - Positive: faster recharge
   - Negative: slower recharge
   - Decays without active input

This enables neurons to adapt their sensitivity based on recent activity patterns.
