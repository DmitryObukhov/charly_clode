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
    fatigue: float = 0.0
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
| `fatigue` | float | 0.0 | Accumulated fatigue level (suppresses recharge when high) |

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
    history="",
    fatigue=0.0
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

### Step 2: Fatigue Recovery
```python
next_neuron.fatigue = max(0.0, current_neuron.fatigue - FATIGUE_DECREMENT)
```

Fatigue decreases by `FATIGUE_DECREMENT` each iteration, recovering toward 0.

### Step 3: Calculate Cumulative Signal
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

### Step 4: Update Elastic Parameters

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

### Step 5: Accumulate Charge (with Fatigue Suppression)
```python
fatigue_ratio = current_neuron.fatigue / FATIGUE_MAX
recharge_multiplier = _apply_fatigue_curve(fatigue_ratio)
effective_recharge = (RECHARGE_RATE + next_neuron.elastic_recharge) * recharge_multiplier
next_neuron.charge = current_neuron.charge + effective_recharge
```

**Formula:**
```
fatigue_ratio = fatigue / FATIGUE_MAX
recharge_multiplier = _apply_fatigue_curve(fatigue_ratio)  # [0, 1]
charge_new = charge_old + (RECHARGE_RATE + elastic_recharge) × recharge_multiplier
```

High fatigue suppresses recharge, preventing rapid repeated firing.

### Step 6: Activation Decision

Three-stage check in priority order:

**6a. Charge Minimum Check:**
```python
if next_neuron.charge < CHARGE_MIN:
    activated = False
```
Neuron cannot fire if charge is below minimum threshold.

**6b. Cyclic Discharge:**
```python
elif current_neuron.charge_cycle > 0 and iteration % current_neuron.charge_cycle == 0:
    activated = True
    next_neuron.charge = 0
```
Generator neurons fire periodically regardless of input.

**6c. Threshold Crossing:**
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

### Step 7: Fatigue Increment
```python
if activated:
    next_neuron.fatigue = min(FATIGUE_MAX, next_neuron.fatigue + FATIGUE_INCREMENT)
```

Firing increases fatigue, which will suppress future recharge until recovery.

### Step 8: Update History
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
| 2 | Fatigue Recovery | `fatigue -= FATIGUE_DECREMENT` (min 0) |
| 3 | Signal Calc | `cumulative_signal = Σ(weight × active)` |
| 4 | Elastic Update | Propagate (×0.75) or decay (-1) |
| 5 | Charge Accum | `charge += (RECHARGE_RATE + elastic_recharge) × recharge_mult` |
| 6a | Charge Check | `charge < CHARGE_MIN` → inactive |
| 6b | Cyclic Fire | `iteration % charge_cycle == 0` → active |
| 6c | Threshold | `signal > NORMAL_TRIGGER + elastic_trigger` → active |
| 7 | Fatigue Increment | If activated: `fatigue += FATIGUE_INCREMENT` (max FATIGUE_MAX) |
| 8 | History | Append '1' or '0', trim to HISTORY_MAXLEN |

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

### Fatigue

| Parameter | Default | Purpose |
|-----------|---------|---------|
| FATIGUE_INCREMENT | 5 | Fatigue added on each activation |
| FATIGUE_DECREMENT | 1 | Fatigue recovered per iteration |
| FATIGUE_MAX | 100 | Maximum fatigue level |
| FATIGUE_SMOOTHNESS | 50 | Response curve shape (0-100, see below) |

### History

| Parameter | Default | Purpose |
|-----------|---------|---------|
| HISTORY_MAXLEN | 256 | Maximum activation history length |

---

## Fatigue System

The fatigue system prevents neurons from firing in rapid succession by suppressing recharge when fatigued.

### Response Curve: `_apply_fatigue_curve(fatigue_ratio)`

Converts fatigue level to a recharge multiplier using a configurable response curve.

**Input:** `fatigue_ratio` = fatigue / FATIGUE_MAX (range [0, 1])

**Output:** `recharge_multiplier` (range [0, 1])
- 0 = full suppression (no recharge)
- 1 = no suppression (full recharge)

**FATIGUE_SMOOTHNESS controls the curve shape:**

| Smoothness | Behavior |
|------------|----------|
| 0% | Binary: full recharge below 50% fatigue, no recharge above |
| 50% | Sigmoid: S-curve transition around 50% fatigue |
| 100% | Linear: proportional suppression |

**Formula (sigmoid mode):**
```python
k = 12.0 * (1.0 - smoothness)
x = (fatigue_ratio - 0.5) * k
sigmoid = 1.0 / (1.0 + exp(-x))
suppression = blend * fatigue_ratio + (1 - blend) * sigmoid
recharge_multiplier = 1.0 - suppression
```

### Fatigue Lifecycle

```
           ┌──────────────────┐
           │ Neuron Activates │
           └────────┬─────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │ fatigue += FATIGUE_INCREMENT│
      │ (capped at FATIGUE_MAX)     │
      └─────────────────────────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │ High fatigue suppresses     │
      │ recharge via curve function │
      └─────────────────────────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │ Each iteration:             │
      │ fatigue -= FATIGUE_DECREMENT│
      │ (until fatigue = 0)         │
      └─────────────────────────────┘
```

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
