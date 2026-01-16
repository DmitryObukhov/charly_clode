#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Charly Neural Substrate

Implements a neuromorphic cellular automaton that connects to a physical
model of reality. The neural substrate consists of neurons organized in
arrays with a connectome (synapse table) defining their connections.

The simulation runs in day/night cycles:
- Day: Sensory input processing and motor output generation
- Night: Learning and connectome modification

Usage:
    python charly.py --config ../config/config.yaml --days 1
"""

import random
import math
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from PIL import Image, ImageDraw
import numpy as np

from physical_model import PhysicalModel


@dataclass
class Neuron:
    """
    Represents a single neuron in the neural substrate.

    Attributes:
        name: Logical name for the neuron (e.g., "N42")
        active: Current activation status
        eq: Emotional quantum value [-EQ_MAX, +EQ_MAX]
        cumulative_signal: Sum of weighted inputs from active sources
        elastic_trigger: Elastic component of activation threshold
        charge: Current charge level
        elastic_recharge: Elastic component of recharge rate
        charge_cycle: Period for cyclic discharge (0 = disabled)
        history: String of '0' and '1' representing activation history
    """
    name: str = ""
    active: bool = False
    eq: int = 0
    cumulative_signal: float = 0.0
    elastic_trigger: float = 0.0
    charge: float = 0.0
    elastic_recharge: float = 0.0
    charge_cycle: int = 0
    history: str = ""

    def clone(self) -> 'Neuron':
        """Create a deep copy of the neuron."""
        return Neuron(
            name=self.name,
            active=self.active,
            eq=self.eq,
            cumulative_signal=self.cumulative_signal,
            elastic_trigger=self.elastic_trigger,
            charge=self.charge,
            elastic_recharge=self.elastic_recharge,
            charge_cycle=self.charge_cycle,
            history=self.history
        )


@dataclass
class Synapse:
    """
    Represents a synaptic connection between two neurons.

    Attributes:
        src: Source neuron index
        dst: Destination neuron index
        weight: Synaptic weight
    """
    src: int
    dst: int
    weight: float


class Connectome:
    """
    Manages synaptic connections between neurons.

    The connectome is indexed by both source and destination for efficient
    lookup during forward and backward propagation.
    """

    def __init__(self):
        """Initialize empty connectome."""
        self.synapses: List[Synapse] = []
        self._by_src: Dict[int, List[Synapse]] = defaultdict(list)
        self._by_dst: Dict[int, List[Synapse]] = defaultdict(list)

    def add(self, src: int, dst: int, weight: float) -> None:
        """Add a synapse to the connectome."""
        synapse = Synapse(src=src, dst=dst, weight=weight)
        self.synapses.append(synapse)
        self._by_src[src].append(synapse)
        self._by_dst[dst].append(synapse)

    def get_inputs(self, dst: int) -> List[Synapse]:
        """Get all synapses targeting a specific neuron."""
        return self._by_dst[dst]

    def get_outputs(self, src: int) -> List[Synapse]:
        """Get all synapses originating from a specific neuron."""
        return self._by_src[src]

    def remove_inputs(self, dst: int) -> None:
        """Remove all synapses targeting a specific neuron."""
        synapses_to_remove = self._by_dst[dst][:]
        for synapse in synapses_to_remove:
            self.synapses.remove(synapse)
            self._by_src[synapse.src].remove(synapse)
        self._by_dst[dst] = []

    def clear(self) -> None:
        """Clear all synapses."""
        self.synapses = []
        self._by_src = defaultdict(list)
        self._by_dst = defaultdict(list)

    def reindex(self) -> None:
        """Rebuild indices from synapse list."""
        self._by_src = defaultdict(list)
        self._by_dst = defaultdict(list)
        for synapse in self.synapses:
            self._by_src[synapse.src].append(synapse)
            self._by_dst[synapse.dst].append(synapse)


class Charly:
    """
    Neural substrate controller that connects neurons to a physical model.

    This class implements the neuromorphic cellular automaton with:
    - Neuron arrays (current and next state)
    - Connectome (synapse table)
    - Input receptors and output actuators
    - Day/night simulation cycles
    - Visualization
    """

    def __init__(self, config: Dict[str, Any], physical_model: PhysicalModel):
        """
        Initialize the Charly neural substrate.

        Args:
            config: Configuration dictionary
            physical_model: Physical model instance to connect to
        """
        self.config = config
        self.physical_model = physical_model
        self.log = print

        # Extract config values with defaults
        self.seed = config.get('SEED', 42)
        self.neuron_count = config.get('NEURON_COUNT', 1920)
        self.head_count = config.get('HEAD_COUNT', 200)
        self.charge_min = config.get('CHARGE_MIN', 100)
        self.recharge_rate = config.get('RECHARGE_RATE', 10)
        self.normal_trigger = config.get('NORMAL_TRIGGER', 500)
        self.eq_max = config.get('EQ_MAX', 255)
        self.elastic_trigger_prop = config.get('ELASTIC_TRIGGER_PROPAGATION', 0.75)
        self.elastic_trigger_deg = config.get('ELASTIC_TRIGGER_DEGRADATION', 1)
        self.elastic_recharge_prop = config.get('ELASTIC_RECHARGE_PROPAGATION', 0.75)
        self.elastic_recharge_deg = config.get('ELASTIC_RECHARGE_DEGRADATION', 1)
        self.history_maxlen = config.get('HISTORY_MAXLEN', 256)
        self.links_min = config.get('LINKS_PER_NEURON_MIN', 10)
        self.links_max = config.get('LINKS_PER_NEURON_MAX', 200)
        self.link_length_max = config.get('LINK_LENGTH_MAX', 200)
        self.default_cumulative_input = config.get('DEFAULT_CUMULATIVE_INPUT', 1000)
        self.star_level = config.get('STAR_LEVEL', 20)
        self.day_steps = config.get('DAY_STEPS', 1080)

        # Visualization colors
        self.pos_color = tuple(config.get('NEURON_POSITIVE_COLOR', [0, 255, 0]))
        self.neg_color = tuple(config.get('NEURON_NEGATIVE_COLOR', [255, 0, 0]))
        self.bg_color = tuple(config.get('NEURON_BG_COLOR', [0, 0, 0]))
        self.head_inactive_color = tuple(config.get('HEAD_INACTIVE_COLOR', [40, 40, 40]))

        # CES strip visualization
        self.ces_bg_color = tuple(config.get('CES_BG_COLOR', [0, 40, 0]))
        self.ces_esp_color = tuple(config.get('CES_ESP_COLOR', [144, 238, 144]))
        self.ces_esn_color = tuple(config.get('CES_ESN_COLOR', [255, 0, 0]))
        self.ces_grid_color = tuple(config.get('CES_GRID_COLOR', [40, 40, 40]))
        self.ces_grid_spacing = config.get('CES_GRID_SPACING', 10)

        # Physical strip visualization
        self.physical_agent_color = tuple(config.get('PHYSICAL_AGENT_COLOR', [0, 100, 255]))
        self.physical_lamp_color = tuple(config.get('PHYSICAL_LAMP_COLOR', [255, 100, 0]))

        # Initialize random generator
        self.rng = random.Random(self.seed)

        # Neural substrate arrays
        self.current: List[Neuron] = []
        self.next: List[Neuron] = []

        # Connectome
        self.connectome = Connectome()

        # Input/output mappings
        self.receptors: Dict[str, List[Dict]] = {}
        self.actuators: Dict[str, List[int]] = {}
        self.innates: Dict[str, Dict] = {}
        self.named: Dict[str, int] = {}

        # Star neurons
        self.stars: List[int] = []

        # Sauron's Finger - dynamic substrate modification
        self.finger_presses: List[Dict] = []

        # History
        self.day_history: List[Dict] = []
        self.iteration = 0
        self.day_index = 0

        # Verify and initialize
        self._verify_physical_model()
        self._init_substrate()
        self._configure_saurons_finger()

        self.log(f"Charly initialized: {self.neuron_count} neurons, head={self.head_count}")

    def set_logger(self, log_func) -> None:
        """Replace the default logging function."""
        self.log = log_func

    def _verify_physical_model(self) -> None:
        """Verify that physical model is compatible with configuration."""
        inputs, outputs = self.physical_model.get_names()

        for input_cfg in self.config.get('inputs', []):
            signal = input_cfg.get('signal')
            if signal and signal not in inputs:
                self.log(f"Warning: Input signal '{signal}' not available in physical model")

        actuator_configs = self.config.get('actuators', [])
        if isinstance(actuator_configs, list):
            for actuator_cfg in actuator_configs:
                if actuator_cfg:
                    name = actuator_cfg.get('name', '')
                    if name and name not in outputs:
                        self.log(f"Warning: Actuator '{name}' not available in physical model")
        elif isinstance(actuator_configs, dict):
            for actuator_name in actuator_configs.keys():
                if actuator_name not in outputs:
                    self.log(f"Warning: Actuator '{actuator_name}' not available in physical model")

    def _init_substrate(self) -> None:
        """Initialize the neural substrate with default values."""
        min_eq = self.config.get('DEFAULT_MIN_EQ', -3)
        max_eq = self.config.get('DEFAULT_MAX_EQ', 3)

        for i in range(self.neuron_count):
            neuron = Neuron(
                name=f"N{i}",
                active=False,
                eq=self.rng.randint(min_eq, max_eq),
                cumulative_signal=0.0,
                elastic_trigger=0.0,
                charge=0.0,
                elastic_recharge=0.0,
                charge_cycle=0,
                history=""
            )
            self.current.append(neuron)
            self.next.append(neuron.clone())

        self._build_connectome()
        self._configure_receptors()
        self._configure_actuators()
        self._configure_innates()
        self._configure_named()
        self._identify_stars()

    def _build_connectome(self) -> None:
        """Build the connectome for all neurons with configurable link distributions."""
        self.log("Building connectome...")

        # Link length distribution parameters
        short_range_pct = self.config.get('LINK_LEN_SHORT_RANGE', 10) / 100.0
        long_range_pct = self.config.get('LINK_LEN_LONG_RANGE', 10) / 100.0
        default_dist = self.config.get('LINK_DISTRIBUTION_DEFAULT', [80, 15, 5])

        # Calculate distance thresholds
        short_max = int(self.link_length_max * short_range_pct)
        long_min = int(self.link_length_max * (1.0 - long_range_pct))

        # Build distribution lookup from segments
        dist_segments = self.config.get('connectome_distribution', [])

        def get_distribution(neuron_idx: int) -> List[int]:
            """Get [short%, mid%, long%] distribution for a neuron index."""
            for seg in dist_segments:
                if seg['start'] <= neuron_idx < seg['end']:
                    return seg.get('distribution', default_dist)
            return default_dist

        for dst_idx in range(self.head_count, self.neuron_count):
            n_links = self.rng.randint(self.links_min, self.links_max)

            # Get distribution for this neuron
            dist = get_distribution(dst_idx)
            short_pct, mid_pct, long_pct = dist[0] / 100.0, dist[1] / 100.0, dist[2] / 100.0

            # Categorize candidates by distance band
            short_candidates = []  # 0 to short_max
            mid_candidates = []    # short_max to long_min
            long_candidates = []   # long_min to link_length_max

            for src_idx in range(self.neuron_count):
                if src_idx == dst_idx:
                    continue
                distance = abs(src_idx - dst_idx)
                if distance > self.link_length_max:
                    continue
                if distance <= short_max:
                    short_candidates.append(src_idx)
                elif distance >= long_min:
                    long_candidates.append(src_idx)
                else:
                    mid_candidates.append(src_idx)

            # Calculate target count for each band
            n_short = int(n_links * short_pct)
            n_long = int(n_links * long_pct)
            n_mid = n_links - n_short - n_long

            # Select from each band (cap at available candidates)
            selected = []

            if short_candidates:
                n_short_actual = min(n_short, len(short_candidates))
                selected.extend(self.rng.sample(short_candidates, n_short_actual))

            if mid_candidates:
                n_mid_actual = min(n_mid, len(mid_candidates))
                selected.extend(self.rng.sample(mid_candidates, n_mid_actual))

            if long_candidates:
                n_long_actual = min(n_long, len(long_candidates))
                selected.extend(self.rng.sample(long_candidates, n_long_actual))

            if not selected:
                continue

            # Assign weights
            raw_weights = [self.rng.random() for _ in selected]
            total_raw = sum(raw_weights)
            if total_raw > 0:
                weights = [w / total_raw * self.default_cumulative_input for w in raw_weights]
            else:
                weights = [self.default_cumulative_input / len(selected)] * len(selected)

            for src_idx, weight in zip(selected, weights):
                self.connectome.add(src_idx, dst_idx, weight)

        self.log(f"Connectome built: {len(self.connectome.synapses)} synapses")

    def _configure_receptors(self) -> None:
        """Configure input receptors from config.

        Supports two modes:
        - 'population': Spread signal across N neurons using Stevens' power law
        - 'range' (legacy): Activate neurons when signal is in a value range
        """
        for input_cfg in self.config.get('inputs', []):
            signal = input_cfg.get('signal')
            if not signal:
                continue

            input_type = input_cfg.get('type', 'range')

            if input_type == 'population':
                # Population coding with Stevens' power law
                neuron_start = input_cfg.get('neuron_start', 0)
                neuron_count = input_cfg.get('neuron_count', 50)
                gamma = input_cfg.get('gamma', 0.4)
                eq_gradient = input_cfg.get('eq_gradient', {'low': -20, 'high': 20})

                # Configure neurons in the population
                eq_low = eq_gradient.get('low', -20)
                eq_high = eq_gradient.get('high', 20)

                for i in range(neuron_count):
                    idx = neuron_start + i
                    if 0 <= idx < self.neuron_count:
                        # Linear interpolation of EQ across the population
                        t = i / (neuron_count - 1) if neuron_count > 1 else 0.5
                        eq = int(eq_low + t * (eq_high - eq_low))
                        name = f"{signal}_{i}"

                        self.current[idx].eq = eq
                        self.current[idx].name = name
                        self.next[idx].eq = eq
                        self.next[idx].name = name

                self.receptors[signal] = {
                    'type': 'population',
                    'neuron_start': neuron_start,
                    'neuron_count': neuron_count,
                    'gamma': gamma,
                    'eq_gradient': eq_gradient
                }
                self.log(f"Configured population receptor '{signal}': {neuron_count} neurons, gamma={gamma}")

            else:
                # Legacy range-based activation
                if signal not in self.receptors:
                    self.receptors[signal] = []

                receptor_config = {
                    'type': 'range',
                    'min_val': input_cfg.get('min_val', 0.0),
                    'max_val': input_cfg.get('max_val', 1.0),
                    'neurons': []
                }

                for neuron_cfg in input_cfg.get('neurons', []):
                    idx = neuron_cfg.get('idx')
                    if idx is not None and 0 <= idx < self.neuron_count:
                        eq = neuron_cfg.get('eq', self.current[idx].eq)
                        name = neuron_cfg.get('name', f"receptor_{signal}_{idx}")

                        self.current[idx].eq = eq
                        self.current[idx].name = name
                        self.next[idx].eq = eq
                        self.next[idx].name = name

                        receptor_config['neurons'].append(idx)

                if isinstance(self.receptors.get(signal), list):
                    self.receptors[signal].append(receptor_config)
                else:
                    self.receptors[signal] = [receptor_config]

        self.log(f"Configured {len(self.receptors)} receptor signals")

    def _configure_actuators(self) -> None:
        """Configure output actuators from config.

        New format: list of {name, start, count, default_eq, color}
        Each actuator gets neurons from start to start+count-1 with assigned EQ.
        """
        actuator_configs = self.config.get('actuators', [])

        # Handle new list format
        if isinstance(actuator_configs, list):
            for actuator_cfg in actuator_configs:
                if not actuator_cfg:
                    continue

                name = actuator_cfg.get('name', 'unnamed')
                start = actuator_cfg.get('start', 0)
                count = actuator_cfg.get('count', 10)
                default_eq = actuator_cfg.get('default_eq', 0)
                color = tuple(actuator_cfg.get('color', [255, 255, 255]))

                # Build list of neuron indices
                indices = []
                for i in range(count):
                    idx = start + i
                    if 0 <= idx < self.neuron_count:
                        indices.append(idx)
                        # Assign EQ to the neuron
                        self.current[idx].eq = default_eq
                        self.current[idx].name = f"{name}_{i}"
                        self.next[idx].eq = default_eq
                        self.next[idx].name = f"{name}_{i}"

                # Wire input connections if specified (adds to existing connections)
                input_cfg = actuator_cfg.get('inputs', None)
                if input_cfg:
                    input_start = input_cfg.get('start', 0)
                    input_count = input_cfg.get('count', 100)
                    weight = input_cfg.get('weight', 100)

                    # Add inputs from source range to motor neurons
                    n_connections = 0
                    for dst_idx in indices:
                        for src_offset in range(input_count):
                            src_idx = input_start + src_offset
                            if 0 <= src_idx < self.neuron_count and src_idx != dst_idx:
                                self.connectome.add(src_idx, dst_idx, weight)
                                n_connections += 1

                    self.log(f"Wired {name}: {n_connections} connections from body")

                self.actuators[name] = {
                    'indices': indices,
                    'color': color,
                    'start': start,
                    'count': count
                }

        # Handle legacy dict format for backwards compatibility
        elif isinstance(actuator_configs, dict):
            for actuator_name, neuron_indices in actuator_configs.items():
                if isinstance(neuron_indices, list):
                    valid_indices = [idx for idx in neuron_indices if 0 <= idx < self.neuron_count]
                    self.actuators[actuator_name] = {
                        'indices': valid_indices,
                        'color': (255, 255, 255),
                        'start': valid_indices[0] if valid_indices else 0,
                        'count': len(valid_indices)
                    }

        self.log(f"Configured {len(self.actuators)} actuators")

    def _configure_innates(self) -> None:
        """Configure innate generator neurons from config."""
        for name, innate_cfg in self.config.get('innates', {}).items():
            idx = innate_cfg.get('idx')
            if idx is None or not (0 <= idx < self.neuron_count):
                continue

            eq = innate_cfg.get('eq', 0)
            charge_cycle = innate_cfg.get('charge_cycle', 0)
            input_connections = innate_cfg.get('inputs', [])

            self.current[idx].eq = eq
            self.current[idx].charge_cycle = charge_cycle
            self.current[idx].name = name
            self.next[idx].eq = eq
            self.next[idx].charge_cycle = charge_cycle
            self.next[idx].name = name

            if input_connections:
                self.connectome.remove_inputs(idx)
                for src_idx, weight in input_connections:
                    if 0 <= src_idx < self.neuron_count:
                        self.connectome.add(src_idx, idx, weight)

            self.innates[name] = {
                'idx': idx,
                'eq': eq,
                'charge_cycle': charge_cycle
            }

        self.log(f"Configured {len(self.innates)} innate generators")

    def _configure_named(self) -> None:
        """Configure named neuron aliases from config."""
        for name, idx in self.config.get('named', {}).items():
            if 0 <= idx < self.neuron_count:
                self.named[name] = idx

        self.log(f"Configured {len(self.named)} named neurons")

    def _identify_stars(self) -> None:
        """Identify star neurons (high-EQ generators)."""
        self.stars = []
        for idx, neuron in enumerate(self.current):
            if abs(neuron.eq) >= self.star_level:
                self.stars.append(idx)

        self.log(f"Identified {len(self.stars)} star neurons")

    def _configure_saurons_finger(self) -> None:
        """Configure Sauron's Finger presses from config."""
        finger_configs = self.config.get('saurons_finger', [])
        if not finger_configs:
            return

        for cfg in finger_configs:
            if not cfg:  # Skip empty/commented entries
                continue

            finger_press = {
                'name': cfg.get('name', 'unnamed'),
                'x': cfg.get('x', 0),
                'r': cfg.get('r', 10),
                'field': cfg.get('field', 'elastic_trigger'),
                'formula': cfg.get('formula', 'PREV'),
                'shape': cfg.get('shape', '1.0'),
                'rows': cfg.get('rows', [])
            }
            self.finger_presses.append(finger_press)

        if self.finger_presses:
            self.log(f"Configured {len(self.finger_presses)} Sauron's Finger presses")

    def _eval_finger_formula(self, formula: str, context: Dict[str, float]) -> float:
        """
        Safely evaluate a finger formula with the given context.

        Args:
            formula: Formula string to evaluate
            context: Dictionary of variable names to values (IDX, X, R, DIST, PREV, etc.)

        Returns:
            Evaluated result as float
        """
        # Safe math functions
        safe_dict = {
            'exp': math.exp,
            'sin': math.sin,
            'cos': math.cos,
            'sqrt': math.sqrt,
            'abs': abs,
            'min': min,
            'max': max,
            'pi': math.pi,
            'e': math.e,
        }
        # Add context variables
        safe_dict.update(context)

        try:
            return float(eval(formula, {"__builtins__": {}}, safe_dict))
        except Exception:
            return 0.0

    def _apply_saurons_finger(self) -> None:
        """Apply active finger presses to the substrate using formula evaluation."""
        for finger in self.finger_presses:
            x = finger['x']
            r = finger['r']
            field = finger['field']
            formula = finger['formula']
            shape_formula = finger['shape']
            rows = finger['rows']

            # Check if any row is active at current iteration
            temporal = 0.0
            for row in rows:
                start = row.get('start', 0)
                end = row.get('end', self.day_steps)
                if start <= self.iteration <= end:
                    row_formula = row.get('formula', '1.0')
                    temporal = self._eval_finger_formula(row_formula, {'ROW': self.iteration})
                    break

            if temporal == 0.0:
                continue

            # Apply to neurons in range [x-r, x+r]
            for idx in range(max(0, x - r), min(self.neuron_count, x + r + 1)):
                neuron = self.current[idx]
                dist = abs(idx - x)

                # Get previous field value
                prev = getattr(neuron, field, 0.0)

                # Evaluate shape formula
                shape_context = {
                    'IDX': idx,
                    'X': x,
                    'R': r,
                    'DIST': dist,
                    'PREV': prev,
                }
                shape_value = self._eval_finger_formula(shape_formula, shape_context)

                # Evaluate main formula
                main_context = {
                    'IDX': idx,
                    'X': x,
                    'R': r,
                    'DIST': dist,
                    'PREV': prev,
                    'SHAPE': shape_value,
                    'TEMPORAL': temporal,
                }
                new_value = self._eval_finger_formula(formula, main_context)

                # Apply to the specified field
                if field == 'elastic_trigger':
                    neuron.elastic_trigger = new_value
                elif field == 'elastic_recharge':
                    neuron.elastic_recharge = new_value
                elif field == 'charge':
                    neuron.charge = new_value
                elif field == 'cumulative_signal':
                    neuron.cumulative_signal = new_value

    def _process_inputs(self) -> None:
        """Process sensory inputs and activate receptor neurons.

        Supports two modes:
        - 'population': Stevens' power law with thermometer coding
        - 'range' (legacy): Binary activation based on value ranges
        """
        sensor_names = list(self.receptors.keys())
        if not sensor_names:
            return

        sensor_values = self.physical_model.get(sensor_names)

        for signal, receptor_cfg in self.receptors.items():
            value = sensor_values.get(signal, 0.0)

            if isinstance(receptor_cfg, dict) and receptor_cfg.get('type') == 'population':
                # Population coding with Stevens' power law
                gamma = receptor_cfg.get('gamma', 0.4)
                neuron_start = receptor_cfg['neuron_start']
                neuron_count = receptor_cfg['neuron_count']

                # Stevens' power law: perceived = physical^gamma
                # Clamp value to [0, 1] range
                value = max(0.0, min(1.0, value))
                perceived = value ** gamma

                # Calculate number of active neurons (thermometer code)
                n_active = round(neuron_count * perceived)

                # Activate neurons from start to start + n_active - 1
                for i in range(n_active):
                    idx = neuron_start + i
                    if 0 <= idx < self.neuron_count:
                        self.current[idx].active = True

            elif isinstance(receptor_cfg, list):
                # Legacy range-based activation
                for cfg in receptor_cfg:
                    if cfg.get('type') == 'range' or 'min_val' in cfg:
                        min_val = cfg.get('min_val', 0.0)
                        max_val = cfg.get('max_val', 1.0)

                        if min_val <= value <= max_val:
                            for idx in cfg.get('neurons', []):
                                self.current[idx].active = True

    def _process_innates(self) -> None:
        """Process innate generator neurons."""
        for name, innate_cfg in self.innates.items():
            idx = innate_cfg['idx']
            charge_cycle = innate_cfg.get('charge_cycle', 0)

            if charge_cycle > 0 and self.iteration % charge_cycle == 0:
                self.current[idx].active = True

    def _calculate_actuator_outputs(self) -> Dict[str, float]:
        """Calculate actuator output values based on neuron activity."""
        outputs = {}

        for actuator_name, actuator_cfg in self.actuators.items():
            indices = actuator_cfg.get('indices', [])
            if not indices:
                outputs[actuator_name] = 0.0
                continue

            active_count = sum(1 for idx in indices if self.current[idx].active)
            outputs[actuator_name] = active_count / len(indices)

        return outputs

    def _process_neuron(self, idx: int) -> None:
        """Process a single neuron and update its state in the next array."""
        current_neuron = self.current[idx]
        next_neuron = self.next[idx]

        # Step 1: Copy base values
        next_neuron.eq = current_neuron.eq
        next_neuron.name = current_neuron.name
        next_neuron.charge_cycle = current_neuron.charge_cycle

        # Step 2: Calculate cumulative signal
        inputs = self.connectome.get_inputs(idx)
        cumulative_signal = 0.0
        has_active_inputs = False

        for synapse in inputs:
            src_neuron = self.current[synapse.src]
            if src_neuron.active:
                cumulative_signal += synapse.weight
                has_active_inputs = True

        next_neuron.cumulative_signal = cumulative_signal

        # Step 3: Update elastic parameters
        if has_active_inputs:
            next_neuron.elastic_trigger = current_neuron.elastic_trigger * self.elastic_trigger_prop
            next_neuron.elastic_recharge = current_neuron.elastic_recharge * self.elastic_recharge_prop
        else:
            next_neuron.elastic_trigger = max(0, current_neuron.elastic_trigger - self.elastic_trigger_deg)
            next_neuron.elastic_recharge = max(0, current_neuron.elastic_recharge - self.elastic_recharge_deg)

        # Step 4: Update charge
        next_neuron.charge = current_neuron.charge + self.recharge_rate + next_neuron.elastic_recharge

        # Step 5: Determine activation
        activated = False

        if next_neuron.charge < self.charge_min:
            activated = False
        elif current_neuron.charge_cycle > 0 and self.iteration % current_neuron.charge_cycle == 0:
            activated = True
            next_neuron.charge = 0
        elif cumulative_signal > (self.normal_trigger + next_neuron.elastic_trigger):
            activated = True
            next_neuron.charge = 0

        next_neuron.active = activated

        # Step 6: Update history
        history = current_neuron.history + ('1' if activated else '0')
        if len(history) > self.history_maxlen:
            history = history[-self.history_maxlen:]
        next_neuron.history = history

    def _calculate_emotional_state(self) -> Tuple[int, int]:
        """Calculate the total emotional state from active neurons."""
        esp = 0
        esn = 0

        for neuron in self.next:
            if neuron.active:
                if neuron.eq > 0:
                    esp += neuron.eq
                elif neuron.eq < 0:
                    esn += neuron.eq

        return esp, esn

    def _collect_star_values(self) -> Dict[int, Tuple[bool, int]]:
        """Collect the activation status and EQ of star neurons."""
        return {idx: (self.current[idx].active, self.current[idx].eq) for idx in self.stars}

    def day_step(self, actuator_values: Dict[str, float] = None) -> Dict[str, Any]:
        """Execute one step of the day phase."""
        if actuator_values is None:
            actuator_values = {name: 0.0 for name in self.actuators}
        self.physical_model.set(actuator_values)

        sensor_names = list(self.receptors.keys())
        sensor_values = self.physical_model.get(sensor_names) if sensor_names else {}
        # Also get physical state for visualization
        physical_state = self.physical_model.get(['agent_pos', 'lamp_pos'])
        sensor_values.update(physical_state)

        for i in range(self.neuron_count):
            self.next[i] = self.current[i].clone()
            self.next[i].active = False

        for i in range(self.head_count):
            self.current[i].active = False
            self.current[i].cumulative_signal = 0.0

        self._process_inputs()
        self._process_innates()
        self._apply_saurons_finger()

        # Copy head neuron activations to next array so they're preserved
        # and included in emotional state calculation
        for i in range(self.head_count):
            self.next[i].active = self.current[i].active

        star_values = self._collect_star_values()

        for idx in range(self.head_count, self.neuron_count):
            self._process_neuron(idx)

        esp, esn = self._calculate_emotional_state()

        for i in range(self.neuron_count):
            self.current[i] = self.next[i].clone()

        new_actuator_values = self._calculate_actuator_outputs()

        step_record = {
            'iteration': self.iteration,
            'esp': esp,
            'esn': esn,
            'inputs': sensor_values,
            'outputs': new_actuator_values,
            'stars': star_values,
            'neurons': [n.clone() for n in self.current]
        }
        self.day_history.append(step_record)

        self.iteration += 1

        return {
            'esp': esp,
            'esn': esn,
            'inputs': sensor_values,
            'outputs': new_actuator_values,
            'stars': star_values
        }

    def run_day(self) -> List[Dict[str, Any]]:
        """Run a complete day phase."""
        self.log(f"Starting day {self.day_index}...")
        self.day_history = []

        actuator_values = None
        results = []

        for step in range(self.day_steps):
            result = self.day_step(actuator_values)
            results.append(result)
            actuator_values = result['outputs']

            if (step + 1) % 100 == 0:
                self.log(f"  Step {step+1}/{self.day_steps}: ESP={result['esp']}, ESN={result['esn']}")

        return results

    def run_night(self) -> None:
        """Run the night phase (learning/connectome modification)."""
        self.log(f"Night phase for day {self.day_index}...")

        for step_record in self.day_history:
            stars = step_record.get('stars', {})
            for star_idx, (was_active, eq) in stars.items():
                if was_active:
                    inputs = self.connectome.get_inputs(star_idx)
                    for synapse in inputs:
                        if eq > 0:
                            synapse.weight *= 1.01
                        elif eq < 0:
                            synapse.weight *= 0.99

        self.connectome.reindex()
        self.day_index += 1

    def visualize(self,
                  width: Optional[int] = None,
                  height: Optional[int] = None,
                  ces_strip_width: int = 0,
                  actuator_strip_width: int = 0,
                  physical_strip_width: int = 0) -> Image.Image:
        """Create visualization of the day's neural activity.

        The neural diagram is scaled to fit width/height, while the CES,
        actuator, and physical strips are rendered at their configured widths.
        """
        if not self.day_history:
            self.log("No history to visualize")
            return Image.new('RGB', (width or 100, height or 100), self.bg_color)

        h_raw = len(self.day_history)
        # Count separators between strips
        n_separators = 0
        if actuator_strip_width > 0:
            n_separators += 1
        if physical_strip_width > 0:
            n_separators += 1
        if ces_strip_width > 0:
            n_separators += 1
        total_strip_width = ces_strip_width + actuator_strip_width + physical_strip_width + n_separators

        # Collect actuator neuron indices for special rendering
        actuator_indices = set()
        for actuator_cfg in self.actuators.values():
            actuator_indices.update(actuator_cfg.get('indices', []))

        # Create neural activity image (without strips)
        neural_array = np.zeros((h_raw, self.neuron_count, 3), dtype=np.uint8)
        neural_array[:, :] = self.bg_color

        for y, step_record in enumerate(self.day_history):
            neurons = step_record.get('neurons', [])

            for x, neuron in enumerate(neurons):
                if x >= self.neuron_count:
                    break

                if neuron.active:
                    eq = neuron.eq
                    brightness = min(1.0, abs(eq) / self.eq_max) if self.eq_max > 0 else 1.0
                    brightness = max(0.2, brightness)

                    if eq > 0:
                        color = tuple(int(c * brightness) for c in self.pos_color)
                    else:
                        color = tuple(int(c * brightness) for c in self.neg_color)

                    neural_array[y, x] = color
                elif x < self.head_count:
                    # Show inactive head neurons as dark grey
                    neural_array[y, x] = self.head_inactive_color
                elif x in actuator_indices:
                    # Show inactive actuator neurons as dark grey (like head)
                    neural_array[y, x] = self.head_inactive_color

        neural_image = Image.fromarray(neural_array, 'RGB')

        # Scale neural image if dimensions specified
        if width is not None or height is not None:
            # Account for both strips when calculating neural diagram width
            target_width = (width or neural_image.width) - total_strip_width
            target_height = height or neural_image.height
            neural_image = neural_image.resize((target_width, target_height), Image.Resampling.NEAREST)

        # If no strips, return scaled neural image
        if total_strip_width <= 0:
            return neural_image

        # Create CES strip
        ces_image = None
        if ces_strip_width > 0:
            esps = [rec['esp'] for rec in self.day_history]
            esns = [abs(rec['esn']) for rec in self.day_history]
            max_esp = max(esps) if esps else 1
            max_esn = max(esns) if esns else 1

            ces_array = np.zeros((h_raw, ces_strip_width, 3), dtype=np.uint8)
            ces_array[:, :] = self.ces_bg_color

            for y, step_record in enumerate(self.day_history):
                esp = step_record['esp']
                esn = abs(step_record['esn'])

                esp_normalized = esp / max_esp if max_esp > 0 else 0
                esn_normalized = esn / max_esn if max_esn > 0 else 0

                # Draw grid lines (vertical lines at regular intervals)
                for gx in range(0, ces_strip_width, self.ces_grid_spacing):
                    ces_array[y, gx] = self.ces_grid_color

                # Draw ESP as light green dot (position indicates value)
                half_width = ces_strip_width // 2
                esp_x = int(esp_normalized * (half_width - 1))
                ces_array[y, esp_x] = self.ces_esp_color

                # Draw ESN as red dot (position indicates value)
                esn_x = half_width + int(esn_normalized * (half_width - 1))
                ces_array[y, esn_x] = self.ces_esn_color

            ces_image = Image.fromarray(ces_array, 'RGB')
            if ces_image.height != neural_image.height:
                ces_image = ces_image.resize((ces_strip_width, neural_image.height), Image.Resampling.NEAREST)

        # Create actuator strip
        actuator_image = None
        if actuator_strip_width > 0 and self.actuators:
            actuator_array = np.zeros((h_raw, actuator_strip_width, 3), dtype=np.uint8)
            actuator_array[:, :] = self.ces_bg_color  # Same background as CES

            # Calculate max output for normalization
            actuator_names = list(self.actuators.keys())
            max_outputs = {}
            for name in actuator_names:
                outputs = [rec['outputs'].get(name, 0.0) for rec in self.day_history]
                max_outputs[name] = max(outputs) if outputs else 1.0

            # Divide strip width among actuators
            n_actuators = len(actuator_names)
            strip_per_actuator = actuator_strip_width // max(n_actuators, 1)

            for y, step_record in enumerate(self.day_history):
                outputs = step_record.get('outputs', {})

                # Draw grid lines
                for gx in range(0, actuator_strip_width, self.ces_grid_spacing):
                    actuator_array[y, gx] = self.ces_grid_color

                # Draw each actuator's output
                for i, name in enumerate(actuator_names):
                    actuator_cfg = self.actuators[name]
                    color = actuator_cfg.get('color', (255, 255, 255))
                    output_val = outputs.get(name, 0.0)
                    max_val = max_outputs.get(name, 1.0)

                    normalized = output_val / max_val if max_val > 0 else 0

                    # Position within this actuator's portion of the strip
                    strip_start = i * strip_per_actuator
                    x_pos = strip_start + int(normalized * (strip_per_actuator - 1))
                    if 0 <= x_pos < actuator_strip_width:
                        actuator_array[y, x_pos] = color

            actuator_image = Image.fromarray(actuator_array, 'RGB')
            if actuator_image.height != neural_image.height:
                actuator_image = actuator_image.resize((actuator_strip_width, neural_image.height), Image.Resampling.NEAREST)

        # Create physical strip (agent and lamp positions)
        physical_image = None
        if physical_strip_width > 0:
            physical_array = np.zeros((h_raw, physical_strip_width, 3), dtype=np.uint8)
            physical_array[:, :] = self.ces_bg_color

            for y, step_record in enumerate(self.day_history):
                inputs = step_record.get('inputs', {})
                agent_pos = inputs.get('agent_pos', 0.5)
                lamp_pos = inputs.get('lamp_pos', 0.5)

                # Draw grid lines
                for gx in range(0, physical_strip_width, self.ces_grid_spacing):
                    physical_array[y, gx] = self.ces_grid_color

                # Draw agent position (blue) - proportionally mapped
                agent_x = int(agent_pos * (physical_strip_width - 1))
                agent_x = max(0, min(physical_strip_width - 1, agent_x))
                physical_array[y, agent_x] = self.physical_agent_color

                # Draw lamp position (orange) - proportionally mapped
                lamp_x = int(lamp_pos * (physical_strip_width - 1))
                lamp_x = max(0, min(physical_strip_width - 1, lamp_x))
                physical_array[y, lamp_x] = self.physical_lamp_color

            physical_image = Image.fromarray(physical_array, 'RGB')
            if physical_image.height != neural_image.height:
                physical_image = physical_image.resize((physical_strip_width, neural_image.height), Image.Resampling.NEAREST)

        # Combine: neural + separator + actuator + separator + physical + separator + CES
        final_width = neural_image.width + total_strip_width
        final_image = Image.new('RGB', (final_width, neural_image.height), self.bg_color)
        final_image.paste(neural_image, (0, 0))

        x_offset = neural_image.width
        separator_color = (255, 255, 255)  # White separator

        if actuator_image:
            # Draw 1px white separator before actuator strip
            for y in range(neural_image.height):
                final_image.putpixel((x_offset, y), separator_color)
            x_offset += 1
            final_image.paste(actuator_image, (x_offset, 0))
            x_offset += actuator_strip_width

        if physical_image:
            # Draw 1px white separator before physical strip
            for y in range(neural_image.height):
                final_image.putpixel((x_offset, y), separator_color)
            x_offset += 1
            final_image.paste(physical_image, (x_offset, 0))
            x_offset += physical_strip_width

        if ces_image:
            # Draw 1px white separator before CES strip
            for y in range(neural_image.height):
                final_image.putpixel((x_offset, y), separator_color)
            x_offset += 1
            final_image.paste(ces_image, (x_offset, 0))

        return final_image

    def save_visualization(self,
                          filename: str,
                          width: Optional[int] = None,
                          height: Optional[int] = None,
                          ces_strip_width: Optional[int] = None,
                          actuator_strip_width: Optional[int] = None,
                          physical_strip_width: Optional[int] = None) -> str:
        """Save visualization to a PNG file."""
        if ces_strip_width is None:
            ces_strip_width = self.config.get('CES_STRIP_WIDTH', 100)
        if actuator_strip_width is None:
            actuator_strip_width = self.config.get('ACTUATOR_STRIP_WIDTH', 100)
        if physical_strip_width is None:
            physical_strip_width = self.config.get('PHYSICAL_STRIP_WIDTH', 100)
        img = self.visualize(width, height,
                            ces_strip_width=ces_strip_width,
                            actuator_strip_width=actuator_strip_width,
                            physical_strip_width=physical_strip_width)
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        img.save(filename, 'PNG')
        self.log(f"Visualization saved to {filename}")
        return filename

    def faceshot(self,
                 w: int,
                 h: int,
                 filename: Optional[str] = None) -> Image.Image:
        """
        Visualize the connectome structure on a circular diagram.

        Neurons are arranged on a circle, colored by EQ (green=positive,
        red=negative). Connections shown as lines with color by weight sign
        and brightness by weight magnitude.

        Args:
            w: Output image width in pixels
            h: Output image height in pixels
            filename: Optional path to save PNG

        Returns:
            PIL Image object
        """
        img = Image.new('RGB', (w, h), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Calculate circle parameters
        center_x = w // 2
        center_y = h // 2
        radius = min(w, h) // 2 - 20

        # Calculate neuron positions on circle
        positions = []
        for i in range(self.neuron_count):
            angle = 2 * math.pi * i / self.neuron_count - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))

        # Find max weight for normalization
        max_weight = max((abs(s.weight) for s in self.connectome.synapses), default=1)

        # Draw connections first (behind neurons)
        for synapse in self.connectome.synapses:
            src_pos = positions[synapse.src]
            dst_pos = positions[synapse.dst]

            # Color by weight sign, brightness by magnitude
            brightness = min(1.0, abs(synapse.weight) / max_weight)
            brightness = max(0.1, brightness)

            if synapse.weight >= 0:
                color = (0, int(255 * brightness), 0)  # Green
            else:
                color = (int(255 * brightness), 0, 0)  # Red

            draw.line([src_pos, dst_pos], fill=color, width=1)

        # Draw neurons on top
        neuron_radius = max(2, min(w, h) // 400)

        for i, (x, y) in enumerate(positions):
            eq = self.current[i].eq

            # Color by EQ
            if eq > 0:
                brightness = min(1.0, abs(eq) / self.eq_max) if self.eq_max > 0 else 1.0
                brightness = max(0.3, brightness)
                color = (0, int(255 * brightness), 0)  # Green
            elif eq < 0:
                brightness = min(1.0, abs(eq) / self.eq_max) if self.eq_max > 0 else 1.0
                brightness = max(0.3, brightness)
                color = (int(255 * brightness), 0, 0)  # Red
            else:
                color = (128, 128, 128)  # Gray for zero EQ

            draw.ellipse(
                [x - neuron_radius, y - neuron_radius,
                 x + neuron_radius, y + neuron_radius],
                fill=color
            )

        # Save if filename provided
        if filename:
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            img.save(filename, 'PNG')
            self.log(f"Faceshot saved to {filename}")

        return img


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Charly Neural Substrate")
    parser.add_argument('--config', '-c', type=str, default='../config/config.yaml',
                       help='Configuration file')
    parser.add_argument('--days', '-d', type=int, default=1,
                       help='Number of days to simulate')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory')

    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    from model_linear import Linear
    physical_model = Linear(world_size=config.get('WORLD_SIZE', 1920))

    charly = Charly(config, physical_model)

    os.makedirs(args.output, exist_ok=True)

    for day in range(args.days):
        print(f"\n=== Day {day + 1} ===")
        results = charly.run_day()
        charly.run_night()

        filename = os.path.join(args.output, f"day_{day:04d}.png")
        charly.save_visualization(filename, width=1920, height=1080)

        faceshot_file = os.path.join(args.output, f"faceshot_{day:04d}.png")
        charly.faceshot(1024, 1024, faceshot_file)

    print("\nSimulation complete!")
