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

        for actuator_name in self.config.get('actuators', {}).keys():
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
        """Build the connectome for all neurons."""
        self.log("Building connectome...")

        for dst_idx in range(self.head_count, self.neuron_count):
            n_links = self.rng.randint(self.links_min, self.links_max)

            candidates = []
            for src_idx in range(self.neuron_count):
                if src_idx == dst_idx:
                    continue
                distance = abs(src_idx - dst_idx)
                if distance <= self.link_length_max:
                    candidates.append(src_idx)

            if len(candidates) > n_links:
                selected = self.rng.sample(candidates, n_links)
            else:
                selected = candidates

            if not selected:
                continue

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
        """Configure output actuators from config."""
        for actuator_name, neuron_indices in self.config.get('actuators', {}).items():
            if isinstance(neuron_indices, list):
                valid_indices = [idx for idx in neuron_indices if 0 <= idx < self.neuron_count]
                self.actuators[actuator_name] = valid_indices

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

            center_idx = cfg.get('center_idx', 0)
            radius = cfg.get('radius', 10)
            shape = cfg.get('shape', 'gaussian')
            pressure = cfg.get('pressure', 0)
            parameter = cfg.get('parameter', 'elastic_trigger')
            iter_start = cfg.get('iter_start', 0)
            iter_end = cfg.get('iter_end', self.day_steps)
            shape_values = cfg.get('shape_values', None)

            # Pre-calculate shape multipliers for the region
            multipliers = self._calculate_finger_shape(radius, shape, shape_values)

            finger_press = {
                'name': cfg.get('name', f'finger_{center_idx}'),
                'center_idx': center_idx,
                'radius': radius,
                'pressure': pressure,
                'parameter': parameter,
                'iter_start': iter_start,
                'iter_end': iter_end,
                'multipliers': multipliers
            }
            self.finger_presses.append(finger_press)

        if self.finger_presses:
            self.log(f"Configured {len(self.finger_presses)} Sauron's Finger presses")

    def _calculate_finger_shape(self, radius: int, shape: str,
                                 custom_values: List[float] = None) -> List[float]:
        """
        Calculate shape multipliers for finger press.

        Args:
            radius: Number of neurons on each side of center
            shape: Shape type ('gaussian', 'linear', 'flat', 'custom')
            custom_values: Custom shape array for 'custom' shape

        Returns:
            List of multipliers [0..1] of length 2*radius+1
        """
        width = 2 * radius + 1

        if shape == 'custom' and custom_values:
            # Use custom values, pad or truncate to fit
            if len(custom_values) >= width:
                return custom_values[:width]
            else:
                # Pad with zeros
                padding = width - len(custom_values)
                left_pad = padding // 2
                right_pad = padding - left_pad
                return [0.0] * left_pad + custom_values + [0.0] * right_pad

        elif shape == 'flat':
            # Uniform pressure across region
            return [1.0] * width

        elif shape == 'linear':
            # Linear falloff from center
            multipliers = []
            for i in range(width):
                distance = abs(i - radius)
                multipliers.append(1.0 - distance / (radius + 1))
            return multipliers

        else:  # gaussian (default)
            # Gaussian falloff from center
            sigma = radius / 2.5  # 2.5 sigma covers most of the radius
            multipliers = []
            for i in range(width):
                distance = abs(i - radius)
                multipliers.append(math.exp(-0.5 * (distance / sigma) ** 2))
            return multipliers

    def _apply_saurons_finger(self) -> None:
        """Apply active finger presses to the substrate."""
        for finger in self.finger_presses:
            # Check if finger is active at current iteration
            if not (finger['iter_start'] <= self.iteration <= finger['iter_end']):
                continue

            center = finger['center_idx']
            radius = finger['radius']
            pressure = finger['pressure']
            parameter = finger['parameter']
            multipliers = finger['multipliers']

            # Apply pressure to neurons in range
            for i, mult in enumerate(multipliers):
                idx = center - radius + i
                if 0 <= idx < self.neuron_count:
                    delta = pressure * mult
                    neuron = self.current[idx]

                    # Apply to the specified parameter
                    if parameter == 'elastic_trigger':
                        neuron.elastic_trigger += delta
                    elif parameter == 'elastic_recharge':
                        neuron.elastic_recharge += delta
                    elif parameter == 'charge':
                        neuron.charge += delta
                    elif parameter == 'cumulative_signal':
                        neuron.cumulative_signal += delta

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

        for actuator_name, neuron_indices in self.actuators.items():
            if not neuron_indices:
                outputs[actuator_name] = 0.0
                continue

            active_count = sum(1 for idx in neuron_indices if self.current[idx].active)
            outputs[actuator_name] = active_count / len(neuron_indices)

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
                  ces_strip_width: int = 0) -> Image.Image:
        """Create visualization of the day's neural activity."""
        if not self.day_history:
            self.log("No history to visualize")
            return Image.new('RGB', (width or 100, height or 100), self.bg_color)

        w_raw = self.neuron_count + ces_strip_width
        h_raw = len(self.day_history)

        img_array = np.zeros((h_raw, w_raw, 3), dtype=np.uint8)
        img_array[:, :] = self.bg_color

        if ces_strip_width > 0:
            esps = [rec['esp'] for rec in self.day_history]
            esns = [abs(rec['esn']) for rec in self.day_history]
            max_esp = max(esps) if esps else 1
            max_esn = max(esns) if esns else 1

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

                    img_array[y, x] = color
                elif x < self.head_count:
                    # Show inactive head neurons as dark grey
                    img_array[y, x] = self.head_inactive_color

            if ces_strip_width > 0:
                esp = step_record['esp']
                esn = abs(step_record['esn'])

                esp_normalized = esp / max_esp if max_esp > 0 else 0
                esn_normalized = esn / max_esn if max_esn > 0 else 0

                strip_start = self.neuron_count

                # Fill CES strip with dark green background
                img_array[y, strip_start:strip_start + ces_strip_width] = self.ces_bg_color

                # Draw grid lines (vertical lines at regular intervals)
                for gx in range(0, ces_strip_width, self.ces_grid_spacing):
                    img_array[y, strip_start + gx] = self.ces_grid_color

                # Draw ESP as light green dot (position indicates value)
                # ESP maps to left half of strip (0 = left edge, max = center)
                half_width = ces_strip_width // 2
                esp_x = int(esp_normalized * (half_width - 1))
                img_array[y, strip_start + esp_x] = self.ces_esp_color

                # Draw ESN as red dot (position indicates value)
                # ESN maps to right half of strip (0 = center, max = right edge)
                esn_x = half_width + int(esn_normalized * (half_width - 1))
                img_array[y, strip_start + esn_x] = self.ces_esn_color

        raw_image = Image.fromarray(img_array, 'RGB')

        if width is not None or height is not None:
            target_width = width or raw_image.width
            target_height = height or raw_image.height
            raw_image = raw_image.resize((target_width, target_height), Image.Resampling.NEAREST)

        return raw_image

    def save_visualization(self,
                          filename: str,
                          width: Optional[int] = None,
                          height: Optional[int] = None,
                          ces_strip_width: Optional[int] = None) -> str:
        """Save visualization to a PNG file."""
        if ces_strip_width is None:
            ces_strip_width = self.config.get('CES_STRIP_WIDTH', 100)
        img = self.visualize(width, height, ces_strip_width=ces_strip_width)
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
