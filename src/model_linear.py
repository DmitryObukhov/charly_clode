#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Physical Model - 1D World Simulation

Implements a 1D world where an agent tracks a moving lamp.
The agent receives pleasure (orgasm) when matching the lamp position,
and pain (terror) when deviating above or below.

Lamp Movement:
- Sine: pos = center + amplitude * sin(2 * pi * iteration / period)
- Linear: pos = (iteration * speed) % 1.0
- File: pos = data[iteration % len(data)]

Agent Movement:
- Controlled by move_up and move_down actuators
- delta = (move_up - move_down) * agent_speed

Sensors:
- orgasm: Intensity of pleasure when matching lamp (0-1)
- terror_up: Intensity of pain when agent > lamp (0-1)
- terror_down: Intensity of pain when agent < lamp (0-1)
- agent_pos: Normalized agent position (0-1)
- lamp_pos: Normalized lamp position (0-1)
- deviation: Signed distance (agent - lamp), normalized

Usage:
    python model_linear.py
"""

import math
from typing import Dict, List, Set, Tuple, Union, Optional
from PIL import Image
import numpy as np

from physical_model import PhysicalModel


class Linear(PhysicalModel):
    """
    1D world physical model with agent tracking a lamp.

    The agent feels pleasure when matching the lamp position
    and pain when deviating in either direction.
    """

    # Visualization colors
    COLOR_AGENT = (0, 0, 255)   # Blue
    COLOR_LAMP = (255, 0, 0)    # Red
    COLOR_BG = (0, 0, 0)        # Black

    def __init__(self,
                 world_size: int = 1000,
                 lamp_mode: str = 'sine',
                 lamp_amplitude: float = 0.4,
                 lamp_period: int = 4000,
                 lamp_center: float = 0.5,
                 lamp_file: str = '',
                 agent_speed: float = 0.001,
                 agent_start: float = 0.5,
                 orgasm_tolerance: float = 0.10,
                 terror_range: float = 0.5,
                 terror_smoothness: float = 50.0,
                 orgasm_smoothness: float = 50.0):
        """
        Initialize the Linear physical model.

        Args:
            world_size: Size of the 1D world (for visualization)
            lamp_mode: 'sine', 'linear', or 'file'
            lamp_amplitude: Amplitude for sine mode (fraction of world)
            lamp_period: Period for sine mode (in steps)
            lamp_center: Center position for sine mode (0-1)
            lamp_file: CSV file path for file mode
            agent_speed: Max movement per step (fraction of world)
            agent_start: Starting position (0-1)
            orgasm_tolerance: Tolerance for "hit" (fraction of world)
            terror_range: Range where terror is felt (fraction of world)
            terror_smoothness: Response curve 0-100 (0=binary, 50=sigmoid, 100=linear)
            orgasm_smoothness: Response curve 0-100 (0=binary, 50=sigmoid, 100=linear)
        """
        super().__init__()

        self.world_size = world_size
        self.lamp_mode = lamp_mode
        self.lamp_amplitude = lamp_amplitude
        self.lamp_period = lamp_period
        self.lamp_center = lamp_center
        self.lamp_file = lamp_file
        self.agent_speed = agent_speed
        self.agent_start = agent_start
        self.orgasm_tolerance = orgasm_tolerance
        self.terror_range = terror_range
        self.terror_smoothness = terror_smoothness / 100.0  # Normalize to 0-1
        self.orgasm_smoothness = orgasm_smoothness / 100.0  # Normalize to 0-1

        # Load file data if needed
        self.lamp_data = []
        if lamp_mode == 'file' and lamp_file:
            self._load_lamp_file(lamp_file)

        # Initialize state (normalized 0-1)
        self.agent_pos = agent_start
        self.lamp_pos = self._calculate_lamp_position(0)

        self.log(f"Linear model initialized: world_size={world_size}")

    def _load_lamp_file(self, filepath: str) -> None:
        """Load lamp position data from CSV file."""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            value = float(line.split(',')[0])
                            self.lamp_data.append(value)
                        except ValueError:
                            continue

            if self.lamp_data:
                # Normalize to 0-1 range
                min_val = min(self.lamp_data)
                max_val = max(self.lamp_data)
                if max_val > min_val:
                    self.lamp_data = [(v - min_val) / (max_val - min_val)
                                      for v in self.lamp_data]
                self.log(f"Loaded {len(self.lamp_data)} lamp positions from {filepath}")
        except Exception as e:
            self.log(f"Warning: Could not load lamp file {filepath}: {e}")
            self.lamp_data = []

    def _calculate_lamp_position(self, iteration: int) -> float:
        """
        Calculate lamp position based on movement mode.

        Returns:
            Lamp position normalized to [0, 1]
        """
        if self.lamp_mode == 'file' and self.lamp_data:
            idx = iteration % len(self.lamp_data)
            return self.lamp_data[idx]

        elif self.lamp_mode == 'linear':
            # Linear sweep from 0 to 1, wrapping
            pos = (iteration * 0.0001) % 1.0
            return pos

        else:  # 'sine' (default)
            # Sinusoidal oscillation around center
            phase = 2 * math.pi * iteration / self.lamp_period
            offset = self.lamp_amplitude * math.sin(phase)
            pos = self.lamp_center + offset
            return max(0.0, min(1.0, pos))

    def _apply_response_curve(self, value: float, smoothness: float) -> float:
        """
        Apply response curve to a normalized value [0,1].

        Args:
            value: Input value in range [0, 1]
            smoothness: Curve shape (0=binary step, 0.5=sigmoid, 1=linear)

        Returns:
            Transformed value in range [0, 1]
        """
        value = max(0.0, min(1.0, value))

        if smoothness <= 0.01:
            # Binary step: 0 below 0.5, 1 at/above 0.5
            return 0.0 if value < 0.5 else 1.0

        elif smoothness >= 0.99:
            # Linear: output = input
            return value

        else:
            # Sigmoid with variable steepness
            # steepness k: higher = steeper (more step-like)
            # At smoothness=0.5, k=6 gives nice S-curve
            # At smoothness→0, k→∞ (step)
            # At smoothness→1, k→0 (linear)

            # Map smoothness to steepness: inverse relationship
            # k = 12 * (1 - smoothness) gives k=6 at smoothness=0.5
            k = 12.0 * (1.0 - smoothness)

            if k < 0.1:
                # Very low steepness = effectively linear
                return value

            # Apply sigmoid: map [0,1] to [-k/2, k/2] then through sigmoid
            x = (value - 0.5) * k
            sigmoid_value = 1.0 / (1.0 + math.exp(-x))

            # Blend sigmoid with linear based on smoothness
            # Higher smoothness = more linear influence
            blend = smoothness
            return blend * value + (1.0 - blend) * sigmoid_value

    def _calculate_sensory_signals(self) -> Dict[str, float]:
        """
        Calculate orgasm and terror signals based on agent-lamp distance.

        Returns:
            Dictionary with orgasm, terror_up, terror_down values (0-1)
        """
        # Signed deviation: positive = agent above lamp
        deviation = self.agent_pos - self.lamp_pos
        abs_deviation = abs(deviation)

        # Orgasm: maximum at center, decreasing to edge of tolerance
        if abs_deviation <= self.orgasm_tolerance:
            # Within tolerance - feel pleasure
            # Raw value: 1.0 at center, 0.0 at edge of tolerance
            raw_orgasm = 1.0 - (abs_deviation / self.orgasm_tolerance)
            # Apply response curve (inverted: we want high at center)
            orgasm = self._apply_response_curve(raw_orgasm, self.orgasm_smoothness)
            terror_up = 0.0
            terror_down = 0.0
        else:
            # Outside tolerance - feel terror
            orgasm = 0.0

            # Calculate raw terror intensity (0 at tolerance edge, 1 at terror_range)
            excess = abs_deviation - self.orgasm_tolerance
            raw_terror = min(1.0, excess / self.terror_range)

            # Apply response curve
            terror_intensity = self._apply_response_curve(raw_terror, self.terror_smoothness)

            if deviation > 0:
                # Agent above lamp
                terror_up = terror_intensity
                terror_down = 0.0
            else:
                # Agent below lamp
                terror_up = 0.0
                terror_down = terror_intensity

        return {
            'orgasm': orgasm,
            'terror_up': terror_up,
            'terror_down': terror_down,
            'deviation': deviation
        }

    def set(self, actuators: Dict[str, float]) -> None:
        """
        Apply actuator values and advance simulation.

        Args:
            actuators: Dictionary with 'move_up' and 'move_down' activation [0, 1]
        """
        # Get actuator values
        move_up = max(0.0, min(1.0, actuators.get('move_up', 0.0)))
        move_down = max(0.0, min(1.0, actuators.get('move_down', 0.0)))

        # Calculate movement
        movement = (move_up - move_down) * self.agent_speed

        # Update agent position with boundary clamping
        new_pos = self.agent_pos + movement
        self.agent_pos = max(0.0, min(1.0, new_pos))

        # Increment iteration and update lamp position
        self.iteration += 1
        self.lamp_pos = self._calculate_lamp_position(self.iteration)

        # Save state to history
        self.save_state()

    def get(self, sensors: Union[List[str], Set[str], None] = None) -> Dict[str, float]:
        """
        Read sensor values.

        Args:
            sensors: List of sensor names to read, or None for all

        Returns:
            Dictionary mapping sensor names to values
        """
        signals = self._calculate_sensory_signals()

        all_sensors = {
            'orgasm': signals['orgasm'],
            'terror_up': signals['terror_up'],
            'terror_down': signals['terror_down'],
            'deviation': signals['deviation'],
            'agent_pos': self.agent_pos,
            'lamp_pos': self.lamp_pos
        }

        if sensors is None:
            return all_sensors

        return {name: all_sensors.get(name, 0.0) for name in sensors}

    def get_names(self) -> Tuple[List[str], List[str]]:
        """
        Get names of sensors and actuators.

        Returns:
            Tuple of (sensor_names, actuator_names)
        """
        return (
            ['orgasm', 'terror_up', 'terror_down', 'deviation', 'agent_pos', 'lamp_pos'],
            ['move_up', 'move_down']
        )

    def reset(self) -> None:
        """Reset model to initial state."""
        self.iteration = 0
        self.agent_pos = self.agent_start
        self.lamp_pos = self._calculate_lamp_position(0)
        self.history = []

    def _get_state_snapshot(self) -> Dict:
        """Capture current state for history."""
        signals = self._calculate_sensory_signals()
        return {
            'agent_pos': self.agent_pos,
            'lamp_pos': self.lamp_pos,
            'orgasm': signals['orgasm'],
            'terror_up': signals['terror_up'],
            'terror_down': signals['terror_down'],
            'deviation': signals['deviation']
        }

    def _create_raw_visualization(self, num_iterations: int) -> Image.Image:
        """
        Create visualization image from history.

        Each row is one iteration, each column is a world position.
        Agent shown in blue, lamp in red.

        Args:
            num_iterations: Number of iterations to visualize

        Returns:
            PIL Image object
        """
        img_array = np.zeros((num_iterations, self.world_size, 3), dtype=np.uint8)

        for row_idx in range(num_iterations):
            if row_idx >= len(self.history):
                break

            state = self.history[row_idx]
            agent_x = int(state['agent_pos'] * (self.world_size - 1))
            lamp_x = int(state['lamp_pos'] * (self.world_size - 1))

            # Clamp positions to valid range
            agent_x = max(0, min(self.world_size - 1, agent_x))
            lamp_x = max(0, min(self.world_size - 1, lamp_x))

            # Draw lamp (red)
            img_array[row_idx, lamp_x] = self.COLOR_LAMP

            # Draw agent (blue, brighter when in orgasm)
            orgasm = state.get('orgasm', 0)
            brightness = 0.3 + 0.7 * orgasm  # 30% to 100% brightness
            agent_color = (0, 0, int(255 * brightness))
            img_array[row_idx, agent_x] = agent_color

        return Image.fromarray(img_array, 'RGB')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Linear Physical Model Test")
    parser.add_argument('--size', type=int, default=1000, help='World size')
    parser.add_argument('--steps', type=int, default=500, help='Simulation steps')
    parser.add_argument('--output', type=str, default='linear_test.png', help='Output file')

    args = parser.parse_args()

    # Create model
    model = Linear(world_size=args.size)

    print(f"Sensors: {model.get_names()[0]}")
    print(f"Actuators: {model.get_names()[1]}")

    # Run simulation - agent tries to follow lamp with simple logic
    for i in range(args.steps):
        sensors = model.get()

        # Simple following logic
        deviation = sensors['deviation']
        if deviation > 0.01:
            # Agent above lamp - move down
            model.set({'move_up': 0.0, 'move_down': 0.5})
        elif deviation < -0.01:
            # Agent below lamp - move up
            model.set({'move_up': 0.5, 'move_down': 0.0})
        else:
            # On target
            model.set({'move_up': 0.0, 'move_down': 0.0})

        if (i + 1) % 100 == 0:
            print(f"Step {i+1}: orgasm={sensors['orgasm']:.3f}, "
                  f"terror_up={sensors['terror_up']:.3f}, "
                  f"terror_down={sensors['terror_down']:.3f}, "
                  f"agent={sensors['agent_pos']:.3f}, lamp={sensors['lamp_pos']:.3f}")

    # Save visualization
    model.save_visualization(args.output, width=args.size, height=args.steps)
    print(f"\nVisualization saved to {args.output}")
