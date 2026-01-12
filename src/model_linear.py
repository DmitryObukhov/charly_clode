#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Physical Model - 1D World Simulation

Implements a 1D world where an agent with a light sensor ("eye") and
two flagella actuators navigates toward or away from a moving lamp.

Mathematical Models:
- Light intensity: I = I_max / (1 + (d / d_0)^2)  [inverse-square falloff]
- Lamp movement: pos = center + amplitude * sin(2 * pi * frequency * iteration)
- Agent movement: delta = (left - right) * FLAGELLA_STRENGTH

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
    1D world physical model with agent and lamp.

    The world is a 1D space where:
    - Agent: Blue dot with light sensor and two flagella
    - Lamp: Red dot moving sinusoidally as light source

    Sensors:
        eye: Light intensity at agent position [0, 1]
        agent_pos: Normalized agent position [0, 1]
        lamp_pos: Normalized lamp position [0, 1]

    Actuators:
        left: Left flagella activation [0, 1] - pushes agent RIGHT
        right: Right flagella activation [0, 1] - pushes agent LEFT
    """

    # Physical constants
    FLAGELLA_STRENGTH = 5.0     # Maximum movement per iteration
    LIGHT_FALLOFF = 200.0       # Reference distance for light falloff
    MAX_ILLUMINATION = 1.0      # Maximum light intensity

    # Visualization colors
    COLOR_AGENT = (0, 0, 255)   # Blue
    COLOR_LAMP = (255, 0, 0)    # Red
    COLOR_BG = (0, 0, 0)        # Black

    def __init__(self,
                 world_size: int = 1920,
                 lamp_mode: str = 'linear',
                 lamp_amplitude: Optional[float] = None,
                 lamp_frequency: float = 2/1080,
                 lamp_center: Optional[float] = None,
                 lamp_speed: Optional[float] = None):
        """
        Initialize the Linear physical model.

        Args:
            world_size: Size of the 1D world in pixels (default: 1920)
            lamp_mode: 'linear' (left to right) or 'sinusoidal' (oscillating)
            lamp_amplitude: Lamp oscillation amplitude for sinusoidal mode (default: world_size/3)
            lamp_frequency: Lamp oscillation frequency for sinusoidal mode (default: 2/1080)
            lamp_center: Center of lamp oscillation for sinusoidal mode (default: world_size/2)
            lamp_speed: Lamp movement speed for linear mode (default: world_size/1080)
        """
        super().__init__()

        self.world_size = world_size
        self.lamp_mode = lamp_mode
        self.lamp_amplitude = lamp_amplitude if lamp_amplitude is not None else world_size / 3
        self.lamp_frequency = lamp_frequency
        self.lamp_center = lamp_center if lamp_center is not None else world_size / 2
        self.lamp_speed = lamp_speed if lamp_speed is not None else world_size / 1080

        # Initialize state
        self.agent_pos = world_size / 2
        self.lamp_pos = self._calculate_lamp_position(0)
        self.illumination = self._calculate_illumination()

        self.log(f"Linear model initialized: world_size={world_size}")

    def _calculate_lamp_position(self, iteration: int) -> float:
        """
        Calculate lamp position based on movement mode.

        Args:
            iteration: Current iteration number

        Returns:
            Lamp position in pixels
        """
        if self.lamp_mode == 'linear':
            # Linear movement from left (0) to right (world_size-1)
            pos = iteration * self.lamp_speed
        else:
            # Sinusoidal oscillation around center
            offset = self.lamp_amplitude * math.sin(2 * math.pi * self.lamp_frequency * iteration)
            pos = self.lamp_center + offset

        return max(0, min(self.world_size - 1, pos))

    def _calculate_illumination(self) -> float:
        """
        Calculate light intensity at agent position.

        Uses inverse-square falloff: I = I_max / (1 + (d / d_0)^2)

        Returns:
            Light intensity in range [0, 1]
        """
        distance = abs(self.agent_pos - self.lamp_pos)
        return self.MAX_ILLUMINATION / (1 + (distance / self.LIGHT_FALLOFF) ** 2)

    def set(self, actuators: Dict[str, float]) -> None:
        """
        Apply actuator values and advance simulation.

        Left flagella pushes agent right, right flagella pushes left.
        Movement = (left - right) * FLAGELLA_STRENGTH

        Args:
            actuators: Dictionary with 'left' and 'right' activation [0, 1]
        """
        # Get and clamp actuator values
        left_activation = max(0.0, min(1.0, actuators.get('left', 0.0)))
        right_activation = max(0.0, min(1.0, actuators.get('right', 0.0)))

        # Calculate movement (left pushes right, right pushes left)
        movement = (left_activation - right_activation) * self.FLAGELLA_STRENGTH

        # Update agent position with boundary clamping
        new_pos = self.agent_pos + movement
        self.agent_pos = max(0, min(self.world_size - 1, new_pos))

        # Increment iteration and update lamp position
        self.iteration += 1
        self.lamp_pos = self._calculate_lamp_position(self.iteration)

        # Update illumination
        self.illumination = self._calculate_illumination()

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
        all_sensors = {
            'eye': self.illumination,
            'agent_pos': self.agent_pos / self.world_size,
            'lamp_pos': self.lamp_pos / self.world_size
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
        return (['eye', 'agent_pos', 'lamp_pos'], ['left', 'right'])

    def reset(self) -> None:
        """Reset model to initial state."""
        self.iteration = 0
        self.agent_pos = self.world_size / 2
        self.lamp_pos = self._calculate_lamp_position(0)
        self.illumination = self._calculate_illumination()
        self.history = []

    def _get_state_snapshot(self) -> Dict:
        """Capture current state for history."""
        return {
            'agent_pos': self.agent_pos,
            'lamp_pos': self.lamp_pos,
            'illumination': self.illumination
        }

    def _create_raw_visualization(self, num_iterations: int) -> Image.Image:
        """
        Create visualization image from history.

        Each row is one iteration, each column is a world position.
        Agent shown in blue (brightness = illumination), lamp in red.

        Args:
            num_iterations: Number of iterations to visualize

        Returns:
            PIL Image object
        """
        # Image dimensions: width = world_size, height = iterations
        img_array = np.zeros((num_iterations, self.world_size, 3), dtype=np.uint8)

        for row_idx in range(num_iterations):
            if row_idx >= len(self.history):
                break

            state = self.history[row_idx]
            agent_x = int(state['agent_pos'])
            lamp_x = int(state['lamp_pos'])
            illumination = state['illumination']

            # Clamp positions to valid range
            agent_x = max(0, min(self.world_size - 1, agent_x))
            lamp_x = max(0, min(self.world_size - 1, lamp_x))

            # Draw lamp (red)
            img_array[row_idx, lamp_x] = self.COLOR_LAMP

            # Draw agent (blue, brightness varies with illumination)
            brightness = max(0.2, illumination)  # Minimum brightness for visibility
            agent_color = (
                0,
                0,
                max(50, int(255 * brightness))
            )
            img_array[row_idx, agent_x] = agent_color

        return Image.fromarray(img_array, 'RGB')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Linear Physical Model Test")
    parser.add_argument('--size', type=int, default=1920, help='World size')
    parser.add_argument('--steps', type=int, default=500, help='Simulation steps')
    parser.add_argument('--output', type=str, default='linear_test.png', help='Output file')

    args = parser.parse_args()

    # Create model
    model = Linear(world_size=args.size)

    print(f"Sensors: {model.get_names()[0]}")
    print(f"Actuators: {model.get_names()[1]}")

    # Run simulation with oscillating actuators
    for i in range(args.steps):
        # Alternate left/right to create movement
        phase = math.sin(i * 0.05)
        left = max(0, phase)
        right = max(0, -phase)

        model.set({'left': left, 'right': right})

        if (i + 1) % 100 == 0:
            sensors = model.get()
            print(f"Step {i+1}: eye={sensors['eye']:.3f}, "
                  f"agent={sensors['agent_pos']:.3f}, lamp={sensors['lamp_pos']:.3f}")

    # Save visualization
    model.save_visualization(args.output, width=args.size, height=args.steps)
    print(f"\nVisualization saved to {args.output}")
