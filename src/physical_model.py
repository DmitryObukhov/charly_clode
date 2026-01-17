#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical Model - Abstract Base Class

Defines the interface for physical world simulations that connect
to the neural substrate. Physical models provide sensors (inputs)
and actuators (outputs) for the agent.

Usage:
    python physical_model.py
"""

import abc
import os
from typing import Dict, List, Set, Tuple, Union, Optional
from PIL import Image
import numpy as np


class PhysicalModel(abc.ABC):
    """
    Abstract base class for physical world simulations.

    Physical models implement the sensor/actuator interface that connects
    to the neural substrate. The model maintains state, history, and
    provides visualization.

    Attributes:
        iteration: Current iteration counter
        history: List of state snapshots for visualization
        log: Logging function (default: print)
    """

    def __init__(self):
        """Initialize the physical model."""
        self.iteration = 0
        self.history: List[Dict] = []
        self.log = print

    def set_logger(self, log_func) -> None:
        """
        Replace the default logging function.

        Args:
            log_func: New logging function to use
        """
        self.log = log_func

    @abc.abstractmethod
    def set(self, actuators: Dict[str, float]) -> None:
        """
        Apply actuator values to the physical model.

        This advances the simulation by one step, applying the control
        inputs and updating the world state.

        Args:
            actuators: Dictionary mapping actuator names to values [0, 1].
                      Values represent activation percentage.
        """
        pass

    @abc.abstractmethod
    def get(self, sensors: Union[List[str], Set[str], None] = None) -> Dict[str, float]:
        """
        Read sensor values from the physical model.

        Args:
            sensors: List or set of sensor names to read.
                    If None, returns all available sensors.

        Returns:
            Dictionary mapping sensor names to their current values.
            Values are typically normalized to [0, 1].
        """
        pass

    @abc.abstractmethod
    def get_names(self) -> Tuple[List[str], List[str]]:
        """
        Get names of all available sensors and actuators.

        Returns:
            Tuple of (sensor_names, actuator_names)
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the model to its initial state and clear history."""
        pass

    @abc.abstractmethod
    def _get_state_snapshot(self) -> Dict:
        """
        Capture current state for history recording.

        Returns:
            Dictionary with all relevant state variables.
        """
        pass

    @abc.abstractmethod
    def _create_raw_visualization(self, num_iterations: int) -> Image.Image:
        """
        Create the raw visualization image without scaling.

        Args:
            num_iterations: Number of iterations to include.

        Returns:
            PIL Image object with the visualization.
        """
        pass

    @abc.abstractmethod
    def save_state_to_file(self, path: str) -> None:
        """
        Save the complete model state to a file.
        
        Args:
            path: Absolute path to the save file.
        """
        pass

    @abc.abstractmethod
    def load_state_from_file(self, path: str) -> None:
        """
        Load the model state from a file.
        
        Args:
            path: Absolute path to the save file.
        """
        pass

    def save_state(self) -> None:
        """Record current state to history."""
        snapshot = self._get_state_snapshot()
        snapshot['iteration'] = self.iteration
        self.history.append(snapshot)

    def get_history(self) -> List[Dict]:
        """Get the recorded history."""
        return self.history

    def clear_history(self) -> None:
        """Clear the recorded history."""
        self.history = []

    def visualize(self,
                  width: Optional[int] = None,
                  height: Optional[int] = None,
                  iterations: int = -1) -> Image.Image:
        """
        Create visualization of the simulation history.

        Args:
            width: Output width (None = use raw width)
            height: Output height (None = use raw height)
            iterations: Number of iterations to visualize (-1 = all)

        Returns:
            PIL Image object with the visualization.
        """
        if iterations < 0:
            iterations = len(self.history)
        iterations = min(iterations, len(self.history))

        if iterations == 0:
            # Return empty image if no history
            w = width or 100
            h = height or 100
            return Image.new('RGB', (w, h), (0, 0, 0))

        # Create raw visualization
        raw_image = self._create_raw_visualization(iterations)

        # Scale if dimensions specified
        if width is not None or height is not None:
            target_width = width or raw_image.width
            target_height = height or raw_image.height
            raw_image = raw_image.resize(
                (target_width, target_height),
                Image.Resampling.NEAREST
            )

        return raw_image

    def save_visualization(self,
                          filename: str,
                          width: Optional[int] = None,
                          height: Optional[int] = None,
                          iterations: int = -1) -> str:
        """
        Save visualization to a PNG file.

        Args:
            filename: Output filename
            width: Output width
            height: Output height
            iterations: Number of iterations to visualize (-1 = all)

        Returns:
            The filename where the image was saved.
        """
        img = self.visualize(width, height, iterations)
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        img.save(filename, 'PNG')
        self.log(f"Visualization saved to {filename}")
        return filename


if __name__ == "__main__":
    print("PhysicalModel is an abstract base class.")
    print("Use a concrete implementation like Linear from model_linear.py")
