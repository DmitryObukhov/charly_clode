#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Charly Simulation - Main Application

CLI application that orchestrates the Charly neuromorphic simulation:
- Loads configuration and creates Charly instance
- Creates the physical model
- Executes day/night simulation cycles
- Saves history and visualizations
- Generates video output

Usage:
    python main.py --days 5 --output output --fps 30
    python main.py --config ../config/config.yaml --days 10
"""

import argparse
import math
import os
import sys
from typing import List, Optional
from datetime import datetime

import yaml
import numpy as np
import cv2

from physical_model import PhysicalModel
from model_linear import Linear
from charly import Charly


def create_video(image_dir: str,
                 output_file: str,
                 fps: int = 30,
                 pattern: str = "day_*.png") -> bool:
    """
    Create video from PNG images.

    Tries OpenCV first, falls back to moviepy.

    Args:
        image_dir: Directory containing PNG files
        output_file: Output video file path
        fps: Frames per second
        pattern: Glob pattern for image files

    Returns:
        True if video was created successfully
    """
    import glob

    # Find all matching images
    image_files = sorted(glob.glob(os.path.join(image_dir, pattern)))

    if not image_files:
        print(f"No images found matching {pattern} in {image_dir}")
        return False

    print(f"Creating video from {len(image_files)} images...")

    # Try OpenCV first
    try:
        import cv2
        import numpy as np
        from PIL import Image

        # Read first image to get dimensions
        first_img = Image.open(image_files[0])
        width, height = first_img.size

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        for img_path in image_files:
            img = Image.open(img_path)
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            video.write(frame)

        video.release()
        print(f"Video saved to {output_file} (using OpenCV)")
        return True

    except ImportError:
        pass

    # Fall back to moviepy
    try:
        from moviepy.editor import ImageSequenceClip

        clip = ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(output_file, codec='libx264')
        print(f"Video saved to {output_file} (using moviepy)")
        return True

    except ImportError:
        print("Warning: Neither opencv-python nor moviepy is installed.")
        print("Install with: pip install opencv-python")
        return False


def run_arc_visualization(config_path: str, output_dir: str, fps: int = 5) -> None:
    """
    Run simulation with arc-based connectome visualization.

    Neurons are displayed as a horizontal chain. Active connections
    are shown as arcs, with brightness based on weight magnitude
    and recency of activation (fades over 128 cycles).

    Args:
        config_path: Path to configuration YAML file
        output_dir: Output directory for final results
    """
    from collections import defaultdict

    # Load configuration
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Display parameters
    display_width = config.get('VISUALIZATION_WIDTH', 1920)
    display_height = config.get('VISUALIZATION_HEIGHT', 1080)
    day_steps = config.get('DAY_STEPS', 20000)
    neuron_count = config.get('NEURON_COUNT', 5000)
    head_count = config.get('HEAD_COUNT', 1000)
    eq_max = config.get('EQ_MAX', 255)

    # Arc visualization parameters
    arc_history_len = 128  # Fade over this many cycles
    chain_y = display_height // 2  # Vertical center for neuron chain
    neuron_spacing = display_width / neuron_count
    max_arc_height = display_height // 2 - 50  # Max arc height

    # Frame averaging parameters
    avg_window = 4  # Average brightness over last N frames
    frame_buffer = []  # Buffer for averaging

    # Colors (BGR for OpenCV)
    bg_color = (0, 0, 0)
    chain_color = (40, 40, 40)
    pos_color = np.array([0, 255, 0])  # Green for positive EQ
    neg_color = np.array([0, 0, 255])  # Red for negative EQ
    neutral_color = np.array([128, 128, 128])

    # Clean and create output directory
    clean_output_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create physical model
    world_size = config.get('WORLD_SIZE', 1000)
    print(f"Creating Linear physical model (world_size={world_size})...")
    physical_model = Linear(
        world_size=world_size,
        lamp_mode=config.get('LAMP_MODE', 'sine'),
        lamp_amplitude=config.get('LAMP_AMPLITUDE', 0.4),
        lamp_period=config.get('LAMP_PERIOD', 4000),
        lamp_center=config.get('LAMP_CENTER', 0.5),
        lamp_file=config.get('LAMP_FILE', ''),
        agent_speed=config.get('AGENT_SPEED', 0.001),
        agent_start=config.get('AGENT_START', 0.5),
        orgasm_tolerance=config.get('ORGASM_TOLERANCE', 0.10),
        terror_range=config.get('TERROR_RANGE', 0.5),
        terror_smoothness=config.get('TERROR_SMOOTHNESS', 50),
        orgasm_smoothness=config.get('ORGASM_SMOOTHNESS', 50)
    )

    # Create Charly instance
    print("Initializing Charly neural substrate...")
    charly = Charly(config, physical_model)

    # Find max weight for normalization
    max_weight = max((abs(s.weight) for s in charly.connectome.synapses), default=1)

    # Arc activation history: {(src, dst): [timestamps of last activations]}
    arc_history = defaultdict(list)

    # Mouse state for Sauron's Finger
    mouse_state = {
        'pressed': False,
        'right_pressed': False,
        'x': 0,
        'neuron_idx': 0
    }

    # Finger parameters for each type
    finger_config = {
        'trigger': {
            'field': 'elastic_trigger',
            'radius': 100,
            'strength': -300,  # Negative = lower threshold = easier to fire
            'color': (0, 255, 255),  # Yellow
            'key': '1'
        },
        'eq': {
            'field': 'eq',
            'radius': 100,
            'strength': 50,  # Positive EQ boost
            'color': (0, 255, 0),  # Green
            'key': '2'
        },
        'fatigue': {
            'field': 'fatigue',
            'radius': 100,
            'strength': -20,  # Negative = reduce fatigue = more active
            'color': (255, 0, 255),  # Magenta
            'key': '3'
        },
        'charge': {
            'field': 'charge',
            'radius': 100,
            'strength': 200,  # Inject charge
            'color': (255, 255, 0),  # Cyan
            'key': '4'
        }
    }
    active_finger = 'trigger'  # Default finger type
    finger_shape_sigma = 0.5

    # Create window
    window_name = "Charly Arc Visualization"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state['pressed'] = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state['pressed'] = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse_state['right_pressed'] = True
        elif event == cv2.EVENT_RBUTTONUP:
            mouse_state['right_pressed'] = False
        mouse_state['x'] = x
        neuron_idx = int(x / display_width * neuron_count)
        mouse_state['neuron_idx'] = max(0, min(neuron_count - 1, neuron_idx))

    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\nStarting arc visualization for {day_steps} steps...")
    print("Controls:")
    print("  1-4: Select finger type (1=Trigger, 2=EQ, 3=Fatigue, 4=Charge)")
    print("  LMB: Apply finger (positive effect)")
    print("  RMB: Apply finger (negative effect)")
    print("  q: Quit, s: Save frame")
    print("=" * 50)

    # Initialize
    charly.day_history = []
    actuator_values = None

    for step in range(day_steps):
        # Apply Sauron's Finger if mouse pressed
        if mouse_state['pressed'] or mouse_state['right_pressed']:
            cfg = finger_config[active_finger]
            finger_radius = cfg['radius']
            # LMB = positive strength, RMB = negative strength
            strength = cfg['strength'] if mouse_state['pressed'] else -cfg['strength']
            field = cfg['field']
            center_idx = mouse_state['neuron_idx']

            for idx in range(max(0, center_idx - finger_radius),
                           min(neuron_count, center_idx + finger_radius + 1)):
                dist = abs(idx - center_idx)
                shape = math.exp(-((dist / finger_radius) ** 2) / (2 * finger_shape_sigma ** 2))
                delta = strength * shape

                # Apply based on field type
                if field == 'elastic_trigger':
                    charly.current[idx].elastic_trigger += delta
                elif field == 'eq':
                    charly.current[idx].eq = int(charly.current[idx].eq + delta)
                elif field == 'fatigue':
                    new_fatigue = charly.current[idx].fatigue + delta
                    charly.current[idx].fatigue = max(0.0, min(100.0, new_fatigue))
                elif field == 'charge':
                    charly.current[idx].charge += delta

        # Execute one step
        result = charly.day_step(actuator_values)
        actuator_values = result['outputs']

        # Record active arcs
        for dst_idx in range(head_count, neuron_count):
            if charly.current[dst_idx].active:
                inputs = charly.connectome.get_inputs(dst_idx)
                for synapse in inputs:
                    if charly.current[synapse.src].active:
                        arc_key = (synapse.src, synapse.dst)
                        arc_history[arc_key].append(step)
                        # Keep only recent history
                        arc_history[arc_key] = [t for t in arc_history[arc_key]
                                                if step - t < arc_history_len]

        # Create display frame
        frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # Draw neuron chain (horizontal line)
        cv2.line(frame, (0, chain_y), (display_width, chain_y), chain_color, 1)

        # Draw arcs for recently active connections
        arcs_to_draw = []
        for (src, dst), timestamps in arc_history.items():
            if not timestamps:
                continue
            # Get most recent activation
            most_recent = max(timestamps)
            age = step - most_recent
            if age >= arc_history_len:
                continue

            # Calculate brightness based on recency
            recency_factor = 1.0 - (age / arc_history_len)

            # Get synapse weight
            synapse = None
            for s in charly.connectome.get_inputs(dst):
                if s.src == src:
                    synapse = s
                    break
            if not synapse:
                continue

            weight_factor = min(1.0, abs(synapse.weight) / max_weight)
            brightness = recency_factor * weight_factor

            # Determine color based on destination neuron EQ
            dst_eq = charly.current[dst].eq
            if dst_eq > 0:
                color = pos_color * brightness
            elif dst_eq < 0:
                color = neg_color * brightness
            else:
                color = neutral_color * brightness

            # Convert to Python int tuple for OpenCV
            color_tuple = (int(color[0]), int(color[1]), int(color[2]))
            arcs_to_draw.append((src, dst, brightness, color_tuple))

        # Sort by brightness (draw dimmer arcs first)
        arcs_to_draw.sort(key=lambda x: x[2])

        # Draw arcs
        for src, dst, brightness, color in arcs_to_draw:
            # Calculate screen positions
            src_x = int(src * display_width / neuron_count)
            dst_x = int(dst * display_width / neuron_count)

            # Arc height proportional to distance
            distance = abs(dst - src)
            arc_height = int((distance / neuron_count) * max_arc_height * 2)
            arc_height = max(10, min(max_arc_height, arc_height))

            # Draw arc using ellipse
            center_x = (src_x + dst_x) // 2
            width = abs(dst_x - src_x)
            if width < 2:
                width = 2

            # Green arcs (positive EQ) go up, red arcs (negative EQ) go down
            dst_eq = charly.current[dst].eq
            if dst_eq >= 0:
                # Green/neutral - arc above the chain
                center_y = chain_y - arc_height // 2
                start_angle, end_angle = 0, 180
            else:
                # Red - arc below the chain
                center_y = chain_y + arc_height // 2
                start_angle, end_angle = 180, 360

            # Draw arc
            axes = (width // 2, arc_height // 2)
            if axes[0] > 0 and axes[1] > 0:
                cv2.ellipse(frame, (center_x, center_y), axes,
                           0, start_angle, end_angle, color, 1, cv2.LINE_AA)

        # Draw active neurons as dots on the chain
        for idx, neuron in enumerate(charly.current):
            x = int(idx * display_width / neuron_count)
            if neuron.active:
                eq = neuron.eq
                if eq > 0:
                    dot_color = (0, 255, 0)
                elif eq < 0:
                    dot_color = (0, 0, 255)
                else:
                    dot_color = (255, 255, 255)
                cv2.circle(frame, (x, chain_y), 2, dot_color, -1)

        # Draw Sauron's Finger indicator
        if mouse_state['pressed'] or mouse_state['right_pressed']:
            cfg = finger_config[active_finger]
            finger_radius = cfg['radius']
            finger_color = cfg['color']
            center_idx = mouse_state['neuron_idx']
            center_x = int(center_idx * display_width / neuron_count)
            left_x = int(max(0, center_idx - finger_radius) * display_width / neuron_count)
            right_x = int(min(neuron_count - 1, center_idx + finger_radius) * display_width / neuron_count)

            cv2.line(frame, (center_x, 0), (center_x, display_height), finger_color, 1)
            cv2.rectangle(frame, (left_x, chain_y - 5), (right_x, chain_y + 5), finger_color, 1)

            # Show finger type and effect
            effect = "+" if mouse_state['pressed'] else "-"
            finger_text = f"[{active_finger.upper()}] {effect}{cfg['field']}"
            cv2.putText(frame, finger_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, finger_color, 1, cv2.LINE_AA)

        # Show active finger type indicator (always visible)
        cfg = finger_config[active_finger]
        type_text = f"Finger: {active_finger.upper()} (1-4 to switch)"
        cv2.putText(frame, type_text, (display_width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, cfg['color'], 1, cv2.LINE_AA)

        # Add frame to buffer for averaging
        frame_buffer.append(frame.astype(np.float32))
        if len(frame_buffer) > avg_window:
            frame_buffer.pop(0)

        # Calculate averaged frame
        if len(frame_buffer) > 0:
            avg_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)
        else:
            avg_frame = frame

        # Info overlay (on averaged frame)
        esp = result['esp']
        esn = result['esn']
        info_text = f"Step: {step+1}/{day_steps}  ESP: {esp}  ESN: {esn}  Arcs: {len(arcs_to_draw)}"
        cv2.putText(avg_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Physical state bar at bottom
        agent_pos = result['inputs'].get('agent_pos', 0.5)
        lamp_pos = result['inputs'].get('lamp_pos', 0.5)
        bar_y = display_height - 30
        cv2.line(avg_frame, (0, bar_y), (display_width, bar_y), (40, 40, 40), 1)
        agent_x = int(agent_pos * display_width)
        lamp_x = int(lamp_pos * display_width)
        cv2.circle(avg_frame, (agent_x, bar_y), 5, (255, 100, 0), -1)  # Agent - blue
        cv2.circle(avg_frame, (lamp_x, bar_y), 5, (0, 100, 255), -1)   # Lamp - orange

        # Save frame as PNG
        frame_file = os.path.join(output_dir, f"arc_{step:06d}.png")
        cv2.imwrite(frame_file, avg_frame)

        # Show frame
        cv2.imshow(window_name, avg_frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nInterrupted by user")
            break
        elif key == ord('1'):
            active_finger = 'trigger'
            print(f"  Finger: TRIGGER (elastic_trigger)")
        elif key == ord('2'):
            active_finger = 'eq'
            print(f"  Finger: EQ (emotional quantum)")
        elif key == ord('3'):
            active_finger = 'fatigue'
            print(f"  Finger: FATIGUE")
        elif key == ord('4'):
            active_finger = 'charge'
            print(f"  Finger: CHARGE")

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1}: ESP={esp}, ESN={esn}, Active arcs={len(arcs_to_draw)}")

    # Cleanup
    cv2.destroyAllWindows()

    # Compile video from saved frames
    print(f"\nCompiling video at {fps} FPS...")
    video_file = os.path.join(output_dir, "arc_simulation.mp4")
    create_video(output_dir, video_file, fps=fps, pattern="arc_*.png")

    print("\nSimulation complete!")
    print(f"Frames saved to: {output_dir}/arc_*.png")
    print(f"Video saved to: {video_file}")


def run_live_visualization(config_path: str, output_dir: str) -> None:
    """
    Run simulation with live scrolling visualization.

    Opens a 1920x1080 window that shows neural activity in real-time.
    Each iteration adds one row. When 1080 rows are filled, the display
    scrolls up to always show the last 1080 iterations.

    Interactive controls:
    - Mouse click & drag: Apply Sauron's Finger (elastic_trigger boost)
    - 'q': Quit simulation
    - 's': Save current frame

    Args:
        config_path: Path to configuration YAML file
        output_dir: Output directory for final results
    """
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Mouse state for Sauron's Finger
    mouse_state = {
        'pressed': False,
        'right_pressed': False,
        'x': 0,
        'y': 0,
        'neuron_idx': 0
    }

    # Finger parameters for each type
    finger_config = {
        'trigger': {'field': 'elastic_trigger', 'radius': 100, 'strength': -300, 'color': (0, 255, 255)},
        'eq': {'field': 'eq', 'radius': 100, 'strength': 50, 'color': (0, 255, 0)},
        'fatigue': {'field': 'fatigue', 'radius': 100, 'strength': -20, 'color': (255, 0, 255)},
        'charge': {'field': 'charge', 'radius': 100, 'strength': 200, 'color': (255, 255, 0)}
    }
    active_finger = 'trigger'
    finger_shape_sigma = 0.5

    # Display parameters
    display_width = config.get('VISUALIZATION_WIDTH', 1920)
    display_height = config.get('VISUALIZATION_HEIGHT', 1080)
    day_steps = config.get('DAY_STEPS', 20000)
    neuron_count = config.get('NEURON_COUNT', 21920)
    head_count = config.get('HEAD_COUNT', 2000)
    eq_max = config.get('EQ_MAX', 255)

    # Strip widths
    ces_strip_width = config.get('CES_STRIP_WIDTH', 100)
    actuator_strip_width = config.get('ACTUATOR_STRIP_WIDTH', 100)
    physical_strip_width = config.get('PHYSICAL_STRIP_WIDTH', 100)
    ces_grid_spacing = config.get('CES_GRID_SPACING', 25)
    ces_range = config.get('CES_RANGE', 5000)

    # Calculate neural diagram width (without strips)
    n_separators = 3  # 3 separators for 3 strips
    total_strip_width = ces_strip_width + actuator_strip_width + physical_strip_width + n_separators
    neural_width = display_width - total_strip_width

    # Colors (BGR for OpenCV)
    pos_color = tuple(reversed(config.get('NEURON_POSITIVE_COLOR', [0, 255, 0])))
    neg_color = tuple(reversed(config.get('NEURON_NEGATIVE_COLOR', [255, 0, 0])))
    bg_color = tuple(reversed(config.get('NEURON_BG_COLOR', [0, 0, 0])))
    head_inactive_color = tuple(reversed(config.get('HEAD_INACTIVE_COLOR', [14, 14, 14])))
    ces_bg_color = tuple(reversed(config.get('CES_BG_COLOR', [14, 14, 14])))
    ces_esp_color = tuple(reversed(config.get('CES_ESP_COLOR', [14, 238, 14])))
    ces_esn_color = tuple(reversed(config.get('CES_ESN_COLOR', [235, 14, 14])))
    ces_grid_color = tuple(reversed(config.get('CES_GRID_COLOR', [24, 24, 24])))
    physical_agent_color = tuple(reversed(config.get('PHYSICAL_AGENT_COLOR', [0, 100, 255])))
    physical_lamp_color = tuple(reversed(config.get('PHYSICAL_LAMP_COLOR', [255, 100, 0])))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create physical model
    world_size = config.get('WORLD_SIZE', 1000)
    print(f"Creating Linear physical model (world_size={world_size})...")
    physical_model = Linear(
        world_size=world_size,
        lamp_mode=config.get('LAMP_MODE', 'sine'),
        lamp_amplitude=config.get('LAMP_AMPLITUDE', 0.4),
        lamp_period=config.get('LAMP_PERIOD', 4000),
        lamp_center=config.get('LAMP_CENTER', 0.5),
        lamp_file=config.get('LAMP_FILE', ''),
        agent_speed=config.get('AGENT_SPEED', 0.001),
        agent_start=config.get('AGENT_START', 0.5),
        orgasm_tolerance=config.get('ORGASM_TOLERANCE', 0.10),
        terror_range=config.get('TERROR_RANGE', 0.5),
        terror_smoothness=config.get('TERROR_SMOOTHNESS', 50),
        orgasm_smoothness=config.get('ORGASM_SMOOTHNESS', 50)
    )

    # Create Charly instance
    print("Initializing Charly neural substrate...")
    charly = Charly(config, physical_model)

    # Get actuator info
    actuator_indices = set()
    actuator_info = {}
    for name, cfg in charly.actuators.items():
        actuator_indices.update(cfg.get('indices', []))
        actuator_info[name] = {
            'color': tuple(reversed(cfg.get('color', [255, 255, 255]))),  # BGR
        }

    # Initialize display buffer (1080 rows of raw neural data)
    # We store raw data and scale to display on render
    raw_buffer = np.zeros((display_height, neuron_count, 3), dtype=np.uint8)
    raw_buffer[:, :] = bg_color

    # CES, actuator, physical strip buffers
    ces_buffer = np.zeros((display_height, ces_strip_width, 3), dtype=np.uint8)
    ces_buffer[:, :] = ces_bg_color

    actuator_buffer = np.zeros((display_height, actuator_strip_width, 3), dtype=np.uint8)
    actuator_buffer[:, :] = ces_bg_color

    physical_buffer = np.zeros((display_height, physical_strip_width, 3), dtype=np.uint8)
    physical_buffer[:, :] = ces_bg_color

    # Tracking for normalization
    max_esp = 1
    max_esn = 1
    max_actuator_output = {name: 0.001 for name in actuator_info}

    # Create window
    window_name = "Charly Neural Substrate - Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    def mouse_callback(event, x, y, flags, param):
        """Handle mouse events for Sauron's Finger."""
        nonlocal mouse_state, neural_width, neuron_count

        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state['pressed'] = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state['pressed'] = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse_state['right_pressed'] = True
        elif event == cv2.EVENT_RBUTTONUP:
            mouse_state['right_pressed'] = False

        # Update position regardless of button state
        mouse_state['x'] = x
        mouse_state['y'] = y

        # Convert screen X to neuron index (only neural area, not strips)
        if x < neural_width:
            neuron_idx = int(x * neuron_count / neural_width)
            neuron_idx = max(0, min(neuron_count - 1, neuron_idx))
            mouse_state['neuron_idx'] = neuron_idx

    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\nStarting live simulation for {day_steps} steps...")
    print("Controls:")
    print("  1-4: Select finger (1=Trigger, 2=EQ, 3=Fatigue, 4=Charge)")
    print("  LMB/RMB: Apply finger (+/-)")
    print("  q: Quit, s: Save frame")
    print("=" * 50)

    # Initialize day
    charly.day_history = []
    actuator_values = None
    row_index = 0

    for step in range(day_steps):
        # Apply Sauron's Finger if mouse is pressed
        if mouse_state['pressed'] or mouse_state['right_pressed']:
            cfg = finger_config[active_finger]
            finger_radius = cfg['radius']
            strength = cfg['strength'] if mouse_state['pressed'] else -cfg['strength']
            field = cfg['field']
            center_idx = mouse_state['neuron_idx']

            for idx in range(max(0, center_idx - finger_radius),
                           min(neuron_count, center_idx + finger_radius + 1)):
                dist = abs(idx - center_idx)
                shape = math.exp(-((dist / finger_radius) ** 2) / (2 * finger_shape_sigma ** 2))
                delta = strength * shape

                if field == 'elastic_trigger':
                    charly.current[idx].elastic_trigger += delta
                elif field == 'eq':
                    charly.current[idx].eq = int(charly.current[idx].eq + delta)
                elif field == 'fatigue':
                    new_fatigue = charly.current[idx].fatigue + delta
                    charly.current[idx].fatigue = max(0.0, min(100.0, new_fatigue))
                elif field == 'charge':
                    charly.current[idx].charge += delta

        # Execute one step
        result = charly.day_step(actuator_values)
        actuator_values = result['outputs']

        esp = result['esp']
        esn = abs(result['esn'])
        inputs = result['inputs']
        outputs = result['outputs']

        # Update max values for normalization
        if esp > max_esp:
            max_esp = esp
        if esn > max_esn:
            max_esn = esn
        for name, val in outputs.items():
            if val > max_actuator_output.get(name, 0):
                max_actuator_output[name] = val

        # Determine which row to update (scrolling logic)
        if row_index < display_height:
            y = row_index
        else:
            # Shift all buffers up by 1 row
            raw_buffer[:-1] = raw_buffer[1:]
            ces_buffer[:-1] = ces_buffer[1:]
            actuator_buffer[:-1] = actuator_buffer[1:]
            physical_buffer[:-1] = physical_buffer[1:]
            y = display_height - 1

            # Clear the new bottom row
            raw_buffer[y, :] = bg_color
            ces_buffer[y, :] = ces_bg_color
            actuator_buffer[y, :] = ces_bg_color
            physical_buffer[y, :] = ces_bg_color

        # Update neural buffer with current neuron states
        for x, neuron in enumerate(charly.current):
            if x >= neuron_count:
                break

            if neuron.active:
                eq = neuron.eq
                brightness = min(1.0, abs(eq) / eq_max) if eq_max > 0 else 1.0
                brightness = max(0.2, brightness)

                if eq > 0:
                    color = tuple(int(c * brightness) for c in pos_color)
                else:
                    color = tuple(int(c * brightness) for c in neg_color)

                raw_buffer[y, x] = color
            elif x < head_count:
                raw_buffer[y, x] = head_inactive_color
            elif x in actuator_indices:
                raw_buffer[y, x] = head_inactive_color
            else:
                raw_buffer[y, x] = bg_color

        # Update CES buffer - bar chart from center (fixed range +/- ces_range)
        ces_buffer[y, :] = ces_bg_color
        # Draw grid
        for gx in range(0, ces_strip_width, ces_grid_spacing):
            ces_buffer[y, gx] = ces_grid_color
        # Center line
        center_x = ces_strip_width // 2
        ces_buffer[y, center_x] = ces_grid_color
        # Draw ESP as green bar from center to right
        esp_normalized = min(1.0, esp / ces_range)
        esp_width = int(esp_normalized * (center_x - 1))
        for bx in range(center_x, center_x + esp_width + 1):
            if bx < ces_strip_width:
                ces_buffer[y, bx] = ces_esp_color
        # Draw ESN as red bar from center to left
        esn_normalized = min(1.0, esn / ces_range)
        esn_width = int(esn_normalized * (center_x - 1))
        for bx in range(center_x - esn_width, center_x):
            if bx >= 0:
                ces_buffer[y, bx] = ces_esn_color

        # Update actuator buffer
        actuator_buffer[y, :] = ces_bg_color
        for gx in range(0, actuator_strip_width, ces_grid_spacing):
            actuator_buffer[y, gx] = ces_grid_color

        actuator_names = list(actuator_info.keys())
        n_actuators = len(actuator_names)
        if n_actuators > 0:
            strip_per_actuator = actuator_strip_width // n_actuators
            for i, name in enumerate(actuator_names):
                color = actuator_info[name]['color']
                output_val = outputs.get(name, 0.0)
                max_val = max_actuator_output.get(name, 0.001)
                normalized = output_val / max_val if max_val > 0 else 0
                strip_start = i * strip_per_actuator
                x_pos = strip_start + int(normalized * (strip_per_actuator - 1))
                if 0 <= x_pos < actuator_strip_width:
                    actuator_buffer[y, x_pos] = color

        # Update physical buffer
        physical_buffer[y, :] = ces_bg_color
        for gx in range(0, physical_strip_width, ces_grid_spacing):
            physical_buffer[y, gx] = ces_grid_color

        agent_pos = inputs.get('agent_pos', 0.5)
        lamp_pos = inputs.get('lamp_pos', 0.5)
        agent_x = int(agent_pos * (physical_strip_width - 1))
        agent_x = max(0, min(physical_strip_width - 1, agent_x))
        lamp_x = int(lamp_pos * (physical_strip_width - 1))
        lamp_x = max(0, min(physical_strip_width - 1, lamp_x))
        physical_buffer[y, agent_x] = physical_agent_color
        physical_buffer[y, lamp_x] = physical_lamp_color

        row_index += 1

        # Compose final display frame
        # Scale neural buffer to neural_width
        neural_scaled = cv2.resize(raw_buffer, (neural_width, display_height),
                                   interpolation=cv2.INTER_NEAREST)

        # Create display frame
        display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        display_frame[:, :neural_width] = neural_scaled

        # Add separator and actuator strip
        x_offset = neural_width
        display_frame[:, x_offset] = (255, 255, 255)  # White separator
        x_offset += 1
        display_frame[:, x_offset:x_offset + actuator_strip_width] = actuator_buffer
        x_offset += actuator_strip_width

        # Add separator and physical strip
        display_frame[:, x_offset] = (255, 255, 255)
        x_offset += 1
        display_frame[:, x_offset:x_offset + physical_strip_width] = physical_buffer
        x_offset += physical_strip_width

        # Add separator and CES strip
        display_frame[:, x_offset] = (255, 255, 255)
        x_offset += 1
        display_frame[:, x_offset:x_offset + ces_strip_width] = ces_buffer

        # Draw Sauron's Finger indicator if mouse is pressed
        if mouse_state['pressed'] or mouse_state['right_pressed']:
            cfg = finger_config[active_finger]
            finger_radius = cfg['radius']
            finger_color = cfg['color']
            center_idx = mouse_state['neuron_idx']

            # Convert neuron indices to screen coordinates
            left_idx = max(0, center_idx - finger_radius)
            right_idx = min(neuron_count - 1, center_idx + finger_radius)
            left_x = int(left_idx * neural_width / neuron_count)
            right_x = int(right_idx * neural_width / neuron_count)
            center_x = int(center_idx * neural_width / neuron_count)

            # Draw semi-transparent overlay for finger area
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (left_x, 0), (right_x, display_height), finger_color, -1)
            cv2.addWeighted(overlay, 0.1, display_frame, 0.9, 0, display_frame)

            # Draw center line
            cv2.line(display_frame, (center_x, 0), (center_x, display_height), finger_color, 1)

            # Draw finger info
            effect = "+" if mouse_state['pressed'] else "-"
            finger_text = f"[{active_finger.upper()}] {effect}{cfg['field']} @ N{center_idx}"
            cv2.putText(display_frame, finger_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, finger_color, 1, cv2.LINE_AA)

        # Show active finger type
        cfg = finger_config[active_finger]
        type_text = f"Finger: {active_finger.upper()} (1-4)"
        cv2.putText(display_frame, type_text, (display_width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, cfg['color'], 1, cv2.LINE_AA)

        # Add step count and timestamp overlay in lower left corner
        timestamp = datetime.now().strftime("%H:%M:%S")
        info_text = f"Step: {step+1}/{day_steps}  Time: {timestamp}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)  # White
        thickness = 1
        text_y = display_height - 15
        cv2.putText(display_frame, info_text, (10, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Show frame
        cv2.imshow(window_name, display_frame)

        # Handle key events (non-blocking)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nSimulation interrupted by user (q pressed)")
            break
        elif key == ord('s'):
            frame_file = os.path.join(output_dir, f"frame_{step:06d}.png")
            cv2.imwrite(frame_file, display_frame)
            print(f"Saved frame to {frame_file}")
        elif key == ord('1'):
            active_finger = 'trigger'
            print(f"  Finger: TRIGGER")
        elif key == ord('2'):
            active_finger = 'eq'
            print(f"  Finger: EQ")
        elif key == ord('3'):
            active_finger = 'fatigue'
            print(f"  Finger: FATIGUE")
        elif key == ord('4'):
            active_finger = 'charge'
            print(f"  Finger: CHARGE")

        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("\nSimulation interrupted by user (window closed)")
            break

        # Progress output every 1000 steps
        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1}/{day_steps}: ESP={esp}, ESN={result['esn']}")

    # Run night phase
    print("\nRunning night phase...")
    charly.run_night()

    # Save final visualization
    final_file = os.path.join(output_dir, "day_0000.png")
    cv2.imwrite(final_file, display_frame)
    print(f"Final frame saved to {final_file}")

    # Wait for key before closing
    print("\nSimulation complete! Press any key in the window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clean_output_directory(output_dir: str) -> None:
    """
    Clean up the output directory before starting a new simulation.

    Removes all PNG, MP4, and other generated files.

    Args:
        output_dir: Directory to clean
    """
    import glob

    if not os.path.exists(output_dir):
        return

    # Patterns of files to remove
    patterns = [
        "day_*.png",
        "faceshot_*.png",
        "physical_*.png",
        "arc_*.png",
        "*.mp4"
    ]

    removed_count = 0
    for pattern in patterns:
        files = glob.glob(os.path.join(output_dir, pattern))
        for f in files:
            try:
                os.remove(f)
                removed_count += 1
            except OSError as e:
                print(f"Warning: Could not remove {f}: {e}")

    if removed_count > 0:
        print(f"Cleaned output directory: removed {removed_count} file(s)")


def run_simulation(config_path: str,
                   num_days: int,
                   output_dir: str,
                   fps: int = 30,
                   no_video: bool = False,
                   log_level: str = "INFO") -> None:
    """
    Run the Charly simulation.

    Args:
        config_path: Path to configuration YAML file
        num_days: Number of days to simulate
        output_dir: Output directory for results
        fps: Video frames per second
        no_video: Skip video generation
        log_level: Logging verbosity
    """
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Clean and create output directory
    clean_output_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create physical model
    world_size = config.get('WORLD_SIZE', 1000)
    print(f"Creating Linear physical model (world_size={world_size})...")
    physical_model = Linear(
        world_size=world_size,
        lamp_mode=config.get('LAMP_MODE', 'sine'),
        lamp_amplitude=config.get('LAMP_AMPLITUDE', 0.4),
        lamp_period=config.get('LAMP_PERIOD', 4000),
        lamp_center=config.get('LAMP_CENTER', 0.5),
        lamp_file=config.get('LAMP_FILE', ''),
        agent_speed=config.get('AGENT_SPEED', 0.001),
        agent_start=config.get('AGENT_START', 0.5),
        orgasm_tolerance=config.get('ORGASM_TOLERANCE', 0.10),
        terror_range=config.get('TERROR_RANGE', 0.5),
        terror_smoothness=config.get('TERROR_SMOOTHNESS', 50),
        orgasm_smoothness=config.get('ORGASM_SMOOTHNESS', 50)
    )

    # Create Charly instance
    print("Initializing Charly neural substrate...")
    charly = Charly(config, physical_model)

    # Run simulation
    print(f"\nStarting simulation for {num_days} day(s)...")
    print("=" * 50)

    for day in range(num_days):
        print(f"\n=== Day {day + 1}/{num_days} ===")

        # Run day phase
        results = charly.run_day()

        # Run night phase
        charly.run_night()

        # Save day visualization
        day_filename = os.path.join(output_dir, f"day_{day:04d}.png")
        charly.save_visualization(
            day_filename,
            width=config.get('VISUALIZATION_WIDTH', 1920),
            height=config.get('VISUALIZATION_HEIGHT', 1080),
            ces_strip_width=config.get('CES_STRIP_WIDTH', 20)
        )

        # Save physical model visualization
        physical_filename = os.path.join(output_dir, f"physical_{day:04d}.png")
        physical_model.save_visualization(
            physical_filename,
            width=config.get('VISUALIZATION_WIDTH', 1920),
            height=config.get('VISUALIZATION_HEIGHT', 1080)
        )

        # Save faceshot
        faceshot_filename = os.path.join(output_dir, f"faceshot_{day:04d}.png")
        charly.faceshot(1024, 1024, faceshot_filename)

        # Print summary
        final_result = results[-1] if results else {}
        print(f"  Final ESP: {final_result.get('esp', 0)}")
        print(f"  Final ESN: {final_result.get('esn', 0)}")

    print("\n" + "=" * 50)
    print("Simulation complete!")

    # Generate video
    if not no_video and num_days > 1:
        video_file = os.path.join(output_dir, "simulation.mp4")
        create_video(output_dir, video_file, fps=fps)

    # Print output summary
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - day_*.png: Neural activity visualizations")
    print(f"  - physical_*.png: Physical model (agent/lamp positions)")
    print(f"  - faceshot_*.png: Connectome structure")
    if not no_video and num_days > 1:
        print(f"  - simulation.mp4: Compiled video")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Charly Neuromorphic Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --days 5
  python main.py --config ../config/config.yaml --days 10 --fps 60
  python main.py --days 3 --no-video --output results
  python main.py --serve --name mysim --config ../config/config.yaml
  python main.py --serve --name mysim --port 8080  # Resume existing
        """
    )

    parser.add_argument('--config', '-c', type=str,
                       default='../config/config.yaml',
                       help='Configuration file path (default: ../config/config.yaml)')

    parser.add_argument('--days', '-d', type=int, default=1,
                       help='Number of days to simulate (default: 1)')

    parser.add_argument('--output', '-o', type=str, default='../output',
                       help='Output directory for results (default: ../output)')

    parser.add_argument('--fps', type=int, default=5,
                       help='Video frames per second (default: 5 for arc, 30 for batch)')

    parser.add_argument('--no-video', action='store_true',
                       help='Skip video generation')

    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging verbosity (default: INFO)')

    parser.add_argument('--live', '-l', action='store_true',
                       help='Run with live visualization window (scrolling display)')

    parser.add_argument('--arc', '-a', action='store_true',
                       help='Run with arc visualization (neurons as chain, connections as arcs)')

    parser.add_argument('--serve', '-s', action='store_true',
                       help='Run as HTTP server with web-based control')

    parser.add_argument('--name', '-n', type=str, default=None,
                       help='Simulation name (required for --serve mode, used as directory name)')

    parser.add_argument('--port', '-p', type=int, default=0,
                       help='HTTP port for server mode (default: auto-allocate)')

    args = parser.parse_args()

    # Handle serve mode
    if args.serve:
        if not args.name:
            print("Error: --name is required for server mode")
            print("Usage: python main.py --serve --name mysim --config ../config/config.yaml")
            sys.exit(1)

        # Config is optional if resuming (will check in server)
        config_path = args.config if os.path.exists(args.config) else None

        try:
            from server import run_server
            run_server(args.name, config_path, args.port)
        except KeyboardInterrupt:
            print("\nServer stopped")
        except Exception as e:
            print(f"Server error: {e}")
            sys.exit(1)
        return

    # Check config file exists (required for non-serve modes)
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Create a config file or specify path with --config")
        sys.exit(1)

    try:
        if args.arc:
            run_arc_visualization(
                config_path=args.config,
                output_dir=args.output,
                fps=args.fps
            )
        elif args.live:
            run_live_visualization(
                config_path=args.config,
                output_dir=args.output
            )
        else:
            run_simulation(
                config_path=args.config,
                num_days=args.days,
                output_dir=args.output,
                fps=args.fps,
                no_video=args.no_video,
                log_level=args.log_level
            )
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
