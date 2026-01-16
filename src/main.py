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


def run_live_visualization(config_path: str, output_dir: str) -> None:
    """
    Run simulation with live scrolling visualization.

    Opens a 1920x1080 window that shows neural activity in real-time.
    Each iteration adds one row. When 1080 rows are filled, the display
    scrolls up to always show the last 1080 iterations.

    Args:
        config_path: Path to configuration YAML file
        output_dir: Output directory for final results
    """
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

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

    print(f"\nStarting live simulation for {day_steps} steps...")
    print("Press 'q' to quit, 's' to save current frame")
    print("=" * 50)

    # Initialize day
    charly.day_history = []
    actuator_values = None
    row_index = 0

    for step in range(day_steps):
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
        """
    )

    parser.add_argument('--config', '-c', type=str,
                       default='../config/config.yaml',
                       help='Configuration file path (default: ../config/config.yaml)')

    parser.add_argument('--days', '-d', type=int, default=1,
                       help='Number of days to simulate (default: 1)')

    parser.add_argument('--output', '-o', type=str, default='../output',
                       help='Output directory for results (default: ../output)')

    parser.add_argument('--fps', type=int, default=30,
                       help='Video frames per second (default: 30)')

    parser.add_argument('--no-video', action='store_true',
                       help='Skip video generation')

    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging verbosity (default: INFO)')

    parser.add_argument('--live', '-l', action='store_true',
                       help='Run with live visualization window (scrolling display)')

    args = parser.parse_args()

    # Check config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Create a config file or specify path with --config")
        sys.exit(1)

    try:
        if args.live:
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
