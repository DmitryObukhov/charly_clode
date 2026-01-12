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

import yaml

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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create physical model
    world_size = config.get('WORLD_SIZE', 1920)
    print(f"Creating Linear physical model (world_size={world_size})...")
    physical_model = Linear(world_size=world_size)

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

    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory for results (default: output)')

    parser.add_argument('--fps', type=int, default=30,
                       help='Video frames per second (default: 30)')

    parser.add_argument('--no-video', action='store_true',
                       help='Skip video generation')

    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging verbosity (default: INFO)')

    args = parser.parse_args()

    # Check config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Create a config file or specify path with --config")
        sys.exit(1)

    try:
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
