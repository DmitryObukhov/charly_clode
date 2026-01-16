# Devlog: 2026-01-16 - Live Visualization Mode

## Session Summary

Added real-time live visualization mode with scrolling display for monitoring neural activity during simulation.

## Changes Made

### 1. Live Visualization Mode (`--live` flag)
- New `run_live_visualization()` function in `main.py`
- Opens 1920x1080 OpenCV window
- Each simulation step adds one row to the display
- Scrolling buffer: when 1080 rows filled, shifts up and adds new row at bottom
- Always shows the last 1080 iterations
- Keyboard controls: `q` to quit, `s` to save frame
- Proper window close detection (X button now works)

### 2. CES Strip Redesign
- Changed from dot display to bar chart
- ESP (green bar) extends from center to right
- ESN (red bar) extends from center to left
- Fixed range normalization (configurable via `CES_RANGE` parameter)
- Default range: +/- 5000

### 3. Information Overlay
- Step count and timestamp displayed in lower left corner
- Format: `Step: X/20000  Time: HH:MM:SS`

### 4. Configuration Changes
- `DAY_STEPS`: 1080 -> 20000 (longer simulation days)
- `NEURON_COUNT`: 21920 -> 5000 (reduced for faster iteration)
- `HEAD_COUNT`: 2000 -> 500
- `TAIL_COUNT`: 20000 -> 4500
- Added `CES_RANGE: 5000` parameter
- Proportionally adjusted all neuron indices:
  - Connectome distribution segments
  - Input receptor positions
  - Actuator positions
  - Named neuron aliases
  - Sauron's Finger positions

## New Config Parameters

```yaml
CES_RANGE: 5000    # Max range for ESP/ESN display (+/-)
DAY_STEPS: 20000   # Iterations per day phase
NEURON_COUNT: 5000 # Total neurons in substrate
```

## Usage

```batch
cd src
python main.py --live
```

## Technical Notes

- OpenCV window creation with `cv2.namedWindow()` and `cv2.WINDOW_NORMAL`
- Window close detection via `cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)`
- BGR color format for OpenCV (reversed from RGB config values)
- Efficient numpy array operations for buffer scrolling
- Text overlay using `cv2.putText()` with anti-aliasing

## Files Modified

- `src/main.py` - Added live visualization function and CLI flag
- `config/config.yaml` - Updated neuron counts, added CES_RANGE
