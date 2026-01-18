#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Charly Simulation Server

HTTP server providing web-based control and monitoring of the simulation.
Supports creating new simulations, resuming from saved state, and real-time
control via a web interface.

Usage:
    python main.py serve --name mysim --config ../config/config.yaml [--port 8080]
"""

import json
import os
import shutil
import socket
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, parse_qs

import yaml

from charly import Charly
from model_linear import Linear


class SimulationState:
    """Enumeration of simulation states."""
    STOPPED = 'stopped'
    RUNNING = 'running'
    PAUSED = 'paused'


class SimulationManager:
    """
    Manages the simulation lifecycle and state.

    Handles initialization, start/stop/pause control, stepping,
    and state persistence.
    """

    def __init__(self, name: str, base_dir: str = 'simulations'):
        """
        Initialize the simulation manager.

        Args:
            name: Simulation name (used as directory name)
            base_dir: Base directory for all simulations
        """
        self.name = name
        self.base_dir = base_dir
        self.sim_dir = os.path.join(base_dir, name)
        self.charly: Optional[Charly] = None
        self.physical_model: Optional[Linear] = None
        self.config: Dict[str, Any] = {}
        self.state = SimulationState.STOPPED
        self.lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._auto_save_interval = 1000  # Save every N iterations
        self._last_save_iteration = 0
        self._actuator_values: Optional[Dict[str, float]] = None
        # Activation history for visualization (last 3 steps)
        self._activation_history: List[List[bool]] = []  # [step-2, step-1, current]

    def initialize(self, config_path: Optional[str] = None) -> bool:
        """
        Initialize or resume simulation.

        If a saved state exists, it will be loaded. Otherwise, a new
        simulation is created from the config file.

        Args:
            config_path: Path to config YAML (required for new simulation)

        Returns:
            True if initialization succeeded
        """
        os.makedirs(self.sim_dir, exist_ok=True)

        config_dest = os.path.join(self.sim_dir, 'config.yaml')
        state_file = os.path.join(self.sim_dir, 'state.json')

        # Determine if we're resuming or starting fresh
        has_saved_state = os.path.exists(state_file)
        has_config = os.path.exists(config_dest)

        if config_path and os.path.exists(config_path):
            # Copy config to simulation directory
            shutil.copy2(config_path, config_dest)
            print(f"Config copied to {config_dest}")
        elif not has_config:
            print(f"Error: No config file. Provide --config for new simulation.")
            return False

        # Load config
        with open(config_dest, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Create physical model
        world_size = self.config.get('WORLD_SIZE', 1920)
        self.physical_model = Linear(world_size=world_size)

        # Create Charly instance
        self.charly = Charly(self.config, self.physical_model)
        self.charly.set_logger(lambda msg: print(f"[Charly] {msg}"))

        # Try to load saved state
        if has_saved_state:
            if self.charly.load_state(self.sim_dir):
                print(f"Resumed simulation '{self.name}' from saved state")
                print(f"  Iteration: {self.charly.iteration}, Day: {self.charly.day_index}")
            else:
                print(f"Warning: Could not load saved state, starting fresh")

        self._last_save_iteration = self.charly.iteration
        self.state = SimulationState.PAUSED
        return True

    def start(self) -> bool:
        """
        Start or resume simulation in background thread.

        Returns:
            True if started successfully
        """
        with self.lock:
            if self.state == SimulationState.RUNNING:
                return True
            if self.charly is None:
                return False

            self._stop_event.clear()
            self.state = SimulationState.RUNNING
            self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self._thread.start()
            return True

    def pause(self) -> bool:
        """
        Pause the simulation.

        Returns:
            True if paused successfully
        """
        with self.lock:
            if self.state != SimulationState.RUNNING:
                return True

            self._stop_event.set()
            self.state = SimulationState.PAUSED

        # Wait for thread to finish current iteration
        if self._thread:
            self._thread.join(timeout=2.0)

        # Save state on pause
        self.save()
        return True

    def stop(self) -> bool:
        """
        Stop simulation and save state.

        Returns:
            True if stopped successfully
        """
        self.pause()
        with self.lock:
            self.state = SimulationState.STOPPED
        return True

    def step(self, count: int = 1) -> Dict[str, Any]:
        """
        Execute N steps while paused.

        Args:
            count: Number of steps to execute

        Returns:
            Result of the last step
        """
        with self.lock:
            if self.state == SimulationState.RUNNING:
                return {'error': 'Cannot step while running'}
            if self.charly is None:
                return {'error': 'Simulation not initialized'}

            result = {}
            for _ in range(count):
                result = self.charly.day_step(self._actuator_values)
                self._actuator_values = result.get('outputs')
                self._update_activation_history()

                # Check for day boundary
                day_steps = self.config.get('DAY_STEPS', 1080)
                if self.charly.iteration % day_steps == 0 and self.charly.iteration > 0:
                    self.charly.run_night()

            return result

    def get_status(self) -> Dict[str, Any]:
        """
        Get current simulation status.

        Returns:
            Dictionary with status information
        """
        with self.lock:
            if self.charly is None:
                return {
                    'name': self.name,
                    'state': self.state,
                    'error': 'Not initialized'
                }

            charly_status = self.charly.get_status()

            # Get physical model state
            physical_state = {}
            if self.physical_model:
                physical_state = self.physical_model.get(['agent_pos', 'lamp_pos'])

            return {
                'name': self.name,
                'state': self.state,
                'iteration': charly_status['iteration'],
                'day_index': charly_status['day_index'],
                'day_steps': self.config.get('DAY_STEPS', 1080),
                'neuron_count': charly_status['neuron_count'],
                'active_neurons': charly_status['active_neurons'],
                'esp': charly_status['esp'],
                'esn': charly_status['esn'],
                'synapse_count': charly_status['synapse_count'],
                'agent_pos': physical_state.get('agent_pos', 0.5),
                'lamp_pos': physical_state.get('lamp_pos', 0.5),
                'auto_save_interval': self._auto_save_interval,
                'last_save_iteration': self._last_save_iteration
            }

    def save(self) -> bool:
        """
        Save current state to disk.

        Returns:
            True if saved successfully
        """
        with self.lock:
            if self.charly is None:
                return False
            result = self.charly.save_state(self.sim_dir)
            if result:
                self._last_save_iteration = self.charly.iteration
            return result

    def get_neurons(self, count: int = 100) -> Dict[str, Any]:
        """
        Get current state of all neurons for visualization with activation history.

        Returns:
            Dictionary with neuron data including activation history for last 3 steps
        """
        with self.lock:
            if self.charly is None:
                return {'error': 'Not initialized', 'neurons': [], 'width': 0, 'height': 0}

            total = len(self.charly.current)

            # Calculate matrix dimensions (as square as possible)
            width = int(total ** 0.5)
            height = (total + width - 1) // width

            neurons = []
            history_len = len(self._activation_history)

            for idx in range(total):
                n = self.charly.current[idx]

                # Activation state: 0=not active, 1=active at step-2, 2=active at step-1, 3=active now
                activation_state = 0
                if history_len >= 1 and idx < len(self._activation_history[-1]) and self._activation_history[-1][idx]:
                    activation_state = 3  # Current step
                elif history_len >= 2 and idx < len(self._activation_history[-2]) and self._activation_history[-2][idx]:
                    activation_state = 2  # Step -1
                elif history_len >= 3 and idx < len(self._activation_history[-3]) and self._activation_history[-3][idx]:
                    activation_state = 1  # Step -2

                neurons.append({
                    'eq': n.eq,
                    's': activation_state  # 0=off, 1=step-2, 2=step-1, 3=current
                })

            return {
                'neurons': neurons,
                'total': total,
                'width': width,
                'height': height
            }

    def _update_activation_history(self) -> None:
        """Update activation history with current neuron states."""
        if self.charly is None:
            return
        current_active = [n.active for n in self.charly.current]
        self._activation_history.append(current_active)
        # Keep only last 3 steps
        if len(self._activation_history) > 3:
            self._activation_history.pop(0)

    def _simulation_loop(self) -> None:
        """Background simulation loop."""
        while not self._stop_event.is_set():
            with self.lock:
                if self.charly is None:
                    break

                # Execute one step
                result = self.charly.day_step(self._actuator_values)
                self._actuator_values = result.get('outputs')
                self._update_activation_history()

                # Check for day boundary
                day_steps = self.config.get('DAY_STEPS', 1080)
                if self.charly.iteration % day_steps == 0 and self.charly.iteration > 0:
                    self.charly.run_night()

                # Auto-save periodically
                if self.charly.iteration - self._last_save_iteration >= self._auto_save_interval:
                    self.charly.save_state(self.sim_dir)
                    self._last_save_iteration = self.charly.iteration

            # Small sleep to prevent CPU spinning and allow other threads
            time.sleep(0.001)


# HTML template embedded as string
HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <title>Charly Simulation - {name}</title>
    <meta charset="utf-8">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            background: #1a1a2e;
            color: #eee;
            font-family: 'Consolas', 'Monaco', monospace;
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        h1 {{ margin: 0; font-size: 24px; }}
        .state-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .state-running {{ background: #4ecca3; color: #000; }}
        .state-paused {{ background: #ffc107; color: #000; }}
        .state-stopped {{ background: #e63946; color: #fff; }}
        .controls {{
            display: flex;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-family: inherit;
            transition: opacity 0.2s;
        }}
        .btn:hover {{ opacity: 0.8; }}
        .btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .btn-start {{ background: #4ecca3; color: #000; }}
        .btn-pause {{ background: #ffc107; color: #000; }}
        .btn-stop {{ background: #e63946; color: #fff; }}
        .btn-step {{ background: #0077b6; color: #fff; }}
        .btn-save {{ background: #6c757d; color: #fff; }}
        .status-panel {{
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #0f3460;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4ecca3;
        }}
        .metric-value.positive {{ color: #4ecca3; }}
        .metric-value.negative {{ color: #e63946; }}
        .metric-label {{
            font-size: 12px;
            opacity: 0.7;
            margin-top: 5px;
        }}
        .progress-section {{
            margin: 20px 0;
        }}
        .progress-bar {{
            background: #0f3460;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #4ecca3, #0077b6);
            height: 100%;
            transition: width 0.3s;
        }}
        .progress-label {{
            text-align: center;
            margin-top: 5px;
            font-size: 12px;
            opacity: 0.7;
        }}
        .log-section {{
            background: #0a0a15;
            border-radius: 4px;
            padding: 10px;
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
        }}
        .log-entry {{ margin: 2px 0; opacity: 0.8; }}
        .visualization-section {{
            margin: 20px 0;
        }}
        .matrix-container {{
            background: #0a0a15;
            border-radius: 8px;
            padding: 10px;
        }}
        .matrix-title {{
            font-size: 12px;
            opacity: 0.7;
            margin-bottom: 10px;
        }}
        #matrix-canvas {{
            display: block;
            width: 100%;
            image-rendering: pixelated;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Charly: <span id="sim-name">{name}</span></h1>
            <span id="state-badge" class="state-badge state-stopped">STOPPED</span>
        </div>

        <div class="controls">
            <button class="btn btn-start" onclick="apiStart()" id="btn-start">Start</button>
            <button class="btn btn-pause" onclick="apiPause()" id="btn-pause">Pause</button>
            <button class="btn btn-stop" onclick="apiStop()" id="btn-stop">Stop</button>
            <button class="btn btn-step" onclick="apiStep(1)" id="btn-step1">Step</button>
            <button class="btn btn-step" onclick="apiStep(10)">+10</button>
            <button class="btn btn-step" onclick="apiStep(100)">+100</button>
            <button class="btn btn-step" onclick="apiStep(1000)">+1000</button>
            <button class="btn btn-save" onclick="apiSave()">Save</button>
        </div>

        <div class="progress-section">
            <div class="progress-bar">
                <div class="progress-fill" id="day-progress" style="width: 0%"></div>
            </div>
            <div class="progress-label">Day <span id="day-num">0</span> Progress: <span id="day-pct">0</span>%</div>
        </div>

        <div class="status-panel">
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="iteration">0</div>
                    <div class="metric-label">Iteration</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="day">0</div>
                    <div class="metric-label">Day</div>
                </div>
                <div class="metric">
                    <div class="metric-value positive" id="esp">0</div>
                    <div class="metric-label">ESP (Positive)</div>
                </div>
                <div class="metric">
                    <div class="metric-value negative" id="esn">0</div>
                    <div class="metric-label">ESN (Negative)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="agent-pos">0.500</div>
                    <div class="metric-label">Agent Position</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="lamp-pos">0.500</div>
                    <div class="metric-label">Lamp Position</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="active">0</div>
                    <div class="metric-label">Active Neurons</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="total">0</div>
                    <div class="metric-label">Total Neurons</div>
                </div>
            </div>
        </div>

        <div class="visualization-section">
            <div class="matrix-container">
                <div class="matrix-title">Neuron Activity Matrix (bright=current, medium=step-1, dim=step-2, black=inactive)</div>
                <canvas id="matrix-canvas"></canvas>
            </div>
        </div>

        <div class="log-section" id="log">
            <div class="log-entry">Server started...</div>
        </div>
    </div>

    <script>
        let pollInterval = null;
        let currentState = 'stopped';

        function log(msg) {{
            const logDiv = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = new Date().toLocaleTimeString() + ' - ' + msg;
            logDiv.appendChild(entry);
            logDiv.scrollTop = logDiv.scrollHeight;
        }}

        async function fetchStatus() {{
            try {{
                const resp = await fetch('/api/status');
                const data = await resp.json();
                updateUI(data);
            }} catch (e) {{
                log('Error fetching status: ' + e.message);
            }}
        }}

        function updateUI(data) {{
            // Update state badge
            const badge = document.getElementById('state-badge');
            badge.textContent = data.state.toUpperCase();
            badge.className = 'state-badge state-' + data.state;
            currentState = data.state;

            // Update metrics
            document.getElementById('iteration').textContent = data.iteration.toLocaleString();
            document.getElementById('day').textContent = data.day_index;
            document.getElementById('esp').textContent = data.esp.toLocaleString();
            document.getElementById('esn').textContent = data.esn.toLocaleString();
            document.getElementById('agent-pos').textContent = data.agent_pos.toFixed(3);
            document.getElementById('lamp-pos').textContent = data.lamp_pos.toFixed(3);
            document.getElementById('active').textContent = data.active_neurons.toLocaleString();
            document.getElementById('total').textContent = data.neuron_count.toLocaleString();

            // Update day progress
            const daySteps = data.day_steps || 1080;
            const dayProgress = (data.iteration % daySteps) / daySteps * 100;
            document.getElementById('day-progress').style.width = dayProgress + '%';
            document.getElementById('day-num').textContent = data.day_index;
            document.getElementById('day-pct').textContent = dayProgress.toFixed(1);

            // Update button states
            document.getElementById('btn-start').disabled = (data.state === 'running');
            document.getElementById('btn-pause').disabled = (data.state !== 'running');
            document.getElementById('btn-step1').disabled = (data.state === 'running');
        }}

        async function apiStart() {{
            log('Starting simulation...');
            await fetch('/api/start', {{ method: 'POST' }});
            startPolling();
            fetchStatus();
        }}

        async function apiPause() {{
            log('Pausing simulation...');
            await fetch('/api/pause', {{ method: 'POST' }});
            stopPolling();
            fetchStatus();
        }}

        async function apiStop() {{
            log('Stopping simulation...');
            await fetch('/api/stop', {{ method: 'POST' }});
            stopPolling();
            fetchStatus();
        }}

        async function apiStep(count) {{
            log('Stepping ' + count + ' iteration(s)...');
            await fetch('/api/step', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ count: count }})
            }});
            fetchStatus();
        }}

        async function apiSave() {{
            log('Saving state...');
            const resp = await fetch('/api/save', {{ method: 'POST' }});
            const data = await resp.json();
            if (data.success) {{
                log('State saved successfully');
            }} else {{
                log('Save failed: ' + (data.error || 'unknown error'));
            }}
        }}

        function startPolling() {{
            if (!pollInterval) {{
                pollInterval = setInterval(fetchStatus, 500);
            }}
        }}

        function stopPolling() {{
            if (pollInterval) {{
                clearInterval(pollInterval);
                pollInterval = null;
            }}
        }}

        // Matrix visualization
        const matrixCanvas = document.getElementById('matrix-canvas');
        const matrixCtx = matrixCanvas.getContext('2d');
        let matrixWidth = 0;
        let matrixHeight = 0;
        let imageData = null;

        async function fetchNeurons() {{
            try {{
                const resp = await fetch('/api/neurons');
                const data = await resp.json();
                if (data.neurons && data.width && data.height) {{
                    drawMatrix(data.neurons, data.width, data.height);
                }}
            }} catch (e) {{
                // Silently fail - will retry on next poll
            }}
        }}

        function drawMatrix(neurons, width, height) {{
            // Resize canvas if dimensions changed
            if (matrixWidth !== width || matrixHeight !== height) {{
                matrixWidth = width;
                matrixHeight = height;
                matrixCanvas.width = width;
                matrixCanvas.height = height;
                imageData = matrixCtx.createImageData(width, height);
            }}

            // Color lookup: brightness levels for activation states
            // State 0: black (inactive)
            // State 1: very dim (step-2)
            // State 2: dim (step-1)
            // State 3: bright (current)
            const brightnessLevels = [0, 60, 140, 255];

            // Fill pixel data
            for (let i = 0; i < neurons.length; i++) {{
                const neuron = neurons[i];
                const pixelIndex = i * 4;

                const state = neuron.s;
                const brightness = brightnessLevels[state] || 0;

                let r = 0, g = 0, b = 0;

                if (state > 0) {{
                    // Active (current or recent): color based on EQ
                    if (neuron.eq >= 0) {{
                        // Positive EQ: green
                        g = brightness;
                    }} else {{
                        // Negative EQ: red
                        r = brightness;
                    }}
                }}
                // State 0: black (r=g=b=0)

                imageData.data[pixelIndex] = r;
                imageData.data[pixelIndex + 1] = g;
                imageData.data[pixelIndex + 2] = b;
                imageData.data[pixelIndex + 3] = 255;  // Alpha
            }}

            // Draw to canvas
            matrixCtx.putImageData(imageData, 0, 0);
        }}

        // Fetch neurons periodically
        setInterval(fetchNeurons, 200);

        // Initial load
        fetchStatus();
        fetchNeurons();
        log('Connected to simulation: {name}');
    </script>
</body>
</html>
'''


class CharlyRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for simulation control."""

    # Class-level reference to simulation manager (set before starting server)
    simulation_manager: Optional[SimulationManager] = None

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _send_json(self, data: Dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_html(self, html: str, status: int = 200):
        """Send HTML response."""
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '/index.html':
            self._serve_html()
        elif parsed.path == '/api/status':
            self._api_status()
        elif parsed.path == '/api/neurons':
            self._api_neurons()
        else:
            self._not_found()

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)

        if parsed.path == '/api/start':
            self._api_start()
        elif parsed.path == '/api/pause':
            self._api_pause()
        elif parsed.path == '/api/stop':
            self._api_stop()
        elif parsed.path == '/api/step':
            self._api_step()
        elif parsed.path == '/api/save':
            self._api_save()
        else:
            self._not_found()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _serve_html(self):
        """Serve the HTML control page."""
        if self.simulation_manager:
            name = self.simulation_manager.name
        else:
            name = 'Unknown'
        html = HTML_TEMPLATE.format(name=name)
        self._send_html(html)

    def _api_status(self):
        """GET /api/status - Return simulation status."""
        if self.simulation_manager:
            status = self.simulation_manager.get_status()
            self._send_json(status)
        else:
            self._send_json({'error': 'No simulation'}, 500)

    def _api_neurons(self):
        """GET /api/neurons - Return neuron states for visualization."""
        if self.simulation_manager:
            data = self.simulation_manager.get_neurons(100)
            self._send_json(data)
        else:
            self._send_json({'error': 'No simulation'}, 500)

    def _api_start(self):
        """POST /api/start - Start simulation."""
        if self.simulation_manager:
            success = self.simulation_manager.start()
            self._send_json({'success': success})
        else:
            self._send_json({'error': 'No simulation'}, 500)

    def _api_pause(self):
        """POST /api/pause - Pause simulation."""
        if self.simulation_manager:
            success = self.simulation_manager.pause()
            self._send_json({'success': success})
        else:
            self._send_json({'error': 'No simulation'}, 500)

    def _api_stop(self):
        """POST /api/stop - Stop simulation."""
        if self.simulation_manager:
            success = self.simulation_manager.stop()
            self._send_json({'success': success})
        else:
            self._send_json({'error': 'No simulation'}, 500)

    def _api_step(self):
        """POST /api/step - Execute N steps."""
        if not self.simulation_manager:
            self._send_json({'error': 'No simulation'}, 500)
            return

        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'

        try:
            data = json.loads(body)
            count = data.get('count', 1)
        except json.JSONDecodeError:
            count = 1

        result = self.simulation_manager.step(count)
        self._send_json({'success': True, 'result': result})

    def _api_save(self):
        """POST /api/save - Force save state."""
        if self.simulation_manager:
            success = self.simulation_manager.save()
            self._send_json({'success': success})
        else:
            self._send_json({'error': 'No simulation'}, 500)

    def _not_found(self):
        """Send 404 response."""
        self._send_json({'error': 'Not found'}, 404)


def find_free_port(start: int = 8000, end: int = 9000) -> int:
    """
    Find an available port in the given range.

    Args:
        start: First port to try
        end: Last port to try

    Returns:
        Available port number

    Raises:
        RuntimeError: If no free port is found
    """
    for port in range(start, end):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"No free ports available in range {start}-{end}")


def run_server(name: str, config_path: Optional[str], port: int = 0) -> None:
    """
    Start the HTTP server for simulation control.

    Args:
        name: Simulation name
        config_path: Path to config YAML (optional if resuming)
        port: HTTP port (0 for auto-allocate)
    """
    # Create and initialize simulation manager
    manager = SimulationManager(name)

    if not manager.initialize(config_path):
        print("Failed to initialize simulation")
        return

    # Find port
    if port == 0:
        port = find_free_port()

    # Set up request handler
    CharlyRequestHandler.simulation_manager = manager

    # Create and start server
    server = HTTPServer(('', port), CharlyRequestHandler)

    print(f"\n{'='*60}")
    print(f"  Charly Simulation Server")
    print(f"  Simulation: {name}")
    print(f"  Directory:  {manager.sim_dir}")
    print(f"{'='*60}")
    print(f"\n  Open in browser: http://localhost:{port}/")
    print(f"\n  Press Ctrl+C to stop the server")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        manager.stop()
        server.shutdown()
        print("Server stopped.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Charly Simulation Server')
    parser.add_argument('--name', '-n', required=True, help='Simulation name')
    parser.add_argument('--config', '-c', help='Config file path')
    parser.add_argument('--port', '-p', type=int, default=0, help='HTTP port')

    args = parser.parse_args()
    run_server(args.name, args.config, args.port)
