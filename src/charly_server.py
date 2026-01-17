#!/usr/bin/env python3
import os
import time
import threading
import argparse
import shutil
import uvicorn
import numpy as np
import base64
import io
import cv2
import yaml
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, Body, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import Charly modules
# Assuming charly.py and physical_model.py are in the same directory or PYTHONPATH
from charly import Charly
from model_linear import Linear
from physical_model import PhysicalModel

# --- Configuration & Defaults ---
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"

# --- Pydantic Models for API ---
class FingerRequest(BaseModel):
    finger_type: str
    x: int
    y: int
    pressed: bool
    right_pressed: bool

class ControlRequest(BaseModel):
    action: str  # start, stop, step, reset
    steps: Optional[int] = 1

# --- Simulator Wrapper ---
class Simulator:
    def __init__(self, sim_dir: str, config_path: str = None):
        self.sim_dir = os.path.abspath(sim_dir)
        self.config_file = os.path.join(self.sim_dir, "config.yaml")
        self.log_file = os.path.join(self.sim_dir, "server.log")

        # Initialize Directory
        self._init_directory(config_path)

        # Load Config
        self.config = self._load_config()
        
        # Threading & Control
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Parameters
        self.world_size = self.config.get('WORLD_SIZE', 1000)
        self.day_steps = self.config.get('DAY_STEPS', 1080)
        self.visual_width = self.config.get('VISUALIZATION_WIDTH', 1920)
        self.visual_height = self.config.get('VISUALIZATION_HEIGHT', 1080)

        # Save Paths
        self.charly_save_path = os.path.join(self.sim_dir, "charly.pkl")
        self.world_save_path = os.path.join(self.sim_dir, "world.pkl")

        # Initialize / Load Simulation
        self._init_simulation()
        
        # Visualization Buffer
        # rolling_idx points to the line currently being written
        self.rolling_buffer = np.zeros((self.visual_height, self.visual_width, 3), dtype=np.uint8)
        self.rolling_idx = 0
        
        # Finger State
        self.finger_state = {
            'active': False,
            'type': 'trigger',
            'neuron_idx': 0,
            'pressed': False,
            'right_pressed': False
        }

    def _init_directory(self, config_path: Optional[str]):
        """Ensure simulation directory exists and has config."""
        if not os.path.exists(self.sim_dir):
            if config_path and os.path.exists(config_path):
                print(f"Creating new simulation directory: {self.sim_dir}")
                os.makedirs(self.sim_dir)
                shutil.copy(config_path, self.config_file)
            else:
                raise ValueError(f"Directory {self.sim_dir} does not exist and no valid config provided.")
        elif not os.path.exists(self.config_file):
            if config_path and os.path.exists(config_path):
                print(f"Copying config to existing directory: {self.sim_dir}")
                shutil.copy(config_path, self.config_file)
            else:
                pass # Assume config exists or might fail later

    def _load_config(self) -> Dict:
        if not os.path.exists(self.config_file):
             raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_physical_model(self) -> PhysicalModel:
        return Linear(
            world_size=self.world_size,
            lamp_mode=self.config.get('LAMP_MODE', 'sine'),
            lamp_amplitude=self.config.get('LAMP_AMPLITUDE', 0.4),
            lamp_period=self.config.get('LAMP_PERIOD', 4000),
            lamp_center=self.config.get('LAMP_CENTER', 0.5),
            agent_speed=self.config.get('AGENT_SPEED', 0.001),
            agent_start=self.config.get('AGENT_START', 0.5),
            orgasm_tolerance=self.config.get('ORGASM_TOLERANCE', 0.10),
            terror_range=self.config.get('TERROR_RANGE', 0.5),
            terror_smoothness=self.config.get('TERROR_SMOOTHNESS', 50),
            orgasm_smoothness=self.config.get('ORGASM_SMOOTHNESS', 50)
        )

    def _init_simulation(self):
        """Load state if exists, else create new."""
        # Create physical model structure
        self.physical_model = self._create_physical_model()

        if os.path.exists(self.charly_save_path) and os.path.exists(self.world_save_path):
            print(f"Resuming simulation from {self.sim_dir}...")
            try:
                self.physical_model.load_state_from_file(self.world_save_path)
                self.charly = Charly.load_state(self.charly_save_path, self.physical_model)
                self.current_step = self.charly.iteration
            except Exception as e:
                print(f"Failed to load save state: {e}. Starting fresh.")
                self.charly = Charly(self.config, self.physical_model)
                self.current_step = 0
                self.physical_model.reset()
        else:
            print(f"Initializing fresh simulation in {self.sim_dir}...")
            self.charly = Charly(self.config, self.physical_model)
            self.current_step = 0
    
    def start(self):
        if not self.running:
            self.running = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("Simulation background loop started.")

    def stop(self):
        if self.running:
            print("Stopping simulation...")
            self.running = False
            self.stop_event.set()
            if self.thread:
                self.thread.join(timeout=2.0)
            self._save()
            print("Simulation stopped and saved.")

    def _save(self):
        """Save current state to disk."""
        # Note: We should be inside a lock when calling this if running,
        # but _save is called from stop() (after loop end) or _single_step (inside lock).
        print(f"Saving state at step {self.current_step}...")
        try:
            self.charly.save_state(self.charly_save_path)
            self.physical_model.save_state_to_file(self.world_save_path)
        except Exception as e:
            print(f"Error saving state: {e}")

    def _run_loop(self):
        """Background simulation loop."""
        while self.running and not self.stop_event.is_set():
            with self.lock:
                self._single_step()
            
            # Tiny sleep to allow context switching/API handling if CPU bound
            # Adjust based on desired speed
            time.sleep(0.001)

    def step(self, n: int = 1):
        """Manual step command."""
        with self.lock:
            for _ in range(n):
                self._single_step()

    def _single_step(self):
        # 1. Apply Sauron's Finger Logic
        if self.finger_state['active']:
            self._apply_finger()

        # 2. Run Charly Step
        result = self.charly.day_step()
        
        # 3. Update Rolling Buffer Visualization
        self._update_visualization(result)
        
        self.current_step += 1
        
        # 4. Handle Day/Night Cycle & Auto-Save
        if self.current_step % self.day_steps == 0:
            self.charly.run_night()
            print(f"End of Day {self.charly.day_index}. Saving...")
            self._save()

    def _apply_finger(self):
        pass # Placeholder

    def _update_visualization(self, result: Dict):
        y = self.rolling_idx
        w = self.visual_width
        
        # Colors (BGR)
        bg = (0, 0, 0)
        pos = (0, 255, 0)
        neg = (0, 0, 255)
        
        row_data = np.zeros((1, w, 3), dtype=np.uint8)
        row_data[:, :] = bg
        
        neurons = self.charly.current
        count = len(neurons)
        
        # Simple mapping
        limit = min(count, w)
        
        # Optimization: vectorizing color assignment would be faster 
        # but iterating is fine for <2000 neurons
        for idx in range(limit):
            n = neurons[idx]
            if n.active:
                row_data[0, idx] = pos if n.eq >= 0 else neg
                
        self.rolling_buffer[y:y+1, :] = row_data
        self.rolling_idx = (self.rolling_idx + 1) % self.visual_height

    def get_visual_frame(self) -> bytes:
        """Return the current rolling visualization as a JPEG image."""
        with self.lock:
            # Copy buffer to avoid tearing during roll/encode
            buf = self.rolling_buffer.copy()
            idx = self.rolling_idx
            
        # Roll buffer so the current line is at the bottom
        # rolling_idx points to the *next* line to write (oldest)
        # So rolling_idx is exactly the shift needed to bring the oldest to the top
        # and the newest (rolling_idx - 1) to the bottom.
        # np.roll shift is positive -> shifts down/right.
        # We want index 'idx' to move to 0.
        # So we roll by -idx.
        rolled = np.roll(buf, -idx, axis=0)
        
        success, encoded_image = cv2.imencode('.jpg', rolled)
        if not success:
            return b""
        return encoded_image.tobytes()

    def get_status(self) -> Dict:
        return {
            "running": self.running,
            "step": self.current_step,
            "day": self.charly.day_index,
            "sim_dir": self.sim_dir
        }

# --- Server Setup ---
app = FastAPI(title="Charly Neural Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Simulator Instance
sim: Optional[Simulator] = None

@app.on_event("startup")
def startup_event():
    global sim
    # If sim wasn't initialized via CLI (e.g. running via uvicorn direct or no args),
    # try to initialize with defaults.
    if sim is None:
        print("Warning: Simulator not initialized via CLI. Attempting default initialization...")
        default_dir = os.path.abspath("./default_sim")
        default_config = os.path.abspath("config/config.yaml")
        
        # Check parent dir for config if not found
        if not os.path.exists(default_config):
             default_config = os.path.abspath("../config/config.yaml")

        try:
            sim = Simulator(default_dir, default_config if os.path.exists(default_config) else None)
            print(f"Initialized default simulator at {default_dir}")
        except Exception as e:
            print(f"Failed to auto-initialize simulator: {e}")

@app.on_event("shutdown")
def shutdown_event():
    if sim:
        sim.stop()

# --- Routes ---

@app.get("/api/status")
def get_status():
    if not sim:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    return sim.get_status()

@app.get("/api/visual/frame")
def get_visual_frame():
    if not sim:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    image_bytes = sim.get_visual_frame()
    return Response(content=image_bytes, media_type="image/jpeg")

@app.post("/api/control/{action}")
def control_sim(action: str):
    if not sim:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    if action == "start":
        sim.start()
    elif action == "stop":
        sim.stop()
    elif action == "step":
        sim.step(1)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
    return {"status": "ok", "action": action}

@app.get("/gui/index.html")
def gui_index():
    return RedirectResponse(url="/index.html")

# Mount static files last to catch-all
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Charly Simulation Server")
    parser.add_argument("sim_dir", type=str, help="Path to simulation directory")
    parser.add_argument("config", nargs="?", type=str, help="Path to initial configuration (UML) file")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    
    args = parser.parse_args()
    
    try:
        sim = Simulator(args.sim_dir, args.config)
        
        # Start server
        uvicorn.run(app, host=DEFAULT_HOST, port=args.port)
    except Exception as e:
        print(f"Fatal Error: {e}")
