import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
import json
import shutil
import signal

# Paths
SERVER_SCRIPT = os.path.abspath("src/charly_server.py")
TEST_DIR = os.path.abspath("test_sim_dir")
CONFIG_PATH = os.path.abspath("config/config.yaml")

API_URL = "http://localhost:8002/api"
PORT = 8002

def cleanup():
    if os.path.exists(TEST_DIR):
        try:
            shutil.rmtree(TEST_DIR)
        except Exception:
            pass

def get_json(url):
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError:
        raise

def post_json(url):
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError:
        raise

def wait_for_server():
    for _ in range(20):
        try:
            get_json(f"{API_URL}/status")
            return True
        except (urllib.error.URLError, ConnectionResetError):
            time.sleep(0.5)
    return False

def test_persistence():
    print("=== Testing Persistence Flow (urllib) ===")
    cleanup()
    
    # 1. Start Server in Init Mode
    print(f"[1] Starting server to init {TEST_DIR}...")
    # Ensure PYTHONPATH is set to src so imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    
    proc = subprocess.Popen(
        [sys.executable, SERVER_SCRIPT, TEST_DIR, CONFIG_PATH, "--port", str(PORT)],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        env=env,
        cwd=os.path.abspath("src")
    )
    
    if not wait_for_server():
        print("Server failed to start!")
        proc.terminate()
        return False
        
    # Check status
    status = get_json(f"{API_URL}/status")
    print(f"Initial Status: {status}")
    assert status['step'] == 0
    assert status['running'] == False
    
    # 2. Run Simulation
    print("[2] Starting simulation...")
    post_json(f"{API_URL}/control/start")
    
    # Wait for some steps
    time.sleep(3)
    
    status = get_json(f"{API_URL}/status")
    print(f"Status after running: {status}")
    steps_run = status['step']
    assert steps_run > 0
    assert status['running'] == True
    
    # 3. Stop and Save
    print("[3] Stopping simulation...")
    post_json(f"{API_URL}/control/stop")
    time.sleep(2) # Wait for save and stop
    
    # Kill server
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
    
    # Verify files
    assert os.path.exists(os.path.join(TEST_DIR, "charly.pkl"))
    assert os.path.exists(os.path.join(TEST_DIR, "world.pkl"))
    print("Save files verified.")
    
    # 4. Resume Simulation
    print("[4] Restarting server to resume...")
    # Note: No config path passed this time
    proc = subprocess.Popen(
        [sys.executable, SERVER_SCRIPT, TEST_DIR, "--port", str(PORT)],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        env=env,
        cwd=os.path.abspath("src")
    )
    
    if not wait_for_server():
        print("Server failed to restart!")
        proc.terminate()
        return False
        
    status = get_json(f"{API_URL}/status")
    print(f"Resumed Status: {status}")
    
    # Verify resumed state
    # Ideally step should be roughly equal to steps_run (maybe +1 due to loop/save timing)
    # But definitely NOT 0
    assert status['step'] >= steps_run, f"Expected step >= {steps_run}, got {status['step']}"
    assert status['running'] == False # Should start paused
    
    print("PASSED: Simulation resumed correctly.")
    
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        
    cleanup()
    return True

if __name__ == "__main__":
    try:
        success = test_persistence()
        if success:
            print("\nALL TESTS PASSED")
            sys.exit(0)
        else:
            print("\nTESTS FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\nEXCEPTION: {e}")
        # cleanup() 
        sys.exit(1)
