const API_URL = "http://localhost:8000/api";

const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnStep = document.getElementById('btn-step');
const statusDiv = document.getElementById('status');
const visImage = document.getElementById('vis-image');

let pollingInterval = null;

async function updateStatus() {
    try {
        const response = await fetch(`${API_URL}/status`);

        if (!response.ok) {
            const err = await response.json();
            statusDiv.textContent = `Error: ${err.detail || response.statusText}`;
            return;
        }

        const data = await response.json();
        statusDiv.textContent = `Time: ${new Date().toLocaleTimeString()} | Step: ${data.step} | Day: ${data.day} | ${data.running ? 'RUNNING' : 'PAUSED'}`;

        // Refresh image
        // Always refresh if we haven't loaded one yet, or if running.
        // Even if paused, we want to see the static state.
        if (data.running || !visImage.src || visImage.src.endsWith('src=""')) {
            visImage.src = `${API_URL}/visual/frame?t=${Date.now()}`;
        }
    } catch (e) {
        console.error(e);
        statusDiv.textContent = "Server Offline";
    }
}

async function sendControl(action) {
    try {
        const res = await fetch(`${API_URL}/control/${action}`, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            alert(`Error: ${err.detail}`);
            return;
        }
        updateStatus();
        visImage.src = `${API_URL}/visual/frame?t=${Date.now()}`;
    } catch (e) {
        alert("Control Command Failed");
    }
}

btnStart.onclick = () => sendControl('start');
btnStop.onclick = () => sendControl('stop');
btnStep.onclick = () => sendControl('step');

// Start polling status
setInterval(updateStatus, 500); // 2Hz polling for status/image
updateStatus();
