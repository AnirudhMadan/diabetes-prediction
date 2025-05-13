import subprocess
import time
import os

# Run app.py (Flask or other backend)
frontend = subprocess.Popen(["python","-m","streamlit", "run", "streamlit_app.py"])


# Optional delay to let the backend initialize
time.sleep(3)

# Run Streamlit frontend
backend = subprocess.Popen(["python", "app2.py"])


# Wait for both to complete (they won't unless stopped manually)
try:
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    backend.terminate()
    frontend.terminate()
