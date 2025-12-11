import subprocess
import time
import os
import signal
import sys
from pathlib import Path

def start_dashboard():
    # Get current directory
    base_dir = Path(__file__).parent.absolute()
    frontend_dir = base_dir / "frontend"
    venv_dir = base_dir / "cryptovision_py39_env"
    
    # Check for virtual environment
    if not venv_dir.exists():
        print(f"❌ Virtual environment not found at {venv_dir}")
        print("Please run ./setup_and_run.sh first or create the environment.")
        sys.exit(1)
        
    print("🚀 Starting CryptoVisionAI Dashboard...")
    
    # Determine python and uvicorn paths
    if sys.platform == "win32":
        python_exe = venv_dir / "Scripts" / "python.exe"
        uvicorn_exe = venv_dir / "Scripts" / "uvicorn.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        uvicorn_exe = venv_dir / "bin" / "uvicorn"
        
    print(f"Using python: {python_exe}")
    
    # Start Backend
    print("\n🐍 Starting Backend (FastAPI)...")
    backend_cmd = [
        str(uvicorn_exe), 
        "src.api.main:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    
    try:
        backend_process = subprocess.Popen(
            backend_cmd,
            cwd=str(base_dir),
            env=os.environ.copy()
        )
    except FileNotFoundError:
         print(f"❌ Failed to start backend. Is uvicorn installed in {venv_dir}?")
         sys.exit(1)
    
    # Start Frontend
    print("\n⚛️  Starting Frontend (Vite)...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_dir),
        env=os.environ.copy()
    )
    
    print("\n✅ Services started!")
    print("   Backend: http://localhost:8000")
    print("   Frontend: http://localhost:5173")
    print("\nPress Ctrl+C to stop all services...")
    
    try:
        while True:
            time.sleep(1)
            if backend_process.poll() is not None:
                print("⚠️ Backend process ended unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("⚠️ Frontend process ended unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
    finally:
        if backend_process.poll() is None:
            backend_process.terminate()
        if frontend_process.poll() is None:
            frontend_process.terminate()
        
        # Wait for processes to exit
        backend_process.wait()
        frontend_process.wait()
        print("👋 Shutdown complete.")

if __name__ == "__main__":
    start_dashboard()
