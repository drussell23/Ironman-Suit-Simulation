#!/usr/bin/env python3
"""
Start the Unity Bridge backend for Iron Man suit visualization.
This provides the REST API that Unity connects to.
"""

import subprocess
import sys
import os
import time
import signal

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✓ FastAPI dependencies found")
        return True
    except ImportError:
        print("✗ Missing FastAPI dependencies")
        print("\nInstall with:")
        print("  pip install fastapi uvicorn pydantic")
        return False

def start_backend():
    """Start the Unity Bridge API."""
    print("\n=== Iron Man Unity Bridge ===")
    print("Starting backend API server...\n")
    
    # Add backend to path
    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    sys.path.insert(0, backend_path)
    
    try:
        # Start uvicorn
        import uvicorn
        from backend.api.unity_bridge import app
        
        print("Server starting on http://localhost:8000")
        print("\nEndpoints:")
        print("  GET  /api/flight/state - Get current flight state")
        print("  POST /api/control      - Send control commands")
        print("  GET  /api/telemetry    - Get detailed telemetry")
        print("  POST /api/reset        - Reset simulation")
        print("\nPress Ctrl+C to stop\n")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError starting server: {e}")
        return 1
    
    return 0

def print_unity_instructions():
    """Print instructions for Unity setup."""
    print("\n" + "="*50)
    print("Unity Setup Instructions")
    print("="*50)
    print("\n1. Open Unity Hub")
    print("2. Add Project: simulation/unity_simulation_ui/Iron-Man-Suit-Simulation")
    print("3. Open with Unity 2022.3.x LTS")
    print("\n4. In Unity:")
    print("   - Open Scenes/SampleScene")
    print("   - Add BackendConnector script to IronManSuit")
    print("   - Set Backend URL to: http://localhost:8000")
    print("   - Press Play")
    print("\n5. Controls in Unity:")
    print("   - WASD: Movement")
    print("   - Space: Thrust")
    print("   - Q/E: Rotate")
    print("\nSee simulation/unity_simulation_ui/UNITY_SETUP.md for details")

def main():
    """Main entry point."""
    # Check dependencies
    if not check_dependencies():
        print_unity_instructions()
        return 1
    
    # Print instructions
    print_unity_instructions()
    
    print("\n" + "="*50)
    input("\nPress Enter to start the backend server...")
    
    # Start backend
    return start_backend()

if __name__ == "__main__":
    sys.exit(main())