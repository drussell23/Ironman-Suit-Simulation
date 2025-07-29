#!/usr/bin/env python3
"""
Start the Enhanced Aerodynamics Backend for Iron Man suit simulation.
This provides full integration between Unity and Python physics calculations.
"""

import subprocess
import sys
import os
import time
import signal
import logging

def check_dependencies():
    """Check if required packages are installed."""
    missing_packages = []
    required_packages = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('pydantic', 'pydantic'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
    ]
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} found")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name} missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_physics_plugin():
    """Check if C++ physics plugin is available."""
    try:
        from backend.aerodynamics.physics_plugin.python.bindings import cfd_compute_forces
        print("✓ C++ Physics Plugin available")
        return True
    except ImportError as e:
        print(f"⚠ C++ Physics Plugin not available: {e}")
        print("  The backend will work with Python-only calculations")
        return False

def build_physics_plugin():
    """Build the C++ physics plugin if needed."""
    plugin_dir = "backend/aerodynamics/physics_plugin"
    build_dir = os.path.join(plugin_dir, "build")
    
    if not os.path.exists(plugin_dir):
        print("⚠ Physics plugin directory not found")
        return False
    
    print("Building C++ physics plugin...")
    
    try:
        # Create build directory
        os.makedirs(build_dir, exist_ok=True)
        
        # Run CMake
        cmake_cmd = ["cmake", "-DENABLE_TESTING=OFF", ".."]
        result = subprocess.run(cmake_cmd, cwd=build_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"CMake failed: {result.stderr}")
            return False
        
        # Build
        build_cmd = ["cmake", "--build", ".", "--", "-j4"]
        result = subprocess.run(build_cmd, cwd=build_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return False
        
        print("✓ Physics plugin built successfully")
        return True
        
    except FileNotFoundError:
        print("⚠ CMake not found. Install CMake to build physics plugin")
        return False
    except Exception as e:
        print(f"⚠ Build failed: {e}")
        return False

def start_backend():
    """Start the aerodynamics backend server."""
    print("\n" + "="*60)
    print("🚀 Iron Man Aerodynamics Backend Starting...")
    print("="*60)
    
    # Add backend to path
    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    sys.path.insert(0, backend_path)
    
    try:
        # Import and start the enhanced backend
        import uvicorn
        from backend.api.aerodynamics_bridge import app
        
        print("\n🌐 Server starting on http://localhost:8001")
        print("\n📡 Enhanced API Endpoints:")
        print("  GET  /api/aerodynamics/state - Complete aerodynamic state")
        print("  POST /api/aerodynamics/state - Update from Unity calculations")
        print("  GET  /api/wind/state        - Wind and turbulence data")
        print("  POST /api/wind/state        - Update wind from Unity")
        print("  GET  /api/sensors/data      - All sensor readings")
        print("  POST /api/control/input     - Control commands from Unity")
        print("  GET  /api/turbulence/data   - Turbulence model data")
        print("  POST /api/simulation/step   - Advance simulation timestep")
        print("  WS   /ws/telemetry          - Real-time telemetry stream")
        
        print("\n🎮 Unity Integration:")
        print("  1. Add AerodynamicsConnector script to IronManSuit")
        print("  2. Set Backend URL to: http://localhost:8001")
        print("  3. Enable desired sync options")
        print("  4. Press Play in Unity")
        
        print("\n🔧 Advanced Features:")
        print("  • Bidirectional physics synchronization")
        print("  • Real-time sensor data exchange")
        print("  • Turbulence model integration")
        print("  • Flight stability control")
        print("  • C++ physics plugin support")
        
        print(f"\n⚡ Physics Plugin: {'Enabled' if check_physics_plugin() else 'Disabled'}")
        print("\nPress Ctrl+C to stop\n")
        
        # Start server
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8001,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down aerodynamics backend...")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return 1
    
    return 0

def print_unity_setup():
    """Print Unity setup instructions."""
    print("\n" + "="*60)
    print("🎮 Unity Setup Instructions")
    print("="*60)
    
    print("\n1. 📁 Open Unity Project:")
    print("   simulation/unity_simulation_ui/Iron-Man-Suit-Simulation")
    
    print("\n2. 🔧 Setup Iron Man Suit GameObject:")
    print("   • Add Rigidbody component")
    print("   • Add all Aerodynamics scripts:")
    print("     - AerodynamicForces")
    print("     - AtmosphericDensity") 
    print("     - StabilityControl")
    print("     - WindInteraction")
    print("     - IMUEmulator")
    print("     - PitotEmulator")
    print("     - KEpsilonModel or SmagorinskyModel")
    print("   • Add AerodynamicsConnector")
    
    print("\n3. ⚙️ Configure AerodynamicsConnector:")
    print("   • Backend URL: http://localhost:8001")
    print("   • Enable desired sync options")
    print("   • Choose physics authority (Unity vs Backend)")
    
    print("\n4. 🎯 Input Setup:")
    print("   • Configure Input Manager for:")
    print("     - Thrust (e.g., Space key)")
    print("     - Pitch/Roll/Yaw (e.g., WASD + QE)")
    
    print("\n5. 🚀 Run Simulation:")
    print("   • Start this backend first")
    print("   • Press Play in Unity")
    print("   • Check connection status in Scene view")

def print_troubleshooting():
    """Print troubleshooting guide."""
    print("\n" + "="*60)
    print("🔧 Troubleshooting")
    print("="*60)
    
    print("\n🔗 Connection Issues:")
    print("   • Check backend is running on port 8001")
    print("   • Verify Unity Backend URL setting")
    print("   • Check firewall settings")
    print("   • Look for CORS errors in browser console")
    
    print("\n⚡ Performance Issues:")
    print("   • Reduce sync frequency in AerodynamicsConnector")
    print("   • Disable unused sync options")
    print("   • Lower turbulence model grid resolution")
    print("   • Use k-epsilon instead of Smagorinsky for better performance")
    
    print("\n🎯 Physics Issues:")
    print("   • Check which system is physics authority")
    print("   • Verify Rigidbody settings (mass, drag)")
    print("   • Adjust PID controller gains in StabilityControl")
    print("   • Check atmospheric model settings")
    
    print("\n📊 Debugging:")
    print("   • Enable Debug.Log in AerodynamicsConnector")
    print("   • Monitor /ws/telemetry WebSocket for real-time data")
    print("   • Check Unity Console for connection errors")
    print("   • Use Gizmos to visualize forces and wind")

def main():
    """Main entry point."""
    print("🚁 Iron Man Aerodynamics Backend")
    print("Advanced Unity-Python Physics Integration")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print_unity_setup()
        return 1
    
    # Try to build physics plugin
    plugin_available = check_physics_plugin()
    if not plugin_available:
        print("\n🔨 Attempting to build physics plugin...")
        if build_physics_plugin():
            plugin_available = check_physics_plugin()
    
    # Print setup instructions
    print_unity_setup()
    print_troubleshooting()
    
    print("\n" + "="*60)
    input("📡 Press Enter to start the aerodynamics backend server...")
    
    # Start backend
    return start_backend()

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)