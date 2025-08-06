#!/usr/bin/env python3
"""
Test script for Iron Man Suit Simulation Backend
Validates all major components before Unity integration
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

def test_basic_imports():
    """Test basic Python package imports."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        print("✓ NumPy and Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import basic packages: {e}")
        return False
    
    return True

def test_ai_imports():
    """Test AI module imports."""
    print("\nTesting AI module imports...")
    
    try:
        from adaptive_ai import (
            DQNAgent, PPOAgent, SACAgent, MultiAgentCoordinator,
            TacticalDecisionEngine, AdaptiveAISystem
        )
        print("✓ AI modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AI modules: {e}")
        return False
    
    return True

def test_aerodynamics_imports():
    """Test aerodynamics module imports."""
    print("\nTesting aerodynamics imports...")
    
    try:
        from aerodynamics.flight_models.flight_dynamics import FlightDynamics
        print("✓ Flight dynamics imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import flight dynamics: {e}")
        return False
    
    return True

def test_basic_ai_functionality():
    """Test basic AI agent functionality."""
    print("\nTesting basic AI functionality...")
    
    try:
        from adaptive_ai import DQNAgent
        
        # Create a simple DQN agent
        state_dim = 4
        action_dim = 2
        agent = DQNAgent(state_dim, action_dim, device="cpu")
        
        # Test action selection
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        
        print(f"✓ DQN agent created and selected action: {action}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test AI functionality: {e}")
        return False

def test_flight_dynamics():
    """Test flight dynamics simulation."""
    print("\nTesting flight dynamics...")
    
    try:
        from aerodynamics.flight_models.flight_dynamics import FlightDynamics
        
        # Create flight dynamics model
        flight_model = FlightDynamics()
        
        # Test basic simulation step
        state = np.array([0.0, 0.0, 1000.0, 0.0, 0.0, 0.0])  # x, y, z, vx, vy, vz
        controls = np.array([0.1, 0.0, 0.0])  # thrust, pitch, roll
        
        # Simulate one step
        dt = 0.01
        new_state = flight_model.step(state, controls, dt)
        
        print(f"✓ Flight dynamics step completed. New state: {new_state[:3]}")  # Show position
        return True
        
    except Exception as e:
        print(f"✗ Failed to test flight dynamics: {e}")
        return False

def test_simulation_runner():
    """Test the main simulation runner."""
    print("\nTesting simulation runner...")
    
    try:
        from aerodynamics.simulations.run_simulation import IronManSimulation
        
        # Create simulation
        sim = IronManSimulation()
        
        # Run a short simulation
        results = sim.run_simulation(duration=1.0, dt=0.01)
        
        print(f"✓ Simulation completed. Final altitude: {results['altitude'][-1]:.1f}m")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test simulation runner: {e}")
        return False

def test_adaptive_ai_system():
    """Test the adaptive AI system integration."""
    print("\nTesting adaptive AI system...")
    
    try:
        from adaptive_ai import AdaptiveAISystem, AISystemConfig
        
        # Create AI system configuration
        config = AISystemConfig(
            use_reinforcement_learning=True,
            use_tactical_decision=True,
            use_predictive_analytics=True,
            use_cognitive_load=True
        )
        
        # Create AI system
        ai_system = AdaptiveAISystem(config)
        
        # Test basic decision making
        state = np.random.randn(10)  # 10-dimensional state
        decision = ai_system.make_decision(state)
        
        print(f"✓ Adaptive AI system created and made decision: {type(decision)}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test adaptive AI system: {e}")
        return False

def run_performance_test():
    """Run a performance test to ensure real-time capability."""
    print("\nRunning performance test...")
    
    try:
        import time
        from adaptive_ai import DQNAgent
        from aerodynamics.flight_models.flight_dynamics import FlightDynamics
        
        # Create components
        agent = DQNAgent(4, 2, device="cpu")
        flight_model = FlightDynamics()
        
        # Performance test
        n_steps = 1000
        start_time = time.time()
        
        for i in range(n_steps):
            state = np.random.randn(4)
            action = agent.select_action(state)
            
            flight_state = np.random.randn(6)
            controls = np.random.randn(3)
            new_state = flight_model.step(flight_state, controls, 0.01)
        
        end_time = time.time()
        elapsed = end_time - start_time
        steps_per_second = n_steps / elapsed
        
        print(f"✓ Performance test completed: {steps_per_second:.1f} steps/second")
        print(f"  Target: >100 steps/second for real-time operation")
        
        if steps_per_second > 100:
            print("  ✓ Real-time performance achieved!")
        else:
            print("  ⚠ Performance below real-time target")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to run performance test: {e}")
        return False

def main():
    """Run all tests."""
    print("Iron Man Suit Simulation - Backend Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_ai_imports,
        test_aerodynamics_imports,
        test_basic_ai_functionality,
        test_flight_dynamics,
        test_simulation_runner,
        test_adaptive_ai_system,
        run_performance_test
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your backend is ready for Unity integration.")
        print("\nNext steps:")
        print("1. Set up Unity with ML-Agents")
        print("2. Create Iron Man suit model")
        print("3. Connect Python backend to Unity")
        print("4. Start testing in 3D environment")
    else:
        print("⚠ Some tests failed. Please fix the issues before Unity integration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 