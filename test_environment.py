#!/usr/bin/env python3
"""
Test script to verify the Iron Man AI environment is properly set up.
"""

import sys
import os


def test_basic_imports():
    """Test basic Python imports."""
    print("🔧 Testing basic Python imports...")

    try:
        import numpy as np

        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False

    try:
        import scipy

        print(f"✅ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"❌ SciPy import failed: {e}")
        return False

    try:
        import pandas as pd

        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False

    try:
        import matplotlib

        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False

    return True


def test_ai_imports():
    """Test AI/ML library imports."""
    print("\n🤖 Testing AI/ML library imports...")

    try:
        import torch

        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        import sklearn

        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False

    return True


def test_ironman_modules():
    """Test Iron Man specific modules."""
    print("\n🛡️ Testing Iron Man AI modules...")

    # Add the backend directory to Python path
    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    try:
        from adaptive_ai.reinforcement_learning import DQN, PPO, SAC

        print("✅ Reinforcement Learning modules imported")
    except ImportError as e:
        print(f"❌ RL modules import failed: {e}")
        return False

    try:
        from adaptive_ai.tactical_decision import ThreatAssessment, MissionPlanner

        print("✅ Tactical Decision modules imported")
    except ImportError as e:
        print(f"❌ Tactical Decision modules import failed: {e}")
        return False

    try:
        from adaptive_ai.behavioral_adaptation import (
            PilotPreferenceModel,
            AdaptiveController,
        )

        print("✅ Behavioral Adaptation modules imported")
    except ImportError as e:
        print(f"❌ Behavioral Adaptation modules import failed: {e}")
        return False

    try:
        from adaptive_ai.predictive_analytics import (
            ThreatPredictor,
            PerformanceOptimizer,
            AnomalyDetector,
        )

        print("✅ Predictive Analytics modules imported")
    except ImportError as e:
        print(f"❌ Predictive Analytics modules import failed: {e}")
        return False

    try:
        from adaptive_ai.cognitive_load import WorkloadAssessor, AutomationManager

        print("✅ Cognitive Load modules imported")
    except ImportError as e:
        print(f"❌ Cognitive Load modules import failed: {e}")
        return False

    return True


def test_advanced_modules():
    """Test advanced AI modules."""
    print("\n🚀 Testing Advanced AI modules...")

    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    try:
        from adaptive_ai.advanced_neural_architectures import (
            TransformerPolicy,
            GraphNeuralNetwork,
        )

        print("✅ Advanced Neural Architectures imported")
    except ImportError as e:
        print(f"❌ Advanced Neural Architectures import failed: {e}")
        return False

    try:
        from adaptive_ai.advanced_reinforcement_learning import (
            AdvancedPPO,
            AdvancedSAC,
            TD3,
        )

        print("✅ Advanced Reinforcement Learning imported")
    except ImportError as e:
        print(f"❌ Advanced RL import failed: {e}")
        return False

    try:
        from adaptive_ai.meta_learning import MAML, Reptile, PrototypicalNetwork

        print("✅ Meta-Learning modules imported")
    except ImportError as e:
        print(f"❌ Meta-Learning import failed: {e}")
        return False

    try:
        from adaptive_ai.neural_evolution import (
            NEAT,
            GeneticAlgorithm,
            EvolutionaryStrategies,
        )

        print("✅ Neural Evolution modules imported")
    except ImportError as e:
        print(f"❌ Neural Evolution import failed: {e}")
        return False

    try:
        from adaptive_ai.advanced_decision_making import MCTS, BayesianOptimizer, MCDA

        print("✅ Advanced Decision Making imported")
    except ImportError as e:
        print(f"❌ Advanced Decision Making import failed: {e}")
        return False

    return True


def test_system_integration():
    """Test the main AI system integration."""
    print("\n⚡ Testing AI System Integration...")

    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    try:
        from adaptive_ai.ai_system_integration import (
            AISystemCoordinator,
            AISystemConfig,
        )

        print("✅ AI System Integration imported")

        # Test configuration
        config = AISystemConfig()
        print(f"   Update frequency: {config.update_frequency} Hz")
        print(f"   Max response time: {config.max_response_time} seconds")

    except ImportError as e:
        print(f"❌ AI System Integration import failed: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality of AI components."""
    print("\n🧪 Testing basic functionality...")

    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    try:
        import numpy as np
        from adaptive_ai.reinforcement_learning import DQN

        # Test DQN initialization
        dqn = DQN(state_dim=10, action_dim=4)
        print("✅ DQN initialized successfully")

        # Test forward pass
        state = np.random.randn(10)
        action = dqn.select_action(state)
        print(f"✅ DQN action selection: {action}")

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

    return True


def main():
    """Main test function."""
    print("🛡️ Iron Man AI Environment Test")
    print("=" * 50)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("AI/ML Libraries", test_ai_imports),
        ("Iron Man Modules", test_ironman_modules),
        ("Advanced AI Modules", test_advanced_modules),
        ("System Integration", test_system_integration),
        ("Basic Functionality", test_basic_functionality),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your Iron Man AI environment is ready!")
        print("\n🚀 Next steps:")
        print("   1. Select the Python interpreter: ironman_env/bin/python")
        print("   2. Start developing your Iron Man suit AI!")
        print("   3. Run: python test_environment.py to verify again")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you're using the ironman_env virtual environment")
        print("   2. Install missing dependencies: pip install -r requirements.txt")
        print("   3. Check Python path and module imports")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
