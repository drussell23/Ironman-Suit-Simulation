"""
Shared fixtures and configuration for adaptive_ai tests
"""

import pytest
import numpy as np
import torch
import os
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


@pytest.fixture
def sample_state():
    """Sample state vector for testing"""
    return np.array([0.5, -0.3, 0.8, 0.1, -0.2, 0.6, 0.4, -0.5])


@pytest.fixture
def sample_action():
    """Sample action vector for testing"""
    return np.array([0.7, -0.2, 0.3])


@pytest.fixture
def sample_observation_space():
    """Sample observation space dimensions"""
    return 8


@pytest.fixture
def sample_action_space():
    """Sample action space dimensions"""
    return 3


@pytest.fixture
def device():
    """Get appropriate device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def ai_config():
    """Default AI system configuration for testing"""
    return {
        "rl_config": {
            "algorithm": "ppo",
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "buffer_size": 10000
        },
        "tactical_config": {
            "threat_threshold": 0.7,
            "planning_horizon": 100,
            "update_frequency": 10
        },
        "behavioral_config": {
            "learning_rate": 0.001,
            "history_size": 1000,
            "adaptation_rate": 0.1
        },
        "predictive_config": {
            "prediction_horizon": 50,
            "model_type": "lstm",
            "update_interval": 5
        },
        "cognitive_config": {
            "workload_threshold": 0.8,
            "automation_levels": 4,
            "assessment_interval": 1.0
        }
    }


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data for testing"""
    return {
        "position": np.array([100.0, 200.0, 500.0]),
        "velocity": np.array([50.0, 0.0, 10.0]),
        "orientation": np.array([0.0, 0.1, 0.0, 0.99]),  # quaternion
        "angular_velocity": np.array([0.0, 0.0, 0.1]),
        "acceleration": np.array([0.0, 0.0, -9.81]),
        "timestamp": 1234567890.0
    }


@pytest.fixture
def sample_threat_data():
    """Sample threat data for testing"""
    return [
        {
            "id": "threat_1",
            "type": "missile",
            "position": np.array([500.0, 300.0, 600.0]),
            "velocity": np.array([-100.0, -50.0, -20.0]),
            "threat_level": 0.8
        },
        {
            "id": "threat_2",
            "type": "aircraft",
            "position": np.array([1000.0, 500.0, 800.0]),
            "velocity": np.array([-50.0, 0.0, 0.0]),
            "threat_level": 0.5
        }
    ]


@pytest.fixture
def sample_pilot_action():
    """Sample pilot action for behavioral adaptation testing"""
    return {
        "control_input": np.array([0.5, -0.2, 0.0]),
        "mode_selection": "manual",
        "timestamp": 1234567890.0
    }


@pytest.fixture
def torch_available():
    """Check if PyTorch is available"""
    try:
        import torch
        return True
    except ImportError:
        return False