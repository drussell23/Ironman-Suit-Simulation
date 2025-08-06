"""
Pytest configuration for AR/VR integration tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_camera_data():
    """Provide mock camera data for testing"""
    return {
        'resolution': (1920, 1080),
        'fov': 90,
        'position': np.array([0, 0, 0]),
        'rotation': np.array([0, 0, 0, 1])
    }


@pytest.fixture
def mock_tracking_data():
    """Provide mock tracking data for testing"""
    return {
        'position': np.array([0, 1.6, 0]),  # Eye height
        'rotation': np.array([0, 0, 0, 1]),
        'velocity': np.array([0, 0, 0]),
        'acceleration': np.array([0, 0, 0]),
        'confidence': 0.95
    }


@pytest.fixture
def mock_hand_data():
    """Provide mock hand tracking data for testing"""
    return {
        'left': {
            'position': np.array([-0.2, 1.2, 0.5]),
            'joints': {
                'wrist': np.array([-0.2, 1.2, 0.5]),
                'thumb_tip': np.array([-0.15, 1.25, 0.5]),
                'index_tip': np.array([-0.15, 1.3, 0.5])
            },
            'confidence': 0.9
        },
        'right': {
            'position': np.array([0.2, 1.2, 0.5]),
            'joints': {
                'wrist': np.array([0.2, 1.2, 0.5]),
                'thumb_tip': np.array([0.15, 1.25, 0.5]),
                'index_tip': np.array([0.15, 1.3, 0.5])
            },
            'confidence': 0.9
        }
    }


@pytest.fixture
def mock_eye_data():
    """Provide mock eye tracking data for testing"""
    return {
        'left_pupil_position': np.array([-0.1, 0]),
        'right_pupil_position': np.array([0.1, 0]),
        'left_pupil_diameter': 3.5,
        'right_pupil_diameter': 3.5,
        'left_eye_openness': 1.0,
        'right_eye_openness': 1.0,
        'confidence': 0.85
    }


@pytest.fixture
def cleanup_threads():
    """Ensure all threads are stopped after tests"""
    yield
    # Cleanup will happen after test completes
    import threading
    for thread in threading.enumerate():
        if thread.name.startswith('test_'):
            thread.join(timeout=1.0)