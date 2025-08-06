"""
AR/VR Integration Module for Iron Man Suit Simulation

This module provides augmented reality (AR) and virtual reality (VR)
capabilities for the Iron Man suit's heads-up display (HUD) and 
immersive control interfaces.
"""

from .augmented_hud import AugmentedHUD
from .virtual_interface import VirtualInterface
from .core import ARVRCore
from .gesture_recognition import GestureRecognizer
from .holographic_display import HolographicDisplay
from .spatial_tracking import SpatialTracker
from .vr_training import VRTrainingEnvironment
from .eye_tracking import EyeTracker
from .ar_overlays import AROverlayManager

__all__ = [
    'AugmentedHUD',
    'VirtualInterface',
    'ARVRCore',
    'GestureRecognizer',
    'HolographicDisplay',
    'SpatialTracker',
    'VRTrainingEnvironment',
    'EyeTracker',
    'AROverlayManager'
]

__version__ = '1.0.0'