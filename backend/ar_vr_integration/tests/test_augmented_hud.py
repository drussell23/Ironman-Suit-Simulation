"""
Tests for Augmented HUD system.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from backend.ar_vr_integration.augmented_hud import (
    AugmentedHUD, HUDMode, ThreatLevel, Target, HUDElement
)


class TestAugmentedHUD:
    """Test cases for AugmentedHUD class"""
    
    @pytest.fixture
    def hud(self):
        """Create HUD instance for testing"""
        return AugmentedHUD(resolution=(1920, 1080))
    
    def test_initialization(self, hud):
        """Test HUD initialization"""
        assert hud.resolution == (1920, 1080)
        assert hud.mode == HUDMode.FLIGHT
        assert hud.is_active is False
        assert len(hud.targets) == 0
        assert len(hud.elements) > 0  # Default elements
    
    def test_start_stop(self, hud):
        """Test starting and stopping HUD"""
        hud.start()
        assert hud.is_active is True
        assert hud.render_thread is not None
        assert hud.render_thread.is_alive()
        
        hud.stop()
        assert hud.is_active is False
    
    def test_mode_switching(self, hud):
        """Test switching between HUD modes"""
        hud.set_mode(HUDMode.COMBAT)
        assert hud.mode == HUDMode.COMBAT
        assert hud.config['max_targets'] == 100
        
        hud.set_mode(HUDMode.STEALTH)
        assert hud.mode == HUDMode.STEALTH
        
        hud.set_mode(HUDMode.SCAN)
        assert hud.mode == HUDMode.SCAN
        assert hud.config['grid_spacing'] == 25
    
    def test_add_target(self, hud):
        """Test adding targets"""
        target = Target(
            id="enemy_1",
            position=np.array([100, 50, 200]),
            velocity=np.array([10, 0, 0]),
            classification="hostile",
            threat_level=ThreatLevel.HIGH,
            distance=150.0,
            angle=45.0,
            last_seen=time.time(),
            confidence=0.9
        )
        
        hud.add_target(target)
        assert "enemy_1" in hud.targets
        assert hud.targets["enemy_1"].classification == "hostile"
    
    def test_remove_target(self, hud):
        """Test removing targets"""
        target = Target(
            id="target_1",
            position=np.array([0, 0, 0]),
            velocity=np.array([0, 0, 0]),
            classification="unknown",
            threat_level=ThreatLevel.NONE,
            distance=100.0,
            angle=0.0,
            last_seen=time.time(),
            confidence=0.5
        )
        
        hud.add_target(target)
        assert "target_1" in hud.targets
        
        hud.remove_target("target_1")
        assert "target_1" not in hud.targets
    
    def test_target_limit(self, hud):
        """Test target limit enforcement"""
        hud.config['max_targets'] = 5
        
        # Add more targets than limit
        for i in range(10):
            target = Target(
                id=f"target_{i}",
                position=np.array([i, i, i]),
                velocity=np.array([0, 0, 0]),
                classification="neutral",
                threat_level=ThreatLevel.NONE if i < 5 else ThreatLevel.LOW,
                distance=100.0,
                angle=0.0,
                last_seen=time.time() - i,  # Older targets have older timestamps
                confidence=0.5
            )
            hud.add_target(target)
        
        # Should not exceed max_targets
        assert len(hud.targets) <= hud.config['max_targets']
    
    def test_add_remove_element(self, hud):
        """Test adding and removing HUD elements"""
        element = HUDElement(
            name="test_element",
            position=(100, 100),
            priority=5
        )
        
        initial_count = len(hud.elements)
        hud.add_element(element)
        assert len(hud.elements) == initial_count + 1
        
        hud.remove_element("test_element")
        assert len(hud.elements) == initial_count
    
    def test_project_3d_to_screen(self, hud):
        """Test 3D to 2D projection"""
        # Point in front of camera
        world_pos = np.array([0, 0, -10])
        screen_pos = hud.project_3d_to_screen(world_pos)
        assert screen_pos is not None
        assert 0 <= screen_pos[0] < hud.resolution[0]
        assert 0 <= screen_pos[1] < hud.resolution[1]
        
        # Point behind camera
        world_pos = np.array([0, 0, 10])
        screen_pos = hud.project_3d_to_screen(world_pos)
        assert screen_pos is None
    
    def test_calibration(self, hud):
        """Test HUD calibration"""
        calibration_data = {
            'fov_horizontal': 90,
            'fov_vertical': 60,
            'eye_offset': [0, 0.05, 0]
        }
        
        hud.calibrate(calibration_data)
        assert hud.calibration['fov_horizontal'] == 90
        assert hud.calibration['fov_vertical'] == 60
        assert np.array_equal(hud.calibration['eye_offset'], np.array([0, 0.05, 0]))
    
    def test_telemetry_update(self, hud):
        """Test telemetry data update"""
        telemetry = {
            'altitude': 1000,
            'speed': 300,
            'heading': 45,
            'power_level': 85
        }
        
        hud.update_telemetry(telemetry)
        assert not hud.update_queue.empty()
    
    def test_frame_rendering(self, hud):
        """Test frame rendering"""
        hud.start()
        time.sleep(0.1)  # Let it render a few frames
        
        frame = hud.get_frame()
        assert frame is not None
        assert frame.shape == (1080, 1920, 4)  # RGBA
        
        hud.stop()
    
    def test_screenshot(self, hud, tmp_path):
        """Test screenshot functionality"""
        hud.start()
        time.sleep(0.1)
        
        screenshot_path = tmp_path / "hud_screenshot.png"
        hud.screenshot(str(screenshot_path))
        
        assert screenshot_path.exists()
        hud.stop()
    
    def test_clear_targets(self, hud):
        """Test clearing all targets"""
        # Add some targets
        for i in range(5):
            target = Target(
                id=f"target_{i}",
                position=np.array([i, i, i]),
                velocity=np.array([0, 0, 0]),
                classification="unknown",
                threat_level=ThreatLevel.LOW,
                distance=100.0,
                angle=0.0,
                last_seen=time.time(),
                confidence=0.7
            )
            hud.add_target(target)
        
        assert len(hud.targets) == 5
        
        hud.clear_targets()
        assert len(hud.targets) == 0


class TestHUDMode:
    """Test HUD mode enumeration"""
    
    def test_mode_values(self):
        """Test HUD mode values"""
        assert HUDMode.COMBAT.value == "combat"
        assert HUDMode.FLIGHT.value == "flight"
        assert HUDMode.SCAN.value == "scan"
        assert HUDMode.NAVIGATION.value == "navigation"
        assert HUDMode.STEALTH.value == "stealth"
        assert HUDMode.ANALYSIS.value == "analysis"


class TestThreatLevel:
    """Test threat level enumeration"""
    
    def test_threat_levels(self):
        """Test threat level values"""
        assert ThreatLevel.NONE.value == 0
        assert ThreatLevel.LOW.value == 1
        assert ThreatLevel.MEDIUM.value == 2
        assert ThreatLevel.HIGH.value == 3
        assert ThreatLevel.CRITICAL.value == 4
    
    def test_threat_comparison(self):
        """Test threat level comparison"""
        assert ThreatLevel.CRITICAL.value > ThreatLevel.HIGH.value
        assert ThreatLevel.HIGH.value > ThreatLevel.MEDIUM.value
        assert ThreatLevel.MEDIUM.value > ThreatLevel.LOW.value
        assert ThreatLevel.LOW.value > ThreatLevel.NONE.value