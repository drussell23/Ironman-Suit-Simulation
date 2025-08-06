"""
Augmented HUD (Heads-Up Display) implementation for Iron Man suit.

This module provides the core augmented reality HUD functionality including
target tracking, environmental awareness, threat assessment, and real-time
data visualization overlays similar to Tony Stark's helmet interface.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import cv2
import threading
import queue
import time


class HUDMode(Enum):
    """Operating modes for the HUD display"""
    COMBAT = "combat"
    FLIGHT = "flight"
    SCAN = "scan"
    NAVIGATION = "navigation"
    STEALTH = "stealth"
    ANALYSIS = "analysis"


class ThreatLevel(Enum):
    """Threat assessment levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Target:
    """Target tracking data structure"""
    id: str
    position: np.ndarray  # 3D position in world space
    velocity: np.ndarray  # 3D velocity vector
    classification: str
    threat_level: ThreatLevel
    distance: float
    angle: float
    last_seen: float
    confidence: float


@dataclass
class HUDElement:
    """Base class for HUD overlay elements"""
    name: str
    position: Tuple[int, int]  # Screen coordinates
    visible: bool = True
    priority: int = 0
    opacity: float = 1.0
    color: Tuple[int, int, int] = (0, 255, 255)  # Default cyan


class AugmentedHUD:
    """
    Main augmented reality HUD system for Iron Man suit.
    
    Provides real-time overlays, target tracking, environmental scanning,
    and tactical information display.
    """
    
    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        self.resolution = resolution
        self.mode = HUDMode.FLIGHT
        self.targets: Dict[str, Target] = {}
        self.elements: List[HUDElement] = []
        self.frame_buffer = np.zeros((resolution[1], resolution[0], 4), dtype=np.uint8)
        
        # System state
        self.is_active = False
        self.fps = 60
        self.last_update_time = time.time()
        
        # Threading for real-time updates
        self.update_queue = queue.Queue()
        self.render_thread = None
        
        # Configuration
        self.config = {
            'target_box_color': (255, 0, 0),  # Red for targets
            'friendly_box_color': (0, 255, 0),  # Green for friendlies
            'neutral_box_color': (255, 255, 0),  # Yellow for neutral
            'grid_color': (0, 128, 255),  # Blue grid
            'text_color': (0, 255, 255),  # Cyan text
            'warning_color': (255, 0, 255),  # Magenta warnings
            'grid_spacing': 50,
            'target_lock_threshold': 0.8,
            'max_targets': 50
        }
        
        # Calibration data
        self.calibration = {
            'fov_horizontal': 120,  # degrees
            'fov_vertical': 80,  # degrees
            'eye_offset': np.array([0, 0, 0]),  # Eye position offset
            'projection_matrix': None
        }
        
        self._initialize_projection()
        self._setup_default_elements()
    
    def _initialize_projection(self):
        """Initialize perspective projection matrix"""
        fov_h = np.radians(self.calibration['fov_horizontal'])
        fov_v = np.radians(self.calibration['fov_vertical'])
        aspect = self.resolution[0] / self.resolution[1]
        
        # Simple perspective projection
        self.calibration['projection_matrix'] = np.array([
            [1/np.tan(fov_h/2), 0, 0, 0],
            [0, aspect/np.tan(fov_v/2), 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
    
    def _setup_default_elements(self):
        """Setup default HUD elements"""
        # Compass
        self.add_element(HUDElement(
            name="compass",
            position=(self.resolution[0]//2, 50),
            priority=10
        ))
        
        # Altitude indicator
        self.add_element(HUDElement(
            name="altitude",
            position=(100, self.resolution[1]//2),
            priority=9
        ))
        
        # Speed indicator
        self.add_element(HUDElement(
            name="speed",
            position=(self.resolution[0]-100, self.resolution[1]//2),
            priority=9
        ))
        
        # Power level
        self.add_element(HUDElement(
            name="power",
            position=(50, self.resolution[1]-50),
            priority=8
        ))
        
        # Mode indicator
        self.add_element(HUDElement(
            name="mode",
            position=(self.resolution[0]//2, self.resolution[1]-50),
            priority=8
        ))
    
    def start(self):
        """Start the HUD system"""
        self.is_active = True
        self.render_thread = threading.Thread(target=self._render_loop)
        self.render_thread.start()
    
    def stop(self):
        """Stop the HUD system"""
        self.is_active = False
        if self.render_thread:
            self.render_thread.join()
    
    def set_mode(self, mode: HUDMode):
        """Change HUD operating mode"""
        self.mode = mode
        self._reconfigure_for_mode()
    
    def _reconfigure_for_mode(self):
        """Reconfigure HUD elements based on mode"""
        if self.mode == HUDMode.COMBAT:
            # Enhanced target tracking
            self.config['max_targets'] = 100
            self.config['target_lock_threshold'] = 0.7
        elif self.mode == HUDMode.STEALTH:
            # Reduced visibility
            for element in self.elements:
                element.opacity = 0.5
        elif self.mode == HUDMode.SCAN:
            # Enhanced environmental scanning
            self.config['grid_spacing'] = 25
    
    def add_target(self, target: Target):
        """Add or update a target"""
        self.targets[target.id] = target
        
        # Limit targets
        if len(self.targets) > self.config['max_targets']:
            # Remove oldest low-priority target
            oldest = min(self.targets.values(), key=lambda t: t.last_seen)
            if oldest.threat_level == ThreatLevel.NONE:
                del self.targets[oldest.id]
    
    def remove_target(self, target_id: str):
        """Remove a target from tracking"""
        if target_id in self.targets:
            del self.targets[target_id]
    
    def add_element(self, element: HUDElement):
        """Add a HUD element"""
        self.elements.append(element)
        self.elements.sort(key=lambda e: e.priority, reverse=True)
    
    def remove_element(self, name: str):
        """Remove a HUD element by name"""
        self.elements = [e for e in self.elements if e.name != name]
    
    def project_3d_to_screen(self, world_pos: np.ndarray) -> Optional[Tuple[int, int]]:
        """Project 3D world position to 2D screen coordinates"""
        # Apply projection matrix
        pos_4d = np.append(world_pos, 1)
        proj = self.calibration['projection_matrix'] @ pos_4d
        
        # Perspective divide
        if proj[3] != 0:
            proj = proj / proj[3]
        
        # Check if behind camera
        if proj[2] > 0:
            return None
        
        # Convert to screen coordinates
        x = int((proj[0] + 1) * self.resolution[0] / 2)
        y = int((1 - proj[1]) * self.resolution[1] / 2)
        
        # Check bounds
        if 0 <= x < self.resolution[0] and 0 <= y < self.resolution[1]:
            return (x, y)
        return None
    
    def _render_loop(self):
        """Main rendering loop"""
        while self.is_active:
            current_time = time.time()
            delta_time = current_time - self.last_update_time
            
            if delta_time >= 1.0 / self.fps:
                self._render_frame()
                self.last_update_time = current_time
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
    
    def _render_frame(self):
        """Render a single frame"""
        # Clear frame buffer
        self.frame_buffer.fill(0)
        
        # Render based on mode
        if self.mode in [HUDMode.FLIGHT, HUDMode.NAVIGATION]:
            self._render_flight_mode()
        elif self.mode == HUDMode.COMBAT:
            self._render_combat_mode()
        elif self.mode == HUDMode.SCAN:
            self._render_scan_mode()
        
        # Render common elements
        self._render_targets()
        self._render_elements()
        self._render_warnings()
    
    def _render_flight_mode(self):
        """Render flight mode specific elements"""
        # Horizon line
        cv2.line(self.frame_buffer, 
                (0, self.resolution[1]//2),
                (self.resolution[0], self.resolution[1]//2),
                (*self.config['grid_color'], 128), 2)
        
        # Pitch ladder
        for i in range(-30, 31, 10):
            y = self.resolution[1]//2 + i * 10
            if 0 <= y < self.resolution[1]:
                cv2.line(self.frame_buffer,
                        (self.resolution[0]//2 - 50, y),
                        (self.resolution[0]//2 + 50, y),
                        (*self.config['grid_color'], 128), 1)
    
    def _render_combat_mode(self):
        """Render combat mode specific elements"""
        # Target acquisition reticle
        center = (self.resolution[0]//2, self.resolution[1]//2)
        cv2.circle(self.frame_buffer, center, 100, (*self.config['target_box_color'], 200), 2)
        cv2.circle(self.frame_buffer, center, 150, (*self.config['target_box_color'], 150), 1)
        
        # Crosshairs
        cv2.line(self.frame_buffer,
                (center[0] - 200, center[1]),
                (center[0] - 100, center[1]),
                (*self.config['target_box_color'], 255), 2)
        cv2.line(self.frame_buffer,
                (center[0] + 100, center[1]),
                (center[0] + 200, center[1]),
                (*self.config['target_box_color'], 255), 2)
        cv2.line(self.frame_buffer,
                (center[0], center[1] - 200),
                (center[0], center[1] - 100),
                (*self.config['target_box_color'], 255), 2)
        cv2.line(self.frame_buffer,
                (center[0], center[1] + 100),
                (center[0], center[1] + 200),
                (*self.config['target_box_color'], 255), 2)
    
    def _render_scan_mode(self):
        """Render scan mode specific elements"""
        # Grid overlay
        spacing = self.config['grid_spacing']
        
        # Vertical lines
        for x in range(0, self.resolution[0], spacing):
            cv2.line(self.frame_buffer,
                    (x, 0), (x, self.resolution[1]),
                    (*self.config['grid_color'], 50), 1)
        
        # Horizontal lines
        for y in range(0, self.resolution[1], spacing):
            cv2.line(self.frame_buffer,
                    (0, y), (self.resolution[0], y),
                    (*self.config['grid_color'], 50), 1)
        
        # Scanning sweep effect
        sweep_x = int((time.time() % 3) * self.resolution[0] / 3)
        cv2.line(self.frame_buffer,
                (sweep_x, 0), (sweep_x, self.resolution[1]),
                (*self.config['grid_color'], 200), 3)
    
    def _render_targets(self):
        """Render all tracked targets"""
        for target in self.targets.values():
            screen_pos = self.project_3d_to_screen(target.position)
            if screen_pos:
                self._draw_target_box(screen_pos, target)
    
    def _draw_target_box(self, pos: Tuple[int, int], target: Target):
        """Draw target tracking box"""
        # Determine color based on classification
        if "friendly" in target.classification.lower():
            color = self.config['friendly_box_color']
        elif "hostile" in target.classification.lower():
            color = self.config['target_box_color']
        else:
            color = self.config['neutral_box_color']
        
        # Box size based on distance
        size = max(30, min(100, int(1000 / (target.distance + 1))))
        
        # Draw box
        cv2.rectangle(self.frame_buffer,
                     (pos[0] - size//2, pos[1] - size//2),
                     (pos[0] + size//2, pos[1] + size//2),
                     (*color, 200), 2)
        
        # Draw corner brackets
        bracket_size = size // 4
        cv2.line(self.frame_buffer,
                (pos[0] - size//2, pos[1] - size//2),
                (pos[0] - size//2 + bracket_size, pos[1] - size//2),
                (*color, 255), 3)
        cv2.line(self.frame_buffer,
                (pos[0] - size//2, pos[1] - size//2),
                (pos[0] - size//2, pos[1] - size//2 + bracket_size),
                (*color, 255), 3)
        
        # Target info
        info_text = f"{target.classification} [{target.distance:.0f}m]"
        cv2.putText(self.frame_buffer, info_text,
                   (pos[0] - size//2, pos[1] - size//2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (*color, 255), 1)
        
        # Threat indicator
        if target.threat_level.value >= ThreatLevel.HIGH.value:
            cv2.circle(self.frame_buffer, (pos[0], pos[1] - size//2 - 20), 5,
                      (*self.config['warning_color'], 255), -1)
    
    def _render_elements(self):
        """Render HUD elements"""
        for element in self.elements:
            if element.visible:
                self._draw_element(element)
    
    def _draw_element(self, element: HUDElement):
        """Draw a specific HUD element"""
        alpha = int(element.opacity * 255)
        color = (*element.color, alpha)
        
        if element.name == "compass":
            self._draw_compass(element.position, color)
        elif element.name == "altitude":
            self._draw_altitude_indicator(element.position, color)
        elif element.name == "speed":
            self._draw_speed_indicator(element.position, color)
        elif element.name == "power":
            self._draw_power_indicator(element.position, color)
        elif element.name == "mode":
            self._draw_mode_indicator(element.position, color)
    
    def _draw_compass(self, pos: Tuple[int, int], color: Tuple[int, int, int, int]):
        """Draw compass indicator"""
        # Simplified compass visualization
        cv2.circle(self.frame_buffer, pos, 40, color[:3], 2)
        cv2.putText(self.frame_buffer, "N", (pos[0]-5, pos[1]-45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color[:3], 2)
    
    def _draw_altitude_indicator(self, pos: Tuple[int, int], color: Tuple[int, int, int, int]):
        """Draw altitude indicator"""
        # Vertical scale
        cv2.line(self.frame_buffer, (pos[0], pos[1]-100), (pos[0], pos[1]+100), color[:3], 2)
        
        # Tick marks
        for i in range(-5, 6):
            y = pos[1] + i * 20
            cv2.line(self.frame_buffer, (pos[0]-10, y), (pos[0]+10, y), color[:3], 1)
    
    def _draw_speed_indicator(self, pos: Tuple[int, int], color: Tuple[int, int, int, int]):
        """Draw speed indicator"""
        # Similar to altitude but on right side
        cv2.line(self.frame_buffer, (pos[0], pos[1]-100), (pos[0], pos[1]+100), color[:3], 2)
        
        for i in range(-5, 6):
            y = pos[1] + i * 20
            cv2.line(self.frame_buffer, (pos[0]-10, y), (pos[0]+10, y), color[:3], 1)
    
    def _draw_power_indicator(self, pos: Tuple[int, int], color: Tuple[int, int, int, int]):
        """Draw power level indicator"""
        # Battery-style indicator
        cv2.rectangle(self.frame_buffer, (pos[0]-30, pos[1]-10), (pos[0]+30, pos[1]+10), color[:3], 2)
        # Fill based on power level (mock)
        cv2.rectangle(self.frame_buffer, (pos[0]-28, pos[1]-8), (pos[0]+15, pos[1]+8), 
                     (*self.config['friendly_box_color'], 200), -1)
    
    def _draw_mode_indicator(self, pos: Tuple[int, int], color: Tuple[int, int, int, int]):
        """Draw current mode indicator"""
        cv2.putText(self.frame_buffer, f"MODE: {self.mode.value.upper()}",
                   (pos[0]-50, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color[:3], 2)
    
    def _render_warnings(self):
        """Render warning messages"""
        # Check for critical threats
        critical_threats = [t for t in self.targets.values() 
                          if t.threat_level == ThreatLevel.CRITICAL]
        
        if critical_threats:
            warning_text = f"WARNING: {len(critical_threats)} CRITICAL THREATS"
            cv2.putText(self.frame_buffer, warning_text,
                       (self.resolution[0]//2 - 150, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                       (*self.config['warning_color'], 255), 3)
    
    def get_frame(self) -> np.ndarray:
        """Get current HUD frame"""
        return self.frame_buffer.copy()
    
    def update_telemetry(self, telemetry: Dict[str, Any]):
        """Update HUD with telemetry data"""
        # This would be called by external systems to update HUD data
        self.update_queue.put(telemetry)
    
    def calibrate(self, calibration_data: Dict[str, Any]):
        """Calibrate HUD projection parameters"""
        if 'fov_horizontal' in calibration_data:
            self.calibration['fov_horizontal'] = calibration_data['fov_horizontal']
        if 'fov_vertical' in calibration_data:
            self.calibration['fov_vertical'] = calibration_data['fov_vertical']
        if 'eye_offset' in calibration_data:
            self.calibration['eye_offset'] = np.array(calibration_data['eye_offset'])
        
        self._initialize_projection()
    
    def set_config(self, config: Dict[str, Any]):
        """Update HUD configuration"""
        self.config.update(config)
    
    def clear_targets(self):
        """Clear all tracked targets"""
        self.targets.clear()
    
    def screenshot(self, filename: str):
        """Save current HUD frame to file"""
        cv2.imwrite(filename, self.frame_buffer)