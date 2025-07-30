"""
Virtual Interface implementation for Iron Man suit.

This module provides immersive 3D holographic interfaces for suit control,
including gesture-based manipulation, voice commands, and virtual control panels
that can be interacted with in 3D space.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from collections import deque


class InterfaceType(Enum):
    """Types of virtual interfaces"""
    CONTROL_PANEL = "control_panel"
    HOLOGRAPHIC_MAP = "holographic_map"
    DATA_VISUALIZATION = "data_visualization"
    WEAPON_SYSTEMS = "weapon_systems"
    COMMUNICATIONS = "communications"
    DIAGNOSTICS = "diagnostics"


class InteractionMode(Enum):
    """Interaction modes for virtual interfaces"""
    GESTURE = "gesture"
    VOICE = "voice"
    GAZE = "gaze"
    HYBRID = "hybrid"


@dataclass
class VirtualObject:
    """Base class for virtual 3D objects"""
    id: str
    name: str
    position: np.ndarray  # 3D position
    rotation: np.ndarray  # Euler angles
    scale: np.ndarray
    visible: bool = True
    interactive: bool = True
    material: Dict[str, Any] = field(default_factory=dict)
    
    def transform_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix"""
        # Simplified transformation matrix
        matrix = np.eye(4)
        matrix[:3, 3] = self.position
        # Add rotation and scale (simplified)
        return matrix


@dataclass
class VirtualButton(VirtualObject):
    """Interactive button in 3D space"""
    callback: Optional[Callable] = None
    pressed: bool = False
    hover: bool = False
    label: str = ""
    size: Tuple[float, float] = (0.1, 0.05)  # Width, height in meters


@dataclass
class VirtualSlider(VirtualObject):
    """Interactive slider in 3D space"""
    min_value: float = 0.0
    max_value: float = 1.0
    current_value: float = 0.5
    callback: Optional[Callable] = None
    orientation: str = "horizontal"  # horizontal or vertical
    length: float = 0.2  # meters


@dataclass
class VirtualPanel(VirtualObject):
    """Container panel for UI elements"""
    width: float = 0.5
    height: float = 0.3
    elements: List[VirtualObject] = field(default_factory=list)
    transparent: bool = True
    opacity: float = 0.8


class VirtualInterface:
    """
    Main virtual interface system for Iron Man suit.
    
    Manages 3D holographic interfaces, gesture recognition,
    and spatial UI elements.
    """
    
    def __init__(self):
        self.interfaces: Dict[str, VirtualPanel] = {}
        self.objects: Dict[str, VirtualObject] = {}
        self.active_interface: Optional[str] = None
        self.interaction_mode = InteractionMode.HYBRID
        
        # Spatial tracking
        self.user_position = np.array([0, 0, 0])
        self.user_orientation = np.array([0, 0, 0])
        self.hand_positions = {
            'left': np.array([0, 0, 0]),
            'right': np.array([0, 0, 0])
        }
        
        # Gesture recognition
        self.gesture_buffer = deque(maxlen=30)  # 1 second at 30fps
        self.recognized_gestures = []
        
        # System state
        self.is_active = False
        self.update_thread = None
        self.last_update_time = time.time()
        
        # Configuration
        self.config = {
            'interface_distance': 0.8,  # meters from user
            'gesture_threshold': 0.8,
            'selection_radius': 0.05,  # meters
            'haptic_feedback': True,
            'audio_feedback': True,
            'animation_speed': 1.0,
            'max_interfaces': 5
        }
        
        # Callbacks
        self.callbacks = {
            'on_select': None,
            'on_gesture': None,
            'on_interface_change': None
        }
        
        self._initialize_default_interfaces()
    
    def _initialize_default_interfaces(self):
        """Create default interface layouts"""
        # Main control panel
        self.create_control_panel()
        
        # Weapon systems interface
        self.create_weapon_interface()
        
        # Communications interface
        self.create_communications_interface()
        
        # Diagnostics interface
        self.create_diagnostics_interface()
    
    def create_control_panel(self):
        """Create main control panel interface"""
        panel = VirtualPanel(
            id="main_control",
            name="Main Control Panel",
            position=np.array([0, 1.5, self.config['interface_distance']]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            width=0.6,
            height=0.4
        )
        
        # Power control slider
        power_slider = VirtualSlider(
            id="power_slider",
            name="Power Level",
            position=np.array([-0.2, 0.1, 0]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            min_value=0,
            max_value=100,
            current_value=75,
            callback=self._on_power_change
        )
        
        # Flight mode button
        flight_button = VirtualButton(
            id="flight_mode",
            name="Flight Mode",
            position=np.array([0, 0, 0]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            label="FLIGHT",
            callback=self._on_flight_mode_toggle
        )
        
        # Combat mode button
        combat_button = VirtualButton(
            id="combat_mode",
            name="Combat Mode",
            position=np.array([0.15, 0, 0]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            label="COMBAT",
            callback=self._on_combat_mode_toggle
        )
        
        # Add elements to panel
        panel.elements.extend([power_slider, flight_button, combat_button])
        
        self.add_interface("main_control", panel)
    
    def create_weapon_interface(self):
        """Create weapon systems interface"""
        panel = VirtualPanel(
            id="weapon_systems",
            name="Weapon Systems",
            position=np.array([0.5, 1.2, self.config['interface_distance']]),
            rotation=np.array([0, -30, 0]),
            scale=np.array([1, 1, 1]),
            width=0.5,
            height=0.5
        )
        
        # Repulsor controls
        repulsor_left = VirtualButton(
            id="repulsor_left",
            name="Left Repulsor",
            position=np.array([-0.1, 0.1, 0]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            label="L-REP",
            callback=lambda: self._fire_weapon("left_repulsor")
        )
        
        repulsor_right = VirtualButton(
            id="repulsor_right",
            name="Right Repulsor",
            position=np.array([0.1, 0.1, 0]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            label="R-REP",
            callback=lambda: self._fire_weapon("right_repulsor")
        )
        
        # Unibeam control
        unibeam = VirtualButton(
            id="unibeam",
            name="Unibeam",
            position=np.array([0, -0.1, 0]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            label="UNIBEAM",
            size=(0.15, 0.08),
            callback=lambda: self._fire_weapon("unibeam")
        )
        
        panel.elements.extend([repulsor_left, repulsor_right, unibeam])
        self.add_interface("weapon_systems", panel)
    
    def create_communications_interface(self):
        """Create communications interface"""
        panel = VirtualPanel(
            id="communications",
            name="Communications",
            position=np.array([-0.5, 1.2, self.config['interface_distance']]),
            rotation=np.array([0, 30, 0]),
            scale=np.array([1, 1, 1]),
            width=0.4,
            height=0.3
        )
        
        # Contact list would be populated dynamically
        # For now, add some example buttons
        for i, contact in enumerate(["JARVIS", "Fury", "Avengers"]):
            button = VirtualButton(
                id=f"contact_{contact}",
                name=f"Contact {contact}",
                position=np.array([0, 0.1 - i*0.08, 0]),
                rotation=np.array([0, 0, 0]),
                scale=np.array([1, 1, 1]),
                label=contact,
                callback=lambda c=contact: self._initiate_communication(c)
            )
            panel.elements.append(button)
        
        self.add_interface("communications", panel)
    
    def create_diagnostics_interface(self):
        """Create diagnostics interface"""
        panel = VirtualPanel(
            id="diagnostics",
            name="System Diagnostics",
            position=np.array([0, 0.8, self.config['interface_distance']]),
            rotation=np.array([15, 0, 0]),
            scale=np.array([1, 1, 1]),
            width=0.8,
            height=0.3
        )
        
        # This would show real-time system data
        # For now, create placeholder elements
        self.add_interface("diagnostics", panel)
    
    def add_interface(self, interface_id: str, panel: VirtualPanel):
        """Add a virtual interface panel"""
        if len(self.interfaces) >= self.config['max_interfaces']:
            # Remove oldest interface
            oldest = list(self.interfaces.keys())[0]
            del self.interfaces[oldest]
        
        self.interfaces[interface_id] = panel
        
        # Add all panel elements to objects dict
        for element in panel.elements:
            self.objects[element.id] = element
    
    def remove_interface(self, interface_id: str):
        """Remove a virtual interface"""
        if interface_id in self.interfaces:
            panel = self.interfaces[interface_id]
            # Remove all associated objects
            for element in panel.elements:
                if element.id in self.objects:
                    del self.objects[element.id]
            del self.interfaces[interface_id]
    
    def show_interface(self, interface_id: str):
        """Show a specific interface"""
        if interface_id in self.interfaces:
            self.interfaces[interface_id].visible = True
            self.active_interface = interface_id
            
            if self.callbacks['on_interface_change']:
                self.callbacks['on_interface_change'](interface_id)
    
    def hide_interface(self, interface_id: str):
        """Hide a specific interface"""
        if interface_id in self.interfaces:
            self.interfaces[interface_id].visible = False
            if self.active_interface == interface_id:
                self.active_interface = None
    
    def start(self):
        """Start the virtual interface system"""
        self.is_active = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.start()
    
    def stop(self):
        """Stop the virtual interface system"""
        self.is_active = False
        if self.update_thread:
            self.update_thread.join()
    
    def _update_loop(self):
        """Main update loop for interfaces"""
        while self.is_active:
            current_time = time.time()
            delta_time = current_time - self.last_update_time
            
            # Update at 60 Hz
            if delta_time >= 1/60:
                self._update_interfaces(delta_time)
                self._process_interactions()
                self._update_animations(delta_time)
                self.last_update_time = current_time
            
            time.sleep(0.001)
    
    def _update_interfaces(self, delta_time: float):
        """Update all interface positions and states"""
        # Update interface positions based on user orientation
        for interface in self.interfaces.values():
            if interface.visible:
                # Keep interfaces facing the user
                self._orient_to_user(interface)
    
    def _orient_to_user(self, interface: VirtualPanel):
        """Orient interface to face the user"""
        # Calculate direction from interface to user
        direction = self.user_position - interface.position
        direction[1] = 0  # Keep interfaces upright
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            
            # Calculate rotation to face user
            angle = np.arctan2(direction[0], direction[2])
            interface.rotation[1] = np.degrees(angle)
    
    def _process_interactions(self):
        """Process user interactions with interfaces"""
        if self.interaction_mode in [InteractionMode.GESTURE, InteractionMode.HYBRID]:
            self._process_gesture_interactions()
        
        if self.interaction_mode in [InteractionMode.GAZE, InteractionMode.HYBRID]:
            self._process_gaze_interactions()
    
    def _process_gesture_interactions(self):
        """Process gesture-based interactions"""
        # Check for pointing gestures
        for hand in ['left', 'right']:
            ray_origin = self.hand_positions[hand]
            ray_direction = self._get_pointing_direction(hand)
            
            # Ray cast to find intersections
            hit_object = self._raycast_interfaces(ray_origin, ray_direction)
            
            if hit_object and isinstance(hit_object, VirtualButton):
                if not hit_object.hover:
                    hit_object.hover = True
                    self._on_hover_enter(hit_object)
                
                # Check for selection gesture (e.g., pinch)
                if self._is_selecting(hand):
                    self._on_object_selected(hit_object)
    
    def _get_pointing_direction(self, hand: str) -> np.ndarray:
        """Get pointing direction from hand position"""
        # Simplified - would use actual hand tracking
        return np.array([0, 0, 1])
    
    def _is_selecting(self, hand: str) -> bool:
        """Check if hand is making selection gesture"""
        # Simplified - would use actual gesture recognition
        return False
    
    def _raycast_interfaces(self, origin: np.ndarray, direction: np.ndarray) -> Optional[VirtualObject]:
        """Find object hit by ray"""
        closest_hit = None
        min_distance = float('inf')
        
        for obj in self.objects.values():
            if obj.visible and obj.interactive:
                # Simplified ray-sphere intersection
                distance = self._ray_object_distance(origin, direction, obj)
                if distance < min_distance and distance < self.config['selection_radius']:
                    min_distance = distance
                    closest_hit = obj
        
        return closest_hit
    
    def _ray_object_distance(self, origin: np.ndarray, direction: np.ndarray, 
                           obj: VirtualObject) -> float:
        """Calculate distance from ray to object"""
        # Simplified - would use proper ray-object intersection
        to_object = obj.position - origin
        projection = np.dot(to_object, direction)
        
        if projection < 0:
            return float('inf')
        
        closest_point = origin + direction * projection
        distance = np.linalg.norm(obj.position - closest_point)
        
        return distance
    
    def _process_gaze_interactions(self):
        """Process gaze-based interactions"""
        # Would integrate with eye tracking
        pass
    
    def _update_animations(self, delta_time: float):
        """Update interface animations"""
        # Animate button presses, slider movements, etc.
        for obj in self.objects.values():
            if isinstance(obj, VirtualButton) and obj.pressed:
                # Simple press animation
                obj.scale = obj.scale * 0.95
                if np.all(obj.scale < 0.9):
                    obj.pressed = False
                    obj.scale = np.array([1, 1, 1])
    
    def _on_hover_enter(self, obj: VirtualObject):
        """Handle hover enter event"""
        if self.config['audio_feedback']:
            # Play hover sound
            pass
        
        if self.config['haptic_feedback']:
            # Trigger haptic feedback
            pass
    
    def _on_object_selected(self, obj: VirtualObject):
        """Handle object selection"""
        if isinstance(obj, VirtualButton):
            obj.pressed = True
            if obj.callback:
                obj.callback()
        elif isinstance(obj, VirtualSlider):
            # Handle slider interaction
            pass
        
        if self.callbacks['on_select']:
            self.callbacks['on_select'](obj)
    
    # Callback methods
    def _on_power_change(self, value: float):
        """Handle power level change"""
        print(f"Power level changed to: {value}%")
    
    def _on_flight_mode_toggle(self):
        """Handle flight mode toggle"""
        print("Flight mode toggled")
    
    def _on_combat_mode_toggle(self):
        """Handle combat mode toggle"""
        print("Combat mode toggled")
    
    def _fire_weapon(self, weapon_type: str):
        """Handle weapon firing"""
        print(f"Firing {weapon_type}")
    
    def _initiate_communication(self, contact: str):
        """Handle communication initiation"""
        print(f"Initiating communication with {contact}")
    
    # Public API methods
    def update_hand_position(self, hand: str, position: np.ndarray):
        """Update tracked hand position"""
        if hand in self.hand_positions:
            self.hand_positions[hand] = position
    
    def update_user_transform(self, position: np.ndarray, orientation: np.ndarray):
        """Update user position and orientation"""
        self.user_position = position
        self.user_orientation = orientation
    
    def register_gesture(self, gesture_data: Dict[str, Any]):
        """Register a recognized gesture"""
        self.gesture_buffer.append(gesture_data)
        
        # Check for gesture patterns
        gesture_type = self._recognize_gesture_pattern()
        if gesture_type:
            self.recognized_gestures.append(gesture_type)
            if self.callbacks['on_gesture']:
                self.callbacks['on_gesture'](gesture_type)
    
    def _recognize_gesture_pattern(self) -> Optional[str]:
        """Recognize gesture patterns from buffer"""
        # Simplified gesture recognition
        # Would use ML models for actual recognition
        return None
    
    def set_callback(self, event: str, callback: Callable):
        """Set event callback"""
        if event in self.callbacks:
            self.callbacks[event] = callback
    
    def get_object(self, object_id: str) -> Optional[VirtualObject]:
        """Get object by ID"""
        return self.objects.get(object_id)
    
    def update_object(self, object_id: str, **kwargs):
        """Update object properties"""
        if object_id in self.objects:
            obj = self.objects[object_id]
            for key, value in kwargs.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
    
    def create_custom_interface(self, interface_id: str, config: Dict[str, Any]) -> VirtualPanel:
        """Create a custom interface from configuration"""
        panel = VirtualPanel(
            id=interface_id,
            name=config.get('name', interface_id),
            position=np.array(config.get('position', [0, 1, self.config['interface_distance']])),
            rotation=np.array(config.get('rotation', [0, 0, 0])),
            scale=np.array(config.get('scale', [1, 1, 1])),
            width=config.get('width', 0.5),
            height=config.get('height', 0.3)
        )
        
        # Add configured elements
        for element_config in config.get('elements', []):
            element_type = element_config.get('type', 'button')
            
            if element_type == 'button':
                element = VirtualButton(
                    id=element_config['id'],
                    name=element_config.get('name', ''),
                    position=np.array(element_config.get('position', [0, 0, 0])),
                    rotation=np.array(element_config.get('rotation', [0, 0, 0])),
                    scale=np.array(element_config.get('scale', [1, 1, 1])),
                    label=element_config.get('label', ''),
                    callback=element_config.get('callback')
                )
            elif element_type == 'slider':
                element = VirtualSlider(
                    id=element_config['id'],
                    name=element_config.get('name', ''),
                    position=np.array(element_config.get('position', [0, 0, 0])),
                    rotation=np.array(element_config.get('rotation', [0, 0, 0])),
                    scale=np.array(element_config.get('scale', [1, 1, 1])),
                    min_value=element_config.get('min', 0),
                    max_value=element_config.get('max', 1),
                    current_value=element_config.get('value', 0.5),
                    callback=element_config.get('callback')
                )
            else:
                continue
            
            panel.elements.append(element)
        
        self.add_interface(interface_id, panel)
        return panel
    
    def get_interface_state(self) -> Dict[str, Any]:
        """Get current state of all interfaces"""
        state = {
            'active_interface': self.active_interface,
            'interaction_mode': self.interaction_mode.value,
            'interfaces': {}
        }
        
        for interface_id, panel in self.interfaces.items():
            state['interfaces'][interface_id] = {
                'visible': panel.visible,
                'position': panel.position.tolist(),
                'rotation': panel.rotation.tolist(),
                'elements': {}
            }
            
            for element in panel.elements:
                element_state = {
                    'type': type(element).__name__,
                    'position': element.position.tolist(),
                    'visible': element.visible
                }
                
                if isinstance(element, VirtualButton):
                    element_state['pressed'] = element.pressed
                    element_state['hover'] = element.hover
                elif isinstance(element, VirtualSlider):
                    element_state['value'] = element.current_value
                
                state['interfaces'][interface_id]['elements'][element.id] = element_state
        
        return state