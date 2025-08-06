"""
AR Overlay Manager for Iron Man suit.

This module provides AR visualization tools, data overlays, environmental
annotations, and contextual information displays in augmented reality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import threading


class OverlayType(Enum):
    """Types of AR overlays"""
    TEXT_LABEL = "text_label"
    INFO_PANEL = "info_panel"
    WAYPOINT = "waypoint"
    DANGER_ZONE = "danger_zone"
    PATH_GUIDANCE = "path_guidance"
    MEASUREMENT = "measurement"
    ANNOTATION = "annotation"
    SCHEMATIC = "schematic"
    TACTICAL_INFO = "tactical_info"
    ENVIRONMENTAL = "environmental"


class AnchorMode(Enum):
    """How overlays are anchored in space"""
    WORLD_LOCKED = "world_locked"  # Fixed in world space
    SCREEN_LOCKED = "screen_locked"  # Fixed on screen
    OBJECT_LOCKED = "object_locked"  # Attached to object
    BILLBOARD = "billboard"  # Always faces camera
    HEAD_LOCKED = "head_locked"  # Follows head movement


class AnimationType(Enum):
    """Overlay animation types"""
    NONE = "none"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    PULSE = "pulse"
    SLIDE_IN = "slide_in"
    EXPAND = "expand"
    ROTATE = "rotate"


@dataclass
class OverlayStyle:
    """Visual style for overlays"""
    color: Tuple[int, int, int, int] = (0, 255, 255, 255)  # RGBA
    background_color: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 128)
    border_color: Optional[Tuple[int, int, int, int]] = (0, 255, 255, 255)
    border_width: int = 2
    font_size: int = 16
    font_family: str = "Arial"
    padding: int = 10
    corner_radius: int = 5
    blur_background: bool = True
    glow_effect: bool = True
    shadow: bool = True


@dataclass
class AROverlay:
    """Base AR overlay element"""
    id: str
    overlay_type: OverlayType
    position: np.ndarray  # 3D world position or 2D screen position
    anchor_mode: AnchorMode
    content: Any
    visible: bool = True
    opacity: float = 1.0
    scale: float = 1.0
    rotation: float = 0.0  # Degrees
    style: OverlayStyle = field(default_factory=OverlayStyle)
    metadata: Dict[str, Any] = field(default_factory=dict)
    animation: Optional[AnimationType] = None
    animation_progress: float = 0.0
    created_time: float = field(default_factory=time.time)
    lifetime: Optional[float] = None  # Seconds, None = permanent
    priority: int = 0  # Higher = rendered on top
    interactive: bool = False
    parent_id: Optional[str] = None  # For hierarchical overlays


@dataclass
class TextOverlay(AROverlay):
    """Text label overlay"""
    text: str = ""
    max_width: Optional[int] = None
    alignment: str = "center"  # left, center, right
    
    def __post_init__(self):
        self.overlay_type = OverlayType.TEXT_LABEL
        self.content = self.text


@dataclass
class InfoPanelOverlay(AROverlay):
    """Information panel with multiple data fields"""
    title: str = ""
    fields: Dict[str, str] = field(default_factory=dict)
    show_icon: bool = True
    icon_type: Optional[str] = None
    expandable: bool = False
    expanded: bool = False
    
    def __post_init__(self):
        self.overlay_type = OverlayType.INFO_PANEL
        self.content = {'title': self.title, 'fields': self.fields}


@dataclass
class WaypointOverlay(AROverlay):
    """Navigation waypoint overlay"""
    label: str = ""
    distance: float = 0.0
    icon: str = "marker"
    show_distance: bool = True
    show_direction: bool = True
    
    def __post_init__(self):
        self.overlay_type = OverlayType.WAYPOINT
        self.content = {'label': self.label, 'distance': self.distance}


@dataclass
class PathOverlay(AROverlay):
    """Path guidance overlay"""
    waypoints: List[np.ndarray] = field(default_factory=list)
    width: float = 0.5
    animated: bool = True
    arrow_spacing: float = 5.0  # meters
    
    def __post_init__(self):
        self.overlay_type = OverlayType.PATH_GUIDANCE
        self.content = self.waypoints


@dataclass
class MeasurementOverlay(AROverlay):
    """Measurement and dimension overlay"""
    start_point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    end_point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    measurement_type: str = "distance"  # distance, area, volume, angle
    unit: str = "meters"
    show_markers: bool = True
    
    def __post_init__(self):
        self.overlay_type = OverlayType.MEASUREMENT
        self.content = self.calculate_measurement()
    
    def calculate_measurement(self) -> Dict[str, Any]:
        """Calculate measurement value"""
        if self.measurement_type == "distance":
            distance = np.linalg.norm(self.end_point - self.start_point)
            return {'value': distance, 'unit': self.unit}
        return {'value': 0, 'unit': self.unit}


class AROverlayManager:
    """
    Manages AR overlays and visualizations.
    
    Handles creation, updates, rendering, and interaction
    with AR overlay elements.
    """
    
    def __init__(self):
        # Overlay storage
        self.overlays: Dict[str, AROverlay] = {}
        self.render_queue: List[str] = []  # Ordered by priority
        
        # Spatial indexing for efficient queries
        self.spatial_index: Dict[Tuple[int, int, int], List[str]] = {}
        self.grid_size = 10.0  # meters
        
        # Templates for common overlays
        self.templates = self._load_templates()
        
        # Animation system
        self.animations: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            'max_overlays': 1000,
            'max_render_distance': 500,  # meters
            'occlusion_enabled': True,
            'auto_scale_distance': True,
            'min_scale': 0.5,
            'max_scale': 2.0,
            'fade_distance': 100,  # Start fading at this distance
            'update_rate': 30,  # Hz
            'spatial_index_enabled': True,
            'clustering_enabled': True,
            'cluster_distance': 5.0  # meters
        }
        
        # Camera/view state
        self.camera_position = np.zeros(3)
        self.camera_rotation = np.array([0, 0, 0, 1])  # Quaternion
        self.camera_fov = 90  # degrees
        self.screen_resolution = (1920, 1080)
        
        # Processing state
        self.is_running = False
        self.update_thread = None
        
        # Performance metrics
        self.metrics = {
            'overlay_count': 0,
            'visible_overlays': 0,
            'update_time': 0,
            'render_time': 0,
            'culled_overlays': 0
        }
        
        # Interaction state
        self.selected_overlay: Optional[str] = None
        self.hover_overlay: Optional[str] = None
        
        # Callbacks
        self.callbacks = {
            'on_overlay_selected': None,
            'on_overlay_hover': None,
            'on_overlay_expired': None
        }
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load overlay templates"""
        return {
            'enemy_info': {
                'type': OverlayType.INFO_PANEL,
                'style': OverlayStyle(
                    color=(255, 0, 0, 255),
                    background_color=(50, 0, 0, 200),
                    border_color=(255, 0, 0, 255)
                ),
                'fields': ['Type', 'Threat Level', 'Distance', 'Status']
            },
            'objective_marker': {
                'type': OverlayType.WAYPOINT,
                'style': OverlayStyle(
                    color=(0, 255, 0, 255),
                    glow_effect=True
                ),
                'icon': 'objective',
                'animation': AnimationType.PULSE
            },
            'danger_zone': {
                'type': OverlayType.DANGER_ZONE,
                'style': OverlayStyle(
                    color=(255, 0, 0, 128),
                    background_color=(255, 0, 0, 64),
                    border_width=3
                ),
                'animation': AnimationType.PULSE
            },
            'measurement_tool': {
                'type': OverlayType.MEASUREMENT,
                'style': OverlayStyle(
                    color=(255, 255, 0, 255),
                    font_size=18
                ),
                'show_markers': True
            }
        }
    
    def start(self):
        """Start overlay manager"""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.start()
    
    def stop(self):
        """Stop overlay manager"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
    
    def create_overlay(self, overlay: AROverlay) -> str:
        """Create new AR overlay"""
        # Check limits
        if len(self.overlays) >= self.config['max_overlays']:
            # Remove oldest overlay
            oldest = min(self.overlays.values(), key=lambda o: o.created_time)
            self.remove_overlay(oldest.id)
        
        # Add to storage
        self.overlays[overlay.id] = overlay
        
        # Add to spatial index if world-locked
        if overlay.anchor_mode == AnchorMode.WORLD_LOCKED:
            self._add_to_spatial_index(overlay)
        
        # Start animation if specified
        if overlay.animation:
            self._start_animation(overlay)
        
        # Update render queue
        self._update_render_queue()
        
        return overlay.id
    
    def create_from_template(self, template_name: str, **kwargs) -> Optional[str]:
        """Create overlay from template"""
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        overlay_type = template['type']
        
        # Create appropriate overlay class
        if overlay_type == OverlayType.TEXT_LABEL:
            overlay = TextOverlay(
                id=f"{template_name}_{len(self.overlays)}",
                position=kwargs.get('position', np.zeros(3)),
                anchor_mode=kwargs.get('anchor_mode', AnchorMode.WORLD_LOCKED),
                text=kwargs.get('text', ''),
                style=template.get('style', OverlayStyle())
            )
        elif overlay_type == OverlayType.INFO_PANEL:
            overlay = InfoPanelOverlay(
                id=f"{template_name}_{len(self.overlays)}",
                position=kwargs.get('position', np.zeros(3)),
                anchor_mode=kwargs.get('anchor_mode', AnchorMode.WORLD_LOCKED),
                title=kwargs.get('title', ''),
                fields=kwargs.get('fields', {}),
                style=template.get('style', OverlayStyle())
            )
        elif overlay_type == OverlayType.WAYPOINT:
            overlay = WaypointOverlay(
                id=f"{template_name}_{len(self.overlays)}",
                position=kwargs.get('position', np.zeros(3)),
                anchor_mode=AnchorMode.WORLD_LOCKED,
                label=kwargs.get('label', ''),
                style=template.get('style', OverlayStyle())
            )
        else:
            return None
        
        # Apply template animation
        if 'animation' in template:
            overlay.animation = template['animation']
        
        return self.create_overlay(overlay)
    
    def remove_overlay(self, overlay_id: str):
        """Remove overlay"""
        if overlay_id in self.overlays:
            overlay = self.overlays[overlay_id]
            
            # Remove from spatial index
            if overlay.anchor_mode == AnchorMode.WORLD_LOCKED:
                self._remove_from_spatial_index(overlay)
            
            # Stop animation
            if overlay_id in self.animations:
                del self.animations[overlay_id]
            
            del self.overlays[overlay_id]
            self._update_render_queue()
    
    def update_overlay(self, overlay_id: str, **kwargs):
        """Update overlay properties"""
        if overlay_id not in self.overlays:
            return
        
        overlay = self.overlays[overlay_id]
        
        # Update position (handle spatial index)
        if 'position' in kwargs and overlay.anchor_mode == AnchorMode.WORLD_LOCKED:
            self._remove_from_spatial_index(overlay)
            overlay.position = kwargs['position']
            self._add_to_spatial_index(overlay)
        
        # Update other properties
        for key, value in kwargs.items():
            if hasattr(overlay, key) and key != 'position':
                setattr(overlay, key, value)
        
        # Update render queue if priority changed
        if 'priority' in kwargs:
            self._update_render_queue()
    
    def _add_to_spatial_index(self, overlay: AROverlay):
        """Add overlay to spatial index"""
        if not self.config['spatial_index_enabled']:
            return
        
        # Calculate grid cell
        cell = self._position_to_grid_cell(overlay.position)
        
        if cell not in self.spatial_index:
            self.spatial_index[cell] = []
        
        self.spatial_index[cell].append(overlay.id)
    
    def _remove_from_spatial_index(self, overlay: AROverlay):
        """Remove overlay from spatial index"""
        if not self.config['spatial_index_enabled']:
            return
        
        cell = self._position_to_grid_cell(overlay.position)
        
        if cell in self.spatial_index and overlay.id in self.spatial_index[cell]:
            self.spatial_index[cell].remove(overlay.id)
            
            if not self.spatial_index[cell]:
                del self.spatial_index[cell]
    
    def _position_to_grid_cell(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert position to grid cell coordinates"""
        return tuple((position / self.grid_size).astype(int))
    
    def _update_render_queue(self):
        """Update render queue based on priority"""
        self.render_queue = sorted(
            self.overlays.keys(),
            key=lambda oid: (self.overlays[oid].priority, oid),
            reverse=True
        )
    
    def _start_animation(self, overlay: AROverlay):
        """Start overlay animation"""
        self.animations[overlay.id] = {
            'type': overlay.animation,
            'start_time': time.time(),
            'duration': 1.0,  # Default duration
            'loop': overlay.animation == AnimationType.PULSE
        }
    
    def _update_loop(self):
        """Main update loop"""
        while self.is_running:
            start_time = time.time()
            
            # Update animations
            self._update_animations()
            
            # Update lifetime
            self._update_lifetimes()
            
            # Update visibility and LOD
            self._update_visibility()
            
            # Cluster overlays if enabled
            if self.config['clustering_enabled']:
                self._update_clusters()
            
            # Calculate metrics
            self.metrics['update_time'] = (time.time() - start_time) * 1000
            self.metrics['overlay_count'] = len(self.overlays)
            
            # Maintain update rate
            sleep_time = 1.0 / self.config['update_rate'] - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _update_animations(self):
        """Update overlay animations"""
        current_time = time.time()
        
        for overlay_id, anim in list(self.animations.items()):
            if overlay_id not in self.overlays:
                del self.animations[overlay_id]
                continue
            
            overlay = self.overlays[overlay_id]
            elapsed = current_time - anim['start_time']
            progress = elapsed / anim['duration']
            
            if progress >= 1.0:
                if anim['loop']:
                    anim['start_time'] = current_time
                    progress = 0.0
                else:
                    overlay.animation = None
                    del self.animations[overlay_id]
                    continue
            
            overlay.animation_progress = progress
            
            # Apply animation effects
            self._apply_animation(overlay, anim['type'], progress)
    
    def _apply_animation(self, overlay: AROverlay, anim_type: AnimationType, progress: float):
        """Apply animation effect to overlay"""
        if anim_type == AnimationType.FADE_IN:
            overlay.opacity = progress
        elif anim_type == AnimationType.FADE_OUT:
            overlay.opacity = 1.0 - progress
        elif anim_type == AnimationType.PULSE:
            overlay.opacity = 0.5 + 0.5 * math.sin(progress * math.pi * 2)
        elif anim_type == AnimationType.EXPAND:
            overlay.scale = progress
        elif anim_type == AnimationType.ROTATE:
            overlay.rotation = progress * 360
    
    def _update_lifetimes(self):
        """Update overlay lifetimes and remove expired"""
        current_time = time.time()
        expired = []
        
        for overlay_id, overlay in self.overlays.items():
            if overlay.lifetime is not None:
                age = current_time - overlay.created_time
                
                if age >= overlay.lifetime:
                    expired.append(overlay_id)
                elif age >= overlay.lifetime - 1.0:  # Fade out last second
                    overlay.animation = AnimationType.FADE_OUT
                    if overlay_id not in self.animations:
                        self._start_animation(overlay)
        
        # Remove expired overlays
        for overlay_id in expired:
            if self.callbacks['on_overlay_expired']:
                self.callbacks['on_overlay_expired'](self.overlays[overlay_id])
            self.remove_overlay(overlay_id)
    
    def _update_visibility(self):
        """Update overlay visibility and LOD"""
        visible_count = 0
        culled_count = 0
        
        for overlay in self.overlays.values():
            # Skip screen-locked overlays
            if overlay.anchor_mode == AnchorMode.SCREEN_LOCKED:
                overlay.visible = True
                visible_count += 1
                continue
            
            # Calculate distance
            distance = np.linalg.norm(overlay.position - self.camera_position)
            
            # Distance culling
            if distance > self.config['max_render_distance']:
                overlay.visible = False
                culled_count += 1
                continue
            
            # Frustum culling (simplified)
            if not self._in_frustum(overlay.position):
                overlay.visible = False
                culled_count += 1
                continue
            
            # Occlusion culling
            if self.config['occlusion_enabled'] and self._is_occluded(overlay):
                overlay.visible = False
                culled_count += 1
                continue
            
            overlay.visible = True
            visible_count += 1
            
            # Auto-scale based on distance
            if self.config['auto_scale_distance']:
                scale = np.clip(50 / distance, self.config['min_scale'], self.config['max_scale'])
                overlay.scale = scale
            
            # Distance-based fading
            if distance > self.config['fade_distance']:
                fade = 1.0 - (distance - self.config['fade_distance']) / \
                       (self.config['max_render_distance'] - self.config['fade_distance'])
                overlay.opacity = min(overlay.opacity, fade)
        
        self.metrics['visible_overlays'] = visible_count
        self.metrics['culled_overlays'] = culled_count
    
    def _in_frustum(self, position: np.ndarray) -> bool:
        """Check if position is in view frustum"""
        # Simplified frustum check
        to_point = position - self.camera_position
        distance = np.linalg.norm(to_point)
        
        if distance < 0.1:  # Too close
            return True
        
        # Project to camera space (simplified)
        forward = np.array([0, 0, 1])  # Assume looking down +Z
        angle = np.arccos(np.clip(np.dot(to_point / distance, forward), -1, 1))
        
        return np.degrees(angle) < self.camera_fov / 2
    
    def _is_occluded(self, overlay: AROverlay) -> bool:
        """Check if overlay is occluded"""
        # Would implement proper occlusion testing
        # For now, return False
        return False
    
    def _update_clusters(self):
        """Update overlay clustering for dense areas"""
        # Group nearby overlays
        clusters = []
        processed = set()
        
        for overlay_id, overlay in self.overlays.items():
            if overlay_id in processed or overlay.anchor_mode != AnchorMode.WORLD_LOCKED:
                continue
            
            # Find nearby overlays
            cluster = [overlay_id]
            cluster_center = overlay.position.copy()
            
            for other_id, other in self.overlays.items():
                if other_id != overlay_id and other_id not in processed:
                    distance = np.linalg.norm(other.position - overlay.position)
                    
                    if distance < self.config['cluster_distance']:
                        cluster.append(other_id)
                        cluster_center += other.position
                        processed.add(other_id)
            
            if len(cluster) > 3:  # Minimum cluster size
                # Create cluster overlay
                cluster_center /= len(cluster)
                self._create_cluster_overlay(cluster, cluster_center)
                
                # Hide individual overlays
                for oid in cluster:
                    self.overlays[oid].visible = False
    
    def _create_cluster_overlay(self, overlay_ids: List[str], position: np.ndarray):
        """Create a cluster overlay"""
        # Would implement cluster visualization
        pass
    
    # Rendering methods
    def render(self, render_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Render overlays for display"""
        render_start = time.time()
        render_list = []
        
        # Update camera state
        if 'camera_position' in render_context:
            self.camera_position = render_context['camera_position']
        if 'camera_rotation' in render_context:
            self.camera_rotation = render_context['camera_rotation']
        
        # Process overlays in render queue order
        for overlay_id in self.render_queue:
            if overlay_id not in self.overlays:
                continue
            
            overlay = self.overlays[overlay_id]
            
            if not overlay.visible:
                continue
            
            # Convert to render data
            render_data = self._overlay_to_render_data(overlay)
            
            if render_data:
                render_list.append(render_data)
        
        self.metrics['render_time'] = (time.time() - render_start) * 1000
        
        return render_list
    
    def _overlay_to_render_data(self, overlay: AROverlay) -> Optional[Dict[str, Any]]:
        """Convert overlay to renderable data"""
        render_data = {
            'id': overlay.id,
            'type': overlay.overlay_type.value,
            'position': overlay.position.tolist(),
            'anchor_mode': overlay.anchor_mode.value,
            'opacity': overlay.opacity,
            'scale': overlay.scale,
            'rotation': overlay.rotation,
            'style': {
                'color': overlay.style.color,
                'background_color': overlay.style.background_color,
                'border_color': overlay.style.border_color,
                'border_width': overlay.style.border_width,
                'font_size': int(overlay.style.font_size * overlay.scale)
            }
        }
        
        # Add type-specific data
        if isinstance(overlay, TextOverlay):
            render_data['text'] = overlay.text
            render_data['alignment'] = overlay.alignment
        elif isinstance(overlay, InfoPanelOverlay):
            render_data['title'] = overlay.title
            render_data['fields'] = overlay.fields
            render_data['expanded'] = overlay.expanded
        elif isinstance(overlay, WaypointOverlay):
            render_data['label'] = overlay.label
            render_data['distance'] = overlay.distance
            render_data['icon'] = overlay.icon
        elif isinstance(overlay, PathOverlay):
            render_data['waypoints'] = [wp.tolist() for wp in overlay.waypoints]
            render_data['animated'] = overlay.animated
        elif isinstance(overlay, MeasurementOverlay):
            render_data['measurement'] = overlay.content
            render_data['start_point'] = overlay.start_point.tolist()
            render_data['end_point'] = overlay.end_point.tolist()
        
        # Transform to screen space if needed
        if overlay.anchor_mode != AnchorMode.SCREEN_LOCKED:
            screen_pos = self._world_to_screen(overlay.position)
            if screen_pos:
                render_data['screen_position'] = screen_pos
            else:
                return None  # Off-screen
        
        return render_data
    
    def _world_to_screen(self, world_pos: np.ndarray) -> Optional[Tuple[int, int]]:
        """Convert world position to screen coordinates"""
        # Simplified projection
        to_point = world_pos - self.camera_position
        distance = np.linalg.norm(to_point)
        
        if distance < 0.1:
            return None
        
        # Project to screen (simplified)
        # Would use proper projection matrix
        screen_x = self.screen_resolution[0] // 2 + to_point[0] * 100
        screen_y = self.screen_resolution[1] // 2 - to_point[1] * 100
        
        if 0 <= screen_x < self.screen_resolution[0] and 0 <= screen_y < self.screen_resolution[1]:
            return (int(screen_x), int(screen_y))
        
        return None
    
    # Public API methods
    def create_text_label(self, text: str, position: np.ndarray, **kwargs) -> str:
        """Create a text label overlay"""
        overlay = TextOverlay(
            id=f"text_{len(self.overlays)}",
            position=position,
            anchor_mode=kwargs.get('anchor_mode', AnchorMode.WORLD_LOCKED),
            text=text,
            style=kwargs.get('style', OverlayStyle())
        )
        
        for key, value in kwargs.items():
            if hasattr(overlay, key):
                setattr(overlay, key, value)
        
        return self.create_overlay(overlay)
    
    def create_waypoint(self, position: np.ndarray, label: str = "", **kwargs) -> str:
        """Create a waypoint marker"""
        overlay = WaypointOverlay(
            id=f"waypoint_{len(self.overlays)}",
            position=position,
            anchor_mode=AnchorMode.WORLD_LOCKED,
            label=label,
            distance=np.linalg.norm(position - self.camera_position)
        )
        
        for key, value in kwargs.items():
            if hasattr(overlay, key):
                setattr(overlay, key, value)
        
        return self.create_overlay(overlay)
    
    def create_path_guidance(self, waypoints: List[np.ndarray], **kwargs) -> str:
        """Create path guidance overlay"""
        overlay = PathOverlay(
            id=f"path_{len(self.overlays)}",
            position=waypoints[0] if waypoints else np.zeros(3),
            anchor_mode=AnchorMode.WORLD_LOCKED,
            waypoints=waypoints
        )
        
        for key, value in kwargs.items():
            if hasattr(overlay, key):
                setattr(overlay, key, value)
        
        return self.create_overlay(overlay)
    
    def create_measurement(self, start: np.ndarray, end: np.ndarray, **kwargs) -> str:
        """Create measurement overlay"""
        overlay = MeasurementOverlay(
            id=f"measurement_{len(self.overlays)}",
            position=(start + end) / 2,  # Center position
            anchor_mode=AnchorMode.WORLD_LOCKED,
            start_point=start,
            end_point=end
        )
        
        for key, value in kwargs.items():
            if hasattr(overlay, key):
                setattr(overlay, key, value)
        
        return self.create_overlay(overlay)
    
    def create_info_panel(self, position: np.ndarray, title: str, 
                         fields: Dict[str, str], **kwargs) -> str:
        """Create information panel overlay"""
        overlay = InfoPanelOverlay(
            id=f"info_{len(self.overlays)}",
            position=position,
            anchor_mode=kwargs.get('anchor_mode', AnchorMode.BILLBOARD),
            title=title,
            fields=fields
        )
        
        for key, value in kwargs.items():
            if hasattr(overlay, key):
                setattr(overlay, key, value)
        
        return self.create_overlay(overlay)
    
    def highlight_object(self, object_id: str, position: np.ndarray, 
                        info: Dict[str, str], **kwargs) -> str:
        """Create highlight overlay for object"""
        # Create info panel
        panel_id = self.create_info_panel(
            position=position + np.array([0, 2, 0]),  # Above object
            title=object_id,
            fields=info,
            style=OverlayStyle(
                color=(0, 255, 255, 255),
                glow_effect=True
            )
        )
        
        # Could also create bounding box, etc.
        
        return panel_id
    
    def get_overlays_near_position(self, position: np.ndarray, 
                                  radius: float) -> List[AROverlay]:
        """Get overlays within radius of position"""
        nearby = []
        
        if self.config['spatial_index_enabled']:
            # Use spatial index for efficiency
            cells_to_check = []
            cell_radius = int(radius / self.grid_size) + 1
            center_cell = self._position_to_grid_cell(position)
            
            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    for dz in range(-cell_radius, cell_radius + 1):
                        cell = (center_cell[0] + dx, center_cell[1] + dy, center_cell[2] + dz)
                        if cell in self.spatial_index:
                            cells_to_check.extend(self.spatial_index[cell])
            
            # Check overlays in cells
            for overlay_id in set(cells_to_check):
                overlay = self.overlays[overlay_id]
                distance = np.linalg.norm(overlay.position - position)
                if distance <= radius:
                    nearby.append(overlay)
        else:
            # Brute force search
            for overlay in self.overlays.values():
                if overlay.anchor_mode == AnchorMode.WORLD_LOCKED:
                    distance = np.linalg.norm(overlay.position - position)
                    if distance <= radius:
                        nearby.append(overlay)
        
        return nearby
    
    def clear_all(self):
        """Remove all overlays"""
        self.overlays.clear()
        self.render_queue.clear()
        self.spatial_index.clear()
        self.animations.clear()
    
    def set_camera_state(self, position: np.ndarray, rotation: np.ndarray, 
                        fov: float = 90):
        """Update camera state for rendering"""
        self.camera_position = position
        self.camera_rotation = rotation
        self.camera_fov = fov
    
    def set_screen_resolution(self, width: int, height: int):
        """Set screen resolution for rendering"""
        self.screen_resolution = (width, height)
    
    def handle_interaction(self, screen_pos: Tuple[int, int], 
                          interaction_type: str = "select") -> Optional[str]:
        """Handle user interaction with overlays"""
        # Find overlay at screen position
        for overlay_id in reversed(self.render_queue):  # Front to back
            overlay = self.overlays[overlay_id]
            
            if not overlay.visible or not overlay.interactive:
                continue
            
            # Check if screen position hits overlay
            # Simplified hit test
            if overlay.anchor_mode == AnchorMode.SCREEN_LOCKED:
                # Direct screen space test
                overlay_screen_pos = overlay.position[:2]
            else:
                # World to screen projection
                overlay_screen_pos = self._world_to_screen(overlay.position)
                
                if not overlay_screen_pos:
                    continue
            
            # Simple radius-based hit test
            distance = math.sqrt(
                (screen_pos[0] - overlay_screen_pos[0])**2 +
                (screen_pos[1] - overlay_screen_pos[1])**2
            )
            
            hit_radius = 50 * overlay.scale  # Simplified
            
            if distance <= hit_radius:
                if interaction_type == "select":
                    self.selected_overlay = overlay_id
                    if self.callbacks['on_overlay_selected']:
                        self.callbacks['on_overlay_selected'](overlay)
                elif interaction_type == "hover":
                    self.hover_overlay = overlay_id
                    if self.callbacks['on_overlay_hover']:
                        self.callbacks['on_overlay_hover'](overlay)
                
                return overlay_id
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """Update configuration"""
        self.config.update(config)
    
    def set_callback(self, event: str, callback: Callable):
        """Set event callback"""
        if event in self.callbacks:
            self.callbacks[event] = callback


# Add missing import at the top
import math