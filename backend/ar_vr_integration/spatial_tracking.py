"""
Spatial Tracking and Mapping system for Iron Man suit AR/VR.

This module provides 6DOF tracking, SLAM (Simultaneous Localization and Mapping),
spatial anchors, and environment understanding for AR/VR experiences.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from collections import deque
import uuid


class TrackingState(Enum):
    """Tracking quality states"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LIMITED = "limited"  # Reduced quality
    LOST = "lost"


class AnchorType(Enum):
    """Types of spatial anchors"""
    POINT = "point"
    PLANE = "plane"
    OBJECT = "object"
    ROOM = "room"
    PERSISTENT = "persistent"  # Saved across sessions


class PlaneType(Enum):
    """Detected plane types"""
    HORIZONTAL_UP = "horizontal_up"  # Floor
    HORIZONTAL_DOWN = "horizontal_down"  # Ceiling
    VERTICAL = "vertical"  # Wall
    ARBITRARY = "arbitrary"  # Any orientation


@dataclass
class Pose:
    """6DOF pose (position and orientation)"""
    position: np.ndarray  # 3D position
    rotation: np.ndarray  # Quaternion (x, y, z, w)
    timestamp: float
    confidence: float = 1.0
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix"""
        mat = np.eye(4)
        
        # Convert quaternion to rotation matrix
        q = self.rotation
        mat[:3, :3] = np.array([
            [1-2*(q[1]**2+q[2]**2), 2*(q[0]*q[1]-q[2]*q[3]), 2*(q[0]*q[2]+q[1]*q[3])],
            [2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[0]**2+q[2]**2), 2*(q[1]*q[2]-q[0]*q[3])],
            [2*(q[0]*q[2]-q[1]*q[3]), 2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[0]**2+q[1]**2)]
        ])
        
        mat[:3, 3] = self.position
        return mat
    
    def inverse(self) -> 'Pose':
        """Get inverse pose"""
        # Inverse rotation
        inv_rot = np.array([-self.rotation[0], -self.rotation[1], 
                           -self.rotation[2], self.rotation[3]])
        
        # Rotate position by inverse rotation
        rot_matrix = self.matrix[:3, :3]
        inv_pos = -rot_matrix.T @ self.position
        
        return Pose(inv_pos, inv_rot, self.timestamp, self.confidence)


@dataclass
class SpatialAnchor:
    """Spatial anchor in the environment"""
    id: str
    anchor_type: AnchorType
    pose: Pose
    created_time: float
    last_seen_time: float
    persistence_id: Optional[str] = None  # For cloud anchors
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_stale(self, timeout: float = 30.0) -> bool:
        """Check if anchor hasn't been seen recently"""
        return time.time() - self.last_seen_time > timeout


@dataclass
class Plane:
    """Detected planar surface"""
    id: str
    plane_type: PlaneType
    center: np.ndarray
    normal: np.ndarray
    extents: np.ndarray  # Width, height
    boundary_points: List[np.ndarray]
    last_updated: float
    
    def point_on_plane(self, point: np.ndarray) -> bool:
        """Check if point lies on plane"""
        distance = abs(np.dot(point - self.center, self.normal))
        return distance < 0.01  # 1cm threshold
    
    def project_point(self, point: np.ndarray) -> np.ndarray:
        """Project point onto plane"""
        distance = np.dot(point - self.center, self.normal)
        return point - distance * self.normal


@dataclass
class PointCloud:
    """3D point cloud data"""
    points: np.ndarray  # Nx3 array
    colors: Optional[np.ndarray] = None  # Nx3 RGB
    normals: Optional[np.ndarray] = None  # Nx3
    confidence: Optional[np.ndarray] = None  # N array
    timestamp: float = 0
    
    def downsample(self, voxel_size: float) -> 'PointCloud':
        """Voxel-based downsampling"""
        # Simple grid-based downsampling
        min_bound = np.min(self.points, axis=0)
        max_bound = np.max(self.points, axis=0)
        
        grid_size = ((max_bound - min_bound) / voxel_size).astype(int) + 1
        voxel_dict = {}
        
        for i, point in enumerate(self.points):
            voxel_idx = ((point - min_bound) / voxel_size).astype(int)
            key = tuple(voxel_idx)
            
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)
        
        # Average points in each voxel
        downsampled_points = []
        downsampled_colors = [] if self.colors is not None else None
        
        for indices in voxel_dict.values():
            downsampled_points.append(np.mean(self.points[indices], axis=0))
            if self.colors is not None:
                downsampled_colors.append(np.mean(self.colors[indices], axis=0))
        
        return PointCloud(
            points=np.array(downsampled_points),
            colors=np.array(downsampled_colors) if downsampled_colors else None,
            timestamp=self.timestamp
        )


class SpatialTracker:
    """
    Main spatial tracking and mapping system.
    
    Provides SLAM, plane detection, spatial anchors, and
    environment reconstruction.
    """
    
    def __init__(self):
        # Tracking state
        self.tracking_state = TrackingState.NOT_INITIALIZED
        self.current_pose = Pose(np.zeros(3), np.array([0, 0, 0, 1]), time.time())
        self.pose_history = deque(maxlen=1000)  # ~16 seconds at 60Hz
        
        # SLAM components
        self.map_points: PointCloud = PointCloud(np.empty((0, 3)))
        self.keyframes: List[Pose] = []
        self.features: Dict[str, np.ndarray] = {}  # Feature descriptors
        
        # Environment understanding
        self.detected_planes: Dict[str, Plane] = {}
        self.spatial_anchors: Dict[str, SpatialAnchor] = {}
        self.room_layout: Optional[Dict[str, Any]] = None
        
        # Sensor data buffers
        self.imu_buffer = deque(maxlen=100)
        self.image_buffer = deque(maxlen=10)
        self.depth_buffer = deque(maxlen=10)
        
        # Configuration
        self.config = {
            'enable_visual_inertial': True,
            'enable_depth_sensing': True,
            'enable_plane_detection': True,
            'enable_persistent_mapping': True,
            'map_voxel_size': 0.05,  # 5cm voxels
            'keyframe_distance': 0.5,  # meters
            'keyframe_angle': 30,  # degrees
            'plane_detection_threshold': 0.02,  # meters
            'max_map_points': 100000,
            'tracking_quality_threshold': 0.7
        }
        
        # Processing threads
        self.is_running = False
        self.tracking_thread = None
        self.mapping_thread = None
        
        # Callbacks
        self.callbacks = {
            'on_tracking_state_changed': None,
            'on_plane_detected': None,
            'on_anchor_created': None,
            'on_map_updated': None
        }
        
        # Performance metrics
        self.metrics = {
            'tracking_fps': 0,
            'mapping_fps': 0,
            'map_point_count': 0,
            'plane_count': 0,
            'anchor_count': 0,
            'tracking_quality': 0
        }
    
    def start(self):
        """Start spatial tracking system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.tracking_state = TrackingState.INITIALIZING
        
        # Start processing threads
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.start()
        
        self.mapping_thread = threading.Thread(target=self._mapping_loop)
        self.mapping_thread.start()
    
    def stop(self):
        """Stop spatial tracking system"""
        self.is_running = False
        
        if self.tracking_thread:
            self.tracking_thread.join()
        
        if self.mapping_thread:
            self.mapping_thread.join()
        
        self.tracking_state = TrackingState.NOT_INITIALIZED
    
    def reset(self):
        """Reset tracking and mapping"""
        self.tracking_state = TrackingState.INITIALIZING
        self.current_pose = Pose(np.zeros(3), np.array([0, 0, 0, 1]), time.time())
        self.pose_history.clear()
        
        self.map_points = PointCloud(np.empty((0, 3)))
        self.keyframes.clear()
        self.features.clear()
        
        self.detected_planes.clear()
        self.spatial_anchors.clear()
        self.room_layout = None
    
    def _tracking_loop(self):
        """Main tracking loop"""
        last_time = time.time()
        
        while self.is_running:
            start_time = time.time()
            
            # Process sensor data
            if self.config['enable_visual_inertial']:
                self._process_visual_inertial()
            
            if self.config['enable_depth_sensing']:
                self._process_depth_data()
            
            # Update pose estimate
            self._update_pose_estimate()
            
            # Check tracking quality
            self._update_tracking_state()
            
            # Calculate metrics
            self.metrics['tracking_fps'] = 1.0 / (time.time() - last_time)
            last_time = time.time()
            
            # Maintain 60Hz update rate
            elapsed = time.time() - start_time
            if elapsed < 1/60:
                time.sleep(1/60 - elapsed)
    
    def _mapping_loop(self):
        """Main mapping loop"""
        last_time = time.time()
        
        while self.is_running:
            start_time = time.time()
            
            if self.tracking_state == TrackingState.TRACKING:
                # Update map
                self._update_map()
                
                # Detect planes
                if self.config['enable_plane_detection']:
                    self._detect_planes()
                
                # Update spatial anchors
                self._update_anchors()
                
                # Check for keyframe
                if self._should_create_keyframe():
                    self._create_keyframe()
                
                # Optimize map
                if len(self.keyframes) % 10 == 0:
                    self._optimize_map()
            
            # Calculate metrics
            self.metrics['mapping_fps'] = 1.0 / (time.time() - last_time)
            self.metrics['map_point_count'] = len(self.map_points.points)
            self.metrics['plane_count'] = len(self.detected_planes)
            self.metrics['anchor_count'] = len(self.spatial_anchors)
            last_time = time.time()
            
            # Run at 30Hz
            elapsed = time.time() - start_time
            if elapsed < 1/30:
                time.sleep(1/30 - elapsed)
    
    def _process_visual_inertial(self):
        """Process visual-inertial odometry"""
        # Get latest IMU data
        if not self.imu_buffer:
            return
        
        imu_data = list(self.imu_buffer)[-10:]  # Last 10 samples
        
        # Get latest image
        if not self.image_buffer:
            return
        
        image_data = self.image_buffer[-1]
        
        # Extract features
        features = self._extract_features(image_data)
        
        # Match with previous features
        if hasattr(self, '_last_features'):
            matches = self._match_features(self._last_features, features)
            
            # Estimate motion
            if len(matches) > 8:
                motion = self._estimate_motion_from_features(matches, imu_data)
                self._apply_motion(motion)
        
        self._last_features = features
    
    def _process_depth_data(self):
        """Process depth sensor data"""
        if not self.depth_buffer:
            return
        
        depth_data = self.depth_buffer[-1]
        
        # Convert to point cloud
        points = self._depth_to_pointcloud(depth_data)
        
        # Add to map (with filtering)
        self._add_points_to_map(points)
    
    def _extract_features(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract visual features from image"""
        # Simplified feature extraction
        # Would use ORB, SIFT, or learned features
        
        features = {
            'keypoints': np.random.rand(100, 2) * np.array([image_data.shape[1], image_data.shape[0]]),
            'descriptors': np.random.rand(100, 128)
        }
        
        return features
    
    def _match_features(self, features1: Dict[str, np.ndarray], 
                       features2: Dict[str, np.ndarray]) -> List[Tuple[int, int]]:
        """Match features between frames"""
        # Simplified matching
        # Would use FLANN or brute-force matching
        
        matches = []
        for i in range(min(len(features1['keypoints']), len(features2['keypoints']))):
            if np.random.rand() > 0.3:  # Random matches for demo
                matches.append((i, i))
        
        return matches
    
    def _estimate_motion_from_features(self, matches: List[Tuple[int, int]], 
                                     imu_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Estimate camera motion from feature matches and IMU"""
        # Simplified motion estimation
        # Would use essential matrix decomposition or PnP
        
        # Integrate IMU data
        angular_velocity = np.mean([d['gyro'] for d in imu_data], axis=0)
        linear_acceleration = np.mean([d['accel'] for d in imu_data], axis=0)
        
        dt = 1/60  # Assume 60Hz
        
        # Simple integration
        rotation_change = angular_velocity * dt
        position_change = linear_acceleration * dt * dt
        
        return {
            'rotation': rotation_change,
            'translation': position_change
        }
    
    def _apply_motion(self, motion: Dict[str, np.ndarray]):
        """Apply motion to current pose"""
        # Update rotation (simplified euler integration)
        euler = motion['rotation']
        rotation_delta = self._euler_to_quaternion(euler)
        self.current_pose.rotation = self._quaternion_multiply(
            self.current_pose.rotation, rotation_delta
        )
        
        # Update position
        self.current_pose.position += motion['translation']
        self.current_pose.timestamp = time.time()
    
    def _depth_to_pointcloud(self, depth_data: np.ndarray) -> PointCloud:
        """Convert depth image to point cloud"""
        # Camera intrinsics (would be calibrated)
        fx, fy = 500, 500  # Focal length
        cx, cy = depth_data.shape[1]/2, depth_data.shape[0]/2
        
        # Generate points
        points = []
        
        for y in range(0, depth_data.shape[0], 10):  # Downsample
            for x in range(0, depth_data.shape[1], 10):
                z = depth_data[y, x]
                if z > 0 and z < 10:  # Valid depth
                    # Back-project to 3D
                    point_3d = np.array([
                        (x - cx) * z / fx,
                        (y - cy) * z / fy,
                        z
                    ])
                    
                    # Transform to world coordinates
                    world_point = self.current_pose.matrix[:3, :3] @ point_3d + self.current_pose.position
                    points.append(world_point)
        
        return PointCloud(
            points=np.array(points) if points else np.empty((0, 3)),
            timestamp=time.time()
        )
    
    def _add_points_to_map(self, new_points: PointCloud):
        """Add points to global map"""
        if len(new_points.points) == 0:
            return
        
        # Concatenate with existing map
        if len(self.map_points.points) == 0:
            self.map_points = new_points
        else:
            self.map_points.points = np.vstack([self.map_points.points, new_points.points])
        
        # Limit map size
        if len(self.map_points.points) > self.config['max_map_points']:
            # Downsample
            self.map_points = self.map_points.downsample(self.config['map_voxel_size'])
    
    def _update_pose_estimate(self):
        """Update current pose estimate"""
        # Add to history
        self.pose_history.append(self.current_pose)
        
        # Apply filtering (simplified Kalman filter)
        if len(self.pose_history) > 3:
            # Smooth position
            positions = np.array([p.position for p in list(self.pose_history)[-3:]])
            self.current_pose.position = np.mean(positions, axis=0)
    
    def _update_tracking_state(self):
        """Update tracking state based on quality metrics"""
        quality = self._calculate_tracking_quality()
        self.metrics['tracking_quality'] = quality
        
        old_state = self.tracking_state
        
        if quality > self.config['tracking_quality_threshold']:
            self.tracking_state = TrackingState.TRACKING
        elif quality > 0.3:
            self.tracking_state = TrackingState.LIMITED
        else:
            self.tracking_state = TrackingState.LOST
        
        # Fire callback if state changed
        if old_state != self.tracking_state and self.callbacks['on_tracking_state_changed']:
            self.callbacks['on_tracking_state_changed'](self.tracking_state)
    
    def _calculate_tracking_quality(self) -> float:
        """Calculate tracking quality metric"""
        # Consider multiple factors
        factors = []
        
        # Feature match ratio
        if hasattr(self, '_last_features'):
            factors.append(0.8)  # Placeholder
        
        # IMU data availability
        if self.imu_buffer:
            factors.append(1.0)
        
        # Map point visibility
        if len(self.map_points.points) > 100:
            factors.append(0.9)
        
        return np.mean(factors) if factors else 0.0
    
    def _update_map(self):
        """Update global map"""
        # Would implement local mapping, loop closure, etc.
        pass
    
    def _detect_planes(self):
        """Detect planar surfaces in point cloud"""
        if len(self.map_points.points) < 100:
            return
        
        # Simplified RANSAC plane fitting
        for _ in range(5):  # Try to find 5 planes
            plane = self._fit_plane_ransac(self.map_points.points)
            
            if plane is not None:
                # Check if similar plane already exists
                merged = False
                for existing_plane in self.detected_planes.values():
                    if self._planes_similar(plane, existing_plane):
                        # Merge planes
                        self._merge_planes(existing_plane, plane)
                        merged = True
                        break
                
                if not merged:
                    # Add new plane
                    plane_id = str(uuid.uuid4())
                    self.detected_planes[plane_id] = plane
                    
                    if self.callbacks['on_plane_detected']:
                        self.callbacks['on_plane_detected'](plane)
    
    def _fit_plane_ransac(self, points: np.ndarray) -> Optional[Plane]:
        """Fit plane using RANSAC"""
        if len(points) < 3:
            return None
        
        best_plane = None
        best_inliers = 0
        
        for _ in range(100):  # RANSAC iterations
            # Sample 3 random points
            indices = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[indices]
            
            # Calculate plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-6:
                continue
            
            normal = normal / np.linalg.norm(normal)
            
            # Count inliers
            distances = np.abs(np.dot(points - p1, normal))
            inliers = np.sum(distances < self.config['plane_detection_threshold'])
            
            if inliers > best_inliers and inliers > len(points) * 0.1:
                best_inliers = inliers
                
                # Fit plane to all inliers
                inlier_points = points[distances < self.config['plane_detection_threshold']]
                center = np.mean(inlier_points, axis=0)
                
                # Determine plane type
                plane_type = self._classify_plane(normal)
                
                # Calculate extents
                projected = inlier_points - np.outer(np.dot(inlier_points - center, normal), normal)
                extents = np.max(projected, axis=0) - np.min(projected, axis=0)
                
                best_plane = Plane(
                    id=str(uuid.uuid4()),
                    plane_type=plane_type,
                    center=center,
                    normal=normal,
                    extents=extents[:2],  # Width, height
                    boundary_points=self._compute_convex_hull(projected),
                    last_updated=time.time()
                )
        
        return best_plane
    
    def _classify_plane(self, normal: np.ndarray) -> PlaneType:
        """Classify plane based on normal direction"""
        # Check angle with gravity (assuming Y is up)
        up_vector = np.array([0, 1, 0])
        angle = np.arccos(np.clip(np.dot(normal, up_vector), -1, 1))
        
        if angle < np.radians(30):  # Within 30 degrees of up
            return PlaneType.HORIZONTAL_UP
        elif angle > np.radians(150):  # Within 30 degrees of down
            return PlaneType.HORIZONTAL_DOWN
        elif np.radians(60) < angle < np.radians(120):  # Roughly vertical
            return PlaneType.VERTICAL
        else:
            return PlaneType.ARBITRARY
    
    def _compute_convex_hull(self, points: np.ndarray) -> List[np.ndarray]:
        """Compute 2D convex hull of points"""
        # Simplified convex hull
        # Would use proper algorithm like Graham scan
        
        if len(points) < 3:
            return list(points)
        
        # Just return bounding box for now
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        
        return [
            min_point,
            np.array([max_point[0], min_point[1], min_point[2]]),
            max_point,
            np.array([min_point[0], max_point[1], max_point[2]])
        ]
    
    def _planes_similar(self, plane1: Plane, plane2: Plane) -> bool:
        """Check if two planes are similar"""
        # Check normal similarity
        normal_dot = np.dot(plane1.normal, plane2.normal)
        if abs(normal_dot) < 0.95:  # ~18 degree threshold
            return False
        
        # Check distance
        distance = abs(np.dot(plane2.center - plane1.center, plane1.normal))
        if distance > 0.1:  # 10cm threshold
            return False
        
        return True
    
    def _merge_planes(self, plane1: Plane, plane2: Plane):
        """Merge two similar planes"""
        # Average properties
        plane1.center = (plane1.center + plane2.center) / 2
        plane1.normal = (plane1.normal + plane2.normal)
        plane1.normal = plane1.normal / np.linalg.norm(plane1.normal)
        
        # Expand extents
        plane1.extents = np.maximum(plane1.extents, plane2.extents)
        
        # Merge boundary points
        all_points = plane1.boundary_points + plane2.boundary_points
        plane1.boundary_points = self._compute_convex_hull(np.array(all_points))
        
        plane1.last_updated = time.time()
    
    def _update_anchors(self):
        """Update spatial anchor visibility"""
        current_time = time.time()
        
        # Check anchor visibility
        for anchor_id, anchor in list(self.spatial_anchors.items()):
            # Simple distance-based visibility
            distance = np.linalg.norm(anchor.pose.position - self.current_pose.position)
            
            if distance < 10:  # 10 meter visibility range
                anchor.last_seen_time = current_time
            elif anchor.is_stale():
                # Remove stale anchors (unless persistent)
                if anchor.anchor_type != AnchorType.PERSISTENT:
                    del self.spatial_anchors[anchor_id]
    
    def _should_create_keyframe(self) -> bool:
        """Check if new keyframe should be created"""
        if not self.keyframes:
            return True
        
        last_keyframe = self.keyframes[-1]
        
        # Check distance
        distance = np.linalg.norm(self.current_pose.position - last_keyframe.position)
        if distance > self.config['keyframe_distance']:
            return True
        
        # Check rotation
        angle = self._quaternion_angle(self.current_pose.rotation, last_keyframe.rotation)
        if np.degrees(angle) > self.config['keyframe_angle']:
            return True
        
        return False
    
    def _create_keyframe(self):
        """Create new keyframe"""
        self.keyframes.append(Pose(
            self.current_pose.position.copy(),
            self.current_pose.rotation.copy(),
            self.current_pose.timestamp,
            self.current_pose.confidence
        ))
        
        # Store associated features
        if hasattr(self, '_last_features'):
            self.features[f"keyframe_{len(self.keyframes)-1}"] = self._last_features
    
    def _optimize_map(self):
        """Optimize map using bundle adjustment"""
        # Would implement pose graph optimization
        pass
    
    # Utility methods
    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to quaternion"""
        roll, pitch, yaw = euler
        
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        return np.array([
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        ])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    def _quaternion_angle(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculate angle between two quaternions"""
        dot = np.clip(np.dot(q1, q2), -1, 1)
        return 2 * np.arccos(abs(dot))
    
    # Public API
    def get_current_pose(self) -> Pose:
        """Get current device pose"""
        return self.current_pose
    
    def create_anchor(self, anchor_type: AnchorType = AnchorType.POINT,
                     pose: Optional[Pose] = None) -> str:
        """Create spatial anchor at current or specified pose"""
        if pose is None:
            pose = self.current_pose
        
        anchor_id = str(uuid.uuid4())
        anchor = SpatialAnchor(
            id=anchor_id,
            anchor_type=anchor_type,
            pose=Pose(pose.position.copy(), pose.rotation.copy(), 
                     pose.timestamp, pose.confidence),
            created_time=time.time(),
            last_seen_time=time.time()
        )
        
        self.spatial_anchors[anchor_id] = anchor
        
        if self.callbacks['on_anchor_created']:
            self.callbacks['on_anchor_created'](anchor)
        
        return anchor_id
    
    def get_anchor(self, anchor_id: str) -> Optional[SpatialAnchor]:
        """Get spatial anchor by ID"""
        return self.spatial_anchors.get(anchor_id)
    
    def delete_anchor(self, anchor_id: str):
        """Delete spatial anchor"""
        if anchor_id in self.spatial_anchors:
            del self.spatial_anchors[anchor_id]
    
    def get_planes(self, plane_type: Optional[PlaneType] = None) -> List[Plane]:
        """Get detected planes"""
        planes = list(self.detected_planes.values())
        
        if plane_type:
            planes = [p for p in planes if p.plane_type == plane_type]
        
        return planes
    
    def raycast(self, origin: np.ndarray, direction: np.ndarray, 
               max_distance: float = 10.0) -> Optional[Dict[str, Any]]:
        """Raycast against tracked geometry"""
        direction = direction / np.linalg.norm(direction)
        
        closest_hit = None
        min_distance = max_distance
        
        # Check planes
        for plane in self.detected_planes.values():
            # Ray-plane intersection
            denom = np.dot(plane.normal, direction)
            if abs(denom) > 1e-6:
                t = np.dot(plane.center - origin, plane.normal) / denom
                
                if 0 < t < min_distance:
                    hit_point = origin + t * direction
                    
                    # Check if hit point is within plane bounds
                    if plane.point_on_plane(hit_point):
                        min_distance = t
                        closest_hit = {
                            'type': 'plane',
                            'object': plane,
                            'point': hit_point,
                            'normal': plane.normal,
                            'distance': t
                        }
        
        # Check anchors
        for anchor in self.spatial_anchors.values():
            # Simple sphere intersection for point anchors
            to_anchor = anchor.pose.position - origin
            projection = np.dot(to_anchor, direction)
            
            if projection > 0:
                closest_point = origin + projection * direction
                distance_to_anchor = np.linalg.norm(anchor.pose.position - closest_point)
                
                if distance_to_anchor < 0.1 and projection < min_distance:  # 10cm radius
                    min_distance = projection
                    closest_hit = {
                        'type': 'anchor',
                        'object': anchor,
                        'point': anchor.pose.position,
                        'normal': -direction,
                        'distance': projection
                    }
        
        return closest_hit
    
    def save_map(self, filename: str):
        """Save current map to file"""
        # Would implement map serialization
        pass
    
    def load_map(self, filename: str):
        """Load map from file"""
        # Would implement map deserialization
        pass
    
    def update_imu_data(self, gyro: np.ndarray, accel: np.ndarray):
        """Update IMU sensor data"""
        self.imu_buffer.append({
            'gyro': gyro,
            'accel': accel,
            'timestamp': time.time()
        })
    
    def update_image_data(self, image: np.ndarray):
        """Update camera image data"""
        self.image_buffer.append(image)
    
    def update_depth_data(self, depth: np.ndarray):
        """Update depth sensor data"""
        self.depth_buffer.append(depth)
    
    def set_config(self, config: Dict[str, Any]):
        """Update configuration"""
        self.config.update(config)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get tracking metrics"""
        return self.metrics.copy()
    
    def set_callback(self, event: str, callback: callable):
        """Set event callback"""
        if event in self.callbacks:
            self.callbacks[event] = callback