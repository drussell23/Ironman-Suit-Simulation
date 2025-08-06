"""
waypoint_planner.py

Provides waypoint-based path planning for autopilot, including methods for
manipulating waypoints, computing distances, bearings, velocity guidance,
cross-track errors, and loiter patterns.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

class WaypointPlanner:
    def __init__(self, waypoints: list, acceptance_radius: float = 1.0):
        """
        :param waypoints: list of [x,y,z] positions
        :param acceptance_radius: distance (m) to consider a waypoint reached
        """
        self.waypoints = [np.array(w, float) for w in waypoints]
        self.acceptance_radius = acceptance_radius
        self.current_index = 0

    def add_waypoint(self, waypoint):
        """Append a new waypoint at the end."""
        self.waypoints.append(np.array(waypoint, float))
        logger.debug(f"Added waypoint: {waypoint}")

    def insert_waypoint(self, index: int, waypoint):
        """Insert a waypoint at a given index."""
        self.waypoints.insert(index, np.array(waypoint, float))
        logger.debug(f"Inserted waypoint at {index}: {waypoint}")

    def clear_waypoints(self):
        """Remove all waypoints and reset index."""
        self.waypoints.clear()
        self.current_index = 0
        logger.debug("Cleared all waypoints")

    def reset(self):
        """Reset to first waypoint."""
        self.current_index = 0
        logger.debug("Reset waypoint index to 0")

    def current_waypoint(self):
        """Return the current target waypoint or None if finished."""
        if self.current_index >= len(self.waypoints):
            return None
        return self.waypoints[self.current_index]

    def get_remaining_waypoints(self):
        """List of waypoints not yet reached."""
        return self.waypoints[self.current_index:]

    def distance_to_waypoint(self, position):
        """Euclidean distance to current waypoint."""
        wp = self.current_waypoint()
        if wp is None:
            return None
        pos = np.array(position, float)
        return np.linalg.norm(wp - pos)

    def bearing_to_waypoint(self, position):
        """Yaw bearing [rad] in horizontal plane towards current waypoint."""
        wp = self.current_waypoint()
        if wp is None:
            return None
        pos = np.array(position, float)
        dx, _, dz = wp - pos
        return np.arctan2(dz, dx)

    def is_done(self):
        """True if all waypoints have been reached."""
        return self.current_index >= len(self.waypoints)

    def update(self, position):
        """
        Check if current waypoint is within acceptance radius. Advance index if so.
        Returns True if moved to next waypoint.
        """
        dist = self.distance_to_waypoint(position)
        if dist is not None and dist <= self.acceptance_radius:
            logger.debug(f"Reached waypoint {self.current_index} at distance {dist:.2f} m")
            self.current_index += 1
            return True
        return False

    def get_desired_velocity(self, position, speed: float):
        """
        Pure pursuit guidance: velocity vector towards current waypoint.
        :param speed: desired speed magnitude
        """
        wp = self.current_waypoint()
        if wp is None:
            return np.zeros(3)
        pos = np.array(position, float)
        vec = wp - pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return np.zeros(3)
        return vec / dist * speed

    def cross_track_error(self, position, previous_wp=None):
        """
        Compute cross-track error to segment from previous_wp to current waypoint.
        If previous_wp is None, uses last reached waypoint.
        """
        if self.current_index == 0:
            return 0.0
        prev = (np.array(previous_wp, float)
                if previous_wp is not None
                else self.waypoints[self.current_index - 1])
        curr = self.current_waypoint()
        pos = np.array(position, float)
        segment = curr - prev
        proj = prev + np.dot(pos - prev, segment) / np.dot(segment, segment) * segment
        return np.linalg.norm(pos - proj)

    def hold_pattern(self, center, radius: float, num_points: int = 36, clockwise: bool = True):
        """
        Create circular loiter waypoints around center in horizontal plane.
        """
        center = np.array(center, float)
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        if clockwise:
            angles = angles[::-1]
        pattern = [
            center + np.array([radius*np.cos(a), 0.0, radius*np.sin(a)])
            for a in angles
        ]
        self.waypoints = pattern
        self.current_index = 0
        logger.debug(f"Generated hold pattern at {center.tolist()} r={radius}m")