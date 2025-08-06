"""
Behavioral Adaptation System for Iron Man Suit

This module provides AI systems that learn and adapt to the pilot's behavior:
- Pilot behavior modeling and prediction
- Adaptive control systems that adjust to pilot preferences
- Preference learning from pilot interactions
- Personalized automation levels
"""

import logging
import numpy as np
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import time

# Try importing ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

logger = logging.getLogger(__name__)


class BehavioralAdaptationError(Exception):
    """Exception for behavioral adaptation errors."""

    pass


class PilotBehaviorModel:
    """Models pilot behavior patterns and preferences."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.behavior_history = deque(maxlen=self.config.get("history_size", 10000))
        self.preference_weights = defaultdict(float)
        self.behavior_clusters = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Behavior categories
        self.behavior_categories = {
            "aggressive": ["high_thrust", "rapid_maneuvers", "close_combat"],
            "defensive": ["evasive_maneuvers", "long_range", "stealth"],
            "balanced": ["moderate_thrust", "tactical_positioning", "versatile"],
            "conservative": ["minimal_thrust", "safe_distances", "defensive_stance"],
        }

    def add_behavior_sample(
        self,
        state: np.ndarray,
        action: np.ndarray,
        context: Dict[str, Any],
        timestamp: float = None,
    ):
        """Add a new behavior sample to the model."""
        if timestamp is None:
            timestamp = time.time()

        sample = {
            "state": state.copy(),
            "action": action.copy(),
            "context": context.copy(),
            "timestamp": timestamp,
            "features": self._extract_behavior_features(state, action, context),
        }

        self.behavior_history.append(sample)

        # Update preference weights
        self._update_preference_weights(sample)

    def _extract_behavior_features(
        self, state: np.ndarray, action: np.ndarray, context: Dict[str, Any]
    ) -> np.ndarray:
        """Extract behavioral features from state, action, and context."""
        features = []

        # Flight dynamics features
        if len(state) >= 6:  # [x, y, z, vx, vy, vz]
            velocity = np.linalg.norm(state[3:6])
            altitude = state[1]
            features.extend([velocity, altitude])

        # Action features
        if len(action) >= 2:  # [thrust, angle_of_attack]
            thrust_magnitude = np.abs(action[0])
            maneuver_intensity = np.abs(action[1])
            features.extend([thrust_magnitude, maneuver_intensity])

        # Context features
        threat_level = context.get("threat_level", 0.0)
        mission_phase = context.get("mission_phase", "patrol")
        energy_level = context.get("energy_level", 1.0)
        features.extend([threat_level, energy_level])

        # Mission phase encoding
        phase_encoding = self._encode_mission_phase(mission_phase)
        features.extend(phase_encoding)

        return np.array(features)

    def _encode_mission_phase(self, phase: str) -> List[float]:
        """One-hot encode mission phase."""
        phases = ["patrol", "combat", "evasion", "approach", "extraction"]
        encoding = [0.0] * len(phases)
        if phase in phases:
            encoding[phases.index(phase)] = 1.0
        return encoding

    def _update_preference_weights(self, sample: Dict):
        """Update preference weights based on behavior patterns."""
        features = sample["features"]
        context = sample["context"]

        # Analyze behavior patterns
        if features[0] > 50:  # High velocity
            self.preference_weights["speed"] += 0.1
        if features[1] > 100:  # High altitude
            self.preference_weights["altitude"] += 0.1
        if features[2] > 1000:  # High thrust
            self.preference_weights["aggressive"] += 0.1
        if context.get("threat_level", 0) > 0.7:
            self.preference_weights["defensive"] += 0.1

        # Normalize weights
        total = sum(self.preference_weights.values())
        if total > 0:
            for key in self.preference_weights:
                self.preference_weights[key] /= total

    def train_model(self):
        """Train the behavior clustering model."""
        if len(self.behavior_history) < 100:
            logger.warning("Insufficient data for training behavior model")
            return

        # Extract features for clustering
        features = np.array([sample["features"] for sample in self.behavior_history])

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Perform clustering
        n_clusters = min(4, len(features) // 10)  # Adaptive number of clusters
        self.behavior_clusters = KMeans(n_clusters=n_clusters, random_state=42)
        self.behavior_clusters.fit(features_scaled)

        self.is_trained = True
        logger.info(f"Behavior model trained with {n_clusters} clusters")

    def predict_behavior(
        self, state: np.ndarray, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict pilot behavior for given state and context."""
        if not self.is_trained:
            return self._default_behavior_prediction()

        # Create dummy action for feature extraction
        dummy_action = np.zeros(2)  # [thrust, angle_of_attack]
        features = self._extract_behavior_features(state, dummy_action, context)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict cluster
        cluster = self.behavior_clusters.predict(features_scaled)[0]

        # Get cluster characteristics
        cluster_center = self.behavior_clusters.cluster_centers_[cluster]
        cluster_center_unscaled = self.scaler.inverse_transform(
            cluster_center.reshape(1, -1)
        )[0]

        # Map to behavior categories
        behavior_prediction = self._map_cluster_to_behavior(cluster_center_unscaled)

        return behavior_prediction

    def _default_behavior_prediction(self) -> Dict[str, float]:
        """Default behavior prediction when model is not trained."""
        return {
            "aggressive": 0.25,
            "defensive": 0.25,
            "balanced": 0.25,
            "conservative": 0.25,
        }

    def _map_cluster_to_behavior(self, cluster_center: np.ndarray) -> Dict[str, float]:
        """Map cluster center to behavior category probabilities."""
        # Extract key features
        velocity = cluster_center[0] if len(cluster_center) > 0 else 0
        thrust = cluster_center[2] if len(cluster_center) > 2 else 0

        # Calculate behavior probabilities
        behaviors = {}

        # Aggressive: high velocity and thrust
        aggressive_score = min(1.0, (velocity / 100) * (thrust / 2000))
        behaviors["aggressive"] = aggressive_score

        # Defensive: moderate velocity, low thrust
        defensive_score = min(1.0, (velocity / 50) * (1 - thrust / 2000))
        behaviors["defensive"] = defensive_score

        # Balanced: moderate everything
        balanced_score = 1.0 - abs(velocity - 50) / 100 - abs(thrust - 1000) / 2000
        behaviors["balanced"] = max(0.0, balanced_score)

        # Conservative: low velocity and thrust
        conservative_score = (1 - velocity / 100) * (1 - thrust / 2000)
        behaviors["conservative"] = conservative_score

        # Normalize
        total = sum(behaviors.values())
        if total > 0:
            for key in behaviors:
                behaviors[key] /= total

        return behaviors

    def get_pilot_profile(self) -> Dict[str, Any]:
        """Get comprehensive pilot behavior profile."""
        if not self.behavior_history:
            return {}

        # Calculate statistics
        velocities = [sample["features"][0] for sample in self.behavior_history]
        thrusts = [sample["features"][2] for sample in self.behavior_history]

        profile = {
            "total_flight_time": len(self.behavior_history),
            "avg_velocity": np.mean(velocities),
            "max_velocity": np.max(velocities),
            "avg_thrust": np.mean(thrusts),
            "max_thrust": np.max(thrusts),
            "preference_weights": dict(self.preference_weights),
            "behavior_categories": self.behavior_categories,
            "last_updated": datetime.now().isoformat(),
        }

        return profile

    def save_model(self, filepath: str):
        """Save the behavior model."""
        model_data = {
            "behavior_history": list(self.behavior_history),
            "preference_weights": dict(self.preference_weights),
            "behavior_clusters": self.behavior_clusters,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "config": self.config,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Behavior model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the behavior model."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.behavior_history = deque(
            model_data["behavior_history"],
            maxlen=self.config.get("history_size", 10000),
        )
        self.preference_weights = defaultdict(float, model_data["preference_weights"])
        self.behavior_clusters = model_data["behavior_clusters"]
        self.scaler = model_data["scaler"]
        self.is_trained = model_data["is_trained"]

        logger.info(f"Behavior model loaded from {filepath}")


class AdaptiveController:
    """Adaptive controller that adjusts to pilot behavior."""

    def __init__(
        self,
        base_controller,
        behavior_model: PilotBehaviorModel,
        config: Optional[Dict] = None,
    ):
        self.base_controller = base_controller
        self.behavior_model = behavior_model
        self.config = config or {}
        self.adaptation_rate = self.config.get("adaptation_rate", 0.1)
        self.adaptation_history = deque(maxlen=1000)

        # Adaptation parameters
        self.control_gains = {
            "thrust_gain": 1.0,
            "maneuver_gain": 1.0,
            "sensitivity_gain": 1.0,
            "automation_level": 0.5,
        }

    def adapt_control(
        self, state: np.ndarray, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Adapt control parameters based on pilot behavior."""
        # Predict pilot behavior
        behavior_prediction = self.behavior_model.predict_behavior(state, context)

        # Calculate adaptation factors
        adaptation_factors = self._calculate_adaptation_factors(
            behavior_prediction, context
        )

        # Update control gains
        self._update_control_gains(adaptation_factors)

        # Store adaptation
        self.adaptation_history.append(
            {
                "state": state.copy(),
                "behavior_prediction": behavior_prediction,
                "adaptation_factors": adaptation_factors,
                "control_gains": self.control_gains.copy(),
                "timestamp": time.time(),
            }
        )

        return self.control_gains

    def _calculate_adaptation_factors(
        self, behavior_prediction: Dict[str, float], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate adaptation factors based on behavior prediction."""
        factors = {}

        # Aggressive behavior: increase responsiveness
        if behavior_prediction.get("aggressive", 0) > 0.5:
            factors["thrust_gain"] = 1.2
            factors["maneuver_gain"] = 1.3
            factors["sensitivity_gain"] = 1.1
            factors["automation_level"] = 0.3  # Less automation for aggressive pilots
        # Defensive behavior: increase automation
        elif behavior_prediction.get("defensive", 0) > 0.5:
            factors["thrust_gain"] = 0.8
            factors["maneuver_gain"] = 0.9
            factors["sensitivity_gain"] = 1.0
            factors["automation_level"] = 0.7  # More automation for defensive pilots
        # Conservative behavior: smooth control
        elif behavior_prediction.get("conservative", 0) > 0.5:
            factors["thrust_gain"] = 0.7
            factors["maneuver_gain"] = 0.8
            factors["sensitivity_gain"] = 0.9
            factors["automation_level"] = 0.8  # High automation for conservative pilots
        # Balanced behavior: moderate adaptation
        else:
            factors["thrust_gain"] = 1.0
            factors["maneuver_gain"] = 1.0
            factors["sensitivity_gain"] = 1.0
            factors["automation_level"] = 0.5

        # Adjust based on threat level
        threat_level = context.get("threat_level", 0.0)
        if threat_level > 0.7:
            factors["automation_level"] = min(0.9, factors["automation_level"] + 0.2)

        # Adjust based on energy level
        energy_level = context.get("energy_level", 1.0)
        if energy_level < 0.3:
            factors["thrust_gain"] *= 0.8  # Conserve energy

        return factors

    def _update_control_gains(self, adaptation_factors: Dict[str, float]):
        """Update control gains using exponential moving average."""
        for key, target_value in adaptation_factors.items():
            if key in self.control_gains:
                current_value = self.control_gains[key]
                new_value = (
                    1 - self.adaptation_rate
                ) * current_value + self.adaptation_rate * target_value
                self.control_gains[key] = new_value

    def get_control_action(
        self, state: np.ndarray, context: Dict[str, Any]
    ) -> np.ndarray:
        """Get adapted control action."""
        # Adapt control parameters
        adapted_gains = self.adapt_control(state, context)

        # Get base control action
        base_action = self.base_controller.get_action(state)

        # Apply adaptation
        adapted_action = self._apply_adaptation(base_action, adapted_gains)

        return adapted_action

    def _apply_adaptation(
        self, base_action: np.ndarray, gains: Dict[str, float]
    ) -> np.ndarray:
        """Apply adaptation gains to base action."""
        adapted_action = base_action.copy()

        if len(adapted_action) >= 1:
            adapted_action[0] *= gains["thrust_gain"]  # Thrust
        if len(adapted_action) >= 2:
            adapted_action[1] *= gains["maneuver_gain"]  # Maneuver

        return adapted_action

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation performance."""
        if not self.adaptation_history:
            return {}

        # Calculate adaptation statistics
        automation_levels = [
            entry["control_gains"]["automation_level"]
            for entry in self.adaptation_history
        ]

        return {
            "total_adaptations": len(self.adaptation_history),
            "avg_automation_level": np.mean(automation_levels),
            "current_gains": self.control_gains,
            "adaptation_rate": self.adaptation_rate,
        }


class PreferenceLearner:
    """Learns pilot preferences from interactions and feedback."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.preferences = defaultdict(float)
        self.feedback_history = deque(
            maxlen=self.config.get("feedback_history_size", 1000)
        )
        self.learning_rate = self.config.get("learning_rate", 0.01)

        # Preference categories
        self.preference_categories = {
            "control_sensitivity": ["high", "medium", "low"],
            "automation_level": ["manual", "assisted", "autonomous"],
            "response_style": ["aggressive", "smooth", "precise"],
            "safety_margin": ["minimal", "standard", "conservative"],
        }

    def add_feedback(
        self,
        context: Dict[str, Any],
        feedback: Dict[str, float],
        timestamp: float = None,
    ):
        """Add pilot feedback to the learning system."""
        if timestamp is None:
            timestamp = time.time()

        feedback_entry = {
            "context": context,
            "feedback": feedback,
            "timestamp": timestamp,
        }

        self.feedback_history.append(feedback_entry)

        # Update preferences based on feedback
        self._update_preferences(feedback_entry)

    def _update_preferences(self, feedback_entry: Dict):
        """Update preferences based on feedback."""
        context = feedback_entry["context"]
        feedback = feedback_entry["feedback"]

        # Extract context features
        threat_level = context.get("threat_level", 0.0)
        mission_phase = context.get("mission_phase", "patrol")
        energy_level = context.get("energy_level", 1.0)

        # Update preferences based on feedback
        for category, score in feedback.items():
            if category in self.preferences:
                # Weighted update based on context
                weight = self._calculate_feedback_weight(context)
                self.preferences[category] += self.learning_rate * weight * score

                # Clamp preferences to reasonable range
                self.preferences[category] = np.clip(
                    self.preferences[category], -1.0, 1.0
                )

    def _calculate_feedback_weight(self, context: Dict[str, Any]) -> float:
        """Calculate weight for feedback based on context importance."""
        weight = 1.0

        # Higher weight for high-threat situations
        threat_level = context.get("threat_level", 0.0)
        weight *= 1.0 + threat_level

        # Higher weight for critical mission phases
        mission_phase = context.get("mission_phase", "patrol")
        if mission_phase in ["combat", "extraction"]:
            weight *= 1.5

        return weight

    def get_preferences(self) -> Dict[str, float]:
        """Get current pilot preferences."""
        return dict(self.preferences)

    def predict_preference(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict pilot preferences for given context."""
        base_preferences = self.get_preferences()

        # Adjust preferences based on context
        adjusted_preferences = base_preferences.copy()

        threat_level = context.get("threat_level", 0.0)
        if threat_level > 0.7:
            # High threat: prefer more automation and safety
            adjusted_preferences["automation_level"] = min(
                1.0, adjusted_preferences.get("automation_level", 0.0) + 0.3
            )
            adjusted_preferences["safety_margin"] = min(
                1.0, adjusted_preferences.get("safety_margin", 0.0) + 0.2
            )

        energy_level = context.get("energy_level", 1.0)
        if energy_level < 0.3:
            # Low energy: prefer conservative control
            adjusted_preferences["response_style"] = max(
                -1.0, adjusted_preferences.get("response_style", 0.0) - 0.3
            )

        return adjusted_preferences

    def save_preferences(self, filepath: str):
        """Save preferences to file."""
        data = {
            "preferences": dict(self.preferences),
            "feedback_history": list(self.feedback_history),
            "config": self.config,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Preferences saved to {filepath}")

    def load_preferences(self, filepath: str):
        """Load preferences from file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.preferences = defaultdict(float, data["preferences"])
        self.feedback_history = deque(
            data["feedback_history"],
            maxlen=self.config.get("feedback_history_size", 1000),
        )

        logger.info(f"Preferences loaded from {filepath}")
