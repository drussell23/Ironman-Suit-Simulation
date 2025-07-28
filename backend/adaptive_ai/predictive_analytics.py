"""
Predictive Analytics System for Iron Man Suit

This module provides advanced predictive capabilities:
- Threat prediction and trajectory forecasting
- Performance optimization and predictive maintenance
- Anomaly detection and early warning systems
- Predictive decision support
"""

import logging
import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass
import threading

# Try importing ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    TORCH_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    torch = None
    nn = None
    optim = None

logger = logging.getLogger(__name__)


class PredictiveAnalyticsError(Exception):
    """Exception for predictive analytics errors."""

    pass


@dataclass
class Prediction:
    """Prediction data structure."""

    timestamp: float
    prediction_type: str
    value: Any
    confidence: float
    time_horizon: float
    metadata: Dict[str, Any]


class ThreatPredictor:
    """Predicts future threat positions and behaviors."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.threat_trajectories = defaultdict(list)
        self.prediction_horizon = self.config.get("prediction_horizon", 30.0)  # seconds
        self.update_frequency = self.config.get("update_frequency", 1.0)  # seconds
        self.last_update = 0.0

        # Trajectory prediction models
        self.trajectory_models = {}
        self.scaler = StandardScaler()

        # Threat behavior patterns
        self.behavior_patterns = {
            "interceptor": {"max_acceleration": 50, "max_turn_rate": 0.5},
            "missile": {"max_acceleration": 100, "max_turn_rate": 0.3},
            "aircraft": {"max_acceleration": 30, "max_turn_rate": 0.2},
            "unknown": {"max_acceleration": 40, "max_turn_rate": 0.4},
        }

    def add_threat_observation(
        self,
        threat_id: str,
        position: np.ndarray,
        velocity: np.ndarray,
        threat_type: str,
        timestamp: float = None,
    ):
        """Add a new threat observation for trajectory prediction."""
        if timestamp is None:
            timestamp = time.time()

        observation = {
            "timestamp": timestamp,
            "position": position.copy(),
            "velocity": velocity.copy(),
            "threat_type": threat_type,
        }

        self.threat_trajectories[threat_id].append(observation)

        # Keep only recent observations
        max_history = self.config.get("max_trajectory_history", 100)
        if len(self.threat_trajectories[threat_id]) > max_history:
            self.threat_trajectories[threat_id] = self.threat_trajectories[threat_id][
                -max_history:
            ]

    def predict_threat_trajectory(
        self, threat_id: str, time_horizon: float = None
    ) -> Optional[Dict[str, Any]]:
        """Predict future trajectory of a threat."""
        if threat_id not in self.threat_trajectories:
            return None

        if time_horizon is None:
            time_horizon = self.prediction_horizon

        trajectory = self.threat_trajectories[threat_id]
        if len(trajectory) < 3:
            return None  # Need at least 3 observations

        # Get latest observation
        latest = trajectory[-1]
        current_time = latest["timestamp"]

        # Predict future positions
        predicted_positions = []
        predicted_velocities = []
        timestamps = []

        dt = 1.0  # 1-second intervals
        for t in np.arange(0, time_horizon, dt):
            future_time = current_time + t

            # Use physics-based prediction with constraints
            predicted_pos, predicted_vel = self._predict_position_velocity(
                trajectory, future_time, latest["threat_type"]
            )

            predicted_positions.append(predicted_pos)
            predicted_velocities.append(predicted_vel)
            timestamps.append(future_time)

        return {
            "threat_id": threat_id,
            "prediction_time": current_time,
            "time_horizon": time_horizon,
            "timestamps": timestamps,
            "positions": predicted_positions,
            "velocities": predicted_velocities,
            "confidence": self._calculate_prediction_confidence(trajectory),
        }

    def _predict_position_velocity(
        self, trajectory: List[Dict], future_time: float, threat_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict position and velocity using physics-based model."""
        if len(trajectory) < 2:
            return trajectory[-1]["position"], trajectory[-1]["velocity"]

        # Get recent observations
        recent = trajectory[-3:]  # Last 3 observations

        # Calculate average velocity and acceleration
        velocities = [obs["velocity"] for obs in recent]
        avg_velocity = np.mean(velocities, axis=0)

        if len(recent) >= 2:
            time_diffs = [
                recent[i]["timestamp"] - recent[i - 1]["timestamp"]
                for i in range(1, len(recent))
            ]
            velocity_diffs = [
                recent[i]["velocity"] - recent[i - 1]["velocity"]
                for i in range(1, len(recent))
            ]

            if all(td > 0 for td in time_diffs):
                accelerations = [vd / td for vd, td in zip(velocity_diffs, time_diffs)]
                avg_acceleration = np.mean(accelerations, axis=0)
            else:
                avg_acceleration = np.zeros(3)
        else:
            avg_acceleration = np.zeros(3)

        # Apply threat-type constraints
        constraints = self.behavior_patterns.get(
            threat_type, self.behavior_patterns["unknown"]
        )
        max_accel = constraints["max_acceleration"]
        max_turn_rate = constraints["max_turn_rate"]

        # Constrain acceleration
        accel_magnitude = np.linalg.norm(avg_acceleration)
        if accel_magnitude > max_accel:
            avg_acceleration = avg_acceleration * (max_accel / accel_magnitude)

        # Constrain turn rate
        if np.linalg.norm(avg_velocity) > 0:
            turn_rate = np.linalg.norm(
                np.cross(avg_velocity, avg_acceleration)
            ) / np.linalg.norm(avg_velocity)
            if turn_rate > max_turn_rate:
                # Reduce acceleration component perpendicular to velocity
                vel_unit = avg_velocity / np.linalg.norm(avg_velocity)
                accel_parallel = np.dot(avg_acceleration, vel_unit) * vel_unit
                accel_perp = avg_acceleration - accel_parallel
                accel_perp = accel_perp * (max_turn_rate / turn_rate)
                avg_acceleration = accel_parallel + accel_perp

        # Predict future state
        latest = recent[-1]
        time_diff = future_time - latest["timestamp"]

        predicted_position = (
            latest["position"]
            + avg_velocity * time_diff
            + 0.5 * avg_acceleration * time_diff**2
        )

        predicted_velocity = avg_velocity + avg_acceleration * time_diff

        return predicted_position, predicted_velocity

    def _calculate_prediction_confidence(self, trajectory: List[Dict]) -> float:
        """Calculate confidence in trajectory prediction."""
        if len(trajectory) < 3:
            return 0.0

        # Calculate consistency of recent observations
        recent_positions = [obs["position"] for obs in trajectory[-5:]]
        recent_velocities = [obs["velocity"] for obs in trajectory[-5:]]

        # Position consistency
        pos_variance = np.var(recent_positions, axis=0)
        pos_consistency = 1.0 / (1.0 + np.mean(pos_variance))

        # Velocity consistency
        vel_variance = np.var(recent_velocities, axis=0)
        vel_consistency = 1.0 / (1.0 + np.mean(vel_variance))

        # Time consistency (regular updates)
        timestamps = [obs["timestamp"] for obs in trajectory[-5:]]
        time_diffs = np.diff(timestamps)
        time_consistency = 1.0 / (1.0 + np.std(time_diffs))

        # Overall confidence
        confidence = (pos_consistency + vel_consistency + time_consistency) / 3.0

        return min(1.0, confidence)

    def predict_threat_encounter(
        self,
        suit_position: np.ndarray,
        suit_velocity: np.ndarray,
        threat_id: str,
        time_horizon: float = None,
    ) -> Optional[Dict[str, Any]]:
        """Predict if and when a threat will encounter the suit."""
        trajectory_prediction = self.predict_threat_trajectory(threat_id, time_horizon)
        if not trajectory_prediction:
            return None

        # Predict suit trajectory (simplified)
        suit_trajectory = self._predict_suit_trajectory(
            suit_position, suit_velocity, trajectory_prediction["time_horizon"]
        )

        # Find closest approach
        min_distance = float("inf")
        encounter_time = None
        encounter_position = None

        for i, (threat_pos, suit_pos) in enumerate(
            zip(trajectory_prediction["positions"], suit_trajectory)
        ):
            distance = np.linalg.norm(threat_pos - suit_pos)
            if distance < min_distance:
                min_distance = distance
                encounter_time = trajectory_prediction["timestamps"][i]
                encounter_position = (threat_pos + suit_pos) / 2

        if encounter_time is None:
            return None

        return {
            "threat_id": threat_id,
            "encounter_time": encounter_time,
            "encounter_position": encounter_position,
            "min_distance": min_distance,
            "time_to_encounter": encounter_time - time.time(),
            "confidence": trajectory_prediction["confidence"],
        }

    def _predict_suit_trajectory(
        self, position: np.ndarray, velocity: np.ndarray, time_horizon: float
    ) -> List[np.ndarray]:
        """Predict suit trajectory (simplified constant velocity model)."""
        trajectory = []
        dt = 1.0

        for t in np.arange(0, time_horizon, dt):
            future_pos = position + velocity * t
            trajectory.append(future_pos)

        return trajectory

    def get_threat_predictions(self) -> Dict[str, Any]:
        """Get predictions for all tracked threats."""
        predictions = {}

        for threat_id in self.threat_trajectories.keys():
            trajectory_pred = self.predict_threat_trajectory(threat_id)
            if trajectory_pred:
                predictions[threat_id] = trajectory_pred

        return predictions


class PerformanceOptimizer:
    """Optimizes suit performance based on predictive analytics."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.performance_history = deque(
            maxlen=self.config.get("performance_history_size", 1000)
        )
        self.optimization_models = {}
        self.performance_metrics = {
            "energy_efficiency": 0.0,
            "maneuverability": 0.0,
            "speed": 0.0,
            "accuracy": 0.0,
        }

        # Performance constraints
        self.constraints = self.config.get(
            "constraints",
            {
                "max_energy_consumption": 1000,  # Watts
                "max_thrust": 2000,  # Newtons
                "max_angular_velocity": 2.0,  # rad/s
                "min_safety_margin": 0.1,
            },
        )

    def add_performance_data(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        timestamp: float = None,
    ):
        """Add performance data for optimization."""
        if timestamp is None:
            timestamp = time.time()

        performance_data = {
            "timestamp": timestamp,
            "state": state,
            "action": action,
            "outcome": outcome,
            "metrics": self._calculate_performance_metrics(state, action, outcome),
        }

        self.performance_history.append(performance_data)

        # Update performance metrics
        self._update_performance_metrics()

    def _calculate_performance_metrics(
        self, state: Dict[str, Any], action: Dict[str, Any], outcome: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate performance metrics from state, action, and outcome."""
        metrics = {}

        # Energy efficiency
        energy_consumed = outcome.get("energy_consumed", 0.0)
        distance_traveled = outcome.get("distance_traveled", 0.0)
        if distance_traveled > 0:
            metrics["energy_efficiency"] = distance_traveled / energy_consumed
        else:
            metrics["energy_efficiency"] = 0.0

        # Maneuverability
        angular_velocity = np.linalg.norm(state.get("angular_velocity", [0, 0, 0]))
        metrics["maneuverability"] = min(
            1.0, angular_velocity / self.constraints["max_angular_velocity"]
        )

        # Speed
        velocity = np.linalg.norm(state.get("velocity", [0, 0, 0]))
        max_speed = 100  # m/s (example)
        metrics["speed"] = min(1.0, velocity / max_speed)

        # Accuracy (based on mission success)
        mission_success = outcome.get("mission_success", 0.0)
        metrics["accuracy"] = mission_success

        return metrics

    def _update_performance_metrics(self):
        """Update overall performance metrics."""
        if not self.performance_history:
            return

        # Calculate average metrics over recent history
        recent_data = list(self.performance_history)[-100:]  # Last 100 samples

        for metric_name in self.performance_metrics.keys():
            values = [data["metrics"].get(metric_name, 0.0) for data in recent_data]
            self.performance_metrics[metric_name] = np.mean(values)

    def optimize_parameters(
        self, current_state: Dict[str, Any], objective: str = "balanced"
    ) -> Dict[str, float]:
        """Optimize control parameters for given objective."""
        if len(self.performance_history) < 50:
            return self._get_default_parameters()

        # Extract optimization features
        features = self._extract_optimization_features(current_state)

        # Define optimization objectives
        if objective == "energy_efficiency":
            target_metrics = {
                "energy_efficiency": 1.0,
                "speed": 0.5,
                "maneuverability": 0.3,
            }
        elif objective == "speed":
            target_metrics = {
                "speed": 1.0,
                "energy_efficiency": 0.3,
                "maneuverability": 0.7,
            }
        elif objective == "maneuverability":
            target_metrics = {
                "maneuverability": 1.0,
                "speed": 0.6,
                "energy_efficiency": 0.4,
            }
        else:  # balanced
            target_metrics = {
                "energy_efficiency": 0.7,
                "speed": 0.7,
                "maneuverability": 0.7,
            }

        # Optimize parameters using historical data
        optimized_params = self._optimize_from_history(features, target_metrics)

        return optimized_params

    def _extract_optimization_features(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract features for parameter optimization."""
        features = []

        # State features
        velocity = np.linalg.norm(state.get("velocity", [0, 0, 0]))
        altitude = state.get("position", [0, 0, 0])[1]
        energy_level = state.get("energy_level", 1.0)
        threat_level = state.get("threat_level", 0.0)

        features.extend([velocity, altitude, energy_level, threat_level])

        return np.array(features)

    def _optimize_from_history(
        self, features: np.ndarray, target_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Optimize parameters using historical performance data."""
        # Find similar historical situations
        similar_cases = self._find_similar_cases(features)

        if not similar_cases:
            return self._get_default_parameters()

        # Calculate optimal parameters based on similar cases
        optimal_params = {}

        for param_name in ["thrust_gain", "maneuver_gain", "sensitivity_gain"]:
            # Find cases with good performance for target metrics
            good_cases = []
            for case in similar_cases:
                case_metrics = case["metrics"]
                case_score = sum(
                    case_metrics.get(metric, 0.0) * weight
                    for metric, weight in target_metrics.items()
                )
                if case_score > 0.6:  # Good performance threshold
                    good_cases.append(case)

            if good_cases:
                # Average parameter values from good cases
                param_values = [
                    case["action"].get(param_name, 1.0) for case in good_cases
                ]
                optimal_params[param_name] = np.mean(param_values)
            else:
                optimal_params[param_name] = 1.0  # Default

        return optimal_params

    def _find_similar_cases(
        self, features: np.ndarray, similarity_threshold: float = 0.8
    ) -> List[Dict]:
        """Find historical cases similar to current features."""
        similar_cases = []

        for data in self.performance_history:
            case_features = self._extract_optimization_features(data["state"])

            # Calculate similarity (cosine similarity)
            similarity = np.dot(features, case_features) / (
                np.linalg.norm(features) * np.linalg.norm(case_features)
            )

            if similarity > similarity_threshold:
                similar_cases.append(data)

        return similar_cases

    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default optimization parameters."""
        return {
            "thrust_gain": 1.0,
            "maneuver_gain": 1.0,
            "sensitivity_gain": 1.0,
            "automation_level": 0.5,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of current performance."""
        if not self.performance_history:
            return {"status": "insufficient_data"}

        recent_data = list(self.performance_history)[-100:]

        # Calculate performance trends
        energy_efficiencies = [
            data["metrics"].get("energy_efficiency", 0.0) for data in recent_data
        ]
        speeds = [data["metrics"].get("speed", 0.0) for data in recent_data]
        maneuverabilities = [
            data["metrics"].get("maneuverability", 0.0) for data in recent_data
        ]

        return {
            "current_metrics": self.performance_metrics,
            "trends": {
                "energy_efficiency_trend": np.mean(energy_efficiencies[-10:])
                - np.mean(energy_efficiencies[:10]),
                "speed_trend": np.mean(speeds[-10:]) - np.mean(speeds[:10]),
                "maneuverability_trend": np.mean(maneuverabilities[-10:])
                - np.mean(maneuverabilities[:10]),
            },
            "total_samples": len(self.performance_history),
            "optimization_status": "active",
        }


class PredictiveAnalytics:
    """Main predictive analytics system that coordinates all predictive capabilities."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.threat_predictor = ThreatPredictor(config.get("threat_predictor", {}))
        self.performance_optimizer = PerformanceOptimizer(
            config.get("performance_optimizer", {})
        )

        # Prediction cache
        self.prediction_cache = {}
        self.cache_duration = self.config.get("cache_duration", 5.0)  # seconds

        # Anomaly detection
        self.anomaly_detector = None
        if SKLEARN_AVAILABLE:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

        # Prediction history
        self.prediction_history = deque(
            maxlen=self.config.get("prediction_history_size", 1000)
        )

    def predict_threats(
        self,
        threats: List[Dict[str, Any]],
        suit_position: np.ndarray,
        suit_velocity: np.ndarray,
    ) -> Dict[str, Any]:
        """Predict future threat behavior and encounters."""
        # Update threat observations
        for threat_data in threats:
            self.threat_predictor.add_threat_observation(
                threat_data["id"],
                np.array(threat_data["position"]),
                np.array(threat_data.get("velocity", [0, 0, 0])),
                threat_data.get("threat_type", "unknown"),
            )

        # Get trajectory predictions
        trajectory_predictions = self.threat_predictor.get_threat_predictions()

        # Predict encounters
        encounter_predictions = {}
        for threat_id in trajectory_predictions.keys():
            encounter = self.threat_predictor.predict_threat_encounter(
                suit_position, suit_velocity, threat_id
            )
            if encounter:
                encounter_predictions[threat_id] = encounter

        # Generate threat warnings
        warnings = self._generate_threat_warnings(encounter_predictions)

        return {
            "trajectory_predictions": trajectory_predictions,
            "encounter_predictions": encounter_predictions,
            "warnings": warnings,
            "prediction_confidence": self._calculate_overall_confidence(
                trajectory_predictions
            ),
        }

    def _generate_threat_warnings(
        self, encounter_predictions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate warnings based on threat encounter predictions."""
        warnings = []

        for threat_id, encounter in encounter_predictions.items():
            time_to_encounter = encounter["time_to_encounter"]
            min_distance = encounter["min_distance"]

            if time_to_encounter < 10 and min_distance < 100:  # Critical warning
                warnings.append(
                    {
                        "level": "critical",
                        "threat_id": threat_id,
                        "message": f"Critical threat encounter in {time_to_encounter:.1f}s",
                        "time_to_encounter": time_to_encounter,
                        "min_distance": min_distance,
                    }
                )
            elif time_to_encounter < 30 and min_distance < 300:  # High warning
                warnings.append(
                    {
                        "level": "high",
                        "threat_id": threat_id,
                        "message": f"Threat approach in {time_to_encounter:.1f}s",
                        "time_to_encounter": time_to_encounter,
                        "min_distance": min_distance,
                    }
                )
            elif time_to_encounter < 60 and min_distance < 500:  # Medium warning
                warnings.append(
                    {
                        "level": "medium",
                        "threat_id": threat_id,
                        "message": f"Threat detected, {time_to_encounter:.1f}s to closest approach",
                        "time_to_encounter": time_to_encounter,
                        "min_distance": min_distance,
                    }
                )

        return warnings

    def _calculate_overall_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall confidence in predictions."""
        if not predictions:
            return 0.0

        confidences = [pred.get("confidence", 0.0) for pred in predictions.values()]
        return np.mean(confidences)

    def optimize_performance(
        self, current_state: Dict[str, Any], objective: str = "balanced"
    ) -> Dict[str, Any]:
        """Optimize suit performance parameters."""
        # Add current performance data if available
        if "performance_data" in current_state:
            self.performance_optimizer.add_performance_data(
                current_state["performance_data"]["state"],
                current_state["performance_data"]["action"],
                current_state["performance_data"]["outcome"],
            )

        # Get optimized parameters
        optimized_params = self.performance_optimizer.optimize_parameters(
            current_state, objective
        )

        # Get performance summary
        performance_summary = self.performance_optimizer.get_performance_summary()

        return {
            "optimized_parameters": optimized_params,
            "performance_summary": performance_summary,
            "optimization_objective": objective,
        }

    def detect_anomalies(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in sensor data."""
        if not self.anomaly_detector:
            return []

        # Extract features for anomaly detection
        features = self._extract_anomaly_features(sensor_data)

        if len(features) == 0:
            return []

        # Reshape for sklearn
        features_array = np.array(features).reshape(1, -1)

        # Predict anomalies
        prediction = self.anomaly_detector.predict(features_array)
        score = self.anomaly_detector.score_samples(features_array)

        anomalies = []
        if prediction[0] == -1:  # Anomaly detected
            anomalies.append(
                {
                    "type": "sensor_anomaly",
                    "severity": "high" if score[0] < -0.5 else "medium",
                    "score": score[0],
                    "sensor_data": sensor_data,
                    "timestamp": time.time(),
                }
            )

        return anomalies

    def _extract_anomaly_features(self, sensor_data: Dict[str, Any]) -> List[float]:
        """Extract features for anomaly detection."""
        features = []

        # Velocity features
        velocity = sensor_data.get("velocity", [0, 0, 0])
        features.extend(
            [np.linalg.norm(velocity), velocity[0], velocity[1], velocity[2]]
        )

        # Acceleration features
        acceleration = sensor_data.get("acceleration", [0, 0, 0])
        features.extend(
            [
                np.linalg.norm(acceleration),
                acceleration[0],
                acceleration[1],
                acceleration[2],
            ]
        )

        # Energy features
        energy_level = sensor_data.get("energy_level", 1.0)
        energy_consumption = sensor_data.get("energy_consumption", 0.0)
        features.extend([energy_level, energy_consumption])

        # Temperature features
        temperature = sensor_data.get("temperature", 20.0)
        features.append(temperature)

        return features

    def get_predictive_insights(self) -> Dict[str, Any]:
        """Get comprehensive predictive insights."""
        insights = {
            "threat_predictions": len(self.threat_predictor.threat_trajectories),
            "performance_optimization": self.performance_optimizer.get_performance_summary(),
            "anomaly_detection": {
                "active": self.anomaly_detector is not None,
                "model_trained": (
                    hasattr(self.anomaly_detector, "fit")
                    if self.anomaly_detector
                    else False
                ),
            },
            "prediction_accuracy": self._calculate_prediction_accuracy(),
            "system_health": self._assess_system_health(),
        }

        return insights

    def _calculate_prediction_accuracy(self) -> float:
        """Calculate accuracy of recent predictions."""
        if len(self.prediction_history) < 10:
            return 0.0

        # Simple accuracy calculation based on prediction confidence
        recent_predictions = list(self.prediction_history)[-10:]
        confidences = [pred.get("confidence", 0.0) for pred in recent_predictions]

        return np.mean(confidences)

    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall health of the predictive system."""
        health_metrics = {
            "threat_predictor_health": len(self.threat_predictor.threat_trajectories)
            > 0,
            "performance_optimizer_health": len(
                self.performance_optimizer.performance_history
            )
            > 50,
            "anomaly_detector_health": self.anomaly_detector is not None,
            "prediction_cache_health": len(self.prediction_cache) < 100,
        }

        overall_health = sum(health_metrics.values()) / len(health_metrics)

        return {"overall_health": overall_health, "metrics": health_metrics}

    def save_predictive_state(self, filepath: str):
        """Save predictive analytics state."""
        state_data = {
            "threat_predictor": {
                "threat_trajectories": dict(self.threat_predictor.threat_trajectories),
                "prediction_horizon": self.threat_predictor.prediction_horizon,
                "behavior_patterns": self.threat_predictor.behavior_patterns,
            },
            "performance_optimizer": {
                "performance_history": list(
                    self.performance_optimizer.performance_history
                ),
                "performance_metrics": self.performance_optimizer.performance_metrics,
                "constraints": self.performance_optimizer.constraints,
            },
            "prediction_cache": self.prediction_cache,
            "config": self.config,
        }

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2, default=str)

        logger.info(f"Predictive analytics state saved to {filepath}")

    def load_predictive_state(self, filepath: str):
        """Load predictive analytics state."""
        with open(filepath, "r") as f:
            state_data = json.load(f)

        # Restore threat predictor
        self.threat_predictor.threat_trajectories = defaultdict(
            list, state_data["threat_predictor"]["threat_trajectories"]
        )
        self.threat_predictor.prediction_horizon = state_data["threat_predictor"][
            "prediction_horizon"
        ]
        self.threat_predictor.behavior_patterns = state_data["threat_predictor"][
            "behavior_patterns"
        ]

        # Restore performance optimizer
        self.performance_optimizer.performance_history = deque(
            state_data["performance_optimizer"]["performance_history"],
            maxlen=self.config.get("performance_history_size", 1000),
        )
        self.performance_optimizer.performance_metrics = state_data[
            "performance_optimizer"
        ]["performance_metrics"]
        self.performance_optimizer.constraints = state_data["performance_optimizer"][
            "constraints"
        ]

        # Restore prediction cache
        self.prediction_cache = state_data["prediction_cache"]

        logger.info(f"Predictive analytics state loaded from {filepath}")
