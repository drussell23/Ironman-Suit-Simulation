"""
Advanced Tactical Decision Making System for Iron Man Suit

This module provides comprehensive tactical decision-making capabilities:
- Threat assessment and prioritization
- Mission planning and execution
- Multi-objective decision optimization
- Real-time tactical adaptation
"""

import logging
import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import threading

# Try importing optimization libraries
try:
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class TacticalDecisionError(Exception):
    """Exception for tactical decision-making errors."""

    pass


class ThreatLevel(Enum):
    """Threat level enumeration."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MissionPhase(Enum):
    """Mission phase enumeration."""

    PREPARATION = "preparation"
    APPROACH = "approach"
    ENGAGEMENT = "engagement"
    EVASION = "evasion"
    EXTRACTION = "extraction"
    COMPLETE = "complete"


@dataclass
class Threat:
    """Threat data structure."""

    id: str
    position: np.ndarray
    velocity: np.ndarray
    threat_level: ThreatLevel
    threat_type: str
    capabilities: Dict[str, Any]
    last_seen: float
    confidence: float


@dataclass
class MissionObjective:
    """Mission objective data structure."""

    id: str
    description: str
    position: np.ndarray
    priority: float
    time_constraint: Optional[float]
    status: str  # 'pending', 'in_progress', 'completed', 'failed'


class ThreatAssessment:
    """Advanced threat assessment and analysis system."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.threats = {}
        self.threat_history = deque(maxlen=self.config.get("threat_history_size", 1000))
        self.assessment_weights = self.config.get(
            "assessment_weights",
            {"distance": 0.3, "velocity": 0.2, "capabilities": 0.3, "behavior": 0.2},
        )

        # Threat type capabilities
        self.threat_capabilities = {
            "missile": {"range": 5000, "speed": 800, "damage": 0.8},
            "aircraft": {"range": 3000, "speed": 600, "damage": 0.6},
            "ground_vehicle": {"range": 1000, "speed": 50, "damage": 0.4},
            "unknown": {"range": 2000, "speed": 400, "damage": 0.5},
        }

    def add_threat(self, threat_data: Dict[str, Any]) -> str:
        """Add or update a threat."""
        threat_id = threat_data["id"]

        threat = Threat(
            id=threat_id,
            position=np.array(threat_data["position"]),
            velocity=np.array(threat_data.get("velocity", [0, 0, 0])),
            threat_level=ThreatLevel(threat_data.get("threat_level", 1)),
            threat_type=threat_data.get("threat_type", "unknown"),
            capabilities=threat_data.get("capabilities", {}),
            last_seen=time.time(),
            confidence=threat_data.get("confidence", 0.5),
        )

        self.threats[threat_id] = threat
        self.threat_history.append(threat)

        return threat_id

    def remove_threat(self, threat_id: str):
        """Remove a threat from tracking."""
        if threat_id in self.threats:
            del self.threats[threat_id]

    def assess_threat_level(
        self, threat: Threat, suit_position: np.ndarray, suit_velocity: np.ndarray
    ) -> float:
        """Assess the threat level of a specific threat."""
        # Calculate distance-based threat
        distance = np.linalg.norm(threat.position - suit_position)
        distance_threat = self._calculate_distance_threat(distance, threat)

        # Calculate velocity-based threat
        relative_velocity = np.linalg.norm(threat.velocity - suit_velocity)
        velocity_threat = self._calculate_velocity_threat(relative_velocity, threat)

        # Calculate capability-based threat
        capability_threat = self._calculate_capability_threat(threat)

        # Calculate behavior-based threat
        behavior_threat = self._calculate_behavior_threat(threat)

        # Weighted combination
        total_threat = (
            self.assessment_weights["distance"] * distance_threat
            + self.assessment_weights["velocity"] * velocity_threat
            + self.assessment_weights["capabilities"] * capability_threat
            + self.assessment_weights["behavior"] * behavior_threat
        )

        return total_threat

    def _calculate_distance_threat(self, distance: float, threat: Threat) -> float:
        """Calculate threat based on distance."""
        threat_range = self.threat_capabilities.get(threat.threat_type, {}).get(
            "range", 2000
        )

        if distance <= threat_range:
            # Within range: threat decreases with distance
            return 1.0 - (distance / threat_range)
        else:
            # Outside range: minimal threat
            return 0.1

    def _calculate_velocity_threat(
        self, relative_velocity: float, threat: Threat
    ) -> float:
        """Calculate threat based on relative velocity."""
        max_speed = self.threat_capabilities.get(threat.threat_type, {}).get(
            "speed", 400
        )

        # Higher relative velocity indicates more aggressive behavior
        return min(1.0, relative_velocity / max_speed)

    def _calculate_capability_threat(self, threat: Threat) -> float:
        """Calculate threat based on capabilities."""
        base_capabilities = self.threat_capabilities.get(threat.threat_type, {})

        # Combine base capabilities with specific threat capabilities
        damage_potential = base_capabilities.get("damage", 0.5)
        if "damage" in threat.capabilities:
            damage_potential = max(damage_potential, threat.capabilities["damage"])

        return damage_potential

    def _calculate_behavior_threat(self, threat: Threat) -> float:
        """Calculate threat based on behavior patterns."""
        # Analyze recent behavior from threat history
        recent_threats = [t for t in self.threat_history if t.id == threat.id]

        if len(recent_threats) < 2:
            return 0.5  # Default threat level

        # Calculate behavior metrics
        velocities = [np.linalg.norm(t.velocity) for t in recent_threats[-5:]]
        avg_velocity = np.mean(velocities)

        # Higher average velocity suggests more aggressive behavior
        max_speed = self.threat_capabilities.get(threat.threat_type, {}).get(
            "speed", 400
        )
        behavior_threat = min(1.0, avg_velocity / max_speed)

        return behavior_threat

    def get_prioritized_threats(
        self, suit_position: np.ndarray, suit_velocity: np.ndarray
    ) -> List[Threat]:
        """Get threats prioritized by threat level."""
        threat_scores = []

        for threat in self.threats.values():
            # Check if threat is still relevant (seen recently)
            if time.time() - threat.last_seen > self.config.get("threat_timeout", 30):
                continue

            threat_score = self.assess_threat_level(
                threat, suit_position, suit_velocity
            )
            threat_scores.append((threat, threat_score))

        # Sort by threat score (highest first)
        threat_scores.sort(key=lambda x: x[1], reverse=True)

        return [threat for threat, score in threat_scores]

    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of current threat situation."""
        if not self.threats:
            return {"total_threats": 0, "highest_threat_level": "NONE"}

        threat_levels = [threat.threat_level.value for threat in self.threats.values()]
        threat_types = [threat.threat_type for threat in self.threats.values()]

        return {
            "total_threats": len(self.threats),
            "highest_threat_level": ThreatLevel(max(threat_levels)).name,
            "threat_types": list(set(threat_types)),
            "avg_threat_level": np.mean(threat_levels),
        }


class MissionPlanner:
    """Advanced mission planning and execution system."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.objectives = {}
        self.current_phase = MissionPhase.PREPARATION
        self.phase_transitions = {
            MissionPhase.PREPARATION: [MissionPhase.APPROACH],
            MissionPhase.APPROACH: [MissionPhase.ENGAGEMENT, MissionPhase.EVASION],
            MissionPhase.ENGAGEMENT: [MissionPhase.EVASION, MissionPhase.EXTRACTION],
            MissionPhase.EVASION: [MissionPhase.APPROACH, MissionPhase.EXTRACTION],
            MissionPhase.EXTRACTION: [MissionPhase.COMPLETE],
        }

        # Mission parameters
        self.mission_start_time = None
        self.phase_start_time = None
        self.phase_timeout = self.config.get("phase_timeout", 300)  # 5 minutes

    def start_mission(self, objectives: List[Dict[str, Any]]):
        """Start a new mission with given objectives."""
        self.mission_start_time = time.time()
        self.phase_start_time = time.time()
        self.current_phase = MissionPhase.PREPARATION

        # Initialize objectives
        self.objectives.clear()
        for obj_data in objectives:
            objective = MissionObjective(
                id=obj_data["id"],
                description=obj_data["description"],
                position=np.array(obj_data["position"]),
                priority=obj_data.get("priority", 1.0),
                time_constraint=obj_data.get("time_constraint"),
                status="pending",
            )
            self.objectives[objective.id] = objective

        logger.info(f"Mission started with {len(objectives)} objectives")

    def update_mission_status(
        self, suit_position: np.ndarray, threat_assessment: ThreatAssessment
    ) -> Dict[str, Any]:
        """Update mission status and determine next actions."""
        # Check phase timeout
        if (
            self.phase_start_time
            and time.time() - self.phase_start_time > self.phase_timeout
        ):
            self._handle_phase_timeout()

        # Update objective status
        self._update_objective_status(suit_position)

        # Determine next phase
        next_phase = self._determine_next_phase(threat_assessment)

        # Generate mission actions
        mission_actions = self._generate_mission_actions(
            suit_position, threat_assessment
        )

        return {
            "current_phase": self.current_phase.value,
            "next_phase": next_phase.value if next_phase else None,
            "objectives": self._get_objective_summary(),
            "actions": mission_actions,
            "mission_progress": self._calculate_mission_progress(),
        }

    def _handle_phase_timeout(self):
        """Handle phase timeout by transitioning to appropriate phase."""
        if self.current_phase == MissionPhase.APPROACH:
            self._transition_phase(MissionPhase.EVASION)
        elif self.current_phase == MissionPhase.ENGAGEMENT:
            self._transition_phase(MissionPhase.EXTRACTION)
        else:
            self._transition_phase(MissionPhase.EXTRACTION)

    def _update_objective_status(self, suit_position: np.ndarray):
        """Update the status of mission objectives."""
        for objective in self.objectives.values():
            if objective.status != "pending":
                continue

            # Check if objective is reached
            distance = np.linalg.norm(objective.position - suit_position)
            if distance < self.config.get("objective_reach_threshold", 50):
                objective.status = "completed"
                logger.info(f"Objective {objective.id} completed")

            # Check time constraints
            if objective.time_constraint and self.mission_start_time:
                elapsed_time = time.time() - self.mission_start_time
                if elapsed_time > objective.time_constraint:
                    objective.status = "failed"
                    logger.warning(f"Objective {objective.id} failed due to timeout")

    def _determine_next_phase(
        self, threat_assessment: ThreatAssessment
    ) -> Optional[MissionPhase]:
        """Determine the next mission phase based on current situation."""
        available_phases = self.phase_transitions.get(self.current_phase, [])

        if not available_phases:
            return None

        # Get threat summary
        threat_summary = threat_assessment.get_threat_summary()

        # Decision logic based on current phase and threats
        if self.current_phase == MissionPhase.PREPARATION:
            return MissionPhase.APPROACH

        elif self.current_phase == MissionPhase.APPROACH:
            if threat_summary["highest_threat_level"] in ["HIGH", "CRITICAL"]:
                return MissionPhase.EVASION
            else:
                return MissionPhase.ENGAGEMENT

        elif self.current_phase == MissionPhase.ENGAGEMENT:
            if threat_summary["highest_threat_level"] == "CRITICAL":
                return MissionPhase.EVASION
            elif self._are_objectives_complete():
                return MissionPhase.EXTRACTION
            else:
                return None  # Stay in current phase

        elif self.current_phase == MissionPhase.EVASION:
            if threat_summary["highest_threat_level"] in ["NONE", "LOW"]:
                return MissionPhase.APPROACH
            else:
                return MissionPhase.EXTRACTION

        return None

    def _are_objectives_complete(self) -> bool:
        """Check if all objectives are completed."""
        return all(obj.status == "completed" for obj in self.objectives.values())

    def _generate_mission_actions(
        self, suit_position: np.ndarray, threat_assessment: ThreatAssessment
    ) -> List[Dict[str, Any]]:
        """Generate specific actions for the current mission phase."""
        actions = []

        if self.current_phase == MissionPhase.PREPARATION:
            actions.append({"type": "prepare_systems", "priority": "high"})

        elif self.current_phase == MissionPhase.APPROACH:
            # Find nearest pending objective
            pending_objectives = [
                obj for obj in self.objectives.values() if obj.status == "pending"
            ]
            if pending_objectives:
                nearest_obj = min(
                    pending_objectives,
                    key=lambda obj: np.linalg.norm(obj.position - suit_position),
                )
                actions.append(
                    {
                        "type": "move_to_position",
                        "target": nearest_obj.position.tolist(),
                        "priority": "high",
                    }
                )

        elif self.current_phase == MissionPhase.ENGAGEMENT:
            # Prioritize objectives by priority and distance
            pending_objectives = [
                obj for obj in self.objectives.values() if obj.status == "pending"
            ]
            if pending_objectives:
                # Sort by priority first, then by distance
                pending_objectives.sort(
                    key=lambda obj: (
                        -obj.priority,
                        np.linalg.norm(obj.position - suit_position),
                    )
                )
                target_obj = pending_objectives[0]
                actions.append(
                    {
                        "type": "execute_objective",
                        "objective_id": target_obj.id,
                        "priority": "high",
                    }
                )

        elif self.current_phase == MissionPhase.EVASION:
            # Get highest priority threat
            prioritized_threats = threat_assessment.get_prioritized_threats(
                suit_position, np.zeros(3)
            )
            if prioritized_threats:
                threat = prioritized_threats[0]
                # Calculate evasion direction (away from threat)
                evasion_direction = suit_position - threat.position
                evasion_direction = evasion_direction / np.linalg.norm(
                    evasion_direction
                )
                evasion_position = suit_position + evasion_direction * 1000  # 1km away

                actions.append(
                    {
                        "type": "evade_threat",
                        "threat_id": threat.id,
                        "direction": evasion_direction.tolist(),
                        "target_position": evasion_position.tolist(),
                        "priority": "critical",
                    }
                )

        elif self.current_phase == MissionPhase.EXTRACTION:
            # Move to extraction point
            extraction_point = self.config.get("extraction_point", [0, 100, 0])
            actions.append(
                {"type": "extract", "target": extraction_point, "priority": "high"}
            )

        return actions

    def _transition_phase(self, new_phase: MissionPhase):
        """Transition to a new mission phase."""
        logger.info(
            f"Mission phase transition: {self.current_phase.value} -> {new_phase.value}"
        )
        self.current_phase = new_phase
        self.phase_start_time = time.time()

    def _get_objective_summary(self) -> Dict[str, Any]:
        """Get summary of mission objectives."""
        status_counts = defaultdict(int)
        for objective in self.objectives.values():
            status_counts[objective.status] += 1

        return {
            "total": len(self.objectives),
            "pending": status_counts["pending"],
            "in_progress": status_counts["in_progress"],
            "completed": status_counts["completed"],
            "failed": status_counts["failed"],
        }

    def _calculate_mission_progress(self) -> float:
        """Calculate overall mission progress (0.0 to 1.0)."""
        if not self.objectives:
            return 0.0

        completed = sum(
            1 for obj in self.objectives.values() if obj.status == "completed"
        )
        total = len(self.objectives)

        return completed / total


class TacticalDecisionEngine:
    """Main tactical decision engine that coordinates all tactical systems."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.threat_assessment = ThreatAssessment(config.get("threat_assessment", {}))
        self.mission_planner = MissionPlanner(config.get("mission_planner", {}))

        # Decision weights
        self.decision_weights = self.config.get(
            "decision_weights",
            {
                "threat_avoidance": 0.4,
                "mission_completion": 0.3,
                "resource_conservation": 0.2,
                "pilot_preference": 0.1,
            },
        )

        # Decision history
        self.decision_history = deque(
            maxlen=self.config.get("decision_history_size", 1000)
        )

    def make_tactical_decision(
        self,
        state: Dict[str, Any],
        pilot_preferences: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Make a comprehensive tactical decision."""
        try:
            # Extract state components
            suit_position = np.array(state.get("position", [0, 0, 0]))
            suit_velocity = np.array(state.get("velocity", [0, 0, 0]))
            suit_energy = state.get("energy_level", 1.0)

            # Update threat assessment
            if "threats" in state:
                for threat_data in state["threats"]:
                    self.threat_assessment.add_threat(threat_data)

            # Get threat analysis
            prioritized_threats = self.threat_assessment.get_prioritized_threats(
                suit_position, suit_velocity
            )
            threat_summary = self.threat_assessment.get_threat_summary()

            # Update mission status
            mission_status = self.mission_planner.update_mission_status(
                suit_position, self.threat_assessment
            )

            # Generate tactical options
            tactical_options = self._generate_tactical_options(
                suit_position,
                suit_velocity,
                suit_energy,
                prioritized_threats,
                mission_status,
                pilot_preferences,
            )

            # Evaluate and select best option
            best_option = self._evaluate_tactical_options(tactical_options, state)

            # Record decision
            decision_record = {
                "timestamp": time.time(),
                "state": state,
                "threat_summary": threat_summary,
                "mission_status": mission_status,
                "selected_option": best_option,
                "all_options": tactical_options,
            }
            self.decision_history.append(decision_record)

            return best_option

        except Exception as e:
            logger.error(f"Error in tactical decision: {e}")
            raise TacticalDecisionError(str(e))

    def _generate_tactical_options(
        self,
        suit_position: np.ndarray,
        suit_velocity: np.ndarray,
        suit_energy: float,
        threats: List[Threat],
        mission_status: Dict[str, Any],
        pilot_preferences: Optional[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Generate possible tactical options."""
        options = []

        # Option 1: Continue mission
        if mission_status["actions"]:
            for action in mission_status["actions"]:
                options.append(
                    {
                        "type": "mission_action",
                        "action": action,
                        "priority": action.get("priority", "medium"),
                        "expected_reward": self._estimate_reward(
                            "mission", action, threats, suit_energy
                        ),
                    }
                )

        # Option 2: Threat response
        if threats:
            for threat in threats[:3]:  # Consider top 3 threats
                # Evasion option
                evasion_direction = suit_position - threat.position
                evasion_direction = evasion_direction / np.linalg.norm(
                    evasion_direction
                )
                evasion_position = suit_position + evasion_direction * 500

                options.append(
                    {
                        "type": "threat_response",
                        "action": "evade",
                        "threat_id": threat.id,
                        "target_position": evasion_position.tolist(),
                        "priority": "high",
                        "expected_reward": self._estimate_reward(
                            "evasion", threat, threats, suit_energy
                        ),
                    }
                )

                # Engagement option (if appropriate)
                if threat.threat_level.value <= ThreatLevel.MEDIUM.value:
                    options.append(
                        {
                            "type": "threat_response",
                            "action": "engage",
                            "threat_id": threat.id,
                            "target_position": threat.position.tolist(),
                            "priority": "medium",
                            "expected_reward": self._estimate_reward(
                                "engagement", threat, threats, suit_energy
                            ),
                        }
                    )

        # Option 3: Resource management
        if suit_energy < 0.3:
            options.append(
                {
                    "type": "resource_management",
                    "action": "conserve_energy",
                    "priority": "high",
                    "expected_reward": self._estimate_reward(
                        "conservation", None, threats, suit_energy
                    ),
                }
            )

        return options

    def _estimate_reward(
        self,
        action_type: str,
        action_data: Any,
        threats: List[Threat],
        suit_energy: float,
    ) -> float:
        """Estimate the expected reward for a tactical option."""
        base_reward = 0.0

        if action_type == "mission":
            base_reward = 1.0  # High reward for mission completion

        elif action_type == "evasion":
            base_reward = 0.8  # Good reward for avoiding threats

        elif action_type == "engagement":
            base_reward = 0.6  # Moderate reward for engagement

        elif action_type == "conservation":
            base_reward = 0.7  # Good reward for resource management

        # Adjust based on threat level
        if threats:
            max_threat_level = max(threat.threat_level.value for threat in threats)
            threat_penalty = max_threat_level * 0.1
            base_reward -= threat_penalty

        # Adjust based on energy level
        if suit_energy < 0.2:
            base_reward *= 0.5  # Severe penalty for low energy

        return max(0.0, base_reward)

    def _evaluate_tactical_options(
        self, options: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate and select the best tactical option."""
        if not options:
            return {"type": "no_action", "reason": "no_options_available"}

        # Score each option
        scored_options = []
        for option in options:
            score = self._score_option(option, state)
            scored_options.append((option, score))

        # Select best option
        best_option, best_score = max(scored_options, key=lambda x: x[1])

        logger.info(
            f"Selected tactical option: {best_option['type']} (score: {best_score:.3f})"
        )

        return best_option

    def _score_option(self, option: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Score a tactical option based on multiple criteria."""
        score = 0.0

        # Base reward
        score += option.get("expected_reward", 0.0)

        # Priority bonus
        priority = option.get("priority", "medium")
        priority_bonus = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.0}
        score += priority_bonus.get(priority, 0.0)

        # Threat avoidance consideration
        if option["type"] == "threat_response" and option["action"] == "evade":
            score += self.decision_weights["threat_avoidance"]

        # Mission completion consideration
        if option["type"] == "mission_action":
            score += self.decision_weights["mission_completion"]

        # Resource conservation consideration
        if option["type"] == "resource_management":
            score += self.decision_weights["resource_conservation"]

        return score

    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of recent tactical decisions."""
        if not self.decision_history:
            return {}

        recent_decisions = list(self.decision_history)[-10:]  # Last 10 decisions

        decision_types = [d["selected_option"]["type"] for d in recent_decisions]
        type_counts = defaultdict(int)
        for decision_type in decision_types:
            type_counts[decision_type] += 1

        return {
            "total_decisions": len(self.decision_history),
            "recent_decision_types": dict(type_counts),
            "last_decision": recent_decisions[-1] if recent_decisions else None,
        }

    def save_tactical_state(self, filepath: str):
        """Save tactical decision system state."""
        state_data = {
            "threat_assessment": {
                "threats": {
                    tid: {
                        "position": threat.position.tolist(),
                        "velocity": threat.velocity.tolist(),
                        "threat_level": threat.threat_level.value,
                        "threat_type": threat.threat_type,
                        "capabilities": threat.capabilities,
                        "last_seen": threat.last_seen,
                        "confidence": threat.confidence,
                    }
                    for tid, threat in self.threat_assessment.threats.items()
                },
                "assessment_weights": self.threat_assessment.assessment_weights,
            },
            "mission_planner": {
                "current_phase": self.mission_planner.current_phase.value,
                "objectives": {
                    oid: {
                        "description": obj.description,
                        "position": obj.position.tolist(),
                        "priority": obj.priority,
                        "time_constraint": obj.time_constraint,
                        "status": obj.status,
                    }
                    for oid, obj in self.mission_planner.objectives.items()
                },
                "mission_start_time": self.mission_planner.mission_start_time,
                "phase_start_time": self.mission_planner.phase_start_time,
            },
            "decision_weights": self.decision_weights,
            "config": self.config,
        }

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2)

        logger.info(f"Tactical state saved to {filepath}")

    def load_tactical_state(self, filepath: str):
        """Load tactical decision system state."""
        with open(filepath, "r") as f:
            state_data = json.load(f)

        # Restore threat assessment
        self.threat_assessment.threats.clear()
        for tid, threat_data in state_data["threat_assessment"]["threats"].items():
            threat = Threat(
                id=tid,
                position=np.array(threat_data["position"]),
                velocity=np.array(threat_data["velocity"]),
                threat_level=ThreatLevel(threat_data["threat_level"]),
                threat_type=threat_data["threat_type"],
                capabilities=threat_data["capabilities"],
                last_seen=threat_data["last_seen"],
                confidence=threat_data["confidence"],
            )
            self.threat_assessment.threats[tid] = threat

        # Restore mission planner
        self.mission_planner.current_phase = MissionPhase(
            state_data["mission_planner"]["current_phase"]
        )
        self.mission_planner.objectives.clear()
        for oid, obj_data in state_data["mission_planner"]["objectives"].items():
            objective = MissionObjective(
                id=oid,
                description=obj_data["description"],
                position=np.array(obj_data["position"]),
                priority=obj_data["priority"],
                time_constraint=obj_data["time_constraint"],
                status=obj_data["status"],
            )
            self.mission_planner.objectives[oid] = objective

        self.mission_planner.mission_start_time = state_data["mission_planner"][
            "mission_start_time"
        ]
        self.mission_planner.phase_start_time = state_data["mission_planner"][
            "phase_start_time"
        ]

        logger.info(f"Tactical state loaded from {filepath}")
