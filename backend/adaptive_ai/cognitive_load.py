"""
Cognitive Load Management System for Iron Man Suit

This module manages pilot cognitive load by:
- Assessing current workload and stress levels
- Dynamically adjusting automation levels
- Balancing manual control with AI assistance
- Providing workload-aware interface adaptations
"""

import logging
import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CognitiveLoadError(Exception):
    """Exception for cognitive load management errors."""

    pass


class WorkloadLevel(Enum):
    """Workload level enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AutomationLevel(Enum):
    """Automation level enumeration."""

    MANUAL = "manual"
    ASSISTED = "assisted"
    SEMI_AUTONOMOUS = "semi_autonomous"
    AUTONOMOUS = "autonomous"


@dataclass
class WorkloadMetrics:
    """Workload metrics data structure."""

    timestamp: float
    overall_load: float
    visual_load: float
    auditory_load: float
    cognitive_load: float
    physical_load: float
    stress_level: float
    fatigue_level: float


@dataclass
class AutomationDecision:
    """Automation decision data structure."""

    timestamp: float
    recommended_level: AutomationLevel
    confidence: float
    reasoning: str
    constraints: Dict[str, Any]


class CognitiveLoadManager:
    """Manages and assesses pilot cognitive load."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.workload_history = deque(
            maxlen=self.config.get("workload_history_size", 1000)
        )
        self.baseline_workload = self.config.get("baseline_workload", 0.3)

        # Workload thresholds
        self.workload_thresholds = self.config.get(
            "workload_thresholds",
            {"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 0.9},
        )

        # Workload factors
        self.workload_factors = {
            "task_complexity": 0.3,
            "environmental_stress": 0.2,
            "time_pressure": 0.2,
            "physical_strain": 0.15,
            "information_overload": 0.15,
        }

        # Pilot state tracking
        self.pilot_state = {
            "fatigue_level": 0.0,
            "stress_level": 0.0,
            "alertness": 1.0,
            "performance_trend": 0.0,
        }

    def assess_workload(
        self, sensor_data: Dict[str, Any], task_data: Dict[str, Any]
    ) -> WorkloadMetrics:
        """Assess current cognitive workload."""
        timestamp = time.time()

        # Extract workload indicators
        visual_load = self._assess_visual_load(sensor_data)
        auditory_load = self._assess_auditory_load(sensor_data)
        cognitive_load = self._assess_cognitive_load(task_data)
        physical_load = self._assess_physical_load(sensor_data)

        # Calculate overall workload
        overall_load = (
            self.workload_factors["task_complexity"] * cognitive_load
            + self.workload_factors["environmental_stress"]
            * (visual_load + auditory_load)
            / 2
            + self.workload_factors["time_pressure"]
            * task_data.get("time_pressure", 0.0)
            + self.workload_factors["physical_strain"] * physical_load
            + self.workload_factors["information_overload"]
            * task_data.get("info_overload", 0.0)
        )

        # Adjust for pilot state
        overall_load *= 1.0 + self.pilot_state["fatigue_level"]
        overall_load *= 1.0 + self.pilot_state["stress_level"]

        # Clamp to valid range
        overall_load = np.clip(overall_load, 0.0, 1.0)

        # Create workload metrics
        metrics = WorkloadMetrics(
            timestamp=timestamp,
            overall_load=overall_load,
            visual_load=visual_load,
            auditory_load=auditory_load,
            cognitive_load=cognitive_load,
            physical_load=physical_load,
            stress_level=self.pilot_state["stress_level"],
            fatigue_level=self.pilot_state["fatigue_level"],
        )

        # Store in history
        self.workload_history.append(metrics)

        # Update pilot state
        self._update_pilot_state(metrics)

        return metrics

    def _assess_visual_load(self, sensor_data: Dict[str, Any]) -> float:
        """Assess visual workload from sensor data."""
        visual_load = 0.0

        # HUD complexity
        hud_elements = sensor_data.get("hud_elements", 0)
        visual_load += min(1.0, hud_elements / 20.0) * 0.3

        # Environmental complexity
        threat_count = sensor_data.get("threat_count", 0)
        visual_load += min(1.0, threat_count / 10.0) * 0.4

        # Visual alerts
        alert_count = sensor_data.get("alert_count", 0)
        visual_load += min(1.0, alert_count / 5.0) * 0.3

        return np.clip(visual_load, 0.0, 1.0)

    def _assess_auditory_load(self, sensor_data: Dict[str, Any]) -> float:
        """Assess auditory workload from sensor data."""
        auditory_load = 0.0

        # Audio alerts
        audio_alerts = sensor_data.get("audio_alerts", 0)
        auditory_load += min(1.0, audio_alerts / 3.0) * 0.5

        # Communication load
        comm_frequency = sensor_data.get("comm_frequency", 0.0)
        auditory_load += min(1.0, comm_frequency / 10.0) * 0.5

        return np.clip(auditory_load, 0.0, 1.0)

    def _assess_cognitive_load(self, task_data: Dict[str, Any]) -> float:
        """Assess cognitive workload from task data."""
        cognitive_load = 0.0

        # Task complexity
        task_complexity = task_data.get("complexity", 0.0)
        cognitive_load += task_complexity * 0.4

        # Decision frequency
        decisions_per_minute = task_data.get("decisions_per_minute", 0.0)
        cognitive_load += min(1.0, decisions_per_minute / 20.0) * 0.3

        # Memory load
        memory_items = task_data.get("memory_items", 0)
        cognitive_load += min(1.0, memory_items / 10.0) * 0.3

        return np.clip(cognitive_load, 0.0, 1.0)

    def _assess_physical_load(self, sensor_data: Dict[str, Any]) -> float:
        """Assess physical workload from sensor data."""
        physical_load = 0.0

        # G-force
        g_force = sensor_data.get("g_force", 1.0)
        physical_load += min(1.0, (g_force - 1.0) / 5.0) * 0.4

        # Maneuver intensity
        maneuver_intensity = sensor_data.get("maneuver_intensity", 0.0)
        physical_load += maneuver_intensity * 0.3

        # Duration of high-intensity activity
        high_intensity_duration = sensor_data.get("high_intensity_duration", 0.0)
        physical_load += min(1.0, high_intensity_duration / 300.0) * 0.3  # 5 minutes

        return np.clip(physical_load, 0.0, 1.0)

    def _update_pilot_state(self, metrics: WorkloadMetrics):
        """Update pilot state based on workload metrics."""
        # Update fatigue (accumulates over time)
        if metrics.overall_load > 0.7:
            self.pilot_state["fatigue_level"] += 0.01
        else:
            self.pilot_state["fatigue_level"] = max(
                0.0, self.pilot_state["fatigue_level"] - 0.005
            )

        # Update stress (responds to immediate workload)
        self.pilot_state["stress_level"] = metrics.overall_load * 0.8

        # Update alertness (inverse of fatigue)
        self.pilot_state["alertness"] = 1.0 - self.pilot_state["fatigue_level"]

        # Update performance trend
        if len(self.workload_history) >= 10:
            recent_loads = [m.overall_load for m in list(self.workload_history)[-10:]]
            self.pilot_state["performance_trend"] = (
                np.mean(recent_loads) - self.baseline_workload
            )

    def get_workload_level(self, metrics: WorkloadMetrics) -> WorkloadLevel:
        """Get workload level from metrics."""
        if metrics.overall_load >= self.workload_thresholds["critical"]:
            return WorkloadLevel.CRITICAL
        elif metrics.overall_load >= self.workload_thresholds["high"]:
            return WorkloadLevel.HIGH
        elif metrics.overall_load >= self.workload_thresholds["medium"]:
            return WorkloadLevel.MEDIUM
        else:
            return WorkloadLevel.LOW

    def get_workload_summary(self) -> Dict[str, Any]:
        """Get summary of workload assessment."""
        if not self.workload_history:
            return {"status": "insufficient_data"}

        recent_metrics = list(self.workload_history)[-10:]

        return {
            "current_workload": (
                recent_metrics[-1].overall_load if recent_metrics else 0.0
            ),
            "workload_trend": np.mean([m.overall_load for m in recent_metrics]),
            "pilot_state": self.pilot_state,
            "workload_distribution": {
                "visual": np.mean([m.visual_load for m in recent_metrics]),
                "auditory": np.mean([m.auditory_load for m in recent_metrics]),
                "cognitive": np.mean([m.cognitive_load for m in recent_metrics]),
                "physical": np.mean([m.physical_load for m in recent_metrics]),
            },
        }


class AutomationLevelController:
    """Controls automation levels based on cognitive load."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.current_level = AutomationLevel.ASSISTED
        self.automation_history = deque(
            maxlen=self.config.get("automation_history_size", 100)
        )

        # Automation level mappings
        self.level_mappings = {
            WorkloadLevel.LOW: AutomationLevel.MANUAL,
            WorkloadLevel.MEDIUM: AutomationLevel.ASSISTED,
            WorkloadLevel.HIGH: AutomationLevel.SEMI_AUTONOMOUS,
            WorkloadLevel.CRITICAL: AutomationLevel.AUTONOMOUS,
        }

        # Safety constraints
        self.safety_constraints = self.config.get(
            "safety_constraints",
            {
                "min_automation_in_combat": AutomationLevel.ASSISTED,
                "max_automation_in_combat": AutomationLevel.SEMI_AUTONOMOUS,
                "emergency_automation": AutomationLevel.AUTONOMOUS,
            },
        )

        # Transition rules
        self.transition_rules = {
            "min_duration": 5.0,  # seconds
            "hysteresis": 0.1,  # prevent rapid switching
            "emergency_threshold": 0.9,
        }

    def determine_automation_level(
        self, workload_metrics: WorkloadMetrics, mission_context: Dict[str, Any]
    ) -> AutomationDecision:
        """Determine appropriate automation level."""
        timestamp = time.time()

        # Get workload-based recommendation
        workload_level = self._get_workload_level(workload_metrics)
        recommended_level = self.level_mappings[workload_level]

        # Apply safety constraints
        constrained_level = self._apply_safety_constraints(
            recommended_level, mission_context
        )

        # Apply transition rules
        final_level = self._apply_transition_rules(constrained_level, workload_metrics)

        # Calculate confidence
        confidence = self._calculate_confidence(workload_metrics, mission_context)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            workload_level, constrained_level, mission_context
        )

        # Create decision
        decision = AutomationDecision(
            timestamp=timestamp,
            recommended_level=final_level,
            confidence=confidence,
            reasoning=reasoning,
            constraints=self._get_active_constraints(mission_context),
        )

        # Store decision
        self.automation_history.append(decision)

        return decision

    def _get_workload_level(self, metrics: WorkloadMetrics) -> WorkloadLevel:
        """Get workload level from metrics."""
        if metrics.overall_load >= 0.9:
            return WorkloadLevel.CRITICAL
        elif metrics.overall_load >= 0.7:
            return WorkloadLevel.HIGH
        elif metrics.overall_load >= 0.4:
            return WorkloadLevel.MEDIUM
        else:
            return WorkloadLevel.LOW

    def _apply_safety_constraints(
        self, recommended_level: AutomationLevel, mission_context: Dict[str, Any]
    ) -> AutomationLevel:
        """Apply safety constraints to automation level."""
        # Combat constraints
        if mission_context.get("in_combat", False):
            min_level = self.safety_constraints["min_automation_in_combat"]
            max_level = self.safety_constraints["max_automation_in_combat"]

            if recommended_level.value < min_level.value:
                recommended_level = min_level
            elif recommended_level.value > max_level.value:
                recommended_level = max_level

        # Emergency constraints
        if mission_context.get("emergency", False):
            recommended_level = self.safety_constraints["emergency_automation"]

        # Energy constraints
        energy_level = mission_context.get("energy_level", 1.0)
        if energy_level < 0.2:
            # Low energy: increase automation to conserve pilot energy
            if recommended_level.value < AutomationLevel.SEMI_AUTONOMOUS.value:
                recommended_level = AutomationLevel.SEMI_AUTONOMOUS

        return recommended_level

    def _apply_transition_rules(
        self, recommended_level: AutomationLevel, metrics: WorkloadMetrics
    ) -> AutomationLevel:
        """Apply transition rules to prevent rapid switching."""
        if not self.automation_history:
            return recommended_level

        last_decision = self.automation_history[-1]
        time_since_last = time.time() - last_decision.timestamp

        # Check minimum duration
        if time_since_last < self.transition_rules["min_duration"]:
            return last_decision.recommended_level

        # Check hysteresis
        workload_change = abs(
            metrics.overall_load
            - self._get_workload_from_level(last_decision.recommended_level)
        )
        if workload_change < self.transition_rules["hysteresis"]:
            return last_decision.recommended_level

        # Emergency override
        if metrics.overall_load >= self.transition_rules["emergency_threshold"]:
            return AutomationLevel.AUTONOMOUS

        return recommended_level

    def _get_workload_from_level(self, level: AutomationLevel) -> float:
        """Get typical workload for automation level."""
        workload_mapping = {
            AutomationLevel.MANUAL: 0.3,
            AutomationLevel.ASSISTED: 0.5,
            AutomationLevel.SEMI_AUTONOMOUS: 0.7,
            AutomationLevel.AUTONOMOUS: 0.9,
        }
        return workload_mapping.get(level, 0.5)

    def _calculate_confidence(
        self, metrics: WorkloadMetrics, mission_context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in automation decision."""
        confidence = 0.8  # Base confidence

        # Adjust based on workload stability
        if len(self.automation_history) >= 5:
            recent_levels = [
                d.recommended_level for d in list(self.automation_history)[-5:]
            ]
            stability = len(set(recent_levels)) / len(recent_levels)
            confidence *= 1.0 - stability * 0.3  # More stable = higher confidence

        # Adjust based on mission context clarity
        context_clarity = 1.0
        if mission_context.get("in_combat", False):
            context_clarity = 0.9
        if mission_context.get("emergency", False):
            context_clarity = 0.7

        confidence *= context_clarity

        return np.clip(confidence, 0.0, 1.0)

    def _generate_reasoning(
        self,
        workload_level: WorkloadLevel,
        automation_level: AutomationLevel,
        mission_context: Dict[str, Any],
    ) -> str:
        """Generate reasoning for automation decision."""
        reasons = []

        # Workload-based reasoning
        reasons.append(f"Workload level: {workload_level.value}")

        # Context-based reasoning
        if mission_context.get("in_combat", False):
            reasons.append("Combat situation detected")
        if mission_context.get("emergency", False):
            reasons.append("Emergency situation")
        if mission_context.get("energy_level", 1.0) < 0.3:
            reasons.append("Low energy levels")

        # Safety reasoning
        if automation_level == AutomationLevel.AUTONOMOUS:
            reasons.append("Maximum automation for safety")
        elif automation_level == AutomationLevel.MANUAL:
            reasons.append("Manual control for pilot preference")

        return "; ".join(reasons)

    def _get_active_constraints(
        self, mission_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get currently active constraints."""
        constraints = {}

        if mission_context.get("in_combat", False):
            constraints["combat_mode"] = True
        if mission_context.get("emergency", False):
            constraints["emergency_mode"] = True
        if mission_context.get("energy_level", 1.0) < 0.3:
            constraints["low_energy"] = True

        return constraints

    def get_automation_summary(self) -> Dict[str, Any]:
        """Get summary of automation decisions."""
        if not self.automation_history:
            return {"status": "no_decisions"}

        recent_decisions = list(self.automation_history)[-10:]

        # Calculate automation statistics
        level_counts = defaultdict(int)
        for decision in recent_decisions:
            level_counts[decision.recommended_level.value] += 1

        return {
            "current_level": self.current_level.value,
            "recent_levels": level_counts,
            "avg_confidence": np.mean([d.confidence for d in recent_decisions]),
            "transition_frequency": self._calculate_transition_frequency(),
            "safety_compliance": self._assess_safety_compliance(),
        }

    def _calculate_transition_frequency(self) -> float:
        """Calculate frequency of automation level transitions."""
        if len(self.automation_history) < 2:
            return 0.0

        transitions = 0
        for i in range(1, len(self.automation_history)):
            if (
                self.automation_history[i].recommended_level
                != self.automation_history[i - 1].recommended_level
            ):
                transitions += 1

        time_span = (
            self.automation_history[-1].timestamp - self.automation_history[0].timestamp
        )

        return transitions / max(time_span, 1.0)  # transitions per second

    def _assess_safety_compliance(self) -> Dict[str, Any]:
        """Assess compliance with safety constraints."""
        if not self.automation_history:
            return {"status": "insufficient_data"}

        recent_decisions = list(self.automation_history)[-20:]

        # Check for emergency situations
        emergency_decisions = [
            d
            for d in recent_decisions
            if d.recommended_level == AutomationLevel.AUTONOMOUS
        ]

        # Check for combat situations
        combat_decisions = [
            d for d in recent_decisions if "combat_mode" in d.constraints
        ]

        return {
            "emergency_responses": len(emergency_decisions),
            "combat_adaptations": len(combat_decisions),
            "safety_violations": 0,  # Would be calculated based on actual violations
        }


class WorkloadBalancer:
    """Balances workload across different systems and interfaces."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.interface_adaptations = {}
        self.system_priorities = self.config.get(
            "system_priorities",
            {
                "flight_control": 1.0,
                "weapons_system": 0.8,
                "communications": 0.6,
                "sensor_management": 0.7,
                "navigation": 0.5,
            },
        )

        # Interface adaptation rules
        self.adaptation_rules = {
            "hud_simplification": {"threshold": 0.7, "priority": 0.9},
            "audio_filtering": {"threshold": 0.6, "priority": 0.7},
            "control_assistance": {"threshold": 0.5, "priority": 0.8},
            "information_filtering": {"threshold": 0.8, "priority": 0.9},
        }

    def balance_workload(
        self,
        workload_metrics: WorkloadMetrics,
        automation_level: AutomationLevel,
        system_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Balance workload across systems and interfaces."""
        adaptations = {}

        # Interface adaptations
        adaptations["interface"] = self._adapt_interfaces(
            workload_metrics, automation_level
        )

        # System priority adjustments
        adaptations["system_priorities"] = self._adjust_system_priorities(
            workload_metrics, system_status
        )

        # Information filtering
        adaptations["information_filtering"] = self._filter_information(
            workload_metrics
        )

        # Control assistance
        adaptations["control_assistance"] = self._provide_control_assistance(
            workload_metrics, automation_level
        )

        return adaptations

    def _adapt_interfaces(
        self, metrics: WorkloadMetrics, automation_level: AutomationLevel
    ) -> Dict[str, Any]:
        """Adapt user interfaces based on workload."""
        adaptations = {}

        # HUD simplification
        if (
            metrics.overall_load
            > self.adaptation_rules["hud_simplification"]["threshold"]
        ):
            adaptations["hud_simplification"] = {
                "enabled": True,
                "level": min(1.0, (metrics.overall_load - 0.7) / 0.3),
                "elements_to_hide": [
                    "detailed_telemetry",
                    "secondary_targets",
                    "background_data",
                ],
            }

        # Audio filtering
        if (
            metrics.auditory_load
            > self.adaptation_rules["audio_filtering"]["threshold"]
        ):
            adaptations["audio_filtering"] = {
                "enabled": True,
                "priority_threshold": 0.8,
                "filtered_channels": ["background_noise", "low_priority_alerts"],
            }

        # Visual alert reduction
        if metrics.visual_load > 0.8:
            adaptations["visual_alerts"] = {
                "critical_only": True,
                "reduced_frequency": True,
                "simplified_icons": True,
            }

        return adaptations

    def _adjust_system_priorities(
        self, metrics: WorkloadMetrics, system_status: Dict[str, Any]
    ) -> Dict[str, float]:
        """Adjust system priorities based on workload."""
        adjusted_priorities = self.system_priorities.copy()

        # Reduce non-critical system priorities under high workload
        if metrics.overall_load > 0.7:
            for system, priority in adjusted_priorities.items():
                if system not in ["flight_control", "weapons_system"]:
                    adjusted_priorities[system] *= 0.7

        # Increase critical system priorities under stress
        if metrics.stress_level > 0.6:
            adjusted_priorities["flight_control"] = min(
                1.0, adjusted_priorities["flight_control"] * 1.2
            )
            adjusted_priorities["weapons_system"] = min(
                1.0, adjusted_priorities["weapons_system"] * 1.1
            )

        return adjusted_priorities

    def _filter_information(self, metrics: WorkloadMetrics) -> Dict[str, Any]:
        """Filter information based on cognitive load."""
        filtering = {
            "enabled": metrics.overall_load
            > self.adaptation_rules["information_filtering"]["threshold"],
            "filters": [],
        }

        if filtering["enabled"]:
            # Filter based on workload components
            if metrics.visual_load > 0.7:
                filtering["filters"].append("reduce_visual_elements")

            if metrics.auditory_load > 0.6:
                filtering["filters"].append("prioritize_audio_alerts")

            if metrics.cognitive_load > 0.8:
                filtering["filters"].append("simplify_decision_data")

            if metrics.physical_load > 0.7:
                filtering["filters"].append("minimize_manual_inputs")

        return filtering

    def _provide_control_assistance(
        self, metrics: WorkloadMetrics, automation_level: AutomationLevel
    ) -> Dict[str, Any]:
        """Provide control assistance based on workload."""
        assistance = {
            "enabled": metrics.overall_load
            > self.adaptation_rules["control_assistance"]["threshold"],
            "features": [],
        }

        if assistance["enabled"]:
            # Add assistance features based on workload
            if metrics.cognitive_load > 0.6:
                assistance["features"].append("decision_support")

            if metrics.physical_load > 0.5:
                assistance["features"].append("control_stabilization")

            if metrics.visual_load > 0.7:
                assistance["features"].append("target_assistance")

            if metrics.overall_load > 0.8:
                assistance["features"].append("predictive_control")

        return assistance

    def get_balancing_summary(self) -> Dict[str, Any]:
        """Get summary of workload balancing."""
        return {
            "active_adaptations": len(self.interface_adaptations),
            "system_priorities": self.system_priorities,
            "adaptation_rules": self.adaptation_rules,
            "balancing_status": "active",
        }
