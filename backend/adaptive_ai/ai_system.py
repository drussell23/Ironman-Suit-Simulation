"""
Main Adaptive AI System for Iron Man Suit

This module integrates all AI components:
- Reinforcement Learning agents
- Tactical Decision Making
- Behavioral Adaptation
- Predictive Analytics
- Cognitive Load Management
- Multi-agent coordination
"""

import logging
import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import threading

from .reinforcement_learning import DQNAgent, PPOAgent, SACAgent, MultiAgentCoordinator
from .tactical_decision import TacticalDecisionEngine, ThreatAssessment, MissionPlanner
from .behavioral_adaptation import (
    PilotBehaviorModel,
    AdaptiveController,
    PreferenceLearner,
)
from .predictive_analytics import (
    PredictiveAnalytics,
    ThreatPredictor,
    PerformanceOptimizer,
)
from .cognitive_load import (
    CognitiveLoadManager,
    AutomationLevelController,
    WorkloadBalancer,
)

logger = logging.getLogger(__name__)


class AISystemError(Exception):
    """Exception for AI system errors."""

    pass


@dataclass
class AISystemConfig:
    """Configuration for the AI system."""

    # Component configurations
    reinforcement_learning: Dict[str, Any]
    tactical_decision: Dict[str, Any]
    behavioral_adaptation: Dict[str, Any]
    predictive_analytics: Dict[str, Any]
    cognitive_load: Dict[str, Any]

    # System parameters
    update_frequency: float = 10.0  # Hz
    decision_timeout: float = 0.1  # seconds
    enable_learning: bool = True
    enable_prediction: bool = True
    enable_adaptation: bool = True

    # Safety parameters
    max_automation_level: str = "semi_autonomous"
    emergency_override: bool = True
    pilot_override_priority: bool = True


class AdaptiveAISystem:
    """Main adaptive AI system that coordinates all AI components."""

    def __init__(self, config: Optional[AISystemConfig] = None):
        self.config = config or AISystemConfig(
            reinforcement_learning={},
            tactical_decision={},
            behavioral_adaptation={},
            predictive_analytics={},
            cognitive_load={},
        )

        # Initialize AI components
        self._initialize_components()

        # System state
        self.system_state = {
            "active": True,
            "last_update": time.time(),
            "decision_count": 0,
            "learning_enabled": self.config.enable_learning,
            "prediction_enabled": self.config.enable_prediction,
            "adaptation_enabled": self.config.enable_adaptation,
        }

        # Decision history
        self.decision_history = deque(maxlen=1000)

        # Performance metrics
        self.performance_metrics = {
            "decision_latency": [],
            "prediction_accuracy": [],
            "adaptation_effectiveness": [],
            "system_health": 1.0,
        }

        # Threading for real-time operation
        self.update_thread = None
        self.running = False

    def _initialize_components(self):
        """Initialize all AI system components."""
        try:
            # Reinforcement Learning
            self.rl_agents = {
                "flight_control": DQNAgent(
                    state_dim=12, action_dim=4, **self.config.reinforcement_learning
                ),
                "weapons_control": PPOAgent(
                    state_dim=8, action_dim=3, **self.config.reinforcement_learning
                ),
                "navigation": SACAgent(
                    state_dim=10, action_dim=3, **self.config.reinforcement_learning
                ),
            }

            self.multi_agent_coordinator = MultiAgentCoordinator(
                self.rl_agents, coordination_strategy="hierarchical"
            )

            # Tactical Decision Making
            self.tactical_engine = TacticalDecisionEngine(self.config.tactical_decision)

            # Behavioral Adaptation
            self.behavior_model = PilotBehaviorModel(self.config.behavioral_adaptation)
            self.adaptive_controller = AdaptiveController(
                base_controller=None,  # Will be set by external system
                behavior_model=self.behavior_model,
                config=self.config.behavioral_adaptation,
            )
            self.preference_learner = PreferenceLearner(
                self.config.behavioral_adaptation
            )

            # Predictive Analytics
            self.predictive_analytics = PredictiveAnalytics(
                self.config.predictive_analytics
            )

            # Cognitive Load Management
            self.cognitive_load_manager = CognitiveLoadManager(
                self.config.cognitive_load
            )
            self.automation_controller = AutomationLevelController(
                self.config.cognitive_load
            )
            self.workload_balancer = WorkloadBalancer(self.config.cognitive_load)

            logger.info("All AI components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI components: {e}")
            raise AISystemError(f"Component initialization failed: {e}")

    def start(self):
        """Start the AI system."""
        if self.running:
            logger.warning("AI system already running")
            return

        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("AI system started")

    def stop(self):
        """Stop the AI system."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        logger.info("AI system stopped")

    def _update_loop(self):
        """Main update loop for the AI system."""
        while self.running:
            try:
                start_time = time.time()

                # Process any pending updates
                self._process_updates()

                # Sleep to maintain update frequency
                elapsed = time.time() - start_time
                sleep_time = max(0.0, 1.0 / self.config.update_frequency - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in AI system update loop: {e}")
                time.sleep(1.0)  # Brief pause on error

    def _process_updates(self):
        """Process pending updates and maintain system state."""
        current_time = time.time()
        self.system_state["last_update"] = current_time

        # Update performance metrics
        self._update_performance_metrics()

        # Check system health
        self._check_system_health()

    def make_decision(
        self, state: Dict[str, Any], pilot_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a comprehensive AI decision."""
        start_time = time.time()

        try:
            # Extract state components
            suit_state = state.get("suit_state", {})
            environment_state = state.get("environment_state", {})
            mission_state = state.get("mission_state", {})

            # 1. Assess cognitive load and automation level
            workload_metrics = self.cognitive_load_manager.assess_workload(
                sensor_data=suit_state, task_data=mission_state
            )

            automation_decision = self.automation_controller.determine_automation_level(
                workload_metrics=workload_metrics, mission_context=mission_state
            )

            # 2. Predict threats and optimize performance
            if self.config.enable_prediction:
                threat_predictions = self.predictive_analytics.predict_threats(
                    threats=environment_state.get("threats", []),
                    suit_position=np.array(suit_state.get("position", [0, 0, 0])),
                    suit_velocity=np.array(suit_state.get("velocity", [0, 0, 0])),
                )

                performance_optimization = (
                    self.predictive_analytics.optimize_performance(
                        current_state=suit_state, objective="balanced"
                    )
                )
            else:
                threat_predictions = {}
                performance_optimization = {}

            # 3. Make tactical decisions
            tactical_decision = self.tactical_engine.make_tactical_decision(
                state={
                    "suit_state": suit_state,
                    "environment_state": environment_state,
                    "mission_state": mission_state,
                    "threat_predictions": threat_predictions,
                    "performance_optimization": performance_optimization,
                },
                pilot_preferences=self.preference_learner.get_preferences(),
            )

            # 4. Adapt behavior based on pilot preferences
            if self.config.enable_adaptation:
                behavior_prediction = self.behavior_model.predict_behavior(
                    state=np.array(suit_state.get("position", [0, 0, 0])),
                    context=mission_state,
                )

                adapted_control = self.adaptive_controller.adapt_control(
                    state=np.array(suit_state.get("position", [0, 0, 0])),
                    context=mission_state,
                )
            else:
                behavior_prediction = {}
                adapted_control = {}

            # 5. Balance workload
            workload_balancing = self.workload_balancer.balance_workload(
                workload_metrics=workload_metrics,
                automation_level=automation_decision.recommended_level,
                system_status=suit_state,
            )

            # 6. Coordinate multi-agent actions
            agent_states = {
                "flight_control": np.array(suit_state.get("flight_state", [0] * 12)),
                "weapons_control": np.array(suit_state.get("weapons_state", [0] * 8)),
                "navigation": np.array(suit_state.get("navigation_state", [0] * 10)),
            }

            coordinated_actions = self.multi_agent_coordinator.coordinate_actions(
                agent_states
            )

            # 7. Integrate all decisions
            integrated_decision = self._integrate_decisions(
                tactical_decision=tactical_decision,
                automation_decision=automation_decision,
                adapted_control=adapted_control,
                coordinated_actions=coordinated_actions,
                workload_balancing=workload_balancing,
                pilot_input=pilot_input,
            )

            # 8. Record decision
            decision_record = {
                "timestamp": time.time(),
                "decision": integrated_decision,
                "workload_metrics": workload_metrics,
                "automation_level": automation_decision.recommended_level.value,
                "processing_time": time.time() - start_time,
            }
            self.decision_history.append(decision_record)

            # 9. Update learning if enabled
            if self.config.enable_learning:
                self._update_learning(state, integrated_decision)

            # 10. Update performance metrics
            self.performance_metrics["decision_latency"].append(
                time.time() - start_time
            )

            return integrated_decision

        except Exception as e:
            logger.error(f"Error in AI decision making: {e}")
            return self._get_fallback_decision(state)

    def _integrate_decisions(
        self,
        tactical_decision: Dict[str, Any],
        automation_decision: AutomationDecision,
        adapted_control: Dict[str, Any],
        coordinated_actions: Dict[str, Any],
        workload_balancing: Dict[str, Any],
        pilot_input: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Integrate all AI decisions into a coherent action plan."""

        # Start with tactical decision
        integrated_decision = tactical_decision.copy()

        # Apply automation level
        integrated_decision["automation_level"] = (
            automation_decision.recommended_level.value
        )
        integrated_decision["automation_confidence"] = automation_decision.confidence
        integrated_decision["automation_reasoning"] = automation_decision.reasoning

        # Apply adapted control parameters
        if adapted_control:
            integrated_decision["control_parameters"] = adapted_control

        # Apply coordinated agent actions
        if coordinated_actions:
            integrated_decision["agent_actions"] = coordinated_actions

        # Apply workload balancing
        if workload_balancing:
            integrated_decision["interface_adaptations"] = workload_balancing

        # Apply pilot override if provided
        if pilot_input and self.config.pilot_override_priority:
            integrated_decision["pilot_override"] = pilot_input
            integrated_decision["automation_level"] = "manual"  # Override automation

        # Add safety checks
        integrated_decision["safety_checks"] = self._perform_safety_checks(
            integrated_decision
        )

        return integrated_decision

    def _perform_safety_checks(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Perform safety checks on the integrated decision."""
        safety_checks = {"passed": True, "warnings": [], "constraints_violated": []}

        # Check automation level constraints
        automation_level = decision.get("automation_level", "assisted")
        if automation_level == "autonomous" and not self.config.emergency_override:
            safety_checks["warnings"].append(
                "Autonomous mode requires emergency override"
            )

        # Check control parameter bounds
        control_params = decision.get("control_parameters", {})
        if "thrust_gain" in control_params and control_params["thrust_gain"] > 2.0:
            safety_checks["constraints_violated"].append(
                "Thrust gain exceeds safety limit"
            )
            safety_checks["passed"] = False

        # Check for conflicting actions
        agent_actions = decision.get("agent_actions", {})
        if len(agent_actions) > 0:
            # Check for conflicts between different agents
            # This is a simplified check - real implementation would be more sophisticated
            pass

        return safety_checks

    def _get_fallback_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get a fallback decision when AI system fails."""
        return {
            "type": "fallback",
            "automation_level": "assisted",
            "actions": ["maintain_current_state"],
            "safety_checks": {"passed": True, "warnings": ["Using fallback decision"]},
            "reasoning": "AI system error - using fallback decision",
        }

    def _update_learning(self, state: Dict[str, Any], decision: Dict[str, Any]):
        """Update learning components with new experience."""
        try:
            # Update behavior model
            if "pilot_behavior" in state:
                self.behavior_model.add_behavior_sample(
                    state=np.array(state["suit_state"].get("position", [0, 0, 0])),
                    action=np.array(decision.get("actions", [0, 0])),
                    context=state.get("mission_state", {}),
                )

            # Update preference learner
            if "pilot_feedback" in state:
                self.preference_learner.add_feedback(
                    context=state.get("mission_state", {}),
                    feedback=state["pilot_feedback"],
                )

            # Update performance optimizer
            if "performance_data" in state:
                self.predictive_analytics.performance_optimizer.add_performance_data(
                    state=state["suit_state"],
                    action=decision,
                    outcome=state["performance_data"],
                )

            # Train RL agents
            for agent_name, agent in self.rl_agents.items():
                if hasattr(agent, "train") and "training_data" in state:
                    agent.train(state["training_data"].get(agent_name, ()))

        except Exception as e:
            logger.error(f"Error updating learning components: {e}")

    def _update_performance_metrics(self):
        """Update system performance metrics."""
        # Calculate average decision latency
        if self.performance_metrics["decision_latency"]:
            recent_latencies = self.performance_metrics["decision_latency"][-100:]
            avg_latency = np.mean(recent_latencies)

            # Update system health based on latency
            if avg_latency < 0.05:  # 50ms
                self.performance_metrics["system_health"] = 1.0
            elif avg_latency < 0.1:  # 100ms
                self.performance_metrics["system_health"] = 0.8
            elif avg_latency < 0.2:  # 200ms
                self.performance_metrics["system_health"] = 0.6
            else:
                self.performance_metrics["system_health"] = 0.4

    def _check_system_health(self):
        """Check overall system health."""
        health_indicators = []

        # Check component health
        try:
            # Test RL agents
            for agent_name, agent in self.rl_agents.items():
                test_state = np.zeros(agent.state_dim)
                agent.select_action(test_state)
            health_indicators.append(1.0)
        except Exception:
            health_indicators.append(0.0)

        # Check tactical engine
        try:
            test_state = {
                "suit_state": {},
                "environment_state": {},
                "mission_state": {},
            }
            self.tactical_engine.make_tactical_decision(test_state)
            health_indicators.append(1.0)
        except Exception:
            health_indicators.append(0.0)

        # Check predictive analytics
        try:
            self.predictive_analytics.get_predictive_insights()
            health_indicators.append(1.0)
        except Exception:
            health_indicators.append(0.0)

        # Update system health
        if health_indicators:
            self.performance_metrics["system_health"] = np.mean(health_indicators)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_state": self.system_state,
            "performance_metrics": self.performance_metrics,
            "component_status": {
                "reinforcement_learning": len(self.rl_agents),
                "tactical_decision": "active",
                "behavioral_adaptation": "active",
                "predictive_analytics": "active",
                "cognitive_load": "active",
            },
            "decision_summary": {
                "total_decisions": len(self.decision_history),
                "avg_latency": (
                    np.mean(self.performance_metrics["decision_latency"][-100:])
                    if self.performance_metrics["decision_latency"]
                    else 0.0
                ),
                "system_health": self.performance_metrics["system_health"],
            },
            "recent_decisions": (
                list(self.decision_history)[-5:] if self.decision_history else []
            ),
        }

    def save_system_state(self, filepath: str):
        """Save the complete AI system state."""
        state_data = {
            "system_state": self.system_state,
            "performance_metrics": self.performance_metrics,
            "config": self.config.__dict__,
            "component_states": {
                "behavior_model": self.behavior_model.get_pilot_profile(),
                "preference_learner": self.preference_learner.get_preferences(),
                "tactical_engine": self.tactical_engine.get_decision_summary(),
                "predictive_analytics": self.predictive_analytics.get_predictive_insights(),
                "cognitive_load": self.cognitive_load_manager.get_workload_summary(),
                "automation_controller": self.automation_controller.get_automation_summary(),
            },
        }

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2, default=str)

        logger.info(f"AI system state saved to {filepath}")

    def load_system_state(self, filepath: str):
        """Load the complete AI system state."""
        with open(filepath, "r") as f:
            state_data = json.load(f)

        # Restore system state
        self.system_state.update(state_data["system_state"])
        self.performance_metrics.update(state_data["performance_metrics"])

        # Restore component states
        # Note: This is a simplified restoration - full restoration would require
        # more sophisticated state management for each component

        logger.info(f"AI system state loaded from {filepath}")

    def reset_system(self):
        """Reset the AI system to initial state."""
        self.stop()

        # Reinitialize components
        self._initialize_components()

        # Reset state
        self.system_state = {
            "active": True,
            "last_update": time.time(),
            "decision_count": 0,
            "learning_enabled": self.config.enable_learning,
            "prediction_enabled": self.config.enable_prediction,
            "adaptation_enabled": self.config.enable_adaptation,
        }

        self.decision_history.clear()
        self.performance_metrics = {
            "decision_latency": [],
            "prediction_accuracy": [],
            "adaptation_effectiveness": [],
            "system_health": 1.0,
        }

        logger.info("AI system reset to initial state")
