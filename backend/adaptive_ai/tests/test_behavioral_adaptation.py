"""
Unit tests for the Behavioral Adaptation module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime, timedelta

from adaptive_ai.behavioral_adaptation import (
    PilotBehaviorModel,
    AdaptiveController,
    PreferenceLearner,
    BehavioralAdaptationError
)


class TestPilotBehaviorModel:
    """Test PilotBehaviorModel functionality"""
    
    def test_initialization(self):
        """Test behavior model initialization"""
        config = {"history_size": 5000, "learning_rate": 0.01}
        model = PilotBehaviorModel(config)
        
        assert model.config == config
        assert len(model.behavior_history) == 0
        assert model.behavior_patterns is not None
        assert model.pilot_profile is not None
    
    def test_record_behavior(self, sample_pilot_action):
        """Test behavior recording"""
        model = PilotBehaviorModel()
        
        # Record behavior
        model.record_behavior(sample_pilot_action)
        
        assert len(model.behavior_history) == 1
        assert model.behavior_history[0]["control_input"] is not None
    
    def test_behavior_history_limit(self, sample_pilot_action):
        """Test behavior history size limit"""
        model = PilotBehaviorModel({"history_size": 5})
        
        # Record more than limit
        for i in range(10):
            action = sample_pilot_action.copy()
            action["timestamp"] = time.time() + i
            model.record_behavior(action)
        
        assert len(model.behavior_history) == 5
    
    def test_analyze_patterns(self, sample_pilot_action):
        """Test behavior pattern analysis"""
        model = PilotBehaviorModel()
        
        # Record multiple behaviors with patterns
        for i in range(20):
            action = sample_pilot_action.copy()
            # Create pattern: alternating aggressive/conservative
            action["control_input"] = np.array([0.8, 0.1, 0.0]) if i % 2 == 0 else np.array([0.2, 0.0, 0.0])
            action["timestamp"] = time.time() + i
            model.record_behavior(action)
        
        patterns = model.analyze_patterns()
        
        assert patterns is not None
        assert "control_style" in patterns
        assert "reaction_time" in patterns
    
    def test_predict_action(self, sample_state):
        """Test action prediction based on pilot behavior"""
        model = PilotBehaviorModel()
        
        # Train model with behavior data
        for i in range(50):
            state = sample_state + np.random.normal(0, 0.1, size=sample_state.shape)
            action = {
                "control_input": np.array([0.5, 0.2 * np.sin(i/10), 0.0]),
                "state": state,
                "timestamp": time.time() + i
            }
            model.record_behavior(action)
        
        # Predict action for new state
        predicted_action = model.predict_action(sample_state)
        
        assert predicted_action is not None
        assert predicted_action.shape == (3,)
    
    def test_update_pilot_profile(self):
        """Test pilot profile updates"""
        model = PilotBehaviorModel()
        
        # Initial profile
        initial_profile = model.pilot_profile.copy()
        
        # Record behaviors
        for i in range(10):
            action = {
                "control_input": np.array([0.9, 0.5, 0.0]),  # Aggressive inputs
                "mode_selection": "manual",
                "timestamp": time.time() + i
            }
            model.record_behavior(action)
        
        # Update profile
        model.update_pilot_profile()
        
        assert model.pilot_profile != initial_profile
        assert model.pilot_profile["aggressiveness"] > initial_profile["aggressiveness"]
    
    def test_adaptation_rate(self):
        """Test behavior model adaptation rate"""
        model = PilotBehaviorModel({"adaptation_rate": 0.5})
        
        # Record contrasting behaviors
        for i in range(10):
            action = {
                "control_input": np.array([0.1, 0.0, 0.0]),  # Conservative
                "timestamp": time.time() + i
            }
            model.record_behavior(action)
        
        initial_profile = model.pilot_profile["aggressiveness"]
        
        # Sudden change to aggressive
        for i in range(5):
            action = {
                "control_input": np.array([0.9, 0.8, 0.0]),  # Aggressive
                "timestamp": time.time() + 10 + i
            }
            model.record_behavior(action)
        
        model.update_pilot_profile()
        
        # With high adaptation rate, should change quickly
        assert abs(model.pilot_profile["aggressiveness"] - initial_profile) > 0.2


class TestAdaptiveController:
    """Test AdaptiveController functionality"""
    
    def test_initialization(self):
        """Test adaptive controller initialization"""
        config = {"adaptation_gain": 0.1, "sensitivity": 0.5}
        controller = AdaptiveController(config)
        
        assert controller.config == config
        assert controller.control_parameters is not None
        assert controller.adaptation_enabled is True
    
    def test_compute_control(self, sample_state):
        """Test control computation"""
        controller = AdaptiveController()
        
        target = sample_state + 0.5
        control = controller.compute_control(sample_state, target)
        
        assert control is not None
        assert isinstance(control, np.ndarray)
        assert control.shape[0] >= 3  # At least 3D control
    
    def test_adapt_parameters(self, sample_pilot_action):
        """Test parameter adaptation"""
        controller = AdaptiveController()
        
        # Get initial parameters
        initial_params = controller.control_parameters.copy()
        
        # Create pilot preference data
        pilot_prefs = {
            "smoothness": 0.8,
            "responsiveness": 0.6,
            "automation_preference": 0.4
        }
        
        # Adapt parameters
        controller.adapt_parameters(pilot_prefs)
        
        assert controller.control_parameters != initial_params
        assert controller.control_parameters["smoothing_factor"] > initial_params["smoothing_factor"]
    
    def test_manual_override(self, sample_state, sample_pilot_action):
        """Test manual override handling"""
        controller = AdaptiveController()
        
        # Compute automatic control
        target = sample_state + 0.5
        auto_control = controller.compute_control(sample_state, target)
        
        # Apply manual override
        manual_input = sample_pilot_action["control_input"]
        final_control = controller.apply_manual_override(auto_control, manual_input)
        
        assert final_control is not None
        assert not np.array_equal(final_control, auto_control)
    
    def test_safety_limits(self, sample_state):
        """Test safety limit enforcement"""
        controller = AdaptiveController({
            "max_acceleration": 9.0,  # 9g limit
            "max_angular_rate": 2.0   # rad/s
        })
        
        # Request extreme control
        target = sample_state * 100  # Unrealistic target
        control = controller.compute_control(sample_state, target)
        
        # Check limits are enforced
        assert np.all(np.abs(control[:3]) <= 9.0 * 9.81)  # Acceleration limit
        assert np.all(np.abs(control[3:]) <= 2.0)  # Angular rate limit
    
    def test_mode_switching(self):
        """Test control mode switching"""
        controller = AdaptiveController()
        
        # Test different modes
        modes = ["aggressive", "normal", "conservative", "precision"]
        
        for mode in modes:
            controller.set_control_mode(mode)
            assert controller.current_mode == mode
            assert controller.control_parameters["mode"] == mode
    
    def test_performance_metrics(self, sample_state):
        """Test controller performance tracking"""
        controller = AdaptiveController()
        
        # Perform multiple control computations
        for i in range(10):
            state = sample_state + np.random.normal(0, 0.1, size=sample_state.shape)
            target = state + 0.2
            control = controller.compute_control(state, target)
            
            # Record performance
            error = np.linalg.norm(target - state)
            controller.record_performance(error, control)
        
        metrics = controller.get_performance_metrics()
        
        assert "average_error" in metrics
        assert "control_effort" in metrics
        assert metrics["average_error"] >= 0


class TestPreferenceLearner:
    """Test PreferenceLearner functionality"""
    
    def test_initialization(self):
        """Test preference learner initialization"""
        config = {"learning_rate": 0.01, "preference_dimensions": 5}
        learner = PreferenceLearner(config)
        
        assert learner.config == config
        assert learner.preferences is not None
        assert learner.preference_history is not None
    
    def test_learn_from_feedback(self):
        """Test learning from pilot feedback"""
        learner = PreferenceLearner()
        
        # Provide feedback
        feedback = {
            "action": "increase_automation",
            "satisfaction": 0.8,
            "context": {"workload": "high", "mission_phase": "combat"}
        }
        
        learner.learn_from_feedback(feedback)
        
        assert len(learner.preference_history) == 1
        assert learner.preferences["automation_level"] > 0.5
    
    def test_implicit_learning(self, sample_pilot_action):
        """Test implicit preference learning"""
        learner = PreferenceLearner()
        
        # Simulate pilot consistently choosing manual control
        for i in range(20):
            action = sample_pilot_action.copy()
            action["mode_selection"] = "manual"
            action["automation_used"] = False
            learner.observe_pilot_action(action)
        
        learner.update_preferences()
        
        assert learner.preferences["manual_control_preference"] > 0.7
    
    def test_context_dependent_preferences(self):
        """Test context-dependent preference learning"""
        learner = PreferenceLearner()
        
        # Different preferences in different contexts
        contexts = [
            {"phase": "takeoff", "preference": "high_automation"},
            {"phase": "combat", "preference": "manual_control"},
            {"phase": "cruise", "preference": "moderate_automation"}
        ]
        
        for context in contexts:
            feedback = {
                "context": {"mission_phase": context["phase"]},
                "action": context["preference"],
                "satisfaction": 0.9
            }
            learner.learn_from_feedback(feedback)
        
        # Get preference for specific context
        combat_pref = learner.get_preference_for_context({"mission_phase": "combat"})
        cruise_pref = learner.get_preference_for_context({"mission_phase": "cruise"})
        
        assert combat_pref["automation_level"] < cruise_pref["automation_level"]
    
    def test_preference_decay(self):
        """Test preference decay over time"""
        learner = PreferenceLearner({"decay_rate": 0.1})
        
        # Set initial preference
        learner.preferences["speed_preference"] = 0.9
        
        # Simulate time passing without reinforcement
        learner.apply_preference_decay(time_elapsed=10.0)
        
        assert learner.preferences["speed_preference"] < 0.9
    
    def test_preference_confidence(self):
        """Test preference confidence tracking"""
        learner = PreferenceLearner()
        
        # Low confidence with few observations
        learner.observe_pilot_action({"mode_selection": "auto"})
        confidence_low = learner.get_preference_confidence()
        
        # Higher confidence with more observations
        for i in range(50):
            learner.observe_pilot_action({"mode_selection": "auto"})
        confidence_high = learner.get_preference_confidence()
        
        assert confidence_high > confidence_low
    
    def test_preference_persistence(self):
        """Test preference saving and loading"""
        learner = PreferenceLearner()
        
        # Learn preferences
        for i in range(10):
            feedback = {
                "action": "smooth_control",
                "satisfaction": 0.85,
                "context": {"phase": "normal"}
            }
            learner.learn_from_feedback(feedback)
        
        # Save preferences
        saved_prefs = learner.export_preferences()
        
        # Create new learner and import
        new_learner = PreferenceLearner()
        new_learner.import_preferences(saved_prefs)
        
        assert new_learner.preferences == learner.preferences
    
    def test_multi_pilot_profiles(self):
        """Test handling multiple pilot profiles"""
        learner = PreferenceLearner()
        
        # Learn preferences for multiple pilots
        pilots = ["pilot_1", "pilot_2"]
        
        for pilot_id in pilots:
            learner.set_pilot_id(pilot_id)
            
            # Different preferences per pilot
            if pilot_id == "pilot_1":
                pref_action = "aggressive"
            else:
                pref_action = "conservative"
            
            for i in range(10):
                feedback = {
                    "action": pref_action,
                    "satisfaction": 0.9,
                    "pilot_id": pilot_id
                }
                learner.learn_from_feedback(feedback)
        
        # Check different profiles learned
        pilot1_prefs = learner.get_pilot_preferences("pilot_1")
        pilot2_prefs = learner.get_pilot_preferences("pilot_2")
        
        assert pilot1_prefs["aggressiveness"] > pilot2_prefs["aggressiveness"]
    
    def test_adaptation_boundaries(self):
        """Test preference adaptation within safe boundaries"""
        learner = PreferenceLearner({
            "min_automation": 0.2,
            "max_automation": 0.8
        })
        
        # Try to learn extreme preference
        for i in range(100):
            feedback = {
                "action": "full_manual",
                "satisfaction": 1.0,
                "context": {"phase": "all"}
            }
            learner.learn_from_feedback(feedback)
        
        learner.update_preferences()
        
        # Check boundaries are respected
        assert learner.preferences["automation_level"] >= 0.2
        assert learner.preferences["automation_level"] <= 0.8