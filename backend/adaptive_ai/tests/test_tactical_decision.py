"""
Unit tests for the Tactical Decision module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

from adaptive_ai.tactical_decision import (
    TacticalDecisionEngine,
    ThreatAssessment,
    MissionPlanner,
    TacticalDecisionError,
    ThreatLevel,
    MissionPhase,
    MissionType,
    ThreatType
)


class TestThreatAssessment:
    """Test ThreatAssessment functionality"""
    
    def test_initialization(self):
        """Test threat assessment initialization"""
        config = {"detection_range": 1000.0, "tracking_limit": 10}
        threat_assessment = ThreatAssessment(config)
        
        assert threat_assessment.config == config
        assert len(threat_assessment.active_threats) == 0
        assert threat_assessment.threat_history is not None
    
    def test_add_threat(self):
        """Test adding new threats"""
        threat_assessment = ThreatAssessment()
        
        threat_data = {
            "id": "threat_1",
            "type": ThreatType.MISSILE,
            "position": np.array([100.0, 200.0, 300.0]),
            "velocity": np.array([50.0, 0.0, 0.0]),
            "detected_at": time.time()
        }
        
        threat_assessment.add_threat("threat_1", threat_data)
        
        assert "threat_1" in threat_assessment.active_threats
        assert threat_assessment.active_threats["threat_1"]["type"] == ThreatType.MISSILE
    
    def test_update_threat(self):
        """Test updating existing threats"""
        threat_assessment = ThreatAssessment()
        
        # Add initial threat
        initial_data = {
            "id": "threat_1",
            "type": ThreatType.AIRCRAFT,
            "position": np.array([100.0, 200.0, 300.0]),
            "velocity": np.array([50.0, 0.0, 0.0]),
            "detected_at": time.time()
        }
        threat_assessment.add_threat("threat_1", initial_data)
        
        # Update threat
        updated_data = {
            "position": np.array([150.0, 200.0, 300.0]),
            "velocity": np.array([60.0, 0.0, 0.0])
        }
        threat_assessment.update_threat("threat_1", updated_data)
        
        assert np.array_equal(
            threat_assessment.active_threats["threat_1"]["position"],
            updated_data["position"]
        )
    
    def test_assess_threat_level(self, sample_sensor_data):
        """Test threat level assessment"""
        threat_assessment = ThreatAssessment()
        
        # Add threat
        threat_data = {
            "id": "threat_1",
            "type": ThreatType.MISSILE,
            "position": np.array([200.0, 300.0, 500.0]),
            "velocity": np.array([-100.0, -100.0, 0.0]),
            "detected_at": time.time()
        }
        threat_assessment.add_threat("threat_1", threat_data)
        
        # Assess threat level
        threat_level = threat_assessment.assess_threat_level(
            "threat_1",
            sample_sensor_data["position"]
        )
        
        assert isinstance(threat_level, ThreatLevel)
        assert threat_level.value >= 0 and threat_level.value <= 4
    
    def test_prioritize_threats(self, sample_sensor_data):
        """Test threat prioritization"""
        threat_assessment = ThreatAssessment()
        
        # Add multiple threats
        threats = [
            {
                "id": "threat_1",
                "type": ThreatType.MISSILE,
                "position": np.array([200.0, 300.0, 500.0]),
                "velocity": np.array([-100.0, -100.0, 0.0]),
                "detected_at": time.time()
            },
            {
                "id": "threat_2",
                "type": ThreatType.AIRCRAFT,
                "position": np.array([1000.0, 1000.0, 1000.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "detected_at": time.time()
            }
        ]
        
        for threat in threats:
            threat_assessment.add_threat(threat["id"], threat)
        
        # Prioritize threats
        prioritized = threat_assessment.prioritize_threats(sample_sensor_data["position"])
        
        assert len(prioritized) == 2
        assert prioritized[0]["id"] == "threat_1"  # Closer threat should be first
    
    def test_remove_threat(self):
        """Test threat removal"""
        threat_assessment = ThreatAssessment()
        
        # Add and remove threat
        threat_assessment.add_threat("threat_1", {"type": ThreatType.MISSILE})
        assert "threat_1" in threat_assessment.active_threats
        
        threat_assessment.remove_threat("threat_1")
        assert "threat_1" not in threat_assessment.active_threats


class TestMissionPlanner:
    """Test MissionPlanner functionality"""
    
    def test_initialization(self):
        """Test mission planner initialization"""
        config = {"planning_horizon": 100, "replan_interval": 10}
        planner = MissionPlanner(config)
        
        assert planner.config == config
        assert planner.current_mission is None
        assert planner.mission_history is not None
    
    def test_create_mission(self):
        """Test mission creation"""
        planner = MissionPlanner()
        
        mission_params = {
            "type": MissionType.COMBAT,
            "objectives": ["neutralize_threats", "protect_area"],
            "constraints": {"max_altitude": 5000, "max_speed": 500},
            "priority": 1
        }
        
        mission = planner.create_mission(mission_params)
        
        assert mission is not None
        assert mission["type"] == MissionType.COMBAT
        assert len(mission["objectives"]) == 2
        assert "id" in mission
    
    def test_update_mission_phase(self):
        """Test mission phase updates"""
        planner = MissionPlanner()
        
        # Create mission
        mission = planner.create_mission({"type": MissionType.PATROL})
        planner.set_current_mission(mission)
        
        # Update phase
        planner.update_mission_phase(MissionPhase.EXECUTION)
        
        assert planner.current_mission["phase"] == MissionPhase.EXECUTION
    
    def test_plan_route(self, sample_sensor_data):
        """Test route planning"""
        planner = MissionPlanner()
        
        start = sample_sensor_data["position"]
        goal = np.array([1000.0, 1000.0, 1000.0])
        
        waypoints = planner.plan_route(start, goal)
        
        assert isinstance(waypoints, list)
        assert len(waypoints) >= 2  # At least start and goal
        assert np.array_equal(waypoints[0], start)
        assert np.array_equal(waypoints[-1], goal)
    
    def test_evaluate_mission_progress(self):
        """Test mission progress evaluation"""
        planner = MissionPlanner()
        
        # Create mission with objectives
        mission = planner.create_mission({
            "type": MissionType.RECONNAISSANCE,
            "objectives": [
                {"id": "obj1", "completed": False},
                {"id": "obj2", "completed": True}
            ]
        })
        planner.set_current_mission(mission)
        
        progress = planner.evaluate_mission_progress()
        
        assert "completion_rate" in progress
        assert progress["completion_rate"] == 0.5  # 1 of 2 completed
    
    def test_abort_mission(self):
        """Test mission abort"""
        planner = MissionPlanner()
        
        # Create and set mission
        mission = planner.create_mission({"type": MissionType.PATROL})
        planner.set_current_mission(mission)
        
        # Abort mission
        planner.abort_mission("Emergency detected")
        
        assert planner.current_mission["phase"] == MissionPhase.ABORTED
        assert planner.current_mission["abort_reason"] == "Emergency detected"


class TestTacticalDecisionEngine:
    """Test TacticalDecisionEngine functionality"""
    
    def test_initialization(self):
        """Test engine initialization"""
        config = {
            "update_frequency": 10.0,
            "decision_threshold": 0.7,
            "planning_horizon": 50
        }
        engine = TacticalDecisionEngine(config)
        
        assert engine.config == config
        assert isinstance(engine.threat_assessment, ThreatAssessment)
        assert isinstance(engine.mission_planner, MissionPlanner)
        assert engine.is_running is False
    
    def test_start_stop(self):
        """Test engine start/stop"""
        engine = TacticalDecisionEngine({"update_frequency": 100.0})
        
        # Start engine
        engine.start()
        assert engine.is_running is True
        assert engine._update_thread is not None
        
        # Stop engine
        engine.stop()
        assert engine.is_running is False
    
    def test_process_sensor_data(self, sample_sensor_data, sample_threat_data):
        """Test sensor data processing"""
        engine = TacticalDecisionEngine()
        
        # Process sensor data
        state = engine.process_sensor_data(sample_sensor_data, sample_threat_data)
        
        assert "suit_position" in state
        assert "active_threats" in state
        assert len(state["active_threats"]) == len(sample_threat_data)
    
    def test_make_tactical_decision(self, sample_sensor_data, sample_threat_data):
        """Test tactical decision making"""
        engine = TacticalDecisionEngine()
        
        # Process data first
        state = engine.process_sensor_data(sample_sensor_data, sample_threat_data)
        
        # Make decision
        decision = engine.make_tactical_decision(state)
        
        assert "action" in decision
        assert "confidence" in decision
        assert "reasoning" in decision
        assert 0 <= decision["confidence"] <= 1
    
    def test_execute_decision(self):
        """Test decision execution"""
        engine = TacticalDecisionEngine()
        
        decision = {
            "action": "evade",
            "parameters": {"direction": np.array([1.0, 0.0, 0.0])},
            "confidence": 0.9
        }
        
        # Execute decision
        result = engine.execute_decision(decision)
        
        assert result is not None
        assert "status" in result
        assert "execution_time" in result
    
    def test_handle_emergency(self):
        """Test emergency handling"""
        engine = TacticalDecisionEngine()
        
        # Create emergency situation
        emergency = {
            "type": "system_failure",
            "severity": "critical",
            "affected_systems": ["propulsion", "navigation"]
        }
        
        response = engine.handle_emergency(emergency)
        
        assert response is not None
        assert "immediate_actions" in response
        assert "contingency_plan" in response
    
    def test_coordination_with_mission(self, sample_sensor_data):
        """Test coordination between threat assessment and mission planning"""
        engine = TacticalDecisionEngine()
        
        # Create mission
        mission = engine.mission_planner.create_mission({
            "type": MissionType.PATROL,
            "objectives": ["patrol_area"]
        })
        engine.mission_planner.set_current_mission(mission)
        
        # Add threat
        threat_data = {
            "id": "threat_1",
            "type": ThreatType.MISSILE,
            "position": np.array([500.0, 500.0, 500.0]),
            "velocity": np.array([-50.0, -50.0, 0.0])
        }
        engine.threat_assessment.add_threat("threat_1", threat_data)
        
        # Make decision considering both mission and threats
        state = engine.process_sensor_data(sample_sensor_data, [threat_data])
        decision = engine.make_tactical_decision(state)
        
        # Should prioritize threat over patrol
        assert decision["action"] in ["evade", "engage", "defend"]
    
    def test_multi_objective_optimization(self):
        """Test multi-objective decision optimization"""
        engine = TacticalDecisionEngine()
        
        objectives = {
            "minimize_risk": 0.8,
            "maximize_efficiency": 0.6,
            "conserve_energy": 0.4
        }
        
        constraints = {
            "max_g_force": 9.0,
            "min_altitude": 100.0,
            "max_speed": 600.0
        }
        
        optimal_decision = engine.optimize_decision(objectives, constraints)
        
        assert optimal_decision is not None
        assert "solution" in optimal_decision
        assert "score" in optimal_decision
    
    def test_decision_history(self):
        """Test decision history tracking"""
        engine = TacticalDecisionEngine()
        
        # Make several decisions
        decisions = []
        for i in range(5):
            decision = {
                "action": f"action_{i}",
                "confidence": 0.8 + i * 0.02,
                "timestamp": time.time()
            }
            engine.record_decision(decision)
            decisions.append(decision)
        
        # Check history
        history = engine.get_decision_history(limit=3)
        assert len(history) == 3
        assert history[0]["action"] == "action_4"  # Most recent first
    
    def test_threat_prediction(self, sample_threat_data):
        """Test threat trajectory prediction"""
        engine = TacticalDecisionEngine()
        
        # Add threat
        threat = sample_threat_data[0]
        engine.threat_assessment.add_threat(threat["id"], threat)
        
        # Predict future position
        future_time = 5.0  # 5 seconds
        predicted_pos = engine.predict_threat_position(threat["id"], future_time)
        
        assert predicted_pos is not None
        assert predicted_pos.shape == (3,)
    
    def test_error_handling(self):
        """Test error handling in tactical decisions"""
        engine = TacticalDecisionEngine()
        
        # Test with invalid state
        with pytest.raises(TacticalDecisionError):
            engine.make_tactical_decision({})  # Empty state
        
        # Test with invalid threat ID
        with pytest.raises(TacticalDecisionError):
            engine.threat_assessment.assess_threat_level(
                "nonexistent_threat",
                np.array([0, 0, 0])
            )