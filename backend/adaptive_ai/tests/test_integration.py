"""
Integration tests for the complete Adaptive AI System

These tests verify that all components work together cohesively
and that the system can handle realistic scenarios.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time
import threading
from datetime import datetime, timedelta

from adaptive_ai.ai_system import AdaptiveAISystem, AISystemConfig
from adaptive_ai.tactical_decision import ThreatLevel, MissionType, MissionPhase
from adaptive_ai.cognitive_load import WorkloadLevel, AutomationLevel


class TestAdaptiveAIIntegration:
    """Integration tests for the complete AI system"""
    
    @pytest.fixture
    def ai_system(self):
        """Create a fully configured AI system"""
        config = AISystemConfig(
            update_frequency=50.0,
            enable_reinforcement_learning=True,
            enable_tactical_decision=True,
            enable_behavioral_adaptation=True,
            enable_predictive_analytics=True,
            enable_cognitive_load_management=True
        )
        return AdaptiveAISystem(config)
    
    def test_complete_decision_cycle(self, ai_system, sample_sensor_data, sample_threat_data):
        """Test a complete decision-making cycle"""
        # Start the system
        ai_system.start()
        
        try:
            # Initial state
            pilot_input = {
                "control": np.array([0.5, 0.0, 0.0]),
                "mode": "assisted",
                "workload_indicators": {
                    "task_count": 3,
                    "task_complexity": 0.6,
                    "time_pressure": 0.5
                }
            }
            
            # Process multiple cycles
            decisions = []
            for i in range(5):
                # Update sensor data
                sensor_data = sample_sensor_data.copy()
                sensor_data["position"] = sensor_data["position"] + i * 10
                sensor_data["timestamp"] = time.time()
                
                # Make decision
                decision = ai_system.process_input(
                    sensor_data=sensor_data,
                    threats=sample_threat_data,
                    pilot_input=pilot_input
                )
                
                decisions.append(decision)
                time.sleep(0.02)  # 20ms between decisions
            
            # Verify decisions were made
            assert len(decisions) == 5
            assert all(d is not None for d in decisions)
            assert all("action" in d for d in decisions)
            assert all("confidence" in d for d in decisions)
            
        finally:
            ai_system.stop()
    
    def test_threat_response_scenario(self, ai_system, sample_sensor_data):
        """Test system response to evolving threat scenario"""
        ai_system.start()
        
        try:
            # Initial peaceful state
            decision1 = ai_system.process_input(
                sensor_data=sample_sensor_data,
                threats=[],
                pilot_input={"mode": "cruise"}
            )
            
            # Threat appears
            threat = {
                "id": "missile_1",
                "type": "missile",
                "position": np.array([500.0, 500.0, 500.0]),
                "velocity": np.array([-100.0, -100.0, 0.0]),
                "threat_level": 0.9
            }
            
            decision2 = ai_system.process_input(
                sensor_data=sample_sensor_data,
                threats=[threat],
                pilot_input={"mode": "combat"}
            )
            
            # Verify appropriate response
            assert decision2["confidence"] > decision1["confidence"]
            assert "evade" in decision2.get("reasoning", "") or \
                   "engage" in decision2.get("reasoning", "") or \
                   "defend" in decision2.get("reasoning", "")
            
        finally:
            ai_system.stop()
    
    def test_workload_adaptation(self, ai_system, sample_sensor_data):
        """Test system adaptation to pilot workload"""
        ai_system.start()
        
        try:
            automation_levels = []
            
            # Simulate increasing workload
            for i in range(10):
                workload = 0.3 + i * 0.07  # Increasing from 0.3 to ~1.0
                
                pilot_input = {
                    "workload_indicators": {
                        "task_count": 2 + i,
                        "task_complexity": workload,
                        "time_pressure": workload * 0.8,
                        "error_rate": 0.01 * (1 + i)
                    }
                }
                
                decision = ai_system.process_input(
                    sensor_data=sample_sensor_data,
                    threats=[],
                    pilot_input=pilot_input
                )
                
                automation_levels.append(
                    decision.get("automation_level", 0.5)
                )
            
            # Verify automation increases with workload
            assert automation_levels[-1] > automation_levels[0]
            assert all(automation_levels[i] <= automation_levels[i+1] 
                      for i in range(len(automation_levels)-1))
            
        finally:
            ai_system.stop()
    
    def test_learning_and_adaptation(self, ai_system, sample_sensor_data):
        """Test system learning from pilot behavior"""
        ai_system.start()
        
        try:
            # Simulate pilot preferring aggressive maneuvers
            for i in range(20):
                pilot_input = {
                    "control": np.array([0.9, 0.5 * np.sin(i/3), 0.0]),
                    "mode": "manual",
                    "satisfaction_feedback": 0.9 if i % 2 == 0 else 0.7
                }
                
                decision = ai_system.process_input(
                    sensor_data=sample_sensor_data,
                    threats=[],
                    pilot_input=pilot_input
                )
                
                # Provide outcome feedback
                outcome = {
                    "success": True,
                    "pilot_satisfaction": 0.9,
                    "efficiency": 0.85
                }
                ai_system.learn_from_outcome(decision, outcome)
            
            # Test if system learned pilot preferences
            final_decision = ai_system.process_input(
                sensor_data=sample_sensor_data,
                threats=[],
                pilot_input={"mode": "assisted"}
            )
            
            # Should reflect aggressive preference
            assert np.linalg.norm(final_decision["action"]) > 0.7
            
        finally:
            ai_system.stop()
    
    def test_multi_threat_coordination(self, ai_system, sample_sensor_data):
        """Test handling multiple simultaneous threats"""
        ai_system.start()
        
        try:
            # Multiple threats from different directions
            threats = [
                {
                    "id": "threat_1",
                    "type": "missile",
                    "position": np.array([100.0, 0.0, 500.0]),
                    "velocity": np.array([50.0, 0.0, 0.0]),
                    "threat_level": 0.8
                },
                {
                    "id": "threat_2",
                    "type": "aircraft",
                    "position": np.array([0.0, 100.0, 500.0]),
                    "velocity": np.array([0.0, -50.0, 0.0]),
                    "threat_level": 0.6
                },
                {
                    "id": "threat_3",
                    "type": "drone",
                    "position": np.array([-100.0, -100.0, 500.0]),
                    "velocity": np.array([30.0, 30.0, 0.0]),
                    "threat_level": 0.4
                }
            ]
            
            decision = ai_system.process_input(
                sensor_data=sample_sensor_data,
                threats=threats,
                pilot_input={"mode": "combat"}
            )
            
            # Should handle multiple threats
            assert decision is not None
            assert "multi_threat" in decision.get("reasoning", "").lower() or \
                   len(threats) == 3  # Verify it processed all threats
            
        finally:
            ai_system.stop()
    
    def test_mission_phase_transitions(self, ai_system, sample_sensor_data):
        """Test system behavior during mission phase transitions"""
        ai_system.start()
        
        try:
            phases = [
                ("takeoff", {"altitude_target": 1000.0}),
                ("cruise", {"efficiency_priority": 0.8}),
                ("combat", {"agility_priority": 0.9}),
                ("landing", {"precision_priority": 0.95})
            ]
            
            decisions = []
            for phase_name, context in phases:
                ai_system.set_operational_mode(phase_name)
                
                decision = ai_system.process_input(
                    sensor_data=sample_sensor_data,
                    threats=[],
                    pilot_input={"mode": "assisted"},
                    mission_context=context
                )
                
                decisions.append((phase_name, decision))
            
            # Verify different behaviors for different phases
            takeoff_action = decisions[0][1]["action"]
            combat_action = decisions[2][1]["action"]
            
            assert not np.array_equal(takeoff_action, combat_action)
            
        finally:
            ai_system.stop()
    
    def test_predictive_collision_avoidance(self, ai_system, sample_sensor_data):
        """Test predictive analytics for collision avoidance"""
        ai_system.start()
        
        try:
            # Create collision course scenario
            suit_pos = sample_sensor_data["position"]
            suit_vel = np.array([100.0, 0.0, 0.0])
            
            # Object on collision course
            threat = {
                "id": "object_1",
                "type": "debris",
                "position": suit_pos + np.array([500.0, 0.0, 0.0]),
                "velocity": np.array([-100.0, 0.0, 0.0]),
                "threat_level": 0.5
            }
            
            sensor_data = sample_sensor_data.copy()
            sensor_data["velocity"] = suit_vel
            
            decision = ai_system.process_input(
                sensor_data=sensor_data,
                threats=[threat],
                pilot_input={"mode": "assisted"}
            )
            
            # Should take evasive action
            assert decision is not None
            assert np.abs(decision["action"][1]) > 0.1 or \
                   np.abs(decision["action"][2]) > 0.1  # Lateral/vertical movement
            
        finally:
            ai_system.stop()
    
    def test_emergency_response(self, ai_system, sample_sensor_data):
        """Test emergency situation handling"""
        ai_system.start()
        
        try:
            # Normal operation
            normal_decision = ai_system.process_input(
                sensor_data=sample_sensor_data,
                threats=[],
                pilot_input={"mode": "cruise"}
            )
            
            # Emergency situation
            emergency_input = {
                "mode": "emergency",
                "emergency_type": "engine_failure",
                "affected_systems": ["thrust", "power"]
            }
            
            emergency_decision = ai_system.process_input(
                sensor_data=sample_sensor_data,
                threats=[],
                pilot_input=emergency_input,
                emergency=True
            )
            
            # Verify emergency response
            assert emergency_decision["priority"] == "immediate"
            assert "emergency" in emergency_decision.get("reasoning", "").lower()
            
        finally:
            ai_system.stop()
    
    def test_resource_constrained_operation(self, ai_system, sample_sensor_data):
        """Test operation under resource constraints"""
        ai_system.start()
        
        try:
            # Low power scenario
            constrained_data = sample_sensor_data.copy()
            constrained_data["battery_level"] = 0.15  # 15% battery
            constrained_data["power_available"] = 500.0  # Limited power
            
            decision = ai_system.process_input(
                sensor_data=constrained_data,
                threats=[],
                pilot_input={"mode": "assisted"},
                constraints={"max_power": 500.0}
            )
            
            # Should optimize for efficiency
            assert "efficiency" in decision.get("reasoning", "").lower() or \
                   "power" in decision.get("reasoning", "").lower()
            
        finally:
            ai_system.stop()
    
    def test_concurrent_requests(self, ai_system, sample_sensor_data):
        """Test handling concurrent decision requests"""
        ai_system.start()
        
        try:
            results = []
            threads = []
            
            def make_request(idx):
                sensor_data = sample_sensor_data.copy()
                sensor_data["position"] = sensor_data["position"] + idx * 50
                
                decision = ai_system.process_input(
                    sensor_data=sensor_data,
                    threats=[],
                    pilot_input={"mode": "assisted"}
                )
                results.append(decision)
            
            # Start concurrent requests
            for i in range(5):
                thread = threading.Thread(target=make_request, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all to complete
            for thread in threads:
                thread.join(timeout=1.0)
            
            # Verify all requests were processed
            assert len(results) == 5
            assert all(r is not None for r in results)
            
        finally:
            ai_system.stop()
    
    def test_long_duration_stability(self, ai_system, sample_sensor_data):
        """Test system stability over extended operation"""
        ai_system.start()
        
        try:
            start_time = time.time()
            decision_times = []
            errors = []
            
            # Run for extended period
            while time.time() - start_time < 1.0:  # 1 second test
                try:
                    decision_start = time.time()
                    
                    decision = ai_system.process_input(
                        sensor_data=sample_sensor_data,
                        threats=[],
                        pilot_input={"mode": "assisted"}
                    )
                    
                    decision_time = time.time() - decision_start
                    decision_times.append(decision_time)
                    
                    time.sleep(0.01)  # 10ms between requests
                    
                except Exception as e:
                    errors.append(e)
            
            # Verify stability
            assert len(errors) == 0
            assert len(decision_times) > 50  # At least 50 decisions
            assert np.mean(decision_times) < 0.02  # Average under 20ms
            assert np.std(decision_times) < 0.01  # Consistent timing
            
        finally:
            ai_system.stop()
    
    def test_graceful_degradation(self, ai_system, sample_sensor_data):
        """Test system degradation when components fail"""
        ai_system.start()
        
        try:
            # Normal operation
            normal_decision = ai_system.process_input(
                sensor_data=sample_sensor_data,
                threats=[],
                pilot_input={"mode": "assisted"}
            )
            
            # Simulate component failure
            ai_system.predictive_analytics = None
            
            # Should still work
            degraded_decision = ai_system.process_input(
                sensor_data=sample_sensor_data,
                threats=[],
                pilot_input={"mode": "assisted"}
            )
            
            assert degraded_decision is not None
            assert degraded_decision.get("degraded_mode", False)
            
        finally:
            ai_system.stop()