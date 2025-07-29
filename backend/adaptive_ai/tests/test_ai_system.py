"""
Unit tests for the main AI System module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
import threading

from adaptive_ai.ai_system import (
    AdaptiveAISystem,
    AISystemConfig,
    AISystemError
)


class TestAISystemConfig:
    """Test AISystemConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = AISystemConfig()
        
        assert config.update_frequency == 100.0
        assert config.enable_reinforcement_learning is True
        assert config.enable_tactical_decision is True
        assert config.enable_behavioral_adaptation is True
        assert config.enable_predictive_analytics is True
        assert config.enable_cognitive_load_management is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = AISystemConfig(
            update_frequency=50.0,
            max_processing_time=0.02,
            enable_predictive_analytics=False
        )
        
        assert config.update_frequency == 50.0
        assert config.max_processing_time == 0.02
        assert config.enable_predictive_analytics is False


class TestAdaptiveAISystem:
    """Test AdaptiveAISystem functionality"""
    
    def test_initialization(self):
        """Test AI system initialization"""
        config = AISystemConfig(update_frequency=50.0)
        system = AdaptiveAISystem(config)
        
        assert system.config == config
        assert system.is_running is False
        assert system.rl_agent is not None
        assert system.tactical_engine is not None
        assert system.behavior_model is not None
        assert system.predictive_analytics is not None
        assert system.cognitive_manager is not None
    
    def test_start_stop(self):
        """Test system start/stop"""
        system = AdaptiveAISystem(AISystemConfig(update_frequency=100.0))
        
        # Start system
        system.start()
        assert system.is_running is True
        assert system._update_thread is not None
        assert system._update_thread.is_alive()
        
        # Stop system
        system.stop()
        assert system.is_running is False
        
        # Give thread time to stop
        time.sleep(0.1)
        assert not system._update_thread.is_alive()
    
    def test_process_input(self, sample_sensor_data, sample_threat_data):
        """Test input processing"""
        system = AdaptiveAISystem()
        
        # Process input
        result = system.process_input(
            sensor_data=sample_sensor_data,
            threats=sample_threat_data,
            pilot_input={"control": [0.5, 0.0, 0.0], "mode": "assisted"}
        )
        
        assert result is not None
        assert "action" in result
        assert "confidence" in result
        assert "reasoning" in result
    
    def test_component_integration(self, sample_sensor_data, sample_threat_data):
        """Test integration between components"""
        system = AdaptiveAISystem()
        
        # Mock components to track interactions
        system.rl_agent.act = Mock(return_value=np.array([0.5, 0.0, 0.0]))
        system.tactical_engine.make_tactical_decision = Mock(
            return_value={"action": "evade", "confidence": 0.8}
        )
        system.behavior_model.predict_action = Mock(
            return_value=np.array([0.4, 0.1, 0.0])
        )
        
        # Process input
        system.process_input(sample_sensor_data, sample_threat_data)
        
        # Verify components were called
        assert system.rl_agent.act.called
        assert system.tactical_engine.make_tactical_decision.called
        assert system.behavior_model.predict_action.called
    
    def test_decision_fusion(self):
        """Test decision fusion from multiple components"""
        system = AdaptiveAISystem()
        
        # Different recommendations from components
        rl_action = np.array([1.0, 0.0, 0.0])
        tactical_decision = {
            "action": "evade",
            "parameters": {"direction": np.array([0.0, 1.0, 0.0])},
            "confidence": 0.9
        }
        behavioral_preference = np.array([0.5, 0.5, 0.0])
        
        # Fuse decisions
        fused = system.fuse_decisions(
            rl_action=rl_action,
            tactical_decision=tactical_decision,
            behavioral_preference=behavioral_preference,
            workload_level=0.7
        )
        
        assert fused is not None
        assert "action" in fused
        assert "source_weights" in fused
        assert isinstance(fused["action"], np.ndarray)
    
    def test_adaptive_weights(self):
        """Test adaptive component weight adjustment"""
        system = AdaptiveAISystem()
        
        # Track performance over time
        for i in range(20):
            performance = {
                "rl_performance": 0.8 - i * 0.01,  # Declining
                "tactical_performance": 0.7 + i * 0.01,  # Improving
                "overall_success": 0.75
            }
            system.update_component_weights(performance)
        
        # Weights should adapt
        assert system.component_weights["tactical"] > system.component_weights["reinforcement"]
    
    def test_emergency_handling(self):
        """Test emergency situation handling"""
        system = AdaptiveAISystem()
        
        # Create emergency
        emergency = {
            "type": "system_failure",
            "severity": "critical",
            "affected_systems": ["propulsion"]
        }
        
        response = system.handle_emergency(emergency)
        
        assert response is not None
        assert response["priority"] == "immediate"
        assert "actions" in response
        assert len(response["actions"]) > 0
    
    def test_learning_cycle(self, sample_sensor_data):
        """Test continuous learning cycle"""
        system = AdaptiveAISystem()
        
        # Simulate multiple cycles
        for i in range(10):
            # Make decision
            state = sample_sensor_data.copy()
            state["position"] = state["position"] + i * 10
            
            decision = system.process_input(state, [])
            
            # Simulate outcome
            outcome = {
                "success": True,
                "error": np.random.random() * 0.1,
                "efficiency": 0.8 + np.random.random() * 0.2
            }
            
            # Learn from outcome
            system.learn_from_outcome(decision, outcome)
        
        # Check learning occurred
        assert len(system.performance_history) > 0
        assert system.total_decisions > 0
    
    def test_mode_switching(self):
        """Test operational mode switching"""
        system = AdaptiveAISystem()
        
        modes = ["combat", "cruise", "stealth", "emergency"]
        
        for mode in modes:
            system.set_operational_mode(mode)
            assert system.current_mode == mode
            
            # Check mode affects behavior
            config = system.get_mode_configuration(mode)
            assert config is not None
            assert "priority" in config
    
    def test_multi_agent_coordination(self):
        """Test coordination with other AI agents"""
        system = AdaptiveAISystem()
        
        # Simulate other agents
        other_agents = [
            {"id": "drone_1", "position": [100, 200, 300], "status": "active"},
            {"id": "drone_2", "position": [200, 300, 400], "status": "active"}
        ]
        
        # Coordinate action
        coordinated = system.coordinate_with_agents(
            own_action=np.array([1.0, 0.0, 0.0]),
            other_agents=other_agents
        )
        
        assert coordinated is not None
        assert "adjusted_action" in coordinated
        assert "coordination_plan" in coordinated
    
    def test_performance_monitoring(self):
        """Test system performance monitoring"""
        system = AdaptiveAISystem()
        
        # Run for a period
        system.start()
        time.sleep(0.2)
        system.stop()
        
        # Get performance metrics
        metrics = system.get_performance_metrics()
        
        assert metrics is not None
        assert "average_decision_time" in metrics
        assert "component_utilization" in metrics
        assert "success_rate" in metrics
    
    def test_state_persistence(self):
        """Test saving and loading system state"""
        system = AdaptiveAISystem()
        
        # Make some decisions to build state
        for i in range(5):
            system.process_input(
                {"position": [i*10, 0, 1000]},
                []
            )
        
        # Save state
        state = system.export_state()
        
        # Create new system and load state
        new_system = AdaptiveAISystem()
        new_system.import_state(state)
        
        assert new_system.total_decisions == system.total_decisions
        assert len(new_system.performance_history) == len(system.performance_history)
    
    def test_resource_management(self):
        """Test computational resource management"""
        system = AdaptiveAISystem(
            AISystemConfig(max_processing_time=0.01)  # 10ms limit
        )
        
        # Process with time constraint
        start_time = time.time()
        result = system.process_input(
            {"position": [0, 0, 1000]},
            [],
            time_budget=0.01
        )
        end_time = time.time()
        
        assert result is not None
        assert (end_time - start_time) < 0.02  # Some margin
        assert result.get("degraded_mode", False) if (end_time - start_time) > 0.01 else True
    
    def test_component_failure_handling(self):
        """Test handling of component failures"""
        system = AdaptiveAISystem()
        
        # Simulate component failure
        system.tactical_engine = None  # Simulate failure
        
        # Should still work with degraded performance
        try:
            result = system.process_input(
                {"position": [0, 0, 1000]},
                []
            )
            assert result is not None
            assert result.get("degraded_mode", False)
        except AISystemError:
            # Should handle gracefully
            pass
    
    def test_parallel_processing(self):
        """Test parallel processing capabilities"""
        system = AdaptiveAISystem()
        
        # Submit multiple requests
        results = []
        threads = []
        
        def process_request(idx):
            result = system.process_input(
                {"position": [idx*10, 0, 1000]},
                []
            )
            results.append(result)
        
        # Start parallel requests
        for i in range(5):
            thread = threading.Thread(target=process_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        assert all(r is not None for r in results)
    
    def test_diagnostic_interface(self):
        """Test diagnostic and debugging interface"""
        system = AdaptiveAISystem()
        
        # Get diagnostics
        diagnostics = system.get_diagnostics()
        
        assert diagnostics is not None
        assert "component_status" in diagnostics
        assert "last_errors" in diagnostics
        assert "performance_stats" in diagnostics
        
        # Check component status
        for component in ["rl", "tactical", "behavioral", "predictive", "cognitive"]:
            assert component in diagnostics["component_status"]