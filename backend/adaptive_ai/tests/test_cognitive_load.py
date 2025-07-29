"""
Unit tests for the Cognitive Load Management module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime, timedelta

from adaptive_ai.cognitive_load import (
    CognitiveLoadManager,
    AutomationLevelController,
    WorkloadBalancer,
    CognitiveLoadError,
    WorkloadLevel,
    AutomationLevel,
    WorkloadMetrics
)


class TestCognitiveLoadManager:
    """Test CognitiveLoadManager functionality"""
    
    def test_initialization(self):
        """Test cognitive load manager initialization"""
        config = {
            "assessment_interval": 1.0,
            "workload_threshold": 0.8,
            "adaptation_rate": 0.1
        }
        manager = CognitiveLoadManager(config)
        
        assert manager.config == config
        assert manager.current_workload is not None
        assert manager.workload_history is not None
        assert manager.is_monitoring is False
    
    def test_assess_workload(self):
        """Test workload assessment"""
        manager = CognitiveLoadManager()
        
        # Create workload indicators
        indicators = {
            "task_count": 5,
            "task_complexity": 0.7,
            "time_pressure": 0.8,
            "error_rate": 0.1,
            "response_time": 1.2,
            "physiological": {
                "heart_rate": 90,
                "eye_movement_rate": 15,
                "pupil_dilation": 0.6
            }
        }
        
        workload = manager.assess_workload(indicators)
        
        assert isinstance(workload, WorkloadMetrics)
        assert 0 <= workload.overall_load <= 1
        assert workload.level in [level for level in WorkloadLevel]
    
    def test_workload_classification(self):
        """Test workload level classification"""
        manager = CognitiveLoadManager()
        
        # Test different workload levels
        test_cases = [
            (0.2, WorkloadLevel.LOW),
            (0.5, WorkloadLevel.MEDIUM),
            (0.75, WorkloadLevel.HIGH),
            (0.9, WorkloadLevel.CRITICAL)
        ]
        
        for load_value, expected_level in test_cases:
            level = manager.classify_workload_level(load_value)
            assert level == expected_level
    
    def test_physiological_monitoring(self):
        """Test physiological indicator monitoring"""
        manager = CognitiveLoadManager()
        
        # Simulate physiological data
        physio_data = {
            "heart_rate": 85,
            "heart_rate_variability": 50,
            "breathing_rate": 16,
            "skin_conductance": 2.5,
            "eye_blink_rate": 20,
            "pupil_diameter": 4.5
        }
        
        stress_level = manager.assess_physiological_stress(physio_data)
        
        assert 0 <= stress_level <= 1
        assert manager.stress_indicators is not None
    
    def test_cognitive_state_prediction(self):
        """Test cognitive state prediction"""
        manager = CognitiveLoadManager()
        
        # Build history
        for i in range(20):
            indicators = {
                "task_count": 3 + i // 5,
                "task_complexity": 0.5 + i * 0.02,
                "time_pressure": 0.6 + i * 0.01,
                "error_rate": 0.05 + i * 0.005
            }
            workload = manager.assess_workload(indicators)
            manager.record_workload(workload)
        
        # Predict future state
        prediction = manager.predict_cognitive_state(time_horizon=5.0)
        
        assert prediction is not None
        assert "predicted_workload" in prediction
        assert "confidence" in prediction
        assert "risk_of_overload" in prediction
    
    def test_adaptive_thresholds(self):
        """Test adaptive workload thresholds"""
        manager = CognitiveLoadManager({"adaptive_thresholds": True})
        
        # Simulate pilot handling high workload well
        for i in range(30):
            workload = WorkloadMetrics(
                overall_load=0.85,
                level=WorkloadLevel.HIGH,
                components={"mental": 0.9, "physical": 0.8},
                timestamp=time.time() + i
            )
            performance = {"error_rate": 0.05, "task_completion": 0.95}
            
            manager.record_workload_with_performance(workload, performance)
        
        # Thresholds should adapt
        new_threshold = manager.get_adaptive_threshold()
        assert new_threshold > 0.8  # Should increase if pilot handles high load well
    
    def test_workload_trends(self):
        """Test workload trend analysis"""
        manager = CognitiveLoadManager()
        
        # Create trending workload
        base_load = 0.4
        for i in range(50):
            load = base_load + i * 0.01  # Increasing trend
            workload = WorkloadMetrics(
                overall_load=load,
                level=manager.classify_workload_level(load),
                components={"mental": load * 1.1, "physical": load * 0.9},
                timestamp=time.time() + i
            )
            manager.record_workload(workload)
        
        trends = manager.analyze_workload_trends()
        
        assert trends is not None
        assert trends["trend_direction"] == "increasing"
        assert trends["rate_of_change"] > 0
        assert "time_to_threshold" in trends


class TestAutomationLevelController:
    """Test AutomationLevelController functionality"""
    
    def test_initialization(self):
        """Test automation controller initialization"""
        config = {
            "min_automation": 0.2,
            "max_automation": 0.9,
            "transition_time": 2.0
        }
        controller = AutomationLevelController(config)
        
        assert controller.config == config
        assert controller.current_level is not None
        assert controller.target_level is not None
        assert controller.transition_active is False
    
    def test_determine_automation_level(self):
        """Test automation level determination"""
        controller = AutomationLevelController()
        
        # Test different workload levels
        test_cases = [
            (WorkloadLevel.LOW, AutomationLevel.MANUAL),
            (WorkloadLevel.MEDIUM, AutomationLevel.ASSISTED),
            (WorkloadLevel.HIGH, AutomationLevel.SEMI_AUTONOMOUS),
            (WorkloadLevel.CRITICAL, AutomationLevel.AUTONOMOUS)
        ]
        
        for workload_level, expected_auto in test_cases:
            workload = WorkloadMetrics(
                overall_load=0.5,
                level=workload_level,
                components={},
                timestamp=time.time()
            )
            
            auto_level = controller.determine_automation_level(workload)
            assert auto_level == expected_auto
    
    def test_smooth_transitions(self):
        """Test smooth automation level transitions"""
        controller = AutomationLevelController({"transition_time": 1.0})
        
        # Set initial level
        controller.set_automation_level(AutomationLevel.MANUAL)
        
        # Request transition
        controller.transition_to_level(AutomationLevel.AUTONOMOUS)
        assert controller.transition_active is True
        
        # Check intermediate values during transition
        intermediate_values = []
        for i in range(10):
            time.sleep(0.1)
            current = controller.get_current_automation_value()
            intermediate_values.append(current)
        
        # Should see gradual increase
        assert all(intermediate_values[i] <= intermediate_values[i+1] 
                  for i in range(len(intermediate_values)-1))
    
    def test_emergency_override(self):
        """Test emergency automation override"""
        controller = AutomationLevelController()
        
        # Set normal level
        controller.set_automation_level(AutomationLevel.ASSISTED)
        
        # Emergency override
        controller.emergency_override(AutomationLevel.AUTONOMOUS)
        
        assert controller.current_level == AutomationLevel.AUTONOMOUS
        assert controller.override_active is True
    
    def test_pilot_preference_integration(self):
        """Test pilot preference integration"""
        controller = AutomationLevelController()
        
        pilot_preferences = {
            "automation_preference": 0.3,  # Prefers manual control
            "trust_in_automation": 0.7,
            "skill_level": 0.9
        }
        
        # High workload but pilot prefers manual
        workload = WorkloadMetrics(
            overall_load=0.8,
            level=WorkloadLevel.HIGH,
            components={},
            timestamp=time.time()
        )
        
        auto_level = controller.determine_automation_level(
            workload,
            pilot_preferences=pilot_preferences
        )
        
        # Should respect preference to some degree
        assert auto_level != AutomationLevel.AUTONOMOUS
    
    def test_context_aware_automation(self):
        """Test context-aware automation decisions"""
        controller = AutomationLevelController()
        
        contexts = [
            {
                "mission_phase": "combat",
                "expected": AutomationLevel.ASSISTED  # Pilot control important
            },
            {
                "mission_phase": "cruise",
                "expected": AutomationLevel.SEMI_AUTONOMOUS  # Can be more automated
            },
            {
                "mission_phase": "landing",
                "expected": AutomationLevel.ASSISTED  # Critical phase
            }
        ]
        
        for context in contexts:
            level = controller.get_context_appropriate_level(
                context["mission_phase"],
                WorkloadLevel.MEDIUM
            )
            assert level == context["expected"]
    
    def test_automation_boundaries(self):
        """Test automation level boundaries"""
        controller = AutomationLevelController({
            "min_automation": 0.3,
            "max_automation": 0.8
        })
        
        # Try to set outside boundaries
        controller.set_automation_value(0.1)
        assert controller.get_current_automation_value() >= 0.3
        
        controller.set_automation_value(0.95)
        assert controller.get_current_automation_value() <= 0.8


class TestWorkloadBalancer:
    """Test WorkloadBalancer functionality"""
    
    def test_initialization(self):
        """Test workload balancer initialization"""
        config = {
            "max_concurrent_tasks": 10,
            "priority_levels": 5,
            "rebalance_interval": 2.0
        }
        balancer = WorkloadBalancer(config)
        
        assert balancer.config == config
        assert balancer.active_tasks is not None
        assert balancer.task_queue is not None
    
    def test_task_allocation(self):
        """Test task allocation"""
        balancer = WorkloadBalancer()
        
        # Add tasks
        tasks = [
            {"id": "task1", "priority": 1, "complexity": 0.3, "type": "navigation"},
            {"id": "task2", "priority": 2, "complexity": 0.5, "type": "combat"},
            {"id": "task3", "priority": 1, "complexity": 0.7, "type": "analysis"},
            {"id": "task4", "priority": 3, "complexity": 0.2, "type": "monitoring"}
        ]
        
        for task in tasks:
            balancer.add_task(task)
        
        # Allocate with workload constraint
        current_workload = 0.6
        allocated = balancer.allocate_tasks(current_workload)
        
        assert allocated is not None
        assert len(allocated) <= len(tasks)
        assert all(t["id"] in [task["id"] for task in tasks] for t in allocated)
    
    def test_priority_handling(self):
        """Test priority-based task handling"""
        balancer = WorkloadBalancer()
        
        # Add tasks with different priorities
        balancer.add_task({"id": "low", "priority": 3, "complexity": 0.5})
        balancer.add_task({"id": "high", "priority": 1, "complexity": 0.5})
        balancer.add_task({"id": "medium", "priority": 2, "complexity": 0.5})
        
        # Get next task
        next_task = balancer.get_next_task()
        
        assert next_task["id"] == "high"  # Highest priority (1)
    
    def test_task_offloading(self):
        """Test task offloading to automation"""
        balancer = WorkloadBalancer()
        
        # Add tasks
        tasks = [
            {"id": "task1", "priority": 2, "complexity": 0.6, "automatable": True},
            {"id": "task2", "priority": 1, "complexity": 0.8, "automatable": False},
            {"id": "task3", "priority": 3, "complexity": 0.4, "automatable": True}
        ]
        
        for task in tasks:
            balancer.add_task(task)
        
        # Offload with high workload
        offloaded = balancer.offload_tasks(
            current_workload=0.9,
            automation_level=AutomationLevel.SEMI_AUTONOMOUS
        )
        
        assert len(offloaded) > 0
        assert all(task["automatable"] for task in offloaded)
    
    def test_dynamic_rebalancing(self):
        """Test dynamic task rebalancing"""
        balancer = WorkloadBalancer()
        
        # Initial task allocation
        initial_tasks = [
            {"id": f"task{i}", "priority": i % 3 + 1, "complexity": 0.3 + i * 0.1}
            for i in range(5)
        ]
        
        for task in initial_tasks:
            balancer.add_task(task)
            balancer.assign_task(task["id"])
        
        # Change in workload
        new_workload = WorkloadMetrics(
            overall_load=0.85,
            level=WorkloadLevel.HIGH,
            components={},
            timestamp=time.time()
        )
        
        rebalanced = balancer.rebalance_tasks(new_workload)
        
        assert rebalanced is not None
        assert "tasks_deferred" in rebalanced
        assert "tasks_automated" in rebalanced
    
    def test_task_interruption_handling(self):
        """Test handling of task interruptions"""
        balancer = WorkloadBalancer()
        
        # Add and start task
        task = {"id": "task1", "priority": 2, "complexity": 0.5, "interruptible": True}
        balancer.add_task(task)
        balancer.assign_task("task1")
        
        # Interrupt with higher priority task
        urgent_task = {"id": "urgent", "priority": 1, "complexity": 0.3}
        
        decision = balancer.handle_interruption(urgent_task)
        
        assert decision is not None
        assert decision["action"] in ["interrupt", "queue", "defer"]
        assert decision["action"] == "interrupt"  # Should interrupt for higher priority
    
    def test_cognitive_resource_tracking(self):
        """Test cognitive resource usage tracking"""
        balancer = WorkloadBalancer()
        
        # Define tasks with resource requirements
        tasks = [
            {
                "id": "visual",
                "resources": {"visual": 0.8, "auditory": 0.1, "motor": 0.2}
            },
            {
                "id": "auditory",
                "resources": {"visual": 0.1, "auditory": 0.9, "motor": 0.1}
            },
            {
                "id": "motor",
                "resources": {"visual": 0.2, "auditory": 0.1, "motor": 0.7}
            }
        ]
        
        # Check resource conflicts
        for task in tasks:
            balancer.add_task(task)
        
        resource_usage = balancer.calculate_resource_usage()
        
        assert "visual" in resource_usage
        assert "auditory" in resource_usage
        assert "motor" in resource_usage
        assert all(0 <= usage <= 1 for usage in resource_usage.values())
    
    def test_performance_impact_prediction(self):
        """Test performance impact prediction"""
        balancer = WorkloadBalancer()
        
        # Current state
        current_tasks = [
            {"id": "task1", "complexity": 0.4, "priority": 2},
            {"id": "task2", "complexity": 0.3, "priority": 1}
        ]
        
        for task in current_tasks:
            balancer.assign_task(task["id"])
        
        # Predict impact of new task
        new_task = {"id": "new", "complexity": 0.5, "priority": 1}
        
        impact = balancer.predict_performance_impact(new_task)
        
        assert impact is not None
        assert "workload_increase" in impact
        assert "performance_degradation" in impact
        assert "recommendation" in impact