"""
Unit tests for the Predictive Analytics module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime, timedelta

from adaptive_ai.predictive_analytics import (
    ThreatPredictor,
    PerformanceOptimizer,
    PredictiveAnalytics,
    PredictiveAnalyticsError,
    Prediction
)


class TestThreatPredictor:
    """Test ThreatPredictor functionality"""
    
    def test_initialization(self):
        """Test threat predictor initialization"""
        config = {
            "prediction_horizon": 30.0,
            "update_interval": 1.0,
            "model_type": "lstm"
        }
        predictor = ThreatPredictor(config)
        
        assert predictor.config == config
        assert predictor.threat_models is not None
        assert predictor.prediction_cache is not None
    
    def test_predict_trajectory(self, sample_threat_data):
        """Test threat trajectory prediction"""
        predictor = ThreatPredictor()
        
        threat = sample_threat_data[0]
        
        # Predict trajectory
        time_horizon = 10.0  # 10 seconds
        trajectory = predictor.predict_trajectory(
            threat["position"],
            threat["velocity"],
            time_horizon
        )
        
        assert trajectory is not None
        assert len(trajectory) > 0
        assert trajectory[-1]["time"] == pytest.approx(time_horizon, rel=0.1)
    
    def test_predict_threat_evolution(self, sample_threat_data):
        """Test threat behavior evolution prediction"""
        predictor = ThreatPredictor()
        
        # Add historical data
        threat_history = []
        for i in range(10):
            threat = sample_threat_data[0].copy()
            threat["position"] = threat["position"] + i * threat["velocity"] * 0.1
            threat["timestamp"] = time.time() + i * 0.1
            threat_history.append(threat)
        
        # Predict evolution
        evolution = predictor.predict_threat_evolution(
            threat_id="threat_1",
            history=threat_history,
            time_horizon=5.0
        )
        
        assert evolution is not None
        assert "behavior_change_probability" in evolution
        assert "predicted_maneuvers" in evolution
    
    def test_collision_prediction(self, sample_sensor_data, sample_threat_data):
        """Test collision prediction"""
        predictor = ThreatPredictor()
        
        # Set up collision course
        suit_pos = sample_sensor_data["position"]
        suit_vel = sample_sensor_data["velocity"]
        
        threat = sample_threat_data[0].copy()
        # Point threat at suit
        threat["velocity"] = (suit_pos - threat["position"]) / 5.0  # 5 second intercept
        
        collision_risk = predictor.predict_collision(
            suit_position=suit_pos,
            suit_velocity=suit_vel,
            threat_position=threat["position"],
            threat_velocity=threat["velocity"]
        )
        
        assert collision_risk is not None
        assert "collision_probability" in collision_risk
        assert "time_to_impact" in collision_risk
        assert collision_risk["collision_probability"] > 0.8  # High risk
    
    def test_multi_threat_analysis(self, sample_sensor_data, sample_threat_data):
        """Test multiple threat analysis"""
        predictor = ThreatPredictor()
        
        # Analyze multiple threats
        analysis = predictor.analyze_multi_threat_scenario(
            suit_state=sample_sensor_data,
            threats=sample_threat_data,
            time_horizon=10.0
        )
        
        assert analysis is not None
        assert "combined_risk" in analysis
        assert "priority_order" in analysis
        assert len(analysis["priority_order"]) == len(sample_threat_data)
    
    def test_evasion_recommendation(self, sample_sensor_data, sample_threat_data):
        """Test evasion maneuver recommendations"""
        predictor = ThreatPredictor()
        
        # Get evasion recommendations
        recommendations = predictor.recommend_evasion(
            suit_state=sample_sensor_data,
            threat=sample_threat_data[0],
            constraints={"max_g": 9.0, "max_speed": 600.0}
        )
        
        assert recommendations is not None
        assert "maneuvers" in recommendations
        assert "success_probability" in recommendations
        assert len(recommendations["maneuvers"]) > 0
    
    def test_model_update(self, sample_threat_data):
        """Test prediction model updates"""
        predictor = ThreatPredictor()
        
        # Generate training data
        actual_trajectories = []
        for i in range(20):
            threat = sample_threat_data[0].copy()
            threat["position"] = threat["position"] + i * threat["velocity"] * 0.5
            threat["timestamp"] = time.time() + i * 0.5
            actual_trajectories.append(threat)
        
        # Update model
        predictor.update_model("threat_1", actual_trajectories)
        
        # Check model was updated
        assert "threat_1" in predictor.threat_models
        assert predictor.threat_models["threat_1"]["last_update"] is not None
    
    def test_uncertainty_quantification(self, sample_threat_data):
        """Test prediction uncertainty quantification"""
        predictor = ThreatPredictor()
        
        # Make prediction with uncertainty
        prediction = predictor.predict_with_uncertainty(
            threat=sample_threat_data[0],
            time_horizon=10.0,
            num_samples=100
        )
        
        assert prediction is not None
        assert "mean_trajectory" in prediction
        assert "confidence_bounds" in prediction
        assert "uncertainty_evolution" in prediction


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer functionality"""
    
    def test_initialization(self):
        """Test performance optimizer initialization"""
        config = {
            "optimization_interval": 5.0,
            "performance_metrics": ["speed", "efficiency", "safety"],
            "learning_rate": 0.01
        }
        optimizer = PerformanceOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.performance_history is not None
        assert optimizer.optimization_models is not None
    
    def test_optimize_energy_usage(self, sample_sensor_data):
        """Test energy usage optimization"""
        optimizer = PerformanceOptimizer()
        
        # Current state
        current_state = {
            "position": sample_sensor_data["position"],
            "velocity": sample_sensor_data["velocity"],
            "power_consumption": 1500.0,  # Watts
            "battery_level": 0.75
        }
        
        # Optimize energy
        optimization = optimizer.optimize_energy_usage(
            current_state=current_state,
            mission_requirements={"min_speed": 50.0, "duration": 3600.0}
        )
        
        assert optimization is not None
        assert "recommended_power_level" in optimization
        assert "estimated_duration" in optimization
        assert optimization["recommended_power_level"] <= current_state["power_consumption"]
    
    def test_optimize_flight_path(self, sample_sensor_data):
        """Test flight path optimization"""
        optimizer = PerformanceOptimizer()
        
        start = sample_sensor_data["position"]
        goal = np.array([1000.0, 1000.0, 1000.0])
        
        # Optimize path
        optimal_path = optimizer.optimize_flight_path(
            start=start,
            goal=goal,
            constraints={"max_altitude": 5000.0, "no_fly_zones": []}
        )
        
        assert optimal_path is not None
        assert "waypoints" in optimal_path
        assert "total_distance" in optimal_path
        assert "estimated_time" in optimal_path
        assert len(optimal_path["waypoints"]) >= 2
    
    def test_predictive_maintenance(self):
        """Test predictive maintenance analysis"""
        optimizer = PerformanceOptimizer()
        
        # System health data
        system_data = {
            "flight_hours": 150.0,
            "component_status": {
                "thrusters": {"health": 0.85, "cycles": 10000},
                "power_system": {"health": 0.92, "cycles": 8000},
                "control_surfaces": {"health": 0.78, "cycles": 12000}
            },
            "error_logs": ["minor_calibration_drift", "temperature_spike"]
        }
        
        maintenance = optimizer.predict_maintenance_needs(system_data)
        
        assert maintenance is not None
        assert "components_at_risk" in maintenance
        assert "recommended_actions" in maintenance
        assert "time_to_failure_estimates" in maintenance
        assert "control_surfaces" in maintenance["components_at_risk"]
    
    def test_anomaly_detection(self):
        """Test performance anomaly detection"""
        optimizer = PerformanceOptimizer()
        
        # Normal performance data
        for i in range(100):
            normal_data = {
                "speed": 100.0 + np.random.normal(0, 5),
                "power_consumption": 1000.0 + np.random.normal(0, 50),
                "temperature": 25.0 + np.random.normal(0, 2),
                "timestamp": time.time() + i
            }
            optimizer.record_performance(normal_data)
        
        # Anomalous data
        anomaly_data = {
            "speed": 100.0,
            "power_consumption": 2000.0,  # Abnormally high
            "temperature": 45.0,  # Abnormally high
            "timestamp": time.time() + 101
        }
        
        anomalies = optimizer.detect_anomalies(anomaly_data)
        
        assert anomalies is not None
        assert len(anomalies) > 0
        assert any(a["metric"] == "power_consumption" for a in anomalies)
    
    def test_adaptive_optimization(self):
        """Test adaptive optimization based on mission phase"""
        optimizer = PerformanceOptimizer()
        
        # Different optimization for different phases
        phases = [
            {"name": "combat", "priority": "agility"},
            {"name": "cruise", "priority": "efficiency"},
            {"name": "stealth", "priority": "signature_reduction"}
        ]
        
        for phase in phases:
            optimization = optimizer.optimize_for_phase(
                phase_name=phase["name"],
                current_state={"speed": 200.0, "altitude": 3000.0}
            )
            
            assert optimization is not None
            assert optimization["optimization_focus"] == phase["priority"]
    
    def test_multi_objective_optimization(self):
        """Test multi-objective performance optimization"""
        optimizer = PerformanceOptimizer()
        
        objectives = {
            "minimize_time": {"weight": 0.4, "target": 300.0},
            "minimize_energy": {"weight": 0.3, "target": 1000.0},
            "maximize_safety": {"weight": 0.3, "target": 0.95}
        }
        
        current_state = {
            "position": np.array([0, 0, 1000]),
            "velocity": np.array([100, 0, 0]),
            "power_available": 5000.0
        }
        
        solution = optimizer.multi_objective_optimize(objectives, current_state)
        
        assert solution is not None
        assert "pareto_solutions" in solution
        assert "recommended_solution" in solution
        assert len(solution["pareto_solutions"]) > 0


class TestPredictiveAnalytics:
    """Test PredictiveAnalytics main class"""
    
    def test_initialization(self):
        """Test predictive analytics system initialization"""
        config = {
            "prediction_horizon": 60.0,
            "update_frequency": 1.0,
            "confidence_threshold": 0.7
        }
        analytics = PredictiveAnalytics(config)
        
        assert analytics.config == config
        assert isinstance(analytics.threat_predictor, ThreatPredictor)
        assert isinstance(analytics.performance_optimizer, PerformanceOptimizer)
        assert analytics.is_running is False
    
    def test_start_stop(self):
        """Test analytics system start/stop"""
        analytics = PredictiveAnalytics({"update_frequency": 100.0})
        
        # Start system
        analytics.start()
        assert analytics.is_running is True
        
        # Stop system
        analytics.stop()
        assert analytics.is_running is False
    
    def test_comprehensive_analysis(self, sample_sensor_data, sample_threat_data):
        """Test comprehensive predictive analysis"""
        analytics = PredictiveAnalytics()
        
        # Perform analysis
        analysis = analytics.analyze(
            suit_state=sample_sensor_data,
            threats=sample_threat_data,
            mission_context={"phase": "patrol", "duration_remaining": 1800.0}
        )
        
        assert analysis is not None
        assert "threat_predictions" in analysis
        assert "performance_recommendations" in analysis
        assert "risk_assessment" in analysis
    
    def test_real_time_prediction(self, sample_sensor_data):
        """Test real-time prediction updates"""
        analytics = PredictiveAnalytics()
        
        predictions = []
        
        # Simulate real-time updates
        for i in range(5):
            state = sample_sensor_data.copy()
            state["position"] = state["position"] + i * state["velocity"] * 0.1
            state["timestamp"] = time.time() + i * 0.1
            
            prediction = analytics.predict_next_state(state, time_delta=0.1)
            predictions.append(prediction)
        
        assert len(predictions) == 5
        assert all("position" in p for p in predictions)
        assert all("confidence" in p for p in predictions)
    
    def test_scenario_simulation(self):
        """Test what-if scenario simulation"""
        analytics = PredictiveAnalytics()
        
        # Define scenario
        scenario = {
            "initial_state": {
                "position": np.array([0, 0, 1000]),
                "velocity": np.array([200, 0, 0])
            },
            "actions": [
                {"time": 0.0, "action": "accelerate", "parameters": {"thrust": 5000}},
                {"time": 5.0, "action": "turn", "parameters": {"angle": 45}},
                {"time": 10.0, "action": "climb", "parameters": {"rate": 10}}
            ],
            "duration": 20.0
        }
        
        simulation = analytics.simulate_scenario(scenario)
        
        assert simulation is not None
        assert "trajectory" in simulation
        assert "performance_metrics" in simulation
        assert "risk_events" in simulation
    
    def test_learning_from_outcomes(self):
        """Test learning from prediction outcomes"""
        analytics = PredictiveAnalytics()
        
        # Make prediction
        prediction = {
            "threat_position": np.array([500, 500, 500]),
            "confidence": 0.8,
            "timestamp": time.time()
        }
        
        # Actual outcome
        actual = {
            "threat_position": np.array([480, 510, 495]),
            "timestamp": time.time() + 5.0
        }
        
        # Learn from outcome
        analytics.learn_from_outcome(prediction, actual)
        
        # Check model adaptation
        assert analytics.prediction_accuracy is not None
        assert "threat_position" in analytics.prediction_accuracy
    
    def test_export_predictions(self):
        """Test prediction export functionality"""
        analytics = PredictiveAnalytics()
        
        # Generate some predictions
        for i in range(10):
            analytics.store_prediction({
                "type": "threat_trajectory",
                "data": {"position": [i*10, i*20, 1000]},
                "timestamp": time.time() + i
            })
        
        # Export predictions
        exported = analytics.export_predictions(
            start_time=time.time() - 10,
            end_time=time.time() + 20
        )
        
        assert exported is not None
        assert len(exported) == 10
        assert all("timestamp" in p for p in exported)