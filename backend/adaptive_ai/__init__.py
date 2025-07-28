"""
Adaptive AI System for Iron Man Suit

This package provides a comprehensive AI system that adapts to the pilot's behavior,
environmental conditions, and mission requirements. It includes:

- Reinforcement Learning: DQN, PPO, and custom agents for flight control
- Tactical Decision Making: High-level mission planning and threat assessment
- Behavioral Adaptation: Learning pilot preferences and adapting control systems
- Multi-Agent Coordination: For drone swarms and team operations
- Predictive Analytics: Anticipating threats and optimizing performance
- Cognitive Load Management: Balancing automation with pilot control
"""

from .reinforcement_learning import (
    ReplayBuffer,
    BaseAgent,
    DQNAgent,
    PPOAgent,
    SACAgent,
    MultiAgentCoordinator
)

from .tactical_decision import (
    TacticalDecisionEngine,
    ThreatAssessment,
    MissionPlanner,
    TacticalDecisionError
)

from .behavioral_adaptation import (
    PilotBehaviorModel,
    AdaptiveController,
    PreferenceLearner,
    BehavioralAdaptationError
)

from .predictive_analytics import (
    ThreatPredictor,
    PerformanceOptimizer,
    PredictiveAnalytics,
    PredictiveAnalyticsError
)

from .cognitive_load import (
    CognitiveLoadManager,
    AutomationLevelController,
    WorkloadBalancer,
    CognitiveLoadError
)

from .ai_system import (
    AdaptiveAISystem,
    AISystemConfig,
    AISystemError
)

__all__ = [
    # Reinforcement Learning
    'ReplayBuffer',
    'BaseAgent', 
    'DQNAgent',
    'PPOAgent',
    'SACAgent',
    'MultiAgentCoordinator',
    
    # Tactical Decision
    'TacticalDecisionEngine',
    'ThreatAssessment',
    'MissionPlanner',
    'TacticalDecisionError',
    
    # Behavioral Adaptation
    'PilotBehaviorModel',
    'AdaptiveController',
    'PreferenceLearner',
    'BehavioralAdaptationError',
    
    # Predictive Analytics
    'ThreatPredictor',
    'PerformanceOptimizer',
    'PredictiveAnalytics',
    'PredictiveAnalyticsError',
    
    # Cognitive Load
    'CognitiveLoadManager',
    'AutomationLevelController',
    'WorkloadBalancer',
    'CognitiveLoadError',
    
    # Main System
    'AdaptiveAISystem',
    'AISystemConfig',
    'AISystemError'
]

__version__ = "1.0.0" 