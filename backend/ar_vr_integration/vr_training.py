"""
VR Training Simulation Environment for Iron Man suit.

This module provides immersive VR training scenarios, combat simulations,
flight training, and emergency response drills in a safe virtual environment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import deque
import json


class ScenarioType(Enum):
    """Types of training scenarios"""
    FLIGHT_BASIC = "flight_basic"
    FLIGHT_ADVANCED = "flight_advanced"
    COMBAT_BASIC = "combat_basic"
    COMBAT_ADVANCED = "combat_advanced"
    RESCUE_OPERATIONS = "rescue_operations"
    EQUIPMENT_FAMILIARIZATION = "equipment_familiarization"
    EMERGENCY_PROTOCOLS = "emergency_protocols"
    TEAM_COORDINATION = "team_coordination"
    STEALTH_OPERATIONS = "stealth_operations"
    CUSTOM = "custom"


class DifficultyLevel(Enum):
    """Training difficulty levels"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


class ObjectiveType(Enum):
    """Types of training objectives"""
    REACH_WAYPOINT = "reach_waypoint"
    DEFEAT_ENEMIES = "defeat_enemies"
    PROTECT_CIVILIANS = "protect_civilians"
    COMPLETE_MANEUVER = "complete_maneuver"
    AVOID_DETECTION = "avoid_detection"
    TIME_TRIAL = "time_trial"
    ACCURACY_TEST = "accuracy_test"
    ENDURANCE_TEST = "endurance_test"


@dataclass
class TrainingObjective:
    """Individual training objective"""
    objective_type: ObjectiveType
    description: str
    target_value: Any
    current_value: Any = 0
    completed: bool = False
    optional: bool = False
    time_limit: Optional[float] = None
    points: int = 100
    
    def check_completion(self) -> bool:
        """Check if objective is completed"""
        if self.objective_type == ObjectiveType.REACH_WAYPOINT:
            return self.current_value  # Boolean flag
        elif self.objective_type == ObjectiveType.DEFEAT_ENEMIES:
            return self.current_value >= self.target_value
        elif self.objective_type == ObjectiveType.COMPLETE_MANEUVER:
            return self.current_value >= self.target_value
        return False


@dataclass
class SimulatedEntity:
    """Entity in the training simulation"""
    id: str
    entity_type: str  # enemy, civilian, object, etc.
    position: np.ndarray
    velocity: np.ndarray
    health: float = 100.0
    active: bool = True
    ai_behavior: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Performance metrics for training session"""
    accuracy: float = 0.0
    reaction_time: float = 0.0
    damage_taken: float = 0.0
    damage_dealt: float = 0.0
    objectives_completed: int = 0
    objectives_total: int = 0
    flight_precision: float = 0.0
    energy_efficiency: float = 0.0
    tactical_score: float = 0.0
    completion_time: float = 0.0
    
    def calculate_overall_score(self) -> float:
        """Calculate overall performance score"""
        weights = {
            'accuracy': 0.2,
            'objectives': 0.3,
            'efficiency': 0.2,
            'tactical': 0.2,
            'speed': 0.1
        }
        
        obj_ratio = self.objectives_completed / max(self.objectives_total, 1)
        time_score = max(0, 1 - self.completion_time / 3600)  # 1 hour baseline
        
        score = (
            weights['accuracy'] * self.accuracy +
            weights['objectives'] * obj_ratio +
            weights['efficiency'] * self.energy_efficiency +
            weights['tactical'] * self.tactical_score +
            weights['speed'] * time_score
        )
        
        return min(100, score * 100)


class VRTrainingEnvironment:
    """
    Main VR training environment system.
    
    Manages training scenarios, objectives, performance tracking,
    and immersive simulation experiences.
    """
    
    def __init__(self):
        # Scenario management
        self.current_scenario: Optional[Dict[str, Any]] = None
        self.scenario_state = "idle"  # idle, loading, running, paused, completed
        self.difficulty = DifficultyLevel.BEGINNER
        
        # Training objectives
        self.objectives: List[TrainingObjective] = []
        self.completed_objectives: List[TrainingObjective] = []
        
        # Simulation entities
        self.entities: Dict[str, SimulatedEntity] = {}
        self.player_state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'orientation': np.array([0, 0, 0, 1]),  # Quaternion
            'health': 100.0,
            'energy': 100.0,
            'weapons_enabled': True,
            'flight_mode': 'hover'
        }
        
        # Performance tracking
        self.metrics = TrainingMetrics()
        self.session_start_time: Optional[float] = None
        self.event_log: deque = deque(maxlen=1000)
        
        # Environment settings
        self.environment = {
            'time_of_day': 12.0,  # 24-hour format
            'weather': 'clear',
            'visibility': 1.0,
            'wind_speed': np.zeros(3),
            'gravity': np.array([0, -9.81, 0]),
            'bounds': np.array([1000, 500, 1000])  # Training area bounds
        }
        
        # Simulation parameters
        self.config = {
            'physics_timestep': 1/60,
            'ai_update_rate': 10,  # Hz
            'render_distance': 500,  # meters
            'collision_enabled': True,
            'damage_enabled': True,
            'respawn_enabled': True,
            'telemetry_enabled': True,
            'adaptive_difficulty': True
        }
        
        # Scenario templates
        self.scenario_templates = self._load_scenario_templates()
        
        # Processing threads
        self.is_running = False
        self.simulation_thread = None
        self.ai_thread = None
        
        # Callbacks
        self.callbacks = {
            'on_objective_complete': None,
            'on_scenario_complete': None,
            'on_player_damaged': None,
            'on_achievement_unlocked': None,
            'on_checkpoint_reached': None
        }
        
        # Training history
        self.training_history: List[Dict[str, Any]] = []
        self.best_scores: Dict[str, float] = {}
    
    def _load_scenario_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined training scenarios"""
        templates = {
            ScenarioType.FLIGHT_BASIC: {
                'name': 'Basic Flight Training',
                'description': 'Learn fundamental flight controls and maneuvers',
                'duration': 600,  # 10 minutes
                'objectives': [
                    {
                        'type': ObjectiveType.REACH_WAYPOINT,
                        'description': 'Reach the first waypoint',
                        'target': {'position': [0, 100, 100]},
                        'points': 100
                    },
                    {
                        'type': ObjectiveType.COMPLETE_MANEUVER,
                        'description': 'Complete a barrel roll',
                        'target': 1,
                        'points': 150
                    },
                    {
                        'type': ObjectiveType.TIME_TRIAL,
                        'description': 'Complete the course in under 5 minutes',
                        'target': 300,
                        'points': 200
                    }
                ],
                'environment': {
                    'weather': 'clear',
                    'time_of_day': 14.0,
                    'wind_speed': [5, 0, 0]
                }
            },
            
            ScenarioType.COMBAT_BASIC: {
                'name': 'Basic Combat Training',
                'description': 'Master repulsor targeting and defensive maneuvers',
                'duration': 900,  # 15 minutes
                'objectives': [
                    {
                        'type': ObjectiveType.DEFEAT_ENEMIES,
                        'description': 'Defeat 10 training drones',
                        'target': 10,
                        'points': 200
                    },
                    {
                        'type': ObjectiveType.ACCURACY_TEST,
                        'description': 'Maintain 70% accuracy',
                        'target': 0.7,
                        'points': 150
                    }
                ],
                'spawn_waves': [
                    {'time': 0, 'enemies': 3, 'type': 'drone_basic'},
                    {'time': 60, 'enemies': 5, 'type': 'drone_basic'},
                    {'time': 120, 'enemies': 7, 'type': 'drone_advanced'}
                ]
            },
            
            ScenarioType.RESCUE_OPERATIONS: {
                'name': 'Search and Rescue',
                'description': 'Locate and evacuate civilians from danger zones',
                'duration': 1200,  # 20 minutes
                'objectives': [
                    {
                        'type': ObjectiveType.PROTECT_CIVILIANS,
                        'description': 'Rescue 15 civilians',
                        'target': 15,
                        'points': 300
                    },
                    {
                        'type': ObjectiveType.TIME_TRIAL,
                        'description': 'Complete within time limit',
                        'target': 1200,
                        'points': 200
                    }
                ],
                'environment': {
                    'weather': 'stormy',
                    'visibility': 0.5,
                    'hazards': ['fire', 'debris', 'unstable_structures']
                }
            },
            
            ScenarioType.STEALTH_OPERATIONS: {
                'name': 'Stealth Infiltration',
                'description': 'Complete objectives without detection',
                'duration': 1800,  # 30 minutes
                'objectives': [
                    {
                        'type': ObjectiveType.AVOID_DETECTION,
                        'description': 'Remain undetected',
                        'target': True,
                        'points': 500
                    },
                    {
                        'type': ObjectiveType.REACH_WAYPOINT,
                        'description': 'Reach extraction point',
                        'target': {'position': [500, 50, 500]},
                        'points': 200
                    }
                ],
                'environment': {
                    'time_of_day': 2.0,  # Night
                    'visibility': 0.3,
                    'enemy_patrols': True
                }
            }
        }
        
        return templates
    
    def start_scenario(self, scenario_type: ScenarioType, 
                      difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
                      custom_config: Optional[Dict[str, Any]] = None):
        """Start a training scenario"""
        if self.scenario_state == "running":
            self.end_scenario()
        
        self.difficulty = difficulty
        
        # Load scenario template
        if scenario_type in self.scenario_templates:
            scenario = self.scenario_templates[scenario_type].copy()
        else:
            scenario = custom_config or {}
        
        # Apply difficulty modifiers
        self._apply_difficulty_modifiers(scenario, difficulty)
        
        # Initialize scenario
        self.current_scenario = scenario
        self.scenario_state = "loading"
        
        # Reset state
        self._reset_simulation_state()
        
        # Load objectives
        self._load_objectives(scenario.get('objectives', []))
        
        # Setup environment
        self._setup_environment(scenario.get('environment', {}))
        
        # Spawn initial entities
        self._spawn_initial_entities(scenario)
        
        # Start simulation
        self.is_running = True
        self.session_start_time = time.time()
        self.scenario_state = "running"
        
        # Start processing threads
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.start()
        
        self.ai_thread = threading.Thread(target=self._ai_update_loop)
        self.ai_thread.start()
        
        # Log event
        self._log_event('scenario_started', {
            'type': scenario_type.value,
            'difficulty': difficulty.value
        })
    
    def end_scenario(self):
        """End current training scenario"""
        if self.scenario_state != "running":
            return
        
        self.is_running = False
        self.scenario_state = "completed"
        
        # Calculate final metrics
        self.metrics.completion_time = time.time() - self.session_start_time
        final_score = self.metrics.calculate_overall_score()
        
        # Save to history
        self._save_session_results(final_score)
        
        # Stop threads
        if self.simulation_thread:
            self.simulation_thread.join()
        if self.ai_thread:
            self.ai_thread.join()
        
        # Fire callback
        if self.callbacks['on_scenario_complete']:
            self.callbacks['on_scenario_complete'](self.metrics, final_score)
        
        # Log event
        self._log_event('scenario_completed', {
            'score': final_score,
            'metrics': self.metrics.__dict__
        })
    
    def pause_scenario(self):
        """Pause training scenario"""
        if self.scenario_state == "running":
            self.scenario_state = "paused"
            self._log_event('scenario_paused', {})
    
    def resume_scenario(self):
        """Resume training scenario"""
        if self.scenario_state == "paused":
            self.scenario_state = "running"
            self._log_event('scenario_resumed', {})
    
    def _reset_simulation_state(self):
        """Reset simulation to initial state"""
        # Clear entities
        self.entities.clear()
        
        # Reset player
        self.player_state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'orientation': np.array([0, 0, 0, 1]),
            'health': 100.0,
            'energy': 100.0,
            'weapons_enabled': True,
            'flight_mode': 'hover'
        }
        
        # Reset metrics
        self.metrics = TrainingMetrics()
        
        # Clear objectives
        self.objectives.clear()
        self.completed_objectives.clear()
        
        # Clear event log
        self.event_log.clear()
    
    def _apply_difficulty_modifiers(self, scenario: Dict[str, Any], 
                                  difficulty: DifficultyLevel):
        """Apply difficulty-based modifiers to scenario"""
        modifiers = {
            DifficultyLevel.BEGINNER: {
                'enemy_health': 0.5,
                'enemy_damage': 0.5,
                'enemy_speed': 0.7,
                'player_damage_taken': 0.5,
                'objective_multiplier': 0.8,
                'time_bonus': 1.5
            },
            DifficultyLevel.INTERMEDIATE: {
                'enemy_health': 0.8,
                'enemy_damage': 0.8,
                'enemy_speed': 0.9,
                'player_damage_taken': 0.8,
                'objective_multiplier': 1.0,
                'time_bonus': 1.2
            },
            DifficultyLevel.ADVANCED: {
                'enemy_health': 1.0,
                'enemy_damage': 1.0,
                'enemy_speed': 1.0,
                'player_damage_taken': 1.0,
                'objective_multiplier': 1.0,
                'time_bonus': 1.0
            },
            DifficultyLevel.EXPERT: {
                'enemy_health': 1.5,
                'enemy_damage': 1.5,
                'enemy_speed': 1.2,
                'player_damage_taken': 1.5,
                'objective_multiplier': 1.2,
                'time_bonus': 0.8
            },
            DifficultyLevel.MASTER: {
                'enemy_health': 2.0,
                'enemy_damage': 2.0,
                'enemy_speed': 1.5,
                'player_damage_taken': 2.0,
                'objective_multiplier': 1.5,
                'time_bonus': 0.6
            }
        }
        
        mods = modifiers[difficulty]
        
        # Apply to objectives
        if 'objectives' in scenario:
            for obj in scenario['objectives']:
                if obj['type'] == ObjectiveType.DEFEAT_ENEMIES:
                    obj['target'] = int(obj['target'] * mods['objective_multiplier'])
                elif obj['type'] == ObjectiveType.TIME_TRIAL:
                    obj['target'] = int(obj['target'] * mods['time_bonus'])
        
        # Store modifiers for use during simulation
        scenario['difficulty_modifiers'] = mods
    
    def _load_objectives(self, objective_configs: List[Dict[str, Any]]):
        """Load objectives from configuration"""
        for config in objective_configs:
            objective = TrainingObjective(
                objective_type=ObjectiveType(config['type']),
                description=config['description'],
                target_value=config['target'],
                points=config.get('points', 100),
                optional=config.get('optional', False),
                time_limit=config.get('time_limit')
            )
            self.objectives.append(objective)
        
        self.metrics.objectives_total = len(self.objectives)
    
    def _setup_environment(self, env_config: Dict[str, Any]):
        """Setup training environment parameters"""
        self.environment.update(env_config)
        
        # Apply weather effects
        if self.environment['weather'] == 'stormy':
            self.environment['wind_speed'] = np.random.randn(3) * 20
            self.environment['visibility'] = 0.5
        elif self.environment['weather'] == 'foggy':
            self.environment['visibility'] = 0.3
        
        # Setup lighting based on time
        time_of_day = self.environment['time_of_day']
        if 6 <= time_of_day <= 18:
            self.environment['lighting'] = 'day'
        elif 4 <= time_of_day < 6 or 18 < time_of_day <= 20:
            self.environment['lighting'] = 'twilight'
        else:
            self.environment['lighting'] = 'night'
    
    def _spawn_initial_entities(self, scenario: Dict[str, Any]):
        """Spawn initial entities for scenario"""
        # Spawn waypoints
        if 'waypoints' in scenario:
            for i, wp in enumerate(scenario['waypoints']):
                self._spawn_entity(
                    entity_type='waypoint',
                    position=np.array(wp['position']),
                    metadata={'index': i, 'required': wp.get('required', True)}
                )
        
        # Spawn civilians for rescue scenarios
        if any(obj.objective_type == ObjectiveType.PROTECT_CIVILIANS 
               for obj in self.objectives):
            self._spawn_civilians(20)  # Spawn more than needed
        
        # Schedule enemy spawns
        if 'spawn_waves' in scenario:
            for wave in scenario['spawn_waves']:
                # Would schedule spawn with timer
                pass
    
    def _spawn_entity(self, entity_type: str, position: np.ndarray, 
                     **kwargs) -> str:
        """Spawn a single entity"""
        entity_id = f"{entity_type}_{len(self.entities)}"
        
        entity = SimulatedEntity(
            id=entity_id,
            entity_type=entity_type,
            position=position,
            velocity=np.zeros(3),
            health=100.0,
            ai_behavior=kwargs.get('ai_behavior', 'default'),
            metadata=kwargs.get('metadata', {})
        )
        
        # Apply difficulty modifiers
        if entity_type.startswith('enemy') and self.current_scenario:
            mods = self.current_scenario.get('difficulty_modifiers', {})
            entity.health *= mods.get('enemy_health', 1.0)
            entity.metadata['damage_multiplier'] = mods.get('enemy_damage', 1.0)
            entity.metadata['speed_multiplier'] = mods.get('enemy_speed', 1.0)
        
        self.entities[entity_id] = entity
        return entity_id
    
    def _spawn_civilians(self, count: int):
        """Spawn civilians for rescue scenarios"""
        bounds = self.environment['bounds']
        
        for i in range(count):
            # Random position within bounds
            position = np.array([
                np.random.uniform(-bounds[0]/2, bounds[0]/2),
                0,  # Ground level
                np.random.uniform(-bounds[2]/2, bounds[2]/2)
            ])
            
            self._spawn_entity(
                entity_type='civilian',
                position=position,
                ai_behavior='wander',
                metadata={
                    'rescued': False,
                    'in_danger': np.random.rand() < 0.3
                }
            )
    
    def _simulation_loop(self):
        """Main physics simulation loop"""
        last_time = time.time()
        
        while self.is_running:
            if self.scenario_state != "running":
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Update physics
            self._update_physics(dt)
            
            # Check collisions
            if self.config['collision_enabled']:
                self._check_collisions()
            
            # Update objectives
            self._update_objectives()
            
            # Check scenario completion
            if self._check_scenario_completion():
                self.end_scenario()
                break
            
            # Maintain timestep
            elapsed = time.time() - current_time
            if elapsed < self.config['physics_timestep']:
                time.sleep(self.config['physics_timestep'] - elapsed)
    
    def _ai_update_loop(self):
        """AI behavior update loop"""
        update_interval = 1.0 / self.config['ai_update_rate']
        
        while self.is_running:
            if self.scenario_state != "running":
                time.sleep(0.1)
                continue
            
            start_time = time.time()
            
            # Update AI entities
            for entity in list(self.entities.values()):
                if entity.active and entity.ai_behavior:
                    self._update_entity_ai(entity)
            
            # Sleep to maintain update rate
            elapsed = time.time() - start_time
            if elapsed < update_interval:
                time.sleep(update_interval - elapsed)
    
    def _update_physics(self, dt: float):
        """Update physics simulation"""
        # Update player physics
        self._update_player_physics(dt)
        
        # Update entity physics
        for entity in self.entities.values():
            if entity.active:
                # Apply velocity
                entity.position += entity.velocity * dt
                
                # Apply gravity if applicable
                if entity.entity_type not in ['waypoint', 'marker']:
                    entity.velocity += self.environment['gravity'] * dt
                
                # Apply bounds
                self._apply_bounds(entity)
    
    def _update_player_physics(self, dt: float):
        """Update player physics"""
        player = self.player_state
        
        # Apply velocity
        player['position'] += player['velocity'] * dt
        
        # Apply drag based on flight mode
        if player['flight_mode'] == 'hover':
            player['velocity'] *= 0.95  # High drag
        else:
            player['velocity'] *= 0.99  # Low drag
        
        # Apply bounds
        bounds = self.environment['bounds']
        player['position'] = np.clip(
            player['position'],
            -bounds / 2,
            bounds / 2
        )
        
        # Energy consumption
        speed = np.linalg.norm(player['velocity'])
        energy_drain = speed * 0.01 * dt  # Simplified
        player['energy'] = max(0, player['energy'] - energy_drain)
    
    def _apply_bounds(self, entity: SimulatedEntity):
        """Keep entity within bounds"""
        bounds = self.environment['bounds']
        entity.position = np.clip(
            entity.position,
            -bounds / 2,
            bounds / 2
        )
        
        # Bounce off bounds
        for i in range(3):
            if abs(entity.position[i]) >= bounds[i] / 2:
                entity.velocity[i] *= -0.5
    
    def _check_collisions(self):
        """Check and handle collisions"""
        player_pos = self.player_state['position']
        
        for entity in list(self.entities.values()):
            if not entity.active:
                continue
            
            distance = np.linalg.norm(entity.position - player_pos)
            
            # Simple sphere collision
            if distance < 5.0:  # 5 meter collision radius
                self._handle_collision(entity)
    
    def _handle_collision(self, entity: SimulatedEntity):
        """Handle collision with entity"""
        if entity.entity_type == 'waypoint':
            # Waypoint reached
            self._log_event('waypoint_reached', {'waypoint': entity.id})
            entity.active = False
            
            # Update objectives
            for obj in self.objectives:
                if obj.objective_type == ObjectiveType.REACH_WAYPOINT:
                    obj.current_value = True
                    
        elif entity.entity_type == 'civilian':
            # Rescue civilian
            if not entity.metadata.get('rescued', False):
                entity.metadata['rescued'] = True
                self._log_event('civilian_rescued', {'civilian': entity.id})
                
                # Update objectives
                for obj in self.objectives:
                    if obj.objective_type == ObjectiveType.PROTECT_CIVILIANS:
                        obj.current_value += 1
                        
        elif entity.entity_type.startswith('enemy'):
            # Take damage from enemy
            if self.config['damage_enabled']:
                damage = 10 * entity.metadata.get('damage_multiplier', 1.0)
                self._apply_damage_to_player(damage)
    
    def _update_entity_ai(self, entity: SimulatedEntity):
        """Update entity AI behavior"""
        if entity.ai_behavior == 'wander':
            # Random wandering
            if np.random.rand() < 0.1:  # 10% chance to change direction
                entity.velocity = np.random.randn(3) * 2
                entity.velocity[1] = 0  # Keep on ground
                
        elif entity.ai_behavior == 'pursue':
            # Pursue player
            to_player = self.player_state['position'] - entity.position
            distance = np.linalg.norm(to_player)
            
            if distance > 0:
                direction = to_player / distance
                speed = 10 * entity.metadata.get('speed_multiplier', 1.0)
                entity.velocity = direction * speed
                
        elif entity.ai_behavior == 'patrol':
            # Patrol between waypoints
            waypoints = entity.metadata.get('patrol_points', [])
            if waypoints:
                current_wp_idx = entity.metadata.get('current_waypoint', 0)
                target = np.array(waypoints[current_wp_idx])
                
                to_target = target - entity.position
                distance = np.linalg.norm(to_target)
                
                if distance < 5:  # Reached waypoint
                    current_wp_idx = (current_wp_idx + 1) % len(waypoints)
                    entity.metadata['current_waypoint'] = current_wp_idx
                else:
                    direction = to_target / distance
                    entity.velocity = direction * 5
    
    def _update_objectives(self):
        """Update objective progress"""
        for obj in self.objectives:
            if not obj.completed:
                # Check completion
                if obj.check_completion():
                    obj.completed = True
                    self.completed_objectives.append(obj)
                    self.metrics.objectives_completed += 1
                    
                    # Fire callback
                    if self.callbacks['on_objective_complete']:
                        self.callbacks['on_objective_complete'](obj)
                    
                    self._log_event('objective_completed', {
                        'objective': obj.description,
                        'points': obj.points
                    })
                
                # Check time limit
                if obj.time_limit and self.session_start_time:
                    elapsed = time.time() - self.session_start_time
                    if elapsed > obj.time_limit and not obj.optional:
                        # Failed time limit
                        self._log_event('objective_failed', {
                            'objective': obj.description,
                            'reason': 'time_limit_exceeded'
                        })
    
    def _check_scenario_completion(self) -> bool:
        """Check if scenario is completed"""
        # Check if all required objectives are complete
        required_objectives = [obj for obj in self.objectives if not obj.optional]
        completed_required = [obj for obj in required_objectives if obj.completed]
        
        if len(completed_required) == len(required_objectives):
            return True
        
        # Check scenario time limit
        if self.current_scenario and 'duration' in self.current_scenario:
            elapsed = time.time() - self.session_start_time
            if elapsed > self.current_scenario['duration']:
                return True
        
        # Check failure conditions
        if self.player_state['health'] <= 0 and not self.config['respawn_enabled']:
            return True
        
        return False
    
    def _apply_damage_to_player(self, damage: float):
        """Apply damage to player"""
        self.player_state['health'] -= damage
        self.metrics.damage_taken += damage
        
        if self.callbacks['on_player_damaged']:
            self.callbacks['on_player_damaged'](damage)
        
        self._log_event('player_damaged', {'damage': damage})
        
        # Check for player death
        if self.player_state['health'] <= 0:
            if self.config['respawn_enabled']:
                self._respawn_player()
            else:
                self._log_event('player_died', {})
    
    def _respawn_player(self):
        """Respawn player at checkpoint"""
        self.player_state['health'] = 100.0
        self.player_state['position'] = np.zeros(3)  # Reset to start
        self.player_state['velocity'] = np.zeros(3)
        
        self._log_event('player_respawned', {})
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log training event"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        }
        self.event_log.append(event)
    
    def _save_session_results(self, score: float):
        """Save training session results"""
        session_data = {
            'timestamp': time.time(),
            'scenario': self.current_scenario.get('name', 'Unknown'),
            'difficulty': self.difficulty.value,
            'score': score,
            'metrics': self.metrics.__dict__,
            'objectives_completed': len(self.completed_objectives),
            'duration': self.metrics.completion_time,
            'events': list(self.event_log)
        }
        
        self.training_history.append(session_data)
        
        # Update best scores
        scenario_name = self.current_scenario.get('name', 'Unknown')
        if scenario_name not in self.best_scores or score > self.best_scores[scenario_name]:
            self.best_scores[scenario_name] = score
    
    # Public API methods
    def apply_player_input(self, input_data: Dict[str, Any]):
        """Apply player control input"""
        if self.scenario_state != "running":
            return
        
        # Movement input
        if 'thrust' in input_data:
            thrust = np.array(input_data['thrust'])
            self.player_state['velocity'] += thrust * 0.1
        
        # Rotation input
        if 'rotation' in input_data:
            self.player_state['orientation'] = np.array(input_data['rotation'])
        
        # Weapon input
        if 'fire_weapon' in input_data and input_data['fire_weapon']:
            self._fire_weapon(input_data.get('weapon_type', 'repulsor'))
        
        # Flight mode
        if 'flight_mode' in input_data:
            self.player_state['flight_mode'] = input_data['flight_mode']
    
    def _fire_weapon(self, weapon_type: str):
        """Handle weapon firing"""
        if not self.player_state['weapons_enabled']:
            return
        
        # Track shots for accuracy
        self.metrics.accuracy = 0.75  # Placeholder
        
        # Check for hit enemies
        player_pos = self.player_state['position']
        
        for entity in list(self.entities.values()):
            if entity.entity_type.startswith('enemy') and entity.active:
                distance = np.linalg.norm(entity.position - player_pos)
                
                if distance < 50:  # Simplified hit detection
                    # Apply damage
                    damage = 25
                    entity.health -= damage
                    self.metrics.damage_dealt += damage
                    
                    if entity.health <= 0:
                        entity.active = False
                        self._log_event('enemy_destroyed', {'enemy': entity.id})
                        
                        # Update objectives
                        for obj in self.objectives:
                            if obj.objective_type == ObjectiveType.DEFEAT_ENEMIES:
                                obj.current_value += 1
    
    def get_scenario_list(self) -> List[Dict[str, Any]]:
        """Get list of available scenarios"""
        scenarios = []
        
        for scenario_type, template in self.scenario_templates.items():
            scenarios.append({
                'type': scenario_type.value,
                'name': template['name'],
                'description': template['description'],
                'duration': template.get('duration', 0),
                'difficulty_levels': [d.value for d in DifficultyLevel]
            })
        
        return scenarios
    
    def get_player_state(self) -> Dict[str, Any]:
        """Get current player state"""
        return self.player_state.copy()
    
    def get_objectives_status(self) -> List[Dict[str, Any]]:
        """Get status of all objectives"""
        status = []
        
        for obj in self.objectives:
            status.append({
                'description': obj.description,
                'type': obj.objective_type.value,
                'progress': f"{obj.current_value}/{obj.target_value}",
                'completed': obj.completed,
                'optional': obj.optional,
                'points': obj.points
            })
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        return {
            **self.metrics.__dict__,
            'overall_score': self.metrics.calculate_overall_score()
        }
    
    def get_training_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent training history"""
        return self.training_history[-limit:]
    
    def get_best_scores(self) -> Dict[str, float]:
        """Get best scores for each scenario"""
        return self.best_scores.copy()
    
    def export_session_data(self, filename: str):
        """Export session data to file"""
        if self.training_history:
            with open(filename, 'w') as f:
                json.dump(self.training_history[-1], f, indent=2)
    
    def set_config(self, config: Dict[str, Any]):
        """Update simulation configuration"""
        self.config.update(config)
    
    def set_callback(self, event: str, callback: Callable):
        """Set event callback"""
        if event in self.callbacks:
            self.callbacks[event] = callback