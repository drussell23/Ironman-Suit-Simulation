"""
Unit tests for the Reinforcement Learning module
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from adaptive_ai.reinforcement_learning import (
    ReplayBuffer,
    BaseAgent,
    DQNAgent,
    PPOAgent,
    SACAgent,
    MultiAgentCoordinator
)


class TestReplayBuffer:
    """Test ReplayBuffer functionality"""
    
    def test_initialization(self):
        """Test buffer initialization"""
        buffer = ReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)
        assert buffer.capacity == 1000
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert len(buffer.buffer) == 0
        assert buffer.max_priority == 1.0
    
    def test_add_experience(self, sample_state, sample_action):
        """Test adding experiences to buffer"""
        buffer = ReplayBuffer(capacity=10)
        
        # Add single experience
        next_state = sample_state + 0.1
        buffer.add(sample_state, sample_action, 1.0, next_state, False)
        
        assert len(buffer.buffer) == 1
        assert len(buffer.priorities) == 1
    
    def test_sample_experiences(self, sample_state, sample_action):
        """Test sampling from buffer"""
        buffer = ReplayBuffer(capacity=100)
        
        # Add multiple experiences
        for i in range(50):
            state = sample_state + i * 0.01
            next_state = state + 0.1
            buffer.add(state, sample_action, float(i), next_state, False)
        
        # Sample batch
        batch = buffer.sample(batch_size=10)
        assert batch is not None
        assert len(batch) == 5  # states, actions, rewards, next_states, dones
        assert batch[0].shape[0] == 10  # batch size
    
    def test_buffer_overflow(self, sample_state, sample_action):
        """Test buffer behavior when capacity is exceeded"""
        buffer = ReplayBuffer(capacity=5)
        
        # Add more than capacity
        for i in range(10):
            buffer.add(sample_state, sample_action, 1.0, sample_state, False)
        
        assert len(buffer.buffer) == 5
        assert len(buffer.priorities) == 5
    
    def test_prioritized_sampling(self, sample_state, sample_action):
        """Test prioritized experience replay"""
        buffer = ReplayBuffer(capacity=100, alpha=0.6)
        
        # Add experiences with different priorities
        for i in range(20):
            priority = (i + 1) / 20.0
            buffer.add(sample_state, sample_action, 1.0, sample_state, False, priority=priority)
        
        # Higher priority experiences should be sampled more often
        batch = buffer.sample(batch_size=10)
        assert batch is not None


@pytest.mark.skipif(not torch, reason="PyTorch not available")
class TestDQNAgent:
    """Test DQN Agent functionality"""
    
    def test_initialization(self, sample_observation_space, sample_action_space):
        """Test DQN agent initialization"""
        agent = DQNAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space,
            learning_rate=3e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        assert agent.observation_space == sample_observation_space
        assert agent.action_space == sample_action_space
        assert agent.epsilon == 1.0
        assert agent.gamma == 0.99
    
    def test_act_exploration(self, sample_state, sample_observation_space, sample_action_space):
        """Test action selection in exploration mode"""
        agent = DQNAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space,
            epsilon=1.0  # Full exploration
        )
        
        # Should return random actions
        actions = []
        for _ in range(10):
            action = agent.act(sample_state)
            actions.append(action)
        
        # Check actions are within valid range
        assert all(0 <= a < sample_action_space for a in actions)
    
    def test_act_exploitation(self, sample_state, sample_observation_space, sample_action_space):
        """Test action selection in exploitation mode"""
        agent = DQNAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space,
            epsilon=0.0  # No exploration
        )
        
        # Should return deterministic actions
        action1 = agent.act(sample_state)
        action2 = agent.act(sample_state)
        
        assert action1 == action2
    
    def test_remember(self, sample_state, sample_action, sample_observation_space, sample_action_space):
        """Test experience storage"""
        agent = DQNAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space
        )
        
        next_state = sample_state + 0.1
        agent.remember(sample_state, 0, 1.0, next_state, False)
        
        # Check experience was stored
        assert len(agent.memory.buffer) == 1
    
    @patch('torch.nn.Module.state_dict')
    def test_update_target_network(self, mock_state_dict, sample_observation_space, sample_action_space):
        """Test target network update"""
        agent = DQNAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space
        )
        
        # Mock state dict
        mock_state_dict.return_value = {'test': torch.tensor([1.0])}
        
        # Update target network
        agent.update_target_network()
        
        # Verify update was called
        assert mock_state_dict.called


@pytest.mark.skipif(not torch, reason="PyTorch not available")
class TestPPOAgent:
    """Test PPO Agent functionality"""
    
    def test_initialization(self, sample_observation_space, sample_action_space):
        """Test PPO agent initialization"""
        agent = PPOAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space,
            learning_rate=3e-4,
            gamma=0.99,
            clip_ratio=0.2,
            epochs=10
        )
        
        assert agent.observation_space == sample_observation_space
        assert agent.action_space == sample_action_space
        assert agent.clip_ratio == 0.2
        assert agent.epochs == 10
    
    def test_act(self, sample_state, sample_observation_space, sample_action_space):
        """Test PPO action selection"""
        agent = PPOAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space
        )
        
        # Get action and log probability
        action, log_prob = agent.act(sample_state)
        
        assert action.shape == (sample_action_space,)
        assert isinstance(log_prob, torch.Tensor)
    
    def test_compute_advantages(self, sample_observation_space, sample_action_space):
        """Test advantage computation"""
        agent = PPOAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space
        )
        
        # Create fake trajectory
        rewards = [1.0, 2.0, 3.0, 4.0]
        values = torch.tensor([0.5, 1.5, 2.5, 3.5])
        dones = [False, False, False, True]
        
        advantages = agent.compute_advantages(rewards, values, dones)
        
        assert isinstance(advantages, torch.Tensor)
        assert advantages.shape[0] == len(rewards)


@pytest.mark.skipif(not torch, reason="PyTorch not available")
class TestSACAgent:
    """Test SAC Agent functionality"""
    
    def test_initialization(self, sample_observation_space, sample_action_space):
        """Test SAC agent initialization"""
        agent = SACAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2
        )
        
        assert agent.observation_space == sample_observation_space
        assert agent.action_space == sample_action_space
        assert agent.alpha == 0.2
        assert agent.tau == 0.005
    
    def test_act(self, sample_state, sample_observation_space, sample_action_space):
        """Test SAC action selection"""
        agent = SACAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space
        )
        
        # Test deterministic action
        action_det = agent.act(sample_state, deterministic=True)
        assert action_det.shape == (sample_action_space,)
        
        # Test stochastic action
        action_stoch = agent.act(sample_state, deterministic=False)
        assert action_stoch.shape == (sample_action_space,)
    
    def test_soft_update(self, sample_observation_space, sample_action_space):
        """Test soft target network update"""
        agent = SACAgent(
            observation_space=sample_observation_space,
            action_space=sample_action_space,
            tau=0.1
        )
        
        # Get initial target parameters
        if hasattr(agent, 'critic_target'):
            initial_params = list(agent.critic_target.parameters())[0].clone()
            
            # Perform soft update
            agent.soft_update()
            
            # Check parameters changed
            updated_params = list(agent.critic_target.parameters())[0]
            assert not torch.equal(initial_params, updated_params)


class TestMultiAgentCoordinator:
    """Test Multi-Agent Coordinator functionality"""
    
    def test_initialization(self):
        """Test coordinator initialization"""
        coordinator = MultiAgentCoordinator(num_agents=3)
        
        assert coordinator.num_agents == 3
        assert len(coordinator.agents) == 0
        assert len(coordinator.communication_graph) == 3
    
    def test_add_agent(self, sample_observation_space, sample_action_space):
        """Test adding agents to coordinator"""
        coordinator = MultiAgentCoordinator(num_agents=2)
        
        # Add DQN agent
        agent1 = Mock(spec=DQNAgent)
        coordinator.add_agent(0, agent1)
        
        assert 0 in coordinator.agents
        assert coordinator.agents[0] == agent1
    
    def test_coordinate_actions(self, sample_state):
        """Test coordinated action selection"""
        coordinator = MultiAgentCoordinator(num_agents=2)
        
        # Add mock agents
        agent1 = Mock()
        agent1.act.return_value = np.array([0.5, 0.5])
        
        agent2 = Mock()
        agent2.act.return_value = np.array([0.3, 0.7])
        
        coordinator.add_agent(0, agent1)
        coordinator.add_agent(1, agent2)
        
        # Get coordinated actions
        observations = {0: sample_state, 1: sample_state}
        actions = coordinator.coordinate_actions(observations)
        
        assert len(actions) == 2
        assert 0 in actions and 1 in actions
    
    def test_update_communication(self):
        """Test communication graph update"""
        coordinator = MultiAgentCoordinator(num_agents=3)
        
        # Update communication
        coordinator.update_communication_graph(0, 1, weight=0.8)
        
        assert coordinator.communication_graph[0][1] == 0.8
        assert coordinator.communication_graph[1][0] == 0.8
    
    def test_get_neighbors(self):
        """Test getting agent neighbors"""
        coordinator = MultiAgentCoordinator(num_agents=3)
        
        # Set up communication
        coordinator.update_communication_graph(0, 1, weight=0.8)
        coordinator.update_communication_graph(0, 2, weight=0.5)
        
        neighbors = coordinator.get_neighbors(0, threshold=0.6)
        
        assert 1 in neighbors
        assert 2 not in neighbors  # Below threshold