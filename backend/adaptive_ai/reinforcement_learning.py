"""
Advanced Reinforcement Learning System for Iron Man Suit

This module provides a comprehensive RL framework including:
- Multiple RL algorithms (DQN, PPO, SAC)
- Multi-agent coordination for drone swarms
- Specialized agents for different suit functions
- Experience replay and curriculum learning
"""

import logging
import random
import numpy as np
from collections import deque
from typing import Any, Deque, Tuple, List, Optional, Dict, Union
from abc import ABC, abstractmethod
import json
import pickle

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Advanced experience replay buffer with prioritized sampling."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        priority: float = None,
    ):
        """Add experience with optional priority."""
        self.buffer.append((state, action, reward, next_state, done))

        if priority is None:
            priority = self.max_priority
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[List, List, List, List, List, List]:
        """Sample experiences with importance sampling weights."""
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self.buffer)} samples, need {batch_size}"
            )

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Extract experiences
        experiences = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*experiences)

        return (
            list(states),
            list(actions),
            list(rewards),
            list(next_states),
            list(dones),
            weights.tolist(),
        )

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for specific experiences."""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.training = True

    @abstractmethod
    def select_action(self, state: np.ndarray, **kwargs) -> Any:
        """Select action given current state."""
        pass

    @abstractmethod
    def train(self, batch: Tuple) -> Dict[str, float]:
        """Train the agent on a batch of experiences."""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save agent parameters."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load agent parameters."""
        pass

    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training


    class DQNAgent(BaseAgent):
    """Deep Q-Network with dueling architecture and double Q-learning."""

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for DQNAgent")

        super().__init__(state_dim, action_dim, kwargs.get("device", "cpu"))

        # Hyperparameters
        self.lr = kwargs.get("lr", 1e-3)
        self.gamma = kwargs.get("gamma", 0.99)
        self.batch_size = kwargs.get("batch_size", 64)
        self.target_update_freq = kwargs.get("target_update_freq", 1000)
        self.epsilon = kwargs.get("epsilon", 0.1)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        self.epsilon_min = kwargs.get("epsilon_min", 0.01)

        # Networks
        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay buffer
        self.buffer = ReplayBuffer(kwargs.get("buffer_size", 100000))
        self.steps = 0

    def select_action(self, state: np.ndarray, **kwargs) -> int:
        """Epsilon-greedy action selection."""
        if self.training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()

    def train(self, batch: Tuple) -> Dict[str, float]:
        """Train the agent."""
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones, weights = self.buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Loss with importance sampling weights
        loss = (weights_t * F.mse_loss(current_q, target_q, reduction="none")).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def save(self, filepath: str):
        """Save agent."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load agent."""
        data = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(data["policy_net"])
        self.target_net.load_state_dict(data["target_net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.epsilon = data["epsilon"]
        self.steps = data["steps"]


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent for continuous control."""

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PPOAgent")

        super().__init__(state_dim, action_dim, kwargs.get("device", "cpu"))

        # Hyperparameters
        self.lr = kwargs.get("lr", 3e-4)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.clip_ratio = kwargs.get("clip_ratio", 0.2)
        self.value_coef = kwargs.get("value_coef", 0.5)
        self.entropy_coef = kwargs.get("entropy_coef", 0.01)

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Memory
        self.memory = []

    def select_action(
        self, state: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, float, float]:
        """Select action and return log probability."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.actor(state_t)
            value = self.critic(state_t)

        return action.cpu().numpy()[0], log_prob.item(), value.item()

    def store_transition(
        self, state, action, reward, next_state, done, log_prob, value
    ):
        """Store transition for PPO training."""
        self.memory.append((state, action, reward, next_state, done, log_prob, value))

    def train(self, batch: Tuple = None) -> Dict[str, float]:
        """Train PPO agent."""
        if len(self.memory) < 64:  # Minimum batch size
            return {}

        # Process memory into training data
        states, actions, rewards, next_states, dones, log_probs, values = zip(
            *self.memory
        )

        # Calculate advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + np.array(values)

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs_t = torch.FloatTensor(log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # PPO update
        for _ in range(10):  # Multiple epochs
            # Actor update
            new_actions, new_log_probs = self.actor(states_t, actions_t)
            ratio = torch.exp(new_log_probs - old_log_probs_t)

            surr1 = ratio * advantages_t
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                * advantages_t
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic update
            new_values = self.critic(states_t)
            critic_loss = F.mse_loss(new_values.squeeze(), returns_t)

            # Entropy bonus
            entropy = self.actor.entropy(states_t).mean()

            # Total loss
            total_loss = (
                actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            )

            # Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # Clear memory
        self.memory.clear()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }

    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            )
            last_advantage = advantages[t]

        return advantages

    def save(self, filepath: str):
        """Save agent."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load agent."""
        data = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])


class SACAgent(BaseAgent):
    """Soft Actor-Critic for continuous control with maximum entropy."""

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SACAgent")

        super().__init__(state_dim, action_dim, kwargs.get("device", "cpu"))

        # Hyperparameters
        self.lr = kwargs.get("lr", 3e-4)
        self.gamma = kwargs.get("gamma", 0.99)
        self.tau = kwargs.get("tau", 0.005)
        self.alpha = kwargs.get("alpha", 0.2)
        self.auto_alpha = kwargs.get("auto_alpha", True)
        self.target_entropy = kwargs.get("target_entropy", -action_dim)

        # Networks
        self.actor = SACActor(state_dim, action_dim).to(self.device)
        self.critic1 = SACCritic(state_dim, action_dim).to(self.device)
        self.critic2 = SACCritic(state_dim, action_dim).to(self.device)
        self.target_critic1 = SACCritic(state_dim, action_dim).to(self.device)
        self.target_critic2 = SACCritic(state_dim, action_dim).to(self.device)

        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)

        # Alpha optimization
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

        # Replay buffer
        self.buffer = ReplayBuffer(kwargs.get("buffer_size", 100000))
        self.batch_size = kwargs.get("batch_size", 64)

    def select_action(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Select action using current policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.actor(state_t)

        return action.cpu().numpy()[0]

    def train(self, batch: Tuple = None) -> Dict[str, float]:
        """Train SAC agent."""
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones, _ = self.buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states_t)
            next_q1 = self.target_critic1(next_states_t, next_actions)
            next_q2 = self.target_critic2(next_states_t, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards_t + self.gamma * (1 - dones_t) * next_q

        current_q1 = self.critic1(states_t, actions_t)
        current_q2 = self.critic2(states_t, actions_t)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        actions_new, log_probs = self.actor(states_t)
        q1_new = self.critic1(states_t, actions_new)
        q2_new = self.critic2(states_t, actions_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Target network update
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return {
            "actor_loss": actor_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "alpha": self.alpha.item() if self.auto_alpha else self.alpha,
        }

    def _soft_update(self, source, target):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save agent."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic1_optimizer": self.critic1_optimizer.state_dict(),
                "critic2_optimizer": self.critic2_optimizer.state_dict(),
                "alpha": self.alpha,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load agent."""
            data = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic1.load_state_dict(data["critic1"])
        self.critic2.load_state_dict(data["critic2"])
        self.target_critic1.load_state_dict(data["target_critic1"])
        self.target_critic2.load_state_dict(data["target_critic2"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(data["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(data["critic2_optimizer"])
        self.alpha = data["alpha"]


class MultiAgentCoordinator:
    """Coordinates multiple agents for complex tasks like drone swarms."""

    def __init__(
        self, agents: Dict[str, BaseAgent], coordination_strategy: str = "centralized"
    ):
        self.agents = agents
        self.coordination_strategy = coordination_strategy
        self.communication_matrix = self._initialize_communication()

    def _initialize_communication(self) -> np.ndarray:
        """Initialize communication matrix between agents."""
        n_agents = len(self.agents)
        return np.ones((n_agents, n_agents)) - np.eye(n_agents)

    def coordinate_actions(self, states: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Coordinate actions across all agents."""
        if self.coordination_strategy == "centralized":
            return self._centralized_coordination(states)
        elif self.coordination_strategy == "decentralized":
            return self._decentralized_coordination(states)
        elif self.coordination_strategy == "hierarchical":
            return self._hierarchical_coordination(states)
else:
            raise ValueError(
                f"Unknown coordination strategy: {self.coordination_strategy}"
            )

    def _centralized_coordination(
        self, states: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Centralized coordination with global state."""
        # Combine all states into global state
        global_state = np.concatenate(list(states.values()))

        # Use a coordinator agent or rule-based system
        # For now, use simple averaging of individual agent decisions
        actions = {}
        for agent_name, agent in self.agents.items():
            actions[agent_name] = agent.select_action(states[agent_name])

        return actions

    def _decentralized_coordination(
        self, states: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Decentralized coordination with local communication."""
        actions = {}
        for agent_name, agent in self.agents.items():
            # Each agent makes decision based on local state
            actions[agent_name] = agent.select_action(states[agent_name])

        # Apply coordination constraints
        actions = self._apply_coordination_constraints(actions, states)

        return actions

    def _hierarchical_coordination(
        self, states: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Hierarchical coordination with leader-follower structure."""
        # Identify leader agent (could be based on role or performance)
        leader_name = list(self.agents.keys())[0]  # Simple: first agent is leader

        # Leader makes high-level decision
        leader_action = self.agents[leader_name].select_action(states[leader_name])

        # Followers adapt based on leader's decision
        actions = {leader_name: leader_action}
        for agent_name, agent in self.agents.items():
            if agent_name != leader_name:
                # Modify state to include leader's decision
                modified_state = np.concatenate([states[agent_name], leader_action])
                actions[agent_name] = agent.select_action(modified_state)

        return actions

    def _apply_coordination_constraints(
        self, actions: Dict[str, Any], states: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Apply coordination constraints to prevent conflicts."""
        # Example: prevent agents from colliding
        # This is a simplified version - real implementation would be more complex

        modified_actions = actions.copy()

        # Check for potential conflicts and resolve them
        agent_names = list(actions.keys())
        for i, agent1 in enumerate(agent_names):
            for j, agent2 in enumerate(agent_names[i + 1 :], i + 1):
                # Simple collision avoidance
                if self._check_conflict(actions[agent1], actions[agent2]):
                    # Resolve conflict by modifying one action
                    modified_actions[agent2] = self._resolve_conflict(
                        actions[agent1], actions[agent2]
                    )

        return modified_actions

    def _check_conflict(self, action1: Any, action2: Any) -> bool:
        """Check if two actions conflict."""
        # Simplified conflict detection
        # In reality, this would check for spatial conflicts, resource conflicts, etc.
        return False  # Placeholder

    def _resolve_conflict(self, action1: Any, action2: Any) -> Any:
        """Resolve conflict between two actions."""
        # Simplified conflict resolution
        # In reality, this would implement sophisticated conflict resolution strategies
        return action2  # Placeholder


# Neural Network Architectures


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class ActorNetwork(nn.Module):
    """Actor network for PPO with continuous actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, action=None):
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)

        std = log_std.exp()
        normal = Normal(mean, std)

        if action is None:
            action = normal.rsample()

        log_prob = normal.log_prob(action).sum(dim=-1)
        return action, log_prob

    def entropy(self, state):
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = Normal(mean, std)
        return normal.entropy().sum(dim=-1)


class CriticNetwork(nn.Module):
    """Critic network for PPO."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state)


class SACActor(nn.Module):
    """Actor network for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)

        std = log_std.exp()
        normal = Normal(mean, std)

        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(dim=-1)

        # Tanh squashing
        action = torch.tanh(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob


class SACCritic(nn.Module):
    """Critic network for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)
