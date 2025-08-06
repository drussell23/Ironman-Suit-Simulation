"""
Advanced Reinforcement Learning for Iron Man Suit

This module provides state-of-the-art RL algorithms:
- Proximal Policy Optimization (PPO) with advanced features
- Soft Actor-Critic (SAC) with temperature auto-tuning
- Twin Delayed Deep Deterministic Policy Gradient (TD3)
- Hierarchical Reinforcement Learning (HRL)
- Multi-task and meta-RL
- Curiosity-driven exploration
- Imitation learning
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, namedtuple
from dataclasses import dataclass
import random
import math

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


@dataclass
class RLConfig:
    """Configuration for RL algorithms."""

    # General parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    buffer_size: int = 100000

    # PPO specific
    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 10
    ppo_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # SAC specific
    sac_alpha: float = 0.2
    sac_auto_alpha: bool = True
    sac_target_entropy: float = -1.0

    # TD3 specific
    td3_noise: float = 0.2
    td3_noise_clip: float = 0.5
    td3_policy_delay: int = 2


class AdvancedPPO(nn.Module):
    """Advanced Proximal Policy Optimization with multiple improvements."""

    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        super().__init__()

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2),  # Mean and log_std
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )

        # Memory
        self.memory = deque(maxlen=config.buffer_size)

        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor and critic."""
        actor_output = self.actor(state)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)

        value = self.critic(state)

        return mean, log_std, value

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            mean, log_std, value = self.forward(state_tensor)

            if deterministic:
                action = mean
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                action = normal.rsample()

            log_prob = (
                torch.distributions.Normal(mean, log_std.exp()).log_prob(action).sum(-1)
            )

        return action.squeeze(0).numpy(), {
            "log_prob": log_prob.item(),
            "value": value.item(),
            "mean": mean.squeeze(0).numpy(),
            "std": log_std.exp().squeeze(0).numpy(),
        }

    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ):
        """Store experience in memory."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append((experience, info))

    def train(self) -> Dict[str, float]:
        """Train the PPO agent."""
        if len(self.memory) < self.config.batch_size:
            return {}

        # Sample batch
        batch = random.sample(self.memory, self.config.batch_size)
        experiences, infos = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.FloatTensor([exp.action for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences])
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.FloatTensor([exp.done for exp in experiences])
        old_log_probs = torch.FloatTensor([info["log_prob"] for info in infos])
        old_values = torch.FloatTensor([info["value"] for info in infos])

        # Compute advantages using GAE
        advantages = self._compute_gae(states, rewards, next_states, dones, old_values)
        returns = advantages + old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for _ in range(self.config.ppo_epochs):
            # Actor update
            mean, log_std, values = self.forward(states)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            log_probs = normal.log_prob(actions).sum(-1)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(
                    ratio,
                    1 - self.config.ppo_clip_ratio,
                    1 + self.config.ppo_clip_ratio,
                )
                * advantages
            )

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -normal.entropy().mean()

            # Critic update
            critic_loss = F.mse_loss(values.squeeze(), returns)

            # Total loss
            total_loss = (
                actor_loss
                + self.config.value_coef * critic_loss
                + self.config.entropy_coef * entropy_loss
            )

            # Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())

        # Clear memory
        self.memory.clear()

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy_loss": np.mean(entropy_losses),
            "avg_reward": (
                np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            ),
        }

    def _compute_gae(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        with torch.no_grad():
            _, _, next_values = self.forward(next_states)
            next_values = next_values.squeeze()

            deltas = rewards + self.config.gamma * next_values * (1 - dones) - values
            advantages = torch.zeros_like(deltas)

            gae = 0
            for t in reversed(range(len(deltas))):
                gae = (
                    deltas[t]
                    + self.config.gamma * self.config.ppo_lambda * (1 - dones[t]) * gae
                )
                advantages[t] = gae

        return advantages


class AdvancedSAC(nn.Module):
    """Advanced Soft Actor-Critic with temperature auto-tuning."""

    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        super().__init__()

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Networks
        self.actor = SACActor(state_dim, action_dim)
        self.critic1 = SACCritic(state_dim, action_dim)
        self.critic2 = SACCritic(state_dim, action_dim)
        self.target_critic1 = SACCritic(state_dim, action_dim)
        self.target_critic2 = SACCritic(state_dim, action_dim)

        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=config.learning_rate
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=config.learning_rate
        )

        # Temperature auto-tuning
        if config.sac_auto_alpha:
            self.log_alpha = nn.Parameter(torch.zeros(1))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)
            self.target_entropy = config.sac_target_entropy
        else:
            self.alpha = config.sac_alpha

        # Memory
        self.memory = deque(maxlen=config.buffer_size)

        # Training statistics
        self.update_count = 0

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor."""
        return self.actor(state)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, _ = self.actor(state_tensor)

            if deterministic:
                # Use mean action for deterministic selection
                mean, _ = self.actor.get_mean_and_std(state_tensor)
                action = mean

        return action.squeeze(0).numpy()

    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in memory."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def train(self) -> Dict[str, float]:
        """Train the SAC agent."""
        if len(self.memory) < self.config.batch_size:
            return {}

        # Sample batch
        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Get current alpha
        if self.config.sac_auto_alpha:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_probs
            target_q = rewards + self.config.gamma * (1 - dones) * next_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        actions_new, log_probs = self.actor(states)
        q1_new = self.critic1(states, actions_new)
        q2_new = self.critic2(states, actions_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        if self.config.sac_auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()

        # Target network update
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        self.update_count += 1

        return {
            "actor_loss": actor_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "alpha": alpha.item() if self.config.sac_auto_alpha else alpha,
            "update_count": self.update_count,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.config.tau * source_param.data
                + (1 - self.config.tau) * target_param.data
            )


class SACActor(nn.Module):
    """Actor network for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(dim=-1)

        # Tanh squashing
        action = torch.tanh(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob

    def get_mean_and_std(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and standard deviation without sampling."""
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        return torch.tanh(mean), std


class SACCritic(nn.Module):
    """Critic network for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class TD3(nn.Module):
    """Twin Delayed Deep Deterministic Policy Gradient."""

    def __init__(self, state_dim: int, action_dim: int, config: RLConfig):
        super().__init__()

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Networks
        self.actor = TD3Actor(state_dim, action_dim)
        self.critic1 = TD3Critic(state_dim, action_dim)
        self.critic2 = TD3Critic(state_dim, action_dim)
        self.target_actor = TD3Actor(state_dim, action_dim)
        self.target_critic1 = TD3Critic(state_dim, action_dim)
        self.target_critic2 = TD3Critic(state_dim, action_dim)

        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=config.learning_rate
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=config.learning_rate
        )

        # Memory
        self.memory = deque(maxlen=config.buffer_size)

        # Training statistics
        self.update_count = 0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor."""
        return self.actor(state)

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_tensor)

            if add_noise:
                noise = torch.randn_like(action) * self.config.td3_noise
                noise = torch.clamp(
                    noise, -self.config.td3_noise_clip, self.config.td3_noise_clip
                )
                action = torch.clamp(action + noise, -1, 1)

        return action.squeeze(0).numpy()

    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in memory."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def train(self) -> Dict[str, float]:
        """Train the TD3 agent."""
        if len(self.memory) < self.config.batch_size:
            return {}

        # Sample batch
        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            noise = torch.randn_like(next_actions) * self.config.td3_noise
            noise = torch.clamp(
                noise, -self.config.td3_noise_clip, self.config.td3_noise_clip
            )
            next_actions = torch.clamp(next_actions + noise, -1, 1)

            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + self.config.gamma * (1 - dones) * next_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update (delayed)
        if self.update_count % self.config.td3_policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target network update
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)
        else:
            actor_loss = torch.tensor(0.0)

        self.update_count += 1

        return {
            "actor_loss": actor_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "update_count": self.update_count,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.config.tau * source_param.data
                + (1 - self.config.tau) * target_param.data
            )


class TD3Actor(nn.Module):
    """Actor network for TD3."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)


class TD3Critic(nn.Module):
    """Critic network for TD3."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class CuriosityModule(nn.Module):
    """Curiosity-driven exploration module."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Forward model (predicts next state)
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Inverse model (predicts action from states)
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Optimizers
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=3e-4)
        self.inverse_optimizer = optim.Adam(self.inverse_model.parameters(), lr=3e-4)

        # Curiosity coefficient
        self.curiosity_coef = 0.1

    def compute_curiosity_reward(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute curiosity reward based on prediction error."""
        # Forward model prediction
        forward_input = torch.cat([state, action], dim=-1)
        predicted_next_state = self.forward_model(forward_input)

        # Forward model loss
        forward_loss = F.mse_loss(predicted_next_state, next_state)

        # Inverse model prediction
        inverse_input = torch.cat([state, next_state], dim=-1)
        predicted_action = self.inverse_model(inverse_input)

        # Inverse model loss
        inverse_loss = F.mse_loss(predicted_action, action)

        # Curiosity reward
        curiosity_reward = self.curiosity_coef * (forward_loss + inverse_loss)

        return curiosity_reward

    def train_curiosity(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ):
        """Train curiosity models."""
        # Forward model training
        forward_input = torch.cat([state, action], dim=-1)
        predicted_next_state = self.forward_model(forward_input)
        forward_loss = F.mse_loss(predicted_next_state, next_state)

        self.forward_optimizer.zero_grad()
        forward_loss.backward()
        self.forward_optimizer.step()

        # Inverse model training
        inverse_input = torch.cat([state, next_state], dim=-1)
        predicted_action = self.inverse_model(inverse_input)
        inverse_loss = F.mse_loss(predicted_action, action)

        self.inverse_optimizer.zero_grad()
        inverse_loss.backward()
        self.inverse_optimizer.step()

        return {
            "forward_loss": forward_loss.item(),
            "inverse_loss": inverse_loss.item(),
        }
