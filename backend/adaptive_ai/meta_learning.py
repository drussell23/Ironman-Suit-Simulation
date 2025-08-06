"""
Meta-Learning for Iron Man Suit

This module provides meta-learning capabilities for rapid adaptation:
- Model-Agnostic Meta-Learning (MAML)
- Reptile algorithm
- Prototypical Networks
- Meta-RL for policy adaptation
- Few-shot learning
- Transfer learning
- Continual learning
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import copy
import math

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning algorithms."""

    # General parameters
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    num_tasks: int = 8
    support_size: int = 5
    query_size: int = 15

    # MAML specific
    maml_first_order: bool = False
    maml_adaptation_steps: int = 1

    # Reptile specific
    reptile_epsilon: float = 1.0

    # Prototypical Networks specific
    prototype_distance: str = "euclidean"  # 'euclidean', 'cosine', 'manhattan'

    # Meta-RL specific
    meta_rl_episodes: int = 10
    meta_rl_horizon: int = 100


class MAML(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) implementation."""

    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        super().__init__()

        self.model = model
        self.config = config

        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=config.outer_lr)

        # Task-specific optimizers (created on-demand)
        self.task_optimizers = {}

        # Training statistics
        self.meta_losses = []
        self.adaptation_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def adapt_to_task(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """Adapt the model to a specific task using MAML."""
        if num_steps is None:
            num_steps = self.config.num_inner_steps

        support_x, support_y = support_data

        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()

        # Task-specific optimizer
        task_optimizer = optim.SGD(adapted_model.parameters(), lr=self.config.inner_lr)

        # Inner loop adaptation
        for step in range(num_steps):
            task_optimizer.zero_grad()

            # Forward pass
            predictions = adapted_model(support_x)
            loss = F.mse_loss(predictions, support_y)

            # Backward pass
            loss.backward()
            task_optimizer.step()

        return adapted_model

    def meta_update(self, tasks: List[Tuple[Tuple, Tuple]]) -> Dict[str, float]:
        """Perform meta-update using multiple tasks."""
        meta_loss = 0.0
        adaptation_losses = []

        for support_data, query_data in tasks:
            # Adapt to support data
            adapted_model = self.adapt_to_task(support_data)

            # Evaluate on query data
            query_x, query_y = query_data
            adapted_model.eval()

            with torch.no_grad():
                query_predictions = adapted_model(query_x)
                task_loss = F.mse_loss(query_predictions, query_y)

            meta_loss += task_loss
            adaptation_losses.append(task_loss.item())

        # Average meta loss
        meta_loss = meta_loss / len(tasks)

        # Meta-optimization
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        # Store statistics
        self.meta_losses.append(meta_loss.item())
        self.adaptation_losses.extend(adaptation_losses)

        return {
            "meta_loss": meta_loss.item(),
            "avg_adaptation_loss": np.mean(adaptation_losses),
            "std_adaptation_loss": np.std(adaptation_losses),
        }

    def evaluate_few_shot(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, float]:
        """Evaluate few-shot learning performance."""
        # Adapt to support data
        adapted_model = self.adapt_to_task(support_data)

        # Evaluate on query data
        query_x, query_y = query_data
        adapted_model.eval()

        with torch.no_grad():
            query_predictions = adapted_model(query_x)
            mse_loss = F.mse_loss(query_predictions, query_y)
            mae_loss = F.l1_loss(query_predictions, query_y)

            # Calculate R-squared
            ss_res = torch.sum((query_y - query_predictions) ** 2)
            ss_tot = torch.sum((query_y - query_y.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

        return {
            "mse_loss": mse_loss.item(),
            "mae_loss": mae_loss.item(),
            "r_squared": r_squared.item(),
        }


class Reptile(nn.Module):
    """Reptile meta-learning algorithm."""

    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        super().__init__()

        self.model = model
        self.config = config

        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=config.outer_lr)

        # Training statistics
        self.meta_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def adapt_to_task(
        self, support_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> nn.Module:
        """Adapt the model to a specific task using Reptile."""
        support_x, support_y = support_data

        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()

        # Task-specific optimizer
        task_optimizer = optim.SGD(adapted_model.parameters(), lr=self.config.inner_lr)

        # Inner loop adaptation
        for step in range(self.config.num_inner_steps):
            task_optimizer.zero_grad()

            # Forward pass
            predictions = adapted_model(support_x)
            loss = F.mse_loss(predictions, support_y)

            # Backward pass
            loss.backward()
            task_optimizer.step()

        return adapted_model

    def reptile_update(self, tasks: List[Tuple[Tuple, Tuple]]) -> Dict[str, float]:
        """Perform Reptile meta-update."""
        meta_loss = 0.0

        for support_data, query_data in tasks:
            # Adapt to support data
            adapted_model = self.adapt_to_task(support_data)

            # Reptile update: move towards adapted parameters
            for param, adapted_param in zip(
                self.model.parameters(), adapted_model.parameters()
            ):
                param.data += self.config.reptile_epsilon * (
                    adapted_param.data - param.data
                )

            # Evaluate on query data for monitoring
            query_x, query_y = query_data
            adapted_model.eval()

            with torch.no_grad():
                query_predictions = adapted_model(query_x)
                task_loss = F.mse_loss(query_predictions, query_y)

            meta_loss += task_loss

        # Average meta loss
        meta_loss = meta_loss / len(tasks)

        # Store statistics
        self.meta_losses.append(meta_loss.item())

        return {"meta_loss": meta_loss.item(), "epsilon": self.config.reptile_epsilon}


class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for few-shot learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        config: MetaLearningConfig,
    ):
        super().__init__()

        self.config = config
        self.embedding_dim = embedding_dim

        # Embedding network
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.outer_lr)

        # Training statistics
        self.losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding network."""
        return self.embedding_net(x)

    def compute_prototypes(
        self, support_embeddings: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute class prototypes from support embeddings."""
        unique_labels = torch.unique(support_labels)
        prototypes = []

        for label in unique_labels:
            mask = support_labels == label
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def compute_distance(
        self, query_embeddings: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Compute distances between query embeddings and prototypes."""
        if self.config.prototype_distance == "euclidean":
            # Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes, p=2)
        elif self.config.prototype_distance == "cosine":
            # Cosine distance
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            distances = 1 - torch.mm(query_norm, proto_norm.t())
        elif self.config.prototype_distance == "manhattan":
            # Manhattan distance
            distances = torch.cdist(query_embeddings, prototypes, p=1)
        else:
            raise ValueError(
                f"Unknown distance metric: {self.config.prototype_distance}"
            )

        return distances

    def forward_episode(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single episode."""
        support_x, support_y = support_data
        query_x, query_y = query_data

        # Compute embeddings
        support_embeddings = self.forward(support_x)
        query_embeddings = self.forward(query_x)

        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_y)

        # Compute distances
        distances = self.compute_distance(query_embeddings, prototypes)

        # Convert distances to logits (negative distances)
        logits = -distances

        return logits, query_y

    def train_episode(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, float]:
        """Train on a single episode."""
        logits, targets = self.forward_episode(support_data, query_data)

        # Compute loss
        loss = F.cross_entropy(logits, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean()

        # Store statistics
        self.losses.append(loss.item())

        return {"loss": loss.item(), "accuracy": accuracy.item()}

    def evaluate_episode(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, float]:
        """Evaluate on a single episode."""
        self.eval()

        with torch.no_grad():
            logits, targets = self.forward_episode(support_data, query_data)

            # Compute loss
            loss = F.cross_entropy(logits, targets)

            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == targets).float().mean()

        return {"loss": loss.item(), "accuracy": accuracy.item()}


class MetaRL(nn.Module):
    """Meta-Reinforcement Learning for policy adaptation."""

    def __init__(self, state_dim: int, action_dim: int, config: MetaLearningConfig):
        super().__init__()

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2),  # Mean and log_std
        )

        # Value network
        self.value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.parameters(), lr=config.outer_lr)

        # Training statistics
        self.meta_returns = []
        self.adaptation_returns = []

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy and value networks."""
        policy_output = self.policy(state)
        mean, log_std = policy_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)

        value = self.value(state)

        return mean, log_std, value

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select action using current policy."""
        mean, log_std, _ = self.forward(state)

        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()

        log_prob = (
            torch.distributions.Normal(mean, log_std.exp()).log_prob(action).sum(-1)
        )

        return action, log_prob

    def adapt_to_task(self, task_env: Any, adaptation_steps: int) -> nn.Module:
        """Adapt policy to a specific task."""
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self)
        adapted_model.train()

        # Task-specific optimizer
        task_optimizer = optim.SGD(adapted_model.parameters(), lr=self.config.inner_lr)

        # Inner loop adaptation
        for step in range(adaptation_steps):
            # Collect adaptation data
            adaptation_data = self._collect_episode_data(task_env, adapted_model)

            # Compute adaptation loss
            adaptation_loss = self._compute_adaptation_loss(adaptation_data)

            # Update adapted model
            task_optimizer.zero_grad()
            adaptation_loss.backward()
            task_optimizer.step()

        return adapted_model

    def meta_update(self, task_envs: List[Any]) -> Dict[str, float]:
        """Perform meta-update using multiple tasks."""
        meta_returns = []
        adaptation_returns = []

        for task_env in task_envs:
            # Adapt to task
            adapted_model = self.adapt_to_task(task_env, self.config.meta_rl_episodes)

            # Evaluate adapted model
            adapted_return = self._evaluate_policy(task_env, adapted_model)
            adaptation_returns.append(adapted_return)

            # Collect meta-update data
            meta_data = self._collect_episode_data(task_env, adapted_model)
            meta_return = self._compute_episode_return(meta_data)
            meta_returns.append(meta_return)

        # Compute meta-loss
        meta_loss = -torch.mean(torch.tensor(meta_returns))

        # Meta-optimization
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        # Store statistics
        self.meta_returns.extend(meta_returns)
        self.adaptation_returns.extend(adaptation_returns)

        return {
            "meta_loss": meta_loss.item(),
            "avg_meta_return": np.mean(meta_returns),
            "avg_adaptation_return": np.mean(adaptation_returns),
        }

    def _collect_episode_data(self, env: Any, model: nn.Module) -> List[Dict]:
        """Collect episode data from environment."""
        episode_data = []
        state = env.reset()

        for step in range(self.config.meta_rl_horizon):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = model.select_action(state_tensor)

            next_state, reward, done, _ = env.step(action.squeeze(0).numpy())

            episode_data.append(
                {
                    "state": state,
                    "action": action.squeeze(0).numpy(),
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                    "log_prob": log_prob.item(),
                }
            )

            state = next_state
            if done:
                break

        return episode_data

    def _compute_adaptation_loss(self, episode_data: List[Dict]) -> torch.Tensor:
        """Compute adaptation loss from episode data."""
        states = torch.FloatTensor([data["state"] for data in episode_data])
        actions = torch.FloatTensor([data["action"] for data in episode_data])
        rewards = torch.FloatTensor([data["reward"] for data in episode_data])

        # Compute policy loss
        mean, log_std, values = self.forward(states)
        log_probs = (
            torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(-1)
        )

        # Compute advantages (simple implementation)
        returns = self._compute_returns(rewards)
        advantages = returns - values.squeeze()

        # Policy gradient loss
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)

        return policy_loss + 0.5 * value_loss

    def _compute_returns(
        self, rewards: torch.Tensor, gamma: float = 0.99
    ) -> torch.Tensor:
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns

    def _compute_episode_return(self, episode_data: List[Dict]) -> float:
        """Compute total episode return."""
        return sum(data["reward"] for data in episode_data)

    def _evaluate_policy(self, env: Any, model: nn.Module) -> float:
        """Evaluate policy on environment."""
        model.eval()
        total_return = 0.0

        with torch.no_grad():
            state = env.reset()

            for step in range(self.config.meta_rl_horizon):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _ = model.select_action(state_tensor, deterministic=True)

                next_state, reward, done, _ = env.step(action.squeeze(0).numpy())
                total_return += reward

                state = next_state
                if done:
                    break

        return total_return


class ContinualLearner(nn.Module):
    """Continual learning module to prevent catastrophic forgetting."""

    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        super().__init__()

        self.model = model
        self.config = config

        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 1000

        # Elastic Weight Consolidation (EWC) parameters
        self.ewc_lambda = 1000.0
        self.fisher_info = {}
        self.optimal_params = {}

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.outer_lr)

        # Training statistics
        self.task_performances = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def compute_fisher_information(self, task_data: Tuple[torch.Tensor, torch.Tensor]):
        """Compute Fisher information matrix for EWC."""
        self.model.train()

        task_x, task_y = task_data
        fisher_info = {}

        # Compute gradients for each parameter
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)

        # Sample multiple times to estimate Fisher information
        num_samples = 100
        for _ in range(num_samples):
            self.optimizer.zero_grad()

            predictions = self.model(task_x)
            loss = F.mse_loss(predictions, task_y)
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data**2

        # Average over samples
        for name in fisher_info:
            fisher_info[name] /= num_samples

        return fisher_info

    def update_ewc_parameters(self, task_data: Tuple[torch.Tensor, torch.Tensor]):
        """Update EWC parameters after learning a task."""
        # Compute Fisher information
        fisher_info = self.compute_fisher_information(task_data)

        # Store optimal parameters and Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
                self.fisher_info[name] = fisher_info[name]

    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        ewc_loss = 0.0

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_info:
                ewc_loss += (
                    self.fisher_info[name] * (param - self.optimal_params[name]) ** 2
                ).sum()

        return self.ewc_lambda * ewc_loss

    def learn_task(
        self, task_data: Tuple[torch.Tensor, torch.Tensor], task_id: str
    ) -> Dict[str, float]:
        """Learn a new task while preventing forgetting."""
        task_x, task_y = task_data

        # Add to replay buffer
        self.replay_buffer.append((task_x, task_y, task_id))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

        # Train on current task
        for epoch in range(10):
            self.optimizer.zero_grad()

            # Current task loss
            predictions = self.model(task_x)
            task_loss = F.mse_loss(predictions, task_y)

            # EWC regularization loss
            ewc_loss = self.compute_ewc_loss()

            # Replay buffer loss (if available)
            replay_loss = 0.0
            if len(self.replay_buffer) > 1:
                replay_batch = random.sample(
                    self.replay_buffer[:-1], min(32, len(self.replay_buffer) - 1)
                )
                for replay_x, replay_y, _ in replay_batch:
                    replay_predictions = self.model(replay_x)
                    replay_loss += F.mse_loss(replay_predictions, replay_y)
                replay_loss /= len(replay_batch)

            # Total loss
            total_loss = task_loss + ewc_loss + 0.1 * replay_loss

            total_loss.backward()
            self.optimizer.step()

        # Update EWC parameters
        self.update_ewc_parameters(task_data)

        # Evaluate performance
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(task_x)
            mse_loss = F.mse_loss(predictions, task_y)

        self.task_performances[task_id] = mse_loss.item()

        return {
            "task_loss": task_loss.item(),
            "ewc_loss": ewc_loss.item(),
            "replay_loss": replay_loss.item(),
            "total_loss": total_loss.item(),
            "task_mse": mse_loss.item(),
        }

    def evaluate_all_tasks(
        self, task_data_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """Evaluate performance on all learned tasks."""
        self.model.eval()
        performances = {}

        with torch.no_grad():
            for task_id, (task_x, task_y) in task_data_dict.items():
                predictions = self.model(task_x)
                mse_loss = F.mse_loss(predictions, task_y)
                performances[task_id] = mse_loss.item()

        return performances
