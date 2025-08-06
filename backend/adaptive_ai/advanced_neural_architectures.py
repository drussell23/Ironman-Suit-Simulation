"""
Advanced Neural Architectures for Iron Man Suit AI

This module provides cutting-edge neural network architectures:
- Transformer-based attention mechanisms
- Graph Neural Networks for multi-agent coordination
- Memory-augmented neural networks
- Meta-learning architectures
- Hierarchical reinforcement learning
- Neural architecture search
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for transformer architectures."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear transformations and reshape
        Q = (
            self.w_q(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = (
            self.w_v(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.w_o(context)

        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class TransformerPolicy(nn.Module):
    """Transformer-based policy network for complex decision making."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
    ):
        super().__init__()

        self.state_embedding = nn.Linear(state_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
        )

    def forward(
        self, state: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed state and add positional encoding
        x = self.state_embedding(state)
        x = self.positional_encoding(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)

        # Global average pooling
        x = x.mean(dim=1)

        # Output heads
        action_logits = self.output_projection(x)
        value = self.value_head(x)

        return action_logits, value


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architectures."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]


class GraphConvolution(nn.Module):
    """Graph convolution layer for graph neural networks."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            output += self.bias

        return output


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for multi-agent coordination."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GraphConvolution(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(hidden_dim, hidden_dim))

        # Output layer
        self.layers.append(GraphConvolution(hidden_dim, output_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:  # Don't apply activation to output layer
                x = self.activation(x)
                x = self.dropout(x)

        return x


class NeuralTuringMachine(nn.Module):
    """Neural Turing Machine for memory-augmented learning."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        memory_size: int = 128,
        memory_dim: int = 20,
        controller_size: int = 100,
    ):
        super().__init__()

        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.controller_size = controller_size

        # Controller network (LSTM)
        self.controller = nn.LSTMCell(input_size + memory_dim, controller_size)

        # Memory
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))

        # Read and write heads
        self.read_head = ReadHead(controller_size, memory_size, memory_dim)
        self.write_head = WriteHead(controller_size, memory_size, memory_dim)

        # Output projection
        self.output_projection = nn.Linear(controller_size + memory_dim, output_size)

        # Initialize memory
        self._init_memory()

    def _init_memory(self):
        nn.init.normal_(self.memory, mean=0, std=0.1)

    def forward(
        self, x: torch.Tensor, prev_state: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        batch_size = x.size(0)

        # Initialize controller state
        if prev_state is None:
            h = torch.zeros(batch_size, self.controller_size, device=x.device)
            c = torch.zeros(batch_size, self.controller_size, device=x.device)
            read_weights = torch.zeros(batch_size, self.memory_size, device=x.device)
            write_weights = torch.zeros(batch_size, self.memory_size, device=x.device)
        else:
            h, c, read_weights, write_weights = prev_state

        # Read from memory
        read_data = self.read_head(h, self.memory, read_weights)

        # Controller input
        controller_input = torch.cat([x, read_data], dim=1)

        # Update controller
        h, c = self.controller(controller_input, (h, c))

        # Write to memory
        write_data, write_weights = self.write_head(h, self.memory, write_weights)
        self.memory.data = write_data

        # Generate output
        output = self.output_projection(torch.cat([h, read_data], dim=1))

        return output, (h, c, read_weights, write_weights)


class ReadHead(nn.Module):
    """Read head for Neural Turing Machine."""

    def __init__(self, controller_size: int, memory_size: int, memory_dim: int):
        super().__init__()

        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(controller_size, memory_size), nn.Softmax(dim=1)
        )

    def forward(
        self,
        controller_output: torch.Tensor,
        memory: torch.Tensor,
        prev_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Calculate attention weights
        attention_weights = self.attention(controller_output)

        # Read from memory
        read_data = torch.matmul(attention_weights, memory)

        return read_data


class WriteHead(nn.Module):
    """Write head for Neural Turing Machine."""

    def __init__(self, controller_size: int, memory_size: int, memory_dim: int):
        super().__init__()

        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # Write mechanism
        self.write_key = nn.Linear(controller_size, memory_dim)
        self.write_strength = nn.Linear(controller_size, 1)
        self.erase_vector = nn.Linear(controller_size, memory_dim)
        self.add_vector = nn.Linear(controller_size, memory_dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(controller_size, memory_size), nn.Softmax(dim=1)
        )

    def forward(
        self,
        controller_output: torch.Tensor,
        memory: torch.Tensor,
        prev_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate write weights
        write_weights = self.attention(controller_output)

        # Generate write data
        write_key = self.write_key(controller_output)
        write_strength = torch.sigmoid(self.write_strength(controller_output))
        erase_vector = torch.sigmoid(self.erase_vector(controller_output))
        add_vector = torch.tanh(self.add_vector(controller_output))

        # Update memory
        erase_matrix = torch.matmul(
            write_weights.unsqueeze(-1), erase_vector.unsqueeze(1)
        )
        add_matrix = torch.matmul(write_weights.unsqueeze(-1), add_vector.unsqueeze(1))

        memory = memory * (1 - erase_matrix) + add_matrix

        return memory, write_weights


class MetaLearner(nn.Module):
    """Meta-learning architecture for rapid adaptation."""

    def __init__(self, model: nn.Module, alpha: float = 0.01, beta: float = 0.001):
        super().__init__()

        self.model = model
        self.alpha = alpha  # Inner loop learning rate
        self.beta = beta  # Outer loop learning rate

        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=beta)

    def adapt(
        self, support_data: Tuple[torch.Tensor, torch.Tensor], num_steps: int = 5
    ) -> nn.Module:
        """Adapt the model to new task using MAML."""
        support_x, support_y = support_data

        # Create a copy of the model for adaptation
        adapted_model = type(self.model)()
        adapted_model.load_state_dict(self.model.state_dict())

        # Inner loop adaptation
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.alpha)

        for _ in range(num_steps):
            inner_optimizer.zero_grad()
            predictions = adapted_model(support_x)
            loss = F.mse_loss(predictions, support_y)
            loss.backward()
            inner_optimizer.step()

        return adapted_model

    def meta_update(self, tasks: List[Tuple[Tuple, Tuple]]):
        """Update meta-parameters using multiple tasks."""
        meta_loss = 0.0

        for support_data, query_data in tasks:
            # Adapt to support data
            adapted_model = self.adapt(support_data)

            # Evaluate on query data
            query_x, query_y = query_data
            with torch.no_grad():
                query_predictions = adapted_model(query_x)
                task_loss = F.mse_loss(query_predictions, query_y)

            meta_loss += task_loss

        # Update meta-parameters
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()


class HierarchicalPolicy(nn.Module):
    """Hierarchical policy network for complex multi-level decision making."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_options: int = 4,
        option_dim: int = 64,
        meta_dim: int = 128,
    ):
        super().__init__()

        self.num_options = num_options
        self.option_dim = option_dim

        # Meta-controller (high-level policy)
        self.meta_controller = nn.Sequential(
            nn.Linear(state_dim, meta_dim),
            nn.ReLU(),
            nn.Linear(meta_dim, meta_dim),
            nn.ReLU(),
            nn.Linear(meta_dim, num_options),
        )

        # Option networks (low-level policies)
        self.options = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim + option_dim, option_dim),
                    nn.ReLU(),
                    nn.Linear(option_dim, option_dim),
                    nn.ReLU(),
                    nn.Linear(option_dim, action_dim),
                )
                for _ in range(num_options)
            ]
        )

        # Option termination networks
        self.termination_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim + option_dim, option_dim // 2),
                    nn.ReLU(),
                    nn.Linear(option_dim // 2, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_options)
            ]
        )

        # Option embeddings
        self.option_embeddings = nn.Parameter(torch.randn(num_options, option_dim))

    def forward(
        self, state: torch.Tensor, current_option: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = state.size(0)

        # Meta-controller output
        meta_logits = self.meta_controller(state)
        meta_probs = F.softmax(meta_logits, dim=1)

        # If no current option, sample from meta-controller
        if current_option is None:
            current_option = torch.multinomial(meta_probs, 1).squeeze(1)

        # Get option embedding
        option_embedding = self.option_embeddings[current_option]

        # Option input
        option_input = torch.cat([state, option_embedding], dim=1)

        # Option output
        option_output = self.options[current_option](option_input)

        # Termination probability
        termination_prob = self.termination_networks[current_option](option_input)

        return {
            "meta_probs": meta_probs,
            "option_output": option_output,
            "termination_prob": termination_prob,
            "current_option": current_option,
        }


class NeuralArchitectureSearch(nn.Module):
    """Neural Architecture Search for optimal network design."""

    def __init__(self, search_space: Dict[str, List], max_layers: int = 10):
        super().__init__()

        self.search_space = search_space
        self.max_layers = max_layers

        # Architecture controller (RNN)
        self.controller = nn.LSTM(
            input_size=len(search_space), hidden_size=100, num_layers=2, dropout=0.1
        )

        # Architecture decoder
        self.decoder = nn.Sequential(
            nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, len(search_space))
        )

    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a new architecture from the search space."""
        # Initialize controller
        h = torch.zeros(2, 1, 100)  # 2 layers, 1 batch, 100 hidden
        c = torch.zeros(2, 1, 100)

        architecture = {}

        for layer in range(self.max_layers):
            # Controller output
            if layer == 0:
                input_tensor = torch.zeros(1, 1, len(self.search_space))
            else:
                input_tensor = torch.tensor(
                    [list(architecture.values())], dtype=torch.float32
                )

            output, (h, c) = self.controller(input_tensor, (h, c))

            # Decode architecture
            logits = self.decoder(output.squeeze())
            probs = F.softmax(logits, dim=0)

            # Sample architecture parameters
            layer_arch = {}
            for i, (param_name, param_values) in enumerate(self.search_space.items()):
                param_idx = torch.multinomial(probs, 1).item()
                layer_arch[param_name] = param_values[param_idx % len(param_values)]

            architecture[f"layer_{layer}"] = layer_arch

        return architecture

    def evaluate_architecture(
        self,
        architecture: Dict[str, Any],
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """Evaluate an architecture and return its performance."""
        # Build model from architecture
        model = self._build_model_from_architecture(architecture)

        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_x, train_y = train_data
        val_x, val_y = val_data

        for epoch in range(10):  # Quick training
            optimizer.zero_grad()
            predictions = model(train_x)
            loss = criterion(predictions, train_y)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        with torch.no_grad():
            val_predictions = model(val_x)
            val_loss = criterion(val_predictions, val_y)

        return val_loss.item()

    def _build_model_from_architecture(self, architecture: Dict[str, Any]) -> nn.Module:
        """Build a neural network from architecture specification."""
        layers = []

        for layer_name, layer_config in architecture.items():
            layer_type = layer_config.get("type", "linear")

            if layer_type == "linear":
                layers.append(
                    nn.Linear(layer_config["input_size"], layer_config["output_size"])
                )
            elif layer_type == "conv":
                layers.append(
                    nn.Conv2d(
                        layer_config["in_channels"],
                        layer_config["out_channels"],
                        layer_config["kernel_size"],
                    )
                )
            elif layer_type == "lstm":
                layers.append(
                    nn.LSTM(layer_config["input_size"], layer_config["hidden_size"])
                )

            # Add activation
            activation = layer_config.get("activation", "relu")
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())

            # Add dropout
            if "dropout" in layer_config:
                layers.append(nn.Dropout(layer_config["dropout"]))

        return nn.Sequential(*layers)


class AttentionMechanism(nn.Module):
    """Advanced attention mechanism for complex pattern recognition."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        attention_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.attention_dim = attention_dim

        # Multi-head projections
        self.query_projection = nn.Linear(query_dim, attention_dim * num_heads)
        self.key_projection = nn.Linear(key_dim, attention_dim * num_heads)
        self.value_projection = nn.Linear(value_dim, attention_dim * num_heads)

        # Output projection
        self.output_projection = nn.Linear(attention_dim * num_heads, query_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(query_dim)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # Project to multi-head space
        Q = self.query_projection(query).view(
            batch_size, -1, self.num_heads, self.attention_dim
        )
        K = self.key_projection(key).view(
            batch_size, -1, self.num_heads, self.attention_dim
        )
        V = self.value_projection(value).view(
            batch_size, -1, self.num_heads, self.attention_dim
        )

        # Transpose for attention computation
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.attention_dim)
        )

        # Output projection
        output = self.output_projection(context)

        # Residual connection and layer normalization
        output = self.layer_norm(query + output)

        return output


class TemporalConvolutionalNetwork(nn.Module):
    """Temporal Convolutional Network for sequence modeling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        )

        # Hidden layers with residual connections
        for _ in range(num_layers - 2):
            self.layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
            )

        # Output layer
        self.layers.append(
            nn.Conv1d(hidden_dim, output_dim, kernel_size, padding=kernel_size // 2)
        )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)

        for i, layer in enumerate(self.layers):
            residual = x if i > 0 and x.size(1) == layer.out_channels else None

            x = layer(x)
            if i < len(self.layers) - 1:  # Don't apply activation to output layer
                x = self.activation(x)
                x = self.dropout(x)

            if residual is not None:
                x = x + residual

        x = x.transpose(1, 2)  # (batch_size, sequence_length, output_dim)
        return x
