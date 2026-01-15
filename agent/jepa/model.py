"""
JEPA (Joint Embedding Predictive Architecture) for code understanding.

This is the "brain" of the agent. It predicts:
- What will happen if we make a change (semantic consequences)
- Whether a change is safe (tests will pass, no type errors)
- What invalidations will occur

The key insight: we predict in EMBEDDING SPACE, not token space.
We never ask the model to generate code - only to predict outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

if TYPE_CHECKING:
    from agent.world_state import WorldStateDiff, WorldStateSnapshot


# ============================================================================
# Encoder Architectures
# ============================================================================


class GraphEncoder(nn.Module):
    """
    Encodes graph structures (module graph, dispatch graph) into embeddings.
    Uses Graph Attention Networks for structure-aware encoding.
    """

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=(i < num_layers - 1))
            )

        final_dim = hidden_dim * num_heads if num_layers > 1 else hidden_dim
        self.output_proj = nn.Linear(final_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for pooling [num_nodes]

        Returns:
            Graph embedding [batch_size, output_dim] or [1, output_dim]
        """
        h = self.node_embedding(x)
        h = F.relu(h)

        for gat in self.gat_layers:
            h = gat(h, edge_index)
            h = F.elu(h)
            h = self.dropout(h)

        # Global pooling
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)

        return self.output_proj(h)


class MethodEncoder(nn.Module):
    """
    Encodes method signatures and metadata.
    Does NOT encode code text - that's the transformer's job.
    """

    def __init__(
        self,
        type_vocab_size: int = 10000,
        type_embed_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 256,
        max_args: int = 10,
    ):
        super().__init__()

        self.type_embedding = nn.Embedding(type_vocab_size, type_embed_dim)
        self.name_embedding = nn.Embedding(type_vocab_size, type_embed_dim)

        # Encode argument types
        self.arg_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=type_embed_dim, nhead=4, batch_first=True),
            num_layers=2,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(type_embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, name_ids: torch.Tensor, arg_type_ids: torch.Tensor, return_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            name_ids: Function name token IDs [batch, 1]
            arg_type_ids: Argument type IDs [batch, max_args]
            return_type_ids: Return type ID [batch, 1]

        Returns:
            Method embedding [batch, output_dim]
        """
        name_embed = self.name_embedding(name_ids).squeeze(1)
        arg_embeds = self.type_embedding(arg_type_ids)

        # Encode argument sequence
        arg_encoding = self.arg_encoder(arg_embeds)
        arg_pooled = arg_encoding.mean(dim=1)

        combined = torch.cat([name_embed, arg_pooled], dim=-1)
        return self.output_proj(combined)


class WorldStateEncoder(nn.Module):
    """
    Encodes the full world state into a single embedding.
    This is the CONTEXT encoder in JEPA terminology.
    """

    def __init__(
        self,
        graph_dim: int = 256,
        method_dim: int = 256,
        test_dim: int = 128,
        output_dim: int = 512,
    ):
        super().__init__()

        self.module_graph_encoder = GraphEncoder(output_dim=graph_dim)
        self.dispatch_graph_encoder = GraphEncoder(output_dim=graph_dim)
        self.method_encoder = MethodEncoder(output_dim=method_dim)

        # Test state encoding
        self.test_encoder = nn.Sequential(
            nn.Linear(4, test_dim),  # pass/fail/coverage/count
            nn.ReLU(),
            nn.Linear(test_dim, test_dim),
        )

        # Combine all components
        total_dim = graph_dim * 2 + method_dim + test_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, world_state_batch: dict) -> torch.Tensor:
        """
        Encode a batch of world states.

        Args:
            world_state_batch: Dictionary containing batched graph data

        Returns:
            World state embeddings [batch_size, output_dim]
        """
        # Encode module graph
        module_embed = self.module_graph_encoder(
            world_state_batch["module_x"],
            world_state_batch["module_edge_index"],
            world_state_batch.get("module_batch"),
        )

        # Encode dispatch graph
        dispatch_embed = self.dispatch_graph_encoder(
            world_state_batch["dispatch_x"],
            world_state_batch["dispatch_edge_index"],
            world_state_batch.get("dispatch_batch"),
        )

        # Encode method table (aggregate)
        method_embed = world_state_batch["method_aggregate"]

        # Encode test state
        test_embed = self.test_encoder(world_state_batch["test_stats"])

        # Fuse all components
        combined = torch.cat([module_embed, dispatch_embed, method_embed, test_embed], dim=-1)
        return self.fusion(combined)


# ============================================================================
# Action Encoder
# ============================================================================


class ActionEncoder(nn.Module):
    """
    Encodes typed actions into embeddings.
    Actions are the atomic operations the agent can perform.
    """

    def __init__(
        self,
        num_action_types: int = 20,
        action_embed_dim: int = 64,
        method_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()

        self.action_type_embedding = nn.Embedding(num_action_types, action_embed_dim)
        self.method_encoder = MethodEncoder(output_dim=method_dim)

        self.fusion = nn.Sequential(
            nn.Linear(action_embed_dim + method_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        action_type: torch.Tensor,
        target_method: dict | None = None,
    ) -> torch.Tensor:
        """
        Encode an action.

        Args:
            action_type: Action type ID [batch]
            target_method: Optional target method info

        Returns:
            Action embedding [batch, output_dim]
        """
        type_embed = self.action_type_embedding(action_type)

        if target_method is not None:
            method_embed = self.method_encoder(
                target_method["name_ids"],
                target_method["arg_type_ids"],
                target_method["return_type_ids"],
            )
        else:
            method_embed = torch.zeros(action_type.shape[0], self.method_encoder.output_proj[-1].out_features)

        combined = torch.cat([type_embed, method_embed], dim=-1)
        return self.fusion(combined)


# ============================================================================
# JEPA Predictor
# ============================================================================


class JEPAPredictor(nn.Module):
    """
    The core JEPA predictor.

    Given:
    - Current world state embedding (context)
    - Action embedding

    Predicts:
    - Next world state embedding (in embedding space!)
    - NOT actual code or tokens

    This is what makes JEPA different from autoregressive models:
    we predict representations, not sequences.
    """

    def __init__(
        self,
        context_dim: int = 512,
        action_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()

        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Transformer for prediction
        self.predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, context_embed: torch.Tensor, action_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the next world state embedding.

        Args:
            context_embed: Current world state [batch, context_dim]
            action_embed: Action to apply [batch, action_dim]

        Returns:
            Predicted next world state [batch, output_dim]
        """
        context = self.context_proj(context_embed)
        action = self.action_proj(action_embed)

        # Stack context and action as sequence
        sequence = torch.stack([context, action], dim=1)  # [batch, 2, hidden]

        # Predict
        predicted = self.predictor(sequence)

        # Take the prediction from the action position
        output = predicted[:, 1, :]  # [batch, hidden]

        return self.output_proj(output)


# ============================================================================
# Full JEPA Model
# ============================================================================


class JEPAWorldModel(nn.Module):
    """
    The complete JEPA world model for Julia code.

    Architecture:
    - Context encoder: WorldStateSnapshot -> embedding
    - Action encoder: Action -> embedding
    - Predictor: (context, action) -> predicted next state embedding
    - Target encoder: WorldStateSnapshot -> embedding (for training targets)

    Training:
    - Predict next state embedding from (current state, action)
    - Target is the actual next state encoded by target encoder
    - Target encoder uses EMA of context encoder (like BYOL)

    Inference:
    - Given current state and proposed action, predict outcome
    - Use predicted embedding to check safety constraints
    """

    def __init__(
        self,
        context_dim: int = 512,
        action_dim: int = 256,
        predictor_hidden: int = 512,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        self.context_encoder = WorldStateEncoder(output_dim=context_dim)
        self.action_encoder = ActionEncoder(output_dim=action_dim)
        self.predictor = JEPAPredictor(
            context_dim=context_dim,
            action_dim=action_dim,
            hidden_dim=predictor_hidden,
            output_dim=context_dim,
        )

        # Target encoder (EMA of context encoder)
        self.target_encoder = WorldStateEncoder(output_dim=context_dim)
        self._init_target_encoder()

        self.ema_decay = ema_decay

        # Safety prediction head
        self.safety_head = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # safe, risky, dangerous
        )

        # Test outcome prediction head
        self.test_head = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # predicted test pass rate
        )

    def _init_target_encoder(self):
        """Initialize target encoder with context encoder weights."""
        for param_t, param_c in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            param_t.data.copy_(param_c.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder with EMA of context encoder."""
        for param_t, param_c in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            param_t.data = self.ema_decay * param_t.data + (1 - self.ema_decay) * param_c.data

    def forward(
        self,
        current_state: dict,
        action: dict,
        next_state: dict | None = None,
    ) -> dict:
        """
        Forward pass for training or inference.

        Args:
            current_state: Batched current world state
            action: Batched action
            next_state: Batched next world state (for training)

        Returns:
            Dictionary with predictions and (optionally) loss
        """
        # Encode current state and action
        context_embed = self.context_encoder(current_state)
        action_embed = self.action_encoder(
            action["type"],
            action.get("target_method"),
        )

        # Predict next state embedding
        predicted_embed = self.predictor(context_embed, action_embed)

        # Auxiliary predictions
        safety_logits = self.safety_head(predicted_embed)
        test_pred = torch.sigmoid(self.test_head(predicted_embed))

        result = {
            "predicted_embed": predicted_embed,
            "safety_logits": safety_logits,
            "test_pass_rate": test_pred,
        }

        # Compute loss if training
        if next_state is not None:
            with torch.no_grad():
                target_embed = self.target_encoder(next_state)

            # JEPA loss: predict target embedding
            embed_loss = F.mse_loss(predicted_embed, target_embed)

            # Auxiliary losses (if labels available)
            result["embed_loss"] = embed_loss
            result["target_embed"] = target_embed

        return result

    def predict_outcome(
        self,
        current_state: "WorldStateSnapshot",
        action: dict,
    ) -> dict:
        """
        Predict the outcome of an action for planning.

        Args:
            current_state: Current world state
            action: Action to evaluate

        Returns:
            Prediction including safety assessment
        """
        self.eval()
        with torch.no_grad():
            # Convert to batched format
            state_batch = self._state_to_batch(current_state)
            action_batch = self._action_to_batch(action)

            result = self.forward(state_batch, action_batch)

            safety_probs = F.softmax(result["safety_logits"], dim=-1)

            return {
                "is_safe": safety_probs[0, 0].item() > 0.5,
                "safety_probs": safety_probs[0].tolist(),
                "predicted_test_pass_rate": result["test_pass_rate"][0].item(),
                "predicted_embedding": result["predicted_embed"][0].cpu().numpy(),
            }

    def _state_to_batch(self, state: "WorldStateSnapshot") -> dict:
        """Convert a single world state to batched format."""
        # Placeholder - actual implementation converts state to tensors
        return {
            "module_x": torch.zeros(10, 128),
            "module_edge_index": torch.zeros(2, 20, dtype=torch.long),
            "dispatch_x": torch.zeros(10, 128),
            "dispatch_edge_index": torch.zeros(2, 20, dtype=torch.long),
            "method_aggregate": torch.zeros(1, 256),
            "test_stats": torch.zeros(1, 4),
        }

    def _action_to_batch(self, action: dict) -> dict:
        """Convert a single action to batched format."""
        return {
            "type": torch.tensor([action.get("type_id", 0)]),
            "target_method": None,
        }


# ============================================================================
# Training Utilities
# ============================================================================


@dataclass
class JEPATrainingConfig:
    """Configuration for JEPA training."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 100
    ema_decay: float = 0.99
    warmup_steps: int = 1000
    gradient_clip: float = 1.0


def create_jepa_model(config: JEPATrainingConfig | None = None) -> JEPAWorldModel:
    """Factory function to create a JEPA model."""
    model = JEPAWorldModel(
        context_dim=512,
        action_dim=256,
        predictor_hidden=512,
        ema_decay=config.ema_decay if config else 0.99,
    )
    return model
