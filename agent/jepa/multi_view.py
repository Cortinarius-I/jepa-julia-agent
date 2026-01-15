"""
Multi-View JEPA Training for Code World Modeling.

Implements the LLM-JEPA insight: training to predict between multiple views
of the same underlying change creates structured embeddings that resist
overfitting—critical when Julia training data is limited.

View pairs we train on:
1. Pre-state ↔ Post-state (state transition prediction)
2. NL goal ↔ Action sequence (intent-to-action alignment)
3. AST diff ↔ Semantic action (syntax-semantics alignment)
4. Action embedding ↔ Consequence embedding (forward model)

Loss function from LLM-JEPA:
    L = L_pred + λ × d(Pred(Enc(View_A)), Enc(View_B))

The JEPA term constrains the mapping between views to a narrow subspace,
creating near-linear transformations that generalize better.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# View Types
# ---------------------------------------------------------------------------


class ViewType(Enum):
    """Types of views for multi-view JEPA training."""
    PRE_STATE = "pre_state"
    POST_STATE = "post_state"
    NL_GOAL = "nl_goal"
    ACTION_SEQUENCE = "action_sequence"
    AST_DIFF = "ast_diff"
    SEMANTIC_ACTION = "semantic_action"
    CONSEQUENCE = "consequence"


@dataclass
class ViewPair:
    """A pair of views for JEPA training."""
    view_a: ViewType
    view_b: ViewType
    weight: float = 1.0  # Weight in combined loss


# Default view pairs to train on
DEFAULT_VIEW_PAIRS = [
    ViewPair(ViewType.PRE_STATE, ViewType.POST_STATE, weight=1.0),
    ViewPair(ViewType.NL_GOAL, ViewType.ACTION_SEQUENCE, weight=0.5),
    ViewPair(ViewType.SEMANTIC_ACTION, ViewType.CONSEQUENCE, weight=1.0),
]


# ---------------------------------------------------------------------------
# View Encoders
# ---------------------------------------------------------------------------


class ViewEncoder(nn.Module):
    """
    Encoder for a specific view type.
    
    Each view type may have a different input format, but all encoders
    produce embeddings in the same shared space.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class NLGoalEncoder(nn.Module):
    """
    Encoder for natural language goals.
    
    Uses a pre-trained language model backbone with a projection head.
    """
    
    def __init__(
        self,
        backbone_dim: int = 768,  # BERT-base hidden size
        output_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        # We'll use the [CLS] token embedding from a language model
        # In practice, this would be a HuggingFace model
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        
        self.freeze_backbone = freeze_backbone
    
    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings: Pre-computed LM embeddings [batch, backbone_dim]
        
        Returns:
            Projected embeddings [batch, output_dim]
        """
        return self.projection(embeddings)


class ActionSequenceEncoder(nn.Module):
    """
    Encoder for sequences of actions.
    
    Uses a transformer to encode variable-length action sequences.
    """
    
    def __init__(
        self,
        action_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        max_actions: int = 32,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.max_actions = max_actions
        
        # Positional encoding
        self.pos_embed = nn.Embedding(max_actions, action_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=action_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Pool and project
        self.projection = nn.Linear(action_dim, output_dim)
    
    def forward(
        self,
        action_embeddings: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            action_embeddings: [batch, seq_len, action_dim]
            mask: Padding mask [batch, seq_len]
        
        Returns:
            Sequence embedding [batch, output_dim]
        """
        batch_size, seq_len, _ = action_embeddings.shape
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=action_embeddings.device)
        action_embeddings = action_embeddings + self.pos_embed(positions)
        
        # Encode
        if mask is not None:
            # Transformer expects True for positions to ignore
            encoded = self.transformer(action_embeddings, src_key_padding_mask=mask)
        else:
            encoded = self.transformer(action_embeddings)
        
        # Pool (mean over non-masked positions)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)
        
        return self.projection(pooled)


class ASTDiffEncoder(nn.Module):
    """
    Encoder for AST diffs (tree-structured changes).
    
    Uses a tree-structured encoder to capture syntax changes.
    """
    
    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 512,
    ):
        super().__init__()
        
        # Simplified: flatten tree to sequence and use transformer
        # Full implementation would use TreeLSTM or similar
        self.encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, diff_embedding: Tensor) -> Tensor:
        """
        Args:
            diff_embedding: Pre-computed diff embedding [batch, node_dim]
        
        Returns:
            Encoded diff [batch, output_dim]
        """
        return self.encoder(diff_embedding)


# ---------------------------------------------------------------------------
# JEPA Predictor
# ---------------------------------------------------------------------------


class JEPAPredictor(nn.Module):
    """
    Predictor network for JEPA.
    
    Following LLM-JEPA: "We leverage the auto-regressive nature of LLM
    and their internal self-attention to define a tied-weights predictor.
    By introducing special [PRED] tokens..."
    
    For our case, we use a simple MLP predictor that maps from one view
    embedding to another.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_d, out_d),
                nn.LayerNorm(out_d),
            ])
            if i < num_layers - 1:
                layers.append(nn.GELU())
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(x)


# ---------------------------------------------------------------------------
# Multi-View JEPA Model
# ---------------------------------------------------------------------------


class MultiViewJEPA(nn.Module):
    """
    Multi-View JEPA for code world modeling.
    
    Trains to predict between multiple view pairs, creating structured
    embeddings where the mapping between views is nearly linear.
    
    Key insight from LLM-JEPA: "Minimizing L_LLM does not implicitly
    minimize L_JEPA—indicating that it is required to add that term
    during training."
    """
    
    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 128,
        nl_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 512,
        view_pairs: Optional[list[ViewPair]] = None,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.view_pairs = view_pairs or DEFAULT_VIEW_PAIRS
        self.ema_decay = ema_decay
        
        # View encoders (online)
        self.encoders = nn.ModuleDict({
            ViewType.PRE_STATE.value: ViewEncoder(state_dim, hidden_dim, output_dim),
            ViewType.POST_STATE.value: ViewEncoder(state_dim, hidden_dim, output_dim),
            ViewType.NL_GOAL.value: NLGoalEncoder(nl_dim, output_dim),
            ViewType.ACTION_SEQUENCE.value: ActionSequenceEncoder(action_dim, hidden_dim, output_dim),
            ViewType.AST_DIFF.value: ASTDiffEncoder(64, hidden_dim, output_dim),
            ViewType.SEMANTIC_ACTION.value: ViewEncoder(action_dim, hidden_dim, output_dim),
            ViewType.CONSEQUENCE.value: ViewEncoder(state_dim, hidden_dim, output_dim),
        })
        
        # Target encoders (EMA updated, no gradients)
        self.target_encoders = nn.ModuleDict({
            ViewType.PRE_STATE.value: ViewEncoder(state_dim, hidden_dim, output_dim),
            ViewType.POST_STATE.value: ViewEncoder(state_dim, hidden_dim, output_dim),
            ViewType.NL_GOAL.value: NLGoalEncoder(nl_dim, output_dim),
            ViewType.ACTION_SEQUENCE.value: ActionSequenceEncoder(action_dim, hidden_dim, output_dim),
            ViewType.AST_DIFF.value: ASTDiffEncoder(64, hidden_dim, output_dim),
            ViewType.SEMANTIC_ACTION.value: ViewEncoder(action_dim, hidden_dim, output_dim),
            ViewType.CONSEQUENCE.value: ViewEncoder(state_dim, hidden_dim, output_dim),
        })
        
        # Initialize target encoders with online encoder weights
        for key in self.encoders.keys():
            self.target_encoders[key].load_state_dict(self.encoders[key].state_dict())
            for param in self.target_encoders[key].parameters():
                param.requires_grad = False
        
        # Predictors for each view pair
        self.predictors = nn.ModuleDict()
        for pair in self.view_pairs:
            key = f"{pair.view_a.value}_to_{pair.view_b.value}"
            self.predictors[key] = JEPAPredictor(output_dim, hidden_dim, output_dim)
    
    def encode(self, view_type: ViewType, x: Tensor, use_target: bool = False) -> Tensor:
        """Encode a view using online or target encoder."""
        encoder_dict = self.target_encoders if use_target else self.encoders
        encoder = encoder_dict[view_type.value]
        return encoder(x)
    
    def predict(self, view_a: ViewType, view_b: ViewType, x: Tensor) -> Tensor:
        """Predict view_b embedding from view_a embedding."""
        key = f"{view_a.value}_to_{view_b.value}"
        if key not in self.predictors:
            raise ValueError(f"No predictor for {view_a} -> {view_b}")
        return self.predictors[key](x)
    
    def forward(
        self,
        views: dict[ViewType, Tensor],
    ) -> dict[str, Tensor]:
        """
        Compute JEPA loss for all view pairs.
        
        Args:
            views: Dict mapping ViewType to input tensors
        
        Returns:
            Dict with 'loss' and individual pair losses
        """
        losses = {}
        total_loss = 0.0
        
        for pair in self.view_pairs:
            if pair.view_a not in views or pair.view_b not in views:
                continue
            
            # Encode view A with online encoder
            enc_a = self.encode(pair.view_a, views[pair.view_a], use_target=False)
            
            # Predict view B from view A
            pred_b = self.predict(pair.view_a, pair.view_b, enc_a)
            
            # Encode view B with target encoder (no gradients)
            with torch.no_grad():
                target_b = self.encode(pair.view_b, views[pair.view_b], use_target=True)
            
            # JEPA loss: cosine similarity (we want to maximize similarity)
            # Following LLM-JEPA: "We use cosine similarity as the metric"
            loss = 1.0 - F.cosine_similarity(pred_b, target_b, dim=-1).mean()
            
            pair_key = f"{pair.view_a.value}_to_{pair.view_b.value}"
            losses[pair_key] = loss
            total_loss = total_loss + pair.weight * loss
        
        losses["total"] = total_loss
        return losses
    
    @torch.no_grad()
    def update_target_encoders(self):
        """Update target encoders with EMA of online encoder weights."""
        for key in self.encoders.keys():
            online_params = self.encoders[key].parameters()
            target_params = self.target_encoders[key].parameters()
            
            for online_p, target_p in zip(online_params, target_params):
                target_p.data = (
                    self.ema_decay * target_p.data +
                    (1.0 - self.ema_decay) * online_p.data
                )


# ---------------------------------------------------------------------------
# Combined Training Loss
# ---------------------------------------------------------------------------


class MultiViewJEPALoss(nn.Module):
    """
    Combined loss following LLM-JEPA formulation:
    
    L = L_pred + λ × L_JEPA
    
    Where:
    - L_pred is the primary prediction loss (e.g., state transition)
    - L_JEPA is the multi-view JEPA loss
    - λ balances the two terms
    
    From LLM-JEPA: "The next token prediction capability is not hindered
    by the presence of the JEPA term... we obtain that employing LLM-JEPA
    only brings additional structure to the LLM latent space without
    altering its generative capabilities."
    """
    
    def __init__(
        self,
        jepa_model: MultiViewJEPA,
        lambda_jepa: float = 1.0,
        prediction_loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.jepa_model = jepa_model
        self.lambda_jepa = lambda_jepa
        self.prediction_loss_fn = prediction_loss_fn or nn.MSELoss()
    
    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        views: dict[ViewType, Tensor],
    ) -> dict[str, Tensor]:
        """
        Compute combined loss.
        
        Args:
            predicted: Predicted embeddings from primary model
            target: Target embeddings
            views: Dict of view tensors for JEPA loss
        
        Returns:
            Dict with 'loss', 'pred_loss', 'jepa_loss', and per-pair losses
        """
        # Primary prediction loss
        pred_loss = self.prediction_loss_fn(predicted, target)
        
        # JEPA loss
        jepa_losses = self.jepa_model(views)
        jepa_loss = jepa_losses["total"]
        
        # Combined loss
        total_loss = pred_loss + self.lambda_jepa * jepa_loss
        
        result = {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "jepa_loss": jepa_loss,
        }
        
        # Add per-pair losses
        for key, value in jepa_losses.items():
            if key != "total":
                result[f"jepa_{key}"] = value
        
        return result


# ---------------------------------------------------------------------------
# Custom Attention Mask (from LLM-JEPA paper)
# ---------------------------------------------------------------------------


def create_block_causal_mask(
    view_a_len: int,
    view_b_len: int,
    device: torch.device,
) -> Tensor:
    """
    Create block-causal attention mask for multi-view encoding.
    
    Following LLM-JEPA implementation:
    "We pack both Text and Code into a single context window, applying
    an attention mask to ensure they do not reference each other."
    
    This allows encoding both views in a single forward pass while
    keeping them independent.
    
    Args:
        view_a_len: Length of view A sequence
        view_b_len: Length of view B sequence
        device: Device for the mask
    
    Returns:
        Attention mask [seq_len, seq_len] where -inf means no attention
    """
    total_len = view_a_len + view_b_len
    mask = torch.full((total_len, total_len), float("-inf"), device=device)
    
    # Block A can attend to itself (causal)
    for i in range(view_a_len):
        for j in range(i + 1):
            mask[i, j] = 0.0
    
    # Block B can attend to itself (causal)
    for i in range(view_a_len, total_len):
        for j in range(view_a_len, i + 1):
            mask[i, j] = 0.0
    
    return mask


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------


class MultiViewJEPATrainer:
    """
    Trainer for Multi-View JEPA with proper EMA updates.
    """
    
    def __init__(
        self,
        model: MultiViewJEPA,
        loss_fn: MultiViewJEPALoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        gradient_clip: float = 1.0,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        
        self.step_count = 0
    
    def train_step(
        self,
        predicted: Tensor,
        target: Tensor,
        views: dict[ViewType, Tensor],
    ) -> dict[str, float]:
        """
        Perform a single training step.
        
        Returns:
            Dict of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        losses = self.loss_fn(predicted, target, views)
        
        # Backward pass
        losses["loss"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update target encoders with EMA
        self.model.jepa_model.update_target_encoders()
        
        # Learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.step_count += 1
        
        # Return scalar losses
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def validate(
        self,
        predicted: Tensor,
        target: Tensor,
        views: dict[ViewType, Tensor],
    ) -> dict[str, float]:
        """
        Validation step (no gradients, no EMA update).
        """
        self.model.eval()
        losses = self.loss_fn(predicted, target, views)
        return {k: v.item() for k, v in losses.items()}


# ---------------------------------------------------------------------------
# Embedding Space Analysis (from LLM-JEPA)
# ---------------------------------------------------------------------------


@torch.no_grad()
def analyze_embedding_structure(
    model: MultiViewJEPA,
    view_a_samples: Tensor,
    view_b_samples: Tensor,
    view_a_type: ViewType,
    view_b_type: ViewType,
) -> dict[str, float]:
    """
    Analyze the structure of learned embeddings.
    
    From LLM-JEPA: "If LLM-JEPA enforces structure in the representation
    space by constraining the mapping from Enc(Text) to Enc(Code) within
    a narrow subspace... the SVD decomposition should yield significantly
    smaller singular values."
    
    Returns:
        Dict with structure metrics
    """
    model.eval()
    
    # Encode both views
    enc_a = model.encode(view_a_type, view_a_samples, use_target=False)
    enc_b = model.encode(view_b_type, view_b_samples, use_target=False)
    
    # Compute difference
    diff = enc_a - enc_b
    
    # SVD analysis
    U, S, V = torch.svd(diff)
    
    # Compute metrics
    top_100_singular = S[:100].mean().item() if len(S) >= 100 else S.mean().item()
    total_singular = S.sum().item()
    rank_estimate = (S > 0.01 * S[0]).sum().item()  # Effective rank
    
    # Linear regression error (how linear is the mapping?)
    # Solve: enc_a @ X = enc_b
    X, _ = torch.lstsq(enc_b, enc_a)
    regression_error = torch.norm(enc_a @ X[:enc_a.shape[1]] - enc_b).item()
    
    return {
        "top_100_singular_mean": top_100_singular,
        "total_singular": total_singular,
        "effective_rank": rank_estimate,
        "linear_regression_error": regression_error,
    }
