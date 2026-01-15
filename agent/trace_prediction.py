"""
Julia Execution Trace Prediction (Embedding Space).

Adapts the CWM (Code World Model) approach of execution trace prediction
to our embedding-space paradigm. Instead of predicting tokens, we predict
the embedding of the next execution state.

From CWM (2025): "We believe that execution trace prediction enables
grounded reasoning about code generation and execution, without requiring
access to live execution environments."

Key differences from CWM:
1. CWM predicts tokens; we predict embeddings
2. CWM traces Python; we trace Julia
3. CWM uses line-level granularity; we use statement/method-level

Julia-specific considerations:
- Multiple dispatch: trace must capture which method is called
- Type inference: capture inferred types at each step
- JIT compilation: first call may differ from subsequent calls
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace Representation
# ---------------------------------------------------------------------------


class TraceEventType(Enum):
    """Types of events in a Julia execution trace."""
    CALL = "call"           # Function call
    RETURN = "return"       # Function return
    LINE = "line"           # Line execution
    EXCEPTION = "exception" # Exception thrown
    DISPATCH = "dispatch"   # Method dispatch decision
    ALLOCATION = "alloc"    # Memory allocation
    TYPE_INFERENCE = "type" # Type inference result


@dataclass
class LocalVariable:
    """A local variable at a trace point."""
    name: str
    type_str: str
    value_repr: str  # String representation of value
    is_changed: bool = False  # Changed since last frame


@dataclass
class TraceFrame:
    """A single frame in an execution trace."""
    event_type: TraceEventType
    
    # Location
    method_name: str
    source_file: str
    line_number: int
    
    # State
    local_variables: list[LocalVariable]
    
    # For CALL events
    argument_types: Optional[list[str]] = None
    
    # For RETURN events
    return_type: Optional[str] = None
    return_value_repr: Optional[str] = None
    
    # For DISPATCH events
    dispatched_method: Optional[str] = None
    dispatch_signature: Optional[str] = None
    
    # For TYPE_INFERENCE events
    inferred_type: Optional[str] = None
    is_type_stable: Optional[bool] = None


@dataclass
class ExecutionTrace:
    """A complete execution trace."""
    function_name: str
    input_args: list[str]
    frames: list[TraceFrame]
    
    # Outcomes
    completed: bool = True
    exception_message: Optional[str] = None
    final_return_type: Optional[str] = None
    total_allocations: int = 0


# ---------------------------------------------------------------------------
# Trace Encoding
# ---------------------------------------------------------------------------


class TraceEventEncoder(nn.Module):
    """
    Encodes a single trace event (frame) to an embedding.
    
    Captures:
    - Event type
    - Method being executed
    - Local variable state
    - Type information
    """
    
    def __init__(
        self,
        num_event_types: int = 7,
        type_vocab_size: int = 1000,
        var_name_vocab_size: int = 5000,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        max_variables: int = 20,
    ):
        super().__init__()
        
        self.max_variables = max_variables
        
        # Event type embedding
        self.event_embed = nn.Embedding(num_event_types, embed_dim)
        
        # Type embedding (for variable types, return types, etc.)
        self.type_embed = nn.Embedding(type_vocab_size, embed_dim, padding_idx=0)
        
        # Variable name embedding
        self.var_name_embed = nn.Embedding(var_name_vocab_size, embed_dim, padding_idx=0)
        
        # Variable state encoder (processes list of variables)
        self.var_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # name + type
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # Aggregation over variables
        self.var_attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # event + vars
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        event_type: Tensor,  # [batch]
        var_names: Tensor,   # [batch, max_vars]
        var_types: Tensor,   # [batch, max_vars]
        var_mask: Tensor,    # [batch, max_vars] - True for valid vars
    ) -> Tensor:
        """
        Encode a trace event.
        
        Returns:
            Event embedding [batch, output_dim]
        """
        batch_size = event_type.size(0)
        
        # Event embedding
        event_emb = self.event_embed(event_type)  # [batch, embed_dim]
        
        # Variable embeddings
        name_emb = self.var_name_embed(var_names)  # [batch, max_vars, embed_dim]
        type_emb = self.type_embed(var_types)      # [batch, max_vars, embed_dim]
        
        # Combine name and type
        var_combined = torch.cat([name_emb, type_emb], dim=-1)
        var_emb = self.var_encoder(var_combined)  # [batch, max_vars, embed_dim]
        
        # Self-attention over variables
        var_attn, _ = self.var_attention(
            var_emb, var_emb, var_emb,
            key_padding_mask=~var_mask
        )
        
        # Pool variables (mean over valid)
        mask_expanded = var_mask.unsqueeze(-1).float()
        var_pooled = (var_attn * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        # Combine event and variables
        combined = torch.cat([event_emb, var_pooled], dim=-1)
        
        return self.output_proj(combined)


class TraceSequenceEncoder(nn.Module):
    """
    Encodes a sequence of trace frames to a single embedding.
    
    Uses a transformer to capture temporal dependencies between frames.
    """
    
    def __init__(
        self,
        frame_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_frames: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.max_frames = max_frames
        
        # Positional encoding
        self.pos_embed = nn.Embedding(max_frames, frame_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=frame_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(frame_dim, output_dim)
    
    def forward(
        self,
        frame_embeddings: Tensor,  # [batch, seq_len, frame_dim]
        mask: Optional[Tensor] = None,  # [batch, seq_len]
    ) -> Tensor:
        """
        Encode a trace sequence.
        
        Returns:
            Trace embedding [batch, output_dim]
        """
        batch_size, seq_len, _ = frame_embeddings.shape
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=frame_embeddings.device)
        frame_embeddings = frame_embeddings + self.pos_embed(positions)
        
        # Transform
        if mask is not None:
            encoded = self.transformer(frame_embeddings, src_key_padding_mask=mask)
        else:
            encoded = self.transformer(frame_embeddings)
        
        # Pool (use last valid position)
        if mask is not None:
            # Find last valid index per batch
            lengths = (~mask).sum(dim=1) - 1
            batch_idx = torch.arange(batch_size, device=encoded.device)
            pooled = encoded[batch_idx, lengths]
        else:
            pooled = encoded[:, -1]
        
        return self.output_proj(pooled)


# ---------------------------------------------------------------------------
# Trace Prediction Model
# ---------------------------------------------------------------------------


class TracePredictor(nn.Module):
    """
    Predicts the next trace state embedding given current state.
    
    This is an auxiliary task that helps the JEPA model understand
    execution dynamics. Unlike CWM which predicts tokens, we predict
    embeddings in the shared representation space.
    """
    
    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 128,
        trace_dim: int = 512,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        # Combine current state and action to predict trace
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, trace_dim),
        )
    
    def forward(
        self,
        state_embedding: Tensor,
        action_embedding: Tensor,
    ) -> Tensor:
        """
        Predict trace embedding for executing action in state.
        
        Args:
            state_embedding: Current world state [batch, state_dim]
            action_embedding: Action to execute [batch, action_dim]
        
        Returns:
            Predicted trace embedding [batch, trace_dim]
        """
        combined = torch.cat([state_embedding, action_embedding], dim=-1)
        return self.predictor(combined)


class TraceStepPredictor(nn.Module):
    """
    Predicts the next frame embedding given previous frames.
    
    This enables step-by-step trace prediction, similar to CWM's
    auto-regressive trace generation.
    """
    
    def __init__(
        self,
        frame_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        
        # Causal transformer for auto-regressive prediction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=frame_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(frame_dim, frame_dim)
        
        # Causal mask
        self._causal_mask = None
    
    def forward(
        self,
        frame_embeddings: Tensor,  # [batch, seq_len, frame_dim]
        memory: Optional[Tensor] = None,  # [batch, mem_len, frame_dim]
    ) -> Tensor:
        """
        Predict next frame embedding auto-regressively.
        
        Returns:
            Predicted next frame embeddings [batch, seq_len, frame_dim]
        """
        seq_len = frame_embeddings.size(1)
        
        # Create causal mask
        if self._causal_mask is None or self._causal_mask.size(0) != seq_len:
            self._causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=frame_embeddings.device) * float('-inf'),
                diagonal=1
            )
        
        # Decode
        if memory is not None:
            decoded = self.decoder(
                frame_embeddings,
                memory,
                tgt_mask=self._causal_mask
            )
        else:
            # Self-attention only
            decoded = self.decoder(
                frame_embeddings,
                frame_embeddings,
                tgt_mask=self._causal_mask
            )
        
        return self.output_proj(decoded)


# ---------------------------------------------------------------------------
# Combined Trace Prediction Model
# ---------------------------------------------------------------------------


class JuliaTracePredictionModel(nn.Module):
    """
    Complete trace prediction model for Julia.
    
    Components:
    1. TraceEventEncoder: Encodes individual frames
    2. TraceSequenceEncoder: Encodes full traces
    3. TracePredictor: Predicts trace from (state, action)
    4. TraceStepPredictor: Auto-regressive frame prediction
    
    This serves as an auxiliary task to improve JEPA's understanding
    of Julia execution semantics.
    """
    
    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 128,
        frame_dim: int = 256,
        trace_dim: int = 512,
        hidden_dim: int = 512,
        type_vocab_size: int = 1000,
        var_vocab_size: int = 5000,
    ):
        super().__init__()
        
        self.frame_dim = frame_dim
        self.trace_dim = trace_dim
        
        # Frame encoder
        self.frame_encoder = TraceEventEncoder(
            type_vocab_size=type_vocab_size,
            var_name_vocab_size=var_vocab_size,
            output_dim=frame_dim,
        )
        
        # Sequence encoder
        self.sequence_encoder = TraceSequenceEncoder(
            frame_dim=frame_dim,
            output_dim=trace_dim,
        )
        
        # State-action to trace predictor
        self.trace_predictor = TracePredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            trace_dim=trace_dim,
            hidden_dim=hidden_dim,
        )
        
        # Auto-regressive frame predictor
        self.step_predictor = TraceStepPredictor(
            frame_dim=frame_dim,
            hidden_dim=hidden_dim,
        )
        
        # Project trace embedding back to state space
        self.trace_to_state = nn.Linear(trace_dim, state_dim)
    
    def encode_trace(
        self,
        event_types: Tensor,
        var_names: Tensor,
        var_types: Tensor,
        var_masks: Tensor,
        frame_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode a full execution trace.
        
        Args:
            event_types: [batch, seq_len]
            var_names: [batch, seq_len, max_vars]
            var_types: [batch, seq_len, max_vars]
            var_masks: [batch, seq_len, max_vars]
            frame_mask: [batch, seq_len] - True for padding
        
        Returns:
            Trace embedding [batch, trace_dim]
        """
        batch_size, seq_len = event_types.shape
        
        # Encode each frame
        frame_embeddings = []
        for t in range(seq_len):
            frame_emb = self.frame_encoder(
                event_types[:, t],
                var_names[:, t],
                var_types[:, t],
                var_masks[:, t],
            )
            frame_embeddings.append(frame_emb)
        
        frame_embeddings = torch.stack(frame_embeddings, dim=1)
        
        # Encode sequence
        return self.sequence_encoder(frame_embeddings, frame_mask)
    
    def predict_trace_embedding(
        self,
        state_embedding: Tensor,
        action_embedding: Tensor,
    ) -> Tensor:
        """
        Predict trace embedding from state and action.
        
        Returns:
            Predicted trace embedding [batch, trace_dim]
        """
        return self.trace_predictor(state_embedding, action_embedding)
    
    def predict_next_frame(
        self,
        frame_embeddings: Tensor,
        state_context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict next frame embedding auto-regressively.
        
        Args:
            frame_embeddings: Previous frames [batch, seq_len, frame_dim]
            state_context: Optional state context [batch, 1, frame_dim]
        
        Returns:
            Predicted next frames [batch, seq_len, frame_dim]
        """
        return self.step_predictor(frame_embeddings, state_context)
    
    def trace_to_state_delta(self, trace_embedding: Tensor) -> Tensor:
        """
        Convert trace embedding to state change embedding.
        
        This allows the trace prediction to inform state prediction.
        """
        return self.trace_to_state(trace_embedding)


# ---------------------------------------------------------------------------
# Auxiliary Training Loss
# ---------------------------------------------------------------------------


class TracePredictionLoss(nn.Module):
    """
    Auxiliary loss for trace prediction.
    
    Components:
    1. Trace embedding prediction loss
    2. Auto-regressive frame prediction loss
    3. (Optional) Cross-consistency with state prediction
    """
    
    def __init__(
        self,
        trace_model: JuliaTracePredictionModel,
        lambda_trace: float = 0.5,
        lambda_step: float = 0.3,
        lambda_consistency: float = 0.2,
    ):
        super().__init__()
        
        self.trace_model = trace_model
        self.lambda_trace = lambda_trace
        self.lambda_step = lambda_step
        self.lambda_consistency = lambda_consistency
    
    def forward(
        self,
        state_embedding: Tensor,
        action_embedding: Tensor,
        target_trace_embedding: Tensor,
        target_frame_embeddings: Tensor,
        next_state_embedding: Optional[Tensor] = None,
        frame_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Compute trace prediction losses.
        
        Returns:
            Dict with 'loss', 'trace_loss', 'step_loss', 'consistency_loss'
        """
        losses = {}
        
        # 1. Trace embedding prediction
        predicted_trace = self.trace_model.predict_trace_embedding(
            state_embedding, action_embedding
        )
        trace_loss = 1.0 - F.cosine_similarity(
            predicted_trace, target_trace_embedding, dim=-1
        ).mean()
        losses["trace_loss"] = trace_loss
        
        # 2. Auto-regressive frame prediction
        predicted_frames = self.trace_model.predict_next_frame(
            target_frame_embeddings[:, :-1],  # All but last
        )
        target_frames = target_frame_embeddings[:, 1:]  # All but first
        
        if frame_mask is not None:
            mask = ~frame_mask[:, 1:].unsqueeze(-1)
            step_loss = F.mse_loss(
                predicted_frames * mask,
                target_frames * mask,
                reduction='sum'
            ) / mask.sum().clamp(min=1)
        else:
            step_loss = F.mse_loss(predicted_frames, target_frames)
        losses["step_loss"] = step_loss
        
        # 3. Cross-consistency with state prediction
        if next_state_embedding is not None:
            # Trace should be consistent with state change
            trace_delta = self.trace_model.trace_to_state_delta(predicted_trace)
            state_delta = next_state_embedding - state_embedding
            
            consistency_loss = 1.0 - F.cosine_similarity(
                trace_delta, state_delta, dim=-1
            ).mean()
            losses["consistency_loss"] = consistency_loss
        else:
            losses["consistency_loss"] = torch.tensor(0.0, device=state_embedding.device)
        
        # Combined loss
        total_loss = (
            self.lambda_trace * trace_loss +
            self.lambda_step * step_loss +
            self.lambda_consistency * losses["consistency_loss"]
        )
        losses["loss"] = total_loss
        
        return losses


# ---------------------------------------------------------------------------
# Julia Tracer Interface (to be implemented with Julia)
# ---------------------------------------------------------------------------


class JuliaTracer:
    """
    Interface to Julia for collecting execution traces.
    
    This would be implemented using juliacall to run Julia code
    with tracing enabled.
    """
    
    def __init__(self, julia_bridge: Any):
        self.julia_bridge = julia_bridge
    
    def trace_function(
        self,
        function_name: str,
        args: list[Any],
        max_frames: int = 128,
    ) -> ExecutionTrace:
        """
        Trace execution of a Julia function.
        
        Args:
            function_name: Name of function to trace
            args: Arguments to pass to function
            max_frames: Maximum frames to collect
        
        Returns:
            ExecutionTrace with collected frames
        """
        # This would call into Julia's debugging/tracing infrastructure
        # Placeholder implementation
        result = self.julia_bridge.call(
            "IRTools.trace_execution",
            function_name,
            args,
            max_frames=max_frames,
        )
        
        return self._parse_trace_result(result)
    
    def _parse_trace_result(self, result: dict) -> ExecutionTrace:
        """Parse Julia trace result into ExecutionTrace."""
        frames = []
        
        for frame_data in result.get("frames", []):
            # Parse local variables
            variables = [
                LocalVariable(
                    name=v["name"],
                    type_str=v["type"],
                    value_repr=v.get("value", "..."),
                    is_changed=v.get("changed", False),
                )
                for v in frame_data.get("locals", [])
            ]
            
            frame = TraceFrame(
                event_type=TraceEventType(frame_data["event"]),
                method_name=frame_data["method"],
                source_file=frame_data.get("file", ""),
                line_number=frame_data.get("line", 0),
                local_variables=variables,
                argument_types=frame_data.get("arg_types"),
                return_type=frame_data.get("return_type"),
                return_value_repr=frame_data.get("return_value"),
                dispatched_method=frame_data.get("dispatched_to"),
                inferred_type=frame_data.get("inferred_type"),
                is_type_stable=frame_data.get("type_stable"),
            )
            frames.append(frame)
        
        return ExecutionTrace(
            function_name=result.get("function", "unknown"),
            input_args=[str(a) for a in result.get("args", [])],
            frames=frames,
            completed=result.get("completed", True),
            exception_message=result.get("exception"),
            final_return_type=result.get("return_type"),
            total_allocations=result.get("allocations", 0),
        )
