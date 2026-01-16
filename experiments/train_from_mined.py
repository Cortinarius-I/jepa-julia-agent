"""
Training script for JEPA using mined transitions from git history.

This script adapts the mined transitions (from scripts/mine_transitions.py)
to the format expected by the JEPA world model.

Key adaptations:
- Uses the simplified graph representation from TransitionDataset
- Maps action types from mining to model action IDs
- Provides module/dispatch graph placeholders (to be improved with Julia bridge)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

# Direct import to avoid pydantic dependency
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent / "agent" / "data"))
import transition_dataset as td
TransitionDataset = td.TransitionDataset
TransitionCollator = td.TransitionCollator
Vocabulary = td.Vocabulary
ACTION_TYPES = td.ACTION_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Simplified JEPA for Mined Data
# ============================================================================


class SimplifiedJEPA(nn.Module):
    """
    A simplified JEPA model that works with our mined transition format.

    Instead of full world state (module graph, dispatch graph, etc.),
    we use:
    - Node features from code definitions
    - Edge structure from sequential relationships
    - Action type and target embeddings

    This serves as a stepping stone until we have full Julia bridge integration.
    """

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_action_types: int = len(ACTION_TYPES),
        vocab_size: int = 10000,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.ema_decay = ema_decay

        # Graph encoder for state representation
        from torch_geometric.nn import GATConv, global_mean_pool

        self.node_proj = nn.Linear(node_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            concat = i < num_gnn_layers - 1
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=concat)
            )

        # After final GAT layer with concat=False, output is hidden_dim (not * num_heads)
        self.state_proj = nn.Linear(hidden_dim, output_dim)

        # Method sequence encoder
        self.method_embed = nn.Embedding(vocab_size, hidden_dim)
        self.method_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.method_proj = nn.Linear(hidden_dim, output_dim)

        # Action encoder
        self.action_type_embed = nn.Embedding(num_action_types, hidden_dim)
        self.action_target_embed = nn.Embedding(vocab_size, hidden_dim)
        self.action_proj = nn.Linear(hidden_dim * 2, output_dim)

        # State fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        # Predictor: (state, action) -> predicted next state
        self.predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Action type prediction head: predict what action was taken from state transition
        # This is a self-supervised auxiliary task
        self.action_type_head = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),  # takes (before_embed, after_embed)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_action_types),
        )

        # Target encoder (EMA copy)
        self.target_encoder = self._build_target_encoder(node_dim, hidden_dim, output_dim,
                                                          num_gnn_layers, num_heads, dropout, vocab_size)
        self._init_target_encoder()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Store for global pooling
        self.global_mean_pool = global_mean_pool

    def _build_target_encoder(self, node_dim, hidden_dim, output_dim, num_gnn_layers, num_heads, dropout, vocab_size):
        """Build a copy of the encoder for EMA targets."""
        from torch_geometric.nn import GATConv

        target = nn.ModuleDict({
            'node_proj': nn.Linear(node_dim, hidden_dim),
            'state_proj': nn.Linear(hidden_dim, output_dim),  # Final GAT has concat=False
            'method_embed': nn.Embedding(vocab_size, hidden_dim),
            'method_proj': nn.Linear(hidden_dim, output_dim),
        })

        # GAT layers
        target['gat_layers'] = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            concat = i < num_gnn_layers - 1
            target['gat_layers'].append(
                GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=concat)
            )

        # Transformer for methods
        target['method_encoder'] = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=2,
        )

        # Fusion
        target['fusion'] = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        return target

    def _init_target_encoder(self):
        """Initialize target encoder with context encoder weights."""
        # Copy node_proj
        self.target_encoder['node_proj'].load_state_dict(self.node_proj.state_dict())
        self.target_encoder['state_proj'].load_state_dict(self.state_proj.state_dict())
        self.target_encoder['method_embed'].load_state_dict(self.method_embed.state_dict())
        self.target_encoder['method_proj'].load_state_dict(self.method_proj.state_dict())
        self.target_encoder['method_encoder'].load_state_dict(self.method_encoder.state_dict())
        self.target_encoder['fusion'].load_state_dict(self.fusion.state_dict())

        for i, gat in enumerate(self.gat_layers):
            self.target_encoder['gat_layers'][i].load_state_dict(gat.state_dict())

        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder with EMA of context encoder."""
        decay = self.ema_decay

        # Update node_proj
        for p_t, p_c in zip(self.target_encoder['node_proj'].parameters(), self.node_proj.parameters()):
            p_t.data = decay * p_t.data + (1 - decay) * p_c.data

        for p_t, p_c in zip(self.target_encoder['state_proj'].parameters(), self.state_proj.parameters()):
            p_t.data = decay * p_t.data + (1 - decay) * p_c.data

        for p_t, p_c in zip(self.target_encoder['method_embed'].parameters(), self.method_embed.parameters()):
            p_t.data = decay * p_t.data + (1 - decay) * p_c.data

        for p_t, p_c in zip(self.target_encoder['method_proj'].parameters(), self.method_proj.parameters()):
            p_t.data = decay * p_t.data + (1 - decay) * p_c.data

        for p_t, p_c in zip(self.target_encoder['method_encoder'].parameters(), self.method_encoder.parameters()):
            p_t.data = decay * p_t.data + (1 - decay) * p_c.data

        for p_t, p_c in zip(self.target_encoder['fusion'].parameters(), self.fusion.parameters()):
            p_t.data = decay * p_t.data + (1 - decay) * p_c.data

        for i, gat in enumerate(self.gat_layers):
            for p_t, p_c in zip(self.target_encoder['gat_layers'][i].parameters(), gat.parameters()):
                p_t.data = decay * p_t.data + (1 - decay) * p_c.data

    def encode_state(self, node_features, edge_index, batch, method_ids, use_target=False):
        """Encode a state using graph and method features."""
        encoder = self.target_encoder if use_target else None

        if use_target:
            # Use target encoder
            h = encoder['node_proj'](node_features)
            h = F.relu(h)

            for gat in encoder['gat_layers']:
                h = gat(h, edge_index)
                h = F.elu(h)

            graph_embed = self.global_mean_pool(h, batch)
            graph_embed = encoder['state_proj'](graph_embed)

            # Encode method sequence
            method_embed = encoder['method_embed'](method_ids)
            method_encoded = encoder['method_encoder'](method_embed)
            method_pooled = method_encoded.mean(dim=1)
            method_pooled = encoder['method_proj'](method_pooled)

            # Fuse
            combined = torch.cat([graph_embed, method_pooled], dim=-1)
            return encoder['fusion'](combined)
        else:
            # Use context encoder
            h = self.node_proj(node_features)
            h = F.relu(h)

            for gat in self.gat_layers:
                h = gat(h, edge_index)
                h = F.elu(h)
                h = self.dropout(h)

            graph_embed = self.global_mean_pool(h, batch)
            graph_embed = self.state_proj(graph_embed)

            # Encode method sequence
            method_embed = self.method_embed(method_ids)
            method_encoded = self.method_encoder(method_embed)
            method_pooled = method_encoded.mean(dim=1)
            method_pooled = self.method_proj(method_pooled)

            # Fuse
            combined = torch.cat([graph_embed, method_pooled], dim=-1)
            return self.fusion(combined)

    def encode_action(self, action_type, action_target):
        """Encode an action."""
        type_embed = self.action_type_embed(action_type)
        target_embed = self.action_target_embed(action_target)
        combined = torch.cat([type_embed, target_embed], dim=-1)
        return self.action_proj(combined)

    def forward(self, batch: dict) -> dict:
        """
        Forward pass.

        Args:
            batch: Dictionary from TransitionCollator containing:
                - before_node_features, before_edge_index, before_batch, before_method_ids
                - after_node_features, after_edge_index, after_batch, after_method_ids
                - action_type, action_target
        """
        # Encode current state
        current_embed = self.encode_state(
            batch["before_node_features"],
            batch["before_edge_index"],
            batch["before_batch"],
            batch["before_method_ids"],
            use_target=False,
        )

        # Encode action
        action_embed = self.encode_action(
            batch["action_type"],
            batch["action_target"],
        )

        # Predict next state
        pred_input = torch.cat([current_embed, action_embed], dim=-1)
        predicted_embed = self.predictor(pred_input)

        # Encode target state (with EMA encoder, no gradients)
        with torch.no_grad():
            target_embed = self.encode_state(
                batch["after_node_features"],
                batch["after_edge_index"],
                batch["after_batch"],
                batch["after_method_ids"],
                use_target=True,
            )

        # Compute embedding loss
        embed_loss = F.mse_loss(predicted_embed, target_embed)

        # Predict action type from state transition (self-supervised auxiliary task)
        # Uses current state and predicted next state embeddings
        state_transition = torch.cat([current_embed, predicted_embed], dim=-1)
        action_type_logits = self.action_type_head(state_transition)
        action_type_loss = F.cross_entropy(action_type_logits, batch["action_type"])

        # Compute cosine similarity for monitoring
        with torch.no_grad():
            cos_sim = F.cosine_similarity(predicted_embed, target_embed, dim=-1).mean()
            action_type_acc = (action_type_logits.argmax(dim=-1) == batch["action_type"]).float().mean()

        return {
            "embed_loss": embed_loss,
            "action_type_loss": action_type_loss,
            "predicted_embed": predicted_embed,
            "target_embed": target_embed,
            "cosine_similarity": cos_sim,
            "action_type_logits": action_type_logits,
            "action_type_acc": action_type_acc,
        }


# ============================================================================
# Training Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 100
    ema_decay: float = 0.99
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    val_split: float = 0.1
    log_every: int = 10
    save_every: int = 10


# ============================================================================
# Trainer
# ============================================================================


class MinedDataTrainer:
    """Trainer for JEPA using mined transitions."""

    def __init__(
        self,
        model: SimplifiedJEPA,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = config.num_epochs * len(train_loader)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        self.global_step = 0
        self.best_val_loss = float("inf")

    def train(self, save_dir: Path) -> dict:
        """Run training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "val_loss": [], "cosine_sim": [], "action_acc": []}

        for epoch in range(self.config.num_epochs):
            train_loss, cos_sim, action_acc = self._train_epoch()
            history["train_loss"].append(train_loss)
            history["cosine_sim"].append(cos_sim)
            history["action_acc"].append(action_acc)

            if self.val_loader:
                val_loss, val_cos_sim, val_action_acc = self._validate()
                history["val_loss"].append(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(save_dir / "best.pt")

                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Cos Sim: {cos_sim:.4f} | Action Acc: {action_acc:.2%}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Cos Sim: {cos_sim:.4f} | "
                    f"Action Acc: {action_acc:.2%}"
                )

            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(save_dir / f"epoch_{epoch + 1}.pt")

        self._save_checkpoint(save_dir / "final.pt")
        return history

    def _train_epoch(self) -> tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_cos_sim = 0.0
        total_action_acc = 0.0
        num_batches = 0

        for batch in self.train_loader:
            batch = self._to_device(batch)

            result = self.model(batch)

            # Combined loss: embedding prediction + action type prediction
            loss = result["embed_loss"] + 0.1 * result["action_type_loss"]

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            self.optimizer.step()
            self.scheduler.step()

            self.model.update_target_encoder()

            total_loss += loss.item()
            total_cos_sim += result["cosine_similarity"].item()
            total_action_acc += result["action_type_acc"].item()
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every == 0:
                logger.debug(f"Step {self.global_step}: loss={loss.item():.4f}")

        return total_loss / num_batches, total_cos_sim / num_batches, total_action_acc / num_batches

    def _validate(self) -> tuple[float, float, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_cos_sim = 0.0
        total_action_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                result = self.model(batch)
                # Combined loss for validation
                loss = result["embed_loss"] + 0.1 * result["action_type_loss"]
                total_loss += loss.item()
                total_cos_sim += result["cosine_similarity"].item()
                total_action_acc += result["action_type_acc"].item()
                num_batches += 1

        return total_loss / num_batches, total_cos_sim / num_batches, total_action_acc / num_batches

    def _to_device(self, batch: dict) -> dict:
        """Move batch to device."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def _save_checkpoint(self, path: Path):
        """Save checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, path)
        logger.info(f"Saved checkpoint to {path}")


# ============================================================================
# Main
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train JEPA from mined transitions")
    parser.add_argument("transitions_path", type=Path, nargs="+",
                       help="Path(s) to transitions (JSONL or Parquet files)")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/mined"),
                       help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--vocab-path", type=Path, help="Path to pre-built vocabulary")
    parser.add_argument("--max-transitions", type=int, default=None,
                       help="Maximum transitions to use (for resource-limited training)")
    args = parser.parse_args()

    # Expand glob patterns (useful for shell expansion)
    all_paths = []
    for p in args.transitions_path:
        if "*" in str(p):
            all_paths.extend(Path(".").glob(str(p)))
        else:
            all_paths.append(p)
    args.transitions_path = all_paths

    if not args.transitions_path:
        logger.error("No transition files found!")
        return

    logger.info(f"Found {len(args.transitions_path)} transition file(s)")
    for p in args.transitions_path:
        logger.info(f"  - {p} ({p.suffix})")

    # Build or load vocabulary from first file
    if args.vocab_path and args.vocab_path.exists():
        logger.info(f"Loading vocabulary from {args.vocab_path}")
        vocab = Vocabulary.load(args.vocab_path)
    else:
        logger.info("Building vocabulary from transitions...")
        vocab = Vocabulary.build_from_transitions(args.transitions_path[0])

        # Save vocabulary
        vocab_path = args.save_dir / "vocab.json"
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)
        logger.info(f"Saved vocabulary to {vocab_path}")

    logger.info(f"Vocabulary size: {len(vocab.token_to_id)}")

    # Load datasets
    all_transitions = []
    for path in args.transitions_path:
        dataset = TransitionDataset(path, vocab)
        all_transitions.extend(dataset.transitions)
        logger.info(f"Loaded {len(dataset)} transitions from {path}")

        # Early exit if we have enough transitions
        if args.max_transitions and len(all_transitions) >= args.max_transitions:
            all_transitions = all_transitions[:args.max_transitions]
            logger.info(f"Reached max transitions limit: {args.max_transitions}")
            break

    # Create combined dataset
    combined_dataset = TransitionDataset.__new__(TransitionDataset)
    combined_dataset.transitions = all_transitions
    combined_dataset.vocab = vocab
    combined_dataset.max_nodes = 100
    combined_dataset.max_methods = 50

    logger.info(f"Total transitions: {len(combined_dataset)}")

    # Split into train/val
    val_size = int(len(combined_dataset) * args.val_split)
    train_size = len(combined_dataset) - val_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data loaders
    collator = TransitionCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Avoid multiprocessing issues
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Create model
    model = SimplifiedJEPA(
        vocab_size=len(vocab.token_to_id),
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create trainer
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
    )

    trainer = MinedDataTrainer(model, config, train_loader, val_loader)

    # Train
    history = trainer.train(args.save_dir)

    logger.info(f"Training complete!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        logger.info(f"Best val loss: {min(history['val_loss']):.4f}")
    logger.info(f"Final cosine similarity: {history['cosine_sim'][-1]:.4f}")
    logger.info(f"Final action type accuracy: {history['action_acc'][-1]:.2%}")


if __name__ == "__main__":
    main()
