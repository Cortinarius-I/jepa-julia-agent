"""
Training script for the JEPA world model.

Training data consists of (state, action, next_state) tuples collected from:
1. Synthetic edits on real Julia repositories
2. Historical commits with semantic analysis
3. Deliberate perturbations with measured outcomes

The training objective is to predict the next state embedding from (current state, action),
using the EMA target encoder as the ground truth signal.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from agent.jepa.model import JEPAWorldModel, JEPATrainingConfig, create_jepa_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================


@dataclass
class Transition:
    """A single (state, action, next_state) transition."""

    state_before: dict
    action: dict
    state_after: dict
    metadata: dict


class TransitionDataset(Dataset):
    """Dataset of world state transitions."""

    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split

        # Load transition files
        self.transitions: list[Transition] = []
        self._load_transitions()

    def _load_transitions(self):
        """Load transitions from disk."""
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist")
            return

        for file_path in split_dir.glob("*.json"):
            with open(file_path) as f:
                data = json.load(f)
                self.transitions.append(
                    Transition(
                        state_before=data["state_before"],
                        action=data["action"],
                        state_after=data["state_after"],
                        metadata=data.get("metadata", {}),
                    )
                )

        logger.info(f"Loaded {len(self.transitions)} transitions for {self.split}")

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict:
        t = self.transitions[idx]
        return {
            "state_before": self._encode_state(t.state_before),
            "action": self._encode_action(t.action),
            "state_after": self._encode_state(t.state_after),
        }

    def _encode_state(self, state: dict) -> dict:
        """Encode a world state for the model."""
        # Placeholder - actual implementation converts to tensors
        return {
            "module_x": torch.zeros(10, 128),
            "module_edge_index": torch.zeros(2, 20, dtype=torch.long),
            "dispatch_x": torch.zeros(10, 128),
            "dispatch_edge_index": torch.zeros(2, 20, dtype=torch.long),
            "method_aggregate": torch.zeros(1, 256),
            "test_stats": torch.tensor([[
                state.get("tests", {}).get("total_passed", 0),
                state.get("tests", {}).get("total_failed", 0),
                state.get("tests", {}).get("coverage", 0.0),
                len(state.get("methods", {}).get("methods", {})),
            ]], dtype=torch.float32),
        }

    def _encode_action(self, action: dict) -> dict:
        """Encode an action for the model."""
        action_type_map = {
            "add_method": 0,
            "modify_method": 1,
            "remove_method": 2,
            "add_field": 3,
            "modify_field": 4,
            "remove_field": 5,
            "add_import": 6,
            "remove_import": 7,
            "rename_symbol": 8,
            "move_definition": 9,
            "add_test": 10,
            "modify_test": 11,
            "remove_test": 12,
        }

        action_type = action_type_map.get(action.get("type", ""), 0)

        return {
            "type": torch.tensor([action_type]),
            "target_method": None,  # Would encode method info if present
        }


def collate_transitions(batch: list[dict]) -> dict:
    """Collate a batch of transitions."""
    # Stack tensors for each component
    def stack_state(key: str):
        return {
            k: torch.cat([b[key][k] for b in batch], dim=0) if k != "module_edge_index" and k != "dispatch_edge_index"
            else torch.cat([b[key][k] for b in batch], dim=1)
            for k in batch[0][key].keys()
        }

    return {
        "state_before": stack_state("state_before"),
        "action": {
            "type": torch.cat([b["action"]["type"] for b in batch]),
            "target_method": None,
        },
        "state_after": stack_state("state_after"),
    }


# ============================================================================
# Training Loop
# ============================================================================


class JEPATrainer:
    """Trainer for the JEPA world model."""

    def __init__(
        self,
        model: JEPAWorldModel,
        config: JEPATrainingConfig,
        train_dataset: TransitionDataset,
        val_dataset: TransitionDataset | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_transitions,
            num_workers=4,
        )

        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_transitions,
                num_workers=4,
            )

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * len(self.train_loader),
        )

        self.global_step = 0

    def train(self, save_dir: str | Path) -> dict:
        """Run the full training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch()
            history["train_loss"].append(train_loss)

            if self.val_loader:
                val_loss = self._validate()
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(save_dir / "best.pt")

            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {history['val_loss'][-1] if self.val_loader else 'N/A':.4f}"
            )

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(save_dir / f"epoch_{epoch + 1}.pt")

        # Save final model
        self._save_checkpoint(save_dir / "final.pt")

        return history

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move to device
            batch = self._to_device(batch)

            # Forward pass
            result = self.model(
                batch["state_before"],
                batch["action"],
                batch["state_after"],
            )

            loss = result["embed_loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            self.optimizer.step()
            self.scheduler.step()

            # Update target encoder (EMA)
            self.model.update_target_encoder()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        return total_loss / num_batches

    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)

                result = self.model(
                    batch["state_before"],
                    batch["action"],
                    batch["state_after"],
                )

                total_loss += result["embed_loss"].item()
                num_batches += 1

        return total_loss / num_batches

    def _to_device(self, batch: dict) -> dict:
        """Move batch to device."""

        def move(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            elif isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            return x

        return move(batch)

    def _save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")


# ============================================================================
# Main
# ============================================================================


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train JEPA world model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to transition data")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Load data
    train_dataset = TransitionDataset(args.data_dir, split="train")
    val_dataset = TransitionDataset(args.data_dir, split="val")

    # Create model
    config = JEPATrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    model = create_jepa_model(config)

    # Train
    trainer = JEPATrainer(model, config, train_dataset, val_dataset)
    history = trainer.train(args.save_dir)

    logger.info(f"Training complete. Final train loss: {history['train_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
