"""
Integrated JEPA Training Pipeline.

This training script integrates all 5 paper-inspired recommendations:

1. Adaptive Test Generation (Agent2World)
   - Generate targeted tests per action type
   - Richer training signal than existing tests alone

2. Verifier-Guided Rejection Sampling (Agent2World)
   - Filter transitions: only keep V(τ) = 1
   - Cleaner gradients from verified data

3. Multi-View JEPA Loss (LLM-JEPA)
   - Predict between view pairs
   - Structured embeddings resist overfitting

4. Knowledge Synthesis (Agent2World)
   - Enrich context with doc retrieval
   - Better understanding of unfamiliar packages

5. Trace Prediction Auxiliary Task (CWM)
   - Predict execution traces in embedding space
   - Ground model in "what code does"

Combined loss:
    L = L_state + λ_jepa * L_multiview + λ_trace * L_trace + λ_safety * L_safety
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Internal imports
from agent.jepa import (
    JEPAWorldModel,
    JEPATrainingConfig,
    create_jepa_model,
    MultiViewJEPA,
    MultiViewJEPALoss,
    ViewType,
)
from agent.test_generator import AdaptiveTestGenerator, test_suite_to_julia
from agent.rejection_sampling import (
    TransitionVerifier,
    RejectionSamplingFilter,
    VerifiedDatasetBuilder,
    Transition,
)
from agent.knowledge_synthesis import (
    KnowledgeSynthesizer,
    PlanningContextEnricher,
)
from agent.trace_prediction import (
    JuliaTracePredictionModel,
    TracePredictionLoss,
)
from agent.world_state import WorldStateSnapshot

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IntegratedTrainingConfig:
    """Configuration for integrated training pipeline."""
    
    # Data
    data_dir: Path = Path("data/transitions")
    output_dir: Path = Path("checkpoints")
    cache_dir: Path = Path("cache")
    
    # Model dimensions
    state_dim: int = 512
    action_dim: int = 128
    hidden_dim: int = 512
    trace_dim: int = 512
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Loss weights (from papers)
    lambda_jepa: float = 1.0      # LLM-JEPA default
    lambda_trace: float = 0.5    # Auxiliary task weight
    lambda_safety: float = 0.3   # Safety prediction weight
    
    # EMA for target networks
    ema_decay: float = 0.996
    
    # Rejection sampling
    enable_rejection_sampling: bool = True
    max_invalidations: int = 10
    test_timeout_seconds: float = 60.0
    
    # Adaptive testing
    enable_adaptive_tests: bool = True
    
    # Knowledge synthesis
    enable_knowledge_synthesis: bool = True
    
    # Trace prediction
    enable_trace_prediction: bool = True
    
    # Checkpointing
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class IntegratedTransitionDataset(Dataset):
    """
    Dataset that includes all views needed for multi-view training.
    
    Each sample contains:
    - pre_state: World state before action
    - action: The action taken
    - post_state: World state after action
    - nl_goal: Natural language description (if available)
    - trace: Execution trace (if available)
    - context: Knowledge context (if available)
    """
    
    def __init__(
        self,
        data_path: Path,
        knowledge_enricher: Optional[PlanningContextEnricher] = None,
    ):
        self.data_path = data_path
        self.knowledge_enricher = knowledge_enricher
        self.samples = self._load_samples()
    
    def _load_samples(self) -> list[dict]:
        """Load samples from JSONL file."""
        samples = []
        
        if self.data_path.suffix == ".jsonl":
            with open(self.data_path) as f:
                for line in f:
                    samples.append(json.loads(line))
        elif self.data_path.is_dir():
            for file in self.data_path.glob("*.jsonl"):
                with open(file) as f:
                    for line in f:
                        samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(samples)} samples from {self.data_path}")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Base data
        result = {
            "pre_state": torch.tensor(sample["pre_state_embedding"], dtype=torch.float32),
            "post_state": torch.tensor(sample["post_state_embedding"], dtype=torch.float32),
            "action": torch.tensor(sample["action_embedding"], dtype=torch.float32),
            "action_type": sample["action_type"],
            "is_safe": sample.get("is_safe", True),
            "tests_pass": sample.get("tests_pass", True),
        }
        
        # Optional: NL goal
        if "nl_goal_embedding" in sample:
            result["nl_goal"] = torch.tensor(sample["nl_goal_embedding"], dtype=torch.float32)
        
        # Optional: Trace
        if "trace_embedding" in sample:
            result["trace"] = torch.tensor(sample["trace_embedding"], dtype=torch.float32)
        
        # Optional: Knowledge context
        if "context_embedding" in sample:
            result["context"] = torch.tensor(sample["context_embedding"], dtype=torch.float32)
        
        return result


# ---------------------------------------------------------------------------
# Combined Model
# ---------------------------------------------------------------------------


class IntegratedJEPAModel(nn.Module):
    """
    Integrated model combining all components.
    
    Components:
    1. JEPAWorldModel: Core state prediction
    2. MultiViewJEPA: Multi-view alignment
    3. TracePredictionModel: Trace prediction auxiliary task
    4. Safety/Test heads: Predict action outcomes
    """
    
    def __init__(self, config: IntegratedTrainingConfig):
        super().__init__()
        
        self.config = config
        
        # 1. Core JEPA world model
        self.world_model = create_jepa_model(JEPATrainingConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            ema_decay=config.ema_decay,
        ))
        
        # 2. Multi-view JEPA (Recommendation #3)
        if config.lambda_jepa > 0:
            self.multi_view = MultiViewJEPA(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.state_dim,
                ema_decay=config.ema_decay,
            )
        else:
            self.multi_view = None
        
        # 3. Trace prediction (Recommendation #5)
        if config.enable_trace_prediction:
            self.trace_model = JuliaTracePredictionModel(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                trace_dim=config.trace_dim,
                hidden_dim=config.hidden_dim,
            )
        else:
            self.trace_model = None
        
        # 4. Safety prediction head
        self.safety_head = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # 5. Test outcome prediction head
        self.test_head = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        pre_state: torch.Tensor,
        action: torch.Tensor,
        post_state: Optional[torch.Tensor] = None,
        nl_goal: Optional[torch.Tensor] = None,
        trace: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass computing all predictions.
        
        Returns dict with:
        - predicted_state: Predicted next state
        - safety_prob: Safety prediction
        - test_prob: Test pass prediction
        - multiview_losses: Per-pair JEPA losses (if enabled)
        - trace_pred: Predicted trace embedding (if enabled)
        """
        results = {}
        
        # 1. Core state prediction
        predicted_state = self.world_model.predict_state(pre_state, action)
        results["predicted_state"] = predicted_state
        
        # 2. Safety and test predictions
        results["safety_prob"] = self.safety_head(predicted_state)
        results["test_prob"] = self.test_head(predicted_state)
        
        # 3. Multi-view JEPA losses
        if self.multi_view is not None:
            views = {ViewType.PRE_STATE: pre_state}
            
            if post_state is not None:
                views[ViewType.POST_STATE] = post_state
            
            if nl_goal is not None:
                views[ViewType.NL_GOAL] = nl_goal
            
            # Action as semantic action view
            views[ViewType.SEMANTIC_ACTION] = action
            
            # Predicted state as consequence view
            views[ViewType.CONSEQUENCE] = predicted_state.detach()
            
            multiview_losses = self.multi_view(views)
            results["multiview_losses"] = multiview_losses
        
        # 4. Trace prediction
        if self.trace_model is not None and trace is not None:
            trace_pred = self.trace_model.predict_trace_embedding(pre_state, action)
            results["trace_pred"] = trace_pred
            results["trace_target"] = trace
        
        return results
    
    @torch.no_grad()
    def update_ema(self):
        """Update EMA target networks."""
        self.world_model.update_target()
        if self.multi_view is not None:
            self.multi_view.update_target_encoders()


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------


class IntegratedLoss(nn.Module):
    """
    Combined loss function.
    
    L = L_state + λ_jepa * L_multiview + λ_trace * L_trace + λ_safety * L_safety
    """
    
    def __init__(self, config: IntegratedTrainingConfig):
        super().__init__()
        
        self.config = config
        
        # State prediction loss (cosine similarity)
        self.state_loss_fn = lambda pred, target: 1.0 - nn.functional.cosine_similarity(
            pred, target, dim=-1
        ).mean()
        
        # Binary losses
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        model_outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}
        
        # 1. State prediction loss
        state_loss = self.state_loss_fn(
            model_outputs["predicted_state"],
            targets["post_state"]
        )
        losses["state_loss"] = state_loss
        
        # 2. Multi-view JEPA loss
        if "multiview_losses" in model_outputs:
            jepa_loss = model_outputs["multiview_losses"]["total"]
            losses["jepa_loss"] = jepa_loss
        else:
            jepa_loss = torch.tensor(0.0, device=state_loss.device)
            losses["jepa_loss"] = jepa_loss
        
        # 3. Trace prediction loss
        if "trace_pred" in model_outputs and "trace_target" in model_outputs:
            trace_loss = self.state_loss_fn(
                model_outputs["trace_pred"],
                model_outputs["trace_target"]
            )
            losses["trace_loss"] = trace_loss
        else:
            trace_loss = torch.tensor(0.0, device=state_loss.device)
            losses["trace_loss"] = trace_loss
        
        # 4. Safety prediction loss
        if "is_safe" in targets:
            safety_loss = self.bce_loss(
                model_outputs["safety_prob"].squeeze(),
                targets["is_safe"].float()
            )
            losses["safety_loss"] = safety_loss
        else:
            safety_loss = torch.tensor(0.0, device=state_loss.device)
            losses["safety_loss"] = safety_loss
        
        # 5. Test prediction loss
        if "tests_pass" in targets:
            test_loss = self.bce_loss(
                model_outputs["test_prob"].squeeze(),
                targets["tests_pass"].float()
            )
            losses["test_loss"] = test_loss
        else:
            test_loss = torch.tensor(0.0, device=state_loss.device)
            losses["test_loss"] = test_loss
        
        # Combined loss
        total_loss = (
            state_loss +
            self.config.lambda_jepa * jepa_loss +
            self.config.lambda_trace * trace_loss +
            self.config.lambda_safety * (safety_loss + test_loss)
        )
        losses["total_loss"] = total_loss
        
        return losses


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class IntegratedTrainer:
    """
    Trainer that integrates all components.
    """
    
    def __init__(
        self,
        config: IntegratedTrainingConfig,
        model: IntegratedJEPAModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        self.loss_fn = IntegratedLoss(config)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.num_epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_steps / total_steps,
        )
        
        # Logging
        self.writer = SummaryWriter(config.output_dir / "logs")
        self.global_step = 0
        
        # Create output dirs
        config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Epoch {epoch}", total=len(self.train_loader))
            
            for batch in self.train_loader:
                # Move to device
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward
                outputs = self.model(
                    pre_state=batch["pre_state"],
                    action=batch["action"],
                    post_state=batch.get("post_state"),
                    nl_goal=batch.get("nl_goal"),
                    trace=batch.get("trace"),
                )
                
                # Compute losses
                losses = self.loss_fn(outputs, batch)
                
                # Backward
                self.optimizer.zero_grad()
                losses["total_loss"].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                
                # Step
                self.optimizer.step()
                self.scheduler.step()
                
                # Update EMA targets
                self.model.update_ema()
                
                # Logging
                for name, value in losses.items():
                    if name not in epoch_losses:
                        epoch_losses[name] = []
                    epoch_losses[name].append(value.item())
                    self.writer.add_scalar(f"train/{name}", value.item(), self.global_step)
                
                self.global_step += 1
                progress.update(task, advance=1)
                
                # Periodic eval and checkpoint
                if self.global_step % self.config.eval_every_n_steps == 0:
                    if self.val_loader is not None:
                        val_losses = self.validate()
                        for name, value in val_losses.items():
                            self.writer.add_scalar(f"val/{name}", value, self.global_step)
                
                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint()
        
        # Return mean losses
        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}
    
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validation pass."""
        self.model.eval()
        val_losses = {}
        
        for batch in self.val_loader:
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            outputs = self.model(
                pre_state=batch["pre_state"],
                action=batch["action"],
                post_state=batch.get("post_state"),
                nl_goal=batch.get("nl_goal"),
                trace=batch.get("trace"),
            )
            
            losses = self.loss_fn(outputs, batch)
            
            for name, value in losses.items():
                if name not in val_losses:
                    val_losses[name] = []
                val_losses[name].append(value.item())
        
        return {k: sum(v) / len(v) for k, v in val_losses.items()}
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        path = self.config.output_dir / f"checkpoint_{self.global_step}.pt"
        torch.save({
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def train(self) -> dict[str, list[float]]:
        """Full training loop."""
        history = {}
        
        console.print("[bold blue]Starting Integrated JEPA Training[/bold blue]")
        console.print(f"Device: {self.config.device}")
        console.print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            console.print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            start_time = time.time()
            
            epoch_losses = self.train_epoch(epoch)
            
            elapsed = time.time() - start_time
            
            # Log
            console.print(f"\n[bold]Epoch {epoch}/{self.config.num_epochs}[/bold] ({elapsed:.1f}s)")
            for name, value in epoch_losses.items():
                console.print(f"  {name}: {value:.4f}")
                if name not in history:
                    history[name] = []
                history[name].append(value)
            
            # Validation
            if self.val_loader is not None:
                val_losses = self.validate()
                console.print("[bold]Validation:[/bold]")
                for name, value in val_losses.items():
                    console.print(f"  {name}: {value:.4f}")
        
        # Final checkpoint
        self.save_checkpoint()
        
        return history


# ---------------------------------------------------------------------------
# Data Preparation Pipeline
# ---------------------------------------------------------------------------


def prepare_training_data(
    config: IntegratedTrainingConfig,
    raw_transitions_path: Path,
    states_path: Path,
    julia_bridge: Optional[object] = None,
) -> Path:
    """
    Prepare training data using rejection sampling.
    
    This implements Recommendations #1 and #2:
    1. Generate adaptive tests per action
    2. Filter to only valid transitions
    """
    console.print("[bold]Preparing training data...[/bold]")
    
    # Load raw transitions
    console.print(f"Loading from {raw_transitions_path}")
    transitions = []
    with open(raw_transitions_path) as f:
        for line in f:
            data = json.loads(line)
            transitions.append(Transition(
                state_hash=data["state_hash"],
                action_type=data["action_type"],
                action_target=data["action_target"],
                action_params=data.get("action_params", {}),
                next_state_hash=data["next_state_hash"],
            ))
    
    console.print(f"Loaded {len(transitions)} raw transitions")
    
    # Load states
    states = {}
    with open(states_path) as f:
        for line in f:
            data = json.loads(line)
            states[data["hash"]] = WorldStateSnapshot(**data)
    
    console.print(f"Loaded {len(states)} states")
    
    if config.enable_rejection_sampling and julia_bridge is not None:
        # Create verifier with adaptive test generation
        test_generator = AdaptiveTestGenerator() if config.enable_adaptive_tests else None
        
        verifier = TransitionVerifier(
            julia_bridge=julia_bridge,
            test_generator=test_generator,
            max_invalidations=config.max_invalidations,
            test_timeout_seconds=config.test_timeout_seconds,
        )
        
        # Create filter
        filter = RejectionSamplingFilter(
            verifier=verifier,
            cache_dir=config.cache_dir / "verification",
        )
        
        # Build verified dataset
        builder = VerifiedDatasetBuilder(
            filter=filter,
            output_dir=config.data_dir / "verified",
        )
        
        splits = builder.build(transitions, states)
        
        console.print(f"[green]Verified dataset:[/green]")
        console.print(f"  Train: {len(splits['train'])}")
        console.print(f"  Val: {len(splits['val'])}")
        console.print(f"  Test: {len(splits['test'])}")
        
        return config.data_dir / "verified"
    else:
        console.print("[yellow]Skipping rejection sampling (no Julia bridge)[/yellow]")
        return raw_transitions_path


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated JEPA Training")
    parser.add_argument("--data-dir", type=Path, default=Path("data/transitions"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-jepa", type=float, default=1.0)
    parser.add_argument("--lambda-trace", type=float, default=0.5)
    parser.add_argument("--disable-multiview", action="store_true")
    parser.add_argument("--disable-trace", action="store_true")
    args = parser.parse_args()
    
    # Config
    config = IntegratedTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_jepa=0.0 if args.disable_multiview else args.lambda_jepa,
        lambda_trace=0.0 if args.disable_trace else args.lambda_trace,
        enable_trace_prediction=not args.disable_trace,
    )
    
    # Dataset
    train_data = IntegratedTransitionDataset(config.data_dir / "train.jsonl")
    val_data = IntegratedTransitionDataset(config.data_dir / "val.jsonl")
    
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    # Model
    model = IntegratedJEPAModel(config)
    
    # Train
    trainer = IntegratedTrainer(config, model, train_loader, val_loader)
    history = trainer.train()
    
    console.print("[bold green]Training complete![/bold green]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
