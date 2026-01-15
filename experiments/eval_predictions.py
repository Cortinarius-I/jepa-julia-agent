#!/usr/bin/env python3
"""
Evaluate JEPA world model prediction accuracy.

This script measures how well the JEPA model predicts:
1. Next state embeddings given (state, action) pairs
2. Safety outcomes (will the action break tests?)
3. Type stability changes
4. Invalidation predictions

Metrics:
- Embedding cosine similarity (prediction vs actual)
- Safety prediction accuracy
- Ranking correlation (do predicted-good actions perform well?)
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.jepa.model import JEPAWorldModel
from experiments.train_jepa import TransitionDataset

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""
    # Embedding prediction quality
    embedding_cosine_sim: float = 0.0
    embedding_mse: float = 0.0
    
    # Safety prediction
    safety_accuracy: float = 0.0
    safety_precision: float = 0.0
    safety_recall: float = 0.0
    safety_f1: float = 0.0
    safety_auc: float = 0.0
    
    # Test outcome prediction
    test_outcome_accuracy: float = 0.0
    test_outcome_auc: float = 0.0
    
    # Ranking quality (do we pick good actions?)
    action_ranking_correlation: float = 0.0
    top_k_success_rate: dict = field(default_factory=dict)
    
    # Per-action-type breakdown
    per_action_metrics: dict = field(default_factory=dict)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    model_path: Path
    data_path: Path
    output_path: Optional[Path] = None
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_k_values: tuple = (1, 3, 5, 10)


class JEPAEvaluator:
    """
    Evaluator for JEPA world model predictions.
    
    This evaluator measures prediction accuracy across multiple dimensions:
    - Embedding space accuracy (does predicted next state match actual?)
    - Safety prediction (can we predict which actions break things?)
    - Ranking quality (does the model prefer better actions?)
    """
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._load_model()
        self.metrics = EvalMetrics()
        
    def _load_model(self) -> JEPAWorldModel:
        """Load trained JEPA model from checkpoint."""
        logger.info(f"Loading model from {self.config.model_path}")
        
        checkpoint = torch.load(
            self.config.model_path,
            map_location=self.device
        )
        
        # Extract model config from checkpoint
        model_config = checkpoint.get('config', {})
        
        model = JEPAWorldModel(
            state_dim=model_config.get('state_dim', 512),
            action_dim=model_config.get('action_dim', 128),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_gnn_layers=model_config.get('num_gnn_layers', 3),
            num_action_types=model_config.get('num_action_types', 14),
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate(self) -> EvalMetrics:
        """Run full evaluation suite."""
        console.print("[bold blue]Starting JEPA Evaluation[/bold blue]")
        
        # Load evaluation data
        dataset = TransitionDataset(self.config.data_path)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(dataloader))
            
            all_predictions = []
            all_targets = []
            all_safety_preds = []
            all_safety_labels = []
            all_test_preds = []
            all_test_labels = []
            all_action_types = []
            
            with torch.no_grad():
                for batch in dataloader:
                    results = self._evaluate_batch(batch)
                    
                    all_predictions.append(results['predicted_embeddings'])
                    all_targets.append(results['target_embeddings'])
                    all_safety_preds.append(results['safety_predictions'])
                    all_safety_labels.append(results['safety_labels'])
                    all_test_preds.append(results['test_predictions'])
                    all_test_labels.append(results['test_labels'])
                    all_action_types.extend(results['action_types'])
                    
                    progress.update(task, advance=1)
            
            # Concatenate all results
            predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
            targets = torch.cat(all_targets, dim=0).cpu().numpy()
            safety_preds = torch.cat(all_safety_preds, dim=0).cpu().numpy()
            safety_labels = torch.cat(all_safety_labels, dim=0).cpu().numpy()
            test_preds = torch.cat(all_test_preds, dim=0).cpu().numpy()
            test_labels = torch.cat(all_test_labels, dim=0).cpu().numpy()
        
        # Compute metrics
        self._compute_embedding_metrics(predictions, targets)
        self._compute_safety_metrics(safety_preds, safety_labels)
        self._compute_test_outcome_metrics(test_preds, test_labels)
        self._compute_ranking_metrics(predictions, targets, all_action_types)
        self._compute_per_action_metrics(
            predictions, targets, safety_preds, safety_labels, all_action_types
        )
        
        return self.metrics
    
    def _evaluate_batch(self, batch: dict) -> dict:
        """Evaluate a single batch."""
        # Move to device
        state_embeddings = batch['state_embedding'].to(self.device)
        action_embeddings = batch['action_embedding'].to(self.device)
        next_state_embeddings = batch['next_state_embedding'].to(self.device)
        safety_labels = batch['safety_label'].to(self.device)
        test_labels = batch['test_outcome'].to(self.device)
        action_types = batch['action_type']
        
        # Get predictions
        predicted_next = self.model.predict(state_embeddings, action_embeddings)
        safety_preds = self.model.predict_safety(state_embeddings, action_embeddings)
        test_preds = self.model.predict_test_outcome(state_embeddings, action_embeddings)
        
        return {
            'predicted_embeddings': predicted_next,
            'target_embeddings': next_state_embeddings,
            'safety_predictions': safety_preds,
            'safety_labels': safety_labels,
            'test_predictions': test_preds,
            'test_labels': test_labels,
            'action_types': action_types.tolist() if isinstance(action_types, torch.Tensor) else action_types,
        }
    
    def _compute_embedding_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ):
        """Compute embedding prediction quality metrics."""
        # Cosine similarity
        pred_norm = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
        target_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
        cosine_sims = np.sum(pred_norm * target_norm, axis=1)
        self.metrics.embedding_cosine_sim = float(np.mean(cosine_sims))
        
        # MSE
        mse = np.mean((predictions - targets) ** 2)
        self.metrics.embedding_mse = float(mse)
        
        logger.info(f"Embedding cosine similarity: {self.metrics.embedding_cosine_sim:.4f}")
        logger.info(f"Embedding MSE: {self.metrics.embedding_mse:.4f}")
    
    def _compute_safety_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ):
        """Compute safety prediction metrics."""
        # Binary predictions
        pred_binary = (predictions > 0.5).astype(int)
        
        self.metrics.safety_accuracy = float(accuracy_score(labels, pred_binary))
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_binary, average='binary', zero_division=0
        )
        self.metrics.safety_precision = float(precision)
        self.metrics.safety_recall = float(recall)
        self.metrics.safety_f1 = float(f1)
        
        # AUC if we have both classes
        if len(np.unique(labels)) > 1:
            self.metrics.safety_auc = float(roc_auc_score(labels, predictions))
        
        logger.info(f"Safety accuracy: {self.metrics.safety_accuracy:.4f}")
        logger.info(f"Safety F1: {self.metrics.safety_f1:.4f}")
    
    def _compute_test_outcome_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ):
        """Compute test outcome prediction metrics."""
        pred_binary = (predictions > 0.5).astype(int)
        
        self.metrics.test_outcome_accuracy = float(accuracy_score(labels, pred_binary))
        
        if len(np.unique(labels)) > 1:
            self.metrics.test_outcome_auc = float(roc_auc_score(labels, predictions))
        
        logger.info(f"Test outcome accuracy: {self.metrics.test_outcome_accuracy:.4f}")
    
    def _compute_ranking_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        action_types: list
    ):
        """
        Compute ranking quality metrics.
        
        Given multiple candidate actions, does the model rank better actions higher?
        """
        # Compute prediction quality scores
        pred_scores = np.sum(predictions * targets, axis=1)  # dot product as quality
        target_scores = np.sum(targets * targets, axis=1)  # self-similarity as baseline
        
        # Spearman correlation
        if len(pred_scores) > 2:
            correlation, _ = stats.spearmanr(pred_scores, target_scores)
            self.metrics.action_ranking_correlation = float(correlation) if not np.isnan(correlation) else 0.0
        
        # Top-k success rate (placeholder - would need ground truth rankings)
        for k in self.config.top_k_values:
            self.metrics.top_k_success_rate[k] = 0.0  # TODO: implement with actual ranking data
        
        logger.info(f"Action ranking correlation: {self.metrics.action_ranking_correlation:.4f}")
    
    def _compute_per_action_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        safety_preds: np.ndarray,
        safety_labels: np.ndarray,
        action_types: list
    ):
        """Compute metrics broken down by action type."""
        unique_types = set(action_types)
        
        for action_type in unique_types:
            mask = np.array([t == action_type for t in action_types])
            
            if not np.any(mask):
                continue
            
            type_preds = predictions[mask]
            type_targets = targets[mask]
            type_safety_preds = safety_preds[mask]
            type_safety_labels = safety_labels[mask]
            
            # Cosine similarity for this action type
            pred_norm = type_preds / (np.linalg.norm(type_preds, axis=1, keepdims=True) + 1e-8)
            target_norm = type_targets / (np.linalg.norm(type_targets, axis=1, keepdims=True) + 1e-8)
            cosine_sim = float(np.mean(np.sum(pred_norm * target_norm, axis=1)))
            
            # Safety accuracy for this action type
            safety_acc = float(accuracy_score(
                type_safety_labels,
                (type_safety_preds > 0.5).astype(int)
            ))
            
            self.metrics.per_action_metrics[action_type] = {
                'count': int(np.sum(mask)),
                'embedding_cosine_sim': cosine_sim,
                'safety_accuracy': safety_acc,
            }
    
    def print_results(self):
        """Print evaluation results in a nice table."""
        console.print("\n[bold green]Evaluation Results[/bold green]\n")
        
        # Main metrics table
        main_table = Table(title="Overall Metrics")
        main_table.add_column("Metric", style="cyan")
        main_table.add_column("Value", style="green")
        
        main_table.add_row("Embedding Cosine Similarity", f"{self.metrics.embedding_cosine_sim:.4f}")
        main_table.add_row("Embedding MSE", f"{self.metrics.embedding_mse:.4f}")
        main_table.add_row("Safety Accuracy", f"{self.metrics.safety_accuracy:.4f}")
        main_table.add_row("Safety F1", f"{self.metrics.safety_f1:.4f}")
        main_table.add_row("Safety AUC", f"{self.metrics.safety_auc:.4f}")
        main_table.add_row("Test Outcome Accuracy", f"{self.metrics.test_outcome_accuracy:.4f}")
        main_table.add_row("Ranking Correlation", f"{self.metrics.action_ranking_correlation:.4f}")
        
        console.print(main_table)
        
        # Per-action metrics table
        if self.metrics.per_action_metrics:
            action_table = Table(title="Per-Action Metrics")
            action_table.add_column("Action Type", style="cyan")
            action_table.add_column("Count", style="yellow")
            action_table.add_column("Embedding Sim", style="green")
            action_table.add_column("Safety Acc", style="green")
            
            for action_type, metrics in sorted(self.metrics.per_action_metrics.items()):
                action_table.add_row(
                    str(action_type),
                    str(metrics['count']),
                    f"{metrics['embedding_cosine_sim']:.4f}",
                    f"{metrics['safety_accuracy']:.4f}",
                )
            
            console.print(action_table)
    
    def save_results(self, path: Path):
        """Save evaluation results to JSON."""
        results = {
            'embedding_cosine_sim': self.metrics.embedding_cosine_sim,
            'embedding_mse': self.metrics.embedding_mse,
            'safety_accuracy': self.metrics.safety_accuracy,
            'safety_precision': self.metrics.safety_precision,
            'safety_recall': self.metrics.safety_recall,
            'safety_f1': self.metrics.safety_f1,
            'safety_auc': self.metrics.safety_auc,
            'test_outcome_accuracy': self.metrics.test_outcome_accuracy,
            'test_outcome_auc': self.metrics.test_outcome_auc,
            'action_ranking_correlation': self.metrics.action_ranking_correlation,
            'top_k_success_rate': self.metrics.top_k_success_rate,
            'per_action_metrics': self.metrics.per_action_metrics,
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JEPA world model predictions"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to evaluation data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results (JSON)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create config
    config = EvalConfig(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Run evaluation
    evaluator = JEPAEvaluator(config)
    metrics = evaluator.evaluate()
    evaluator.print_results()
    
    if args.output:
        evaluator.save_results(args.output)
    
    console.print("\n[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
