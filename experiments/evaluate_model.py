"""
Evaluation script for trained JEPA model.

Computes various metrics to assess prediction accuracy:
- Cosine similarity between predicted and actual next states
- MSE loss distribution
- Per-action-type accuracy
- Embedding visualization (optional)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Import from training script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "data"))
import transition_dataset as td
TransitionDataset = td.TransitionDataset
TransitionCollator = td.TransitionCollator
Vocabulary = td.Vocabulary
ACTION_TYPES = td.ACTION_TYPES

from train_from_mined import SimplifiedJEPA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, vocab_path: Path, device: str = "cpu"):
    """Load trained model from checkpoint."""
    # Load vocabulary
    vocab = Vocabulary.load(vocab_path)
    logger.info(f"Loaded vocabulary with {len(vocab.token_to_id)} tokens")

    # Create model
    model = SimplifiedJEPA(vocab_size=len(vocab.token_to_id))

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    logger.info(f"  Global step: {checkpoint.get('global_step', 'N/A')}")

    return model, vocab, checkpoint


def evaluate_model(
    model: SimplifiedJEPA,
    dataloader: DataLoader,
    device: str = "cpu",
) -> dict:
    """
    Evaluate model on a dataset.

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_cos_sims = []
    all_mse_losses = []
    per_action_cos_sims = defaultdict(list)
    per_action_mse = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            result = model(batch)

            # Per-sample metrics
            predicted = result["predicted_embed"]
            target = result["target_embed"]

            # Cosine similarity per sample
            cos_sims = F.cosine_similarity(predicted, target, dim=-1)
            all_cos_sims.extend(cos_sims.cpu().numpy())

            # MSE per sample
            mse_per_sample = ((predicted - target) ** 2).mean(dim=-1)
            all_mse_losses.extend(mse_per_sample.cpu().numpy())

            # Per-action-type metrics
            action_types = batch["action_type"].cpu().numpy()
            for i, action_type in enumerate(action_types):
                per_action_cos_sims[action_type].append(cos_sims[i].item())
                per_action_mse[action_type].append(mse_per_sample[i].item())

    # Compute summary statistics
    all_cos_sims = np.array(all_cos_sims)
    all_mse_losses = np.array(all_mse_losses)

    results = {
        "num_samples": len(all_cos_sims),
        "cosine_similarity": {
            "mean": float(all_cos_sims.mean()),
            "std": float(all_cos_sims.std()),
            "min": float(all_cos_sims.min()),
            "max": float(all_cos_sims.max()),
            "median": float(np.median(all_cos_sims)),
            "percentile_25": float(np.percentile(all_cos_sims, 25)),
            "percentile_75": float(np.percentile(all_cos_sims, 75)),
        },
        "mse_loss": {
            "mean": float(all_mse_losses.mean()),
            "std": float(all_mse_losses.std()),
            "min": float(all_mse_losses.min()),
            "max": float(all_mse_losses.max()),
            "median": float(np.median(all_mse_losses)),
        },
        "per_action_type": {},
    }

    # Per-action-type results
    for action_type in sorted(per_action_cos_sims.keys()):
        action_name = ACTION_TYPES[action_type] if action_type < len(ACTION_TYPES) else f"unknown_{action_type}"
        cos_sims = np.array(per_action_cos_sims[action_type])
        mse_vals = np.array(per_action_mse[action_type])

        results["per_action_type"][action_name] = {
            "count": len(cos_sims),
            "cosine_similarity_mean": float(cos_sims.mean()),
            "cosine_similarity_std": float(cos_sims.std()),
            "mse_mean": float(mse_vals.mean()),
        }

    return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("JEPA Model Evaluation Results")
    print("=" * 60)

    print(f"\nTotal samples evaluated: {results['num_samples']}")

    print("\n--- Cosine Similarity (higher is better, max 1.0) ---")
    cos = results["cosine_similarity"]
    print(f"  Mean:      {cos['mean']:.4f} ± {cos['std']:.4f}")
    print(f"  Median:    {cos['median']:.4f}")
    print(f"  Range:     [{cos['min']:.4f}, {cos['max']:.4f}]")
    print(f"  IQR:       [{cos['percentile_25']:.4f}, {cos['percentile_75']:.4f}]")

    print("\n--- MSE Loss (lower is better) ---")
    mse = results["mse_loss"]
    print(f"  Mean:      {mse['mean']:.6f} ± {mse['std']:.6f}")
    print(f"  Median:    {mse['median']:.6f}")
    print(f"  Range:     [{mse['min']:.6f}, {mse['max']:.6f}]")

    print("\n--- Per Action Type ---")
    print(f"{'Action Type':<20} {'Count':>8} {'Cos Sim':>10} {'MSE':>12}")
    print("-" * 52)
    for action_name, stats in sorted(results["per_action_type"].items(),
                                      key=lambda x: -x[1]["count"]):
        print(f"{action_name:<20} {stats['count']:>8} "
              f"{stats['cosine_similarity_mean']:>10.4f} "
              f"{stats['mse_mean']:>12.6f}")

    # Quality assessment
    print("\n--- Quality Assessment ---")
    mean_cos = cos['mean']
    if mean_cos >= 0.95:
        quality = "Excellent"
    elif mean_cos >= 0.90:
        quality = "Very Good"
    elif mean_cos >= 0.80:
        quality = "Good"
    elif mean_cos >= 0.70:
        quality = "Fair"
    else:
        quality = "Needs Improvement"

    print(f"  Overall Quality: {quality} (cosine similarity: {mean_cos:.4f})")

    # Check for action types with lower performance
    weak_actions = []
    for action_name, stats in results["per_action_type"].items():
        if stats["cosine_similarity_mean"] < mean_cos - 0.1:
            weak_actions.append((action_name, stats["cosine_similarity_mean"]))

    if weak_actions:
        print(f"\n  Action types with lower accuracy:")
        for action, score in weak_actions:
            print(f"    - {action}: {score:.4f}")

    print("\n" + "=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained JEPA model")
    parser.add_argument("--checkpoint", type=Path,
                       default=Path("checkpoints/jepa-model-1/best.pt"),
                       help="Path to model checkpoint")
    parser.add_argument("--vocab", type=Path,
                       default=Path("checkpoints/jepa-model-1/vocab.json"),
                       help="Path to vocabulary file")
    parser.add_argument("--data", type=Path, nargs="+",
                       default=[Path("data/transitions-1")],
                       help="Path(s) to transition data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")
    args = parser.parse_args()

    # Load model
    model, vocab, checkpoint = load_model(args.checkpoint, args.vocab, args.device)

    # Find data files
    data_files = []
    for path in args.data:
        if path.is_dir():
            data_files.extend(path.glob("*.parquet"))
            data_files.extend(path.glob("*.jsonl"))
        else:
            data_files.append(path)

    if not data_files:
        logger.error("No data files found!")
        return

    logger.info(f"Found {len(data_files)} data files")

    # Load all transitions
    all_transitions = []
    for path in data_files:
        dataset = TransitionDataset(path, vocab)
        all_transitions.extend(dataset.transitions)
        logger.info(f"  {path.name}: {len(dataset)} transitions")

    # Create combined dataset
    combined_dataset = TransitionDataset.__new__(TransitionDataset)
    combined_dataset.transitions = all_transitions
    combined_dataset.vocab = vocab
    combined_dataset.max_nodes = 100
    combined_dataset.max_methods = 50

    logger.info(f"Total transitions for evaluation: {len(combined_dataset)}")

    # Create dataloader
    collator = TransitionCollator()
    dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Evaluate
    logger.info("Running evaluation...")
    results = evaluate_model(model, dataloader, args.device)

    # Print results
    print_results(results)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
