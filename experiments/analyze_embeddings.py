"""
SVD Analysis of JEPA Embedding Structure.

Evaluates the learned embedding space following LLM-JEPA methodology:
- SVD decomposition to analyze the transformation subspace
- Linearity of mappings between before/after states
- Effective rank of embedding differences
- Clustering analysis by action type

From LLM-JEPA: "If LLM-JEPA enforces structure in the representation space
by constraining the mapping from Enc(Text) to Enc(Code) within a narrow
subspace... the SVD decomposition should yield significantly smaller
singular values."
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

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
    vocab = Vocabulary.load(vocab_path)
    logger.info(f"Loaded vocabulary with {len(vocab.token_to_id)} tokens")

    model = SimplifiedJEPA(vocab_size=len(vocab.token_to_id))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    return model, vocab, checkpoint


@torch.no_grad()
def extract_embeddings(
    model: SimplifiedJEPA,
    dataloader: DataLoader,
    device: str = "cpu",
) -> dict:
    """
    Extract embeddings for all samples.

    Returns:
        Dict with:
        - before_embeds: [N, embed_dim] embeddings of before states
        - after_embeds: [N, embed_dim] embeddings of after states
        - predicted_embeds: [N, embed_dim] predicted next states
        - action_types: [N] action type indices
    """
    model.eval()

    before_embeds = []
    after_embeds = []
    predicted_embeds = []
    action_types = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Encode before state
        before_embed = model.encode_state(
            batch["before_node_features"],
            batch["before_edge_index"],
            batch["before_batch"],
            batch["before_method_ids"],
            use_target=False,
        )

        # Get action embedding and predict
        action_embed = model.encode_action(
            batch["action_type"],
            batch["action_target"],
        )
        combined = torch.cat([before_embed, action_embed], dim=-1)
        predicted_embed = model.predictor(combined)

        # Encode after state with target encoder
        after_embed = model.encode_state(
            batch["after_node_features"],
            batch["after_edge_index"],
            batch["after_batch"],
            batch["after_method_ids"],
            use_target=True,
        )

        before_embeds.append(before_embed.cpu())
        after_embeds.append(after_embed.cpu())
        predicted_embeds.append(predicted_embed.cpu())
        action_types.append(batch["action_type"].cpu())

    return {
        "before_embeds": torch.cat(before_embeds, dim=0),
        "after_embeds": torch.cat(after_embeds, dim=0),
        "predicted_embeds": torch.cat(predicted_embeds, dim=0),
        "action_types": torch.cat(action_types, dim=0),
    }


def analyze_svd(
    before_embeds: torch.Tensor,
    after_embeds: torch.Tensor,
    predicted_embeds: torch.Tensor,
) -> dict:
    """
    Perform SVD analysis on embedding differences.

    Following LLM-JEPA: analyzes whether the mapping is constrained
    to a narrow subspace (indicated by smaller singular values).
    """
    # Compute differences
    actual_diff = after_embeds - before_embeds  # True state change
    predicted_diff = predicted_embeds - before_embeds  # Model's prediction
    prediction_error = predicted_embeds - after_embeds  # Prediction error

    results = {}

    for name, diff_matrix in [
        ("actual_transition", actual_diff),
        ("predicted_transition", predicted_diff),
        ("prediction_error", prediction_error),
    ]:
        # Compute SVD
        U, S, V = torch.svd(diff_matrix)
        S = S.numpy()

        # Metrics from LLM-JEPA
        results[name] = {
            # Top singular values
            "top_1_singular": float(S[0]),
            "top_5_singular_sum": float(S[:5].sum()),
            "top_10_singular_sum": float(S[:10].sum()),
            "top_20_singular_sum": float(S[:20].sum()),
            "total_singular_sum": float(S.sum()),

            # Concentration (what fraction is in top k)
            "top_5_fraction": float(S[:5].sum() / S.sum()),
            "top_10_fraction": float(S[:10].sum() / S.sum()),
            "top_20_fraction": float(S[:20].sum() / S.sum()),

            # Effective rank (number of significant singular values)
            "effective_rank_01": int((S > 0.01 * S[0]).sum()),
            "effective_rank_001": int((S > 0.001 * S[0]).sum()),

            # Singular value decay
            "singular_values_top_20": S[:20].tolist(),
        }

    return results


def analyze_linearity(
    before_embeds: torch.Tensor,
    after_embeds: torch.Tensor,
) -> dict:
    """
    Analyze if the before->after mapping is approximately linear.

    From LLM-JEPA: "We regress the mapping between views using linear
    regression and measure the residual error."
    """
    # Solve: before_embeds @ W + b ≈ after_embeds
    # Using least squares: minimize ||before @ W - after||^2

    # Add bias term
    N = before_embeds.shape[0]
    before_with_bias = torch.cat([
        before_embeds,
        torch.ones(N, 1)
    ], dim=1)

    # Solve linear regression
    # X @ W = Y  ->  W = (X^T X)^-1 X^T Y
    try:
        XtX = before_with_bias.T @ before_with_bias
        XtY = before_with_bias.T @ after_embeds
        W = torch.linalg.solve(XtX + 1e-6 * torch.eye(XtX.shape[0]), XtY)

        # Compute predictions and errors
        after_predicted = before_with_bias @ W
        residuals = after_embeds - after_predicted

        # Metrics
        mse = (residuals ** 2).mean().item()
        mae = residuals.abs().mean().item()
        r2_per_dim = 1 - (residuals ** 2).sum(dim=0) / ((after_embeds - after_embeds.mean(dim=0)) ** 2).sum(dim=0)
        r2_mean = r2_per_dim.mean().item()

        # Cosine similarity of linear predictions
        cos_sim = F.cosine_similarity(after_predicted, after_embeds, dim=-1).mean().item()

        return {
            "mse": mse,
            "mae": mae,
            "r2_mean": r2_mean,
            "cosine_similarity": cos_sim,
            "residual_norm": float(torch.norm(residuals).item()),
            "weight_matrix_norm": float(torch.norm(W).item()),
        }
    except Exception as e:
        logger.warning(f"Linear regression failed: {e}")
        return {"error": str(e)}


def analyze_per_action_type(
    before_embeds: torch.Tensor,
    after_embeds: torch.Tensor,
    predicted_embeds: torch.Tensor,
    action_types: torch.Tensor,
) -> dict:
    """
    Analyze embedding structure per action type.
    """
    results = {}
    unique_actions = action_types.unique()

    for action_idx in unique_actions:
        mask = action_types == action_idx
        action_name = ACTION_TYPES[action_idx.item()] if action_idx.item() < len(ACTION_TYPES) else f"unknown_{action_idx.item()}"

        before = before_embeds[mask]
        after = after_embeds[mask]
        predicted = predicted_embeds[mask]

        if len(before) < 5:
            continue

        # State change magnitude
        actual_diff = (after - before).norm(dim=-1)
        predicted_diff = (predicted - before).norm(dim=-1)
        prediction_error = (predicted - after).norm(dim=-1)

        # Cosine similarity
        cos_sim = F.cosine_similarity(predicted, after, dim=-1)

        # Clustering: how tight is the distribution of changes?
        actual_changes = after - before
        change_centroid = actual_changes.mean(dim=0)
        change_variance = ((actual_changes - change_centroid) ** 2).mean().item()

        results[action_name] = {
            "count": int(mask.sum().item()),
            "actual_change_magnitude_mean": float(actual_diff.mean().item()),
            "actual_change_magnitude_std": float(actual_diff.std().item()),
            "predicted_change_magnitude_mean": float(predicted_diff.mean().item()),
            "prediction_error_mean": float(prediction_error.mean().item()),
            "cosine_similarity_mean": float(cos_sim.mean().item()),
            "cosine_similarity_std": float(cos_sim.std().item()),
            "change_variance": change_variance,
        }

    return results


def analyze_embedding_geometry(
    before_embeds: torch.Tensor,
    after_embeds: torch.Tensor,
) -> dict:
    """
    Analyze geometric properties of the embedding space.
    """
    # Norms
    before_norms = before_embeds.norm(dim=-1)
    after_norms = after_embeds.norm(dim=-1)

    # Angular distribution
    cos_before_after = F.cosine_similarity(before_embeds, after_embeds, dim=-1)

    # Pairwise distances within before states
    N = min(500, len(before_embeds))  # Subsample for efficiency
    indices = torch.randperm(len(before_embeds))[:N]
    before_sample = before_embeds[indices]
    pairwise_dists = torch.cdist(before_sample, before_sample)

    # Get upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones_like(pairwise_dists), diagonal=1).bool()
    pairwise_dists_flat = pairwise_dists[mask]

    return {
        "before_norm_mean": float(before_norms.mean().item()),
        "before_norm_std": float(before_norms.std().item()),
        "after_norm_mean": float(after_norms.mean().item()),
        "after_norm_std": float(after_norms.std().item()),
        "before_after_cosine_mean": float(cos_before_after.mean().item()),
        "before_after_cosine_std": float(cos_before_after.std().item()),
        "pairwise_distance_mean": float(pairwise_dists_flat.mean().item()),
        "pairwise_distance_std": float(pairwise_dists_flat.std().item()),
    }


def print_results(results: dict):
    """Pretty print SVD analysis results."""
    print("\n" + "=" * 70)
    print("JEPA Embedding Structure Analysis (SVD)")
    print("=" * 70)

    print(f"\nTotal samples analyzed: {results['num_samples']}")
    print(f"Embedding dimension: {results['embed_dim']}")

    # SVD Analysis
    print("\n" + "-" * 70)
    print("SVD Analysis of State Transitions")
    print("-" * 70)

    for name in ["actual_transition", "predicted_transition", "prediction_error"]:
        svd = results["svd"][name]
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Top-1 singular value:  {svd['top_1_singular']:.4f}")
        print(f"  Top-5 fraction:        {svd['top_5_fraction']:.2%}")
        print(f"  Top-10 fraction:       {svd['top_10_fraction']:.2%}")
        print(f"  Effective rank (1%):   {svd['effective_rank_01']}")
        print(f"  Effective rank (0.1%): {svd['effective_rank_001']}")

    # Linearity Analysis
    print("\n" + "-" * 70)
    print("Linearity Analysis (Linear Regression: before -> after)")
    print("-" * 70)
    lin = results["linearity"]
    if "error" not in lin:
        print(f"  R² (mean across dims): {lin['r2_mean']:.4f}")
        print(f"  Cosine similarity:     {lin['cosine_similarity']:.4f}")
        print(f"  MSE:                   {lin['mse']:.6f}")
        print(f"  MAE:                   {lin['mae']:.6f}")

        # Interpret linearity
        if lin['r2_mean'] > 0.8:
            interp = "Highly linear (good structure)"
        elif lin['r2_mean'] > 0.5:
            interp = "Moderately linear"
        else:
            interp = "Non-linear (complex transformation)"
        print(f"  Interpretation:        {interp}")
    else:
        print(f"  Error: {lin['error']}")

    # Geometry
    print("\n" + "-" * 70)
    print("Embedding Geometry")
    print("-" * 70)
    geo = results["geometry"]
    print(f"  Before state norm:     {geo['before_norm_mean']:.4f} ± {geo['before_norm_std']:.4f}")
    print(f"  After state norm:      {geo['after_norm_mean']:.4f} ± {geo['after_norm_std']:.4f}")
    print(f"  Before-After cosine:   {geo['before_after_cosine_mean']:.4f} ± {geo['before_after_cosine_std']:.4f}")
    print(f"  Pairwise distance:     {geo['pairwise_distance_mean']:.4f} ± {geo['pairwise_distance_std']:.4f}")

    # Per-action analysis
    print("\n" + "-" * 70)
    print("Per-Action-Type Analysis")
    print("-" * 70)
    print(f"{'Action Type':<20} {'Count':>8} {'Change Mag':>12} {'Pred Error':>12} {'Cos Sim':>10}")
    print("-" * 64)

    for action_name, stats in sorted(
        results["per_action_type"].items(),
        key=lambda x: -x[1]["count"]
    ):
        print(f"{action_name:<20} {stats['count']:>8} "
              f"{stats['actual_change_magnitude_mean']:>12.4f} "
              f"{stats['prediction_error_mean']:>12.4f} "
              f"{stats['cosine_similarity_mean']:>10.4f}")

    # Overall assessment
    print("\n" + "-" * 70)
    print("Overall Assessment")
    print("-" * 70)

    # Key metrics for LLM-JEPA style analysis
    actual_svd = results["svd"]["actual_transition"]
    error_svd = results["svd"]["prediction_error"]

    print(f"\nTransition Subspace:")
    print(f"  - {actual_svd['top_10_fraction']:.1%} of variance in top 10 dimensions")
    print(f"  - Effective rank: {actual_svd['effective_rank_01']} (1% threshold)")

    print(f"\nPrediction Quality:")
    print(f"  - Error effective rank: {error_svd['effective_rank_01']}")
    print(f"  - Error top-10 fraction: {error_svd['top_10_fraction']:.1%}")

    if actual_svd['top_10_fraction'] > 0.8:
        print(f"\n✓ Transitions are concentrated in a low-dimensional subspace")
    else:
        print(f"\n⚠ Transitions spread across many dimensions")

    if lin.get('r2_mean', 0) > 0.5:
        print(f"✓ State transitions are approximately linear (R²={lin['r2_mean']:.2f})")
    else:
        print(f"⚠ State transitions are non-linear")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SVD Analysis of JEPA Embeddings")
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

    logger.info(f"Total transitions for analysis: {len(combined_dataset)}")

    # Create dataloader
    collator = TransitionCollator()
    dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Extract embeddings
    logger.info("Extracting embeddings...")
    embeddings = extract_embeddings(model, dataloader, args.device)

    # Run analyses
    logger.info("Running SVD analysis...")
    svd_results = analyze_svd(
        embeddings["before_embeds"],
        embeddings["after_embeds"],
        embeddings["predicted_embeds"],
    )

    logger.info("Running linearity analysis...")
    linearity_results = analyze_linearity(
        embeddings["before_embeds"],
        embeddings["after_embeds"],
    )

    logger.info("Running per-action analysis...")
    per_action_results = analyze_per_action_type(
        embeddings["before_embeds"],
        embeddings["after_embeds"],
        embeddings["predicted_embeds"],
        embeddings["action_types"],
    )

    logger.info("Analyzing embedding geometry...")
    geometry_results = analyze_embedding_geometry(
        embeddings["before_embeds"],
        embeddings["after_embeds"],
    )

    # Compile results
    results = {
        "num_samples": len(combined_dataset),
        "embed_dim": embeddings["before_embeds"].shape[1],
        "svd": svd_results,
        "linearity": linearity_results,
        "per_action_type": per_action_results,
        "geometry": geometry_results,
    }

    # Print results
    print_results(results)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
