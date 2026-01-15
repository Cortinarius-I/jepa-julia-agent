"""
Training Data Loader for JEPA Model.

Loads mined transitions from JSONL files and converts them to the
tensor format expected by the JEPA model.

Key components:
1. TransitionDataset: PyTorch Dataset for transitions
2. TransitionCollator: Collates batches with variable-size graphs
3. Vocabulary building for type/symbol tokens
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)


# ============================================================================
# Vocabulary
# ============================================================================

@dataclass
class Vocabulary:
    """Vocabulary for tokenizing Julia symbols and types."""
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_ID = 0
    UNK_ID = 1

    @classmethod
    def build_from_transitions(
        cls,
        transitions_path: Path,
        min_freq: int = 2,
        max_vocab_size: int = 10000,
    ) -> "Vocabulary":
        """Build vocabulary from transition files."""
        counter: Counter[str] = Counter()

        with open(transitions_path) as f:
            for line in f:
                t = json.loads(line)
                # Extract tokens from action
                if t["action"].get("target_symbol"):
                    counter[t["action"]["target_symbol"]] += 1

                # Extract tokens from file contents (function/type names)
                for content in t["files_after"].values():
                    tokens = cls._extract_julia_tokens(content)
                    counter.update(tokens)

        # Build vocabulary with special tokens
        token_to_id = {cls.PAD_TOKEN: cls.PAD_ID, cls.UNK_TOKEN: cls.UNK_ID}

        for token, freq in counter.most_common(max_vocab_size - 2):
            if freq >= min_freq:
                token_to_id[token] = len(token_to_id)

        id_to_token = {v: k for k, v in token_to_id.items()}

        logger.info(f"Built vocabulary with {len(token_to_id)} tokens")
        return cls(token_to_id, id_to_token)

    @staticmethod
    def _extract_julia_tokens(code: str) -> list[str]:
        """Extract identifiers from Julia code."""
        import re
        # Simple regex for Julia identifiers
        pattern = r'\b([A-Za-z_][A-Za-z0-9_!]*)\b'
        return re.findall(pattern, code)

    def encode(self, token: str) -> int:
        """Encode a token to its ID."""
        return self.token_to_id.get(token, self.UNK_ID)

    def encode_sequence(self, tokens: list[str], max_len: int) -> list[int]:
        """Encode a sequence of tokens, padding/truncating as needed."""
        ids = [self.encode(t) for t in tokens[:max_len]]
        ids += [self.PAD_ID] * (max_len - len(ids))
        return ids

    def decode(self, token_id: int) -> str:
        """Decode a token ID to its string."""
        return self.id_to_token.get(token_id, self.UNK_TOKEN)

    def save(self, path: Path) -> None:
        """Save vocabulary to file."""
        with open(path, "w") as f:
            json.dump(self.token_to_id, f)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from file."""
        with open(path) as f:
            token_to_id = json.load(f)
        id_to_token = {v: k for k, v in token_to_id.items()}
        return cls(token_to_id, id_to_token)


# ============================================================================
# Action Type Mapping
# ============================================================================

ACTION_TYPES = [
    "ADD_METHOD",
    "MODIFY_METHOD",
    "REMOVE_METHOD",
    "ADD_FIELD",
    "MODIFY_FIELD",
    "REMOVE_FIELD",
    "ADD_IMPORT",
    "REMOVE_IMPORT",
    "RENAME_SYMBOL",
    "MOVE_DEFINITION",
    "ADD_TEST",
    "MODIFY_TEST",
    "REMOVE_TEST",
    "ADD_TYPE",
    "MODIFY_TYPE",
    "UNKNOWN",
]

ACTION_TO_ID = {action: i for i, action in enumerate(ACTION_TYPES)}


# ============================================================================
# Transition Dataset
# ============================================================================

@dataclass
class ProcessedTransition:
    """A processed transition ready for model input."""
    # World state before (simplified representation)
    state_before: dict
    # World state after
    state_after: dict
    # Action
    action_type_id: int
    action_target_symbol_id: int
    # Metadata
    commit_sha: str
    repo: str


class TransitionDataset(Dataset):
    """
    PyTorch Dataset for code transitions.

    Loads transitions from JSONL and converts to model-ready format.
    """

    def __init__(
        self,
        transitions_path: Path,
        vocab: Vocabulary,
        max_nodes: int = 100,
        max_methods: int = 50,
        valid_only: bool = True,
    ):
        self.transitions_path = transitions_path
        self.vocab = vocab
        self.max_nodes = max_nodes
        self.max_methods = max_methods

        # Load transitions
        self.transitions: list[dict] = []
        with open(transitions_path) as f:
            for line in f:
                t = json.loads(line)
                if valid_only and not t.get("is_valid", False):
                    continue
                self.transitions.append(t)

        logger.info(f"Loaded {len(self.transitions)} transitions from {transitions_path}")

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict:
        """Get a single transition as tensors."""
        t = self.transitions[idx]

        # Extract state representations
        state_before = self._encode_state(t["files_before"])
        state_after = self._encode_state(t["files_after"])

        # Encode action
        action_type_id = ACTION_TO_ID.get(t["action"]["type"], ACTION_TO_ID["UNKNOWN"])
        target_symbol = t["action"].get("target_symbol") or ""
        target_symbol_id = self.vocab.encode(target_symbol)

        return {
            # State before
            "before_node_features": state_before["node_features"],
            "before_edge_index": state_before["edge_index"],
            "before_method_ids": state_before["method_ids"],

            # State after
            "after_node_features": state_after["node_features"],
            "after_edge_index": state_after["edge_index"],
            "after_method_ids": state_after["method_ids"],

            # Action
            "action_type": torch.tensor(action_type_id, dtype=torch.long),
            "action_target": torch.tensor(target_symbol_id, dtype=torch.long),

            # Metadata
            "commit_sha": t["commit_sha"],
            "repo": t["repo"],
            "lines_changed": t["lines_changed"],
        }

    def _encode_state(self, files: dict[str, str]) -> dict:
        """
        Encode file contents into a graph representation.

        This is a simplified encoding - in production we'd use
        the Julia bridge to get proper AST/semantic info.
        """
        # Extract all functions and types from files
        all_tokens = []
        for content in files.values():
            tokens = self._extract_definitions(content)
            all_tokens.extend(tokens)

        # Create nodes for each definition
        num_nodes = min(len(all_tokens), self.max_nodes)
        if num_nodes == 0:
            num_nodes = 1  # Ensure at least one node
            all_tokens = ["<EMPTY>"]

        # Node features: one-hot of token ID
        node_ids = [self.vocab.encode(t) for t in all_tokens[:num_nodes]]
        node_features = torch.zeros(num_nodes, 128)
        for i, tok_id in enumerate(node_ids):
            node_features[i, tok_id % 128] = 1.0  # Simple hash embedding

        # Create edges (connect sequential definitions)
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # Bidirectional

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Method IDs (padded)
        method_ids = self.vocab.encode_sequence(all_tokens, self.max_methods)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "method_ids": torch.tensor(method_ids, dtype=torch.long),
        }

    def _extract_definitions(self, code: str) -> list[str]:
        """Extract function and type definition names from Julia code."""
        import re
        definitions = []

        # Function definitions
        func_patterns = [
            r"function\s+([\w.]+)\s*\(",
            r"([\w.]+)\s*\([^)]*\)\s*=",
            r"macro\s+(\w+)\s*\(",
        ]
        for pattern in func_patterns:
            definitions.extend(re.findall(pattern, code))

        # Type definitions
        type_patterns = [
            r"struct\s+(\w+)",
            r"abstract\s+type\s+(\w+)",
        ]
        for pattern in type_patterns:
            definitions.extend(re.findall(pattern, code))

        return definitions


# ============================================================================
# Collation
# ============================================================================

class TransitionCollator:
    """
    Collates batches of transitions.

    Handles variable-size graphs by using PyG's Batch.
    """

    def __call__(self, batch: list[dict]) -> dict:
        """Collate a batch of transitions."""
        # Separate graph data from other tensors
        before_graphs = []
        after_graphs = []

        for item in batch:
            before_graphs.append(Data(
                x=item["before_node_features"],
                edge_index=item["before_edge_index"],
            ))
            after_graphs.append(Data(
                x=item["after_node_features"],
                edge_index=item["after_edge_index"],
            ))

        # Batch graphs
        before_batch = Batch.from_data_list(before_graphs)
        after_batch = Batch.from_data_list(after_graphs)

        # Stack other tensors
        action_types = torch.stack([item["action_type"] for item in batch])
        action_targets = torch.stack([item["action_target"] for item in batch])
        before_method_ids = torch.stack([item["before_method_ids"] for item in batch])
        after_method_ids = torch.stack([item["after_method_ids"] for item in batch])

        return {
            # Before state
            "before_node_features": before_batch.x,
            "before_edge_index": before_batch.edge_index,
            "before_batch": before_batch.batch,
            "before_method_ids": before_method_ids,

            # After state
            "after_node_features": after_batch.x,
            "after_edge_index": after_batch.edge_index,
            "after_batch": after_batch.batch,
            "after_method_ids": after_method_ids,

            # Action
            "action_type": action_types,
            "action_target": action_targets,

            # Metadata
            "commit_sha": [item["commit_sha"] for item in batch],
            "repo": [item["repo"] for item in batch],
            "lines_changed": torch.tensor([item["lines_changed"] for item in batch]),
        }


# ============================================================================
# Data Loading Utilities
# ============================================================================

def create_dataloader(
    transitions_path: Path,
    vocab: Vocabulary,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    valid_only: bool = True,
) -> DataLoader:
    """Create a DataLoader for transitions."""
    dataset = TransitionDataset(
        transitions_path=transitions_path,
        vocab=vocab,
        valid_only=valid_only,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=TransitionCollator(),
        pin_memory=True,
    )


def load_all_transitions(
    transitions_dir: Path,
    vocab: Vocabulary | None = None,
    batch_size: int = 32,
) -> tuple[DataLoader, Vocabulary]:
    """
    Load all transitions from a directory.

    Combines multiple JSONL files and creates train/val split.
    """
    # Find all JSONL files
    jsonl_files = list(transitions_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {transitions_dir}")

    logger.info(f"Found {len(jsonl_files)} transition files")

    # Build vocabulary if not provided
    if vocab is None:
        vocab = Vocabulary.build_from_transitions(jsonl_files[0])
        for f in jsonl_files[1:]:
            # Could extend vocabulary here if needed
            pass

    # For now, just use the first file (can extend to combine later)
    dataloader = create_dataloader(
        jsonl_files[0],
        vocab,
        batch_size=batch_size,
    )

    return dataloader, vocab


# ============================================================================
# Main / Testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test transition data loading")
    parser.add_argument("transitions_path", type=Path, help="Path to transitions JSONL")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")

    args = parser.parse_args()

    # Build vocabulary
    print("Building vocabulary...")
    vocab = Vocabulary.build_from_transitions(args.transitions_path)
    print(f"Vocabulary size: {len(vocab.token_to_id)}")

    # Create dataloader
    print("\nCreating dataloader...")
    dataloader = create_dataloader(
        args.transitions_path,
        vocab,
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f"Dataset size: {len(dataloader.dataset)}")

    # Test iteration
    print("\nTesting batch iteration...")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  Before nodes: {batch['before_node_features'].shape}")
        print(f"  Before edges: {batch['before_edge_index'].shape}")
        print(f"  After nodes: {batch['after_node_features'].shape}")
        print(f"  Action types: {batch['action_type']}")
        print(f"  Repos: {batch['repo']}")

        if i >= 2:
            break

    print("\nData loading test complete!")
