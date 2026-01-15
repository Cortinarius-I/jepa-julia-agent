#!/usr/bin/env python3
"""
Tests for Parquet support in the training pipeline.

Tests:
1. JSONL to Parquet conversion
2. Parquet to JSONL round-trip
3. TransitionDataset loading from Parquet
4. Vocabulary building from Parquet
5. Data integrity (JSONL vs Parquet produce same results)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestParquetConversion(unittest.TestCase):
    """Test JSONL <-> Parquet conversion."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.test_transitions = [
            {
                "repo": "TestRepo",
                "commit_sha": "abc123",
                "parent_sha": "def456",
                "commit_message": "Add feature",
                "commit_date": "2024-01-15T12:00:00",
                "action": {
                    "type": "ADD_METHOD",
                    "target_file": "src/test.jl",
                    "target_symbol": "test_func",
                    "confidence": 0.9,
                },
                "files_before": {"src/test.jl": "# empty"},
                "files_after": {"src/test.jl": "function test_func(x)\n    x + 1\nend"},
                "source_files_changed": ["src/test.jl"],
                "test_files_changed": [],
                "lines_changed": 3,
                "is_valid": True,
                "validation_errors": [],
            },
            {
                "repo": "TestRepo",
                "commit_sha": "xyz789",
                "parent_sha": "abc123",
                "commit_message": "Modify method",
                "commit_date": "2024-01-15T13:00:00",
                "action": {
                    "type": "MODIFY_METHOD",
                    "target_file": "src/test.jl",
                    "target_symbol": "test_func",
                    "confidence": 0.8,
                },
                "files_before": {"src/test.jl": "function test_func(x)\n    x + 1\nend"},
                "files_after": {"src/test.jl": "function test_func(x, y=0)\n    x + y + 1\nend"},
                "source_files_changed": ["src/test.jl"],
                "test_files_changed": [],
                "lines_changed": 2,
                "is_valid": True,
                "validation_errors": [],
            },
        ]

        # Create temp directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        cls.jsonl_path = Path(cls.temp_dir) / "test.jsonl"
        cls.parquet_path = Path(cls.temp_dir) / "test.parquet"

        # Write test JSONL
        with open(cls.jsonl_path, "w") as f:
            for t in cls.test_transitions:
                f.write(json.dumps(t) + "\n")

    def test_jsonl_to_parquet(self):
        """Test converting JSONL to Parquet."""
        try:
            import pyarrow
        except ImportError:
            self.skipTest("pyarrow not installed")

        from scripts.convert_to_parquet import jsonl_to_parquet

        stats = jsonl_to_parquet([self.jsonl_path], self.parquet_path)

        self.assertEqual(stats["transitions"], 2)
        self.assertTrue(self.parquet_path.exists())
        self.assertGreater(stats["compression_ratio"], 0)

    def test_parquet_to_jsonl_roundtrip(self):
        """Test Parquet -> JSONL roundtrip preserves data."""
        try:
            import pyarrow
        except ImportError:
            self.skipTest("pyarrow not installed")

        from scripts.convert_to_parquet import jsonl_to_parquet, parquet_to_jsonl

        # Convert to Parquet
        jsonl_to_parquet([self.jsonl_path], self.parquet_path)

        # Convert back to JSONL
        roundtrip_path = Path(self.temp_dir) / "roundtrip.jsonl"
        count = parquet_to_jsonl(self.parquet_path, roundtrip_path)

        self.assertEqual(count, 2)

        # Verify data integrity
        with open(roundtrip_path) as f:
            roundtrip_data = [json.loads(line) for line in f]

        self.assertEqual(len(roundtrip_data), 2)
        self.assertEqual(roundtrip_data[0]["commit_sha"], "abc123")
        self.assertEqual(roundtrip_data[1]["action"]["type"], "MODIFY_METHOD")


class TestTransitionDatasetParquet(unittest.TestCase):
    """Test TransitionDataset with Parquet files."""

    @classmethod
    def setUpClass(cls):
        """Create test Parquet file."""
        try:
            import pyarrow
        except ImportError:
            cls.parquet_available = False
            return

        cls.parquet_available = True

        cls.test_transitions = [
            {
                "repo": "TestRepo",
                "commit_sha": f"commit{i}",
                "parent_sha": f"parent{i}",
                "commit_message": f"Change {i}",
                "commit_date": "2024-01-15T12:00:00",
                "action": {
                    "type": "ADD_METHOD",
                    "target_file": "src/test.jl",
                    "target_symbol": f"func{i}",
                    "confidence": 0.9,
                },
                "files_before": {"src/test.jl": ""},
                "files_after": {"src/test.jl": f"function func{i}(x)\n    x\nend"},
                "source_files_changed": ["src/test.jl"],
                "test_files_changed": [],
                "lines_changed": 3,
                "is_valid": True,
                "validation_errors": [],
            }
            for i in range(10)
        ]

        cls.temp_dir = tempfile.mkdtemp()
        cls.jsonl_path = Path(cls.temp_dir) / "test.jsonl"
        cls.parquet_path = Path(cls.temp_dir) / "test.parquet"

        # Write JSONL
        with open(cls.jsonl_path, "w") as f:
            for t in cls.test_transitions:
                f.write(json.dumps(t) + "\n")

        # Convert to Parquet
        from scripts.convert_to_parquet import jsonl_to_parquet
        jsonl_to_parquet([cls.jsonl_path], cls.parquet_path)

    def setUp(self):
        if not self.parquet_available:
            self.skipTest("pyarrow not installed")

    def test_load_from_parquet(self):
        """Test loading TransitionDataset from Parquet."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "data"))
        from transition_dataset import TransitionDataset, Vocabulary

        # Build vocab
        vocab = Vocabulary.build_from_transitions(self.parquet_path)

        # Load dataset
        dataset = TransitionDataset(self.parquet_path, vocab)

        self.assertEqual(len(dataset), 10)

    def test_parquet_jsonl_equivalence(self):
        """Test that Parquet and JSONL produce equivalent datasets."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "data"))
        from transition_dataset import TransitionDataset, Vocabulary

        # Build vocab from JSONL
        vocab = Vocabulary.build_from_transitions(self.jsonl_path)

        # Load both formats
        ds_jsonl = TransitionDataset(self.jsonl_path, vocab)
        ds_parquet = TransitionDataset(self.parquet_path, vocab)

        # Same length
        self.assertEqual(len(ds_jsonl), len(ds_parquet))

        # Same data (check first item)
        item_jsonl = ds_jsonl[0]
        item_parquet = ds_parquet[0]

        self.assertEqual(item_jsonl["commit_sha"], item_parquet["commit_sha"])
        self.assertEqual(
            item_jsonl["action_type"].item(),
            item_parquet["action_type"].item()
        )

    def test_vocabulary_from_parquet(self):
        """Test building vocabulary from Parquet."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "data"))
        from transition_dataset import Vocabulary

        vocab_jsonl = Vocabulary.build_from_transitions(self.jsonl_path)
        vocab_parquet = Vocabulary.build_from_transitions(self.parquet_path)

        # Same vocabulary size
        self.assertEqual(len(vocab_jsonl.token_to_id), len(vocab_parquet.token_to_id))

    def test_getitem_returns_tensors(self):
        """Test that __getitem__ returns proper tensors."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "agent" / "data"))
        from transition_dataset import TransitionDataset, Vocabulary
        import torch

        vocab = Vocabulary.build_from_transitions(self.parquet_path)
        dataset = TransitionDataset(self.parquet_path, vocab)

        item = dataset[0]

        # Check tensor types
        self.assertIsInstance(item["before_node_features"], torch.Tensor)
        self.assertIsInstance(item["after_node_features"], torch.Tensor)
        self.assertIsInstance(item["action_type"], torch.Tensor)
        self.assertIsInstance(item["action_target"], torch.Tensor)


class TestParquetCompression(unittest.TestCase):
    """Test Parquet compression benefits."""

    def test_compression_ratio(self):
        """Test that Parquet achieves meaningful compression."""
        try:
            import pyarrow
        except ImportError:
            self.skipTest("pyarrow not installed")

        from scripts.convert_to_parquet import jsonl_to_parquet

        # Create larger test data
        transitions = []
        for i in range(100):
            transitions.append({
                "repo": "TestRepo",
                "commit_sha": f"commit{i:04d}",
                "parent_sha": f"parent{i:04d}",
                "commit_message": f"This is a longer commit message for change number {i}",
                "commit_date": "2024-01-15T12:00:00",
                "action": {
                    "type": "MODIFY_METHOD",
                    "target_file": f"src/module{i % 10}/file{i % 5}.jl",
                    "target_symbol": f"function_name_{i}",
                    "confidence": 0.85,
                },
                "files_before": {f"src/test{i}.jl": f"# File {i} before\nfunction old{i}()\nend"},
                "files_after": {f"src/test{i}.jl": f"# File {i} after\nfunction new{i}(x)\n    x + {i}\nend"},
                "source_files_changed": [f"src/test{i}.jl"],
                "test_files_changed": [],
                "lines_changed": 5,
                "is_valid": True,
                "validation_errors": [],
            })

        temp_dir = tempfile.mkdtemp()
        jsonl_path = Path(temp_dir) / "large.jsonl"
        parquet_path = Path(temp_dir) / "large.parquet"

        with open(jsonl_path, "w") as f:
            for t in transitions:
                f.write(json.dumps(t) + "\n")

        stats = jsonl_to_parquet([jsonl_path], parquet_path)

        # Should achieve at least 2x compression
        self.assertGreater(stats["compression_ratio"], 2.0)


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestParquetConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestTransitionDatasetParquet))
    suite.addTests(loader.loadTestsFromTestCase(TestParquetCompression))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
