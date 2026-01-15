"""
Convert JSONL transitions to Parquet format.

Parquet provides:
- Better compression (3-5x typical for JSON-like data)
- Faster loading for large datasets
- Schema enforcement
- Columnar access for selective loading

Use this when dataset size exceeds ~100MB or 10k+ transitions.

Usage:
    python scripts/convert_to_parquet.py data/transitions/*.jsonl -o data/transitions/all.parquet
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def jsonl_to_parquet(
    jsonl_paths: list[Path],
    output_path: Path,
    valid_only: bool = True,
) -> dict:
    """
    Convert JSONL transition files to Parquet.

    Args:
        jsonl_paths: List of input JSONL files
        output_path: Output Parquet file path
        valid_only: Only include valid transitions

    Returns:
        Statistics about the conversion
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "PyArrow is required for Parquet support. "
            "Install with: pip install pyarrow"
        )

    # Collect all transitions
    transitions = []
    for path in jsonl_paths:
        logger.info(f"Reading {path}...")
        with open(path) as f:
            for line in f:
                t = json.loads(line)
                if valid_only and not t.get("is_valid", False):
                    continue
                transitions.append(t)

    logger.info(f"Collected {len(transitions)} transitions")

    if not transitions:
        raise ValueError("No transitions to convert")

    # Flatten structure for Parquet
    # Store complex nested fields as JSON strings
    records = []
    for t in transitions:
        records.append({
            "repo": t["repo"],
            "commit_sha": t["commit_sha"],
            "parent_sha": t["parent_sha"],
            "commit_message": t["commit_message"],
            "commit_date": t["commit_date"],
            "action_type": t["action"]["type"],
            "action_target_file": t["action"].get("target_file"),
            "action_target_symbol": t["action"].get("target_symbol"),
            "action_confidence": t["action"].get("confidence", 0.0),
            "files_before_json": json.dumps(t.get("files_before", {})),
            "files_after_json": json.dumps(t.get("files_after", {})),
            "source_files_changed": t.get("source_files_changed", []),
            "test_files_changed": t.get("test_files_changed", []),
            "lines_changed": t.get("lines_changed", 0),
            "is_valid": t.get("is_valid", False),
            "validation_errors_json": json.dumps(t.get("validation_errors", [])),
        })

    # Create PyArrow table
    table = pa.Table.from_pylist(records)

    # Write to Parquet with compression
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        output_path,
        compression="snappy",  # Good balance of speed/compression
    )

    # Get file size
    file_size = output_path.stat().st_size
    original_size = sum(p.stat().st_size for p in jsonl_paths)
    compression_ratio = original_size / file_size if file_size > 0 else 0

    stats = {
        "transitions": len(transitions),
        "output_path": str(output_path),
        "file_size_bytes": file_size,
        "original_size_bytes": original_size,
        "compression_ratio": compression_ratio,
    }

    logger.info(f"Wrote {len(transitions)} transitions to {output_path}")
    logger.info(f"File size: {file_size / 1024:.1f} KB (compression ratio: {compression_ratio:.1f}x)")

    return stats


def parquet_to_jsonl(
    parquet_path: Path,
    output_path: Path,
) -> int:
    """
    Convert Parquet back to JSONL.

    Args:
        parquet_path: Input Parquet file
        output_path: Output JSONL file

    Returns:
        Number of records converted
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("PyArrow is required. Install with: pip install pyarrow")

    table = pq.read_table(parquet_path)
    records = table.to_pylist()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for r in records:
            # Reconstruct nested structure
            transition = {
                "repo": r["repo"],
                "commit_sha": r["commit_sha"],
                "parent_sha": r["parent_sha"],
                "commit_message": r["commit_message"],
                "commit_date": r["commit_date"],
                "action": {
                    "type": r["action_type"],
                    "target_file": r["action_target_file"],
                    "target_symbol": r["action_target_symbol"],
                    "confidence": r["action_confidence"],
                },
                "files_before": json.loads(r["files_before_json"]),
                "files_after": json.loads(r["files_after_json"]),
                "source_files_changed": r["source_files_changed"],
                "test_files_changed": r["test_files_changed"],
                "lines_changed": r["lines_changed"],
                "is_valid": r["is_valid"],
                "validation_errors": json.loads(r["validation_errors_json"]),
            }
            f.write(json.dumps(transition) + "\n")

    logger.info(f"Wrote {len(records)} transitions to {output_path}")
    return len(records)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert transitions between JSONL and Parquet")
    parser.add_argument("input_files", type=Path, nargs="+", help="Input files")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    parser.add_argument("--to-jsonl", action="store_true", help="Convert Parquet to JSONL")
    parser.add_argument("--all", action="store_true", help="Include invalid transitions")

    args = parser.parse_args()

    if args.to_jsonl:
        # Parquet -> JSONL
        if len(args.input_files) != 1:
            parser.error("Exactly one Parquet file expected for --to-jsonl")
        count = parquet_to_jsonl(args.input_files[0], args.output)
        print(f"Converted {count} records to JSONL")
    else:
        # JSONL -> Parquet
        stats = jsonl_to_parquet(
            args.input_files,
            args.output,
            valid_only=not args.all,
        )
        print(f"Converted {stats['transitions']} transitions")
        print(f"Compression ratio: {stats['compression_ratio']:.1f}x")


if __name__ == "__main__":
    main()
