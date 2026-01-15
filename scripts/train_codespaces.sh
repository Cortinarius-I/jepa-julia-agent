#!/bin/bash
# =============================================================================
# JEPA Training Script for GitHub Codespaces
# =============================================================================
#
# This script mines Julia package git histories and trains the JEPA model
# within GitHub Codespaces resource limits (60 core-hours, 15GB storage).
#
# Usage:
#   ./scripts/train_codespaces.sh [options]
#
# Options:
#   --repos N       Number of repos to mine (default: 8)
#   --epochs N      Training epochs (default: 50)
#   --quick         Quick mode: 4 repos, 25 epochs (~2 hours)
#   --full          Full mode: 12 repos, 100 epochs (~8 hours)
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Defaults (balanced for ~4-6 hours on 4-core Codespace)
MAX_REPOS=8
EPOCHS=50
BATCH_SIZE=16
MAX_COMMITS_PER_REPO=500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repos)
            MAX_REPOS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --quick)
            MAX_REPOS=4
            EPOCHS=25
            MAX_COMMITS_PER_REPO=300
            echo "Quick mode: 4 repos, 25 epochs"
            shift
            ;;
        --full)
            MAX_REPOS=12
            EPOCHS=100
            MAX_COMMITS_PER_REPO=800
            echo "Full mode: 12 repos, 100 epochs"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Julia repositories ordered by size (smallest first for faster iteration)
REPOS=(
    "https://github.com/JuliaIO/JSON.jl"           # ~440 commits, data serialization
    "https://github.com/JuliaData/CSV.jl"          # ~775 commits, file I/O
    "https://github.com/JuliaDiff/ForwardDiff.jl"  # ~843 commits, autodiff
    "https://github.com/JuliaNLSolvers/Optim.jl"   # ~1004 commits, optimization
    "https://github.com/JuliaStats/Distributions.jl" # ~1942 commits, statistics
    "https://github.com/JuliaGraphs/Graphs.jl"     # ~2335 commits, graphs
    "https://github.com/JuliaWeb/HTTP.jl"          # ~2311 commits, networking
    "https://github.com/FluxML/Zygote.jl"          # ~2455 commits, autodiff
    "https://github.com/fonsp/Pluto.jl"            # ~2698 commits, notebooks
    "https://github.com/JuliaData/DataFrames.jl"   # ~2840 commits, data
    "https://github.com/FluxML/Flux.jl"            # ~5035 commits, ML
    "https://github.com/jump-dev/JuMP.jl"          # ~5129 commits, optimization
)

# =============================================================================
# Setup
# =============================================================================

echo "=============================================="
echo "JEPA Training for GitHub Codespaces"
echo "=============================================="
echo "Config: $MAX_REPOS repos, $EPOCHS epochs, $MAX_COMMITS_PER_REPO commits/repo"
echo ""

# Create directories
mkdir -p repos transitions checkpoints

# Check/install dependencies
echo "Checking dependencies..."
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch..."
    pip install torch --quiet
fi

if ! python -c "import torch_geometric" 2>/dev/null; then
    echo "Installing PyTorch Geometric..."
    pip install torch-geometric --quiet
fi

if ! python -c "import pyarrow" 2>/dev/null; then
    echo "Installing PyArrow..."
    pip install pyarrow --quiet
fi

echo "Dependencies OK"
echo ""

# =============================================================================
# Phase 1: Mine Transitions (directly to Parquet)
# =============================================================================

echo "=============================================="
echo "Phase 1: Mining Transitions to Parquet"
echo "=============================================="

PARQUET_FILES=()
for i in "${!REPOS[@]}"; do
    if [ $i -ge $MAX_REPOS ]; then
        echo "Reached max repos limit ($MAX_REPOS)"
        break
    fi

    REPO_URL="${REPOS[$i]}"
    REPO_NAME=$(basename "$REPO_URL" .git)

    echo ""
    echo "[$((i+1))/$MAX_REPOS] Mining $REPO_NAME..."
    echo "  URL: $REPO_URL"

    # Clone with limited depth
    if [ ! -d "repos/$REPO_NAME" ]; then
        echo "  Cloning (depth=$MAX_COMMITS_PER_REPO)..."
        git clone --depth=$MAX_COMMITS_PER_REPO "$REPO_URL" "repos/$REPO_NAME" 2>/dev/null
    else
        echo "  Already cloned"
    fi

    # Mine transitions directly to Parquet
    OUTPUT_FILE="transitions/${REPO_NAME}.parquet"
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "  Mining transitions to Parquet..."
        python scripts/mine_transitions.py \
            "repos/$REPO_NAME" \
            -o "$OUTPUT_FILE" \
            -n $MAX_COMMITS_PER_REPO \
            2>&1 | grep -E "(Processed|Valid|Invalid|Wrote|action)" || true
    else
        echo "  Parquet file already exists"
    fi

    if [ -f "$OUTPUT_FILE" ]; then
        PARQUET_FILES+=("$OUTPUT_FILE")
        SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "  Output: $OUTPUT_FILE ($SIZE)"
    fi
done

echo ""
echo "Mined ${#PARQUET_FILES[@]} Parquet files"

# =============================================================================
# Phase 2: Train Model
# =============================================================================

echo ""
echo "=============================================="
echo "Phase 2: Training JEPA Model"
echo "=============================================="

echo "Starting training..."
echo "  Parquet files: ${#PARQUET_FILES[@]}"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Save dir: checkpoints/"
echo ""

# Train from all Parquet files
python experiments/train_from_mined.py \
    transitions/*.parquet \
    --save-dir checkpoints \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 1e-4 \
    --val-split 0.1

# =============================================================================
# Phase 3: Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="

echo ""
echo "Storage usage:"
du -sh repos/ transitions/ checkpoints/
echo ""
echo "Total:"
du -sh .

echo ""
echo "Checkpoints:"
ls -la checkpoints/

echo ""
echo "To download the trained model:"
echo "  - Via gh CLI: gh codespace cp remote:checkpoints/best.pt ."
echo "  - Via browser: Download from File Explorer"
echo ""
echo "Done!"
