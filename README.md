# JEPA-Julia-Agent

**A JEPA-brained, execution-grounded coding agent for Julia**

This repository implements a hybrid coding agent that treats a Julia codebase as an **executable semantic world**, not a sequence of tokens. The agent learns to predict the *semantic consequences* of code edits using a **Joint Embedding Predictive Architecture (JEPA)** and uses transformers only for **localized code rendering**, never for planning or global reasoning.

The primary objective is **reliable, non-hallucinating code modification**, with a clear path toward grounded code generation.

---

## 1. Core Idea (TL;DR)

- **JEPA = brain** → predicts what will happen if we change code  
- **Execution = truth** → compiler, inference, and tests define correctness  
- **Transformers = typists** → only fill in small code fragments when asked  
- **Actions are semantic** → no free-form generation at the planning level  

This design avoids context-window limits, reduces hallucinations by construction, and enables long-horizon planning over large repositories.

---

## 2. System Architecture

```
Julia Repo
   ↓
World-State Extractor
   ↓
Semantic World State (graphs + metrics)
   ↓
JEPA World Model
   ↓
Planner over Typed Actions
   ↓
Deterministic Executor (AST edits)
   ↓
Compiler / Inference / Tests
   ↺ (feedback to JEPA)
```

Transformers are invoked **only** inside the executor when an action requires code text.

---

## 3. Repository Structure

```
jepa-julia-agent/
│
├── README.md
├── CHANGELOG.md          # Version history
├── STATUS.md             # Current project status
├── pyproject.toml
├── requirements.txt
│
├── julia/
│   ├── Project.toml
│   ├── src/
│   │   ├── WorldState.jl
│   │   ├── IRTools.jl
│   │   ├── Actions.jl
│   │   ├── Executor.jl
│   │   ├── TestGenerator.jl  # NEW: Adaptive test generation
│   │   └── TraceCollector.jl # NEW: Execution trace collection
│   └── scripts/
│       ├── extract_state.jl
│       ├── run_tests.jl
│       └── collect_metrics.jl
│
├── agent/
│   ├── world_state.py
│   ├── encoders/              # Graph neural network encoders
│   ├── data/                  # NEW: Training data pipeline
│   │   ├── transition_dataset.py  # PyTorch Dataset + PyG Collator
│   │   └── __init__.py
│   ├── jepa/
│   │   ├── model.py           # Core JEPA model
│   │   └── multi_view.py      # Multi-view JEPA (LLM-JEPA)
│   ├── planner.py
│   ├── agent_loop.py
│   ├── test_generator.py      # Adaptive test generation
│   ├── rejection_sampling.py  # Verifier-guided filtering
│   ├── knowledge_synthesis.py # Doc retrieval
│   └── trace_prediction.py    # Embedding-space traces
│
├── data/
│   ├── repos/                # Cloned Julia repositories
│   ├── transitions/          # Mined transitions (Parquet default)
│   └── embeddings/
│
├── scripts/
│   ├── mine_transitions.py   # Git history mining (17 action types, Parquet output)
│   ├── convert_to_parquet.py # JSONL ↔ Parquet conversion
│   ├── validate_julia.jl     # Julia syntax validation
│   └── train_codespaces.sh   # NEW: Training script for GitHub Codespaces
│
├── tests/
│   ├── test_julia_bridge.py  # Julia bridge tests (10 tests)
│   └── test_parquet.py       # NEW: Parquet support tests (7 tests)
│
├── transformer/
│   └── render.py              # Constrained code completion
│
├── experiments/
│   ├── train_jepa.py
│   ├── train_integrated.py    # Combined pipeline (all 5 recs)
│   ├── train_from_mined.py    # NEW: Train from mined transitions
│   └── eval_predictions.py
│
└── docs/
    ├── design.md
    ├── action_grammar.md
    └── roadmap.md
```

---

## 4. Related Work & Design Insights

This architecture is informed by recent advances in world models for code:

| Paper | Key Insight | How We Use It |
|-------|-------------|---------------|
| **LLM-JEPA** (2025) | Multi-view JEPA creates structured embedding spaces resistant to overfitting | Train on view pairs: (NL↔action), (pre↔post state) |
| **CWM** (Meta, 2025) | Execution trace prediction grounds reasoning in "what code does" | Consider Julia traces as auxiliary training signal |
| **Agent2World** (2025) | Adaptive testing + verifier-guided sampling improves world model quality | Generate targeted tests per action type |

**Key differentiators of our approach**:
- **Embedding-space prediction** (vs. CWM's token-space traces)
- **Typed action grammar** (vs. free-form patches)
- **Julia-specific semantics** (dispatch, type inference, invalidations)

See `docs/roadmap.md` for detailed analysis.

---

## 5. Initial Julia World-State Schema (v0)

The **world state** is the agent’s view of the repository. It is *semantic, executable, and compact*.

```julia
struct WorldState
    modules::ModuleGraph
    methods::MethodTableState
    dispatch::DispatchGraph
    types::TypeInferenceState
    tests::TestState
    invalidations::InvalidationState
end
```

(See README in canvas for full schema breakdown.)

---

## 6. Quick Start

```bash
# Mine transitions from a Julia repository (outputs Parquet by default)
python scripts/mine_transitions.py path/to/Julia/Package.jl
# → data/transitions/Package.jl.parquet

# Train on mined transitions (supports both Parquet and JSONL)
python experiments/train_from_mined.py data/transitions/*.parquet --epochs 50

# Results from initial training (674 transitions, 2 repos):
# - Cosine similarity: 0.91 (target >0.85) ✅
# - Best val loss: 0.054
# - Action inference: 100% classified (17 action types)

# Run tests
python tests/test_julia_bridge.py              # Julia bridge (mock mode)
python tests/test_julia_bridge.py --with-julia # Julia bridge (real Julia)
python tests/test_parquet.py                   # Parquet support (7 tests)
```

### Training in GitHub Codespaces

For large-scale training without local GPU, use the Codespaces script:

```bash
# Quick mode: 4 repos, 25 epochs (~2 hours)
./scripts/train_codespaces.sh --quick

# Default: 8 repos, 50 epochs (~4-6 hours)
./scripts/train_codespaces.sh

# Full mode: 12 repos, 100 epochs (~8 hours)
./scripts/train_codespaces.sh --full

# Download trained model
gh codespace cp remote:checkpoints/best.pt .
```

The script mines 8-12 Julia packages directly to Parquet (6x compression), then trains the JEPA model. Fits within Codespaces limits (60 core-hours, 15GB storage).

---

## 7. End Vision

A coding agent that:
- Understands large Julia codebases without context windows
- Plans changes before making them
- Rejects unsafe edits automatically
- Earns the right to generate code
