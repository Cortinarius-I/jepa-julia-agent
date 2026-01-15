# Changelog

All notable changes to the JEPA-Julia-Agent project.

## [0.3.1] - 2026-01-15 - Improved Action Inference & Julia Bridge Testing

Reduced UNKNOWN action rate to 0% and added comprehensive Julia bridge tests.

### Added

**Julia Bridge Tests**
- `tests/test_julia_bridge.py`: Comprehensive test suite for Python-Julia bridge
  - 8 mock mode tests (no Julia required)
  - 2 integration tests with real Julia (via `--with-julia` flag)
  - Tests: mock fallback, world state extraction, action execution, lazy init

**New Action Types** (3 new types added to `scripts/mine_transitions.py`)
- `MODIFY_DOCS`: Docstring and comment changes
- `MODIFY_MACRO`: Julia macro annotation changes (`@deprecate`, `@inline`, etc.)
- `FIX_TYPO`: Small fixes with high character similarity (>80%)

### Improved

**Action Inference** (`scripts/mine_transitions.py`)
- Added `_check_macro_change()`: Detects 12 common Julia macros
- Added `_check_docstring_change()`: Detects triple-quoted docs and markdown-style content
- Added `_check_typo_fix()`: Uses `SequenceMatcher` for similarity detection
- Improved `_check_body_modification()`: More aggressive fallback (any significant change → MODIFY_METHOD)
- **Result: 0% UNKNOWN rate** (was 14.8%)

**Julia Bridge** (`agent/agent_loop.py`)
- `MockJulia.WorldState.extract_world_state()` now returns proper mock structure
- `MockJulia.WorldState.to_json()` properly serializes with `json.dumps()`
- Added graceful exception handling for Julia initialization failures
- Auto-installs JSON3/StructTypes packages when using juliacall

### Metrics

| Metric | Before | After |
|--------|--------|-------|
| UNKNOWN action rate | 14.8% | **0%** ✅ |
| Action types | 14 | 17 |
| Julia bridge tests | 0 | 10 |

---

## [0.3.0] - 2026-01-15 - Training Data Pipeline

Complete training data pipeline from git history mining to model training.

### Added

**Git History Mining**
- `scripts/mine_transitions.py`: Mines (state, action, state') transitions from Julia repos
  - Walks git history, filters to src/ changes
  - Infers semantic actions from diffs (ADD_METHOD, MODIFY_METHOD, etc.)
  - Improved heuristics for qualified names, short-form functions, macros
  - Body modification fallback using hunk context
  - Field/const change detection
- `scripts/validate_julia.jl`: Julia syntax validation via `Meta.parseall`
  - Checks for `:incomplete` and `:error` expression heads
  - Extracts function/type definitions
  - No external dependencies

**Training Data Loader**
- `agent/data/transition_dataset.py`: PyTorch Dataset for transitions
  - `Vocabulary`: Token→ID mapping with UNK handling
  - `TransitionDataset`: Loads JSONL, encodes graphs
  - `TransitionCollator`: Batches variable-size graphs with PyG
  - `create_dataloader()`, `load_all_transitions()` utilities
- `agent/data/__init__.py`: Module exports with torch availability check

**Training from Mined Data**
- `experiments/train_from_mined.py`: End-to-end training script
  - `SimplifiedJEPA`: GAT-based encoder for code graphs
  - EMA target encoder for stable training
  - Vocabulary building and saving
  - Train/val split with checkpoint saving
  - Achieved **0.91 cosine similarity** on initial run

### Data Collected

| Repository | Valid Transitions | Action Distribution |
|------------|------------------|---------------------|
| DataStructures.jl | 551 | MODIFY_METHOD: 313, FIX_TYPO: 78, ADD_METHOD: 73 |
| JSON3.jl | 123 | MODIFY_METHOD: 52, ADD_IMPORT: 21 |
| **Total** | **674** | 17 action types, 0% UNKNOWN |

### Metrics Achieved

- Embedding cosine similarity: **0.91** (target: >0.85) ✅
- Best validation loss: 0.054
- Model parameters: 7.1M

---

## [0.2.0] - 2025-01-14 - Paper Recommendations

Implemented 5 major recommendations from LLM-JEPA, CWM, and Agent2World papers.

### Added

**1. Adaptive Test Generation** (Agent2World-inspired)
- `agent/test_generator.py`: Python test generation strategies
- `julia/src/TestGenerator.jl`: Julia test execution
- Per-action-type test strategies (ADD_METHOD, MODIFY_METHOD, etc.)
- Property-based tests (idempotent, commutative, associative)
- TestSuite and TestCase data structures
- `test_suite_to_julia()` for Julia interop

**2. Verifier-Guided Rejection Sampling** (Agent2World-inspired)
- `agent/rejection_sampling.py`: Full filtering pipeline
- TransitionVerifier: syntax, compile, tests, invalidations
- RejectionSamplingFilter: V(τ) = 1 filtering
- VerifiedDatasetBuilder: train/val/test splits
- Verification caching for efficiency
- Reward signal computation

**3. Multi-View JEPA Loss** (LLM-JEPA-inspired)
- `agent/jepa/multi_view.py`: Complete implementation
- ViewType enum: PRE_STATE, POST_STATE, NL_GOAL, etc.
- ViewEncoder, NLGoalEncoder, ActionSequenceEncoder
- JEPAPredictor for cross-view prediction
- MultiViewJEPA with EMA target encoders
- MultiViewJEPALoss: L = L_pred + λ × L_JEPA
- Block-causal attention mask
- `analyze_embedding_structure()` for SVD analysis

**4. Knowledge Synthesis** (Agent2World-inspired)
- `agent/knowledge_synthesis.py`: Doc retrieval system
- DocStringExtractor: parse Julia docstrings
- PackageInfoExtractor: Project.toml + README
- TypeExtractor: struct/type definitions
- KnowledgeSynthesizer: aggregate knowledge
- PlanningContextEnricher: enrich planning context
- Related symbols graph construction

**5. Trace Prediction** (CWM-inspired, embedding-space adapted)
- `agent/trace_prediction.py`: Python trace models
- `julia/src/TraceCollector.jl`: Julia trace collection
- TraceEventEncoder, TraceSequenceEncoder
- TracePredictor: (state, action) → trace_embedding
- TraceStepPredictor: auto-regressive frames
- JuliaTracePredictionModel: complete model
- TracePredictionLoss: auxiliary training loss

**Integrated Training**
- `experiments/train_integrated.py`: Combined pipeline
- IntegratedJEPAModel: all components unified
- IntegratedLoss: L_state + λ_jepa × L_multiview + λ_trace × L_trace
- IntegratedTrainer: full training with EMA
- TensorBoard logging, checkpointing

### Changed
- `agent/__init__.py`: Export new modules (v0.2.0)
- `agent/jepa/__init__.py`: Export multi-view components
- `julia/Project.toml`: Add TestGenerator, TraceCollector

---

## [0.1.0] - 2025-01-14

### Added

**Core Architecture**
- JEPA world model (`agent/jepa/model.py`)
  - GraphEncoder with GAT layers for module/dispatch graphs
  - MethodEncoder for signature embeddings
  - WorldStateEncoder combining all graph encoders
  - ActionEncoder for typed action embeddings
  - JEPAPredictor: (context, action) → predicted state
  - Safety and test outcome prediction heads
  - EMA target encoder for stable training

- Typed action grammar (`julia/src/Actions.jl`)
  - 14 semantic action types: ADD_METHOD, MODIFY_METHOD, REMOVE_METHOD, etc.
  - Validation, application, and reversal functions per action
  - CompositeAction for multi-step operations

- World state extraction (`julia/src/WorldState.jl`)
  - ModuleGraph: dependency DAG
  - MethodTableState: signatures, locations
  - DispatchGraph: method dispatch relationships
  - TypeInferenceState: inferred types per method
  - TestState: test results with timing
  - InvalidationState: method invalidation tracking
  - JSON serialization for Python interop

- Deterministic executor (`julia/src/Executor.jl`)
  - Apply actions with validation
  - CodeFragment with {{HOLE}} templates
  - `render_code_fragment()` - ONLY transformer invocation point
  - Rollback capability via action inverses

- Planner (`agent/planner.py`)
  - Beam search in embedding space
  - PlanningGoal with constraints
  - Safety-filtered plan selection
  - PlanExecutor with rollback

- Graph encoders (`agent/encoders/graph_encoders.py`)
  - ModuleGraphEncoder (GAT-based)
  - DispatchGraphEncoder (GIN-based)
  - TypeHierarchyEncoder (GraphSAGE)
  - CallGraphEncoder
  - MethodSignatureEncoder
  - CompositeGraphEncoder

- Constrained transformer renderer (`transformer/render.py`)
  - CodeRenderer with hard token limits
  - Low temperature (0.2) for determinism
  - Syntax validation before returning
  - MethodBodyRenderer, TestRenderer specializations

**Training & Evaluation**
- Training pipeline (`experiments/train_jepa.py`)
  - TransitionDataset for (state, action, next_state) tuples
  - JEPATrainer with EMA updates
  - Embedding prediction loss (no reconstruction)
  - Checkpoint saving and validation

- Evaluation script (`experiments/eval_predictions.py`)
  - Embedding cosine similarity
  - Safety prediction accuracy/AUC
  - Test outcome prediction
  - Action ranking correlation
  - Per-action-type breakdown

**Infrastructure**
- Python-Julia bridge via juliacall
- CLI entry point with typer
- Rich console UI with progress tracking
- Project configuration (pyproject.toml, Project.toml)

**Documentation**
- Design document (`docs/design.md`)
- Action grammar specification (`docs/action_grammar.md`)
- Roadmap (`docs/roadmap.md`)
- README with architecture overview

**Julia Scripts**
- `extract_state.jl`: Extract world state from repo
- `run_tests.jl`: Run tests with result collection
- `collect_metrics.jl`: Gather repository metrics

### Design Decisions
- Semantic over tokens: graphs not text
- Prediction over generation: embedding space, not token sequences
- Constrained transformer use: small templates only
- Execution grounding: compiler/tests define correctness
- Reversibility: all actions have inverses

---

## [0.0.1] - 2025-01-14

### Added
- Initial project structure
- Core concept validation
- README with vision statement
