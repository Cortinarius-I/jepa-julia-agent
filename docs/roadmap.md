# JEPA-Julia-Agent Roadmap

This document outlines the development roadmap for the JEPA-brained Julia coding agent.

---

## Phase 1: Foundation

**Status: ✓ Complete**

### 1.1 Core Architecture
- [x] Define world state schema (modules, methods, dispatch, types, tests, invalidations)
- [x] Implement typed action grammar (14 semantic actions)
- [x] Design JEPA world model architecture
- [x] Create Python-Julia bridge via juliacall
- [x] Implement deterministic executor with rollback
- [x] Build transformer renderer with hard constraints

### 1.2 Infrastructure
- [x] Project scaffolding (Python + Julia)
- [x] Configuration files (pyproject.toml, Project.toml)
- [x] Documentation (design.md, action_grammar.md)
- [x] CLI entry point with typer
- [x] Training pipeline skeleton

---

## Phase 1.5: Paper Recommendations

**Status: ✓ Complete**

Based on analysis of LLM-JEPA, CWM, and Agent2World papers.

### 1.5.1 Adaptive Test Generation (Agent2World)
- [x] Per-action-type test strategies
- [x] Property-based test generation
- [x] Julia test execution interface
- [x] TestSuite/TestCase data structures

### 1.5.2 Verifier-Guided Rejection Sampling (Agent2World)
- [x] TransitionVerifier (syntax, compile, tests, invalidations)
- [x] RejectionSamplingFilter with caching
- [x] VerifiedDatasetBuilder for splits
- [x] Reward signal computation

### 1.5.3 Multi-View JEPA Loss (LLM-JEPA)
- [x] ViewType enum and ViewPair configuration
- [x] View encoders (NL, Action Sequence, AST)
- [x] JEPAPredictor for cross-view prediction
- [x] MultiViewJEPA with EMA targets
- [x] Block-causal attention mask
- [x] Embedding structure analysis (SVD)

### 1.5.4 Knowledge Synthesis (Agent2World)
- [x] DocString/PackageInfo/Type extractors
- [x] KnowledgeSynthesizer for aggregation
- [x] PlanningContextEnricher
- [x] Related symbols graph

### 1.5.5 Trace Prediction (CWM, adapted to embeddings)
- [x] TraceFrame/ExecutionTrace data structures
- [x] TraceEventEncoder, TraceSequenceEncoder
- [x] TracePredictor: (state, action) → embedding
- [x] TracePredictionLoss as auxiliary task
- [x] JuliaTracer interface

### 1.5.6 Integrated Training Pipeline
- [x] IntegratedJEPAModel combining all components
- [x] IntegratedLoss: L = Σ λᵢ × Lᵢ
- [x] train_integrated.py with full training loop

---

## Phase 2: Data Collection

**Status: ✓ Initial Pipeline Complete**

### 2.1 Repository Mining
- [x] Git history mining pipeline (`scripts/mine_transitions.py`)
- [x] Julia syntax validation (`scripts/validate_julia.jl`)
- [x] Mined DataStructures.jl (325 transitions)
- [x] Mined JSON3.jl (114 transitions)
- [ ] Curate list of well-tested Julia packages (>80% coverage)
- [ ] Scale to 500+ repositories

### 2.2 Transition Generation
- [x] **Historical commits**: Parse git history into action sequences
- [x] **Action inference**: 17 action types with confidence scoring (0% UNKNOWN)
- [ ] **Synthetic edits**: Generate random valid actions, execute, measure outcomes
- [ ] **Perturbation analysis**: Introduce bugs, measure test/type impacts
- [ ] **Rejection sampling**: Apply verifier to filter transitions

### 2.3 Data Format
- [x] JSONL schema: (state, action, next_state, validation)
- [x] PyTorch Dataset + PyG Collator (`agent/data/transition_dataset.py`)
- [x] Vocabulary building for Julia symbols
- [x] Parquet storage (6x compression, default format)
- [x] Codespaces training script for scalable cloud training
- [ ] Include multi-view data (NL goals, traces)

### Progress: 674 transitions from 2 repositories, 0% UNKNOWN action inference, 0.91 cosine sim achieved
### Target: 100k+ verified transitions from 500+ repositories

### Codespaces Training
The `scripts/train_codespaces.sh` script enables scalable training:
- Mines 8-12 Julia packages directly to Parquet
- Fits within Codespaces limits (60 core-hours, 15GB storage)
- Three modes: `--quick` (2h), default (4-6h), `--full` (8h)
- Target repos: JSON.jl, CSV.jl, ForwardDiff.jl, Optim.jl, Distributions.jl, Graphs.jl, HTTP.jl, Zygote.jl, Pluto.jl, DataFrames.jl, Flux.jl, JuMP.jl

---

## Phase 3: World State Extraction

**Status: Pending**

### 3.1 Module Graph Extraction
- [ ] Use CodeTracking.jl to find all modules
- [ ] Extract exports, imports, dependencies
- [ ] Build dependency DAG

### 3.2 Method Table Extraction
- [ ] Enumerate methods via Base.methods
- [ ] Extract signatures (name, types, where clauses)
- [ ] Track source locations

### 3.3 Dispatch Graph
- [ ] Analyze method ambiguities
- [ ] Build call graph via static analysis
- [ ] Detect specialization patterns

### 3.4 Type Inference State
- [ ] Integrate with Cthulhu.jl for @code_warntype
- [ ] Track inferred types per method
- [ ] Detect type instabilities

### 3.5 Test State
- [ ] Parse Test.jl test sets
- [ ] Track pass/fail per test
- [ ] Measure coverage via Coverage.jl

### 3.6 Invalidation Tracking
- [ ] Hook into SnoopCompile.jl
- [ ] Track method invalidations on edits
- [ ] Build invalidation graph

---

## Phase 4: JEPA Training

**Status: Pending**

### 4.1 Encoder Training
- [ ] Pre-train graph encoder on module/dispatch graphs
- [ ] Pre-train method encoder on signature data
- [ ] Validate embedding quality

### 4.2 JEPA Training
- [ ] Train predictor: (state, action) → next_state embedding
- [ ] Train safety head: predict test failures
- [ ] Train invalidation head: predict recompilation
- [ ] Implement EMA target encoder updates

### 4.3 Ablations
- [ ] Compare: GNN vs transformer for graph encoding
- [ ] Compare: joint vs separate state/action encoders
- [ ] Measure: prediction horizon (1-step vs multi-step)

### Target Metrics
- Embedding cosine similarity: >0.85
- Safety prediction AUC: >0.90
- Test outcome accuracy: >0.80

---

## Phase 5: Planner Development

**Status: Pending**

### 5.1 Goal Specification
- [ ] Natural language → goal embedding pipeline
- [ ] Constraint specification (must pass tests, no invalidations)
- [ ] Multi-objective weighting

### 5.2 Search Algorithms
- [ ] Beam search in embedding space
- [ ] Monte Carlo Tree Search variant
- [ ] Best-first search with JEPA heuristic

### 5.3 Plan Validation
- [ ] Safety filtering before execution
- [ ] Rollback on failure
- [ ] Plan explanation generation

---

## Phase 6: Integration & Evaluation

**Status: Pending**

### 6.1 End-to-End Testing
- [ ] Benchmark on standard refactoring tasks
- [ ] Measure success rate vs baseline (GPT-4, Claude)
- [ ] Evaluate on real open-source issues

### 6.2 Safety Evaluation
- [ ] Adversarial testing (malicious goals)
- [ ] Hallucination rate measurement
- [ ] Rollback success rate

### 6.3 Performance
- [ ] Latency benchmarks (planning + execution)
- [ ] Memory usage profiling
- [ ] Scaling to large repositories (>100k LOC)

---

## Phase 7: Extensions

**Status: Future**

### 7.1 Multi-Language Support
- [ ] Generalize world state schema
- [ ] Language-specific extractors (Python, Rust)
- [ ] Shared JEPA embedding space

### 7.2 Interactive Mode
- [ ] User feedback integration
- [ ] Preference learning
- [ ] Explanation generation

### 7.3 Continuous Learning
- [ ] Online JEPA updates from successful edits
- [ ] Active learning for ambiguous cases
- [ ] Federated learning across repos

---

## Success Criteria

### Minimum Viable Product
1. Successfully complete 10 refactoring tasks on real repos
2. Zero test regressions on all completed tasks
3. <10% rollback rate during planning
4. <30s latency for single-action plans

### Production Ready
1. 90%+ success rate on benchmark tasks
2. Handles repos with >50k LOC
3. Sub-5s planning latency
4. Comprehensive safety guarantees documented

---

## Timeline (Estimated)

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Foundation | 2 weeks | - |
| Phase 2: Data Collection | 4 weeks | Phase 1 |
| Phase 3: World State | 3 weeks | Phase 1 |
| Phase 4: JEPA Training | 4 weeks | Phase 2, 3 |
| Phase 5: Planner | 3 weeks | Phase 4 |
| Phase 6: Evaluation | 2 weeks | Phase 5 |
| Phase 7: Extensions | Ongoing | Phase 6 |

**Total to MVP: ~18 weeks**

---

## Insights from Related Work

Based on analysis of recent papers (LLM-JEPA, CWM, Agent2World), we've identified several improvements:

### From LLM-JEPA (Huang, LeCun, Balestriero 2025)
- **Multi-view JEPA loss**: Train to predict between multiple views of the same change
- **Near-linear mappings**: JEPA creates structured embedding spaces
- **Overfitting resistance**: Critical for limited Julia training data

### From CWM (Meta FAIR 2025)
- **Execution trace prediction**: Could add Julia trace prediction as auxiliary training signal
- **Grounded reasoning**: "Models that understand what code does, not just what it looks like"
- **RL synergy**: World models help agents focus on rewards vs. dynamics

### From Agent2World (HKU 2025)
- **Adaptive testing**: Generate targeted tests based on observed errors, not just run existing tests
- **MDP formalization**: Make the sequential decision-making structure more explicit
- **Knowledge synthesis**: Add context gathering for unfamiliar packages

---

## Recommended Architecture Updates

### High Priority

1. **Multi-View JEPA Training** (from LLM-JEPA)
   - Add view pairs: (NL goal ↔ action sequence), (pre-state ↔ post-state), (AST diff ↔ semantic action)
   - Use custom attention mask for efficient view encoding

2. **Adaptive Test Generation** (from Agent2World)
   - Generate property-based tests targeting the specific action
   - Test synthesizer produces unit tests based on action type and affected methods
   - Provides richer training signal than fixed test suites

3. **Verifier-Guided Rejection Sampling** (from Agent2World)
   - Filter training data: only keep transitions where action leads to valid, test-passing code
   - Creates cleaner training signal

### Medium Priority

4. **Julia Trace Prediction** (from CWM)
   - Add auxiliary task: predict execution trace for affected methods
   - Captures runtime semantics, not just static properties
   - May help with type stability prediction

5. **Knowledge Synthesis Stage** (from Agent2World)
   - Before planning, gather context about unfamiliar modules
   - Use documentation retrieval or web search
   - Feed into world state representation

### Lower Priority

6. **Neural Debugger Capabilities** (from CWM)
   - Predict execution without running code
   - Could help with "what-if" analysis during planning

---

## Open Questions

1. **Embedding Dimensionality**: What's the right size for state embeddings? 256? 512? 1024?

2. **Action Granularity**: Are 14 action types enough? Too many?

3. **Graph Encoding**: GAT vs GraphSAGE vs GIN for module graphs?

4. **Training Data Scale**: How many transitions needed for good generalization?

5. **Multi-Step Planning**: How far ahead can JEPA predict accurately?

6. **Transfer Learning**: Can a JEPA trained on one repo transfer to others?

7. **View Selection** (NEW): Which view pairs provide the strongest training signal for Julia?

8. **Trace Granularity** (NEW): Line-level vs. method-level vs. statement-level traces?

9. **Test Generation** (NEW): How to generate meaningful Julia tests automatically?

---

## Contributing

See CONTRIBUTING.md (to be created) for guidelines on:
- Adding new action types
- Improving world state extraction
- Contributing training data
- Reporting safety issues
