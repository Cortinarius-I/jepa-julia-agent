"""
JEPA Julia Agent

A JEPA-brained, execution-grounded coding agent for Julia.

Key components:
- World state representation and diffing
- JEPA-based planning in embedding space
- Multi-view training for structured embeddings
- Adaptive test generation for richer training signal
- Verifier-guided rejection sampling for clean gradients
- Knowledge synthesis for context enrichment
- Trace prediction for execution grounding
"""
from agent.world_state import WorldStateSnapshot, WorldStateDiff, compute_diff
from agent.planner import JEPAPlanner, PlanningGoal, ActionSequence
from agent.agent_loop import JEPAAgent

# Test generation (Recommendation #1)
from agent.test_generator import (
    AdaptiveTestGenerator,
    TestSuite,
    TestCase,
    TestResult,
    test_suite_to_julia,
)

# Rejection sampling (Recommendation #2)
from agent.rejection_sampling import (
    TransitionVerifier,
    RejectionSamplingFilter,
    VerifiedDatasetBuilder,
    VerificationResult,
    VerificationStatus,
    Transition,
)

# Knowledge synthesis (Recommendation #4)
from agent.knowledge_synthesis import (
    KnowledgeSynthesizer,
    KnowledgeContext,
    PlanningContextEnricher,
    DocString,
    PackageInfo,
    TypeInfo,
)

# Trace prediction (Recommendation #5)
from agent.trace_prediction import (
    JuliaTracePredictionModel,
    TracePredictionLoss,
    TraceEventEncoder,
    TraceSequenceEncoder,
    ExecutionTrace,
    TraceFrame,
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "WorldStateSnapshot",
    "WorldStateDiff",
    "compute_diff",
    "JEPAPlanner",
    "PlanningGoal",
    "ActionSequence",
    "JEPAAgent",
    # Test generation
    "AdaptiveTestGenerator",
    "TestSuite",
    "TestCase",
    "TestResult",
    "test_suite_to_julia",
    # Rejection sampling
    "TransitionVerifier",
    "RejectionSamplingFilter",
    "VerifiedDatasetBuilder",
    "VerificationResult",
    "VerificationStatus",
    "Transition",
    # Knowledge synthesis
    "KnowledgeSynthesizer",
    "KnowledgeContext",
    "PlanningContextEnricher",
    "DocString",
    "PackageInfo",
    "TypeInfo",
    # Trace prediction
    "JuliaTracePredictionModel",
    "TracePredictionLoss",
    "TraceEventEncoder",
    "TraceSequenceEncoder",
    "ExecutionTrace",
    "TraceFrame",
]
