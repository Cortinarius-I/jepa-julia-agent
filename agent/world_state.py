"""
World State representation in Python.

This module defines the semantic world state for a Julia codebase,
mirroring the Julia-side definitions for interoperability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx
import numpy as np
from pydantic import BaseModel


# ============================================================================
# Module Graph
# ============================================================================


class ModuleNode(BaseModel):
    """A Julia module in the dependency graph."""

    name: str
    parent: str | None = None
    submodules: list[str] = []
    imports: list[str] = []
    exports: list[str] = []
    file_path: str | None = None


class ModuleGraph(BaseModel):
    """The full module dependency graph."""

    nodes: dict[str, ModuleNode] = {}
    root: str = "Main"

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for GNN processing."""
        G = nx.DiGraph()
        for name, node in self.nodes.items():
            G.add_node(name, **node.model_dump())
            for imp in node.imports:
                G.add_edge(imp, name, relation="imports")
            for sub in node.submodules:
                G.add_edge(name, sub, relation="contains")
        return G


# ============================================================================
# Method Table
# ============================================================================


class MethodSignature(BaseModel):
    """A Julia method signature."""

    name: str
    arg_types: list[str]
    where_params: list[str] = []
    return_type: str | None = None

    def to_vector(self, type_encoder: Any = None) -> np.ndarray:
        """Encode signature as a vector for JEPA."""
        # Placeholder - actual encoding uses learned embeddings
        return np.zeros(256, dtype=np.float32)


class MethodInfo(BaseModel):
    """Full method information."""

    signature: MethodSignature
    module_name: str
    file: str
    line: int
    is_generated: bool = False
    world_age: int = 0


class MethodTableState(BaseModel):
    """State of all method tables."""

    methods: dict[str, list[MethodInfo]] = {}
    method_count: int = 0


# ============================================================================
# Dispatch Graph
# ============================================================================


class DispatchEdge(BaseModel):
    """A dispatch relationship between methods."""

    caller_sig: MethodSignature
    callee_sig: MethodSignature
    call_site_file: str
    call_site_line: int
    is_concrete: bool = True


class DispatchGraph(BaseModel):
    """Call graph with dispatch information."""

    edges: list[DispatchEdge] = []
    ambiguities: list[tuple[MethodSignature, MethodSignature]] = []

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX for GNN processing."""
        G = nx.DiGraph()
        for edge in self.edges:
            caller_key = f"{edge.caller_sig.name}:{edge.caller_sig.arg_types}"
            callee_key = f"{edge.callee_sig.name}:{edge.callee_sig.arg_types}"
            G.add_edge(caller_key, callee_key, is_concrete=edge.is_concrete)
        return G


# ============================================================================
# Type Inference State
# ============================================================================


class InferredType(BaseModel):
    """Inferred type for an expression."""

    expr_hash: int
    inferred: str
    is_concrete: bool = True
    confidence: float = 1.0


class TypeInferenceState(BaseModel):
    """Cached type inference results."""

    inferred_types: dict[int, InferredType] = {}
    inference_errors: list[str] = []


# ============================================================================
# Test State
# ============================================================================


class TestResult(BaseModel):
    """Result of a single test."""

    name: str
    file: str
    passed: bool
    error_message: str | None = None
    duration_ms: float = 0.0


class TestState(BaseModel):
    """State of all tests."""

    results: list[TestResult] = []
    total_passed: int = 0
    total_failed: int = 0
    coverage: float = 0.0


# ============================================================================
# Invalidation State
# ============================================================================


class InvalidationReason(str, Enum):
    """Reasons for method invalidation."""

    NEW_METHOD = "new_method"
    REDEFINITION = "redefinition"
    TYPE_CHANGE = "type_change"
    IMPORT_CHANGE = "import_change"


class InvalidationEvent(BaseModel):
    """Record of a method invalidation."""

    trigger_method: MethodSignature
    invalidated_methods: list[MethodSignature]
    reason: InvalidationReason
    timestamp: float


class InvalidationState(BaseModel):
    """Tracks method invalidations."""

    recent_events: list[InvalidationEvent] = []
    total_invalidations: int = 0
    hot_spots: list[MethodSignature] = []


# ============================================================================
# Full World State
# ============================================================================


class WorldStateSnapshot(BaseModel):
    """Complete semantic state of a Julia codebase."""

    modules: ModuleGraph = ModuleGraph()
    methods: MethodTableState = MethodTableState()
    dispatch: DispatchGraph = DispatchGraph()
    types: TypeInferenceState = TypeInferenceState()
    tests: TestState = TestState()
    invalidations: InvalidationState = InvalidationState()
    timestamp: float = 0.0
    repo_hash: str = ""

    def to_embedding(self, encoder: Any = None) -> np.ndarray:
        """
        Encode the entire world state as a fixed-size vector.
        This is the main input to JEPA's context encoder.
        """
        # Combine embeddings from all components
        # Placeholder - actual implementation uses learned encoders
        return np.zeros(1024, dtype=np.float32)


# ============================================================================
# World State Diff
# ============================================================================


@dataclass
class WorldStateDiff:
    """Semantic difference between two world states."""

    added_methods: list[MethodInfo] = field(default_factory=list)
    removed_methods: list[MethodInfo] = field(default_factory=list)
    modified_methods: list[tuple[MethodInfo, MethodInfo]] = field(default_factory=list)
    type_changes: list[tuple[InferredType, InferredType]] = field(default_factory=list)
    newly_passing_tests: list[TestResult] = field(default_factory=list)
    newly_failing_tests: list[TestResult] = field(default_factory=list)
    invalidations: list[InvalidationEvent] = field(default_factory=list)

    def is_safe(self) -> bool:
        """Check if this diff represents a safe change."""
        return len(self.newly_failing_tests) == 0

    def to_vector(self) -> np.ndarray:
        """Encode diff as a vector for JEPA training."""
        return np.zeros(512, dtype=np.float32)


def compute_diff(before: WorldStateSnapshot, after: WorldStateSnapshot) -> WorldStateDiff:
    """Compute the semantic diff between two world states."""
    diff = WorldStateDiff()

    # Compare methods
    before_methods = {
        f"{m.signature.name}:{m.signature.arg_types}": m
        for methods in before.methods.methods.values()
        for m in methods
    }
    after_methods = {
        f"{m.signature.name}:{m.signature.arg_types}": m
        for methods in after.methods.methods.values()
        for m in methods
    }

    for key, method in after_methods.items():
        if key not in before_methods:
            diff.added_methods.append(method)
        elif before_methods[key] != method:
            diff.modified_methods.append((before_methods[key], method))

    for key, method in before_methods.items():
        if key not in after_methods:
            diff.removed_methods.append(method)

    # Compare tests
    before_tests = {t.name: t for t in before.tests.results}
    after_tests = {t.name: t for t in after.tests.results}

    for name, test in after_tests.items():
        if name in before_tests:
            if test.passed and not before_tests[name].passed:
                diff.newly_passing_tests.append(test)
            elif not test.passed and before_tests[name].passed:
                diff.newly_failing_tests.append(test)

    return diff
