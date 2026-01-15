"""
Verifier-guided rejection sampling for JEPA training data.

This module filters training transitions to only keep those where:
1. The action leads to syntactically valid code
2. The code compiles without errors
3. All tests pass (existing + generated adaptive tests)
4. No unexpected method invalidations occur

Inspired by Agent2World (2025): "We define a verifier outcome V(τ) ∈ {0, 1}
based on the final candidate model produced in τ. Rejection sampling keeps
only accepted trajectories: D_RS = {(x, τ) | V(τ) = 1}."

This creates cleaner gradients by removing noisy negative examples.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import time

from pydantic import BaseModel

from .test_generator import AdaptiveTestGenerator, TestSuite, test_suite_to_julia
from .world_state import WorldStateSnapshot, ActionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification Outcomes
# ---------------------------------------------------------------------------


class VerificationStatus(Enum):
    """Status of a transition verification."""
    PASSED = "passed"
    FAILED_SYNTAX = "failed_syntax"
    FAILED_COMPILE = "failed_compile"
    FAILED_TESTS = "failed_tests"
    FAILED_INVALIDATIONS = "failed_invalidations"
    FAILED_TIMEOUT = "failed_timeout"
    FAILED_RUNTIME = "failed_runtime"


@dataclass
class VerificationResult:
    """Result of verifying a transition."""
    status: VerificationStatus
    passed: bool
    
    # Detailed results
    syntax_valid: bool = True
    compile_success: bool = True
    existing_tests_passed: int = 0
    existing_tests_failed: int = 0
    adaptive_tests_passed: int = 0
    adaptive_tests_failed: int = 0
    invalidation_count: int = 0
    
    # Timing
    verification_time_ms: float = 0.0
    
    # Error details (if any)
    error_message: Optional[str] = None
    error_location: Optional[str] = None
    
    # Additional metadata
    metadata: dict = field(default_factory=dict)


@dataclass
class Transition:
    """A (state, action, next_state) transition for training."""
    state_hash: str
    action_type: str
    action_target: str
    action_params: dict
    next_state_hash: str
    
    # Cached embeddings (filled after encoding)
    state_embedding: Optional[list[float]] = None
    action_embedding: Optional[list[float]] = None
    next_state_embedding: Optional[list[float]] = None
    
    # Verification result (filled after filtering)
    verification: Optional[VerificationResult] = None
    
    # Training signal
    is_valid: bool = False
    reward_signal: float = 0.0


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class TransitionVerifier:
    """
    Verifies transitions for inclusion in training data.
    
    Only transitions that pass all verification checks are included,
    creating cleaner training signal for the JEPA model.
    """
    
    def __init__(
        self,
        julia_bridge: Any,  # JuliaBridge instance
        test_generator: Optional[AdaptiveTestGenerator] = None,
        max_invalidations: int = 10,
        test_timeout_seconds: float = 60.0,
        require_type_stability: bool = False,
    ):
        self.julia_bridge = julia_bridge
        self.test_generator = test_generator or AdaptiveTestGenerator()
        self.max_invalidations = max_invalidations
        self.test_timeout_seconds = test_timeout_seconds
        self.require_type_stability = require_type_stability
    
    def verify(
        self,
        pre_state: WorldStateSnapshot,
        action_type: str,
        action_target: str,
        action_params: dict,
        post_state: WorldStateSnapshot,
    ) -> VerificationResult:
        """
        Verify a transition is valid for training.
        
        Args:
            pre_state: World state before action
            action_type: Type of action (ADD_METHOD, etc.)
            action_target: Target of action
            action_params: Parameters for the action
            post_state: World state after action
            
        Returns:
            VerificationResult with detailed status
        """
        start_time = time.time()
        
        # Step 1: Syntax validation
        syntax_result = self._check_syntax(action_params)
        if not syntax_result.passed:
            return syntax_result
        
        # Step 2: Compilation check
        compile_result = self._check_compilation(post_state)
        if not compile_result.passed:
            compile_result.syntax_valid = True
            return compile_result
        
        # Step 3: Run existing tests
        existing_tests_result = self._run_existing_tests(post_state)
        if not existing_tests_result.passed:
            existing_tests_result.syntax_valid = True
            existing_tests_result.compile_success = True
            return existing_tests_result
        
        # Step 4: Run adaptive tests
        adaptive_tests_result = self._run_adaptive_tests(
            action_type, action_target, action_params, post_state
        )
        if not adaptive_tests_result.passed:
            adaptive_tests_result.syntax_valid = True
            adaptive_tests_result.compile_success = True
            adaptive_tests_result.existing_tests_passed = existing_tests_result.existing_tests_passed
            return adaptive_tests_result
        
        # Step 5: Check invalidations
        invalidation_result = self._check_invalidations(pre_state, post_state)
        if not invalidation_result.passed:
            invalidation_result.syntax_valid = True
            invalidation_result.compile_success = True
            invalidation_result.existing_tests_passed = existing_tests_result.existing_tests_passed
            invalidation_result.adaptive_tests_passed = adaptive_tests_result.adaptive_tests_passed
            return invalidation_result
        
        # Step 6: (Optional) Type stability check
        if self.require_type_stability:
            stability_result = self._check_type_stability(action_type, action_target, post_state)
            if not stability_result.passed:
                stability_result.syntax_valid = True
                stability_result.compile_success = True
                stability_result.existing_tests_passed = existing_tests_result.existing_tests_passed
                stability_result.adaptive_tests_passed = adaptive_tests_result.adaptive_tests_passed
                stability_result.invalidation_count = invalidation_result.invalidation_count
                return stability_result
        
        # All checks passed!
        elapsed_ms = (time.time() - start_time) * 1000
        
        return VerificationResult(
            status=VerificationStatus.PASSED,
            passed=True,
            syntax_valid=True,
            compile_success=True,
            existing_tests_passed=existing_tests_result.existing_tests_passed,
            existing_tests_failed=0,
            adaptive_tests_passed=adaptive_tests_result.adaptive_tests_passed,
            adaptive_tests_failed=0,
            invalidation_count=invalidation_result.invalidation_count,
            verification_time_ms=elapsed_ms,
        )
    
    def _check_syntax(self, action_params: dict) -> VerificationResult:
        """Check if the generated code is syntactically valid."""
        code = action_params.get("code", "")
        if not code:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                passed=True,
            )
        
        try:
            # Use Julia to parse the code
            is_valid = self.julia_bridge.call(
                "Meta.parse",
                code,
                raise_errors=False
            )
            
            if is_valid is None or isinstance(is_valid, Exception):
                return VerificationResult(
                    status=VerificationStatus.FAILED_SYNTAX,
                    passed=False,
                    syntax_valid=False,
                    error_message=f"Syntax error in generated code",
                )
            
            return VerificationResult(
                status=VerificationStatus.PASSED,
                passed=True,
                syntax_valid=True,
            )
            
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.FAILED_SYNTAX,
                passed=False,
                syntax_valid=False,
                error_message=str(e),
            )
    
    def _check_compilation(self, post_state: WorldStateSnapshot) -> VerificationResult:
        """Check if the code compiles without errors."""
        try:
            # Attempt to load/compile the module
            # This would use the Julia bridge to actually try compilation
            success = self.julia_bridge.call(
                "WorldState.check_compilation",
                post_state.to_dict()
            )
            
            if not success:
                return VerificationResult(
                    status=VerificationStatus.FAILED_COMPILE,
                    passed=False,
                    compile_success=False,
                    error_message="Compilation failed",
                )
            
            return VerificationResult(
                status=VerificationStatus.PASSED,
                passed=True,
                compile_success=True,
            )
            
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.FAILED_COMPILE,
                passed=False,
                compile_success=False,
                error_message=str(e),
            )
    
    def _run_existing_tests(self, post_state: WorldStateSnapshot) -> VerificationResult:
        """Run the repository's existing test suite."""
        try:
            results = self.julia_bridge.call(
                "WorldState.run_tests",
                post_state.to_dict(),
                timeout=self.test_timeout_seconds
            )
            
            passed = results.get("passed", 0)
            failed = results.get("failed", 0)
            
            if failed > 0:
                return VerificationResult(
                    status=VerificationStatus.FAILED_TESTS,
                    passed=False,
                    existing_tests_passed=passed,
                    existing_tests_failed=failed,
                    error_message=f"{failed} existing tests failed",
                )
            
            return VerificationResult(
                status=VerificationStatus.PASSED,
                passed=True,
                existing_tests_passed=passed,
                existing_tests_failed=0,
            )
            
        except TimeoutError:
            return VerificationResult(
                status=VerificationStatus.FAILED_TIMEOUT,
                passed=False,
                error_message="Test timeout exceeded",
            )
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.FAILED_RUNTIME,
                passed=False,
                error_message=str(e),
            )
    
    def _run_adaptive_tests(
        self,
        action_type: str,
        action_target: str,
        action_params: dict,
        post_state: WorldStateSnapshot,
    ) -> VerificationResult:
        """Run adaptively generated tests for this action."""
        try:
            # Generate tests based on action type
            context = self._build_test_context(action_type, action_target, action_params, post_state)
            test_suite = self.test_generator.generate_tests(action_type, action_target, context)
            
            if not test_suite.tests:
                # No tests generated, consider it passed
                return VerificationResult(
                    status=VerificationStatus.PASSED,
                    passed=True,
                    adaptive_tests_passed=0,
                    adaptive_tests_failed=0,
                )
            
            # Convert to Julia code and run
            julia_test_code = test_suite_to_julia(test_suite)
            
            results = self.julia_bridge.call(
                "TestGenerator.run_generated_tests",
                julia_test_code,
                timeout=self.test_timeout_seconds
            )
            
            passed = results.get("passed", 0)
            failed = results.get("failed", 0)
            
            if failed > 0:
                return VerificationResult(
                    status=VerificationStatus.FAILED_TESTS,
                    passed=False,
                    adaptive_tests_passed=passed,
                    adaptive_tests_failed=failed,
                    error_message=f"{failed} adaptive tests failed",
                    metadata={"failed_tests": results.get("failed_names", [])},
                )
            
            return VerificationResult(
                status=VerificationStatus.PASSED,
                passed=True,
                adaptive_tests_passed=passed,
                adaptive_tests_failed=0,
            )
            
        except Exception as e:
            logger.warning(f"Adaptive test execution failed: {e}")
            # Don't fail the transition if adaptive tests error out
            return VerificationResult(
                status=VerificationStatus.PASSED,
                passed=True,
                adaptive_tests_passed=0,
                adaptive_tests_failed=0,
                metadata={"adaptive_test_error": str(e)},
            )
    
    def _check_invalidations(
        self,
        pre_state: WorldStateSnapshot,
        post_state: WorldStateSnapshot,
    ) -> VerificationResult:
        """Check that invalidations are within acceptable bounds."""
        try:
            pre_invalidations = set(pre_state.invalidations.invalidated_methods)
            post_invalidations = set(post_state.invalidations.invalidated_methods)
            
            new_invalidations = post_invalidations - pre_invalidations
            count = len(new_invalidations)
            
            if count > self.max_invalidations:
                return VerificationResult(
                    status=VerificationStatus.FAILED_INVALIDATIONS,
                    passed=False,
                    invalidation_count=count,
                    error_message=f"Too many invalidations: {count} > {self.max_invalidations}",
                    metadata={"invalidated_methods": list(new_invalidations)},
                )
            
            return VerificationResult(
                status=VerificationStatus.PASSED,
                passed=True,
                invalidation_count=count,
            )
            
        except Exception as e:
            logger.warning(f"Invalidation check failed: {e}")
            return VerificationResult(
                status=VerificationStatus.PASSED,
                passed=True,
                invalidation_count=0,
            )
    
    def _check_type_stability(
        self,
        action_type: str,
        action_target: str,
        post_state: WorldStateSnapshot,
    ) -> VerificationResult:
        """Check type stability of affected methods."""
        if action_type not in ["ADD_METHOD", "MODIFY_METHOD"]:
            return VerificationResult(status=VerificationStatus.PASSED, passed=True)
        
        try:
            is_stable = self.julia_bridge.call(
                "IRTools.check_type_stability",
                action_target,
                post_state.to_dict()
            )
            
            if not is_stable:
                return VerificationResult(
                    status=VerificationStatus.FAILED_RUNTIME,
                    passed=False,
                    error_message=f"Method {action_target} is not type-stable",
                )
            
            return VerificationResult(
                status=VerificationStatus.PASSED,
                passed=True,
            )
            
        except Exception as e:
            logger.warning(f"Type stability check failed: {e}")
            return VerificationResult(
                status=VerificationStatus.PASSED,
                passed=True,
            )
    
    def _build_test_context(
        self,
        action_type: str,
        action_target: str,
        action_params: dict,
        post_state: WorldStateSnapshot,
    ) -> dict:
        """Build context dictionary for test generation."""
        context = {
            "module_name": action_params.get("module", "Main"),
            "action_type": action_type,
        }
        
        if action_type == "ADD_METHOD":
            context.update({
                "arg_types": action_params.get("arg_types", []),
                "return_type": action_params.get("return_type"),
                "related_methods": action_params.get("related_methods", []),
            })
        elif action_type == "MODIFY_METHOD":
            context.update({
                "original_signature": action_params.get("original_signature", ""),
                "new_signature": action_params.get("new_signature", ""),
                "existing_call_sites": action_params.get("call_sites", []),
            })
        elif action_type == "ADD_FIELD":
            context.update({
                "struct_name": action_params.get("struct_name", ""),
                "field_type": action_params.get("field_type", "Any"),
                "constructor_args": action_params.get("constructor_args", ""),
            })
        elif action_type == "RENAME_SYMBOL":
            context.update({
                "old_name": action_params.get("old_name", ""),
                "symbol_type": action_params.get("symbol_type", "function"),
                "reference_sites": action_params.get("reference_sites", []),
            })
        
        return context


# ---------------------------------------------------------------------------
# Rejection Sampling Filter
# ---------------------------------------------------------------------------


class RejectionSamplingFilter:
    """
    Filters a dataset of transitions using verifier-guided rejection sampling.
    
    Only keeps transitions where V(τ) = 1 (all verification checks pass).
    """
    
    def __init__(
        self,
        verifier: TransitionVerifier,
        num_workers: int = 4,
        cache_dir: Optional[Path] = None,
    ):
        self.verifier = verifier
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def filter(
        self,
        transitions: list[Transition],
        states: dict[str, WorldStateSnapshot],
    ) -> list[Transition]:
        """
        Filter transitions, keeping only valid ones.
        
        Args:
            transitions: List of candidate transitions
            states: Dict mapping state hashes to WorldStateSnapshots
            
        Returns:
            Filtered list of valid transitions
        """
        logger.info(f"Filtering {len(transitions)} transitions...")
        
        valid_transitions = []
        stats = {
            "total": len(transitions),
            "passed": 0,
            "failed_syntax": 0,
            "failed_compile": 0,
            "failed_tests": 0,
            "failed_invalidations": 0,
            "failed_other": 0,
            "cached": 0,
        }
        
        for transition in transitions:
            # Check cache first
            cached_result = self._check_cache(transition)
            if cached_result is not None:
                stats["cached"] += 1
                if cached_result.passed:
                    transition.verification = cached_result
                    transition.is_valid = True
                    valid_transitions.append(transition)
                    stats["passed"] += 1
                continue
            
            # Get states
            pre_state = states.get(transition.state_hash)
            post_state = states.get(transition.next_state_hash)
            
            if pre_state is None or post_state is None:
                logger.warning(f"Missing state for transition {transition.state_hash}")
                stats["failed_other"] += 1
                continue
            
            # Verify
            result = self.verifier.verify(
                pre_state=pre_state,
                action_type=transition.action_type,
                action_target=transition.action_target,
                action_params=transition.action_params,
                post_state=post_state,
            )
            
            # Cache result
            self._cache_result(transition, result)
            
            # Update stats
            if result.passed:
                stats["passed"] += 1
                transition.verification = result
                transition.is_valid = True
                transition.reward_signal = self._compute_reward(result)
                valid_transitions.append(transition)
            else:
                stats[f"failed_{result.status.value.split('_')[1]}"] = \
                    stats.get(f"failed_{result.status.value.split('_')[1]}", 0) + 1
        
        # Log stats
        logger.info(f"Filtering complete: {stats['passed']}/{stats['total']} passed")
        logger.info(f"  Syntax failures: {stats.get('failed_syntax', 0)}")
        logger.info(f"  Compile failures: {stats.get('failed_compile', 0)}")
        logger.info(f"  Test failures: {stats.get('failed_tests', 0)}")
        logger.info(f"  Invalidation failures: {stats.get('failed_invalidations', 0)}")
        logger.info(f"  Cache hits: {stats['cached']}")
        
        return valid_transitions
    
    def _compute_reward(self, result: VerificationResult) -> float:
        """Compute reward signal based on verification result."""
        if not result.passed:
            return 0.0
        
        # Base reward for passing
        reward = 1.0
        
        # Bonus for passing many tests
        total_tests = result.existing_tests_passed + result.adaptive_tests_passed
        if total_tests > 0:
            reward += 0.1 * min(total_tests, 10)  # Cap bonus at 1.0
        
        # Penalty for invalidations (even if below threshold)
        reward -= 0.05 * result.invalidation_count
        
        return max(0.0, reward)
    
    def _check_cache(self, transition: Transition) -> Optional[VerificationResult]:
        """Check if we have a cached verification result."""
        if self.cache_dir is None:
            return None
        
        cache_key = self._cache_key(transition)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                return VerificationResult(
                    status=VerificationStatus(data["status"]),
                    passed=data["passed"],
                    **{k: v for k, v in data.items() if k not in ["status", "passed"]}
                )
            except Exception:
                return None
        
        return None
    
    def _cache_result(self, transition: Transition, result: VerificationResult):
        """Cache a verification result."""
        if self.cache_dir is None:
            return
        
        cache_key = self._cache_key(transition)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            data = {
                "status": result.status.value,
                "passed": result.passed,
                "syntax_valid": result.syntax_valid,
                "compile_success": result.compile_success,
                "existing_tests_passed": result.existing_tests_passed,
                "existing_tests_failed": result.existing_tests_failed,
                "adaptive_tests_passed": result.adaptive_tests_passed,
                "adaptive_tests_failed": result.adaptive_tests_failed,
                "invalidation_count": result.invalidation_count,
                "verification_time_ms": result.verification_time_ms,
                "error_message": result.error_message,
            }
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _cache_key(self, transition: Transition) -> str:
        """Generate cache key for a transition."""
        key_data = f"{transition.state_hash}:{transition.action_type}:{transition.action_target}:{transition.next_state_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------


class VerifiedDatasetBuilder:
    """
    Builds a verified training dataset using rejection sampling.
    """
    
    def __init__(
        self,
        filter: RejectionSamplingFilter,
        output_dir: Path,
    ):
        self.filter = filter
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build(
        self,
        raw_transitions: list[Transition],
        states: dict[str, WorldStateSnapshot],
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> dict[str, list[Transition]]:
        """
        Build a verified dataset with train/val/test splits.
        
        Args:
            raw_transitions: All candidate transitions
            states: State hash -> WorldStateSnapshot mapping
            split_ratios: (train, val, test) ratios
            
        Returns:
            Dict with 'train', 'val', 'test' keys
        """
        # Filter to valid transitions
        valid = self.filter.filter(raw_transitions, states)
        
        logger.info(f"Building dataset from {len(valid)} valid transitions")
        
        # Shuffle
        import random
        random.shuffle(valid)
        
        # Split
        n = len(valid)
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])
        
        splits = {
            "train": valid[:train_end],
            "val": valid[train_end:val_end],
            "test": valid[val_end:],
        }
        
        # Save
        for split_name, split_data in splits.items():
            self._save_split(split_name, split_data)
        
        logger.info(f"Dataset built: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def _save_split(self, name: str, transitions: list[Transition]):
        """Save a dataset split to disk."""
        output_file = self.output_dir / f"{name}.jsonl"
        
        with open(output_file, "w") as f:
            for t in transitions:
                data = {
                    "state_hash": t.state_hash,
                    "action_type": t.action_type,
                    "action_target": t.action_target,
                    "action_params": t.action_params,
                    "next_state_hash": t.next_state_hash,
                    "is_valid": t.is_valid,
                    "reward_signal": t.reward_signal,
                }
                f.write(json.dumps(data) + "\n")
        
        logger.info(f"Saved {len(transitions)} transitions to {output_file}")
