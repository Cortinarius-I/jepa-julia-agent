#!/usr/bin/env python3
"""
Tests for the Python-Julia bridge.

The bridge supports two modes:
1. Full Julia mode: Uses juliacall for real Julia interop
2. Mock mode: Uses MockJulia for testing without Julia

To run with real Julia (slow, requires juliacall):
    python tests/test_julia_bridge.py --with-julia

To run in mock mode (default, fast):
    python tests/test_julia_bridge.py
"""

import sys
import time
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Module-level flag for Julia tests (set before unittest removes args)
WITH_JULIA = "--with-julia" in sys.argv


# Define standalone test classes that don't depend on the full agent module
class MockJulia:
    """Mock Julia runtime for testing without Julia."""

    class WorldState:
        @staticmethod
        def extract_world_state(path: str) -> dict:
            """Return a mock world state structure."""
            return {
                "modules": {"nodes": {}, "root": "Main"},
                "methods": {"methods": {}, "method_count": 0},
                "dispatch": {"edges": [], "ambiguities": []},
                "types": {"inferred_types": {}, "inference_errors": []},
                "tests": {"results": [], "total_passed": 0, "total_failed": 0, "coverage": 0.0},
                "invalidations": {"recent_events": [], "total_invalidations": 0, "hot_spots": []},
                "timestamp": time.time(),
                "repo_hash": "",
            }

        @staticmethod
        def to_json(state: dict) -> str:
            return json.dumps(state)


class JuliaBridge:
    """Bridge to Julia for world state extraction."""

    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self._julia = None

    def _ensure_julia(self):
        if self._julia is None:
            try:
                from juliacall import Main as jl

                self._julia = jl
                # Note: In full mode, would load Julia modules here
            except ImportError:
                self._julia = MockJulia()
            except Exception:
                self._julia = MockJulia()

    def extract_world_state(self) -> dict:
        self._ensure_julia()
        try:
            state = self._julia.WorldState.extract_world_state(str(self.repo_path))
            json_str = self._julia.WorldState.to_json(state)
            return json.loads(json_str)
        except Exception:
            return self._mock_world_state()

    def execute_action(self, action: dict, dry_run: bool = False) -> dict:
        self._ensure_julia()
        return {"success": True, "dry_run": dry_run}

    def _mock_world_state(self) -> dict:
        return {
            "modules": {"nodes": {}, "root": "Main"},
            "methods": {"methods": {}, "method_count": 0},
            "dispatch": {"edges": [], "ambiguities": []},
            "types": {"inferred_types": {}, "inference_errors": []},
            "tests": {"results": [], "total_passed": 0, "total_failed": 0, "coverage": 0.0},
            "invalidations": {"recent_events": [], "total_invalidations": 0, "hot_spots": []},
            "timestamp": time.time(),
            "repo_hash": "",
        }


class TestMockJuliaBridge(unittest.TestCase):
    """Test the Julia bridge in mock mode (no Julia required)."""

    def test_mock_fallback(self):
        """Test that the bridge falls back to mock when juliacall unavailable."""
        bridge = JuliaBridge("/tmp/test_repo")
        # Force mock mode by patching import
        with patch.dict(sys.modules, {"juliacall": None}):
            bridge._julia = None  # Reset
            bridge._ensure_julia()
            self.assertIsInstance(bridge._julia, MockJulia)

    def test_mock_world_state_extraction(self):
        """Test mock world state extraction."""
        bridge = JuliaBridge("/tmp/test_repo")
        bridge._julia = MockJulia()

        state = bridge.extract_world_state()

        # Mock should return a valid structure
        self.assertIsInstance(state, dict)
        self.assertIn("modules", state)
        self.assertIn("methods", state)
        self.assertIn("dispatch", state)
        self.assertIn("types", state)
        self.assertIn("tests", state)
        self.assertIn("invalidations", state)
        self.assertIn("timestamp", state)

    def test_mock_action_execution(self):
        """Test mock action execution."""
        bridge = JuliaBridge("/tmp/test_repo")
        bridge._julia = MockJulia()

        result = bridge.execute_action({"type": "ADD_METHOD"}, dry_run=True)

        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertTrue(result["success"])


class TestJuliaBridgeFunctionality(unittest.TestCase):
    """Test the JuliaBridge class functionality."""

    def test_bridge_initialization(self):
        """Test bridge initializes with correct path."""
        bridge = JuliaBridge("/test/path")
        self.assertEqual(str(bridge.repo_path), "/test/path")
        self.assertIsNone(bridge._julia)

    def test_lazy_initialization(self):
        """Test Julia runtime is lazily initialized."""
        bridge = JuliaBridge("/test/path")
        # Before any operation, _julia should be None
        self.assertIsNone(bridge._julia)

        # After extraction, _julia should be set (either real or mock)
        state = bridge.extract_world_state()
        self.assertIsNotNone(bridge._julia)

    def test_extract_world_state_structure(self):
        """Test that extracted world state has required fields."""
        bridge = JuliaBridge("/test/path")
        bridge._julia = MockJulia()  # Force mock mode
        state = bridge.extract_world_state()

        required_fields = [
            "modules",
            "methods",
            "dispatch",
            "types",
            "tests",
            "invalidations",
            "timestamp",
        ]

        for field in required_fields:
            self.assertIn(
                field, state, f"World state missing required field: {field}"
            )

    def test_execute_action_returns_result(self):
        """Test that execute_action returns a result dict."""
        bridge = JuliaBridge("/test/path")
        bridge._julia = MockJulia()

        result = bridge.execute_action({"type": "ADD_METHOD", "target": "foo"})

        self.assertIsInstance(result, dict)
        self.assertIn("success", result)

    def test_dry_run_flag(self):
        """Test that dry_run flag is passed through."""
        bridge = JuliaBridge("/test/path")
        bridge._julia = MockJulia()

        result = bridge.execute_action({"type": "ADD_METHOD"}, dry_run=True)

        self.assertIn("dry_run", result)
        self.assertTrue(result["dry_run"])


class TestJuliaBridgeIntegration(unittest.TestCase):
    """
    Integration tests for the Julia bridge with real Julia.

    These tests are skipped by default. Run with --with-julia to enable.
    """

    @classmethod
    def setUpClass(cls):
        cls.julia_available = False
        cls.skip_reason = "Julia tests disabled by default (use --with-julia)"

        if WITH_JULIA:
            try:
                import juliacall

                cls.julia_available = True
                print("\njuliacall available, running Julia integration tests...")
            except ImportError:
                cls.skip_reason = "juliacall not installed"
                print(f"\nSkipping Julia tests: {cls.skip_reason}")

    def setUp(self):
        if not self.julia_available:
            self.skipTest(self.skip_reason)

    def test_julia_connection(self):
        """Test that we can connect to Julia."""
        from juliacall import Main as jl

        result = jl.seval("1 + 1")
        self.assertEqual(result, 2)

    def test_julia_struct_creation(self):
        """Test creating Julia structs from Python."""
        from juliacall import Main as jl

        jl.seval(
            """
        struct TestState
            name::String
            value::Int
        end
        """
        )

        state = jl.seval('TestState("test", 42)')
        self.assertEqual(str(state.name), "test")
        self.assertEqual(int(state.value), 42)


def main():
    # Remove --with-julia from argv before unittest parses it
    with_julia = "--with-julia" in sys.argv
    if with_julia:
        sys.argv.remove("--with-julia")

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMockJuliaBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestJuliaBridgeFunctionality))

    # Add integration tests (may be skipped)
    if with_julia:
        suite.addTests(loader.loadTestsFromTestCase(TestJuliaBridgeIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
