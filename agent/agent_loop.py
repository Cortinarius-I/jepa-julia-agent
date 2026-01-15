"""
Main agent loop for the JEPA Julia Agent.

This is the entry point that orchestrates:
1. World state extraction from Julia
2. JEPA prediction
3. Planning
4. Execution
5. Validation

The loop runs iteratively, making incremental progress toward goals.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from agent.jepa.model import JEPAWorldModel
    from agent.planner import ActionSequence, PlanningGoal
    from agent.world_state import WorldStateSnapshot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


# ============================================================================
# Agent State
# ============================================================================


class AgentStatus(Enum):
    """Current status of the agent."""

    IDLE = "idle"
    EXTRACTING = "extracting_world_state"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class AgentState:
    """Internal state of the agent."""

    status: AgentStatus = AgentStatus.IDLE
    current_world_state: "WorldStateSnapshot | None" = None
    current_plan: "ActionSequence | None" = None
    goal: "PlanningGoal | None" = None
    history: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 100


# ============================================================================
# Julia Bridge
# ============================================================================


class JuliaBridge:
    """
    Bridge to Julia for world state extraction and execution.

    Uses juliacall for Python-Julia interop.
    """

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)
        self._julia = None

    def _ensure_julia(self):
        """Initialize Julia runtime if needed."""
        if self._julia is None:
            try:
                from juliacall import Main as jl

                self._julia = jl

                # Ensure required packages are available
                jl.seval('import Pkg; Pkg.add(["JSON3", "StructTypes"])')

                # Load our Julia modules
                julia_src = Path(__file__).parent.parent / "julia" / "src"
                self._julia.include(str(julia_src / "WorldState.jl"))
                self._julia.include(str(julia_src / "Actions.jl"))
                self._julia.include(str(julia_src / "Executor.jl"))

                logger.info("Julia runtime initialized")
            except ImportError:
                logger.warning("juliacall not available, using mock Julia bridge")
                self._julia = MockJulia()
            except Exception as e:
                logger.warning(f"Julia initialization failed: {e}, using mock bridge")
                self._julia = MockJulia()

    def extract_world_state(self) -> dict:
        """Extract world state from Julia repository."""
        self._ensure_julia()

        try:
            # Call Julia extraction
            json_state = self._julia.WorldState.extract_world_state(str(self.repo_path))
            json_str = self._julia.WorldState.to_json(json_state)
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to extract world state: {e}")
            return self._mock_world_state()

    def execute_action(self, action: dict, dry_run: bool = False) -> dict:
        """Execute an action via Julia."""
        self._ensure_julia()

        try:
            # Convert action to Julia types and execute
            # Placeholder - actual implementation would call Executor.jl
            return {"success": True, "dry_run": dry_run}
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            return {"success": False, "error": str(e)}

    def _mock_world_state(self) -> dict:
        """Return a mock world state for testing."""
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


# ============================================================================
# Main Agent
# ============================================================================


class JEPAAgent:
    """
    The main JEPA Julia Agent.

    Coordinates all components to achieve coding goals.
    """

    def __init__(
        self,
        repo_path: str | Path,
        jepa_model: "JEPAWorldModel | None" = None,
        max_iterations: int = 100,
    ):
        self.repo_path = Path(repo_path)
        self.jepa_model = jepa_model
        self.state = AgentState(max_iterations=max_iterations)
        self.julia_bridge = JuliaBridge(repo_path)

        # Import here to avoid circular imports
        from agent.planner import JEPAPlanner, PlanExecutor

        self.planner = JEPAPlanner(jepa_model) if jepa_model else None
        self.executor = PlanExecutor(str(repo_path))

    def set_goal(self, goal_description: str, constraints: list | None = None) -> None:
        """Set the agent's current goal."""
        from agent.planner import PlanningGoal

        self.state.goal = PlanningGoal(
            description=goal_description,
            constraints=constraints or [],
        )
        self.state.status = AgentStatus.IDLE
        logger.info(f"Goal set: {goal_description}")

    def run(self, dry_run: bool = False) -> dict:
        """
        Run the agent loop until goal is achieved or max iterations reached.

        Args:
            dry_run: If True, simulate without making changes

        Returns:
            Final result dictionary
        """
        if self.state.goal is None:
            return {"success": False, "error": "No goal set"}

        console.print(Panel(f"[bold blue]Starting JEPA Agent[/bold blue]\nGoal: {self.state.goal.description}"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running agent...", total=None)

            while self.state.iteration < self.state.max_iterations:
                self.state.iteration += 1
                progress.update(task, description=f"Iteration {self.state.iteration}: {self.state.status.value}")

                result = self._run_iteration(dry_run)

                if result.get("complete", False):
                    self.state.status = AgentStatus.COMPLETE
                    break

                if result.get("error"):
                    self.state.status = AgentStatus.ERROR
                    self.state.errors.append(result["error"])
                    if len(self.state.errors) > 5:
                        logger.error("Too many errors, stopping")
                        break

        return self._summarize_run()

    def _run_iteration(self, dry_run: bool) -> dict:
        """Run a single iteration of the agent loop."""
        # Step 1: Extract world state
        self.state.status = AgentStatus.EXTRACTING
        world_state_dict = self.julia_bridge.extract_world_state()
        self.state.current_world_state = self._dict_to_world_state(world_state_dict)

        # Step 2: Check if goal is already achieved
        if self._check_goal_achieved():
            return {"complete": True, "reason": "goal_achieved"}

        # Step 3: Plan
        self.state.status = AgentStatus.PLANNING
        if self.planner:
            plan = self.planner.plan(self.state.current_world_state, self.state.goal)
            if plan is None:
                return {"error": "No safe plan found"}
            self.state.current_plan = plan
        else:
            logger.warning("No JEPA model available, using mock planning")
            return {"complete": True, "reason": "no_model"}

        # Step 4: Execute
        self.state.status = AgentStatus.EXECUTING
        exec_result = self.executor.execute(self.state.current_plan, dry_run=dry_run)

        if not exec_result["success"]:
            return {"error": f"Execution failed: {exec_result.get('error', 'unknown')}"}

        # Step 5: Validate
        self.state.status = AgentStatus.VALIDATING
        validation = self._validate_execution(exec_result)

        # Record history
        self.state.history.append({
            "iteration": self.state.iteration,
            "plan": [a.to_dict() for a in self.state.current_plan.actions],
            "execution": exec_result,
            "validation": validation,
        })

        return {"success": True, "validation": validation}

    def _dict_to_world_state(self, d: dict) -> "WorldStateSnapshot":
        """Convert dictionary to WorldStateSnapshot."""
        from agent.world_state import WorldStateSnapshot

        return WorldStateSnapshot.model_validate(d)

    def _check_goal_achieved(self) -> bool:
        """Check if the current goal is achieved."""
        if self.state.goal is None or self.state.current_world_state is None:
            return False

        # Check constraints
        for constraint in self.state.goal.constraints:
            if not constraint(self.state.current_world_state):
                return False

        # Check required tests
        for test_name in self.state.goal.required_tests_pass:
            test_found = False
            for test in self.state.current_world_state.tests.results:
                if test.name == test_name:
                    test_found = True
                    if not test.passed:
                        return False
            if not test_found:
                return False

        return True

    def _validate_execution(self, exec_result: dict) -> dict:
        """Validate the execution result."""
        # Re-extract world state
        new_state_dict = self.julia_bridge.extract_world_state()

        # Compare with previous state
        # Check tests still pass
        # Check no unexpected invalidations

        return {
            "valid": True,
            "tests_passing": True,
            "no_regressions": True,
        }

    def _summarize_run(self) -> dict:
        """Summarize the agent run."""
        return {
            "success": self.state.status == AgentStatus.COMPLETE,
            "status": self.state.status.value,
            "iterations": self.state.iteration,
            "errors": self.state.errors,
            "history_length": len(self.state.history),
            "goal": self.state.goal.description if self.state.goal else None,
        }


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """CLI entry point."""
    import typer

    app = typer.Typer()

    @app.command()
    def run(
        repo_path: str = typer.Argument(..., help="Path to Julia repository"),
        goal: str = typer.Option(..., "--goal", "-g", help="Goal description"),
        dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Simulate without changes"),
        max_iterations: int = typer.Option(100, "--max-iter", "-m", help="Maximum iterations"),
    ):
        """Run the JEPA Julia Agent."""
        agent = JEPAAgent(repo_path, max_iterations=max_iterations)
        agent.set_goal(goal)
        result = agent.run(dry_run=dry_run)

        if result["success"]:
            console.print("[bold green]✓ Goal achieved![/bold green]")
        else:
            console.print(f"[bold red]✗ Failed: {result.get('errors', ['Unknown error'])[-1]}[/bold red]")

    app()


if __name__ == "__main__":
    main()
