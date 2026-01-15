"""
Planner for the JEPA Julia Agent.

The planner:
1. Takes a high-level goal (e.g., "add feature X", "fix bug Y")
2. Searches over possible action sequences
3. Uses JEPA to predict outcomes of each sequence
4. Selects the safest path that achieves the goal

Key principle: Planning happens in EMBEDDING SPACE.
We never generate code during planning - only when executing.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from agent.jepa.model import JEPAWorldModel
    from agent.world_state import WorldStateSnapshot


# ============================================================================
# Action Types (matching Julia side)
# ============================================================================


class ActionType(Enum):
    """Types of actions the agent can take."""

    ADD_METHOD = "add_method"
    MODIFY_METHOD = "modify_method"
    REMOVE_METHOD = "remove_method"
    ADD_FIELD = "add_field"
    MODIFY_FIELD = "modify_field"
    REMOVE_FIELD = "remove_field"
    ADD_IMPORT = "add_import"
    REMOVE_IMPORT = "remove_import"
    RENAME_SYMBOL = "rename_symbol"
    MOVE_DEFINITION = "move_definition"
    ADD_TEST = "add_test"
    MODIFY_TEST = "modify_test"
    REMOVE_TEST = "remove_test"


# ============================================================================
# Action Representation
# ============================================================================


@dataclass
class PlannedAction:
    """An action proposed by the planner."""

    action_type: ActionType
    target_module: str
    target_symbol: str | None = None
    parameters: dict = field(default_factory=dict)
    priority: float = 1.0  # Higher = more important

    def to_dict(self) -> dict:
        """Convert to dictionary for JEPA encoding."""
        return {
            "type": self.action_type.value,
            "type_id": list(ActionType).index(self.action_type),
            "target_module": self.target_module,
            "target_symbol": self.target_symbol,
            "parameters": self.parameters,
        }


@dataclass
class ActionSequence:
    """A sequence of actions forming a plan."""

    actions: list[PlannedAction]
    predicted_safety: float = 1.0
    predicted_test_pass_rate: float = 1.0
    goal_distance: float = float("inf")  # Distance to goal in embedding space

    @property
    def score(self) -> float:
        """Combined score for ranking plans."""
        # Balance safety, test success, and goal achievement
        return self.predicted_safety * 0.4 + self.predicted_test_pass_rate * 0.3 + (1.0 / (1.0 + self.goal_distance)) * 0.3


# ============================================================================
# Goal Representation
# ============================================================================


@dataclass
class PlanningGoal:
    """A goal for the planner to achieve."""

    description: str
    target_embedding: np.ndarray | None = None  # Target state in embedding space
    constraints: list[Callable[[WorldStateSnapshot], bool]] = field(default_factory=list)
    required_tests_pass: list[str] = field(default_factory=list)
    max_actions: int = 10


# ============================================================================
# Planner Implementation
# ============================================================================


class JEPAPlanner:
    """
    Plans action sequences using JEPA predictions.

    Uses beam search in embedding space to find safe, effective plans.
    """

    def __init__(
        self,
        jepa_model: "JEPAWorldModel",
        beam_width: int = 5,
        max_depth: int = 10,
        safety_threshold: float = 0.7,
    ):
        self.jepa = jepa_model
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.safety_threshold = safety_threshold

        # Action generators for different situations
        self.action_generators: list[Callable] = []

    def plan(
        self,
        current_state: "WorldStateSnapshot",
        goal: PlanningGoal,
    ) -> ActionSequence | None:
        """
        Find a plan to achieve the goal from current state.

        Args:
            current_state: Current world state
            goal: Goal to achieve

        Returns:
            Best action sequence, or None if no safe plan found
        """
        # Initialize beam with empty sequence
        initial = ActionSequence(
            actions=[],
            predicted_safety=1.0,
            predicted_test_pass_rate=1.0,
            goal_distance=self._compute_goal_distance(current_state, goal),
        )

        beam = [initial]
        best_complete = None

        for depth in range(self.max_depth):
            candidates = []

            for sequence in beam:
                # Generate possible next actions
                possible_actions = self._generate_actions(current_state, sequence, goal)

                for action in possible_actions:
                    # Predict outcome using JEPA
                    extended = self._extend_sequence(sequence, action, current_state, goal)

                    if extended.predicted_safety >= self.safety_threshold:
                        candidates.append(extended)

                        # Check if goal is achieved
                        if self._is_goal_achieved(extended, goal):
                            if best_complete is None or extended.score > best_complete.score:
                                best_complete = extended

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x.score, reverse=True)
            beam = candidates[: self.beam_width]

            if not beam:
                break

            # Early termination if we have a good complete plan
            if best_complete and best_complete.score > 0.9:
                break

        return best_complete

    def _generate_actions(
        self,
        state: "WorldStateSnapshot",
        current_sequence: ActionSequence,
        goal: PlanningGoal,
    ) -> list[PlannedAction]:
        """Generate candidate actions to extend the current sequence."""
        actions = []

        # Use registered action generators
        for generator in self.action_generators:
            actions.extend(generator(state, current_sequence, goal))

        # Default action generation based on goal analysis
        if not actions:
            actions = self._default_action_generation(state, goal)

        return actions

    def _default_action_generation(
        self,
        state: "WorldStateSnapshot",
        goal: PlanningGoal,
    ) -> list[PlannedAction]:
        """Default action generation when no specific generators match."""
        actions = []

        # Analyze goal description for action hints
        desc_lower = goal.description.lower()

        if "add" in desc_lower or "create" in desc_lower or "implement" in desc_lower:
            # Suggest add actions
            actions.append(
                PlannedAction(
                    action_type=ActionType.ADD_METHOD,
                    target_module="Main",
                    parameters={"goal_hint": goal.description},
                )
            )

        if "fix" in desc_lower or "modify" in desc_lower or "change" in desc_lower:
            # Suggest modify actions
            actions.append(
                PlannedAction(
                    action_type=ActionType.MODIFY_METHOD,
                    target_module="Main",
                    parameters={"goal_hint": goal.description},
                )
            )

        if "test" in desc_lower:
            actions.append(
                PlannedAction(
                    action_type=ActionType.ADD_TEST,
                    target_module="Test",
                    parameters={"goal_hint": goal.description},
                )
            )

        return actions

    def _extend_sequence(
        self,
        sequence: ActionSequence,
        action: PlannedAction,
        current_state: "WorldStateSnapshot",
        goal: PlanningGoal,
    ) -> ActionSequence:
        """Extend a sequence with a new action and predict outcome."""
        # Use JEPA to predict outcome
        prediction = self.jepa.predict_outcome(current_state, action.to_dict())

        new_actions = sequence.actions + [action]

        return ActionSequence(
            actions=new_actions,
            predicted_safety=min(sequence.predicted_safety, prediction["safety_probs"][0]),
            predicted_test_pass_rate=prediction["predicted_test_pass_rate"],
            goal_distance=self._embedding_distance(
                prediction["predicted_embedding"], goal.target_embedding
            )
            if goal.target_embedding is not None
            else sequence.goal_distance * 0.9,  # Assume progress
        )

    def _compute_goal_distance(
        self,
        state: "WorldStateSnapshot",
        goal: PlanningGoal,
    ) -> float:
        """Compute distance from current state to goal."""
        if goal.target_embedding is None:
            return float("inf")

        current_embedding = state.to_embedding()
        return self._embedding_distance(current_embedding, goal.target_embedding)

    def _embedding_distance(
        self,
        embedding1: np.ndarray | None,
        embedding2: np.ndarray | None,
    ) -> float:
        """Compute distance between two embeddings."""
        if embedding1 is None or embedding2 is None:
            return float("inf")
        return float(np.linalg.norm(embedding1 - embedding2))

    def _is_goal_achieved(
        self,
        sequence: ActionSequence,
        goal: PlanningGoal,
    ) -> bool:
        """Check if the sequence achieves the goal."""
        # Check action count
        if len(sequence.actions) > goal.max_actions:
            return False

        # Check goal distance threshold
        if sequence.goal_distance < 0.1:  # Close enough to goal
            return True

        return False

    def register_action_generator(self, generator: Callable) -> None:
        """Register a custom action generator."""
        self.action_generators.append(generator)


# ============================================================================
# Plan Executor
# ============================================================================


class PlanExecutor:
    """
    Executes planned action sequences.

    This is where transformers get invoked for code generation.
    The executor:
    1. Takes an ActionSequence from the planner
    2. Executes each action in order
    3. Validates results after each step
    4. Rolls back if something goes wrong
    """

    def __init__(self, repo_path: str, transformer_endpoint: str = "http://localhost:8000/render"):
        self.repo_path = repo_path
        self.transformer_endpoint = transformer_endpoint
        self.execution_history: list[dict] = []

    def execute(
        self,
        plan: ActionSequence,
        dry_run: bool = False,
    ) -> dict:
        """
        Execute a planned action sequence.

        Args:
            plan: The action sequence to execute
            dry_run: If True, simulate without making changes

        Returns:
            Execution result with success status and details
        """
        results = []
        rollback_stack = []

        for i, action in enumerate(plan.actions):
            try:
                result = self._execute_single_action(action, dry_run)
                results.append(result)

                if not result["success"]:
                    # Execution failed - roll back
                    self._rollback(rollback_stack)
                    return {
                        "success": False,
                        "failed_at": i,
                        "error": result["error"],
                        "results": results,
                    }

                rollback_stack.append(result["rollback_info"])

            except Exception as e:
                self._rollback(rollback_stack)
                return {
                    "success": False,
                    "failed_at": i,
                    "error": str(e),
                    "results": results,
                }

        return {
            "success": True,
            "results": results,
        }

    def _execute_single_action(self, action: PlannedAction, dry_run: bool) -> dict:
        """Execute a single action."""
        # This would call the Julia Executor
        # For now, return a placeholder
        return {
            "success": True,
            "action": action.to_dict(),
            "dry_run": dry_run,
            "rollback_info": {"action": action},
        }

    def _rollback(self, rollback_stack: list[dict]) -> None:
        """Roll back executed actions in reverse order."""
        for rollback_info in reversed(rollback_stack):
            # Execute inverse action
            pass


# ============================================================================
# High-Level API
# ============================================================================


def plan_and_execute(
    jepa_model: "JEPAWorldModel",
    current_state: "WorldStateSnapshot",
    goal_description: str,
    repo_path: str,
    dry_run: bool = False,
) -> dict:
    """
    High-level API for planning and executing code changes.

    Args:
        jepa_model: Trained JEPA model
        current_state: Current world state
        goal_description: Natural language description of goal
        repo_path: Path to Julia repository
        dry_run: If True, simulate without making changes

    Returns:
        Execution result
    """
    # Create goal
    goal = PlanningGoal(
        description=goal_description,
        max_actions=10,
    )

    # Plan
    planner = JEPAPlanner(jepa_model)
    plan = planner.plan(current_state, goal)

    if plan is None:
        return {
            "success": False,
            "error": "No safe plan found",
        }

    # Execute
    executor = PlanExecutor(repo_path)
    result = executor.execute(plan, dry_run=dry_run)

    return result
