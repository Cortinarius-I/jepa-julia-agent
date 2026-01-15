"""
Transition Mining from Git History.

This script extracts (state, action, state') transitions from Julia package
git histories. It focuses on commits that represent single semantic actions
suitable for JEPA training.

Key principles:
1. Only extract commits with small, focused changes
2. Filter to commits that modify .jl files in src/
3. Infer the semantic action from the diff
4. Validate transitions (syntax, parseable)
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Path to Julia validator script
JULIA_VALIDATOR = Path(__file__).parent / "validate_julia.jl"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MiningConfig:
    """Configuration for transition mining."""
    repo_path: Path
    output_path: Path
    max_files_changed: int = 3  # Skip commits touching too many files
    max_lines_changed: int = 100  # Skip large refactors
    min_lines_changed: int = 1  # Skip empty/trivial commits
    skip_merge_commits: bool = True
    skip_docs_only: bool = True
    source_dirs: tuple[str, ...] = ("src/",)
    test_dirs: tuple[str, ...] = ("test/",)
    use_julia_validation: bool = True  # Validate Julia syntax (slower but more accurate)


# ============================================================================
# Action Types (matching julia/src/Actions.jl)
# ============================================================================

class ActionType(Enum):
    ADD_METHOD = "ADD_METHOD"
    MODIFY_METHOD = "MODIFY_METHOD"
    REMOVE_METHOD = "REMOVE_METHOD"
    ADD_FIELD = "ADD_FIELD"
    MODIFY_FIELD = "MODIFY_FIELD"
    REMOVE_FIELD = "REMOVE_FIELD"
    ADD_IMPORT = "ADD_IMPORT"
    REMOVE_IMPORT = "REMOVE_IMPORT"
    RENAME_SYMBOL = "RENAME_SYMBOL"
    MOVE_DEFINITION = "MOVE_DEFINITION"
    ADD_TEST = "ADD_TEST"
    MODIFY_TEST = "MODIFY_TEST"
    REMOVE_TEST = "REMOVE_TEST"
    ADD_TYPE = "ADD_TYPE"
    MODIFY_TYPE = "MODIFY_TYPE"
    # New action types for better coverage
    MODIFY_DOCS = "MODIFY_DOCS"  # Docstring/comment changes
    MODIFY_MACRO = "MODIFY_MACRO"  # Macro annotation changes
    FIX_TYPO = "FIX_TYPO"  # Small fixes (typos, whitespace)
    UNKNOWN = "UNKNOWN"


@dataclass
class InferredAction:
    """An action inferred from a git diff."""
    action_type: ActionType
    target_file: str
    target_symbol: str | None = None
    signature: str | None = None
    diff_hunk: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "type": self.action_type.value,
            "target_file": self.target_file,
            "target_symbol": self.target_symbol,
            "signature": self.signature,
            "confidence": self.confidence,
        }


# ============================================================================
# Git History Walker
# ============================================================================

@dataclass
class CommitInfo:
    """Information about a single commit."""
    sha: str
    parent_sha: str | None
    author: str
    date: datetime
    message: str
    files_changed: list[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0


class GitHistoryWalker:
    """Walks through git history extracting commit information."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def _run_git(self, *args: str) -> str:
        """Run a git command and return output."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git command failed: {result.stderr}")
        return result.stdout

    def get_commits(self, limit: int | None = None) -> Iterator[CommitInfo]:
        """Iterate through commits from newest to oldest."""
        # Get commit list with stats
        format_str = "%H|%P|%an|%aI|%s"
        cmd = ["log", f"--format={format_str}", "--numstat"]
        if limit:
            cmd.append(f"-n{limit}")

        output = self._run_git(*cmd)

        current_commit = None
        for line in output.strip().split("\n"):
            if not line:
                continue

            if "|" in line and line.count("|") >= 4:
                # This is a commit header line
                if current_commit:
                    yield current_commit

                parts = line.split("|")
                sha = parts[0]
                parents = parts[1].split() if parts[1] else []
                parent_sha = parents[0] if parents else None

                current_commit = CommitInfo(
                    sha=sha,
                    parent_sha=parent_sha,
                    author=parts[2],
                    date=datetime.fromisoformat(parts[3]),
                    message=parts[4],
                )
            elif current_commit and "\t" in line:
                # This is a numstat line: insertions\tdeletions\tfilename
                parts = line.split("\t")
                if len(parts) == 3:
                    ins, dels, filename = parts
                    if ins != "-":  # Binary files show "-"
                        current_commit.insertions += int(ins)
                        current_commit.deletions += int(dels)
                    current_commit.files_changed.append(filename)

        if current_commit:
            yield current_commit

    def get_diff(self, sha: str, parent_sha: str | None = None) -> str:
        """Get the diff for a specific commit."""
        if parent_sha:
            return self._run_git("diff", parent_sha, sha)
        else:
            # First commit - diff against empty tree
            return self._run_git("diff", "4b825dc642cb6eb9a060e54bf8d69288fbee4904", sha)

    def get_file_at_commit(self, sha: str, file_path: str) -> str | None:
        """Get file contents at a specific commit."""
        try:
            return self._run_git("show", f"{sha}:{file_path}")
        except RuntimeError:
            return None

    def checkout_commit(self, sha: str) -> None:
        """Checkout a specific commit (detached HEAD)."""
        self._run_git("checkout", sha, "--quiet")

    def get_current_sha(self) -> str:
        """Get current HEAD sha."""
        return self._run_git("rev-parse", "HEAD").strip()


# ============================================================================
# Julia Syntax Validation
# ============================================================================

def validate_julia_syntax(code: str) -> tuple[bool, str | None, list[str], list[str]]:
    """
    Validate Julia code syntax using the Julia validator script.

    Returns: (is_valid, error_message, functions, types)
    """
    try:
        result = subprocess.run(
            ["julia", str(JULIA_VALIDATOR)],
            input=code,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, f"Julia validator failed: {result.stderr}", [], []

        data = json.loads(result.stdout.strip())
        return (
            data["valid"],
            data.get("parse_error"),
            data.get("functions", []),
            data.get("types", []),
        )
    except subprocess.TimeoutExpired:
        return False, "Julia validation timed out", [], []
    except Exception as e:
        return False, f"Validation error: {e}", [], []


# ============================================================================
# Action Inference from Diffs
# ============================================================================

class ActionInferrer:
    """
    Infers semantic actions from git diffs.

    Improved heuristics:
    - Detects function definitions including qualified names (Base.show)
    - Handles short-form functions (f(x) = ...)
    - Detects macro definitions (@macro)
    - Detects const assignments
    - Better handling of multiline function signatures
    - Detects docstrings associated with definitions
    """

    # Regex patterns for Julia constructs
    # Function patterns - handles various forms
    FUNC_DEF_PATTERNS = [
        # function name(args...) or function Module.name(args...)
        re.compile(r"^\s*function\s+([\w.]+)\s*\("),
        # function name(args...) where {T}
        re.compile(r"^\s*function\s+([\w.]+)\s*\{"),
        # Short form: name(args...) = expr
        re.compile(r"^\s*([\w.]+)\s*\([^)]*\)\s*=(?!=)"),
        # Short form with where: name(args...) where T = expr
        re.compile(r"^\s*([\w.]+)\s*\([^)]*\)\s+where\s+"),
        # Macro definition
        re.compile(r"^\s*macro\s+(\w+)\s*\("),
    ]

    # Type/struct patterns
    TYPE_DEF_PATTERNS = [
        re.compile(r"^\s*(mutable\s+)?struct\s+(\w+)"),
        re.compile(r"^\s*abstract\s+type\s+(\w+)"),
        re.compile(r"^\s*primitive\s+type\s+(\w+)"),
    ]

    # Const pattern
    CONST_PATTERN = re.compile(r"^\s*const\s+(\w+)\s*=")

    # Import/export patterns
    IMPORT_PATTERNS = [
        re.compile(r"^\s*using\s+"),
        re.compile(r"^\s*import\s+"),
        re.compile(r"^\s*export\s+"),
    ]

    # Field pattern (inside struct)
    FIELD_PATTERN = re.compile(r"^\s*(\w+)\s*::\s*[\w{},\s]+")

    # Macro patterns (common Julia macros)
    MACRO_PATTERNS = [
        re.compile(r"^\s*@deprecate\s+"),
        re.compile(r"^\s*@inline\s+"),
        re.compile(r"^\s*@noinline\s+"),
        re.compile(r"^\s*@boundscheck\s+"),
        re.compile(r"^\s*@inbounds\s+"),
        re.compile(r"^\s*@simd\s+"),
        re.compile(r"^\s*@assert\s+"),
        re.compile(r"^\s*@debug\s+"),
        re.compile(r"^\s*@warn\s+"),
        re.compile(r"^\s*@error\s+"),
        re.compile(r"^\s*@test\s+"),
        re.compile(r"^\s*@testset\s+"),
    ]

    # Docstring pattern (triple-quoted strings)
    DOCSTRING_PATTERN = re.compile(r'^\s*"""')

    def infer_action(self, diff: str, filename: str) -> InferredAction:
        """Infer the semantic action from a diff hunk."""
        lines = diff.split("\n")

        added_lines = [l[1:] for l in lines if l.startswith("+") and not l.startswith("+++")]
        removed_lines = [l[1:] for l in lines if l.startswith("-") and not l.startswith("---")]

        # Try each detector in order of specificity
        detectors = [
            self._check_function_change,
            self._check_type_change,
            self._check_const_change,
            self._check_import_change,
            self._check_macro_change,  # NEW: Macro annotations
            self._check_docstring_change,  # NEW: Docstrings/comments
            self._check_typo_fix,  # NEW: Small typo fixes
            self._check_body_modification,  # Fallback for changes inside functions
        ]

        for detector in detectors:
            action = detector(added_lines, removed_lines, filename, diff)
            if action.action_type != ActionType.UNKNOWN:
                action.diff_hunk = diff
                return action

        # Default to unknown
        return InferredAction(
            action_type=ActionType.UNKNOWN,
            target_file=filename,
            diff_hunk=diff,
            confidence=0.1,
        )

    def _extract_function_names(self, lines: list[str]) -> set[str]:
        """Extract function names from a list of lines."""
        funcs = set()
        for line in lines:
            for pattern in self.FUNC_DEF_PATTERNS:
                if m := pattern.match(line):
                    funcs.add(m.group(1))
                    break
        return funcs

    def _check_function_change(
        self, added: list[str], removed: list[str], filename: str, full_diff: str
    ) -> InferredAction:
        """Check if diff represents a function addition/modification/removal."""
        added_funcs = self._extract_function_names(added)
        removed_funcs = self._extract_function_names(removed)

        # Determine action type
        only_added = added_funcs - removed_funcs
        only_removed = removed_funcs - added_funcs
        modified = added_funcs & removed_funcs

        if only_added and not only_removed and not modified:
            func_name = next(iter(only_added))
            return InferredAction(
                action_type=ActionType.ADD_METHOD,
                target_file=filename,
                target_symbol=func_name,
                confidence=0.85,
            )
        elif only_removed and not only_added and not modified:
            func_name = next(iter(only_removed))
            return InferredAction(
                action_type=ActionType.REMOVE_METHOD,
                target_file=filename,
                target_symbol=func_name,
                confidence=0.85,
            )
        elif modified:
            # Same function name appears in both - signature change or body modification
            func_name = next(iter(modified))
            return InferredAction(
                action_type=ActionType.MODIFY_METHOD,
                target_file=filename,
                target_symbol=func_name,
                confidence=0.75,
            )
        elif added_funcs and removed_funcs:
            # Different functions added and removed - could be rename
            # Check if only one of each and similar names
            if len(added_funcs) == 1 and len(removed_funcs) == 1:
                added_name = next(iter(added_funcs))
                removed_name = next(iter(removed_funcs))
                return InferredAction(
                    action_type=ActionType.RENAME_SYMBOL,
                    target_file=filename,
                    target_symbol=f"{removed_name} -> {added_name}",
                    confidence=0.6,
                )
            # Multiple changes - treat as modification
            return InferredAction(
                action_type=ActionType.MODIFY_METHOD,
                target_file=filename,
                target_symbol=next(iter(added_funcs)),
                confidence=0.5,
            )

        return InferredAction(action_type=ActionType.UNKNOWN, target_file=filename)

    def _check_type_change(
        self, added: list[str], removed: list[str], filename: str, full_diff: str
    ) -> InferredAction:
        """Check if diff represents a type/struct change."""
        added_types = set()
        removed_types = set()

        for line in added:
            for pattern in self.TYPE_DEF_PATTERNS:
                if m := pattern.match(line):
                    # Group 2 for struct (group 1 is 'mutable'), group 1 for abstract/primitive
                    type_name = m.group(2) if m.lastindex >= 2 else m.group(1)
                    added_types.add(type_name)
                    break

        for line in removed:
            for pattern in self.TYPE_DEF_PATTERNS:
                if m := pattern.match(line):
                    type_name = m.group(2) if m.lastindex >= 2 else m.group(1)
                    removed_types.add(type_name)
                    break

        only_added = added_types - removed_types
        only_removed = removed_types - added_types
        modified = added_types & removed_types

        if only_added and not only_removed:
            return InferredAction(
                action_type=ActionType.ADD_TYPE,
                target_file=filename,
                target_symbol=next(iter(only_added)),
                confidence=0.9,
            )
        elif only_removed and not only_added:
            return InferredAction(
                action_type=ActionType.MODIFY_TYPE,  # Using MODIFY_TYPE for removal
                target_file=filename,
                target_symbol=next(iter(only_removed)),
                confidence=0.9,
            )
        elif modified:
            return InferredAction(
                action_type=ActionType.MODIFY_TYPE,
                target_file=filename,
                target_symbol=next(iter(modified)),
                confidence=0.8,
            )

        # Check for field changes (lines with :: inside struct context)
        added_fields = [l for l in added if self.FIELD_PATTERN.match(l)]
        removed_fields = [l for l in removed if self.FIELD_PATTERN.match(l)]

        if added_fields and not removed_fields:
            return InferredAction(
                action_type=ActionType.ADD_FIELD,
                target_file=filename,
                confidence=0.7,
            )
        elif removed_fields and not added_fields:
            return InferredAction(
                action_type=ActionType.REMOVE_FIELD,
                target_file=filename,
                confidence=0.7,
            )
        elif added_fields or removed_fields:
            return InferredAction(
                action_type=ActionType.MODIFY_FIELD,
                target_file=filename,
                confidence=0.6,
            )

        return InferredAction(action_type=ActionType.UNKNOWN, target_file=filename)

    def _check_const_change(
        self, added: list[str], removed: list[str], filename: str, full_diff: str
    ) -> InferredAction:
        """Check if diff represents a const addition/modification."""
        added_consts = set()
        removed_consts = set()

        for line in added:
            if m := self.CONST_PATTERN.match(line):
                added_consts.add(m.group(1))

        for line in removed:
            if m := self.CONST_PATTERN.match(line):
                removed_consts.add(m.group(1))

        if added_consts and not removed_consts:
            return InferredAction(
                action_type=ActionType.ADD_FIELD,  # Using ADD_FIELD for const
                target_file=filename,
                target_symbol=next(iter(added_consts)),
                confidence=0.75,
            )
        elif removed_consts and not added_consts:
            return InferredAction(
                action_type=ActionType.REMOVE_FIELD,
                target_file=filename,
                target_symbol=next(iter(removed_consts)),
                confidence=0.75,
            )
        elif added_consts or removed_consts:
            return InferredAction(
                action_type=ActionType.MODIFY_FIELD,
                target_file=filename,
                target_symbol=next(iter(added_consts | removed_consts)),
                confidence=0.65,
            )

        return InferredAction(action_type=ActionType.UNKNOWN, target_file=filename)

    def _check_import_change(
        self, added: list[str], removed: list[str], filename: str, full_diff: str
    ) -> InferredAction:
        """Check if diff represents an import/export change."""
        added_imports = any(
            any(p.match(line) for p in self.IMPORT_PATTERNS)
            for line in added
        )
        removed_imports = any(
            any(p.match(line) for p in self.IMPORT_PATTERNS)
            for line in removed
        )

        if added_imports and not removed_imports:
            return InferredAction(
                action_type=ActionType.ADD_IMPORT,
                target_file=filename,
                confidence=0.8,
            )
        elif removed_imports and not added_imports:
            return InferredAction(
                action_type=ActionType.REMOVE_IMPORT,
                target_file=filename,
                confidence=0.8,
            )
        elif added_imports and removed_imports:
            # Modified imports
            return InferredAction(
                action_type=ActionType.ADD_IMPORT,  # Treat as add (modification)
                target_file=filename,
                confidence=0.6,
            )

        return InferredAction(action_type=ActionType.UNKNOWN, target_file=filename)

    def _check_macro_change(
        self, added: list[str], removed: list[str], filename: str, full_diff: str
    ) -> InferredAction:
        """Check if diff represents a macro annotation change."""
        added_macros = any(
            any(p.match(line) for p in self.MACRO_PATTERNS)
            for line in added
        )
        removed_macros = any(
            any(p.match(line) for p in self.MACRO_PATTERNS)
            for line in removed
        )

        if added_macros or removed_macros:
            return InferredAction(
                action_type=ActionType.MODIFY_MACRO,
                target_file=filename,
                confidence=0.7,
            )

        return InferredAction(action_type=ActionType.UNKNOWN, target_file=filename)

    def _check_docstring_change(
        self, added: list[str], removed: list[str], filename: str, full_diff: str
    ) -> InferredAction:
        """Check if diff represents a docstring or comment change."""
        # Check for docstring changes (triple-quoted)
        added_docs = any(self.DOCSTRING_PATTERN.match(l) for l in added)
        removed_docs = any(self.DOCSTRING_PATTERN.match(l) for l in removed)

        # Check for pure comment changes
        added_comments = [l for l in added if l.strip().startswith("#")]
        removed_comments = [l for l in removed if l.strip().startswith("#")]

        # Check if changes look like documentation content (markdown-style)
        doc_content_pattern = re.compile(r"^\s*[-*]\s+`?\w+`?\s*[-:]")  # - `param` or * param:
        added_doc_content = [l for l in added if doc_content_pattern.match(l)]
        removed_doc_content = [l for l in removed if doc_content_pattern.match(l)]

        # Check if all changes are comments/docs
        non_comment_added = [l for l in added if l.strip() and not l.strip().startswith("#")]
        non_comment_removed = [l for l in removed if l.strip() and not l.strip().startswith("#")]

        # If we have docstring changes or mostly comment changes
        if added_docs or removed_docs:
            return InferredAction(
                action_type=ActionType.MODIFY_DOCS,
                target_file=filename,
                confidence=0.75,
            )

        # If all changes are comments
        if (added_comments or removed_comments) and not non_comment_added and not non_comment_removed:
            return InferredAction(
                action_type=ActionType.MODIFY_DOCS,
                target_file=filename,
                confidence=0.8,
            )

        # If changes look like documentation content (markdown-style docs)
        if added_doc_content or removed_doc_content:
            # Check if the changes are mostly doc-like
            total_changes = len(added) + len(removed)
            doc_changes = len(added_doc_content) + len(removed_doc_content) + len(added_comments) + len(removed_comments)
            if doc_changes > 0 and doc_changes >= total_changes * 0.5:
                return InferredAction(
                    action_type=ActionType.MODIFY_DOCS,
                    target_file=filename,
                    confidence=0.7,
                )

        return InferredAction(action_type=ActionType.UNKNOWN, target_file=filename)

    def _check_typo_fix(
        self, added: list[str], removed: list[str], filename: str, full_diff: str
    ) -> InferredAction:
        """
        Check if diff represents a small typo fix or whitespace change.

        A typo fix is characterized by:
        - Very small changes (1-3 lines)
        - High similarity between added and removed content
        - Often just whitespace or single character differences
        """
        # Filter significant lines
        sig_added = [l for l in added if l.strip()]
        sig_removed = [l for l in removed if l.strip()]

        # Must be small change
        if len(sig_added) > 3 or len(sig_removed) > 3:
            return InferredAction(action_type=ActionType.UNKNOWN, target_file=filename)

        # If we have matching counts, check similarity
        if len(sig_added) == len(sig_removed) and len(sig_added) > 0:
            # Calculate character-level similarity
            from difflib import SequenceMatcher

            total_similarity = 0
            for a, r in zip(sig_added, sig_removed):
                ratio = SequenceMatcher(None, a.strip(), r.strip()).ratio()
                total_similarity += ratio

            avg_similarity = total_similarity / len(sig_added)

            # High similarity (>0.8) suggests typo fix
            if avg_similarity > 0.8:
                return InferredAction(
                    action_type=ActionType.FIX_TYPO,
                    target_file=filename,
                    confidence=0.65 + (avg_similarity - 0.8) * 0.5,  # 0.65-0.75
                )

        # Pure whitespace changes
        if not sig_added and not sig_removed:
            # Only whitespace changed
            if added or removed:
                return InferredAction(
                    action_type=ActionType.FIX_TYPO,
                    target_file=filename,
                    confidence=0.6,
                )

        return InferredAction(action_type=ActionType.UNKNOWN, target_file=filename)

    def _check_body_modification(
        self, added: list[str], removed: list[str], filename: str, full_diff: str
    ) -> InferredAction:
        """
        Fallback detector for modifications inside function bodies.

        If we have non-trivial added/removed lines but couldn't detect a specific
        pattern, this is likely a function body modification.
        """
        # Filter out empty lines and comments
        significant_added = [l for l in added if l.strip() and not l.strip().startswith("#")]
        significant_removed = [l for l in removed if l.strip() and not l.strip().startswith("#")]

        if not significant_added and not significant_removed:
            return InferredAction(action_type=ActionType.UNKNOWN, target_file=filename)

        # Look for context in the diff to find which function we're in
        # Parse the @@ hunk headers to find function context
        hunk_pattern = re.compile(r"@@ .* @@\s*(.*)")
        for line in full_diff.split("\n"):
            if m := hunk_pattern.match(line):
                context = m.group(1)
                # Try to extract function name from context
                for pattern in self.FUNC_DEF_PATTERNS:
                    if fm := pattern.match(context):
                        return InferredAction(
                            action_type=ActionType.MODIFY_METHOD,
                            target_file=filename,
                            target_symbol=fm.group(1),
                            confidence=0.6,
                        )

        # If we have ANY significant changes, classify as MODIFY_METHOD
        # This is more aggressive than before - we'd rather have a low-confidence
        # classification than UNKNOWN
        if significant_added or significant_removed:
            # Try to infer from filename if it's a test file
            if "test" in filename.lower():
                return InferredAction(
                    action_type=ActionType.MODIFY_TEST,
                    target_file=filename,
                    confidence=0.55,
                )
            return InferredAction(
                action_type=ActionType.MODIFY_METHOD,
                target_file=filename,
                confidence=0.5,
            )

        return InferredAction(action_type=ActionType.UNKNOWN, target_file=filename)


# ============================================================================
# Transition Extraction
# ============================================================================

@dataclass
class Transition:
    """A single (state, action, state') transition."""
    repo: str
    commit_sha: str
    parent_sha: str | None
    commit_message: str
    commit_date: str
    action: dict
    files_before: dict[str, str]  # filename -> content
    files_after: dict[str, str]   # filename -> content
    source_files_changed: list[str]
    test_files_changed: list[str]
    lines_changed: int
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "commit_sha": self.commit_sha,
            "parent_sha": self.parent_sha,
            "commit_message": self.commit_message,
            "commit_date": self.commit_date,
            "action": self.action,
            "files_before": self.files_before,
            "files_after": self.files_after,
            "source_files_changed": self.source_files_changed,
            "test_files_changed": self.test_files_changed,
            "lines_changed": self.lines_changed,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


class TransitionExtractor:
    """Extracts transitions from git history."""

    def __init__(self, config: MiningConfig):
        self.config = config
        self.walker = GitHistoryWalker(config.repo_path)
        self.inferrer = ActionInferrer()

    def extract_transitions(self, limit: int | None = None) -> Iterator[Transition]:
        """Extract transitions from the repository."""
        repo_name = self.config.repo_path.name

        for commit in self.walker.get_commits(limit=limit):
            # Skip merge commits
            if self.config.skip_merge_commits and " " in (commit.parent_sha or ""):
                logger.debug(f"Skipping merge commit {commit.sha[:8]}")
                continue

            # Filter files to source/test dirs
            source_files = [
                f for f in commit.files_changed
                if any(f.startswith(d) for d in self.config.source_dirs)
                and f.endswith(".jl")
            ]
            test_files = [
                f for f in commit.files_changed
                if any(f.startswith(d) for d in self.config.test_dirs)
                and f.endswith(".jl")
            ]

            # Skip if no source files changed
            if not source_files:
                if self.config.skip_docs_only:
                    logger.debug(f"Skipping docs-only commit {commit.sha[:8]}")
                    continue

            # Skip if too many files changed
            if len(source_files) > self.config.max_files_changed:
                logger.debug(f"Skipping large commit {commit.sha[:8]}: {len(source_files)} files")
                continue

            # Skip if too many lines changed
            total_lines = commit.insertions + commit.deletions
            if total_lines > self.config.max_lines_changed:
                logger.debug(f"Skipping large commit {commit.sha[:8]}: {total_lines} lines")
                continue

            if total_lines < self.config.min_lines_changed:
                logger.debug(f"Skipping trivial commit {commit.sha[:8]}")
                continue

            # Get the diff
            diff = self.walker.get_diff(commit.sha, commit.parent_sha)

            # Infer action (use first source file for now)
            if source_files:
                action = self.inferrer.infer_action(diff, source_files[0])
            else:
                action = InferredAction(
                    action_type=ActionType.UNKNOWN,
                    target_file="",
                    confidence=0.1,
                )

            # Get file contents before and after
            files_before = {}
            files_after = {}

            for f in source_files:
                if commit.parent_sha:
                    content = self.walker.get_file_at_commit(commit.parent_sha, f)
                    if content:
                        files_before[f] = content

                content = self.walker.get_file_at_commit(commit.sha, f)
                if content:
                    files_after[f] = content

            # Validate transition
            validation_errors = self._validate_transition(
                files_before, files_after, action,
                use_julia_validation=self.config.use_julia_validation,
            )

            transition = Transition(
                repo=repo_name,
                commit_sha=commit.sha,
                parent_sha=commit.parent_sha,
                commit_message=commit.message,
                commit_date=commit.date.isoformat(),
                action=action.to_dict(),
                files_before=files_before,
                files_after=files_after,
                source_files_changed=source_files,
                test_files_changed=test_files,
                lines_changed=total_lines,
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors,
            )

            yield transition

    def _validate_transition(
        self,
        files_before: dict[str, str],
        files_after: dict[str, str],
        action: InferredAction,
        use_julia_validation: bool = True,
    ) -> list[str]:
        """Validate a transition."""
        errors = []

        # Check that we have file contents
        if not files_after:
            errors.append("No file contents after commit")
            return errors  # Can't validate further without content

        # Check action confidence
        if action.confidence < 0.5:
            errors.append(f"Low action confidence: {action.confidence}")

        # Check action type is known
        if action.action_type == ActionType.UNKNOWN:
            errors.append("Could not infer action type")

        # Julia syntax validation
        if use_julia_validation:
            for filename, content in files_after.items():
                is_valid, parse_error, _, _ = validate_julia_syntax(content)
                if not is_valid:
                    errors.append(f"Julia syntax error in {filename}: {parse_error}")

        return errors


# ============================================================================
# Main Pipeline
# ============================================================================

def mine_transitions(
    repo_path: Path,
    output_path: Path,
    limit: int | None = None,
    use_julia_validation: bool = True,
) -> None:
    """Mine transitions from a repository and save to JSONL."""
    config = MiningConfig(
        repo_path=repo_path,
        output_path=output_path,
        use_julia_validation=use_julia_validation,
    )

    extractor = TransitionExtractor(config)

    valid_count = 0
    invalid_count = 0
    action_counts: dict[str, int] = {}

    with open(output_path, "w") as f:
        for transition in extractor.extract_transitions(limit=limit):
            # Write to JSONL
            f.write(json.dumps(transition.to_dict()) + "\n")

            # Track stats
            if transition.is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            action_type = transition.action["type"]
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

            if (valid_count + invalid_count) % 50 == 0:
                logger.info(f"Processed {valid_count + invalid_count} commits...")

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Mining complete for {repo_path.name}")
    logger.info(f"{'='*50}")
    logger.info(f"Valid transitions: {valid_count}")
    logger.info(f"Invalid transitions: {invalid_count}")
    logger.info(f"Output: {output_path}")
    logger.info(f"\nAction type distribution:")
    for action_type, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {action_type}: {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mine transitions from Julia git history")
    parser.add_argument("repo_path", type=Path, help="Path to the repository")
    parser.add_argument("--output", "-o", type=Path, help="Output JSONL file")
    parser.add_argument("--limit", "-n", type=int, help="Limit number of commits")
    parser.add_argument("--no-julia-validation", action="store_true",
                        help="Skip Julia syntax validation (faster but less accurate)")

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"data/transitions/{args.repo_path.name}.jsonl")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    mine_transitions(
        args.repo_path,
        args.output,
        args.limit,
        use_julia_validation=not args.no_julia_validation,
    )
