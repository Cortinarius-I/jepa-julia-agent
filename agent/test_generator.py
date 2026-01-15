"""
Adaptive test generation for JEPA training.

This module generates targeted tests based on action types, providing
richer training signal than running existing tests alone.

Inspired by Agent2World (2025): Testing Team "dynamically synthesizes
test cases based on the specific errors exhibited by each world model,
enabling precision-guided debugging rather than generic checks."
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Test Case Representation
# ---------------------------------------------------------------------------


class ExpectedBehavior(Enum):
    """Expected behavior of a test."""
    SHOULD_PASS = "should_pass"
    SHOULD_ERROR = "should_error"
    SHOULD_RETURN = "should_return"


@dataclass
class TestCase:
    """A single generated test case."""
    name: str
    code: str
    expected_behavior: ExpectedBehavior = ExpectedBehavior.SHOULD_PASS
    expected_value: Any = None
    timeout_ms: int = 5000


@dataclass
class TestResult:
    """Result of running a test case."""
    name: str
    passed: bool
    elapsed_ms: float
    error: Optional[str] = None
    actual_value: Any = None


@dataclass
class TestSuite:
    """A suite of test cases for a specific action."""
    action_type: str
    target: str
    tests: list[TestCase] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""


# ---------------------------------------------------------------------------
# Test Generation Strategies
# ---------------------------------------------------------------------------


class TestGenerationStrategy:
    """Base class for action-specific test generation."""
    
    def generate(self, context: dict) -> list[TestCase]:
        raise NotImplementedError


class AddMethodStrategy(TestGenerationStrategy):
    """Generate tests for ADD_METHOD actions."""
    
    def __init__(
        self,
        method_name: str,
        module_name: str,
        arg_types: list[str],
        return_type: Optional[str] = None,
    ):
        self.method_name = method_name
        self.module_name = module_name
        self.arg_types = arg_types
        self.return_type = return_type
    
    def generate(self, context: dict) -> list[TestCase]:
        tests = []
        
        # Test 1: Method exists
        tests.append(TestCase(
            name=f"{self.method_name}_exists",
            code=f"""
@test isdefined({self.module_name}, :{self.method_name})
""".strip(),
        ))
        
        # Test 2: Method is callable with correct types
        if self.arg_types:
            type_tuple = ", ".join(self.arg_types)
            tests.append(TestCase(
                name=f"{self.method_name}_has_method",
                code=f"""
@test hasmethod({self.module_name}.{self.method_name}, Tuple{{{type_tuple}}})
""".strip(),
            ))
        
        # Test 3: Basic invocation
        sample_args = self._generate_sample_args()
        if sample_args:
            tests.append(TestCase(
                name=f"{self.method_name}_callable",
                code=f"""
result = {self.module_name}.{self.method_name}({sample_args})
@test result !== nothing || true
""".strip(),
                timeout_ms=10000,
            ))
        
        # Test 4: Return type (if specified)
        if self.return_type and sample_args:
            tests.append(TestCase(
                name=f"{self.method_name}_return_type",
                code=f"""
result = {self.module_name}.{self.method_name}({sample_args})
@test result isa {self.return_type}
""".strip(),
            ))
        
        # Test 5: Edge cases for numeric types
        if any(t in ["Int", "Float64", "Number"] for t in self.arg_types):
            zero_args = self._generate_zero_args()
            tests.append(TestCase(
                name=f"{self.method_name}_handles_zeros",
                code=f"""
@test try {self.module_name}.{self.method_name}({zero_args}); true catch; true end
""".strip(),
            ))
        
        return tests
    
    def _generate_sample_args(self) -> str:
        """Generate sample arguments based on types."""
        samples = []
        for t in self.arg_types:
            if t in ["Int", "Int64", "Integer"]:
                samples.append("1")
            elif t in ["Float64", "Float32", "AbstractFloat"]:
                samples.append("1.0")
            elif t in ["String", "AbstractString"]:
                samples.append('"test"')
            elif t.startswith("Vector"):
                samples.append("[]")
            elif t == "Bool":
                samples.append("true")
            elif t == "Symbol":
                samples.append(":test")
            else:
                samples.append("nothing")
        return ", ".join(samples)
    
    def _generate_zero_args(self) -> str:
        """Generate zero/empty arguments for edge case testing."""
        zeros = []
        for t in self.arg_types:
            if t in ["Int", "Int64", "Integer"]:
                zeros.append("0")
            elif t in ["Float64", "Float32", "AbstractFloat"]:
                zeros.append("0.0")
            elif t in ["String", "AbstractString"]:
                zeros.append('""')
            else:
                zeros.append("nothing")
        return ", ".join(zeros)


class ModifyMethodStrategy(TestGenerationStrategy):
    """Generate tests for MODIFY_METHOD actions."""
    
    def __init__(
        self,
        method_name: str,
        module_name: str,
        original_signature: str = "",
        new_signature: str = "",
    ):
        self.method_name = method_name
        self.module_name = module_name
        self.original_signature = original_signature
        self.new_signature = new_signature
    
    def generate(self, context: dict) -> list[TestCase]:
        tests = []
        
        # Test 1: Method still exists
        tests.append(TestCase(
            name=f"{self.method_name}_still_exists",
            code=f"""
@test isdefined({self.module_name}, :{self.method_name})
""".strip(),
        ))
        
        # Test 2: Has at least one method
        tests.append(TestCase(
            name=f"{self.method_name}_has_methods",
            code=f"""
@test length(methods({self.module_name}.{self.method_name})) > 0
""".strip(),
        ))
        
        # Test 3: Regression tests from existing call sites
        existing_calls = context.get("existing_call_sites", [])
        for i, call in enumerate(existing_calls):
            tests.append(TestCase(
                name=f"{self.method_name}_regression_{i}",
                code=f"""
@test try {call}; true catch; false end
""".strip(),
                timeout_ms=10000,
            ))
        
        return tests


class RemoveMethodStrategy(TestGenerationStrategy):
    """Generate tests for REMOVE_METHOD actions."""
    
    def __init__(self, method_name: str, module_name: str):
        self.method_name = method_name
        self.module_name = module_name
    
    def generate(self, context: dict) -> list[TestCase]:
        tests = []
        
        # Test 1: Method no longer exists (or has fewer methods)
        tests.append(TestCase(
            name=f"{self.method_name}_removed",
            code=f"""
# Either undefined or has fewer methods than before
methods_count = isdefined({self.module_name}, :{self.method_name}) ? 
    length(methods({self.module_name}.{self.method_name})) : 0
@test methods_count < {context.get('original_method_count', 999)}
""".strip(),
        ))
        
        # Test 2: Dependent code still works (if any)
        dependent_methods = context.get("dependent_methods", [])
        for method in dependent_methods:
            tests.append(TestCase(
                name=f"dependent_{method}_still_works",
                code=f"""
@test isdefined({self.module_name}, :{method})
""".strip(),
            ))
        
        return tests


class AddFieldStrategy(TestGenerationStrategy):
    """Generate tests for ADD_FIELD actions."""
    
    def __init__(self, struct_name: str, field_name: str, field_type: str):
        self.struct_name = struct_name
        self.field_name = field_name
        self.field_type = field_type
    
    def generate(self, context: dict) -> list[TestCase]:
        tests = []
        
        # Test 1: Field exists
        tests.append(TestCase(
            name=f"{self.struct_name}_has_{self.field_name}",
            code=f"""
@test hasfield({self.struct_name}, :{self.field_name})
""".strip(),
        ))
        
        # Test 2: Field type is correct
        tests.append(TestCase(
            name=f"{self.struct_name}_{self.field_name}_type",
            code=f"""
@test fieldtype({self.struct_name}, :{self.field_name}) <: {self.field_type}
""".strip(),
        ))
        
        # Test 3: Struct still constructible
        constructor_args = context.get("constructor_args", "")
        if constructor_args:
            tests.append(TestCase(
                name=f"{self.struct_name}_constructible",
                code=f"""
obj = {self.struct_name}({constructor_args})
@test obj !== nothing
""".strip(),
            ))
        
        return tests


class RenameSymbolStrategy(TestGenerationStrategy):
    """Generate tests for RENAME_SYMBOL actions."""
    
    def __init__(
        self,
        old_name: str,
        new_name: str,
        symbol_type: str = "function",
    ):
        self.old_name = old_name
        self.new_name = new_name
        self.symbol_type = symbol_type
    
    def generate(self, context: dict) -> list[TestCase]:
        tests = []
        module = context.get("module_name", "Main")
        
        # Test 1: New name exists
        tests.append(TestCase(
            name=f"{self.new_name}_exists",
            code=f"""
@test isdefined({module}, :{self.new_name})
""".strip(),
        ))
        
        # Test 2: Old name is gone (unless aliased)
        if not context.get("keep_alias", False):
            tests.append(TestCase(
                name=f"{self.old_name}_gone",
                code=f"""
@test !isdefined({module}, :{self.old_name})
""".strip(),
            ))
        
        return tests


class AddImportStrategy(TestGenerationStrategy):
    """Generate tests for ADD_IMPORT actions."""
    
    def __init__(self, module_name: str, imported_symbols: list[str]):
        self.module_name = module_name
        self.imported_symbols = imported_symbols
    
    def generate(self, context: dict) -> list[TestCase]:
        tests = []
        target_module = context.get("target_module", "Main")
        
        for sym in self.imported_symbols:
            tests.append(TestCase(
                name=f"import_{sym}_available",
                code=f"""
@test isdefined({target_module}, :{sym})
""".strip(),
            ))
        
        return tests


class AddTestStrategy(TestGenerationStrategy):
    """Generate meta-tests for ADD_TEST actions."""
    
    def __init__(self, test_name: str, test_file: str):
        self.test_name = test_name
        self.test_file = test_file
    
    def generate(self, context: dict) -> list[TestCase]:
        tests = []
        
        # Test: The test itself passes
        tests.append(TestCase(
            name=f"{self.test_name}_passes",
            code=f"""
# Run the newly added test
include("{self.test_file}")
@test true  # If we got here, the test passed
""".strip(),
            timeout_ms=30000,
        ))
        
        return tests


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------


class AdaptiveTestGenerator:
    """
    Generate targeted tests based on action type.
    
    This provides richer training signal than running existing tests alone
    by testing the specific consequences of each edit.
    """
    
    STRATEGY_MAP = {
        "ADD_METHOD": AddMethodStrategy,
        "MODIFY_METHOD": ModifyMethodStrategy,
        "REMOVE_METHOD": RemoveMethodStrategy,
        "ADD_FIELD": AddFieldStrategy,
        "MODIFY_FIELD": AddFieldStrategy,  # Similar to add
        "RENAME_SYMBOL": RenameSymbolStrategy,
        "ADD_IMPORT": AddImportStrategy,
        "ADD_TEST": AddTestStrategy,
    }
    
    def generate_tests(
        self,
        action_type: str,
        target: str,
        context: dict,
    ) -> TestSuite:
        """
        Generate a test suite for the given action.
        
        Args:
            action_type: One of ADD_METHOD, MODIFY_METHOD, etc.
            target: Target of the action (method name, field name, etc.)
            context: Additional context needed for test generation
            
        Returns:
            TestSuite with targeted tests for this action.
        """
        strategy_class = self.STRATEGY_MAP.get(action_type)
        
        if strategy_class is None:
            # Return empty suite for unhandled actions
            return TestSuite(action_type=action_type, target=target)
        
        # Create strategy based on action type
        strategy = self._create_strategy(strategy_class, action_type, target, context)
        
        # Generate tests
        tests = strategy.generate(context)
        
        return TestSuite(
            action_type=action_type,
            target=target,
            tests=tests,
            setup_code=context.get("setup_code", ""),
            teardown_code=context.get("teardown_code", ""),
        )
    
    def _create_strategy(
        self,
        strategy_class: type,
        action_type: str,
        target: str,
        context: dict,
    ) -> TestGenerationStrategy:
        """Create strategy instance based on action type."""
        if action_type == "ADD_METHOD":
            return strategy_class(
                method_name=target,
                module_name=context.get("module_name", "Main"),
                arg_types=context.get("arg_types", []),
                return_type=context.get("return_type"),
            )
        elif action_type == "MODIFY_METHOD":
            return strategy_class(
                method_name=target,
                module_name=context.get("module_name", "Main"),
                original_signature=context.get("original_signature", ""),
                new_signature=context.get("new_signature", ""),
            )
        elif action_type == "REMOVE_METHOD":
            return RemoveMethodStrategy(
                method_name=target,
                module_name=context.get("module_name", "Main"),
            )
        elif action_type in ["ADD_FIELD", "MODIFY_FIELD"]:
            return strategy_class(
                struct_name=context.get("struct_name", ""),
                field_name=target,
                field_type=context.get("field_type", "Any"),
            )
        elif action_type == "RENAME_SYMBOL":
            return strategy_class(
                old_name=context.get("old_name", ""),
                new_name=target,
                symbol_type=context.get("symbol_type", "function"),
            )
        elif action_type == "ADD_IMPORT":
            return strategy_class(
                module_name=context.get("source_module", ""),
                imported_symbols=context.get("imported_symbols", [target]),
            )
        elif action_type == "ADD_TEST":
            return strategy_class(
                test_name=target,
                test_file=context.get("test_file", ""),
            )
        else:
            # Shouldn't reach here due to STRATEGY_MAP check
            raise ValueError(f"Unknown action type: {action_type}")
    
    def generate_property_tests(
        self,
        method_name: str,
        properties: list[str],
        context: dict,
    ) -> list[TestCase]:
        """
        Generate property-based tests.
        
        Properties can include: idempotent, commutative, associative, invertible
        """
        tests = []
        module = context.get("module_name", "Main")
        sample = context.get("sample_input", "1")
        
        for prop in properties:
            if prop == "idempotent":
                tests.append(TestCase(
                    name=f"{method_name}_idempotent",
                    code=f"""
x = {sample}
@test {module}.{method_name}({module}.{method_name}(x)) == {module}.{method_name}(x)
""".strip(),
                    timeout_ms=10000,
                ))
            elif prop == "commutative":
                samples = context.get("sample_inputs", "1, 2")
                tests.append(TestCase(
                    name=f"{method_name}_commutative",
                    code=f"""
x, y = {samples}
@test {module}.{method_name}(x, y) == {module}.{method_name}(y, x)
""".strip(),
                    timeout_ms=10000,
                ))
            elif prop == "associative":
                samples = context.get("sample_inputs", "1, 2, 3")
                tests.append(TestCase(
                    name=f"{method_name}_associative",
                    code=f"""
x, y, z = {samples}
@test {module}.{method_name}({module}.{method_name}(x, y), z) == {module}.{method_name}(x, {module}.{method_name}(y, z))
""".strip(),
                    timeout_ms=10000,
                ))
        
        return tests


# ---------------------------------------------------------------------------
# Utility for Julia interop
# ---------------------------------------------------------------------------


def test_suite_to_julia(suite: TestSuite) -> str:
    """Convert a TestSuite to Julia test code."""
    lines = [
        "using Test",
        "",
    ]
    
    if suite.setup_code:
        lines.append("# Setup")
        lines.append(suite.setup_code)
        lines.append("")
    
    lines.append(f'@testset "{suite.action_type}: {suite.target}" begin')
    
    for test in suite.tests:
        lines.append(f'    @testset "{test.name}" begin')
        # Indent the test code
        for line in test.code.split("\n"):
            lines.append(f"        {line}")
        lines.append("    end")
    
    lines.append("end")
    
    if suite.teardown_code:
        lines.append("")
        lines.append("# Teardown")
        lines.append(suite.teardown_code)
    
    return "\n".join(lines)
