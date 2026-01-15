"""
Transformer-based code rendering.

This is the ONLY module that invokes transformers for code generation.
Transformers are used exclusively for filling in small code fragments,
never for planning or reasoning.

The key constraint: transformers receive:
- A template with {{HOLE}} markers
- Surrounding context
- Type constraints
- A hard token limit

They return ONLY the code to fill the hole.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ============================================================================
# Code Fragment Specification
# ============================================================================


@dataclass
class CodeFragment:
    """
    Specification for code to be generated.

    The template contains {{HOLE}} markers where code should be inserted.
    The transformer's job is to fill these holes, nothing more.
    """

    template: str
    context_before: str
    context_after: str
    constraints: dict[str, Any]
    max_tokens: int = 100


@dataclass
class RenderResult:
    """Result of rendering a code fragment."""

    generated_code: str
    tokens_used: int
    confidence: float
    validation_passed: bool
    validation_errors: list[str]


# ============================================================================
# Code Renderer
# ============================================================================


class CodeRenderer:
    """
    Renders code fragments using a transformer.

    This is a constrained code completion task, not open-ended generation.
    """

    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.2,  # Low temperature for deterministic output
        top_p: float = 0.95,
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.top_p = top_p

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info("Model loaded")

    def render(self, fragment: CodeFragment) -> RenderResult:
        """
        Render a code fragment by filling in the holes.

        Args:
            fragment: The code fragment specification

        Returns:
            RenderResult with generated code
        """
        self._load_model()

        # Build the prompt
        prompt = self._build_prompt(fragment)

        # Generate
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=fragment.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Extract generated code
        generated = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Clean up the generated code
        generated = self._clean_generated(generated, fragment)

        # Validate
        validation_errors = self._validate(generated, fragment)

        return RenderResult(
            generated_code=generated,
            tokens_used=len(self._tokenizer.encode(generated)),
            confidence=self._estimate_confidence(outputs),
            validation_passed=len(validation_errors) == 0,
            validation_errors=validation_errors,
        )

    def _build_prompt(self, fragment: CodeFragment) -> str:
        """Build the prompt for the transformer."""
        constraint_text = "\n".join(f"- {k}: {v}" for k, v in fragment.constraints.items())

        prompt = f"""# Fill in the {{{{HOLE}}}} in the following Julia code.
# Return ONLY the code that goes in the hole, nothing else.
# Do not include any explanation or markdown.

# CONSTRAINTS:
{constraint_text}

# CONTEXT BEFORE:
{fragment.context_before}

# TEMPLATE (fill the HOLE):
{fragment.template}

# CONTEXT AFTER:
{fragment.context_after}

# CODE TO FILL HOLE:
"""
        return prompt

    def _clean_generated(self, generated: str, fragment: CodeFragment) -> str:
        """Clean up generated code."""
        # Remove any markdown code blocks
        if "```" in generated:
            lines = generated.split("\n")
            clean_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code or not any(line.strip().startswith(x) for x in ["```", "#"]):
                    clean_lines.append(line)
            generated = "\n".join(clean_lines)

        # Remove leading/trailing whitespace
        generated = generated.strip()

        # Truncate at max tokens
        tokens = self._tokenizer.encode(generated)
        if len(tokens) > fragment.max_tokens:
            generated = self._tokenizer.decode(tokens[: fragment.max_tokens])

        return generated

    def _validate(self, generated: str, fragment: CodeFragment) -> list[str]:
        """Validate generated code against constraints."""
        errors = []

        # Check token limit
        tokens = self._tokenizer.encode(generated)
        if len(tokens) > fragment.max_tokens:
            errors.append(f"Generated code exceeds max tokens ({len(tokens)} > {fragment.max_tokens})")

        # Try to parse as Julia (basic syntax check)
        # This would use juliacall in production
        if not self._check_julia_syntax(generated):
            errors.append("Generated code has syntax errors")

        # Check type constraints if specified
        if "return_type" in fragment.constraints:
            # Would use Julia type inference to verify
            pass

        return errors

    def _check_julia_syntax(self, code: str) -> bool:
        """Check if code has valid Julia syntax."""
        # Basic heuristic checks
        # In production, this would call Julia's parser

        # Check balanced brackets
        brackets = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False
                opening = stack.pop()
                if brackets[opening] != char:
                    return False

        return len(stack) == 0

    def _estimate_confidence(self, outputs: Any) -> float:
        """Estimate confidence in the generated code."""
        # Would use output logits to estimate confidence
        # For now, return a placeholder
        return 0.8


# ============================================================================
# Specialized Renderers
# ============================================================================


class MethodBodyRenderer(CodeRenderer):
    """Renderer specialized for method bodies."""

    def _build_prompt(self, fragment: CodeFragment) -> str:
        """Build prompt specialized for method bodies."""
        func_name = fragment.constraints.get("function_name", "f")
        arg_types = fragment.constraints.get("arg_types", [])
        return_type = fragment.constraints.get("return_type", "Any")

        args_str = ", ".join(f"arg{i}::{t}" for i, t in enumerate(arg_types))

        prompt = f"""# Implement the body of this Julia function.
# Return ONLY the function body code, no function signature.

# Function signature: {func_name}({args_str}) -> {return_type}

# Context before:
{fragment.context_before}

# The function should:
{fragment.constraints.get('description', 'Implement the required functionality')}

# Function body:
"""
        return prompt


class TestRenderer(CodeRenderer):
    """Renderer specialized for test code."""

    def _build_prompt(self, fragment: CodeFragment) -> str:
        """Build prompt specialized for tests."""
        test_for = fragment.constraints.get("test_for", "the function")

        prompt = f"""# Write a Julia test for {test_for}.
# Return ONLY the test code using @test or @testset macros.

# Function being tested:
{fragment.context_before}

# Test code:
"""
        return prompt


# ============================================================================
# API Endpoint for Julia Bridge
# ============================================================================


class RenderServer:
    """
    Simple server for the Julia executor to call.

    In production, this would be a proper HTTP server.
    """

    def __init__(self):
        self.renderer = CodeRenderer()
        self.method_renderer = MethodBodyRenderer()
        self.test_renderer = TestRenderer()

    def render(self, request: dict) -> dict:
        """Handle a render request."""
        fragment = CodeFragment(
            template=request["template"],
            context_before=request.get("context_before", ""),
            context_after=request.get("context_after", ""),
            constraints=request.get("constraints", {}),
            max_tokens=request.get("max_tokens", 100),
        )

        render_type = request.get("type", "general")

        if render_type == "method_body":
            result = self.method_renderer.render(fragment)
        elif render_type == "test":
            result = self.test_renderer.render(fragment)
        else:
            result = self.renderer.render(fragment)

        return {
            "generated_code": result.generated_code,
            "tokens_used": result.tokens_used,
            "confidence": result.confidence,
            "validation_passed": result.validation_passed,
            "validation_errors": result.validation_errors,
        }


# ============================================================================
# Standalone Usage
# ============================================================================


def main():
    """Test the renderer."""
    renderer = CodeRenderer()

    fragment = CodeFragment(
        template="""
function add_vectors(a::Vector{Float64}, b::Vector{Float64})
    {{HOLE}}
end
""",
        context_before="# Vector addition with bounds checking",
        context_after="# More code follows...",
        constraints={
            "function_name": "add_vectors",
            "arg_types": ["Vector{Float64}", "Vector{Float64}"],
            "return_type": "Vector{Float64}",
            "description": "Add two vectors element-wise with bounds checking",
        },
        max_tokens=50,
    )

    result = renderer.render(fragment)
    print(f"Generated code:\n{result.generated_code}")
    print(f"\nTokens used: {result.tokens_used}")
    print(f"Confidence: {result.confidence}")
    print(f"Validation passed: {result.validation_passed}")
    if result.validation_errors:
        print(f"Errors: {result.validation_errors}")


if __name__ == "__main__":
    main()
