"""
Transformer-based code rendering.

This module contains the ONLY components that invoke language models
for code generation. They are used exclusively for filling in small
code templates, never for planning or reasoning.
"""
from transformer.render import CodeRenderer, CodeFragment, RenderResult

__all__ = ["CodeRenderer", "CodeFragment", "RenderResult"]
