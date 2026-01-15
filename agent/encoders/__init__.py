"""
Encoders for world state components.

This package provides specialized graph neural network encoders
for different components of the Julia world state:

- ModuleGraphEncoder: Encodes module dependency graphs
- DispatchGraphEncoder: Encodes method dispatch relationships
- TypeHierarchyEncoder: Encodes type subtype relationships
- CallGraphEncoder: Encodes static call graphs
- MethodSignatureEncoder: Encodes individual method signatures
- CompositeGraphEncoder: Combines all encoders for full world state
"""

from .graph_encoders import (
    ModuleGraphEncoder,
    DispatchGraphEncoder,
    TypeHierarchyEncoder,
    CallGraphEncoder,
    MethodSignatureEncoder,
    CompositeGraphEncoder,
)

__all__ = [
    "ModuleGraphEncoder",
    "DispatchGraphEncoder",
    "TypeHierarchyEncoder",
    "CallGraphEncoder",
    "MethodSignatureEncoder",
    "CompositeGraphEncoder",
]
