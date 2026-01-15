"""
    WorldState

The semantic world state representation for a Julia codebase.
This module defines the data structures that represent a Julia repository
as an executable semantic graph, not as tokens.

The world state captures:
- Module dependency graph
- Method table state (which methods exist, their signatures)
- Dispatch graph (how method dispatch resolves)
- Type inference state (inferred types for expressions)
- Test state (which tests pass/fail)
- Invalidation state (what gets invalidated when code changes)
"""
module WorldState

export ModuleNode, ModuleGraph
export MethodSignature, MethodInfo, MethodTableState
export DispatchEdge, DispatchGraph
export TypeInferenceState, InferredType
export TestResult, TestState
export InvalidationEvent, InvalidationState
export WorldStateSnapshot, extract_world_state, diff_world_states

using JSON3
using StructTypes

# ============================================================================
# Module Graph
# ============================================================================

"""
    ModuleNode

Represents a Julia module in the dependency graph.
"""
struct ModuleNode
    name::Symbol
    parent::Union{Symbol, Nothing}
    submodules::Vector{Symbol}
    imports::Vector{Symbol}        # Modules this module imports from
    exports::Vector{Symbol}        # Symbols this module exports
    file_path::Union{String, Nothing}
end

StructTypes.StructType(::Type{ModuleNode}) = StructTypes.Struct()

"""
    ModuleGraph

The full module dependency graph for a codebase.
"""
struct ModuleGraph
    nodes::Dict{Symbol, ModuleNode}
    root::Symbol
end

StructTypes.StructType(::Type{ModuleGraph}) = StructTypes.Struct()

function ModuleGraph()
    ModuleGraph(Dict{Symbol, ModuleNode}(), :Main)
end

# ============================================================================
# Method Table State
# ============================================================================

"""
    MethodSignature

A compact representation of a method signature.
"""
struct MethodSignature
    name::Symbol
    arg_types::Vector{Any}        # Type constraints on arguments
    where_params::Vector{Symbol}  # Type parameters
    return_type::Union{Any, Nothing}
end

StructTypes.StructType(::Type{MethodSignature}) = StructTypes.Struct()

"""
    MethodInfo

Full information about a method definition.
"""
struct MethodInfo
    signature::MethodSignature
    module_name::Symbol
    file::String
    line::Int
    is_generated::Bool
    world_age::UInt64
end

StructTypes.StructType(::Type{MethodInfo}) = StructTypes.Struct()

"""
    MethodTableState

The state of all method tables in the codebase.
"""
struct MethodTableState
    methods::Dict{Symbol, Vector{MethodInfo}}  # function name -> methods
    method_count::Int
end

StructTypes.StructType(::Type{MethodTableState}) = StructTypes.Struct()

function MethodTableState()
    MethodTableState(Dict{Symbol, Vector{MethodInfo}}(), 0)
end

# ============================================================================
# Dispatch Graph
# ============================================================================

"""
    DispatchEdge

Represents a dispatch relationship: when you call f(x::T), which method runs?
"""
struct DispatchEdge
    caller_sig::MethodSignature
    callee_sig::MethodSignature
    call_site_file::String
    call_site_line::Int
    is_concrete::Bool  # True if dispatch is statically resolved
end

StructTypes.StructType(::Type{DispatchEdge}) = StructTypes.Struct()

"""
    DispatchGraph

The call graph with dispatch resolution information.
"""
struct DispatchGraph
    edges::Vector{DispatchEdge}
    ambiguities::Vector{Tuple{MethodSignature, MethodSignature}}  # Ambiguous dispatch pairs
end

StructTypes.StructType(::Type{DispatchGraph}) = StructTypes.Struct()

function DispatchGraph()
    DispatchGraph(Vector{DispatchEdge}(), Vector{Tuple{MethodSignature, MethodSignature}}())
end

# ============================================================================
# Type Inference State
# ============================================================================

"""
    InferredType

The inferred type for an expression, with confidence.
"""
struct InferredType
    expr_hash::UInt64
    inferred::Any
    is_concrete::Bool
    confidence::Float64  # 1.0 = definitely this type, lower = uncertainty
end

StructTypes.StructType(::Type{InferredType}) = StructTypes.Struct()

"""
    TypeInferenceState

Cached type inference results.
"""
struct TypeInferenceState
    inferred_types::Dict{UInt64, InferredType}  # expr_hash -> type
    inference_errors::Vector{String}
end

StructTypes.StructType(::Type{TypeInferenceState}) = StructTypes.Struct()

function TypeInferenceState()
    TypeInferenceState(Dict{UInt64, InferredType}(), Vector{String}())
end

# ============================================================================
# Test State
# ============================================================================

"""
    TestResult

The result of running a single test.
"""
struct TestResult
    name::String
    file::String
    passed::Bool
    error_message::Union{String, Nothing}
    duration_ms::Float64
end

StructTypes.StructType(::Type{TestResult}) = StructTypes.Struct()

"""
    TestState

The state of all tests in the codebase.
"""
struct TestState
    results::Vector{TestResult}
    total_passed::Int
    total_failed::Int
    coverage::Float64  # 0.0 to 1.0
end

StructTypes.StructType(::Type{TestState}) = StructTypes.Struct()

function TestState()
    TestState(Vector{TestResult}(), 0, 0, 0.0)
end

# ============================================================================
# Invalidation State
# ============================================================================

"""
    InvalidationEvent

Records what gets invalidated when code changes.
"""
struct InvalidationEvent
    trigger_method::MethodSignature
    invalidated_methods::Vector{MethodSignature}
    reason::Symbol  # :new_method, :redefinition, :type_change, etc.
    timestamp::Float64
end

StructTypes.StructType(::Type{InvalidationEvent}) = StructTypes.Struct()

"""
    InvalidationState

Tracks method invalidations.
"""
struct InvalidationState
    recent_events::Vector{InvalidationEvent}
    total_invalidations::Int
    hot_spots::Vector{MethodSignature}  # Methods that cause many invalidations
end

StructTypes.StructType(::Type{InvalidationState}) = StructTypes.Struct()

function InvalidationState()
    InvalidationState(Vector{InvalidationEvent}(), 0, Vector{MethodSignature}())
end

# ============================================================================
# Full World State
# ============================================================================

"""
    WorldStateSnapshot

The complete semantic state of a Julia codebase at a point in time.
"""
struct WorldStateSnapshot
    modules::ModuleGraph
    methods::MethodTableState
    dispatch::DispatchGraph
    types::TypeInferenceState
    tests::TestState
    invalidations::InvalidationState
    timestamp::Float64
    repo_hash::String  # Git commit or content hash
end

StructTypes.StructType(::Type{WorldStateSnapshot}) = StructTypes.Struct()

function WorldStateSnapshot()
    WorldStateSnapshot(
        ModuleGraph(),
        MethodTableState(),
        DispatchGraph(),
        TypeInferenceState(),
        TestState(),
        InvalidationState(),
        time(),
        ""
    )
end

# ============================================================================
# World State Extraction
# ============================================================================

"""
    extract_world_state(repo_path::String) -> WorldStateSnapshot

Extract the semantic world state from a Julia repository.
"""
function extract_world_state(repo_path::String)
    snapshot = WorldStateSnapshot()
    
    # TODO: Implement extraction logic using:
    # - CodeTracking for method locations
    # - MethodAnalysis for method tables
    # - Cthulhu for type inference
    # - JuliaSyntax for AST parsing
    
    @info "Extracting world state from $repo_path"
    
    snapshot
end

"""
    diff_world_states(before::WorldStateSnapshot, after::WorldStateSnapshot) -> Dict

Compute the semantic difference between two world states.
Returns a dictionary of changes by category.
"""
function diff_world_states(before::WorldStateSnapshot, after::WorldStateSnapshot)
    diff = Dict{Symbol, Any}()
    
    # Method changes
    diff[:added_methods] = []
    diff[:removed_methods] = []
    diff[:modified_methods] = []
    
    # Type changes
    diff[:type_changes] = []
    
    # Test changes
    diff[:newly_passing] = []
    diff[:newly_failing] = []
    
    # Invalidations caused by this transition
    diff[:invalidations] = []
    
    # TODO: Implement actual diffing logic
    
    diff
end

# ============================================================================
# Serialization
# ============================================================================

"""
    to_json(snapshot::WorldStateSnapshot) -> String

Serialize a world state snapshot to JSON for Python interop.
"""
function to_json(snapshot::WorldStateSnapshot)
    JSON3.write(snapshot)
end

"""
    from_json(json_str::String) -> WorldStateSnapshot

Deserialize a world state snapshot from JSON.
"""
function from_json(json_str::String)
    JSON3.read(json_str, WorldStateSnapshot)
end

end # module
