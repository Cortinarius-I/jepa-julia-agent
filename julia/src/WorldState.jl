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
Uses static analysis of Julia source files.
"""
function extract_world_state(repo_path::String)
    @info "Extracting world state from $repo_path"

    # Find all Julia source files
    src_files = find_julia_files(repo_path)
    @info "Found $(length(src_files)) Julia files"

    # Extract module graph
    modules = extract_module_graph(repo_path, src_files)

    # Extract method table
    methods = extract_method_table(src_files)

    # Extract dispatch graph (simplified: call relationships)
    dispatch = extract_dispatch_graph(src_files, methods)

    # Type inference (placeholder - requires runtime analysis)
    types = TypeInferenceState()

    # Test state (scan test files)
    tests = extract_test_state(repo_path)

    # Invalidation state (placeholder - requires runtime tracking)
    invalidations = InvalidationState()

    # Compute repo hash
    repo_hash = compute_repo_hash(repo_path)

    WorldStateSnapshot(
        modules,
        methods,
        dispatch,
        types,
        tests,
        invalidations,
        time(),
        repo_hash
    )
end

"""
    find_julia_files(repo_path::String) -> Vector{String}

Find all .jl files in the repository.
"""
function find_julia_files(repo_path::String)
    files = String[]
    for (root, dirs, filenames) in walkdir(repo_path)
        # Skip hidden directories and common non-source dirs
        filter!(d -> !startswith(d, ".") && d ∉ ["deps", "docs", "examples"], dirs)
        for f in filenames
            if endswith(f, ".jl")
                push!(files, joinpath(root, f))
            end
        end
    end
    files
end

"""
    extract_module_graph(repo_path::String, files::Vector{String}) -> ModuleGraph

Extract the module dependency graph from source files.
"""
function extract_module_graph(repo_path::String, files::Vector{String})
    nodes = Dict{Symbol, ModuleNode}()

    for file in files
        try
            content = read(file, String)
            extract_modules_from_source!(nodes, content, file)
        catch e
            @warn "Failed to parse $file: $e"
        end
    end

    # Determine root module (usually matches repo name)
    root = isempty(nodes) ? :Main : first(keys(nodes))

    ModuleGraph(nodes, root)
end

"""
    extract_modules_from_source!(nodes, content, file)

Parse a Julia file and extract module definitions.
"""
function extract_modules_from_source!(nodes::Dict{Symbol, ModuleNode}, content::String, file::String)
    lines = split(content, '\n')

    current_module = nothing
    imports = Symbol[]
    exports = Symbol[]
    submodules = Symbol[]

    for line in lines
        stripped = strip(line)

        # Match module definition: module Foo / baremodule Foo
        m = match(r"^(?:bare)?module\s+(\w+)", stripped)
        if m !== nothing
            if current_module !== nothing
                # Save previous module
                nodes[current_module] = ModuleNode(
                    current_module, nothing, submodules, imports, exports, file
                )
            end
            current_module = Symbol(m.captures[1])
            imports = Symbol[]
            exports = Symbol[]
            submodules = Symbol[]
            continue
        end

        # Match using statements: using Foo, Bar / using Foo: bar, baz
        m = match(r"^using\s+(.+)", stripped)
        if m !== nothing
            # Parse module names (simplified)
            parts = split(m.captures[1], r"[,:]")
            for part in parts
                part = strip(part)
                if !isempty(part)
                    # Extract first identifier (module name)
                    m2 = match(r"^(\w+)", part)
                    if m2 !== nothing
                        push!(imports, Symbol(m2.captures[1]))
                    end
                end
            end
            continue
        end

        # Match import statements
        m = match(r"^import\s+(.+)", stripped)
        if m !== nothing
            parts = split(m.captures[1], r"[,:]")
            for part in parts
                part = strip(part)
                if !isempty(part)
                    m2 = match(r"^(\w+)", part)
                    if m2 !== nothing
                        push!(imports, Symbol(m2.captures[1]))
                    end
                end
            end
            continue
        end

        # Match export statements
        m = match(r"^export\s+(.+)", stripped)
        if m !== nothing
            parts = split(m.captures[1], ",")
            for part in parts
                part = strip(part)
                if !isempty(part)
                    push!(exports, Symbol(part))
                end
            end
            continue
        end
    end

    # Save last module
    if current_module !== nothing
        nodes[current_module] = ModuleNode(
            current_module, nothing, submodules, unique(imports), unique(exports), file
        )
    end

    nothing
end

"""
    extract_method_table(files::Vector{String}) -> MethodTableState

Extract method definitions from source files.
"""
function extract_method_table(files::Vector{String})
    methods = Dict{Symbol, Vector{MethodInfo}}()
    method_count = 0

    for file in files
        try
            content = read(file, String)
            extract_methods_from_source!(methods, content, file)
        catch e
            @warn "Failed to extract methods from $file: $e"
        end
    end

    # Count total methods
    for (_, ms) in methods
        method_count += length(ms)
    end

    MethodTableState(methods, method_count)
end

"""
    extract_methods_from_source!(methods, content, file)

Parse a Julia file and extract function definitions.
"""
function extract_methods_from_source!(methods::Dict{Symbol, Vector{MethodInfo}}, content::String, file::String)
    lines = split(content, '\n')
    current_module = :Main

    # Patterns for function definitions
    # function foo(x, y) / foo(x, y) = / function foo(x::T, y::S) where {T, S}
    func_patterns = [
        r"^function\s+(\w+)\s*\(([^)]*)\)(\s*where\s*\{([^}]*)\})?",
        r"^(\w+)\s*\(([^)]*)\)\s*=",
        r"^function\s+(\w+)\s*\(",  # Multiline function
    ]

    for (line_num, line) in enumerate(lines)
        stripped = strip(line)

        # Track current module
        m = match(r"^(?:bare)?module\s+(\w+)", stripped)
        if m !== nothing
            current_module = Symbol(m.captures[1])
            continue
        end

        # Try to match function definitions
        for pattern in func_patterns
            m = match(pattern, stripped)
            if m !== nothing
                func_name = Symbol(m.captures[1])

                # Parse argument types (simplified)
                args = length(m.captures) >= 2 && m.captures[2] !== nothing ? m.captures[2] : ""
                arg_types = parse_arg_types(args)

                # Parse where clause
                where_params = Symbol[]
                if length(m.captures) >= 4 && m.captures[4] !== nothing
                    where_str = m.captures[4]
                    for part in split(where_str, ",")
                        part = strip(part)
                        m2 = match(r"^(\w+)", part)
                        if m2 !== nothing
                            push!(where_params, Symbol(m2.captures[1]))
                        end
                    end
                end

                sig = MethodSignature(func_name, arg_types, where_params, nothing)
                info = MethodInfo(sig, current_module, file, line_num, false, UInt64(0))

                if !haskey(methods, func_name)
                    methods[func_name] = MethodInfo[]
                end
                push!(methods[func_name], info)
                break
            end
        end
    end

    nothing
end

"""
    parse_arg_types(args::String) -> Vector{Any}

Parse function argument types from the argument string.
"""
function parse_arg_types(args::String)
    types = Any[]
    if isempty(strip(args))
        return types
    end

    # Split by comma (simplified - doesn't handle nested types properly)
    for arg in split(args, ",")
        arg = strip(arg)
        if isempty(arg)
            continue
        end

        # Match name::Type pattern
        m = match(r"::(.+)$", arg)
        if m !== nothing
            type_str = strip(m.captures[1])
            # Remove default values
            type_str = replace(type_str, r"\s*=.*$" => "")
            push!(types, type_str)
        else
            push!(types, "Any")
        end
    end

    types
end

"""
    extract_dispatch_graph(files, methods) -> DispatchGraph

Extract call relationships from source files.
"""
function extract_dispatch_graph(files::Vector{String}, method_table::MethodTableState)
    edges = DispatchEdge[]

    # Get all known function names
    known_funcs = Set(keys(method_table.methods))

    for file in files
        try
            content = read(file, String)
            extract_calls_from_source!(edges, content, file, known_funcs)
        catch e
            @warn "Failed to extract calls from $file: $e"
        end
    end

    DispatchGraph(edges, Vector{Tuple{MethodSignature, MethodSignature}}())
end

"""
    extract_calls_from_source!(edges, content, file, known_funcs)

Extract function call relationships from source.
"""
function extract_calls_from_source!(edges::Vector{DispatchEdge}, content::String, file::String, known_funcs::Set{Symbol})
    lines = split(content, '\n')
    current_func = nothing
    current_module = :Main

    for (line_num, line) in enumerate(lines)
        stripped = strip(line)

        # Track current module
        m = match(r"^(?:bare)?module\s+(\w+)", stripped)
        if m !== nothing
            current_module = Symbol(m.captures[1])
            continue
        end

        # Track current function (simplified)
        m = match(r"^function\s+(\w+)", stripped)
        if m !== nothing
            current_func = Symbol(m.captures[1])
            continue
        end

        m = match(r"^(\w+)\s*\(.*\)\s*=", stripped)
        if m !== nothing
            current_func = Symbol(m.captures[1])
        end

        # Look for function calls
        if current_func !== nothing
            # Find all identifier( patterns
            for m in eachmatch(r"\b(\w+)\s*\(", line)
                callee_name = Symbol(m.captures[1])
                if callee_name in known_funcs && callee_name != current_func
                    caller_sig = MethodSignature(current_func, Any[], Symbol[], nothing)
                    callee_sig = MethodSignature(callee_name, Any[], Symbol[], nothing)

                    edge = DispatchEdge(
                        caller_sig,
                        callee_sig,
                        file,
                        line_num,
                        false  # Can't determine static dispatch without type info
                    )
                    push!(edges, edge)
                end
            end
        end

        # End of function
        if startswith(stripped, "end")
            current_func = nothing
        end
    end

    nothing
end

"""
    extract_test_state(repo_path::String) -> TestState

Scan test files and extract test information.
"""
function extract_test_state(repo_path::String)
    test_dir = joinpath(repo_path, "test")
    results = TestResult[]

    if isdir(test_dir)
        for (root, dirs, files) in walkdir(test_dir)
            for f in files
                if endswith(f, ".jl")
                    file_path = joinpath(root, f)
                    try
                        content = read(file_path, String)
                        # Count @test macros
                        test_count = count(r"@test\b", content)
                        testset_count = count(r"@testset\b", content)

                        # Add a summary result for this file
                        push!(results, TestResult(
                            f,
                            file_path,
                            true,  # Assume passing (we're not running tests)
                            nothing,
                            0.0
                        ))
                    catch e
                        @warn "Failed to read test file $file_path: $e"
                    end
                end
            end
        end
    end

    TestState(results, length(results), 0, 0.0)
end

"""
    compute_repo_hash(repo_path::String) -> String

Compute a hash of the repository state.
"""
function compute_repo_hash(repo_path::String)
    # Try to get git hash first
    try
        git_dir = joinpath(repo_path, ".git")
        if isdir(git_dir)
            head_file = joinpath(git_dir, "HEAD")
            if isfile(head_file)
                ref = strip(read(head_file, String))
                if startswith(ref, "ref: ")
                    ref_path = joinpath(git_dir, ref[6:end])
                    if isfile(ref_path)
                        return strip(read(ref_path, String))[1:8]
                    end
                else
                    return ref[1:8]
                end
            end
        end
    catch e
        @warn "Failed to get git hash: $e"
    end

    # Fall back to timestamp
    string(hash(time()))
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
