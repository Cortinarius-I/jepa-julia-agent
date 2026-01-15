#!/usr/bin/env julia
"""
Collect metrics from a Julia repository for JEPA training.

Metrics include:
- Method count and complexity
- Type stability statistics
- Dispatch analysis
- Test coverage

Usage:
    julia collect_metrics.jl <repo_path> [output_file]
"""

include(joinpath(@__DIR__, "..", "src", "WorldState.jl"))
include(joinpath(@__DIR__, "..", "src", "IRTools.jl"))

using .WorldState
using .IRTools
using JSON3

struct RepositoryMetrics
    method_count::Int
    module_count::Int
    type_stable_methods::Int
    type_unstable_methods::Int
    test_count::Int
    dispatch_edges::Int
    average_method_complexity::Float64
end

function collect_metrics(repo_path::String)
    @info "Collecting metrics from $repo_path"
    
    # Extract world state
    state = extract_world_state(repo_path)
    
    # Count methods
    method_count = state.methods.method_count
    
    # Count modules
    module_count = length(state.modules.nodes)
    
    # Count tests
    test_count = length(state.tests.results)
    
    # Count dispatch edges
    dispatch_edges = length(state.dispatch.edges)
    
    # Analyze type stability (placeholder - would use IRTools)
    type_stable = 0
    type_unstable = 0
    
    # Calculate average complexity (placeholder)
    avg_complexity = 0.0
    
    RepositoryMetrics(
        method_count,
        module_count,
        type_stable,
        type_unstable,
        test_count,
        dispatch_edges,
        avg_complexity
    )
end

function main(args)
    if length(args) < 1
        println("Usage: julia collect_metrics.jl <repo_path> [output_file]")
        return 1
    end
    
    repo_path = args[1]
    output_file = length(args) >= 2 ? args[2] : nothing
    
    metrics = collect_metrics(repo_path)
    
    # Serialize
    json_str = JSON3.write(metrics)
    
    if output_file !== nothing
        write(output_file, json_str)
        @info "Wrote metrics to $output_file"
    else
        println(json_str)
    end
    
    # Print summary
    @info "Repository Metrics:" metrics.method_count metrics.module_count metrics.test_count
    
    return 0
end

exit(main(ARGS))
