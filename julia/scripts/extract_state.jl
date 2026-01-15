#!/usr/bin/env julia
"""
Extract world state from a Julia repository.

Usage:
    julia extract_state.jl <repo_path> [output_file]
"""

# Load our modules
include(joinpath(@__DIR__, "..", "src", "WorldState.jl"))
include(joinpath(@__DIR__, "..", "src", "IRTools.jl"))

using .WorldState
using .IRTools
using JSON3

function main(args)
    if length(args) < 1
        println("Usage: julia extract_state.jl <repo_path> [output_file]")
        return 1
    end
    
    repo_path = args[1]
    output_file = length(args) >= 2 ? args[2] : nothing
    
    @info "Extracting world state from $repo_path"
    
    # Extract the world state
    state = extract_world_state(repo_path)
    
    # Serialize to JSON
    json_str = to_json(state)
    
    if output_file !== nothing
        write(output_file, json_str)
        @info "Wrote state to $output_file"
    else
        println(json_str)
    end
    
    return 0
end

exit(main(ARGS))
