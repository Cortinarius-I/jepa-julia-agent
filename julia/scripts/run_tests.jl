#!/usr/bin/env julia
"""
Run tests and collect results for world state.

Usage:
    julia run_tests.jl <repo_path> [output_file]
"""

include(joinpath(@__DIR__, "..", "src", "WorldState.jl"))

using .WorldState
using Test
using JSON3

function run_tests_with_results(repo_path::String)
    results = TestResult[]
    
    # Find test files
    test_dir = joinpath(repo_path, "test")
    if !isdir(test_dir)
        @warn "No test directory found at $test_dir"
        return results
    end
    
    # Look for runtests.jl
    runtests = joinpath(test_dir, "runtests.jl")
    if !isfile(runtests)
        @warn "No runtests.jl found"
        return results
    end
    
    # Run tests with timing
    @info "Running tests from $runtests"
    
    start_time = time()
    try
        # This is a simplified version - real implementation would
        # hook into Julia's test framework more deeply
        include(runtests)
        
        push!(results, TestResult(
            "runtests",
            runtests,
            true,
            nothing,
            (time() - start_time) * 1000
        ))
    catch e
        push!(results, TestResult(
            "runtests",
            runtests,
            false,
            string(e),
            (time() - start_time) * 1000
        ))
    end
    
    results
end

function main(args)
    if length(args) < 1
        println("Usage: julia run_tests.jl <repo_path> [output_file]")
        return 1
    end
    
    repo_path = args[1]
    output_file = length(args) >= 2 ? args[2] : nothing
    
    @info "Running tests in $repo_path"
    
    results = run_tests_with_results(repo_path)
    
    # Build test state
    passed = count(r -> r.passed, results)
    failed = count(r -> !r.passed, results)
    
    state = TestState(results, passed, failed, 0.0)
    
    # Serialize
    json_str = JSON3.write(state)
    
    if output_file !== nothing
        write(output_file, json_str)
        @info "Wrote test results to $output_file"
    else
        println(json_str)
    end
    
    return failed > 0 ? 1 : 0
end

exit(main(ARGS))
