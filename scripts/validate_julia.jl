#!/usr/bin/env julia
"""
Julia syntax validator for transition mining.

This script reads Julia code from stdin and validates:
1. Syntax (can it be parsed?)
2. Basic semantic checks

Returns JSON with validation results (hand-serialized to avoid dependencies).
"""

function validate_code(code::String)
    functions = String[]
    types = String[]

    # Try to parse
    try
        exprs = Meta.parseall(code)

        # Count expressions and extract function/type names
        n_exprs = 0
        parse_errors = String[]

        if exprs.head == :toplevel
            for expr in exprs.args
                if expr isa Expr
                    # Check for incomplete/error expressions
                    if expr.head == :incomplete
                        push!(parse_errors, expr.args[1])
                    elseif expr.head == :error
                        push!(parse_errors, "Parse error: " * string(expr))
                    else
                        n_exprs += 1
                        extract_definitions!(expr, functions, types)
                    end
                end
            end
        end

        if !isempty(parse_errors)
            return (valid=false, parse_error=join(parse_errors, "; "), expressions=n_exprs, functions=functions, types=types)
        end

        return (valid=true, parse_error=nothing, expressions=n_exprs, functions=functions, types=types)
    catch e
        return (valid=false, parse_error=string(e), expressions=0, functions=functions, types=types)
    end
end

function to_json(result)
    funcs_json = "[" * join(["\"" * escape_string(f) * "\"" for f in result.functions], ",") * "]"
    types_json = "[" * join(["\"" * escape_string(t) * "\"" for t in result.types], ",") * "]"
    error_json = result.parse_error === nothing ? "null" : "\"" * escape_string(result.parse_error) * "\""

    return """{\"valid\":$(result.valid),\"parse_error\":$(error_json),\"expressions\":$(result.expressions),\"functions\":$(funcs_json),\"types\":$(types_json)}"""
end

function extract_definitions!(expr::Expr, functions::Vector{String}, types::Vector{String})
    if expr.head == :function || expr.head == :(=)
        # Check if it's a function definition
        if length(expr.args) >= 1
            call_expr = expr.args[1]
            if call_expr isa Expr && call_expr.head == :call
                fname = call_expr.args[1]
                if fname isa Symbol
                    push!(functions, string(fname))
                elseif fname isa Expr && fname.head == :(.)
                    # Qualified name like Base.show
                    push!(functions, string(fname))
                end
            elseif call_expr isa Expr && call_expr.head == :where
                # Function with where clause
                inner = call_expr.args[1]
                if inner isa Expr && inner.head == :call
                    fname = inner.args[1]
                    if fname isa Symbol
                        push!(functions, string(fname))
                    end
                end
            end
        end
    elseif expr.head == :struct || expr.head == :abstract
        # Type definition
        if length(expr.args) >= 2
            type_expr = expr.args[2]
            if type_expr isa Symbol
                push!(types, string(type_expr))
            elseif type_expr isa Expr && type_expr.head == :(<:)
                push!(types, string(type_expr.args[1]))
            elseif type_expr isa Expr && type_expr.head == :curly
                push!(types, string(type_expr.args[1]))
            end
        end
    elseif expr.head == :macrocall
        # Skip macro calls but recurse into their arguments
        for arg in expr.args
            if arg isa Expr
                extract_definitions!(arg, functions, types)
            end
        end
    else
        # Recurse into nested expressions
        for arg in expr.args
            if arg isa Expr
                extract_definitions!(arg, functions, types)
            end
        end
    end
end

# Main: read from stdin or file argument
function main()
    code = if length(ARGS) > 0 && isfile(ARGS[1])
        read(ARGS[1], String)
    else
        read(stdin, String)
    end

    result = validate_code(code)
    println(to_json(result))
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
