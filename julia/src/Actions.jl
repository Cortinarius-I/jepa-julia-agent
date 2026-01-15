"""
    Actions

Defines the typed action grammar for code modifications.
Actions are semantic operations on code, NOT free-form text generation.

The key insight: instead of asking an LLM "write code to do X",
we have a finite set of well-defined actions that the planner can choose from.
The LLM is only invoked for small, localized code fragments within an action.

This design:
- Prevents hallucination at the planning level
- Makes actions composable and reversible
- Enables prediction of action consequences
"""
module Actions

export Action, ActionType
export AddMethodAction, ModifyMethodAction, RemoveMethodAction
export AddFieldAction, ModifyFieldAction, RemoveFieldAction
export AddImportAction, RemoveImportAction
export RenameAction, MoveAction
export AddTestAction, ModifyTestAction, RemoveTestAction
export CompositeAction
export validate_action, apply_action, reverse_action

using ..WorldState

# ============================================================================
# Action Type Enumeration
# ============================================================================

@enum ActionType begin
    ADD_METHOD
    MODIFY_METHOD
    REMOVE_METHOD
    ADD_FIELD
    MODIFY_FIELD
    REMOVE_FIELD
    ADD_IMPORT
    REMOVE_IMPORT
    RENAME_SYMBOL
    MOVE_DEFINITION
    ADD_TEST
    MODIFY_TEST
    REMOVE_TEST
    COMPOSITE
end

# ============================================================================
# Abstract Action Interface
# ============================================================================

"""
    Action

Abstract base type for all actions.
Every action must be:
- Typed (we know exactly what kind of change it is)
- Validatable (we can check if it's safe before applying)
- Reversible (we can undo it)
- Predictable (JEPA can predict its consequences)
"""
abstract type Action end

"""
    validate_action(action::Action, state::WorldStateSnapshot) -> (Bool, String)

Check if an action is valid in the current world state.
Returns (is_valid, error_message_or_empty).
"""
function validate_action end

"""
    apply_action(action::Action, state::WorldStateSnapshot) -> WorldStateSnapshot

Apply an action and return the new world state.
Assumes validation has already passed.
"""
function apply_action end

"""
    reverse_action(action::Action) -> Action

Return the inverse action that would undo this one.
"""
function reverse_action end

# ============================================================================
# Method Actions
# ============================================================================

"""
    AddMethodAction

Add a new method definition.
"""
struct AddMethodAction <: Action
    target_module::Symbol
    function_name::Symbol
    signature::MethodSignature
    body_template::String  # Template with holes for LLM to fill
    insertion_point::Union{Tuple{String, Int}, Nothing}  # (file, line) or auto
end

function validate_action(action::AddMethodAction, state::WorldStateSnapshot)
    # Check: module exists
    if !haskey(state.modules.nodes, action.target_module)
        return (false, "Module $(action.target_module) does not exist")
    end
    
    # Check: method doesn't already exist with exact signature
    if haskey(state.methods.methods, action.function_name)
        for m in state.methods.methods[action.function_name]
            if m.signature == action.signature
                return (false, "Method with identical signature already exists")
            end
        end
    end
    
    (true, "")
end

function reverse_action(action::AddMethodAction)
    RemoveMethodAction(action.target_module, action.function_name, action.signature)
end

"""
    ModifyMethodAction

Modify an existing method's body.
"""
struct ModifyMethodAction <: Action
    target_module::Symbol
    function_name::Symbol
    signature::MethodSignature
    modification_type::Symbol  # :replace_body, :wrap, :add_dispatch, etc.
    new_body_template::String
    preserve_semantics::Bool  # If true, must pass same tests
end

function validate_action(action::ModifyMethodAction, state::WorldStateSnapshot)
    # Check: method exists
    if !haskey(state.methods.methods, action.function_name)
        return (false, "Function $(action.function_name) does not exist")
    end
    
    found = false
    for m in state.methods.methods[action.function_name]
        if m.signature == action.signature
            found = true
            break
        end
    end
    
    if !found
        return (false, "Method with specified signature not found")
    end
    
    (true, "")
end

"""
    RemoveMethodAction

Remove a method definition.
"""
struct RemoveMethodAction <: Action
    target_module::Symbol
    function_name::Symbol
    signature::MethodSignature
end

function validate_action(action::RemoveMethodAction, state::WorldStateSnapshot)
    # Check: method exists
    if !haskey(state.methods.methods, action.function_name)
        return (false, "Function $(action.function_name) does not exist")
    end
    
    # Check: removing this won't break dispatch for existing calls
    # This is where JEPA predictions become valuable
    
    (true, "")
end

# ============================================================================
# Field/Property Actions
# ============================================================================

"""
    AddFieldAction

Add a field to a struct or mutable struct.
"""
struct AddFieldAction <: Action
    target_module::Symbol
    struct_name::Symbol
    field_name::Symbol
    field_type::Any
    default_value::Union{String, Nothing}
    position::Union{Int, Nothing}  # Position in struct, or nothing for append
end

"""
    ModifyFieldAction

Modify a field's type or default value.
"""
struct ModifyFieldAction <: Action
    target_module::Symbol
    struct_name::Symbol
    field_name::Symbol
    new_type::Union{Any, Nothing}
    new_default::Union{String, Nothing}
end

"""
    RemoveFieldAction

Remove a field from a struct.
"""
struct RemoveFieldAction <: Action
    target_module::Symbol
    struct_name::Symbol
    field_name::Symbol
end

# ============================================================================
# Import/Export Actions
# ============================================================================

"""
    AddImportAction

Add an import statement.
"""
struct AddImportAction <: Action
    target_module::Symbol
    import_from::Symbol
    symbols::Vector{Symbol}  # Empty means import entire module
    import_type::Symbol  # :import, :using, :import_as
end

"""
    RemoveImportAction

Remove an import statement.
"""
struct RemoveImportAction <: Action
    target_module::Symbol
    import_from::Symbol
    symbols::Vector{Symbol}
end

# ============================================================================
# Refactoring Actions
# ============================================================================

"""
    RenameAction

Rename a symbol throughout the codebase.
"""
struct RenameAction <: Action
    target_module::Symbol
    old_name::Symbol
    new_name::Symbol
    scope::Symbol  # :local, :module, :global
end

function validate_action(action::RenameAction, state::WorldStateSnapshot)
    # Check: old name exists
    # Check: new name doesn't conflict
    # Check: rename won't break external API (if scope is :global)
    
    (true, "")
end

"""
    MoveAction

Move a definition to a different module or file.
"""
struct MoveAction <: Action
    symbol_name::Symbol
    from_module::Symbol
    to_module::Symbol
    update_imports::Bool
end

# ============================================================================
# Test Actions
# ============================================================================

"""
    AddTestAction

Add a new test.
"""
struct AddTestAction <: Action
    test_name::String
    test_file::String
    test_body_template::String
    test_for::Union{MethodSignature, Nothing}  # What this test covers
end

"""
    ModifyTestAction

Modify an existing test.
"""
struct ModifyTestAction <: Action
    test_name::String
    test_file::String
    new_body_template::String
end

"""
    RemoveTestAction

Remove a test.
"""
struct RemoveTestAction <: Action
    test_name::String
    test_file::String
end

# ============================================================================
# Composite Actions
# ============================================================================

"""
    CompositeAction

A sequence of actions to be applied atomically.
"""
struct CompositeAction <: Action
    actions::Vector{Action}
    name::String
    description::String
end

function validate_action(action::CompositeAction, state::WorldStateSnapshot)
    current_state = state
    
    for (i, sub_action) in enumerate(action.actions)
        valid, msg = validate_action(sub_action, current_state)
        if !valid
            return (false, "Sub-action $i failed validation: $msg")
        end
        # Would need to simulate state change here for full validation
    end
    
    (true, "")
end

function reverse_action(action::CompositeAction)
    CompositeAction(
        reverse.(reverse(action.actions)),
        "Reverse: $(action.name)",
        "Reversal of: $(action.description)"
    )
end

# ============================================================================
# Action Encoding for JEPA
# ============================================================================

"""
    encode_action(action::Action) -> Vector{Float32}

Encode an action as a fixed-size vector for JEPA input.
This is used by the Python JEPA model.
"""
function encode_action(action::Action)
    # Return a vector that captures:
    # - Action type (one-hot)
    # - Target location embedding
    # - Semantic features of the change
    
    # Placeholder - actual encoding done in Python
    Float32[]
end

"""
    action_to_dict(action::Action) -> Dict

Convert an action to a dictionary for JSON serialization.
"""
function action_to_dict(action::Action)
    Dict(
        "type" => string(typeof(action)),
        "fields" => Dict(string(f) => getfield(action, f) for f in fieldnames(typeof(action)))
    )
end

end # module
