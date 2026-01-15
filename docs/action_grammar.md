# Action Grammar

This document defines the typed action grammar for the JEPA Julia Agent.

## Principles

1. **Actions are typed**: Every action has a known type with defined semantics
2. **Actions are reversible**: Every action has an inverse
3. **Actions are validatable**: We can check preconditions before execution
4. **Actions are predictable**: JEPA can predict consequences

## Action Types

### Method Actions

#### `ADD_METHOD`

Add a new method definition to a module.

```julia
struct AddMethodAction
    target_module::Symbol       # Where to add
    function_name::Symbol       # Name of the function
    signature::MethodSignature  # Type signature
    body_template::String       # Template with {{HOLE}} for LLM
    insertion_point::Location   # Where in the file
end
```

**Preconditions:**
- Module exists
- No method with identical signature exists
- Insertion point is valid

**Postconditions:**
- Method exists in method table
- Method is callable with specified signature

**Inverse:** `REMOVE_METHOD` with same signature

---

#### `MODIFY_METHOD`

Modify an existing method's body.

```julia
struct ModifyMethodAction
    target_module::Symbol
    function_name::Symbol
    signature::MethodSignature
    modification_type::Symbol  # :replace_body, :wrap, :add_dispatch
    new_body_template::String
    preserve_semantics::Bool   # Must pass same tests?
end
```

**Preconditions:**
- Method exists with specified signature
- Modification type is valid for method

**Postconditions:**
- Method body is updated
- If preserve_semantics, tests still pass

**Inverse:** Store original body and `MODIFY_METHOD` back

---

#### `REMOVE_METHOD`

Remove a method definition.

```julia
struct RemoveMethodAction
    target_module::Symbol
    function_name::Symbol
    signature::MethodSignature
end
```

**Preconditions:**
- Method exists
- No other methods depend on this one (or explicit override)

**Postconditions:**
- Method removed from method table
- Dispatch updated

**Inverse:** `ADD_METHOD` with stored body

---

### Field Actions

#### `ADD_FIELD`

Add a field to a struct.

```julia
struct AddFieldAction
    target_module::Symbol
    struct_name::Symbol
    field_name::Symbol
    field_type::Any
    default_value::Union{String, Nothing}
    position::Union{Int, Nothing}
end
```

**Preconditions:**
- Struct exists
- Field name not already used
- Type is valid

**Postconditions:**
- Struct has new field
- Constructors updated

**Inverse:** `REMOVE_FIELD`

---

#### `MODIFY_FIELD`

Modify a field's type or default.

```julia
struct ModifyFieldAction
    target_module::Symbol
    struct_name::Symbol
    field_name::Symbol
    new_type::Union{Any, Nothing}
    new_default::Union{String, Nothing}
end
```

**Preconditions:**
- Field exists
- New type is compatible with existing usage

**Postconditions:**
- Field updated
- Dependent code checked

**Inverse:** Store original and `MODIFY_FIELD` back

---

#### `REMOVE_FIELD`

Remove a field from a struct.

```julia
struct RemoveFieldAction
    target_module::Symbol
    struct_name::Symbol
    field_name::Symbol
end
```

**Preconditions:**
- Field exists
- No other code accesses this field (or explicit override)

**Postconditions:**
- Field removed
- Constructors updated

**Inverse:** `ADD_FIELD` with stored info

---

### Import Actions

#### `ADD_IMPORT`

Add an import statement.

```julia
struct AddImportAction
    target_module::Symbol
    import_from::Symbol
    symbols::Vector{Symbol}  # Empty = whole module
    import_type::Symbol      # :import, :using, :import_as
end
```

**Preconditions:**
- Module to import from exists
- No conflicting imports

**Postconditions:**
- Symbols available in target module

**Inverse:** `REMOVE_IMPORT`

---

#### `REMOVE_IMPORT`

Remove an import statement.

```julia
struct RemoveImportAction
    target_module::Symbol
    import_from::Symbol
    symbols::Vector{Symbol}
end
```

**Preconditions:**
- Import exists
- Removing won't break references

**Postconditions:**
- Symbols no longer imported

**Inverse:** `ADD_IMPORT`

---

### Refactoring Actions

#### `RENAME_SYMBOL`

Rename a symbol throughout the codebase.

```julia
struct RenameAction
    target_module::Symbol
    old_name::Symbol
    new_name::Symbol
    scope::Symbol  # :local, :module, :global
end
```

**Preconditions:**
- Old name exists
- New name doesn't conflict
- Scope is valid

**Postconditions:**
- All references updated
- No broken references

**Inverse:** `RENAME_SYMBOL` back

---

#### `MOVE_DEFINITION`

Move a definition to a different module.

```julia
struct MoveAction
    symbol_name::Symbol
    from_module::Symbol
    to_module::Symbol
    update_imports::Bool
end
```

**Preconditions:**
- Symbol exists in source
- Destination can accept it
- No circular dependencies created

**Postconditions:**
- Symbol in new location
- Imports updated if requested

**Inverse:** `MOVE_DEFINITION` back

---

### Test Actions

#### `ADD_TEST`

Add a new test.

```julia
struct AddTestAction
    test_name::String
    test_file::String
    test_body_template::String
    test_for::Union{MethodSignature, Nothing}
end
```

**Preconditions:**
- Test name unique
- Test file exists or can be created

**Postconditions:**
- Test runs as part of test suite

**Inverse:** `REMOVE_TEST`

---

#### `MODIFY_TEST`

Modify an existing test.

```julia
struct ModifyTestAction
    test_name::String
    test_file::String
    new_body_template::String
end
```

**Preconditions:**
- Test exists

**Postconditions:**
- Test updated

**Inverse:** Store original and `MODIFY_TEST` back

---

#### `REMOVE_TEST`

Remove a test.

```julia
struct RemoveTestAction
    test_name::String
    test_file::String
end
```

**Preconditions:**
- Test exists

**Postconditions:**
- Test no longer runs

**Inverse:** `ADD_TEST` with stored body

---

### Composite Actions

#### `COMPOSITE`

A sequence of actions executed atomically.

```julia
struct CompositeAction
    actions::Vector{Action}
    name::String
    description::String
end
```

**Preconditions:**
- All sub-actions valid in sequence

**Postconditions:**
- All sub-actions applied

**Inverse:** Reverse sequence of inverse actions

---

## Action Encoding for JEPA

Each action is encoded as a fixed-size vector for JEPA:

```python
def encode_action(action):
    # One-hot encoding of action type
    type_onehot = one_hot(action.type, num_types)
    
    # Embedding of target location
    location_embed = embed_location(action.target_module, action.target_symbol)
    
    # Embedding of action-specific parameters
    param_embed = embed_parameters(action.parameters)
    
    return concat(type_onehot, location_embed, param_embed)
```

## Safety Constraints

Actions are rejected if:

1. **Type mismatch**: Action would create type errors
2. **Test regression**: Action would break existing tests
3. **Dependency violation**: Action would create unresolved references
4. **Invalidation cascade**: Action would invalidate too many methods
5. **Ambiguity creation**: Action would create dispatch ambiguities

These are checked both by validation (before execution) and prediction (via JEPA).
