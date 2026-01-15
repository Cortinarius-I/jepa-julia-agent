# JEPA Julia Agent: Design Document

## Overview

The JEPA Julia Agent is a code modification system that treats Julia codebases as **executable semantic worlds** rather than token sequences. The agent learns to predict the consequences of code changes and uses this knowledge to plan safe, effective modifications.

## Core Philosophy

### Why JEPA for Code?

Traditional LLM-based coding agents suffer from:

1. **Context window limits**: Can't see the whole codebase
2. **Hallucination**: Generate plausible-looking but incorrect code
3. **No planning**: Make changes without predicting consequences
4. **Token-level thinking**: Treat code as text, not semantic structure

JEPA (Joint Embedding Predictive Architecture) addresses these by:

1. **Embedding-space prediction**: Predict outcomes, not tokens
2. **Semantic world state**: Represent code as graphs and relations
3. **Execution grounding**: Validate against compiler and tests
4. **Typed actions**: Constrain the action space to safe operations

### The Key Insight

> Code modification is better modeled as navigation through a semantic state space than as sequence generation.

Instead of asking "what code should I generate?", we ask:
- "What is the current semantic state of the codebase?"
- "What action should I take?"
- "What will the new state be if I take this action?"
- "Is that new state safe (tests pass, types check)?"

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     JEPA World Model                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Context   │    │   Action    │    │  Predictor  │     │
│  │   Encoder   │ ×  │   Encoder   │ -> │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│        ↑                  ↑                  ↓              │
│        │                  │           Predicted State       │
└────────│──────────────────│──────────────────│──────────────┘
         │                  │                  │
    World State        Action Type        Safety Check
         ↑                  ↑                  ↓
┌────────│──────────────────│──────────────────│──────────────┐
│        │            Planner                  │              │
│        │     (Beam Search in Embedding Space)│              │
└────────│──────────────────────────────────────│──────────────┘
         ↑                                     ↓
         │                              Selected Plan
         │                                     ↓
┌────────│─────────────────────────────────────│──────────────┐
│   Julia Codebase                        Executor            │
│                                              │              │
│   ┌──────────────┐                    ┌──────────────┐     │
│   │ World State  │ <----------------> │  AST Editor  │     │
│   │  Extractor   │                    └──────────────┘     │
│   └──────────────┘                           │              │
│                                              ↓              │
│                                    ┌──────────────┐        │
│                                    │ Transformer  │        │
│                                    │  (Code Fill) │        │
│                                    └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. World State (Julia → Python)

The semantic representation of a Julia codebase:

```julia
struct WorldState
    modules::ModuleGraph       # Module dependency graph
    methods::MethodTableState  # All method definitions
    dispatch::DispatchGraph    # Call graph with dispatch info
    types::TypeInferenceState  # Inferred type information
    tests::TestState           # Test results
    invalidations::InvalidationState  # Method invalidation tracking
end
```

This is NOT tokens or AST - it's semantic structure that captures:
- What functions exist and their relationships
- How dispatch resolves for different types
- What tests exist and whether they pass
- What gets invalidated when something changes

### 2. Typed Actions

Actions are semantic operations, not free-form text:

```julia
@enum ActionType begin
    ADD_METHOD
    MODIFY_METHOD
    REMOVE_METHOD
    ADD_FIELD
    RENAME_SYMBOL
    # ... etc
end
```

Each action type has:
- **Preconditions**: What must be true for the action to be valid
- **Postconditions**: What the action guarantees
- **Inverse**: How to undo the action
- **Embedding**: How to encode for JEPA

### 3. JEPA World Model

The core prediction model:

```python
class JEPAWorldModel(nn.Module):
    context_encoder: WorldStateEncoder  # State → embedding
    action_encoder: ActionEncoder       # Action → embedding
    predictor: JEPAPredictor           # (state, action) → next_state embedding
    target_encoder: WorldStateEncoder   # EMA target for training
```

Training uses the standard JEPA objective:
- Predict target embedding from (context, action)
- Target is EMA of context encoder applied to actual next state
- No reconstruction, no generation - just embedding prediction

### 4. Planner

Beam search in embedding space:

```python
def plan(current_state, goal):
    beam = [initial_sequence]
    
    for depth in range(max_depth):
        for sequence in beam:
            for action in generate_actions(sequence, goal):
                predicted_state = jepa.predict(current_state, action)
                
                if is_safe(predicted_state):
                    extended = sequence + [action]
                    candidates.append(extended)
        
        beam = top_k(candidates, by=score)
    
    return best_complete_plan
```

The key: we evaluate actions by **predicting** their outcomes, not by generating code.

### 5. Executor

Applies actions deterministically, invoking transformers only for code fragments:

```python
def execute(action):
    # Determine what code needs to be generated
    fragment = create_code_fragment(action)
    
    # Invoke transformer to fill the template
    code = transformer.render(fragment)
    
    # Apply the code to the AST
    apply_to_ast(code, action.location)
    
    # Validate
    run_tests()
    check_types()
```

Transformers see:
- A template with `{{HOLE}}` markers
- Type constraints
- Surrounding context
- Hard token limits

They return ONLY the code to fill the hole.

## Training Data Collection

Training requires (state, action, next_state) tuples. Sources:

1. **Synthetic Edits**: Apply random valid actions to real repos
2. **Historical Commits**: Parse commits as actions, extract state before/after
3. **Deliberate Perturbations**: Make changes and measure consequences

Key: we need to capture the CONSEQUENCES of edits, not just the edits themselves.

## Safety Properties

By construction, the system provides:

1. **Bounded Generation**: Transformers only fill small holes
2. **Validation Before Commit**: JEPA predicts outcomes before execution
3. **Rollback Capability**: All actions have inverses
4. **Test Grounding**: Success is defined by tests passing
5. **Type Safety**: Julia's type system catches errors

## Future Directions

1. **Learning from Failures**: Use failed edits as negative examples
2. **Active Learning**: Query user when uncertain
3. **Multi-Repo Learning**: Transfer knowledge across codebases
4. **Hierarchical Planning**: Decompose large goals into subgoals
5. **Explanation Generation**: Explain why actions are safe/unsafe

## Comparison to Traditional Approaches

| Aspect | Traditional LLM Agent | JEPA Agent |
|--------|----------------------|------------|
| Representation | Tokens | Semantic graph |
| Context limit | Fixed window | Unlimited (embedded) |
| Planning | None or CoT | Beam search in embedding space |
| Generation | Free-form | Constrained templates |
| Validation | Post-hoc | Predictive |
| Hallucination | Common | Reduced by construction |

## Conclusion

The JEPA Julia Agent represents a fundamentally different approach to LLM-based code modification. By treating code as a semantic world and predicting consequences before acting, we can build agents that are more reliable, less prone to hallucination, and capable of reasoning about large codebases.
