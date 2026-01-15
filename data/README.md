# Data Directory

This directory contains training and evaluation data for the JEPA world model.

## Structure

```
data/
├── repos/          # Cloned Julia repositories for analysis
├── transitions/    # (state, action, next_state) training tuples
└── embeddings/     # Pre-computed state embeddings
```

## Data Collection Pipeline

1. **Clone repositories** → `data/repos/`
2. **Extract world states** → Generate snapshots
3. **Generate transitions** → Apply actions, record outcomes
4. **Compute embeddings** → Pre-compute for efficiency

## Recommended Repositories

Start with well-tested Julia packages:
- DataFrames.jl
- Flux.jl
- DifferentialEquations.jl
- Plots.jl
- JuMP.jl

## Data Format

### Transitions (JSON Lines)
```json
{
  "repo": "DataFrames.jl",
  "state_hash": "abc123",
  "action": {"type": "ADD_METHOD", "target": "..."},
  "next_state_hash": "def456",
  "test_passed": true,
  "invalidations": 0
}
```

### Embeddings (NPZ)
```python
{
  "state_hash": np.array([...]),  # 512-dim
  "action_hash": np.array([...]), # 128-dim
}
```
