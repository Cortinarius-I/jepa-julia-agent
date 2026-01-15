"""
JEPA World Model for code understanding.

Components:
- JEPAWorldModel: Main world model for state prediction
- MultiViewJEPA: Multi-view training (LLM-JEPA style)
- View encoders for different modalities
"""
from agent.jepa.model import (
    JEPAWorldModel,
    JEPATrainingConfig,
    create_jepa_model,
    WorldStateEncoder,
    ActionEncoder,
    JEPAPredictor,
)

from agent.jepa.multi_view import (
    MultiViewJEPA,
    MultiViewJEPALoss,
    MultiViewJEPATrainer,
    ViewType,
    ViewPair,
    ViewEncoder,
    NLGoalEncoder,
    ActionSequenceEncoder,
    create_block_causal_mask,
    analyze_embedding_structure,
)

__all__ = [
    # Core model
    "JEPAWorldModel",
    "JEPATrainingConfig",
    "create_jepa_model",
    "WorldStateEncoder",
    "ActionEncoder",
    "JEPAPredictor",
    # Multi-view (LLM-JEPA style)
    "MultiViewJEPA",
    "MultiViewJEPALoss",
    "MultiViewJEPATrainer",
    "ViewType",
    "ViewPair",
    "ViewEncoder",
    "NLGoalEncoder",
    "ActionSequenceEncoder",
    "create_block_causal_mask",
    "analyze_embedding_structure",
]
