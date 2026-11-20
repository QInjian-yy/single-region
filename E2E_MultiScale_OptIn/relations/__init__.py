"""Import a new module here to register it; the training loop stays unchanged."""
from .registry import build_relation, normalize_relation_config, register_relation
from .identity import IdentityRelation
from .semantic_topk import SemanticTopKRelation
from .global_mhsa import GlobalMHSARelation

__all__ = ["build_relation", "normalize_relation_config", "register_relation",
           "IdentityRelation", "SemanticTopKRelation", "GlobalMHSARelation"]
