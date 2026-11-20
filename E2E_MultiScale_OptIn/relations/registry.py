"""Factories for one full-WSI region relation, selected independently of MIL."""
from __future__ import annotations

_REGISTRY = {}


def register_relation(name):
    def register(module_class):
        if name in _REGISTRY:
            raise ValueError(f"Region relation already registered: {name}")
        _REGISTRY[name] = module_class
        return module_class
    return register


def normalize_relation_config(settings=None):
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("region_relation must be a mapping")
    settings = dict(settings)
    enabled = settings.pop("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("region_relation.enabled must be a boolean")
    if not enabled:
        return {"enabled": False}
    name = settings.pop("name", "identity")
    if not isinstance(name, str) or name not in _REGISTRY:
        raise ValueError(f"Unknown region relation {name!r}; choose {sorted(_REGISTRY)}")
    return {"enabled": True, "name": name,
            **_REGISTRY[name].normalize_config(settings)}


def build_relation(config=None, dim=256):
    """Accept either the region_relation block or a model config containing it."""
    if type(dim) is not int or dim < 1:
        raise ValueError("Region embedding dimension must be a positive integer")
    if isinstance(config, dict) and "region_relation" in config:
        config = config["region_relation"]
    settings = normalize_relation_config(config)
    name = settings.pop("name", "identity")
    settings.pop("enabled")
    return _REGISTRY[name](dim=dim, **settings)
