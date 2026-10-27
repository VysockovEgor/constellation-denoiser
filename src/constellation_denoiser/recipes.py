"""Training recipes per family x level chosen by E025 (results/recipes.json); read by E026-E033.

    from constellation_denoiser.recipes import load_recipe
    recipe = load_recipe("F2")      # {"lr": {...LRConfig...}, "spec_overrides": {...}, "source": "..."}

Until E025 has been summarized on the server the fallback recipe is returned: cosine_to_zero,
peak 3e-3, no warmup, c_mult 30 (the E019/E020 best long regime). ``source`` says which one was used,
and every run that uses a recipe stores it (task meta ``recipe_source``).
"""

from __future__ import annotations

import json
import os

from .lr_schedules import LRConfig
from .paths import PROJECT_ROOT

RECIPES_PATH = PROJECT_ROOT / "results" / "recipes.json"
FALLBACK_LR = {"peak_lr": 3e-3, "schedule": "cosine_to_zero", "warmup": 0, "c_mult": 30.0, "floor_frac": 0.1,
               "optimizer": "adam", "weight_decay": 0.0, "horizon": None}
FAMILY_LEVELS = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13")


def recipes_path(smoke: bool | None = None) -> "os.PathLike":
    if smoke is None:
        smoke = os.environ.get("SMOKE", "0") == "1"
    smoke_path = PROJECT_ROOT / "results_smoke" / "recipes.json"
    return smoke_path if smoke and smoke_path.is_file() else RECIPES_PATH


def load_all(smoke: bool | None = None) -> dict:
    path = recipes_path(smoke)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("recipes", {})


def load_recipe(family_level: str, smoke: bool | None = None) -> dict:
    """Recipe of one E025 pair (F1 ... F13); the fallback if E025 has not produced it."""
    if family_level not in FAMILY_LEVELS:
        raise KeyError(f"unknown family/level {family_level!r}; expected one of {FAMILY_LEVELS}")
    item = load_all(smoke).get(family_level)
    if item is None:
        return {"lr": dict(FALLBACK_LR), "spec_overrides": {}, "source": "fallback: cosine_to_zero 3e-3 wu0 cm30"}
    recipe = item["recipe"]
    lr = LRConfig.from_dict({**recipe["lr"], "horizon": None}).to_dict()
    return {"lr": lr, "spec_overrides": dict(recipe.get("spec_overrides", {})),
            "source": f"{recipes_path(smoke).name}: {family_level} = {recipe.get('name', 'unknown')}"}


def lr_tag(lr: dict) -> str:
    return LRConfig.from_dict(lr).tag()
