"""YAML model loader with preset resolution and deep-merge for sweeps.

Loads a .moneta.yaml file from disk, resolves any preset references,
and returns a validated ScenarioModel.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from moneta import MonetaError
from moneta.parser.models import ScenarioModel
from moneta.presets import get_preset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(path: Path) -> ScenarioModel:
    """Load a .moneta.yaml file and return a validated ScenarioModel.

    This is the main entry point for loading model files.
    """
    raw = _load_yaml(path)
    resolved = _resolve_presets(raw)
    try:
        return ScenarioModel.model_validate(resolved)
    except Exception as exc:
        raise MonetaError(f"Validation error in '{path}': {exc}") from exc


def load_model_from_string(yaml_string: str) -> ScenarioModel:
    """Parse a YAML string and return a validated ScenarioModel.

    Useful for testing without writing files to disk.
    """
    try:
        raw = yaml.safe_load(yaml_string)
    except yaml.YAMLError as exc:
        raise MonetaError(f"Malformed YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise MonetaError(
            f"Expected a YAML mapping at the top level, got {type(raw).__name__}"
        )

    resolved = _resolve_presets(raw)
    try:
        return ScenarioModel.model_validate(resolved)
    except Exception as exc:
        raise MonetaError(f"Validation error: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    """Load and parse a YAML file, returning the raw dict.

    Raises MonetaError for file-not-found or malformed YAML.
    """
    path = Path(path)
    if not path.exists():
        raise MonetaError(f"Model file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MonetaError(f"Cannot read model file '{path}': {exc}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise MonetaError(f"Malformed YAML in '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise MonetaError(
            f"Expected a YAML mapping at the top level of '{path}', "
            f"got {type(data).__name__}"
        )
    return data


def _resolve_presets(raw_dict: dict) -> dict:
    """Walk the raw dict and resolve any preset references.

    Looks for ``preset: name`` in growth sections (under assets) and
    inflation sections (under global). Replaces the preset reference
    with the preset's config data.
    """
    result = copy.deepcopy(raw_dict)

    # Resolve asset growth presets
    assets = result.get("assets")
    if isinstance(assets, dict):
        for asset_name, asset_data in assets.items():
            if not isinstance(asset_data, dict):
                continue
            growth = asset_data.get("growth")
            if isinstance(growth, dict) and "preset" in growth and len(growth) == 1:
                preset_name = growth["preset"]
                asset_data["growth"] = get_preset(preset_name)

    # Resolve global inflation preset
    global_cfg = result.get("global")
    if isinstance(global_cfg, dict):
        inflation = global_cfg.get("inflation")
        if isinstance(inflation, dict) and "preset" in inflation and len(inflation) == 1:
            preset_name = inflation["preset"]
            global_cfg["inflation"] = get_preset(preset_name)

    return result


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*, returning a new dict.

    - Dict values are recursively merged.
    - List values and scalars in *overrides* replace the base entirely.
    - Keys in *base* that are not in *overrides* are preserved.
    """
    merged = copy.deepcopy(base)
    for key, override_val in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(override_val, dict)
        ):
            merged[key] = deep_merge(merged[key], override_val)
        else:
            merged[key] = copy.deepcopy(override_val)
    return merged
