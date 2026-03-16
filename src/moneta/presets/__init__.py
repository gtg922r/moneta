"""Built-in preset configurations for common financial models.

Presets are bundled YAML files in the data/ subdirectory. They provide
ready-made stochastic model parameters (e.g., S&P 500 GBM, US inflation).
"""

from __future__ import annotations

import importlib.resources
from typing import Any

import yaml

from moneta import MonetaError


def _data_package() -> importlib.resources.abc.Traversable:
    """Return the traversable for the bundled data/ package."""
    return importlib.resources.files("moneta.presets") / "data"


def list_presets() -> list[str]:
    """Return sorted list of available preset names (without .yaml extension)."""
    data_dir = _data_package()
    names: list[str] = []
    for item in data_dir.iterdir():
        name = str(item.name)
        if name.endswith(".yaml"):
            names.append(name.removesuffix(".yaml"))
    return sorted(names)


def get_preset(name: str) -> dict[str, Any]:
    """Load a bundled preset by name and return its config dict.

    Raises MonetaError if the preset is not found, listing available presets.
    """
    data_dir = _data_package()
    preset_file = data_dir / f"{name}.yaml"

    try:
        content = preset_file.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        available = list_presets()
        raise MonetaError(
            f"Unknown preset '{name}'. "
            f"Available presets: {', '.join(available)}"
        )

    data = yaml.safe_load(content)
    if not isinstance(data, dict):
        raise MonetaError(
            f"Preset '{name}' is malformed — expected a YAML mapping, "
            f"got {type(data).__name__}"
        )
    return data
