"""Parameter loading utilities with layered overrides support.

This module loads the root-level ``params.yaml`` and optionally layers a
product-level ``params.yaml`` (e.g., under ``products/<product_id>/``) on top
when present. The product overrides are applied as a deep merge so that only
the differing keys need to be specified in the product file.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from box import ConfigBox
from dotenv import find_dotenv, load_dotenv


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    Parameters
    ----------
    path: Path
        Path to a YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed YAML content.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f.read())
    return data if isinstance(data, dict) else {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries (``override`` takes precedence).

    This function returns a new dictionary and does not mutate inputs.

    Parameters
    ----------
    base: Dict[str, Any]
        Base dictionary providing default values.
    override: Dict[str, Any]
        Dictionary whose values override those in ``base``.

    Returns
    -------
    Dict[str, Any]
        Merged dictionary with overrides applied.
    """
    result: Dict[str, Any] = {}
    for key in base.keys() | override.keys():
        base_val = base.get(key)
        over_val = override.get(key)
        if isinstance(base_val, dict) and isinstance(over_val, dict):
            result[key] = _deep_merge(base_val, over_val)
        elif over_val is not None:
            result[key] = over_val
        else:
            result[key] = base_val
    return result


def parse_params() -> Dict[str, Any]:
    """Load root params and layer product overrides if available.

    The root params file is resolved from the ``PROJECT_ROOT`` environment
    variable (configured via ``.env``). If a ``params.yaml`` exists in the
    current working directory and is different from the root file, its values
    will be deeply merged on top of the root defaults. Additionally, if the
    ``PRODUCT_PARAMS`` environment variable points to an existing file, that
    file will be applied last as an extra override.

    Returns
    -------
    Dict[str, Any]
        Fully merged parameters.
    """
    load_dotenv(find_dotenv())

    project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
    root_params_path = project_root / "params.yaml"
    if not root_params_path.exists():
        raise FileNotFoundError(f"Root params.yaml not found at: {root_params_path}")

    params: Dict[str, Any] = _load_yaml(root_params_path)

    # Layer current working directory params.yaml if present (e.g., a product)
    cwd_params_path = (Path.cwd() / "params.yaml").resolve()
    if cwd_params_path.exists() and cwd_params_path != root_params_path:
        params = _deep_merge(params, _load_yaml(cwd_params_path))

    # Optional final override via explicit environment variable
    env_override_path = os.getenv("PRODUCT_PARAMS")
    if env_override_path:
        extra_path = Path(env_override_path).resolve()
        if extra_path.exists():
            params = _deep_merge(params, _load_yaml(extra_path))

    return params


# module-level cache for a mutable global config object
_GLOBAL_CFG: Optional[ConfigBox] = None


def update_config(overrides: dict) -> None:
    """
    Shallow/deep merge overrides into the cached config so subsequent get_config()
    calls return the merged result.
    """
    global _GLOBAL_CFG
    if _GLOBAL_CFG is None:
        _GLOBAL_CFG = ConfigBox(parse_params())
    merged = _deep_merge(_GLOBAL_CFG.to_dict(), overrides)
    _GLOBAL_CFG = ConfigBox(merged)


def get_config(subset: str | None = None) -> ConfigBox:
    """Return configuration as a ``ConfigBox`` for convenient attribute access.

    Parameters
    ----------
    subset: str | None
        Optional top-level key to return a sub-config.

    Returns
    -------
    ConfigBox
        Full or subset configuration wrapped in ``ConfigBox``.
    """
    global _GLOBAL_CFG
    if _GLOBAL_CFG is None:
        _GLOBAL_CFG = ConfigBox(parse_params())
    if subset is not None:
        return _GLOBAL_CFG[subset]
    return _GLOBAL_CFG


def get_config_old(subset: str | None = None) -> ConfigBox:
    """Return configuration as a ``ConfigBox`` for convenient attribute access.

    Parameters
    ----------
    subset: str | None
        Optional top-level key to return a sub-config.

    Returns
    -------
    ConfigBox
        Full or subset configuration wrapped in ``ConfigBox``.
    """
    if subset is not None:
        return ConfigBox(parse_params())[subset]
    return ConfigBox(parse_params())


if __name__ == "__main__":
    print(get_config())
