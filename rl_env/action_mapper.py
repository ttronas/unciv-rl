"""
Encode Python-side actions into the wire format understood by the Kotlin server,
and decode server responses back to Python dicts.

The Kotlin server expects a JSON body with 12 semantic camelCase fields
(see RLAction data class in RLActionExecutor.kt).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_env.constants import (
    N_MACRO_ACTIONS,
    MAX_ENTITIES,
    MAX_TILES,
    MAX_IMPROVEMENTS,
    MAX_TECHS,
    MAX_POLICIES,
    MAX_PROD_ITEMS,
    MAX_AGENTS,
    N_UNIT_SUBACTIONS,
    N_CITY_SUBACTIONS,
    N_DIPLOMACY_SUBACTIONS,
    MACRO_NAMES,
    MACRO_END_TURN,
    MACRO_UNIT_MOVE,
    MACRO_UNIT_ATTACK,
    MACRO_UNIT_ABILITY,
    MACRO_CITY_ACTION,
    MACRO_TECH_RESEARCH,
    MACRO_POLICY_ADOPT,
    MACRO_DIPLOMACY,
    MACRO_WORKER_BUILD,
    MACRO_FOUNDER_SETTLE,
    UNIT_SUBACTION_NAMES,
    CITY_SUBACTION_NAMES,
    DIPLOMACY_SUBACTION_NAMES,
)

# Bounds for validation
_BOUNDS = {
    "macro": N_MACRO_ACTIONS,
    "unitTarget": MAX_ENTITIES,
    "unitSubaction": N_UNIT_SUBACTIONS,
    "cityTarget": MAX_ENTITIES,
    "citySubaction": N_CITY_SUBACTIONS,
    "productionTarget": MAX_PROD_ITEMS,
    "techTarget": MAX_TECHS,
    "policyTarget": MAX_POLICIES,
    "diplomacyTarget": MAX_AGENTS,
    "diplomacySubaction": N_DIPLOMACY_SUBACTIONS,
    "tileTarget": MAX_TILES,
    "improvementTarget": MAX_IMPROVEMENTS,
}


def encode_action(action: dict[str, int] | np.ndarray) -> dict[str, int]:
    """
    Convert a semantic action dict into the camelCase JSON payload expected by
    ``POST /rl/action/{gameId}``.

    Accepts:
    - **Semantic snake_case dict** (keys from action_space):
      ``macro``, ``unit_target``, ``unit_subaction``, ``city_target``,
      ``city_subaction``, ``production_target``, ``tech_target``,
      ``policy_target``, ``diplomacy_target``, ``diplomacy_subaction``,
      ``tile_target``, ``improvement_target``.
    - **camelCase dict** (already wire-format, passed through).
    - **5-element array** ``[macro, unit_target, tile_target, subaction, arg]``
      (legacy compatibility).

    Returns a dict with 12 camelCase integer fields.
    """
    if isinstance(action, (np.ndarray, list, tuple)):
        arr = [int(x) for x in action]
        if len(arr) != 5:
            raise ValueError(f"Action array must be length 5, got {len(arr)}")
        macro = arr[0]
        wire = _array_to_wire(macro, arr)
    elif _is_camel(action):
        # Already camelCase — just ensure all 12 fields exist
        wire = {k: int(action.get(k, 0)) for k in _BOUNDS}
        wire["macro"] = int(action.get("macro", 0))
    else:
        macro = int(action.get("macro", 0))
        wire = {
            "macro": macro,
            "unitTarget": int(action.get("unit_target", 0)),
            "unitSubaction": int(action.get("unit_subaction", 0)),
            "cityTarget": int(action.get("city_target", 0)),
            "citySubaction": int(action.get("city_subaction", 0)),
            "productionTarget": int(action.get("production_target", 0)),
            "techTarget": int(action.get("tech_target", 0)),
            "policyTarget": int(action.get("policy_target", 0)),
            "diplomacyTarget": int(action.get("diplomacy_target", 0)),
            "diplomacySubaction": int(action.get("diplomacy_subaction", 0)),
            "tileTarget": int(action.get("tile_target", 0)),
            "improvementTarget": int(action.get("improvement_target", 0)),
        }

    _validate(wire)
    return wire


def decode_action_result(result: dict[str, Any]) -> tuple[bool, str, float]:
    """Unpack an ``ActionResult`` response dict → (success, message, reward)."""
    return (
        bool(result.get("success", False)),
        str(result.get("message", "")),
        float(result.get("reward", 0.0)),
    )


def describe_action(action: dict[str, int]) -> str:
    """Return a human-readable description of a wire or semantic action dict."""
    # Support both camelCase and snake_case
    macro = int(action.get("macro", 0))
    macro_name = MACRO_NAMES[macro] if macro < len(MACRO_NAMES) else f"macro_{macro}"

    if macro == MACRO_END_TURN:
        return "end_turn"
    if macro == MACRO_UNIT_MOVE:
        unit = action.get("unitTarget", action.get("unit_target", 0))
        tile = action.get("tileTarget", action.get("tile_target", 0))
        return f"unit_move(unit={unit}, tile={tile})"
    if macro == MACRO_UNIT_ATTACK:
        unit = action.get("unitTarget", action.get("unit_target", 0))
        tile = action.get("tileTarget", action.get("tile_target", 0))
        return f"unit_attack(unit={unit}, tile={tile})"
    if macro == MACRO_UNIT_ABILITY:
        unit = action.get("unitTarget", action.get("unit_target", 0))
        sub = action.get("unitSubaction", action.get("unit_subaction", 0))
        sub_name = UNIT_SUBACTION_NAMES[sub] if sub < len(UNIT_SUBACTION_NAMES) else f"sub_{sub}"
        return f"unit_ability(unit={unit}, sub={sub_name})"
    if macro == MACRO_CITY_ACTION:
        city = action.get("cityTarget", action.get("city_target", 0))
        sub = action.get("citySubaction", action.get("city_subaction", 0))
        sub_name = CITY_SUBACTION_NAMES[sub] if sub < len(CITY_SUBACTION_NAMES) else f"sub_{sub}"
        prod = action.get("productionTarget", action.get("production_target", 0))
        return f"city_action(city={city}, sub={sub_name}, prod={prod})"
    if macro == MACRO_TECH_RESEARCH:
        return f"tech_research(tech={action.get('techTarget', action.get('tech_target', 0))})"
    if macro == MACRO_POLICY_ADOPT:
        return f"policy_adopt(policy={action.get('policyTarget', action.get('policy_target', 0))})"
    if macro == MACRO_DIPLOMACY:
        target = action.get("diplomacyTarget", action.get("diplomacy_target", 0))
        sub = action.get("diplomacySubaction", action.get("diplomacy_subaction", 0))
        sub_name = DIPLOMACY_SUBACTION_NAMES[sub] if sub < len(DIPLOMACY_SUBACTION_NAMES) else f"sub_{sub}"
        return f"diplomacy(target={target}, sub={sub_name})"
    if macro == MACRO_WORKER_BUILD:
        unit = action.get("unitTarget", action.get("unit_target", 0))
        tile = action.get("tileTarget", action.get("tile_target", 0))
        impr = action.get("improvementTarget", action.get("improvement_target", 0))
        return f"worker_build(unit={unit}, tile={tile}, impr={impr})"
    if macro == MACRO_FOUNDER_SETTLE:
        return f"founder_settle(unit={action.get('unitTarget', action.get('unit_target', 0))})"
    return f"{macro_name}(...)"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_camel(action: dict) -> bool:
    return any(k in action for k in ("unitTarget", "cityTarget", "tileTarget"))


def _array_to_wire(macro: int, arr: list[int]) -> dict[str, int]:
    """Convert legacy 5-element array to 12-field wire dict."""
    _, a1, a2, a3, a4 = arr
    base = {k: 0 for k in _BOUNDS}
    base["macro"] = macro
    if macro == MACRO_UNIT_MOVE:
        base["unitTarget"] = a1; base["tileTarget"] = a2
    elif macro == MACRO_UNIT_ATTACK:
        base["unitTarget"] = a1; base["tileTarget"] = a2
    elif macro == MACRO_UNIT_ABILITY:
        base["unitTarget"] = a1; base["unitSubaction"] = a3
    elif macro == MACRO_CITY_ACTION:
        base["cityTarget"] = a1; base["citySubaction"] = a3; base["productionTarget"] = a2
    elif macro == MACRO_TECH_RESEARCH:
        base["techTarget"] = a1
    elif macro == MACRO_POLICY_ADOPT:
        base["policyTarget"] = a1
    elif macro == MACRO_DIPLOMACY:
        base["diplomacyTarget"] = a1; base["diplomacySubaction"] = a3
    elif macro == MACRO_WORKER_BUILD:
        base["unitTarget"] = a1; base["tileTarget"] = a2; base["improvementTarget"] = a4
    elif macro == MACRO_FOUNDER_SETTLE:
        base["unitTarget"] = a1
    return base


def _validate(wire: dict[str, int]) -> None:
    macro = wire.get("macro", 0)
    if not (0 <= macro < N_MACRO_ACTIONS):
        raise ValueError(f"macro={macro} out of range [0, {N_MACRO_ACTIONS})")
