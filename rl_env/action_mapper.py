"""
Encode Python-side actions into the wire format understood by the Kotlin server,
and decode server responses back to Python dicts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_env.constants import (
    N_MACRO_ACTIONS,
    MAX_TARGETS,
    MAX_SUBACTIONS,
    MAX_ARG1,
    MAX_ARG2,
    MACRO_NAMES,
    MACRO_END_TURN,
    MACRO_UNIT_ACTION,
    MACRO_CITY_ACTION,
    MACRO_TECH_ACTION,
    MACRO_DIPLOMACY_ACTION,
    MACRO_POLICY_ACTION,
    MACRO_SETTLER_ACTION,
    MACRO_IMPROVE_TILE,
    UNIT_SUBACTION_FOUND_CITY,
    UNIT_SUBACTION_NAMES,
    CITY_SUBACTION_NAMES,
    DIPLOMACY_SUBACTION_NAMES,
)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_action(action: dict[str, int] | np.ndarray) -> dict[str, int]:
    """
    Validate and convert an action into the JSON-serialisable payload expected
    by ``POST /rl/action/{gameId}``.

    The function accepts three input formats:

    1. **Semantic dict** (keys matching :meth:`rl_env.spaces.UncivSpaces.action_space`):
       ``macro``, ``unit_target``, ``unit_subaction``, ``city_target``,
       ``city_subaction``, ``tech_target``, ``diplomacy_target``,
       ``diplomacy_subaction``, ``policy_target``, ``settler_target``,
       ``production_items``.  The appropriate target/subaction/arg1/arg2 fields
       are selected based on the value of ``macro``.

    2. **Wire dict** (legacy, keys ``macro``, ``target``, ``subaction``,
       ``arg1``, ``arg2``): passed through as-is after bounds validation.

    3. **Array** ``[macro, target, subaction, arg1, arg2]``: 5-element
       integer array treated as wire format.

    Returns
    -------
    dict with integer values for ``macro``, ``target``, ``subaction``,
    ``arg1``, and ``arg2``.

    Raises
    ------
    ValueError
        If any component is outside its valid range.
    """
    if isinstance(action, (np.ndarray, list, tuple)):
        action_array = list(action)
        if len(action_array) != 5:
            raise ValueError(
                f"Action array must have 5 elements [macro, target, subaction, arg1, arg2], "
                f"got {len(action_array)}"
            )
        macro, target, subaction, arg1, arg2 = (int(x) for x in action_array)
    elif _is_semantic(action):
        macro = int(action.get("macro", 0))
        target, subaction, arg1, arg2 = _decode_semantic(macro, action)
    else:
        # Wire format: target, subaction, arg1, arg2 supplied directly
        macro = int(action.get("macro", 0))
        target = int(action.get("target", 0))
        subaction = int(action.get("subaction", 0))
        arg1 = int(action.get("arg1", 0))
        arg2 = int(action.get("arg2", 0))

    _validate_bounds("macro", macro, N_MACRO_ACTIONS)
    _validate_bounds("target", target, MAX_TARGETS)
    _validate_bounds("subaction", subaction, MAX_SUBACTIONS)
    _validate_bounds("arg1", arg1, MAX_ARG1)
    _validate_bounds("arg2", arg2, MAX_ARG2)

    return {
        "macro": macro,
        "target": target,
        "subaction": subaction,
        "arg1": arg1,
        "arg2": arg2,
    }


def decode_action_result(result: dict[str, Any]) -> tuple[bool, str, float]:
    """
    Unpack an ``ActionResult`` response dict.

    Returns
    -------
    success : bool
    message : str
    reward : float
    """
    return (
        bool(result.get("success", False)),
        str(result.get("message", "")),
        float(result.get("reward", 0.0)),
    )


# ---------------------------------------------------------------------------
# Helper: action-space description (for humans / debugging)
# ---------------------------------------------------------------------------

def describe_action(action: dict[str, int]) -> str:
    """Return a human-readable description of an action dict."""
    macro = action.get("macro", 0)
    target = action.get("target", 0)
    subaction = action.get("subaction", 0)
    arg1 = action.get("arg1", 0)
    arg2 = action.get("arg2", 0)

    macro_name = MACRO_NAMES[macro] if macro < len(MACRO_NAMES) else f"macro_{macro}"

    if macro == 0:
        return "end_turn"
    if macro == 1:
        sub_name = (
            UNIT_SUBACTION_NAMES[subaction]
            if subaction < len(UNIT_SUBACTION_NAMES)
            else f"unit_sub_{subaction}"
        )
        return f"unit_action(target={target}, sub={sub_name}, arg1={arg1})"
    if macro == 2:
        sub_name = (
            CITY_SUBACTION_NAMES[subaction]
            if subaction < len(CITY_SUBACTION_NAMES)
            else f"city_sub_{subaction}"
        )
        return f"city_action(target={target}, sub={sub_name}, arg1={arg1})"
    if macro == 3:
        return f"tech_action(tech_idx={target})"
    if macro == 4:
        sub_name = (
            DIPLOMACY_SUBACTION_NAMES[subaction]
            if subaction < len(DIPLOMACY_SUBACTION_NAMES)
            else f"dipl_sub_{subaction}"
        )
        return f"diplomacy_action(civ_idx={target}, sub={sub_name})"
    if macro == 5:
        return f"policy_action(policy_idx={target})"
    if macro == 6:
        return f"settler_action(unit_idx={target})"
    if macro == 7:
        return f"improve_tile(tile_idx={target}, worker_idx={arg1}, impr_idx={arg2})"

    return f"{macro_name}(target={target}, sub={subaction}, arg1={arg1}, arg2={arg2})"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_semantic(action: dict) -> bool:
    """Return True when *action* uses the semantic key format from action_space."""
    return any(k in action for k in (
        "unit_target", "city_target", "tech_target",
        "diplomacy_target", "policy_target", "settler_target",
    ))


def _decode_semantic(macro: int, action: dict) -> tuple[int, int, int, int]:
    """Map semantic action keys to (target, subaction, arg1, arg2) wire format."""
    if macro == MACRO_END_TURN:
        return 0, 0, 0, 0
    if macro == MACRO_UNIT_ACTION:
        return (
            int(action.get("unit_target", 0)),
            int(action.get("unit_subaction", 0)),
            0,  # tile/promotion index not in semantic space; server validates
            0,
        )
    if macro == MACRO_CITY_ACTION:
        return (
            int(action.get("city_target", 0)),
            int(action.get("city_subaction", 0)),
            int(action.get("production_items", 0)),  # arg1 = production item index
            0,
        )
    if macro == MACRO_TECH_ACTION:
        return int(action.get("tech_target", 0)), 0, 0, 0
    if macro == MACRO_DIPLOMACY_ACTION:
        return (
            int(action.get("diplomacy_target", 0)),
            int(action.get("diplomacy_subaction", 0)),
            0,
            0,
        )
    if macro == MACRO_POLICY_ACTION:
        return int(action.get("policy_target", 0)), 0, 0, 0
    if macro == MACRO_SETTLER_ACTION:
        return int(action.get("settler_target", 0)), UNIT_SUBACTION_FOUND_CITY, 0, 0
    if macro == MACRO_IMPROVE_TILE:
        # wire: target=tile_idx, subaction=improve_type, arg1=worker_entity_idx
        # semantic: unit_target=worker entity (no explicit tile_target in space)
        return 0, 0, int(action.get("unit_target", 0)), 0
    return 0, 0, 0, 0


def _validate_bounds(name: str, value: int, max_exclusive: int) -> None:
    if not (0 <= value < max_exclusive):
        raise ValueError(
            f"Action component '{name}' = {value} is out of bounds [0, {max_exclusive})."
        )
