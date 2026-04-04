"""
Parse the JSON observation returned by the Kotlin server into NumPy arrays.

The resulting dict matches the observation space defined in :mod:`rl_env.spaces`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from rl_env.constants import (
    ENTITY_FEATURES,
    MAX_ENTITIES,
    MAX_TILES,
    N_MAP_CHANNELS,
    N_SCALAR_FEATURES,
)


def parse_observation(obs_json: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw ``RLObservationResponse`` JSON dict from the server into a
    dict of NumPy arrays matching the Gymnasium observation space.
    """
    scalars = _parse_scalars(obs_json["scalars"])
    entities, visibility_mask = _parse_entities(obs_json["entities"])
    map_planes = _parse_map_planes(obs_json.get("mapPlanes"))

    info = {
        "gameId": obs_json.get("gameId", ""),
        "agentCivId": obs_json.get("agentCivId", ""),
        "agentIndex": obs_json.get("agentIndex", 0),
        "allAgents": obs_json.get("allAgents", []),
        "done": obs_json.get("done", False),
        "winner": obs_json.get("winner"),
        "scores": obs_json.get("scores", {}),
        "catalogues": obs_json.get("catalogues", {}),
        "fogOfWar": obs_json.get("fogOfWar", []),
    }

    return {
        "scalars": scalars,
        "entities": entities,
        "map_planes": map_planes,
        "visibility_mask": visibility_mask,
        "info": info,
    }


def parse_action_mask(mask_json: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Convert a raw ``RLActionMaskResponse`` JSON dict into a dict of boolean
    NumPy arrays keyed to match :meth:`rl_env.spaces.UncivSpaces.action_mask_space`.
    """
    def _bool_arr(lst: list) -> np.ndarray:
        return np.array(lst, dtype=np.int8)

    return {
        "macro": _bool_arr(mask_json.get("macroMask", [])),
        "unit_target": _bool_arr(mask_json.get("unitTargetMask", [False] * MAX_ENTITIES)),
        "unit_subaction": _bool_arr(mask_json.get("unitSubactionMask", [])),
        "city_target": _bool_arr(mask_json.get("cityTargetMask", [False] * MAX_ENTITIES)),
        "city_subaction": _bool_arr(mask_json.get("citySubactionMask", [])),
        "production_target": _bool_arr(mask_json.get("productionTargetMask", [])),
        "tech_target": _bool_arr(mask_json.get("techTargetMask", [])),
        "policy_target": _bool_arr(mask_json.get("policyTargetMask", [])),
        "diplomacy_target": _bool_arr(mask_json.get("diplomacyTargetMask", [])),
        "diplomacy_subaction": _bool_arr(mask_json.get("diplomacySubactionMask", [])),
        "tile_target": _bool_arr(mask_json.get("tileTargetMask", [False] * MAX_TILES)),
        "improvement_target": _bool_arr(mask_json.get("improvementTargetMask", [])),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_scalars(scalars_json: dict[str, Any]) -> np.ndarray:
    """
    Extract the 22 scalars in the agreed-upon order (matches SCALAR_* constants).
    """
    arr = np.array(
        [
            float(scalars_json.get("turn", 0)),
            float(scalars_json.get("gold", 0)),
            float(scalars_json.get("sciencePerTurn", 0.0)),
            float(scalars_json.get("culturePerTurn", 0.0)),
            float(scalars_json.get("faithPerTurn", 0.0)),
            float(scalars_json.get("happiness", 0)),
            float(scalars_json.get("netGoldPerTurn", 0.0)),
            float(scalars_json.get("cityCount", 0)),
            float(scalars_json.get("totalPopulation", 0)),
            float(scalars_json.get("techProgressFraction", 0.0)),
            float(scalars_json.get("policyProgressFraction", 0.0)),
            float(scalars_json.get("eraIndex", 0)),
            float(scalars_json.get("isInGoldenAge", 0)),
            float(scalars_json.get("goldenAgeTurnsLeft", 0)),
            float(scalars_json.get("freePolicies", 0)),
            float(scalars_json.get("currentTechIndex", -1)),
            float(scalars_json.get("warsCount", 0)),
            float(scalars_json.get("score", 0)),
            float(scalars_json.get("scienceVictoryProgress", 0.0)),
            float(scalars_json.get("cultureVictoryProgress", 0.0)),
            float(scalars_json.get("dominationVictoryProgress", 0.0)),
            float(scalars_json.get("diploVictoryProgress", 0.0)),
        ],
        dtype=np.float32,
    )
    assert arr.shape == (N_SCALAR_FEATURES,), (
        f"Expected {N_SCALAR_FEATURES} scalars, got {arr.shape}"
    )
    return arr


def _parse_entities(entities_json: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    entity_arr = np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
    visibility_mask = np.zeros(MAX_ENTITIES, dtype=np.int8)

    for i, ent in enumerate(entities_json[:MAX_ENTITIES]):
        features = ent.get("features", [])
        length = min(len(features), ENTITY_FEATURES)
        entity_arr[i, :length] = features[:length]
        if features and features[0] != 0:  # entityType != 0 → real entity
            visibility_mask[i] = 1

    return entity_arr, visibility_mask


def _parse_map_planes(map_planes_json: dict[str, Any] | None) -> np.ndarray:
    if map_planes_json is None:
        return np.zeros((N_MAP_CHANNELS, MAX_TILES), dtype=np.float32)

    channels = map_planes_json.get("channels", N_MAP_CHANNELS)
    height = map_planes_json.get("height", 1)
    width = map_planes_json.get("width", 1)
    data = map_planes_json.get("data", [])

    tile_count = height * width
    raw = np.array(data, dtype=np.float32)
    expected = channels * tile_count
    if len(raw) < expected:
        warnings.warn(
            f"MapPlanes data has {len(raw)} elements but expected {expected}. Padding with zeros.",
            RuntimeWarning,
            stacklevel=2,
        )
        raw = np.pad(raw, (0, expected - len(raw)))
    reshaped = raw[:expected].reshape(channels, tile_count)

    result = np.zeros((N_MAP_CHANNELS, MAX_TILES), dtype=np.float32)
    ch = min(channels, N_MAP_CHANNELS)
    tc = min(tile_count, MAX_TILES)
    result[:ch, :tc] = reshaped[:ch, :tc]
    return result
