"""
Parse the JSON observation returned by the Kotlin server into NumPy arrays.

The resulting dict matches the observation space defined in :mod:`rl_env.spaces`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_env.constants import (
    ENTITY_FEATURES,
    MAX_ENTITIES,
    N_MAP_CHANNELS,
    N_SCALAR_FEATURES,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_observation(obs_json: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw ``RLObservationResponse`` JSON dict from the server into a
    dict of NumPy arrays matching the Gymnasium observation space.

    Parameters
    ----------
    obs_json:
        The JSON object returned by ``GET /rl/state/{gameId}``.

    Returns
    -------
    dict with keys:
        * ``"scalars"``   – ``np.ndarray`` of shape ``(N_SCALAR_FEATURES,)``
        * ``"entities"``  – ``np.ndarray`` of shape ``(MAX_ENTITIES, ENTITY_FEATURES)``
        * ``"map_planes"``– ``np.ndarray`` of shape ``(N_MAP_CHANNELS, tile_count)``
        * ``"visibility_mask"`` – ``np.ndarray`` of shape ``(MAX_ENTITIES,)``
        * ``"info"``      – raw Python dict with non-array metadata
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
    NumPy arrays, keyed to match the :meth:`rl_env.spaces.UncivSpaces.action_mask_space`.
    """
    def _bool_arr(lst: list[bool]) -> np.ndarray:
        return np.array(lst, dtype=np.int8)

    return {
        "macro": _bool_arr(mask_json.get("macroMask", [])),
        "unit_target": _bool_arr(mask_json.get("unitTargetMask", [False] * MAX_ENTITIES)),
        "unit_subaction": _bool_arr(mask_json.get("unitSubactionMask", [])),
        "city_target": _bool_arr(mask_json.get("cityTargetMask", [False] * MAX_ENTITIES)),
        "tech_target": _bool_arr(mask_json.get("techTargetMask", [])),
        "diplomacy_target": _bool_arr(mask_json.get("diplomacyTargetMask", [])),
        "diplomacy_subaction": _bool_arr(mask_json.get("diplomacySubactionMask", [])),
        "policy_target": _bool_arr(mask_json.get("policyTargetMask", [])),
        "settler_target": _bool_arr(mask_json.get("settlerTargetMask", [False] * MAX_ENTITIES)),
        "production_items": _bool_arr(mask_json.get("productionItemMask", [])),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_scalars(scalars_json: dict[str, Any]) -> np.ndarray:
    """
    Extract the N_SCALAR_FEATURES scalars in the agreed-upon order.

    Scalar order (must match constants.py and the Python spaces definition):
      0  turn
      1  currentPlayerId
      2  gold
      3  sciencePerTurn
      4  culturePerTurn
      5  faithPerTurn
      6  happiness
      7  netGoldPerTurn
      8  cityCount
      9  totalPopulation
     10  techProgressCurrent
     11  policyProgressCurrent
     12  warStateFlags
     13  victoryProgress
    """
    arr = np.array(
        [
            float(scalars_json.get("turn", 0)),
            float(scalars_json.get("currentPlayerId", 0)),
            float(scalars_json.get("gold", 0)),
            float(scalars_json.get("sciencePerTurn", 0.0)),
            float(scalars_json.get("culturePerTurn", 0.0)),
            float(scalars_json.get("faithPerTurn", 0.0)),
            float(scalars_json.get("happiness", 0)),
            float(scalars_json.get("netGoldPerTurn", 0.0)),
            float(scalars_json.get("cityCount", 0)),
            float(scalars_json.get("totalPopulation", 0)),
            float(scalars_json.get("techProgressCurrent", 0.0)),
            float(scalars_json.get("policyProgressCurrent", 0.0)),
            float(scalars_json.get("warStateFlags", 0)),
            float(scalars_json.get("victoryProgress", 0.0)),
        ],
        dtype=np.float32,
    )
    assert arr.shape == (N_SCALAR_FEATURES,), (
        f"Expected {N_SCALAR_FEATURES} scalars, got {arr.shape}"
    )
    return arr


def _parse_entities(entities_json: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the padded entity list into a float32 array and a visibility mask.

    Returns
    -------
    entities : np.ndarray, shape (MAX_ENTITIES, ENTITY_FEATURES)
    visibility_mask : np.ndarray of int8, shape (MAX_ENTITIES,)
        1 for real (non-padding) entity slots, 0 for padding.
    """
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
    """
    Convert the optional ``MapPlanes`` payload into a float32 array of shape
    ``(N_MAP_CHANNELS, tile_count)``.

    If *map_planes_json* is ``None`` (server did not include map planes), a
    zero array of shape ``(N_MAP_CHANNELS, MAX_ENTITIES)`` is returned instead,
    where ``MAX_ENTITIES`` serves as a placeholder tile count.
    """
    if map_planes_json is None:
        return np.zeros((N_MAP_CHANNELS, MAX_ENTITIES), dtype=np.float32)

    channels = map_planes_json.get("channels", N_MAP_CHANNELS)
    height = map_planes_json.get("height", 1)
    width = map_planes_json.get("width", 1)
    data = map_planes_json.get("data", [])

    tile_count = height * width
    raw = np.array(data, dtype=np.float32)
    # data is stored as channel × tileCount (flat)
    expected = channels * tile_count
    if len(raw) < expected:
        import warnings
        warnings.warn(
            f"MapPlanes data has {len(raw)} elements but expected {expected} "
            f"({channels} channels × {tile_count} tiles). Padding with zeros.",
            RuntimeWarning,
            stacklevel=2,
        )
        raw = np.pad(raw, (0, expected - len(raw)))
    return raw[:expected].reshape(channels, tile_count)
