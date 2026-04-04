"""
Flatten observation wrapper.

Converts the nested ``Dict`` observation from :class:`rl_env.UncivEnv` into a
single 1-D ``Box`` observation by concatenating all numeric arrays.  The
``action_mask`` sub-dict is preserved in ``info`` (not included in the flat
vector) so that masking-aware algorithms can still access it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pettingzoo.utils.wrappers import BaseWrapper


class FlattenObsWrapper(BaseWrapper):
    """
    Flatten the dict observation to a 1-D float32 ``Box``.

    Concatenation order:
      1. ``scalars``         – (N_SCALAR_FEATURES,)
      2. ``entities`` (flat) – (MAX_ENTITIES × ENTITY_FEATURES,)
      3. ``map_planes`` (flat) – (N_MAP_CHANNELS × tile_count,)
      4. ``visibility_mask`` – (MAX_ENTITIES,)

    The ``action_mask`` sub-dict is **not** included in the flat vector; it is
    passed through ``info["action_mask"]`` unchanged.
    """

    def observe(self, agent: str) -> np.ndarray:
        obs_dict = self.env.observe(agent)
        return self._flatten(obs_dict)

    def _flatten(self, obs_dict: dict[str, Any]) -> np.ndarray:
        scalars = obs_dict["scalars"].flatten()
        entities = obs_dict["entities"].flatten()
        map_planes = obs_dict["map_planes"].flatten()
        vis_mask = obs_dict["visibility_mask"].flatten().astype(np.float32)
        return np.concatenate([scalars, entities, map_planes, vis_mask])

    def reset(self, seed: int | None = None, options: dict | None = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        flat_obs = {agent: self._flatten(obs) for agent, obs in observations.items()}
        return flat_obs, infos

    def step(self, action) -> None:
        """Delegate to the wrapped AECEnv; returns None per PettingZoo AEC convention."""
        self.env.step(action)
