"""
Gymnasium observation and action space definitions for the Unciv RL environment.

Shapes are derived from the constants in :mod:`rl_env.constants`.  Clients that
need to override the defaults (e.g. to support a larger map or more entities)
should subclass :class:`UncivSpaces` and pass it to :class:`rl_env.UncivEnv`.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from rl_env.constants import (
    ENTITY_FEATURES,
    MAX_ENTITIES,
    MAX_TILES,
    N_MAP_CHANNELS,
    N_MACRO_ACTIONS,
    N_SCALAR_FEATURES,
    N_UNIT_SUBACTIONS,
    N_CITY_SUBACTIONS,
    N_DIPLOMACY_SUBACTIONS,
    MAX_TECHS,
    MAX_POLICIES,
    MAX_PROD_ITEMS,
    MAX_IMPROVEMENTS,
    MAX_AGENTS,
)


class UncivSpaces:
    """
    Container for all observation and action space objects.

    Parameters
    ----------
    num_agents:
        Number of RL-controlled agents in the game.
    num_techs:
        Number of technologies in the ruleset.
    num_policies:
        Number of policies in the ruleset.
    num_production_items:
        Number of buildable construction items in the ruleset.
    num_improvements:
        Number of tile improvements in the ruleset.
    """

    def __init__(
        self,
        num_agents: int = MAX_AGENTS,
        num_techs: int = MAX_TECHS,
        num_policies: int = MAX_POLICIES,
        num_production_items: int = MAX_PROD_ITEMS,
        num_improvements: int = MAX_IMPROVEMENTS,
    ) -> None:
        self.num_agents = num_agents
        self.num_techs = num_techs
        self.num_policies = num_policies
        self.num_production_items = num_production_items
        self.num_improvements = num_improvements

    # ------------------------------------------------------------------
    # Observation space
    # ------------------------------------------------------------------

    def observation_space(self) -> spaces.Dict:
        """
        Returns a :class:`gymnasium.spaces.Dict` describing one agent's observation.

        Keys
        ----
        scalars:
            ``Box(shape=(N_SCALAR_FEATURES,), dtype=float32)``
        entities:
            ``Box(shape=(MAX_ENTITIES, ENTITY_FEATURES), dtype=float32)``
        map_planes:
            ``Box(shape=(N_MAP_CHANNELS, MAX_TILES), dtype=float32)``
        visibility_mask:
            ``MultiBinary(MAX_ENTITIES)``
        action_mask:
            Nested ``Dict`` matching :meth:`action_mask_space`.
        """
        return spaces.Dict(
            {
                "scalars": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(N_SCALAR_FEATURES,),
                    dtype=np.float32,
                ),
                "entities": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(MAX_ENTITIES, ENTITY_FEATURES),
                    dtype=np.float32,
                ),
                "map_planes": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(N_MAP_CHANNELS, MAX_TILES),
                    dtype=np.float32,
                ),
                "visibility_mask": spaces.MultiBinary(MAX_ENTITIES),
                "action_mask": self.action_mask_space(),
            }
        )

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def action_space(self) -> spaces.Dict:
        """
        Returns a :class:`gymnasium.spaces.Dict` with 12 keys matching the
        semantic action representation.
        """
        return spaces.Dict(
            {
                "macro": spaces.Discrete(N_MACRO_ACTIONS),
                "unit_target": spaces.Discrete(MAX_ENTITIES),
                "unit_subaction": spaces.Discrete(N_UNIT_SUBACTIONS),
                "city_target": spaces.Discrete(MAX_ENTITIES),
                "city_subaction": spaces.Discrete(N_CITY_SUBACTIONS),
                "production_target": spaces.Discrete(self.num_production_items),
                "tech_target": spaces.Discrete(self.num_techs),
                "policy_target": spaces.Discrete(self.num_policies),
                "diplomacy_target": spaces.Discrete(self.num_agents),
                "diplomacy_subaction": spaces.Discrete(N_DIPLOMACY_SUBACTIONS),
                "tile_target": spaces.Discrete(MAX_TILES),
                "improvement_target": spaces.Discrete(self.num_improvements),
            }
        )

    def action_mask_space(self) -> spaces.Dict:
        """Returns a ``Dict`` of ``MultiBinary`` masks mirroring the action space."""
        return spaces.Dict(
            {
                "macro": spaces.MultiBinary(N_MACRO_ACTIONS),
                "unit_target": spaces.MultiBinary(MAX_ENTITIES),
                "unit_subaction": spaces.MultiBinary(N_UNIT_SUBACTIONS),
                "city_target": spaces.MultiBinary(MAX_ENTITIES),
                "city_subaction": spaces.MultiBinary(N_CITY_SUBACTIONS),
                "production_target": spaces.MultiBinary(self.num_production_items),
                "tech_target": spaces.MultiBinary(self.num_techs),
                "policy_target": spaces.MultiBinary(self.num_policies),
                "diplomacy_target": spaces.MultiBinary(self.num_agents),
                "diplomacy_subaction": spaces.MultiBinary(N_DIPLOMACY_SUBACTIONS),
                "tile_target": spaces.MultiBinary(MAX_TILES),
                "improvement_target": spaces.MultiBinary(self.num_improvements),
            }
        )
