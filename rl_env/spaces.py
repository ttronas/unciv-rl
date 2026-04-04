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
    MAX_ARG1,
    MAX_ARG2,
    MAX_SUBACTIONS,
    MAX_TARGETS,
    N_MAP_CHANNELS,
    N_MACRO_ACTIONS,
    N_SCALAR_FEATURES,
)


class UncivSpaces:
    """
    Container for all observation and action space objects.

    Parameters
    ----------
    num_agents:
        Number of RL-controlled agents in the game (needed for the action mask
        visibility_mask shape and diplomacy target mask).
    num_techs:
        Number of technologies in the ruleset (for action mask).
    num_policies:
        Number of policies in the ruleset (for action mask).
    num_production_items:
        Number of buildable construction items in the ruleset.
    """

    def __init__(
        self,
        num_agents: int = 4,
        num_techs: int = 100,
        num_policies: int = 80,
        num_production_items: int = 200,
    ) -> None:
        self.num_agents = num_agents
        self.num_techs = num_techs
        self.num_policies = num_policies
        self.num_production_items = num_production_items

    # ------------------------------------------------------------------
    # Observation space
    # ------------------------------------------------------------------

    def observation_space(self) -> spaces.Dict:
        """
        Returns a :class:`gymnasium.spaces.Dict` describing one agent's
        observation.

        Keys
        ----
        scalars:
            ``Box(shape=(N_SCALAR_FEATURES,), dtype=float32)`` – 14 per-civ
            scalar statistics.
        entities:
            ``Box(shape=(MAX_ENTITIES, ENTITY_FEATURES), dtype=float32)`` –
            padded entity list (units then cities).
        map_planes:
            ``Box(shape=(N_MAP_CHANNELS, tile_count), dtype=float32)`` where
            ``tile_count`` is the flat tile count.  May contain zeros if the
            server was not asked to include map planes.
        visibility_mask:
            ``MultiBinary(MAX_ENTITIES)`` – 1 for real entities, 0 for padding.
        action_mask:
            A nested ``Dict`` with the same sub-keys as the action space
            (``macro``, ``target``, ``subaction``, ``arg1``, ``arg2``),
            each being a ``MultiBinary`` mask over valid indices.
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
                    shape=(N_MAP_CHANNELS, MAX_ENTITIES),  # MAX_ENTITIES as tile_count upper bound
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
        Returns a :class:`gymnasium.spaces.Dict` describing a single step's
        action.

        Keys
        ----
        macro:
            ``Discrete(N_MACRO_ACTIONS)`` – which top-level action category.
        target:
            ``Discrete(MAX_TARGETS)`` – which entity / tech / civ is targeted.
        subaction:
            ``Discrete(MAX_SUBACTIONS)`` – sub-action type within the macro.
        arg1:
            ``Discrete(MAX_ARG1)`` – first argument (e.g. move destination,
            promotion index, production item index).
        arg2:
            ``Discrete(MAX_ARG2)`` – second argument (e.g. improvement type).
        """
        return spaces.Dict(
            {
                "macro": spaces.Discrete(N_MACRO_ACTIONS),
                "target": spaces.Discrete(MAX_TARGETS),
                "subaction": spaces.Discrete(MAX_SUBACTIONS),
                "arg1": spaces.Discrete(MAX_ARG1),
                "arg2": spaces.Discrete(MAX_ARG2),
            }
        )

    def action_mask_space(self) -> spaces.Dict:
        """
        Returns a ``Dict`` of ``MultiBinary`` masks that mirror the action space.
        """
        return spaces.Dict(
            {
                "macro": spaces.MultiBinary(N_MACRO_ACTIONS),
                "unit_target": spaces.MultiBinary(MAX_ENTITIES),
                "unit_subaction": spaces.MultiBinary(MAX_SUBACTIONS),
                "city_target": spaces.MultiBinary(MAX_ENTITIES),
                "tech_target": spaces.MultiBinary(self.num_techs),
                "diplomacy_target": spaces.MultiBinary(self.num_agents),
                "diplomacy_subaction": spaces.MultiBinary(3),
                "policy_target": spaces.MultiBinary(self.num_policies),
                "settler_target": spaces.MultiBinary(MAX_ENTITIES),
                "production_items": spaces.MultiBinary(self.num_production_items),
            }
        )
