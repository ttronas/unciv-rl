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
    MAX_SUBACTIONS,
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
            (``macro``, ``unit_target``, ``unit_subaction``, ``city_target``,
            ``city_subaction``, ``tech_target``, ``diplomacy_target``,
            ``diplomacy_subaction``, ``policy_target``, ``settler_target``,
            ``production_items``), each being a ``MultiBinary`` mask over valid
            indices.
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

        The keys deliberately match those of :meth:`action_mask_space` so that
        ``action_space.sample(obs["action_mask"])`` works out of the box in any
        training framework.

        Keys
        ----
        macro:
            ``Discrete(N_MACRO_ACTIONS)`` – which top-level action category.
        unit_target:
            ``Discrete(MAX_ENTITIES)`` – which unit entity to act on.
        unit_subaction:
            ``Discrete(MAX_SUBACTIONS)`` – sub-action type for unit commands
            (move, attack, fortify, …).
        city_target:
            ``Discrete(MAX_ENTITIES)`` – which city entity to act on.
        city_subaction:
            ``Discrete(3)`` – city sub-action (set production / buy production
            / sell building).
        tech_target:
            ``Discrete(num_techs)`` – which technology to research.
        diplomacy_target:
            ``Discrete(num_agents)`` – which civilisation to interact with.
        diplomacy_subaction:
            ``Discrete(3)`` – diplomacy sub-action (declare war / offer peace /
            open borders).
        policy_target:
            ``Discrete(num_policies)`` – which social policy to adopt.
        settler_target:
            ``Discrete(MAX_ENTITIES)`` – which settler unit to use for city
            founding.
        production_items:
            ``Discrete(num_production_items)`` – which item to build in the
            selected city.
        """
        return spaces.Dict(
            {
                "macro": spaces.Discrete(N_MACRO_ACTIONS),
                "unit_target": spaces.Discrete(MAX_ENTITIES),
                "unit_subaction": spaces.Discrete(MAX_SUBACTIONS),
                "city_target": spaces.Discrete(MAX_ENTITIES),
                "city_subaction": spaces.Discrete(3),
                "tech_target": spaces.Discrete(self.num_techs),
                "diplomacy_target": spaces.Discrete(self.num_agents),
                "diplomacy_subaction": spaces.Discrete(3),
                "policy_target": spaces.Discrete(self.num_policies),
                "settler_target": spaces.Discrete(MAX_ENTITIES),
                "production_items": spaces.Discrete(self.num_production_items),
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
                "city_subaction": spaces.MultiBinary(3),
                "tech_target": spaces.MultiBinary(self.num_techs),
                "diplomacy_target": spaces.MultiBinary(self.num_agents),
                "diplomacy_subaction": spaces.MultiBinary(3),
                "policy_target": spaces.MultiBinary(self.num_policies),
                "settler_target": spaces.MultiBinary(MAX_ENTITIES),
                "production_items": spaces.MultiBinary(self.num_production_items),
            }
        )
