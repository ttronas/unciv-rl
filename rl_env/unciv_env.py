"""
PettingZoo AEC environment for the Unciv Kotlin game engine.

The environment wraps the Unciv RL server's REST API, exposing a standard
:class:`pettingzoo.utils.env.AECEnv` interface that is compatible with
RLlib, Stable-Baselines3 (via :class:`pettingzoo.utils.wrappers.BaseWrapper`),
and CleanRL.

Usage
-----
.. code-block:: python

    from rl_env import UncivEnv

    env = UncivEnv(base_url="http://localhost:8080", num_agents=2, num_ai=2)
    observations, infos = env.reset(seed=42)

    while env.agents:
        agent = env.agent_selection
        obs = observations[agent]
        action_mask = infos[agent]["action_mask"]
        # ... choose action ...
        env.step(action)
        observations[agent] = env.observe(agent)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector

from rl_env.action_mapper import decode_action_result, encode_action
from rl_env.client import UncivRLClient
from rl_env.constants import (
    ENTITY_FEATURES,
    MACRO_END_TURN,
    MAX_ENTITIES,
    N_MAP_CHANNELS,
    N_SCALAR_FEATURES,
)
from rl_env.obs_parser import parse_action_mask, parse_observation
from rl_env.spaces import UncivSpaces


class UncivEnv(AECEnv):
    """
    PettingZoo AEC environment wrapping the Unciv RL server.

    Parameters
    ----------
    base_url:
        Base URL of the running Unciv server, e.g. ``"http://localhost:8080"``.
    num_agents:
        Number of RL-controlled civilisations.
    num_ai:
        Number of AI-controlled civilisations.
    num_city_states:
        Number of city-state civs.
    difficulty:
        Unciv difficulty level (e.g. ``"Prince"``).
    include_map_planes:
        Whether to include the spatial map planes in observations.  Disabling
        this reduces observation size significantly.
    no_barbarians:
        Disable barbarian units for cleaner RL episodes.
    timeout:
        Per-request HTTP timeout in seconds.
    render_mode:
        ``"human"`` prints a one-line state summary to stdout; ``None``
        (default) renders nothing.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "unciv_rl_v0",
        "is_parallelizable": False,
    }

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        num_agents: int = 1,
        num_ai: int = 3,
        num_city_states: int = 6,
        difficulty: str = "Prince",
        include_map_planes: bool = True,
        no_barbarians: bool = True,
        timeout: float = 60.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self._client = UncivRLClient(base_url=base_url, timeout=timeout)
        self._new_game_kwargs = dict(
            num_agents=num_agents,
            num_ai=num_ai,
            num_city_states=num_city_states,
            difficulty=difficulty,
            include_map_planes=include_map_planes,
            no_barbarians=no_barbarians,
        )
        self.render_mode = render_mode

        # These will be set in reset() once the server creates a game
        self._game_id: str | None = None
        self._agent_civ_ids: list[str] = []
        self._spaces: UncivSpaces | None = None

        # PettingZoo required attributes
        self.possible_agents: list[str] = []
        self.agents: list[str] = []
        self.agent_selection: str = ""

        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict] = {}
        self._cumulative_rewards: dict[str, float] = {}

        self._last_obs: dict[str, dict] = {}
        self._agent_selector: AgentSelector | None = None
        self._obs_space_cache: dict[str, Any] = {}
        self._action_space_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Spaces (built lazily after reset())
    # ------------------------------------------------------------------

    @property
    def observation_spaces(self) -> dict[str, Any]:
        self._require_reset()
        return dict(self._obs_space_cache)

    @property
    def action_spaces(self) -> dict[str, Any]:
        self._require_reset()
        return dict(self._action_space_cache)

    def observation_space(self, agent: str) -> Any:
        self._require_reset()
        return self._obs_space_cache[agent]

    def action_space(self, agent: str) -> Any:
        self._require_reset()
        return self._action_space_cache[agent]

    # ------------------------------------------------------------------
    # PettingZoo core API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """
        Start a new game and return initial observations and infos.

        Parameters
        ----------
        seed:
            Optional random seed forwarded to the Kotlin server.
        options:
            Ignored; reserved for future use.

        Returns
        -------
        observations : dict[agent → obs_dict]
        infos : dict[agent → info_dict]
        """
        kwargs = dict(self._new_game_kwargs)
        if seed is not None:
            kwargs["seed"] = seed

        response = self._client.new_game(**kwargs)
        self._game_id = response["gameId"]
        self._agent_civ_ids = response["agentCivIds"]

        # Agent names are the civ ID strings
        self.possible_agents = list(self._agent_civ_ids)
        self.agents = list(self.possible_agents)

        # Build spaces once we know the catalogue sizes
        catalogues = response["observation"].get("catalogues", {})
        self._spaces = UncivSpaces(
            num_agents=len(self.possible_agents),
            num_techs=max(len(catalogues.get("techNames", [])), 1),
            num_policies=max(len(catalogues.get("policyNames", [])), 1),
            num_production_items=max(len(catalogues.get("productionItemNames", [])), 1),
            num_improvements=max(len(catalogues.get("improvementNames", [])), 1),
        )

        self._obs_space_cache = {a: self._spaces.observation_space() for a in self.possible_agents}
        self._action_space_cache = {a: self._spaces.action_space() for a in self.possible_agents}

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = response["currentAgent"]

        # Fetch initial observations and masks for all agents
        obs_json = response["observation"]
        mask_json = self._client.get_action_mask(self._game_id)
        observations: dict[str, dict] = {}
        infos: dict[str, dict] = {}

        for agent in self.agents:
            parsed = parse_observation(obs_json)
            action_mask = parse_action_mask(mask_json)
            obs_dict = {
                "scalars": parsed["scalars"],
                "entities": parsed["entities"],
                "map_planes": parsed["map_planes"],
                "visibility_mask": parsed["visibility_mask"],
                "action_mask": action_mask,
            }
            observations[agent] = obs_dict
            infos[agent] = {
                **parsed["info"],
                "action_mask": action_mask,
            }

        self._last_obs = observations
        self.infos = infos
        return observations, infos

    def step(self, action: dict[str, int] | np.ndarray | None) -> None:
        """
        Execute *action* for the current agent and advance to the next agent.

        If *action* is ``None`` (agent is terminated/truncated), the turn is
        ended immediately with a no-op.

        Side effects
        ------------
        * Updates ``self.agent_selection`` to the next active agent.
        * Updates ``self.rewards``, ``self.terminations``, ``self.truncations``,
          and ``self.infos`` for all agents.
        """
        if self._game_id is None:
            raise RuntimeError("Call reset() before step().")

        current_agent = self.agent_selection

        if self.terminations[current_agent] or self.truncations[current_agent]:
            # Skip the agent's turn cleanly
            self._was_dead_step(action)
            return

        # Encode and send the action
        if action is None:
            wire_action = encode_action({"macro": MACRO_END_TURN})
        else:
            wire_action = encode_action(action)

        result_json = self._client.step(self._game_id, wire_action)
        _success, _message, reward = decode_action_result(result_json)

        # Assign reward to the current agent
        self.rewards[current_agent] = reward

        # Check game termination
        done_json = self._client.get_done(self._game_id)
        done = done_json.get("done", False)
        winner = done_json.get("winner")

        if done:
            for agent in self.agents:
                self.terminations[agent] = True
            self.agents = []
        else:
            # Advance agent selection (the server already moved to next agent after END_TURN)
            if wire_action.get("macro") == MACRO_END_TURN:
                state_json = self._client.get_state(self._game_id)
                next_agent = state_json.get("agentCivId", current_agent)
                self.agent_selection = next_agent
            # Non-turn-ending actions keep the same agent active

        # Update cumulative rewards
        for agent in self.possible_agents:
            self._cumulative_rewards[agent] = self._cumulative_rewards.get(agent, 0.0) + self.rewards.get(agent, 0.0)
            self.rewards[agent] = 0.0

        # Update infos for the current agent
        if not done and self.agents:
            obs_json = self._client.get_state(self._game_id)
            mask_json = self._client.get_action_mask(self._game_id)
            parsed = parse_observation(obs_json)
            action_mask = parse_action_mask(mask_json)
            self.infos[self.agent_selection] = {
                **parsed["info"],
                "action_mask": action_mask,
                "winner": winner,
            }
            self._last_obs[self.agent_selection] = {
                "scalars": parsed["scalars"],
                "entities": parsed["entities"],
                "map_planes": parsed["map_planes"],
                "visibility_mask": parsed["visibility_mask"],
                "action_mask": action_mask,
            }
        else:
            # Game over: update info for all agents
            for agent in self.possible_agents:
                self.infos[agent] = {"done": True, "winner": winner}

        if self.render_mode == "human":
            self.render()

    def observe(self, agent: str) -> dict:
        """
        Return the latest observation for *agent*.

        This method fetches a fresh observation from the server when *agent* is
        the current player; otherwise it returns the cached last observation.
        """
        if self._game_id is None:
            raise RuntimeError("Call reset() before observe().")
        if agent == self.agent_selection:
            obs_json = self._client.get_state(self._game_id)
            parsed = parse_observation(obs_json)
            mask_json = self._client.get_action_mask(self._game_id)
            action_mask = parse_action_mask(mask_json)
            obs = {
                "scalars": parsed["scalars"],
                "entities": parsed["entities"],
                "map_planes": parsed["map_planes"],
                "visibility_mask": parsed["visibility_mask"],
                "action_mask": action_mask,
            }
            self._last_obs[agent] = obs
            return obs
        # Return cached observation for non-active agents
        return self._last_obs.get(
            agent,
            {
                "scalars": np.zeros(N_SCALAR_FEATURES, dtype=np.float32),
                "entities": np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32),
                "map_planes": np.zeros((N_MAP_CHANNELS, MAX_ENTITIES), dtype=np.float32),
                "visibility_mask": np.zeros(MAX_ENTITIES, dtype=np.int8),
                "action_mask": {},
            },
        )

    def action_mask(self, agent: str) -> dict[str, np.ndarray]:
        """Return the current legal action mask for *agent*."""
        if self._game_id is None:
            raise RuntimeError("Call reset() before action_mask().")
        mask_json = self._client.get_action_mask(self._game_id)
        return parse_action_mask(mask_json)

    def render(self) -> str | None:
        """Render the environment (``"human"`` mode prints a short summary)."""
        if self.render_mode != "human" or self._game_id is None:
            return None
        done_json = self._client.get_done(self._game_id)
        scores = done_json.get("scores", {})
        score_str = ", ".join(f"{k}:{v}" for k, v in scores.items())
        summary = (
            f"[UncivEnv] agent={self.agent_selection}  "
            f"done={done_json.get('done', False)}  scores={{{score_str}}}"
        )
        print(summary)
        return summary

    def close(self) -> None:
        """Release resources (no persistent connection to close)."""
        self._game_id = None
        self.agents = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_reset(self) -> None:
        if self._game_id is None:
            raise RuntimeError(
                "UncivEnv has not been reset yet.  Call env.reset() first."
            )

    def _was_dead_step(self, action: Any) -> None:
        """Handle a step call when the current agent is already terminated."""
        # PettingZoo convention: remove the agent from the active list
        agent = self.agent_selection
        if agent in self.agents:
            self.agents.remove(agent)
        if self.agents:
            self._agent_selector = AgentSelector(self.agents)
            self.agent_selection = self._agent_selector.next()
        else:
            self.agent_selection = ""
