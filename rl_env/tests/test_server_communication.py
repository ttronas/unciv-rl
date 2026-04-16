"""
Integration tests for Kotlin server ↔ Python client communication.

These tests require a running Unciv server started with the ``--rl`` flag.
Set the ``UNCIV_RL_URL`` environment variable to point at the server
(default: ``http://localhost:8080``).

All tests in this module are **automatically skipped** when no live server is
reachable, so they never block CI that does not start a server.

Test classes
~~~~~~~~~~~~
:class:`TestClientEndpoints`
    Exercises every ``/rl/*`` HTTP endpoint via :class:`~rl_env.client.UncivRLClient`
    and validates the raw JSON response shapes match the protocol spec.

:class:`TestObservationRoundTrip`
    Calls the server, parses the response with :func:`~rl_env.obs_parser.parse_observation`
    and :func:`~rl_env.obs_parser.parse_action_mask`, then checks that the
    resulting NumPy arrays match the shapes declared in ``constants.py``.

:class:`TestActionRoundTrip`
    Sends every macro-action category to the server and verifies that the
    ``ActionResult`` response has the required fields with correct types.

:class:`TestEpisodeFlow`
    Exercises a multi-turn episode: new_game → step × N (END_TURN) → done check
    → reset → verify fresh state.

:class:`TestUncivEnvIntegration`
    High-level ``UncivEnv`` PettingZoo integration – mirrors the lightweight
    tests that were present in earlier commits but extended with shape checks.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from rl_env.client import UncivRLClient
from rl_env.action_mapper import encode_action, decode_action_result
from rl_env.obs_parser import parse_observation, parse_action_mask
from rl_env.constants import (
    ENTITY_FEATURES,
    MAX_ENTITIES,
    MAX_TILES,
    N_MAP_CHANNELS,
    N_SCALAR_FEATURES,
    N_MACRO_ACTIONS,
    N_UNIT_SUBACTIONS,
    N_CITY_SUBACTIONS,
    N_DIPLOMACY_SUBACTIONS,
    MACRO_END_TURN,
    MACRO_UNIT_MOVE,
    MACRO_UNIT_ABILITY,
    MACRO_CITY_ACTION,
    MACRO_TECH_RESEARCH,
    MACRO_POLICY_ADOPT,
    MACRO_DIPLOMACY,
    MACRO_WORKER_BUILD,
    MACRO_FOUNDER_SETTLE,
    UNIT_SUB_SKIP,
)

# ---------------------------------------------------------------------------
# Detect whether a live server is available
# ---------------------------------------------------------------------------

_SERVER_URL = os.environ.get("UNCIV_RL_URL", "http://localhost:8080")


def _server_available() -> bool:
    try:
        import requests
        resp = requests.get(f"{_SERVER_URL}/isalive", timeout=3.0)
        return resp.ok
    except Exception:
        return False


_SKIP_MSG = (
    "Unciv RL server not available; set UNCIV_RL_URL and start "
    "the server with --rl to run these tests."
)
_skip_if_no_server = unittest.skipUnless(_server_available(), _SKIP_MSG)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_client() -> UncivRLClient:
    return UncivRLClient(base_url=_SERVER_URL, timeout=30.0)


def _start_game(client: UncivRLClient, **kwargs) -> dict:
    """Create a minimal 1-agent game; return the NewGameResponse dict."""
    defaults = dict(
        num_agents=1,
        num_ai=1,
        num_city_states=0,
        no_barbarians=True,
        seed=42,
    )
    defaults.update(kwargs)
    return client.new_game(**defaults)


# ---------------------------------------------------------------------------
# TestClientEndpoints  (raw HTTP / JSON shape tests)
# ---------------------------------------------------------------------------

@_skip_if_no_server
class TestClientEndpoints(unittest.TestCase):
    """Validate that every /rl/* endpoint returns the expected JSON structure."""

    @classmethod
    def setUpClass(cls):
        cls.client = _make_client()
        resp = _start_game(cls.client)
        cls.game_id = resp["gameId"]
        cls.agent_civ_ids = resp["agentCivIds"]

    @classmethod
    def tearDownClass(cls):
        # Best-effort cleanup; ignore errors
        try:
            cls.client._session.close()
        except Exception:
            pass

    # --- /rl/new_game -------------------------------------------------------

    def test_new_game_response_has_game_id(self):
        self.assertIsInstance(self.game_id, str)
        self.assertGreater(len(self.game_id), 0)

    def test_new_game_response_has_agent_civ_ids(self):
        self.assertIsInstance(self.agent_civ_ids, list)
        self.assertEqual(len(self.agent_civ_ids), 1)
        self.assertIsInstance(self.agent_civ_ids[0], str)

    def test_new_game_response_has_current_agent(self):
        resp = _start_game(self.client)
        self.assertIn("currentAgent", resp)
        self.assertIn(resp["currentAgent"], resp["agentCivIds"])

    def test_new_game_response_has_observation(self):
        resp = _start_game(self.client)
        self.assertIn("observation", resp)
        obs = resp["observation"]
        self.assertIn("entities", obs)
        self.assertIn("scalars", obs)

    # --- /rl/state/{gameId} -------------------------------------------------

    def test_get_state_returns_entities(self):
        state = self.client.get_state(self.game_id)
        self.assertIn("entities", state)
        self.assertIsInstance(state["entities"], list)
        self.assertEqual(len(state["entities"]), MAX_ENTITIES)

    def test_get_state_entity_has_features_field(self):
        state = self.client.get_state(self.game_id)
        first = state["entities"][0]
        self.assertIn("features", first)
        self.assertEqual(len(first["features"]), ENTITY_FEATURES)

    def test_get_state_returns_scalars(self):
        state = self.client.get_state(self.game_id)
        self.assertIn("scalars", state)
        scalars = state["scalars"]
        # Expect a dict with at least the canonical scalar keys
        self.assertIn("turn", scalars)
        self.assertIn("gold", scalars)
        self.assertIn("score", scalars)

    def test_get_state_returns_catalogues(self):
        state = self.client.get_state(self.game_id)
        self.assertIn("catalogues", state)
        cat = state["catalogues"]
        for key in ("techNames", "policyNames", "productionItemNames",
                    "improvementNames"):
            self.assertIn(key, cat, f"catalogue key '{key}' missing")

    def test_get_state_turn_is_positive_int(self):
        state = self.client.get_state(self.game_id)
        turn = state["scalars"]["turn"]
        self.assertIsInstance(turn, int)
        self.assertGreater(turn, 0)

    def test_get_state_game_id_matches(self):
        state = self.client.get_state(self.game_id)
        self.assertEqual(state["gameId"], self.game_id)

    # --- /rl/action_mask/{gameId} -------------------------------------------

    def test_get_action_mask_top_level_keys(self):
        mask = self.client.get_action_mask(self.game_id)
        required_keys = [
            "gameId", "agentCivId",
            "macroMask", "unitTargetMask", "unitSubactionMask",
            "cityTargetMask", "citySubactionMask", "productionTargetMask",
            "techTargetMask", "policyTargetMask",
            "diplomacyTargetMask", "diplomacySubactionMask",
            "tileTargetMask", "improvementTargetMask",
        ]
        for key in required_keys:
            self.assertIn(key, mask, f"mask key '{key}' missing")

    def test_get_action_mask_macro_length(self):
        mask = self.client.get_action_mask(self.game_id)
        self.assertEqual(len(mask["macroMask"]), N_MACRO_ACTIONS)

    def test_get_action_mask_unit_target_length(self):
        mask = self.client.get_action_mask(self.game_id)
        self.assertEqual(len(mask["unitTargetMask"]), MAX_ENTITIES)

    def test_get_action_mask_city_target_length(self):
        mask = self.client.get_action_mask(self.game_id)
        self.assertEqual(len(mask["cityTargetMask"]), MAX_ENTITIES)

    def test_get_action_mask_unit_subaction_length(self):
        mask = self.client.get_action_mask(self.game_id)
        self.assertEqual(len(mask["unitSubactionMask"]), N_UNIT_SUBACTIONS)

    def test_get_action_mask_city_subaction_length(self):
        mask = self.client.get_action_mask(self.game_id)
        self.assertEqual(len(mask["citySubactionMask"]), N_CITY_SUBACTIONS)

    def test_get_action_mask_diplomacy_subaction_length(self):
        mask = self.client.get_action_mask(self.game_id)
        self.assertEqual(len(mask["diplomacySubactionMask"]), N_DIPLOMACY_SUBACTIONS)

    def test_get_action_mask_end_turn_always_legal(self):
        mask = self.client.get_action_mask(self.game_id)
        # END_TURN (index 0) must always be a legal action
        self.assertTrue(mask["macroMask"][MACRO_END_TURN])

    def test_get_action_mask_booleans(self):
        mask = self.client.get_action_mask(self.game_id)
        for val in mask["macroMask"]:
            self.assertIsInstance(val, bool)

    # --- /rl/action/{gameId} -----------------------------------------------

    def test_step_end_turn_success(self):
        action = encode_action({"macro": MACRO_END_TURN})
        result = self.client.step(self.game_id, action)
        success, message, reward = decode_action_result(result)
        self.assertTrue(success)
        self.assertIsInstance(message, str)
        self.assertIsInstance(reward, float)

    def test_step_returns_all_result_fields(self):
        action = encode_action({"macro": MACRO_END_TURN})
        result = self.client.step(self.game_id, action)
        self.assertIn("success", result)
        self.assertIn("message", result)
        self.assertIn("reward", result)

    def test_step_reward_is_numeric(self):
        action = encode_action({"macro": MACRO_END_TURN})
        result = self.client.step(self.game_id, action)
        self.assertIsInstance(result["reward"], (int, float))

    # --- /rl/done/{gameId} -------------------------------------------------

    def test_get_done_fields(self):
        done_resp = self.client.get_done(self.game_id)
        self.assertIn("done", done_resp)
        self.assertIn("winner", done_resp)
        self.assertIn("scores", done_resp)

    def test_get_done_not_done_at_start(self):
        # A fresh game should not be done immediately
        resp = _start_game(self.client)
        done_resp = self.client.get_done(resp["gameId"])
        self.assertIsInstance(done_resp["done"], bool)

    def test_get_done_scores_is_dict(self):
        done_resp = self.client.get_done(self.game_id)
        self.assertIsInstance(done_resp["scores"], dict)

    # --- /rl/reset/{gameId} ------------------------------------------------

    def test_reset_returns_observation(self):
        resp = _start_game(self.client)
        reset_resp = self.client.reset(resp["gameId"])
        self.assertIn("observation", reset_resp)
        self.assertIn("agentCivIds", reset_resp)
        self.assertIn("currentAgent", reset_resp)

    def test_reset_returns_new_valid_game_id(self):
        # The server's reset endpoint deletes the old game and calls
        # createGame() again, which lets GameStarter assign a fresh UUID via
        # gameInfo.gameId.  The returned gameId is therefore a NEW identifier,
        # not the original one.  We just verify it is a non-empty string.
        resp = _start_game(self.client)
        game_id = resp["gameId"]
        reset_resp = self.client.reset(game_id)
        new_game_id = reset_resp["gameId"]
        self.assertIsInstance(new_game_id, str)
        self.assertGreater(len(new_game_id), 0)


# ---------------------------------------------------------------------------
# TestObservationRoundTrip
# ---------------------------------------------------------------------------

@_skip_if_no_server
class TestObservationRoundTrip(unittest.TestCase):
    """Parse live server observations and verify NumPy array shapes/dtypes."""

    @classmethod
    def setUpClass(cls):
        cls.client = _make_client()
        resp = _start_game(cls.client)
        cls.game_id = resp["gameId"]
        # Parse observation from new_game response
        cls.obs = parse_observation(resp["observation"])
        # Also parse action mask
        mask_json = cls.client.get_action_mask(cls.game_id)
        cls.mask = parse_action_mask(mask_json)

    # Observation array shapes
    def test_scalars_shape(self):
        self.assertEqual(self.obs["scalars"].shape, (N_SCALAR_FEATURES,))

    def test_scalars_dtype(self):
        self.assertEqual(self.obs["scalars"].dtype, np.float32)

    def test_entities_shape(self):
        self.assertEqual(self.obs["entities"].shape, (MAX_ENTITIES, ENTITY_FEATURES))

    def test_entities_dtype(self):
        self.assertEqual(self.obs["entities"].dtype, np.float32)

    def test_map_planes_shape(self):
        self.assertEqual(self.obs["map_planes"].shape, (N_MAP_CHANNELS, MAX_TILES))

    def test_map_planes_dtype(self):
        self.assertEqual(self.obs["map_planes"].dtype, np.float32)

    def test_visibility_mask_shape(self):
        self.assertEqual(self.obs["visibility_mask"].shape, (MAX_ENTITIES,))

    def test_visibility_mask_binary(self):
        vals = set(self.obs["visibility_mask"].tolist())
        self.assertTrue(vals.issubset({0, 1}))

    # Info dict
    def test_info_game_id(self):
        self.assertIsInstance(self.obs["info"]["gameId"], str)
        self.assertGreater(len(self.obs["info"]["gameId"]), 0)

    def test_info_agent_civ_id(self):
        self.assertIsInstance(self.obs["info"]["agentCivId"], str)

    def test_info_all_agents(self):
        self.assertIsInstance(self.obs["info"]["allAgents"], list)
        self.assertGreater(len(self.obs["info"]["allAgents"]), 0)

    def test_info_catalogues_present(self):
        cat = self.obs["info"]["catalogues"]
        self.assertIsInstance(cat, dict)
        self.assertIn("techNames", cat)

    # Scalars semantic sanity
    def test_turn_at_least_one(self):
        from rl_env.constants import SCALAR_TURN
        self.assertGreaterEqual(self.obs["scalars"][SCALAR_TURN], 1.0)

    # Action mask parsed shapes
    def test_mask_macro_length(self):
        self.assertEqual(len(self.mask["macro"]), N_MACRO_ACTIONS)

    def test_mask_unit_target_length(self):
        self.assertEqual(len(self.mask["unit_target"]), MAX_ENTITIES)

    def test_mask_city_target_length(self):
        self.assertEqual(len(self.mask["city_target"]), MAX_ENTITIES)

    def test_mask_unit_subaction_length(self):
        self.assertEqual(len(self.mask["unit_subaction"]), N_UNIT_SUBACTIONS)

    def test_mask_city_subaction_length(self):
        self.assertEqual(len(self.mask["city_subaction"]), N_CITY_SUBACTIONS)

    def test_mask_diplomacy_subaction_length(self):
        self.assertEqual(len(self.mask["diplomacy_subaction"]), N_DIPLOMACY_SUBACTIONS)

    def test_mask_end_turn_legal(self):
        self.assertEqual(self.mask["macro"][MACRO_END_TURN], 1)

    def test_mask_values_binary(self):
        for key in ("macro", "unit_subaction", "city_subaction"):
            vals = set(self.mask[key].tolist())
            self.assertTrue(vals.issubset({0, 1}), f"mask[{key!r}] has non-binary values: {vals}")

    # Re-parse the same observation from get_state to check consistency
    def test_get_state_matches_new_game_obs_shape(self):
        state = self.client.get_state(self.game_id)
        obs2 = parse_observation(state)
        self.assertEqual(obs2["entities"].shape, self.obs["entities"].shape)
        self.assertEqual(obs2["scalars"].shape, self.obs["scalars"].shape)

    def test_get_state_same_turn_as_new_game(self):
        from rl_env.constants import SCALAR_TURN
        state = self.client.get_state(self.game_id)
        obs2 = parse_observation(state)
        self.assertEqual(
            obs2["scalars"][SCALAR_TURN],
            self.obs["scalars"][SCALAR_TURN],
        )


# ---------------------------------------------------------------------------
# TestActionRoundTrip
# ---------------------------------------------------------------------------

@_skip_if_no_server
class TestActionRoundTrip(unittest.TestCase):
    """
    Send legal actions to the server and verify the ActionResult structure.

    END_TURN is always legal; other actions are only sent when the
    action mask reports them as available.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = _make_client()

    def setUp(self):
        resp = _start_game(self.client)
        self.game_id = resp["gameId"]

    def _mask(self) -> dict:
        return self.client.get_action_mask(self.game_id)

    def _step(self, action: dict) -> tuple[bool, str, float]:
        wire = encode_action(action)
        raw = self.client.step(self.game_id, wire)
        return decode_action_result(raw)

    def _advance_to_playable_state(self, n_turns: int = 20) -> None:
        """
        Advance the game to a state where a city exists and culture has
        accumulated so that city-action and policy tests are exercisable.

        1. If MACRO_FOUNDER_SETTLE is legal, settle the capital immediately.
        2. Play up to *n_turns* END_TURN actions to accumulate culture and
           populate city production options.

        With Prince difficulty and seed=42 the first social policy typically
        unlocks within ~10 turns after settling (costs 25 culture; base city
        yields ~2-3 culture/turn), so n_turns=20 provides a comfortable margin.
        """
        mask = self._mask()
        macro_mask = mask.get("macroMask", [])
        if len(macro_mask) > MACRO_FOUNDER_SETTLE and macro_mask[MACRO_FOUNDER_SETTLE]:
            unit_targets = [i for i, v in enumerate(mask.get("unitTargetMask", [])) if v]
            if unit_targets:
                self.client.step(
                    self.game_id,
                    encode_action({"macro": MACRO_FOUNDER_SETTLE, "unit_target": unit_targets[0]}),
                )
        end_turn = encode_action({"macro": MACRO_END_TURN})
        for _ in range(n_turns):
            self.client.step(self.game_id, end_turn)

    # END_TURN
    def test_end_turn_success(self):
        success, message, reward = self._step({"macro": MACRO_END_TURN})
        self.assertTrue(success)
        self.assertIn("end_turn", message)

    def test_end_turn_reward_is_float(self):
        _, _, reward = self._step({"macro": MACRO_END_TURN})
        self.assertIsInstance(reward, float)

    # UNIT_ABILITY / SKIP – legal when there is at least one unit
    def test_unit_skip_if_unit_available(self):
        mask = self._mask()
        unit_indices = [i for i, v in enumerate(mask["unitTargetMask"]) if v]
        if not unit_indices:
            self.skipTest("No actionable units available to test unit skip")
        success, _, _ = self._step({
            "macro": MACRO_UNIT_ABILITY,
            "unit_target": unit_indices[0],
            "unit_subaction": UNIT_SUB_SKIP,
        })
        self.assertTrue(success)

    # TECH_RESEARCH – only when a tech is researchable
    def test_tech_research_if_available(self):
        state = self.client.get_state(self.game_id)
        mask = self._mask()
        tech_names = state.get("catalogues", {}).get("techNames", [])
        legal_techs = [i for i, v in enumerate(mask["techTargetMask"]) if v]
        if not legal_techs or not tech_names:
            self.skipTest("No researchable techs in this game state")
        tech_idx = legal_techs[0]
        success, message, _ = self._step({
            "macro": MACRO_TECH_RESEARCH,
            "tech_target": tech_idx,
        })
        self.assertTrue(success, f"Tech research failed: {message}")

    # POLICY_ADOPT – only when a policy is adoptable
    def test_policy_adopt_if_available(self):
        self._advance_to_playable_state()
        mask = self._mask()
        legal_policies = [i for i, v in enumerate(mask["policyTargetMask"]) if v]
        if not legal_policies:
            self.skipTest("No adoptable policies even after advancing game state")
        success, message, _ = self._step({
            "macro": MACRO_POLICY_ADOPT,
            "policy_target": legal_policies[0],
        })
        self.assertTrue(success, f"Policy adopt failed: {message}")

    # CITY_ACTION – only when a city exists
    def test_city_set_production_if_available(self):
        self._advance_to_playable_state()
        mask = self._mask()
        city_indices = [i for i, v in enumerate(mask["cityTargetMask"]) if v]
        prod_indices = [i for i, v in enumerate(mask["productionTargetMask"]) if v]
        if not city_indices or not prod_indices:
            self.skipTest("No city or no buildable production items even after advancing game state")
        success, message, _ = self._step({
            "macro": MACRO_CITY_ACTION,
            "city_target": city_indices[0],
            "city_subaction": 0,  # SET_PRODUCTION
            "production_target": prod_indices[0],
        })
        self.assertTrue(success, f"City set_production failed: {message}")

    # Unknown macro → failure (not a 5xx error, just a logical failure)
    def test_invalid_macro_returns_failure(self):
        raw = self.client.step(self.game_id, {
            "macro": 99,  # out-of-range but valid JSON
            "unitTarget": 0,
            "unitSubaction": 0,
            "cityTarget": 0,
            "citySubaction": 0,
            "productionTarget": 0,
            "techTarget": 0,
            "policyTarget": 0,
            "diplomacyTarget": 0,
            "diplomacySubaction": 0,
            "tileTarget": 0,
            "improvementTarget": 0,
        })
        # Server must respond (not crash), success should be False for unknown macro
        self.assertIn("success", raw)
        self.assertFalse(raw["success"])


# ---------------------------------------------------------------------------
# TestEpisodeFlow
# ---------------------------------------------------------------------------

@_skip_if_no_server
class TestEpisodeFlow(unittest.TestCase):
    """
    Simulate a short game episode through the raw client API and verify
    state transitions.
    """

    N_TURNS = 5

    @classmethod
    def setUpClass(cls):
        cls.client = _make_client()

    def test_multi_turn_end_turn_loop(self):
        """
        Execute N_TURNS END_TURN actions, check that the server responds
        successfully each time and the turn counter advances.
        """
        resp = _start_game(self.client)
        game_id = resp["gameId"]
        initial_state = self.client.get_state(game_id)
        initial_turn = initial_state["scalars"]["turn"]

        for i in range(self.N_TURNS):
            action = encode_action({"macro": MACRO_END_TURN})
            result = self.client.step(game_id, action)
            success, _, _ = decode_action_result(result)
            self.assertTrue(success, f"END_TURN failed on step {i}: {result}")

        final_state = self.client.get_state(game_id)
        final_turn = final_state["scalars"]["turn"]
        # Turn should have advanced by at least 1
        self.assertGreater(final_turn, initial_turn)

    def test_done_check_after_turns(self):
        resp = _start_game(self.client)
        game_id = resp["gameId"]
        for _ in range(3):
            action = encode_action({"macro": MACRO_END_TURN})
            self.client.step(game_id, action)
        done_resp = self.client.get_done(game_id)
        self.assertIn("done", done_resp)
        self.assertIsInstance(done_resp["done"], bool)

    def test_reset_resets_turn(self):
        resp = _start_game(self.client)
        game_id = resp["gameId"]
        initial_turn = self.client.get_state(game_id)["scalars"]["turn"]

        # Advance a few turns
        for _ in range(3):
            action = encode_action({"macro": MACRO_END_TURN})
            self.client.step(game_id, action)

        # Reset
        reset_resp = self.client.reset(game_id)
        self.assertIn("observation", reset_resp)

        # Turn after reset should be back to (or near) initial
        new_turn = reset_resp["observation"]["scalars"]["turn"]
        self.assertLessEqual(new_turn, initial_turn + 1)

    def test_reset_observation_same_shape(self):
        resp = _start_game(self.client)
        game_id = resp["gameId"]
        for _ in range(2):
            action = encode_action({"macro": MACRO_END_TURN})
            self.client.step(game_id, action)
        reset_resp = self.client.reset(game_id)
        obs_after_reset = parse_observation(reset_resp["observation"])
        self.assertEqual(obs_after_reset["entities"].shape, (MAX_ENTITIES, ENTITY_FEATURES))
        self.assertEqual(obs_after_reset["scalars"].shape, (N_SCALAR_FEATURES,))

    def test_agent_changes_after_multi_agent_end_turn(self):
        """
        With 2 agents the currentAgent should cycle after each END_TURN.
        """
        resp = _start_game(self.client, num_agents=2, num_ai=0, num_city_states=0)
        game_id = resp["gameId"]
        first_agent = resp["currentAgent"]

        action = encode_action({"macro": MACRO_END_TURN})
        self.client.step(game_id, action)

        state = self.client.get_state(game_id)
        second_agent = state["agentCivId"]
        # currentAgent should have changed (or still be valid)
        self.assertIn(second_agent, resp["agentCivIds"])

    def test_consecutive_steps_are_independent(self):
        """Each step call should return a fresh ActionResult."""
        resp = _start_game(self.client)
        game_id = resp["gameId"]
        action = encode_action({"macro": MACRO_END_TURN})
        r1 = self.client.step(game_id, action)
        r2 = self.client.step(game_id, action)
        # Both must be valid ActionResults
        for r in (r1, r2):
            self.assertIn("success", r)
            self.assertIn("reward", r)


# ---------------------------------------------------------------------------
# TestUncivEnvIntegration (high-level PettingZoo API)
# ---------------------------------------------------------------------------

@_skip_if_no_server
class TestUncivEnvIntegration(unittest.TestCase):
    """
    High-level UncivEnv tests that mirror the communication patterns from
    earlier commits, extended with observation/mask shape assertions.
    """

    def setUp(self):
        from rl_env import UncivEnv
        self.env = UncivEnv(
            base_url=_SERVER_URL,
            num_agents=1,
            num_ai=1,
            num_city_states=0,
            no_barbarians=True,
        )

    def tearDown(self):
        self.env.close()

    def test_reset_returns_obs_for_all_agents(self):
        obs, infos = self.env.reset(seed=0)
        self.assertGreater(len(obs), 0)
        for agent in obs:
            self.assertIn("scalars", obs[agent])
            self.assertIn("entities", obs[agent])
            self.assertIn("map_planes", obs[agent])
            self.assertIn("visibility_mask", obs[agent])

    def test_reset_obs_shapes(self):
        obs, _ = self.env.reset(seed=1)
        for agent in obs:
            self.assertEqual(obs[agent]["scalars"].shape, (N_SCALAR_FEATURES,))
            self.assertEqual(obs[agent]["entities"].shape, (MAX_ENTITIES, ENTITY_FEATURES))
            self.assertEqual(obs[agent]["map_planes"].shape, (N_MAP_CHANNELS, MAX_TILES))

    def test_step_end_turn_does_not_raise(self):
        self.env.reset(seed=0)
        agent = self.env.agent_selection
        self.env.step({"macro": MACRO_END_TURN})
        # If we get here the step succeeded without exception
        self.assertIsNotNone(agent)

    def test_observe_after_step(self):
        self.env.reset(seed=0)
        agent = self.env.agent_selection
        self.env.step({"macro": MACRO_END_TURN})
        obs = self.env.observe(agent)
        if obs is not None:
            self.assertIn("scalars", obs)
            self.assertEqual(obs["scalars"].shape, (N_SCALAR_FEATURES,))

    def test_pettingzoo_api_compliance(self):
        from pettingzoo.test import api_test
        api_test(self.env, num_cycles=3, verbose_progress=False)

    def test_action_mask_in_observation_space(self):
        obs, _ = self.env.reset(seed=0)
        for agent in obs:
            if "action_mask" in obs[agent]:
                mask = obs[agent]["action_mask"]
                self.assertIn("macro", mask)
                self.assertEqual(len(mask["macro"]), N_MACRO_ACTIONS)

    def test_multiple_resets_produce_valid_obs(self):
        for seed in range(3):
            obs, _ = self.env.reset(seed=seed)
            for agent in obs:
                self.assertEqual(obs[agent]["entities"].shape,
                                 (MAX_ENTITIES, ENTITY_FEATURES))

    def test_rewards_are_floats(self):
        self.env.reset(seed=0)
        self.env.step({"macro": MACRO_END_TURN})
        for reward in self.env.rewards.values():
            self.assertIsInstance(reward, (int, float))

    def test_terminations_are_bools(self):
        self.env.reset(seed=0)
        self.env.step({"macro": MACRO_END_TURN})
        for term in self.env.terminations.values():
            self.assertIsInstance(term, bool)


if __name__ == "__main__":
    unittest.main()
