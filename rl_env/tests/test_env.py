"""
Tests for the Unciv RL environment.

These tests are split into two categories:

Unit tests (no server required)
    - :class:`TestObsParser` – parse_observation / parse_action_mask
    - :class:`TestActionMapper` – encode_action / describe_action
    - :class:`TestSpaces` – Gymnasium space definitions

Integration tests (require a running Unciv server with --rl flag)
    Skipped automatically when the server is not reachable.
    Set env var UNCIV_RL_URL to override the default http://localhost:8080.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Unit tests – no server needed
# ---------------------------------------------------------------------------

from rl_env.constants import (
    ENTITY_FEATURES,
    MAX_ENTITIES,
    MAX_SUBACTIONS,
    N_MAP_CHANNELS,
    N_SCALAR_FEATURES,
    MACRO_END_TURN,
    MACRO_UNIT_ACTION,
    UNIT_SUBACTION_MOVE,
)
from rl_env.action_mapper import decode_action_result, describe_action, encode_action
from rl_env.obs_parser import parse_action_mask, parse_observation
from rl_env.spaces import UncivSpaces


# ---------------------------------------------------------------------------
# Helpers: minimal fake server responses
# ---------------------------------------------------------------------------

def _make_scalars_json() -> dict:
    return {
        "turn": 5,
        "currentPlayerId": 0,
        "gold": 120,
        "sciencePerTurn": 8.5,
        "culturePerTurn": 3.2,
        "faithPerTurn": 1.0,
        "happiness": 4,
        "netGoldPerTurn": 2.5,
        "cityCount": 2,
        "totalPopulation": 10,
        "techProgressCurrent": 0.42,
        "policyProgressCurrent": 0.1,
        "warStateFlags": 0,
        "victoryProgress": 0.05,
        "isInGoldenAge": 0,
        "goldenAgeTurnsRemaining": 0,
        "eraIndex": 1,
        "freePolicies": 0,
        "currentTechIndex": 3,
    }


def _make_entity_json(entity_type: int = 2) -> dict:
    """Return a minimal entity dict with the given entityType at index 0."""
    features = [entity_type] + [0] * (ENTITY_FEATURES - 1)
    return {"features": features}


def _make_obs_json(num_entities: int = 3) -> dict:
    entities = [_make_entity_json(2) for _ in range(num_entities)]
    # Pad to MAX_ENTITIES
    entities += [{"features": [0] * ENTITY_FEATURES}] * (MAX_ENTITIES - num_entities)
    return {
        "gameId": "test-game-id",
        "agentCivId": "Rome",
        "agentIndex": 0,
        "allAgents": ["Rome", "Greece"],
        "scalars": _make_scalars_json(),
        "entities": entities,
        "mapPlanes": {
            "channels": N_MAP_CHANNELS,
            "height": 10,
            "width": 1,
            "data": list(range(N_MAP_CHANNELS * 10)),
        },
        "fogOfWar": [],
        "catalogues": {
            "terrainTypes": ["Plains", "Grassland", "Desert"],
            "featureTypes": ["", "Forest", "Jungle"],
            "resourceTypes": ["", "Iron", "Gold"],
            "improvementTypes": ["", "Farm", "Mine"],
            "techNames": ["Agriculture", "Mining", "Writing"],
            "policyNames": ["Liberty", "Tradition"],
            "productionItemNames": ["Monument", "Warrior"],
        },
        "done": False,
        "winner": None,
        "scores": {"Rome": 100, "Greece": 80},
    }


def _make_mask_json(num_agents: int = 2) -> dict:
    return {
        "gameId": "test-game-id",
        "agentCivId": "Rome",
        "macroMask": [True, True, False, True, False, False, False, False],
        "unitTargetMask": ([True] * 3 + [False] * (MAX_ENTITIES - 3)),
        "unitSubactionMask": [True, False, True, True, False, False, False, False, False],
        "cityTargetMask": ([False] * MAX_ENTITIES),
        "citySubactionMask": [True, False, False],
        "techTargetMask": [True, False, True],
        "diplomacyTargetMask": [False] * num_agents,
        "diplomacySubactionMask": [False, False, False],
        "policyTargetMask": [True, False],
        "settlerTargetMask": ([False] * MAX_ENTITIES),
        "productionItemMask": [True, False],
    }


# ---------------------------------------------------------------------------
# TestObsParser
# ---------------------------------------------------------------------------

class TestObsParser(unittest.TestCase):

    def test_parse_observation_scalars_shape(self):
        obs = parse_observation(_make_obs_json())
        self.assertEqual(obs["scalars"].shape, (N_SCALAR_FEATURES,))
        self.assertEqual(obs["scalars"].dtype, np.float32)

    def test_parse_observation_scalars_values(self):
        obs = parse_observation(_make_obs_json())
        self.assertAlmostEqual(float(obs["scalars"][0]), 5.0)   # turn
        self.assertAlmostEqual(float(obs["scalars"][2]), 120.0) # gold

    def test_parse_observation_entities_shape(self):
        obs = parse_observation(_make_obs_json(num_entities=5))
        self.assertEqual(obs["entities"].shape, (MAX_ENTITIES, ENTITY_FEATURES))
        self.assertEqual(obs["entities"].dtype, np.float32)

    def test_parse_observation_entities_padding(self):
        obs = parse_observation(_make_obs_json(num_entities=3))
        # First 3 entities have type = 2 (unit)
        for i in range(3):
            self.assertEqual(obs["entities"][i, 0], 2.0)
        # Remaining slots should be zero-padded
        for i in range(3, MAX_ENTITIES):
            self.assertEqual(obs["entities"][i, 0], 0.0)

    def test_visibility_mask(self):
        obs = parse_observation(_make_obs_json(num_entities=3))
        # First 3 slots have entityType=2, so visibility_mask should be 1
        self.assertTrue(all(obs["visibility_mask"][:3] == 1))
        # Padded slots should be 0
        self.assertTrue(all(obs["visibility_mask"][3:] == 0))

    def test_parse_observation_map_planes_shape(self):
        obs = parse_observation(_make_obs_json())
        self.assertEqual(obs["map_planes"].shape[0], N_MAP_CHANNELS)
        self.assertEqual(obs["map_planes"].dtype, np.float32)

    def test_parse_observation_no_map_planes(self):
        raw = _make_obs_json()
        raw["mapPlanes"] = None
        obs = parse_observation(raw)
        self.assertEqual(obs["map_planes"].shape[0], N_MAP_CHANNELS)

    def test_parse_action_mask_shapes(self):
        mask = parse_action_mask(_make_mask_json())
        self.assertEqual(len(mask["macro"]), 8)
        self.assertEqual(len(mask["unit_target"]), MAX_ENTITIES)
        self.assertEqual(len(mask["unit_subaction"]), MAX_SUBACTIONS)
        self.assertEqual(len(mask["city_subaction"]), 3)
        self.assertEqual(len(mask["tech_target"]), 3)
        self.assertEqual(len(mask["policy_target"]), 2)

    def test_parse_action_mask_values(self):
        mask = parse_action_mask(_make_mask_json())
        self.assertEqual(mask["macro"][MACRO_END_TURN], 1)   # end_turn valid
        self.assertEqual(mask["macro"][2], 0)                # city_action invalid


# ---------------------------------------------------------------------------
# TestActionMapper
# ---------------------------------------------------------------------------

class TestActionMapper(unittest.TestCase):

    def test_encode_action_dict(self):
        action = {"macro": 1, "target": 5, "subaction": 0, "arg1": 3, "arg2": 0}
        encoded = encode_action(action)
        self.assertEqual(encoded["macro"], 1)
        self.assertEqual(encoded["target"], 5)

    def test_encode_action_array(self):
        action = np.array([0, 0, 0, 0, 0])
        encoded = encode_action(action)
        self.assertEqual(encoded["macro"], 0)

    def test_encode_action_defaults_to_zero(self):
        encoded = encode_action({"macro": 0})
        self.assertEqual(encoded["target"], 0)
        self.assertEqual(encoded["subaction"], 0)
        self.assertEqual(encoded["arg1"], 0)
        self.assertEqual(encoded["arg2"], 0)

    def test_encode_action_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            encode_action({"macro": 999})  # exceeds N_MACRO_ACTIONS

    def test_decode_action_result(self):
        result = {"success": True, "message": "Unit moved", "reward": 3.5}
        success, message, reward = decode_action_result(result)
        self.assertTrue(success)
        self.assertEqual(message, "Unit moved")
        self.assertAlmostEqual(reward, 3.5)

    def test_describe_action_end_turn(self):
        desc = describe_action({"macro": MACRO_END_TURN})
        self.assertEqual(desc, "end_turn")

    def test_describe_action_unit_move(self):
        desc = describe_action({
            "macro": MACRO_UNIT_ACTION,
            "target": 2,
            "subaction": UNIT_SUBACTION_MOVE,
            "arg1": 10,
            "arg2": 0,
        })
        self.assertIn("unit_action", desc)
        self.assertIn("move", desc)


# ---------------------------------------------------------------------------
# TestSpaces
# ---------------------------------------------------------------------------

class TestSpaces(unittest.TestCase):

    def test_observation_space_keys(self):
        sp = UncivSpaces()
        obs_space = sp.observation_space()
        self.assertIn("scalars", obs_space.spaces)
        self.assertIn("entities", obs_space.spaces)
        self.assertIn("map_planes", obs_space.spaces)
        self.assertIn("visibility_mask", obs_space.spaces)
        self.assertIn("action_mask", obs_space.spaces)

    def test_action_space_keys(self):
        sp = UncivSpaces()
        act_space = sp.action_space()
        self.assertIn("macro", act_space.spaces)
        self.assertIn("unit_target", act_space.spaces)
        self.assertIn("unit_subaction", act_space.spaces)
        self.assertIn("city_target", act_space.spaces)
        self.assertIn("city_subaction", act_space.spaces)
        self.assertIn("tech_target", act_space.spaces)
        self.assertIn("diplomacy_target", act_space.spaces)
        self.assertIn("diplomacy_subaction", act_space.spaces)
        self.assertIn("policy_target", act_space.spaces)
        self.assertIn("settler_target", act_space.spaces)
        self.assertIn("production_items", act_space.spaces)

    def test_observation_space_scalar_shape(self):
        sp = UncivSpaces()
        obs_space = sp.observation_space()
        self.assertEqual(obs_space["scalars"].shape, (N_SCALAR_FEATURES,))

    def test_observation_space_entity_shape(self):
        sp = UncivSpaces()
        obs_space = sp.observation_space()
        self.assertEqual(obs_space["entities"].shape, (MAX_ENTITIES, ENTITY_FEATURES))

    def test_action_space_macro_discrete(self):
        from gymnasium import spaces as gym_spaces
        sp = UncivSpaces()
        act_space = sp.action_space()
        self.assertIsInstance(act_space["macro"], gym_spaces.Discrete)


# ---------------------------------------------------------------------------
# Integration tests (live server)
# ---------------------------------------------------------------------------

_SERVER_URL = os.environ.get("UNCIV_RL_URL", "http://localhost:8080")


def _server_available() -> bool:
    try:
        import requests
        resp = requests.get(f"{_SERVER_URL}/isalive", timeout=3.0)
        return resp.ok
    except Exception:
        return False


@unittest.skipUnless(
    _server_available(),
    "Unciv RL server not available; set UNCIV_RL_URL and ensure the server is running with --rl"
)
class TestIntegrationEnv(unittest.TestCase):
    """End-to-end integration tests that require a running server."""

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

    def test_reset_returns_observations(self):
        observations, infos = self.env.reset()
        self.assertIn(self.env.possible_agents[0], observations)
        obs = observations[self.env.possible_agents[0]]
        self.assertIn("scalars", obs)
        self.assertIn("entities", obs)

    def test_step_end_turn(self):
        self.env.reset()
        agent = self.env.agent_selection
        # End the turn
        self.env.step({"macro": MACRO_END_TURN})
        # Environment should still be alive (1 agent vs 1 AI)
        self.assertIsNotNone(self.env.agent_selection)

    def test_pettingzoo_api_compliance(self):
        try:
            from pettingzoo.test import api_test
            api_test(self.env, num_cycles=5, verbose_progress=False)
        except ImportError:
            self.skipTest("pettingzoo.test.api_test not available")


if __name__ == "__main__":
    unittest.main()
