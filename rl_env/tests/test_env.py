"""
Tests for the Unciv RL environment.

Unit tests (no server required)
    - :class:`TestObsParser` – parse_observation / parse_action_mask
    - :class:`TestActionMapper` – encode_action / describe_action
    - :class:`TestSpaces` – Gymnasium space definitions

Integration tests (require a running Unciv server with --rl flag)
    Set env var UNCIV_RL_URL to override the default http://localhost:8080.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

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
    MACRO_UNIT_ATTACK,
    MACRO_UNIT_ABILITY,
    MACRO_CITY_ACTION,
    MACRO_TECH_RESEARCH,
    MACRO_POLICY_ADOPT,
    MACRO_DIPLOMACY,
    MACRO_WORKER_BUILD,
    MACRO_FOUNDER_SETTLE,
    UNIT_SUB_FORTIFY,
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
        "gold": 120,
        "sciencePerTurn": 8.5,
        "culturePerTurn": 3.2,
        "faithPerTurn": 1.0,
        "happiness": 4,
        "netGoldPerTurn": 2.5,
        "cityCount": 2,
        "totalPopulation": 10,
        "techProgressFraction": 0.42,
        "policyProgressFraction": 0.1,
        "eraIndex": 1,
        "isInGoldenAge": 0,
        "goldenAgeTurnsLeft": 0,
        "freePolicies": 0,
        "currentTechIndex": 3,
        "warsCount": 1,
        "score": 250,
        "scienceVictoryProgress": 0.1,
        "cultureVictoryProgress": 0.05,
        "dominationVictoryProgress": 0.0,
        "diploVictoryProgress": 0.0,
    }


def _make_entity_json(entity_type: int = 2) -> dict:
    features = [entity_type] + [0] * (ENTITY_FEATURES - 1)
    return {"features": features}


def _make_obs_json(num_entities: int = 3) -> dict:
    entities = [_make_entity_json(2) for _ in range(num_entities)]
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
            "improvementNames": ["Farm", "Mine", "Trading Post"],
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
        "macroMask": [True, True, False, True, False, False, False, False, False, False],
        "unitTargetMask": ([True] * 3 + [False] * (MAX_ENTITIES - 3)),
        "unitSubactionMask": [True, False, True, True, False, False, False],
        "cityTargetMask": ([False] * MAX_ENTITIES),
        "citySubactionMask": [True, False, False, False],
        "productionTargetMask": [True, False],
        "techTargetMask": [True, False, True],
        "diplomacyTargetMask": [False] * num_agents,
        "diplomacySubactionMask": [False] * N_DIPLOMACY_SUBACTIONS,
        "policyTargetMask": [True, False],
        "tileTargetMask": [False, True, False],
        "improvementTargetMask": [True, False, False],
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
        self.assertAlmostEqual(float(obs["scalars"][0]), 5.0)    # turn
        self.assertAlmostEqual(float(obs["scalars"][1]), 120.0)  # gold
        self.assertAlmostEqual(float(obs["scalars"][17]), 250.0) # score

    def test_parse_observation_entities_shape(self):
        obs = parse_observation(_make_obs_json(num_entities=5))
        self.assertEqual(obs["entities"].shape, (MAX_ENTITIES, ENTITY_FEATURES))
        self.assertEqual(obs["entities"].dtype, np.float32)

    def test_parse_observation_entities_padding(self):
        obs = parse_observation(_make_obs_json(num_entities=3))
        for i in range(3):
            self.assertEqual(obs["entities"][i, 0], 2.0)
        for i in range(3, MAX_ENTITIES):
            self.assertEqual(obs["entities"][i, 0], 0.0)

    def test_visibility_mask(self):
        obs = parse_observation(_make_obs_json(num_entities=3))
        self.assertTrue(all(obs["visibility_mask"][:3] == 1))
        self.assertTrue(all(obs["visibility_mask"][3:] == 0))

    def test_parse_observation_map_planes_shape(self):
        obs = parse_observation(_make_obs_json())
        self.assertEqual(obs["map_planes"].shape, (N_MAP_CHANNELS, MAX_TILES))
        self.assertEqual(obs["map_planes"].dtype, np.float32)

    def test_parse_observation_no_map_planes(self):
        raw = _make_obs_json()
        raw["mapPlanes"] = None
        obs = parse_observation(raw)
        self.assertEqual(obs["map_planes"].shape, (N_MAP_CHANNELS, MAX_TILES))

    def test_parse_action_mask_shapes(self):
        mask = parse_action_mask(_make_mask_json())
        self.assertEqual(len(mask["macro"]), N_MACRO_ACTIONS)
        self.assertEqual(len(mask["unit_target"]), MAX_ENTITIES)
        self.assertEqual(len(mask["unit_subaction"]), N_UNIT_SUBACTIONS)
        self.assertEqual(len(mask["city_subaction"]), N_CITY_SUBACTIONS)
        self.assertEqual(len(mask["tech_target"]), 3)
        self.assertEqual(len(mask["policy_target"]), 2)
        self.assertEqual(len(mask["tile_target"]), 3)
        self.assertEqual(len(mask["improvement_target"]), 3)

    def test_parse_action_mask_values(self):
        mask = parse_action_mask(_make_mask_json())
        self.assertEqual(mask["macro"][MACRO_END_TURN], 1)
        self.assertEqual(mask["macro"][MACRO_CITY_ACTION], 0)


# ---------------------------------------------------------------------------
# TestActionMapper
# ---------------------------------------------------------------------------

class TestActionMapper(unittest.TestCase):

    def test_encode_action_semantic_dict(self):
        action = {
            "macro": MACRO_UNIT_MOVE,
            "unit_target": 5,
            "tile_target": 42,
        }
        encoded = encode_action(action)
        self.assertEqual(encoded["macro"], MACRO_UNIT_MOVE)
        self.assertEqual(encoded["unitTarget"], 5)
        self.assertEqual(encoded["tileTarget"], 42)

    def test_encode_action_camel_dict(self):
        action = {"macro": MACRO_UNIT_ATTACK, "unitTarget": 3, "tileTarget": 7}
        encoded = encode_action(action)
        self.assertEqual(encoded["macro"], MACRO_UNIT_ATTACK)
        self.assertEqual(encoded["unitTarget"], 3)

    def test_encode_action_array(self):
        action = np.array([MACRO_END_TURN, 0, 0, 0, 0])
        encoded = encode_action(action)
        self.assertEqual(encoded["macro"], MACRO_END_TURN)

    def test_encode_action_defaults_to_zero(self):
        encoded = encode_action({"macro": MACRO_END_TURN})
        self.assertEqual(encoded["unitTarget"], 0)
        self.assertEqual(encoded["tileTarget"], 0)

    def test_encode_action_out_of_bounds_raises(self):
        with self.assertRaises(ValueError):
            encode_action({"macro": 999})

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
            "macro": MACRO_UNIT_MOVE,
            "unitTarget": 2,
            "tileTarget": 10,
        })
        self.assertIn("unit_move", desc)

    def test_describe_action_unit_ability(self):
        desc = describe_action({
            "macro": MACRO_UNIT_ABILITY,
            "unitTarget": 1,
            "unitSubaction": UNIT_SUB_FORTIFY,
        })
        self.assertIn("unit_ability", desc)
        self.assertIn("fortify", desc)


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
        expected = [
            "macro", "unit_target", "unit_subaction",
            "city_target", "city_subaction", "production_target",
            "tech_target", "policy_target",
            "diplomacy_target", "diplomacy_subaction",
            "tile_target", "improvement_target",
        ]
        for key in expected:
            self.assertIn(key, act_space.spaces)

    def test_action_mask_space_keys_match_action_space(self):
        sp = UncivSpaces()
        act_keys = set(sp.action_space().spaces.keys())
        mask_keys = set(sp.action_mask_space().spaces.keys())
        self.assertEqual(act_keys, mask_keys)

    def test_observation_space_scalar_shape(self):
        sp = UncivSpaces()
        obs_space = sp.observation_space()
        self.assertEqual(obs_space["scalars"].shape, (N_SCALAR_FEATURES,))

    def test_observation_space_entity_shape(self):
        sp = UncivSpaces()
        obs_space = sp.observation_space()
        self.assertEqual(obs_space["entities"].shape, (MAX_ENTITIES, ENTITY_FEATURES))

    def test_observation_space_map_planes_shape(self):
        sp = UncivSpaces()
        obs_space = sp.observation_space()
        self.assertEqual(obs_space["map_planes"].shape, (N_MAP_CHANNELS, MAX_TILES))

    def test_action_space_macro_discrete(self):
        from gymnasium import spaces as gym_spaces
        sp = UncivSpaces()
        act_space = sp.action_space()
        self.assertIsInstance(act_space["macro"], gym_spaces.Discrete)
        self.assertEqual(act_space["macro"].n, N_MACRO_ACTIONS)


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

    def test_pettingzoo_api_compliance(self):
        from pettingzoo.test import api_test
        api_test(self.env, num_cycles=5, verbose_progress=False)

    def test_reset_returns_observations(self):
        obs, infos = self.env.reset(seed=0)
        self.assertTrue(len(obs) > 0)
        for agent in self.env.agents:
            self.assertIn("scalars", obs[agent])
            self.assertIn("entities", obs[agent])

    def test_step_end_turn(self):
        self.env.reset(seed=0)
        agent = self.env.agent_selection
        self.env.step({"macro": MACRO_END_TURN})
        _ = self.env.observe(agent)


if __name__ == "__main__":
    unittest.main()
