# RL Environment – PettingZoo Wrapper for Unciv

This document explains how to use the PettingZoo Reinforcement-Learning (RL) environment
wrapper built on top of the existing Unciv multiplayer server.

---

## Architecture Overview

```
Python RL agent                           Kotlin game engine
──────────────────        HTTP             ──────────────────────────
  pettingzoo.AECEnv  ←──────────────→   UncivServer (Ktor/Netty)
  rl_env/unciv_env.py                     server/.../RLGameManager.kt
  rl_env/client.py                        server/.../RLObservation.kt
  rl_env/obs_parser.py                    server/.../RLActionExecutor.kt
  rl_env/spaces.py                        server/.../RLActionMask.kt
```

The Kotlin server manages game state; the Python package translates HTTP JSON
responses into NumPy arrays and exposes a standard PettingZoo interface.

---

## Starting the RL-Enabled Server

### Prerequisites

1. Build the project:
   ```bash
   ./gradlew server:jar
   ```
2. The server JAR must be run from (or with `-f` pointing to) a directory that
   contains the `jsons/` folder from `android/assets/jsons/` – the ruleset data
   the game engine reads at start-up.

### Running the server

```bash
# From the repo root – jsons/ is at android/assets/jsons/
cd android/assets
java -jar ../../server/build/libs/server.jar --rl
```

Additional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--rl` | off | Enable RL endpoints (`/rl/*`) |
| `-p <port>` | 8080 | Server port |
| `--no-auth` | auth on | Disable multiplayer auth (recommended for RL) |
| `--no-chat` | chat on | Disable chat WebSockets (saves memory) |

Example minimal RL server launch:

```bash
java -jar server.jar --rl --no-auth --no-chat -p 8080
```

---

## Python Package Installation

The Python package lives in `rl_env/` in the repository root.  Install it
(along with its dependencies) with:

```bash
pip install pettingzoo gymnasium numpy requests
# then install the package itself in editable mode:
pip install -e .
```

---

## Observation Space

Each agent receives a `gymnasium.spaces.Dict` observation with the following keys:

### `scalars` – `Box(shape=(14,), dtype=float32)`

| Index | Name | Description |
|-------|------|-------------|
| 0 | `turn` | Current game turn |
| 1 | `currentPlayerId` | Agent's index in the `allAgents` list |
| 2 | `gold` | Current gold treasury |
| 3 | `sciencePerTurn` | Science yield for the next turn |
| 4 | `culturePerTurn` | Culture yield |
| 5 | `faithPerTurn` | Faith yield |
| 6 | `happiness` | Net happiness |
| 7 | `netGoldPerTurn` | Net gold income (income − maintenance) |
| 8 | `cityCount` | Number of owned cities |
| 9 | `totalPopulation` | Sum of all city populations |
| 10 | `techProgressCurrent` | Fraction of current tech completed [0..1] |
| 11 | `policyProgressCurrent` | Fraction toward next free policy [0..1] |
| 12 | `warStateFlags` | Bitmask: bit `i` set → at war with agent `i` |
| 13 | `victoryProgress` | Fraction of milestones completed [0..1] |

### `entities` – `Box(shape=(512, 25), dtype=float32)`

Each row encodes one city or unit (512 rows padded; real entities are identified
by `features[0] != 0`):

| Index | Name | Values |
|-------|------|--------|
| 0 | `entity_type` | 0=padding, 1=city, 2=unit |
| 1 | `owner_id` | Agent index (−1 = enemy/unknown) |
| 2 | `x` | Hex x-coordinate |
| 3 | `y` | Hex y-coordinate |
| 4 | `visible` | 1 if visible to this agent |
| 5 | `hp` | Current HP |
| 6 | `max_hp` | Maximum HP |
| 7 | `movement` | Remaining movement × 10 (int) |
| 8 | `strength` | Melee combat strength |
| 9 | `ranged_strength` | Ranged strength (0 if melee) |
| 10 | `population` | City population (0 for units) |
| 11 | `production_stock` | Production progress |
| 12 | `build_queue_id` | Index of current construction (−1 if none) |
| 13 | `is_capital` | 1 if this is the agent's capital |
| 14 | `is_garrisoned` | 1 if a unit is stationed in the city |
| 15 | `is_fortified` | 1 if unit is fortified |
| 16 | `has_action_available` | 1 if unit can still act this turn |
| 17 | `terrain_type` | Index into `catalogues.terrainTypes` |
| 18 | `feature_type` | Index into `catalogues.featureTypes` |
| 19 | `resource_type` | Index into `catalogues.resourceTypes` |
| 20 | `improvement_type` | Index into `catalogues.improvementTypes` |
| 21 | `road_level` | 0=none, 1=road, 2=railroad |
| 22 | `city_defenses` | City defence HP |
| 23 | `founded_turn` | Turn the city was founded |
| 24 | `last_action_turn` | Reserved (always 0) |

### `map_planes` – `Box(shape=(11, tile_count), dtype=float32)`

`tile_count` equals the number of tiles in the game map (varies by map size).
Each row corresponds to one spatial channel:

| Channel | Name | Values |
|---------|------|--------|
| 0 | `terrain_type` | Index into `catalogues.terrainTypes` |
| 1 | `feature_type` | Index into `catalogues.featureTypes` |
| 2 | `resource_type` | Index into `catalogues.resourceTypes` |
| 3 | `ownership` | Agent index, −1 = neutral |
| 4 | `visibility` | 0=unexplored, 1=fog-of-war, 2=visible |
| 5 | `city_presence` | 1 if a city occupies this tile |
| 6 | `unit_presence` | 1 if agent's unit is on this tile |
| 7 | `road_level` | 0=none, 1=road, 2=railroad |
| 8 | `improvement` | Index into `catalogues.improvementTypes` |
| 9 | `zoc` | 1 if inside enemy zone of control |
| 10 | `threat_level` | 0 (far) → 4 (immediate danger) |

### `visibility_mask` – `MultiBinary(512)`

`1` for real entity slots, `0` for padding.

### `action_mask` – nested `Dict`

Sub-keys mirror the action space; each is a `MultiBinary` indicating which
choices are currently legal (see *Action Space* below).

---

## Action Space

Each step takes a `gymnasium.spaces.Dict`:

| Key | Space | Description |
|-----|-------|-------------|
| `macro` | `Discrete(8)` | Top-level action category |
| `target` | `Discrete(512)` | Entity index / tech index / civ index |
| `subaction` | `Discrete(8)` | Sub-action within the macro |
| `arg1` | `Discrete(64)` | First argument |
| `arg2` | `Discrete(64)` | Second argument |

### Macro action reference table

| Code | Name | target | subaction | arg1 | arg2 |
|------|------|--------|-----------|------|------|
| 0 | `end_turn` | — | — | — | — |
| 1 | `unit_action` | unit entity index | 0=move, 1=attack, 2=fortify, 3=skip, 4=promote, 5=pillage, 6=disband, 7=found_city | destination tile index (move) or promotion index (promote) | — |
| 2 | `city_action` | city entity index | 0=set_production, 1=buy_production, 2=sell_building | production item index | — |
| 3 | `tech_action` | tech index | 0=research | — | — |
| 4 | `diplomacy_action` | civ index | 0=declare_war, 1=offer_peace, 2=open_borders | — | — |
| 5 | `policy_action` | policy index | 0=adopt | — | — |
| 6 | `settler_action` | settler unit entity index | 7=found_city | — | — |
| 7 | `improve_tile` | tile zero-based index | 0=road, 1=improvement, 2=clear_feature | worker entity index | improvement name index |

---

## Using the Environment

### With RLlib

```python
import ray
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from rl_env import UncivEnv

ray.init()

def env_creator(config):
    return PettingZooEnv(UncivEnv(
        base_url="http://localhost:8080",
        num_agents=2, num_ai=2,
    ))

# Register
from ray.tune.registry import register_env
register_env("unciv_rl", env_creator)
```

### With Stable-Baselines3 (via SuperSuit)

```python
import supersuit as ss
from rl_env import UncivEnv

env = UncivEnv(base_url="http://localhost:8080", num_agents=1, num_ai=3)
# Flatten observations for SB3
env = ss.flatten_v0(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
```

### With CleanRL

CleanRL expects a `gymnasium.Env`; wrap the AEC env with
`pettingzoo.utils.wrappers.BaseWrapper` or the built-in `FlattenObsWrapper`:

```python
from rl_env import UncivEnv
from rl_env.wrappers import FlattenObsWrapper

env = FlattenObsWrapper(UncivEnv(base_url="http://localhost:8080"))
observations, infos = env.reset()
```

---

## REST API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/rl/new_game` | Start a new game; returns initial observation |
| GET | `/rl/state/{gameId}` | Full observation for the current player |
| POST | `/rl/action/{gameId}` | Execute an action; returns `ActionResult` |
| GET | `/rl/action_mask/{gameId}` | Compute legal action mask |
| POST | `/rl/reset/{gameId}` | Reset to a fresh game (same config) |
| GET | `/rl/done/{gameId}` | Check termination status and scores |

All endpoints return JSON.  No authentication is required for RL endpoints.

---

## Running the Unit Tests

```bash
# From the repository root
python -m pytest rl_env/tests/ -v
```

To run the integration tests you must have the server running and set the
`UNCIV_RL_URL` environment variable (or leave it at the default
`http://localhost:8080`).  Then, in `rl_env/tests/test_env.py`, change:

```python
_SKIP_INTEGRATION = False
```
