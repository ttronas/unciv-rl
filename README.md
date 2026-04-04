# Unciv - Civ V remake for Android & Desktop

![](/extraImages/GithubPreviewImage.jpg)

[![Google Play](https://img.shields.io/static/v1?label=Google&message=Play&logo=google-play)](https://play.google.com/store/apps/details?id=com.unciv.app)
[![F-Droid](https://img.shields.io/f-droid/v/com.unciv.app?logo=f-droid)](https://f-droid.org/en/packages/com.unciv.app/)
[![itch.io](https://img.shields.io/static/v1?label=itch.io&message=Unciv&color=607D8B&logo=itch.io)](https://yairm210.itch.io/unciv)
[![Flathub](https://img.shields.io/flathub/v/io.github.yairm210.unciv?logo=flathub)](https://flathub.org/apps/details/io.github.yairm210.unciv)
[![AUR](https://img.shields.io/aur/version/unciv-bin?logo=arch-linux)](https://aur.archlinux.org/packages/unciv-bin)
[![pi-apps](https://img.shields.io/badge/dynamic/json?color=c51a4a&label=Pi-Apps&logo=raspberry-pi&query=%24.Unciv.Version&url=https%3A%2F%2Fraw.githubusercontent.com%2FBotspot%2Fpi-apps-analytics%2Fmain%2Fpackage_data_v2.json)](https://github.com/Botspot/pi-apps)
![Brew](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fformulae.brew.sh%2Fapi%2Fformula%2Funciv.json&query=%24.versions.stable&logo=homebrew&label=Brew)
[![Chocolatey](https://img.shields.io/chocolatey/v/unciv?logo=chocolatey)](https://community.chocolatey.org/packages/unciv)
[![scoop-games](https://img.shields.io/scoop/v/unciv?bucket=games)](https://github.com/Calinou/scoop-games)
 
[![Build status](https://github.com/yairm210/Unciv/actions/workflows/buildAndTest.yml/badge.svg)](https://github.com/yairm210/Unciv/actions/workflows/buildAndTest.yml)
[![Discord](https://img.shields.io/discord/586194543280390151?color=%237289DA&logo=discord&logoColor=%23FFFFFF)](https://discord.gg/bjrB4Xw)

---

## RL Environment

The `rl_env/` directory contains a [PettingZoo](https://pettingzoo.farama.org/) AEC multi-agent environment that wraps the Unciv game engine via a REST API.  The Kotlin server exposes `/rl/*` endpoints when launched with the `--rl` flag; the Python package connects to these endpoints.

### Observation Space

Each agent receives a `Dict` observation with the following keys:

| Key | Shape | Description |
|---|---|---|
| `scalars` | `(22,) float32` | Global civilisation state (see table below) |
| `entities` | `(512, 32) float32` | All visible units and cities, padded to 512 |
| `map_planes` | `(14, 2048) float32` | 14-channel spatial map, flattened to 2048 tiles |
| `visibility_mask` | `(512,) int8` | 1 for real entities, 0 for padding slots |
| `action_mask` | `Dict` | Legal action mask (see Action Space) |

#### Scalar features (22 values, index order)

| # | Name | Description |
|---|---|---|
| 0 | `turn` | Current game turn |
| 1 | `gold` | Gold treasury |
| 2 | `sciencePerTurn` | Science yield per turn |
| 3 | `culturePerTurn` | Culture yield per turn |
| 4 | `faithPerTurn` | Faith yield per turn |
| 5 | `happiness` | Net happiness |
| 6 | `netGoldPerTurn` | Net gold per turn (income − expenses) |
| 7 | `cityCount` | Number of cities |
| 8 | `totalPopulation` | Sum of all city populations |
| 9 | `techProgressFraction` | Current tech research progress [0, 1] |
| 10 | `policyProgressFraction` | Culture toward next policy [0, 1] |
| 11 | `eraIndex` | Era (0=Ancient, 1=Classical, …) |
| 12 | `isInGoldenAge` | 1 if currently in a golden age |
| 13 | `goldenAgeTurnsLeft` | Turns remaining in current golden age |
| 14 | `freePolicies` | Free policy slots available now |
| 15 | `currentTechIndex` | Index of tech being researched (−1=none) |
| 16 | `warsCount` | Number of active wars |
| 17 | `score` | Total victory score |
| 18 | `scienceVictoryProgress` | Science victory completion [0, 1] |
| 19 | `cultureVictoryProgress` | Cultural victory completion [0, 1] |
| 20 | `dominationVictoryProgress` | Domination victory completion [0, 1] |
| 21 | `diploVictoryProgress` | Diplomatic victory completion [0, 1] |

#### Entity features (32 values per entity)

| # | Name | Notes |
|---|---|---|
| 0 | `entity_type` | 0=padding, 1=city, 2=unit |
| 1 | `owner_id` | Agent index (−1=enemy/neutral) |
| 2–3 | `x`, `y` | Hex grid coordinates |
| 4 | `visible` | 1 if currently visible |
| 5–6 | `hp`, `max_hp` | Hit points |
| 7–8 | `strength`, `ranged_strength` | Combat stats |
| 9–10 | `movement`, `max_movement` | Remaining / base movement × 10 |
| 11 | `is_civilian` | 1 if civilian unit |
| 12 | `is_ranged` | 1 if has ranged attack |
| 13 | `can_found_city` | 1 if settler |
| 14 | `can_improve` | 1 if worker |
| 15 | `is_fortified` | 1 if fortified |
| 16 | `can_act` | 1 if has actions remaining |
| 17 | `promotions` | Number of promotions taken |
| 18–20 | `population`, `food_stock`, `food_needed` | City growth stats |
| 21–22 | `prod_stock`, `build_queue_id` | City production progress and item |
| 23–24 | `is_capital`, `is_garrisoned` | City flags |
| 25–29 | `terrain`, `feature`, `resource`, `improvement`, `road_level` | Tile under entity |
| 30 | `founded_turn` | Turn city was founded |
| 31 | `unit_class` | 0=city/pad, 1=land, 2=naval, 3=air, 4=civilian |

#### Map channels (14)

| # | Name | Description |
|---|---|---|
| 0 | terrain | Base terrain index |
| 1 | feature | Terrain feature index |
| 2 | resource | Resource index |
| 3 | ownership | Owner agent index (−1=neutral) |
| 4 | visibility | 0=unexplored, 1=fog, 2=visible |
| 5 | city_presence | 1 if city present |
| 6 | own_unit | 1 if agent's own unit |
| 7 | enemy_unit | 1 if visible enemy unit |
| 8 | road_level | 0/1/2 |
| 9 | improvement | Improvement index |
| 10 | zoc | 1 if in enemy zone of control |
| 11 | threat | Threat estimate 0–4 |
| 12 | own_territory | 1 if owned by agent's civ |
| 13 | fresh_water | 1 if adjacent to river/lake |

---

### Action Space

Actions are represented as a `Dict` with **12 keys**.  Only the keys relevant to the chosen `macro` need to be set; unused fields are ignored.

| Key | Size | Used by macro |
|---|---|---|
| `macro` | 10 | always |
| `unit_target` | 512 | UNIT_MOVE, UNIT_ATTACK, UNIT_ABILITY, WORKER_BUILD, FOUNDER_SETTLE |
| `unit_subaction` | 7 | UNIT_ABILITY |
| `city_target` | 512 | CITY_ACTION |
| `city_subaction` | 4 | CITY_ACTION |
| `production_target` | dynamic | CITY_ACTION |
| `tech_target` | dynamic | TECH_RESEARCH |
| `policy_target` | dynamic | POLICY_ADOPT |
| `diplomacy_target` | dynamic | DIPLOMACY |
| `diplomacy_subaction` | 7 | DIPLOMACY |
| `tile_target` | 2048 | UNIT_MOVE, UNIT_ATTACK, CITY BUY_TILE, WORKER_BUILD |
| `improvement_target` | dynamic | WORKER_BUILD |

#### Macro actions (10)

| Code | Name | Description |
|---|---|---|
| 0 | `end_turn` | End the agent's turn |
| 1 | `unit_move` | Move unit (`unit_target`) to tile (`tile_target`) |
| 2 | `unit_attack` | Attack tile (`tile_target`) with unit (`unit_target`) |
| 3 | `unit_ability` | Unit non-move ability: fortify / heal / skip / promote / pillage / disband / found_city |
| 4 | `city_action` | City management: set/buy production, sell building, buy tile |
| 5 | `tech_research` | Queue a technology for research |
| 6 | `policy_adopt` | Adopt a social policy |
| 7 | `diplomacy` | Declare war, offer peace, open borders, friendship, denounce, … |
| 8 | `worker_build` | Worker builds an improvement on a tile |
| 9 | `founder_settle` | Settler founds a city at its current tile |

#### Unit sub-actions (for `unit_ability`)

`fortify` (0), `heal` (1), `skip` (2), `promote` (3), `pillage` (4), `disband` (5), `found_city` (6)

#### City sub-actions (for `city_action`)

`set_production` (0), `buy_production` (1), `sell_building` (2), `buy_tile` (3)

#### Diplomacy sub-actions

`declare_war` (0), `offer_peace` (1), `open_borders` (2), `friendship` (3), `denounce` (4), `research_agreement` (5), `defensive_pact` (6)

---

### Quickstart

```python
from rl_env import UncivEnv
from rl_env.constants import MACRO_END_TURN

env = UncivEnv(base_url="http://localhost:8080", num_agents=2, num_ai=0)
observations, infos = env.reset(seed=42)

while env.agents:
    agent = env.agent_selection
    action_mask = infos[agent]["action_mask"]
    # Sample a legal action using the mask
    action = env.action_space(agent).sample(action_mask)
    env.step(action)
env.close()
```

Start the server with:

```bash
./gradlew server:run --args="--rl --port 8080"
```

---

## What is this?

An open source, moddability-focused Android and Desktop remake of Civ V, made with LibGDX.

## Is this any good?

Depends what you're looking for. If you're in the market for high-res graphics, amazing soundtracks, animations etc, I highly recommend Firaxis's Civ-V-like game, "Civilization V".

If you want a small, fast, moddable, FOSS, in-depth 4X that can still run on a potato, you've come to the right place :)

## How do I install?

- **Android** - [Google Play](https://play.google.com/store/apps/details?id=com.unciv.app) or [F-droid](https://f-droid.org/en/packages/com.unciv.app/)
- **Linux** - [itch.io](https://yairm210.itch.io/unciv), Flatpak via [Flathub](https://flathub.org/apps/details/io.github.yairm210.unciv), or [AUR](https://aur.archlinux.org/packages/unciv-bin)
- **Windows** - [Grab the MSI](https://github.com/yairm210/Unciv/releases/latest/download/Unciv.msi), or get from [itch.io](https://yairm210.itch.io/unciv), [Chocolatey](https://community.chocolatey.org/packages/unciv), or [Scoop](https://github.com/Calinou/scoop-games)
- **Raspberry Pi** - [Pi-apps](https://github.com/Botspot/pi-apps)
- **MacOS** - Via [Brew](https://brew.sh/) (`brew update && brew install unciv`) or install [with this guide](https://yairm210.github.io/Unciv/Other/Installing-on-macOS/) 
- Jars, APKs and Windows/Linux builds also available in [Releases](https://github.com/yairm210/Unciv/releases) (run jar with `java -jar Unciv.jar`) - *not recommended* since we update frequently and you will quickly become out-of-date
- [Build from scratch](https://yairm210.github.io/Unciv/Developers/Building-Locally/#without-android-studio) if that's your thing

## What's the roadmap?

In this order:

* Polish!
    * UI+UX improvements ([suggestions welcome!](https://github.com/yairm210/Unciv/issues/new?assignees=&labels=feature&template=feature_request.md&title=Feature+request%3A+))
    * Better automation, AI etc. in-game
* G&K mechanics - see [#4697](https://www.github.com/yairm210/Unciv/issues/4697)
* BNW mechanics - trade routes, world congress, etc.

## Contributing

Programmers start [here](https://yairm210.github.io/Unciv/Developers/Building-Locally/)!

Translators start [here](https://yairm210.github.io/Unciv/Translating/Translating/)! Language completion status [here](https://github.com/yairm210/Unciv/blob/master/android/assets/jsons/translations/completionPercentages.properties) 

Modders start [here](https://yairm210.github.io/Unciv/Modders/Mods/)!

You can join us in any of the open issues, or work on improving anything you want - once you're finished, issue a pull request and it'll go into the next version!

If not, you can help by spreading the word - vote for Unciv where you can, mention it on Reddit or Twitter etc, and help us with new ideas of how to get the word out!


## FAQ

### How about iOS?

I'm not planning on it. It means paying money to Apple, yet another release path, and since I don't have an iOS device it means I can't test it properly.

### Steam release?

Steam has decided that they don't want to host Unciv, they probably don't want to risk legal issues with Firaxis (although those should be non-existent, see below).
 
### Will you implement {feature}?

If it's in the original Civ V, then yes!

If not, then the feature won't be added to the base game - possibly it will be added as a way to mod the game, which is constantly expanding.

#### Why not? This is its own game, why not add features that weren't in Civ V?

Having a clear vision is important for actually getting things done.

Anyone can make a suggestion. Not all are good, viable, or simple. Not many can actually implement stuff.

As an open source project, this stuff is done in our spare time, of which there isn't much.

We need a clear-cut criteria to decide what to work on and what not to work on.

#### Will you implement Civ VI?

Considering how long it took to get this far, no.

### How can I learn to play? Where's the wiki?

All the tutorial information is available in-game at menu > civilopedia > tutorials

All the information is included in the amazing [Civ V wiki](https://civilization.fandom.com/wiki/)

Since this is a Civ V clone, you can search Google for how to play Civ V and there are loads of answers =)

Alternatively, you could [join us on Discord](https://discord.gg/bjrB4Xw) and ask there =D

### Aren't you basically making a Civ V clone? Is that even legal?

According to the [US Copyright Office FL-108](https://upload.wikimedia.org/wikipedia/commons/9/96/U.S._Copyright_Office_fl108.pdf), intellectual property rights *do not* apply to mechanics - as I'm sure you know, there are a billion Flappy Bird knockoffs.

It is definitely illegal:
 - To use any assets from the original game (images, sound etc) - they belong to Firaxis

It is probably illegal (no solid sources on this):
 - To use the Civilization name
 - To impersonate the Civ games (so calling yourself civi|zation with a similar logo, for instance)

Interestingly, [Civilization is a registered trademark](https://tsdr.uspto.gov/#caseNumber=74166752&caseType=SERIAL_NO&searchType=statusSearch), but it looks like it's only *that particular logo* which is trademarked, so technically you could make another game called "Civilization" and it'll stick. In any case we're not going there :) 

## Run with Docker [![Docker](https://github.com/yairm210/Unciv/actions/workflows/dockerPublish.yml/badge.svg)](https://github.com/yairm210/Unciv/actions/workflows/dockerPublish.yml)

If you have docker compose installed:

 ```$ docker compose build && docker compose up```

and then goto http://localhost:6901/vnc.html?password=headless

If just docker:

```$ docker build . -t unciv && docker run -d -p 6901:6901 -p 5901:5901 unciv  ```

Or just use our already built one:

```$ docker run -d -p 6901:6901 -p 5901:5901 ghcr.io/yairm210/unciv ```

and then goto http://localhost:6901/vnc.html?password=headless
## [Credits and 3rd parties](docs/Credits.md)
