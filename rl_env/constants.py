"""
Constants shared between the Python RL environment and the tests.

These values must stay in sync with the Kotlin-side constants defined in
``server/src/com/unciv/app/server/RLObservation.kt``.
"""

# ---------------------------------------------------------------------------
# Entity / observation constants
# ---------------------------------------------------------------------------

#: Maximum number of entities (units + cities) in the padded entity list.
MAX_ENTITIES: int = 512

#: Number of float features per entity (matches ENTITY_FEATURES in Kotlin).
ENTITY_FEATURES: int = 28

#: Number of spatial map channels (matches N_MAP_CHANNELS in Kotlin).
N_MAP_CHANNELS: int = 13

#: Number of scalar features in the ScalarObservation.
N_SCALAR_FEATURES: int = 19

# ---------------------------------------------------------------------------
# Action-space constants
# ---------------------------------------------------------------------------

#: Maximum number of macro action categories.
N_MACRO_ACTIONS: int = 8

#: Maximum number of action targets (entity index, tech index, etc.).
MAX_TARGETS: int = 512

#: Maximum number of sub-actions per macro.
MAX_SUBACTIONS: int = 9

#: Maximum number of first arguments (e.g. production item index).
MAX_ARG1: int = 64

#: Maximum number of second arguments (e.g. improvement index).
MAX_ARG2: int = 64

# ---------------------------------------------------------------------------
# Macro action index constants
# ---------------------------------------------------------------------------

MACRO_END_TURN: int = 0
MACRO_UNIT_ACTION: int = 1
MACRO_CITY_ACTION: int = 2
MACRO_TECH_ACTION: int = 3
MACRO_DIPLOMACY_ACTION: int = 4
MACRO_POLICY_ACTION: int = 5
MACRO_SETTLER_ACTION: int = 6
MACRO_IMPROVE_TILE: int = 7

MACRO_NAMES = [
    "end_turn",
    "unit_action",
    "city_action",
    "tech_action",
    "diplomacy_action",
    "policy_action",
    "settler_action",
    "improve_tile",
]

# ---------------------------------------------------------------------------
# Unit sub-action index constants
# ---------------------------------------------------------------------------

UNIT_SUBACTION_MOVE: int = 0
UNIT_SUBACTION_ATTACK: int = 1
UNIT_SUBACTION_FORTIFY: int = 2
UNIT_SUBACTION_SKIP: int = 3
UNIT_SUBACTION_PROMOTE: int = 4
UNIT_SUBACTION_PILLAGE: int = 5
UNIT_SUBACTION_DISBAND: int = 6
UNIT_SUBACTION_FOUND_CITY: int = 7
UNIT_SUBACTION_HEAL: int = 8

UNIT_SUBACTION_NAMES = [
    "move",
    "attack",
    "fortify",
    "skip",
    "promote",
    "pillage",
    "disband",
    "found_city",
    "heal",
]

# ---------------------------------------------------------------------------
# City sub-action constants
# ---------------------------------------------------------------------------

CITY_SUBACTION_SET_PRODUCTION: int = 0
CITY_SUBACTION_BUY_PRODUCTION: int = 1
CITY_SUBACTION_SELL_BUILDING: int = 2

CITY_SUBACTION_NAMES = [
    "set_production",
    "buy_production",
    "sell_building",
]

# ---------------------------------------------------------------------------
# Diplomacy sub-action constants
# ---------------------------------------------------------------------------

DIPLOMACY_DECLARE_WAR: int = 0
DIPLOMACY_OFFER_PEACE: int = 1
DIPLOMACY_OPEN_BORDERS: int = 2

DIPLOMACY_SUBACTION_NAMES = [
    "declare_war",
    "offer_peace",
    "open_borders",
]

# ---------------------------------------------------------------------------
# Entity type codes
# ---------------------------------------------------------------------------

ENTITY_TYPE_PADDING: int = 0
ENTITY_TYPE_CITY: int = 1
ENTITY_TYPE_UNIT: int = 2

# ---------------------------------------------------------------------------
# Feature indices inside each entity vector (see RLObservation.kt for docs)
# ---------------------------------------------------------------------------

FEAT_ENTITY_TYPE: int = 0
FEAT_OWNER_ID: int = 1
FEAT_X: int = 2
FEAT_Y: int = 3
FEAT_VISIBLE: int = 4
FEAT_HP: int = 5
FEAT_MAX_HP: int = 6
FEAT_MOVEMENT: int = 7
FEAT_STRENGTH: int = 8
FEAT_RANGED_STRENGTH: int = 9
FEAT_POPULATION: int = 10
FEAT_PRODUCTION_STOCK: int = 11
FEAT_BUILD_QUEUE_ID: int = 12
FEAT_IS_CAPITAL: int = 13
FEAT_IS_GARRISONED: int = 14
FEAT_IS_FORTIFIED: int = 15
FEAT_HAS_ACTION: int = 16
FEAT_TERRAIN_TYPE: int = 17
FEAT_FEATURE_TYPE: int = 18
FEAT_RESOURCE_TYPE: int = 19
FEAT_IMPROVEMENT_TYPE: int = 20
FEAT_ROAD_LEVEL: int = 21
FEAT_CITY_DEFENSES: int = 22
FEAT_FOUNDED_TURN: int = 23
FEAT_LAST_ACTION_TURN: int = 24
#: Number of promotions taken by a unit (0 for cities).
FEAT_PROMOTIONS_COUNT: int = 25
#: Food stored toward next population growth (0 for units).
FEAT_FOOD_STOCK: int = 26
#: Food needed to reach next population growth (0 for units).
FEAT_FOOD_NEEDED: int = 27

# ---------------------------------------------------------------------------
# Map channel indices (see RLObservation.kt for docs)
# ---------------------------------------------------------------------------

CHAN_TERRAIN: int = 0
CHAN_FEATURE: int = 1
CHAN_RESOURCE: int = 2
CHAN_OWNERSHIP: int = 3
CHAN_VISIBILITY: int = 4
CHAN_CITY_PRESENCE: int = 5
CHAN_UNIT_PRESENCE: int = 6
CHAN_ROAD_LEVEL: int = 7
CHAN_IMPROVEMENT: int = 8
CHAN_ZOC: int = 9
CHAN_THREAT: int = 10
#: 1 if a visible enemy unit is present on this tile.
CHAN_ENEMY_UNIT: int = 11
#: 1 if the tile is owned by the observing civilisation.
CHAN_OWN_TERRITORY: int = 12

# ---------------------------------------------------------------------------
# Scalar observation indices (see RLObservation.kt for docs)
# ---------------------------------------------------------------------------

SCALAR_TURN: int = 0
SCALAR_CURRENT_PLAYER_ID: int = 1
SCALAR_GOLD: int = 2
SCALAR_SCIENCE_PER_TURN: int = 3
SCALAR_CULTURE_PER_TURN: int = 4
SCALAR_FAITH_PER_TURN: int = 5
SCALAR_HAPPINESS: int = 6
SCALAR_NET_GOLD_PER_TURN: int = 7
SCALAR_CITY_COUNT: int = 8
SCALAR_TOTAL_POPULATION: int = 9
SCALAR_TECH_PROGRESS: int = 10
SCALAR_POLICY_PROGRESS: int = 11
SCALAR_WAR_FLAGS: int = 12
SCALAR_VICTORY_PROGRESS: int = 13
#: 1 if the civilisation is currently in a golden age, 0 otherwise.
SCALAR_IS_GOLDEN_AGE: int = 14
#: Turns remaining in the current golden age (0 if not in one).
SCALAR_GOLDEN_AGE_TURNS: int = 15
#: Era number of the civilisation (0=Ancient, 1=Classical, …).
SCALAR_ERA_INDEX: int = 16
#: Number of free social-policy slots available to adopt right now.
SCALAR_FREE_POLICIES: int = 17
#: Index of the technology currently being researched (−1 if none).
SCALAR_CURRENT_TECH_INDEX: int = 18
