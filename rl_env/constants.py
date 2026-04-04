"""
Constants shared between the Python RL environment and the tests.

These values must stay in sync with the Kotlin-side constants defined in
``server/src/com/unciv/app/server/RLObservation.kt``.
"""

# ─── Observation dimensions ───────────────────────────────────────────────
MAX_ENTITIES: int = 512       # padded entity list (units + cities)
MAX_TILES: int = 2048         # padded flat tile count
ENTITY_FEATURES: int = 32     # float features per entity
N_MAP_CHANNELS: int = 14      # spatial map channels
N_SCALAR_FEATURES: int = 22   # global scalar features

# ─── Action space ─────────────────────────────────────────────────────────
N_MACRO_ACTIONS: int = 10
N_UNIT_SUBACTIONS: int = 7
N_CITY_SUBACTIONS: int = 4
N_DIPLOMACY_SUBACTIONS: int = 7

# Runtime-determined (override from catalogue at env reset)
MAX_TECHS: int = 150
MAX_POLICIES: int = 100
MAX_PROD_ITEMS: int = 300
MAX_IMPROVEMENTS: int = 50
MAX_AGENTS: int = 8

# ─── Macro action codes ───────────────────────────────────────────────────
MACRO_END_TURN: int = 0       # end the current agent's turn
MACRO_UNIT_MOVE: int = 1      # move unit to tile (unit_target + tile_target)
MACRO_UNIT_ATTACK: int = 2    # attack tile  (unit_target + tile_target)
MACRO_UNIT_ABILITY: int = 3   # non-move unit ability (unit_target + unit_subaction)
MACRO_CITY_ACTION: int = 4    # city management (city_target + city_subaction + production_target / tile_target)
MACRO_TECH_RESEARCH: int = 5  # research technology (tech_target)
MACRO_POLICY_ADOPT: int = 6   # adopt social policy (policy_target)
MACRO_DIPLOMACY: int = 7      # diplomatic action (diplomacy_target + diplomacy_subaction)
MACRO_WORKER_BUILD: int = 8   # build improvement (unit_target + tile_target + improvement_target)
MACRO_FOUNDER_SETTLE: int = 9 # found city at settler's current tile (unit_target)

MACRO_NAMES: list = [
    "end_turn", "unit_move", "unit_attack", "unit_ability",
    "city_action", "tech_research", "policy_adopt", "diplomacy",
    "worker_build", "founder_settle",
]

# ─── Unit sub-action codes (for MACRO_UNIT_ABILITY) ───────────────────────
UNIT_SUB_FORTIFY: int = 0     # fortify unit
UNIT_SUB_HEAL: int = 1        # wait in place to heal
UNIT_SUB_SKIP: int = 2        # skip remaining movement
UNIT_SUB_PROMOTE: int = 3     # take first available promotion
UNIT_SUB_PILLAGE: int = 4     # pillage current tile's improvement
UNIT_SUB_DISBAND: int = 5     # permanently remove unit
UNIT_SUB_FOUND_CITY: int = 6  # found city at current tile (alias for MACRO_FOUNDER_SETTLE)

UNIT_SUBACTION_NAMES: list = [
    "fortify", "heal", "skip", "promote", "pillage", "disband", "found_city",
]

# ─── City sub-action codes (for MACRO_CITY_ACTION) ────────────────────────
CITY_SUB_SET_PRODUCTION: int = 0  # change production item (production_target)
CITY_SUB_BUY_PRODUCTION: int = 1  # buy item with gold (production_target)
CITY_SUB_SELL_BUILDING: int = 2   # sell a built building (production_target)
CITY_SUB_BUY_TILE: int = 3        # purchase adjacent tile with gold (tile_target)

CITY_SUBACTION_NAMES: list = [
    "set_production", "buy_production", "sell_building", "buy_tile",
]

# ─── Diplomacy sub-action codes (for MACRO_DIPLOMACY) ────────────────────
DIPL_SUB_DECLARE_WAR: int = 0
DIPL_SUB_OFFER_PEACE: int = 1
DIPL_SUB_OPEN_BORDERS: int = 2
DIPL_SUB_FRIENDSHIP: int = 3      # declaration of friendship
DIPL_SUB_DENOUNCE: int = 4
DIPL_SUB_RESEARCH_AGREEMENT: int = 5
DIPL_SUB_DEFENSIVE_PACT: int = 6

DIPLOMACY_SUBACTION_NAMES: list = [
    "declare_war", "offer_peace", "open_borders", "friendship",
    "denounce", "research_agreement", "defensive_pact",
]

# ─── Entity type codes ────────────────────────────────────────────────────
ENTITY_PAD: int = 0
ENTITY_CITY: int = 1
ENTITY_UNIT: int = 2

# ─── Entity feature indices (32 features) ────────────────────────────────
# [0-4] identity / position
FEAT_ENTITY_TYPE: int = 0   # 0=pad, 1=city, 2=unit
FEAT_OWNER_ID: int = 1       # agent index (-1 = enemy / neutral)
FEAT_X: int = 2              # hex grid x
FEAT_Y: int = 3              # hex grid y
FEAT_VISIBLE: int = 4        # 1 if currently visible to agent
# [5-9] combat
FEAT_HP: int = 5             # unit: 0-100; city: defense HP
FEAT_MAX_HP: int = 6         # unit: 100; city: max defense HP
FEAT_STRENGTH: int = 7       # melee strength (0 for cities/civilians)
FEAT_RANGED_STRENGTH: int = 8  # ranged strength (0 if melee/city)
# [9-17] unit kinematics / capabilities
FEAT_MOVEMENT: int = 9       # remaining movement × 10 (0 for cities)
FEAT_MAX_MOVEMENT: int = 10  # base movement × 10 (0 for cities)
FEAT_IS_CIVILIAN: int = 11   # 1 if civilian unit
FEAT_IS_RANGED: int = 12     # 1 if unit has ranged attack
FEAT_CAN_FOUND_CITY: int = 13  # 1 if settler
FEAT_CAN_IMPROVE: int = 14   # 1 if worker
FEAT_IS_FORTIFIED: int = 15  # 1 if fortified
FEAT_CAN_ACT: int = 16       # 1 if unit still has actions this turn
FEAT_PROMOTIONS: int = 17    # number of promotions taken
# [18-24] city demographics / production
FEAT_POPULATION: int = 18    # city population (0 for units)
FEAT_FOOD_STOCK: int = 19    # food stored toward next growth (0 for units)
FEAT_FOOD_NEEDED: int = 20   # food needed for next growth (0 for units)
FEAT_PROD_STOCK: int = 21    # production progress in current item (0 for units)
FEAT_BUILD_QUEUE_ID: int = 22  # city: current item index (−1=none); unit: −1
FEAT_IS_CAPITAL: int = 23    # 1 if capital city
FEAT_IS_GARRISONED: int = 24  # 1 if military unit present in city
# [25-31] terrain under entity
FEAT_TERRAIN: int = 25       # base terrain index
FEAT_FEATURE: int = 26       # primary terrain feature index (0=none)
FEAT_RESOURCE: int = 27      # resource index (0=none)
FEAT_IMPROVEMENT: int = 28   # tile improvement index (0=none)
FEAT_ROAD_LEVEL: int = 29    # 0=None, 1=Road, 2=Railroad
FEAT_FOUNDED_TURN: int = 30  # city: turn founded (0 for units)
FEAT_UNIT_CLASS: int = 31    # 1=land, 2=naval, 3=air, 4=civilian, 0=city/pad

# ─── Map channel indices (14 channels) ────────────────────────────────────
CHAN_TERRAIN: int = 0        # base terrain index
CHAN_FEATURE: int = 1        # primary terrain feature index
CHAN_RESOURCE: int = 2       # resource index (0=none)
CHAN_OWNERSHIP: int = 3      # owning agent index (−1=neutral/enemy)
CHAN_VISIBILITY: int = 4     # 0=unexplored, 1=fog-of-war, 2=visible
CHAN_CITY_PRESENCE: int = 5  # 1 if any city on this tile
CHAN_OWN_UNIT: int = 6       # 1 if agent's own unit present
CHAN_ENEMY_UNIT: int = 7     # 1 if visible enemy unit present
CHAN_ROAD_LEVEL: int = 8     # 0/1/2
CHAN_IMPROVEMENT: int = 9    # improvement index
CHAN_ZOC: int = 10           # 1 if inside enemy zone of control
CHAN_THREAT: int = 11        # integer threat estimate 0–4
CHAN_OWN_TERRITORY: int = 12  # 1 if tile owned by agent's civilisation
CHAN_FRESH_WATER: int = 13   # 1 if adjacent to river/lake (affects farms)

# ─── Scalar feature indices (22 features) ────────────────────────────────
# Raw values from server; normalise in training pipeline.
SCALAR_TURN: int = 0          # current game turn
SCALAR_GOLD: int = 1          # gold treasury
SCALAR_SCIENCE: int = 2       # science yield per turn
SCALAR_CULTURE: int = 3       # culture yield per turn
SCALAR_FAITH: int = 4         # faith yield per turn
SCALAR_HAPPINESS: int = 5     # net happiness
SCALAR_NET_GOLD: int = 6      # net gold per turn (income − expenses)
SCALAR_CITIES: int = 7        # number of cities owned
SCALAR_POPULATION: int = 8    # total population across all cities
SCALAR_TECH_PROGRESS: int = 9   # fraction of current tech researched [0,1]
SCALAR_POLICY_PROGRESS: int = 10  # fraction toward next policy [0,1]
SCALAR_ERA: int = 11          # era index (0=Ancient, 1=Classical, …)
SCALAR_GOLDEN_AGE: int = 12   # 1 if in golden age
SCALAR_GOLDEN_AGE_TURNS: int = 13  # turns remaining in golden age
SCALAR_FREE_POLICIES: int = 14  # free policy slots available now
SCALAR_CURRENT_TECH: int = 15  # index of tech being researched (−1=none)
SCALAR_WARS: int = 16         # number of active wars
SCALAR_SCORE: int = 17        # total victory score
SCALAR_VICTORY_SCIENCE: int = 18   # science victory progress [0,1]
SCALAR_VICTORY_CULTURE: int = 19   # cultural victory progress [0,1]
SCALAR_VICTORY_DOMINATION: int = 20  # domination victory progress [0,1]
SCALAR_VICTORY_DIPLO: int = 21     # diplomatic victory progress [0,1]
