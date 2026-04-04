package com.unciv.app.server

import com.unciv.logic.GameInfo
import com.unciv.logic.city.City
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.civilization.diplomacy.DiplomaticStatus
import com.unciv.logic.map.mapunit.MapUnit
import com.unciv.logic.map.tile.RoadStatus
import com.unciv.logic.map.tile.Tile
import com.unciv.models.ruleset.tile.TerrainType
import com.unciv.ui.screens.victoryscreen.RankingType
import kotlinx.serialization.Serializable
import kotlin.math.roundToInt

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Maximum padded entity list length (units + cities per game). */
const val MAX_ENTITIES = 512

/** Number of float features encoded per entity. */
const val ENTITY_FEATURES = 25

/** Number of spatial map channels. */
const val N_MAP_CHANNELS = 11

// ---------------------------------------------------------------------------
// Serializable data classes
// ---------------------------------------------------------------------------

/**
 * The 14 scalar game-state features for the current agent's civilisation.
 */
@Serializable
data class ScalarObservation(
    /** Current game turn. */
    val turn: Int,
    /** Index of the agent's civ in [RLObservationResponse.allAgents]. */
    val currentPlayerId: Int,
    /** Gold treasury. */
    val gold: Int,
    /** Science yield for the upcoming turn. */
    val sciencePerTurn: Float,
    /** Culture yield for the upcoming turn. */
    val culturePerTurn: Float,
    /** Faith yield for the upcoming turn. */
    val faithPerTurn: Float,
    /** Net happiness. */
    val happiness: Int,
    /** Net gold yield for the upcoming turn (income − expenses including unit maintenance). */
    val netGoldPerTurn: Float,
    /** Number of cities owned by the agent. */
    val cityCount: Int,
    /** Sum of all city populations. */
    val totalPopulation: Int,
    /**
     * Research progress fraction [0..1] for the technology currently being studied.
     * 0 if no technology is selected.
     */
    val techProgressCurrent: Float,
    /**
     * Policy progress fraction [0..1]: storedCulture / cost of the next policy.
     * 0 if no culture policy is available.
     */
    val policyProgressCurrent: Float,
    /**
     * Bitmask: bit i is set when the agent is at war with civilisation index i.
     * Civilisation order is the same as [RLObservationResponse.allAgents].
     */
    val warStateFlags: Int,
    /**
     * Overall victory progress [0..1]: fraction of the most-advanced victory milestone
     * that the agent has completed.
     */
    val victoryProgress: Float
)

/**
 * A single encoded entity (city or unit).  Slots beyond the real entity count are
 * padded with all-zero values; callers can test [entityType] == 0 to identify padding.
 *
 * Feature layout (same order as the Python [ENTITY_FEATURES] constant):
 *  0  entity_type          1=city  2=unit  0=padding
 *  1  owner_id             index into allAgents (-1 = enemy/neutral not in agent list)
 *  2  x                    hex x-coordinate
 *  3  y                    hex y-coordinate
 *  4  visible              1 if currently visible to the agent, 0 otherwise
 *  5  hp                   current HP (0-100 for units; city defence HP for cities)
 *  6  max_hp               maximum HP
 *  7  movement             remaining movement points × 10 (int)
 *  8  strength             melee combat strength (0 for cities)
 *  9  ranged_strength      ranged strength (0 for melee/cities)
 * 10  population           city population (0 for units)
 * 11  production_stock     city production progress (0 for units)
 * 12  build_queue_id       index of the current construction in the available list (-1 if none)
 * 13  is_capital           1 if capital city
 * 14  is_garrisoned        1 if a unit is inside the city
 * 15  is_fortified         1 if the unit is fortified
 * 16  has_action_available 1 if the unit has remaining movement or can still act
 * 17  terrain_type         index of the tile's base terrain in [RLObservationResponse.terrainTypes]
 * 18  feature_type         index of the tile's primary terrain feature (0 = none)
 * 19  resource_type        index of the tile's resource (0 = none)
 * 20  improvement_type     index of the tile's improvement (0 = none)
 * 21  road_level           0=None 1=Road 2=Railroad
 * 22  city_defenses        city defence HP (0 for units)
 * 23  founded_turn         game turn the city was founded (0 for units)
 * 24  last_action_turn     last turn the entity performed an action (placeholder, 0 for now)
 */
@Serializable
data class EntityObservation(
    val features: List<Int>
)

/**
 * A multi-channel spatial map.  Data is stored in row-major order:
 * index = channel * (height * width) + row * width + col
 *
 * Channel meanings (matching N_MAP_CHANNELS = 11):
 *  0  terrain_type   index into [RLObservationResponse.terrainTypes]
 *  1  feature_type   index into [RLObservationResponse.featureTypes]
 *  2  resource_type  index into [RLObservationResponse.resourceTypes]
 *  3  ownership      index into allAgents, -1 = neutral/unknown
 *  4  visibility     0=unexplored  1=fog-of-war (explored but not currently visible)  2=visible
 *  5  city_presence  1 if a city is on this tile
 *  6  unit_presence  1 if a unit belonging to the agent is on this tile
 *  7  road_level     0=None 1=Road 2=Railroad
 *  8  improvement    index into [RLObservationResponse.improvementTypes]
 *  9  zoc            1 if this tile is inside an enemy zone of control
 * 10  threat_level   integer threat estimate (0–4)
 */
@Serializable
data class MapPlanes(
    val channels: Int,
    val height: Int,
    val width: Int,
    /** Flat list, length = channels × height × width. */
    val data: List<Int>
)

/** Per-tile fog-of-war annotation. */
@Serializable
data class FogOfWarEntry(
    val tileIndex: Int,
    val visibleToAgent: Boolean,
    val lastSeenTurn: Int
)

/**
 * Catalogue lists used to map integer IDs back to names.
 * Indices are stable for a given game (same ruleset).
 */
@Serializable
data class RLCatalogues(
    val terrainTypes: List<String>,
    val featureTypes: List<String>,
    val resourceTypes: List<String>,
    val improvementTypes: List<String>,
    val techNames: List<String>,
    val policyNames: List<String>,
    val productionItemNames: List<String>
)

/** Full observation response returned by GET /rl/state/{gameId} */
@Serializable
data class RLObservationResponse(
    val gameId: String,
    /** civID of the agent whose POV this observation represents. */
    val agentCivId: String,
    /** Index of [agentCivId] in [allAgents]. */
    val agentIndex: Int,
    /** Ordered list of all RL-agent civ IDs (no AI or spectator civs). */
    val allAgents: List<String>,
    val scalars: ScalarObservation,
    /** Padded to [MAX_ENTITIES] entries; padding entries have all-zero features. */
    val entities: List<EntityObservation>,
    /** Spatial map planes (may be omitted if mapPlanes is null). */
    val mapPlanes: MapPlanes?,
    /** One entry per explored tile. */
    val fogOfWar: List<FogOfWarEntry>,
    /** Index-to-name catalogues for categorical features. */
    val catalogues: RLCatalogues,
    /** True once a winner has been declared or turn limit exceeded. */
    val done: Boolean,
    /** Name of the winning civ, or null if the game is still in progress. */
    val winner: String?,
    /** Score of every agent at the time of this observation. */
    val scores: Map<String, Int>
)

// ---------------------------------------------------------------------------
// Observation builder
// ---------------------------------------------------------------------------

/**
 * Extracts a complete [RLObservationResponse] from [gameInfo] for the given
 * agent civilisation.  This is a pure read-only pass over the game state.
 *
 * @param includeMapPlanes Set to false to skip the (potentially large) spatial planes.
 */
fun buildObservation(
    gameInfo: GameInfo,
    agentCivId: String,
    allAgents: List<String>,
    includeMapPlanes: Boolean = true
): RLObservationResponse {

    val civ = gameInfo.getCivilization(agentCivId)
    val agentIndex = allAgents.indexOf(agentCivId)
    val ruleset = gameInfo.ruleset

    // ---- catalogues -------------------------------------------------------
    val terrainList = ruleset.terrains.keys.sorted()
    val featureList = listOf("") + ruleset.terrains.values
        .filter { it.type == TerrainType.TerrainFeature }.map { it.name }.sorted()
    val resourceList = listOf("") + ruleset.tileResources.keys.sorted()
    val improvementList = listOf("") + ruleset.tileImprovements.keys.sorted()
    val techList = ruleset.technologies.keys.sorted()
    val policyList = ruleset.policies.keys.sorted()

    // Production items = all buildings + units from the ruleset
    val productionList = (ruleset.buildings.keys + ruleset.units.keys).distinct().sorted()

    val catalogues = RLCatalogues(
        terrainTypes = terrainList,
        featureTypes = featureList,
        resourceTypes = resourceList,
        improvementTypes = improvementList,
        techNames = techList,
        policyNames = policyList,
        productionItemNames = productionList
    )

    // ---- scalars ----------------------------------------------------------
    val statsNext = civ.stats.statsForNextTurn
    val techName = civ.tech.currentTechnologyName()
    val techProgress: Float = if (techName != null) {
        val cost = civ.tech.costOfTech(techName).toFloat()
        if (cost > 0f) civ.tech.researchOfTech(techName) / cost else 0f
    } else 0f

    val cultureCostNextPolicy = civ.policies.getCultureNeededForNextPolicy()
    val policyProgress: Float =
        if (cultureCostNextPolicy > 0) civ.policies.storedCulture.toFloat() / cultureCostNextPolicy
        else 0f

    var warFlags = 0
    for ((idx, otherId) in allAgents.withIndex()) {
        if (otherId == agentCivId) continue
        val otherCiv = gameInfo.getCivilizationOrNull(otherId) ?: continue
        val dm = civ.getDiplomacyManager(otherCiv) ?: continue
        if (dm.diplomaticStatus == DiplomaticStatus.War) warFlags = warFlags or (1 shl idx)
    }

    // Victory progress: fraction of milestones completed across all victory types
    val allMilestones = ruleset.victories.values.flatMap { it.milestoneObjects }
    val victoryProgress: Float = if (allMilestones.isEmpty()) 0f else
        allMilestones.count { it.hasBeenCompletedBy(civ) }.toFloat() / allMilestones.size

    // Net gold per turn (includes unit maintenance as negative contribution)
    val netGoldPerTurn = statsNext.gold

    val scalars = ScalarObservation(
        turn = gameInfo.turns,
        currentPlayerId = agentIndex,
        gold = civ.gold,
        sciencePerTurn = statsNext.science,
        culturePerTurn = statsNext.culture,
        faithPerTurn = statsNext.faith,
        happiness = civ.getHappiness(),
        netGoldPerTurn = netGoldPerTurn,
        cityCount = civ.cities.size,
        totalPopulation = civ.cities.sumOf { it.population.population },
        techProgressCurrent = techProgress.coerceIn(0f, 1f),
        policyProgressCurrent = policyProgress.coerceIn(0f, 1f),
        warStateFlags = warFlags,
        victoryProgress = victoryProgress.coerceIn(0f, 1f)
    )

    // ---- entities ---------------------------------------------------------
    val civIdToIndex: Map<String, Int> = allAgents.mapIndexed { i, id -> id to i }.toMap()

    val entities = mutableListOf<EntityObservation>()

    // Units first (all civs, filtered by visibility for non-agent civs)
    for (tile in gameInfo.tileMap.values) {
        for (unit in tile.getUnits()) {
            val enc = encodeUnit(unit, civ, civIdToIndex, terrainList, featureList,
                resourceList, improvementList)
            if (enc != null) entities += enc
        }
    }
    // Then cities
    for (gameCiv in gameInfo.civilizations) {
        if (gameCiv.isDefeated()) continue
        for (city in gameCiv.cities) {
            entities += encodeCity(city, civ, civIdToIndex, terrainList, featureList,
                resourceList, improvementList, productionList)
        }
    }

    // Pad to MAX_ENTITIES
    val padding = EntityObservation(features = List(ENTITY_FEATURES) { 0 })
    while (entities.size < MAX_ENTITIES) entities += padding
    val paddedEntities = entities.take(MAX_ENTITIES)

    // ---- map planes -------------------------------------------------------
    val planes: MapPlanes? = if (includeMapPlanes) buildMapPlanes(
        gameInfo, civ, civIdToIndex, terrainList, featureList, resourceList, improvementList
    ) else null

    // ---- fog of war -------------------------------------------------------
    val fogOfWar = gameInfo.tileMap.values
        .filter { it.isExplored(civ) }
        .map { tile ->
            val lastTurn = tile.history.maxOfOrNull { it.key } ?: 0
            FogOfWarEntry(
                tileIndex = tile.zeroBasedIndex,
                visibleToAgent = tile.isVisible(civ),
                lastSeenTurn = lastTurn
            )
        }

    // ---- done / winner ----------------------------------------------------
    val done = gameInfo.victoryData != null
    val winner = gameInfo.victoryData?.winningCiv

    val scores = allAgents.associateWith { id ->
        gameInfo.getCivilizationOrNull(id)?.getStatForRanking(RankingType.Score) ?: 0
    }

    return RLObservationResponse(
        gameId = gameInfo.gameId,
        agentCivId = agentCivId,
        agentIndex = agentIndex,
        allAgents = allAgents,
        scalars = scalars,
        entities = paddedEntities,
        mapPlanes = planes,
        fogOfWar = fogOfWar,
        catalogues = catalogues,
        done = done,
        winner = winner,
        scores = scores
    )
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

private fun encodeUnit(
    unit: MapUnit,
    observer: Civilization,
    civIdToIndex: Map<String, Int>,
    terrainList: List<String>,
    featureList: List<String>,
    resourceList: List<String>,
    improvementList: List<String>
): EntityObservation? {
    val tile = unit.getTile()
    val visible = tile.isVisible(observer)

    // Only expose enemy/neutral units that are currently visible to the observer
    if (unit.civ != observer && !visible) return null

    val ownerId = civIdToIndex[unit.civ.civID] ?: -1
    val terrainIdx = terrainList.indexOf(tile.baseTerrain).coerceAtLeast(0)
    val featureIdx = tile.terrainFeatures.firstOrNull()?.let { f ->
        featureList.indexOf(f).let { if (it < 0) 0 else it }
    } ?: 0
    val resourceIdx = tile.resource?.let { r ->
        resourceList.indexOf(r).let { if (it < 0) 0 else it }
    } ?: 0
    val improvIdx = tile.improvement?.let { imp ->
        improvementList.indexOf(imp).let { if (it < 0) 0 else it }
    } ?: 0
    val roadLevel = when (tile.getUnpillagedRoad()) {
        RoadStatus.None -> 0; RoadStatus.Road -> 1; RoadStatus.Railroad -> 2
    }

    val features = listOf(
        2,                                                    //  0 entity_type = unit
        ownerId,                                              //  1 owner_id
        tile.position.x,                                      //  2 x
        tile.position.y,                                      //  3 y
        if (visible) 1 else 0,                               //  4 visible
        unit.health,                                          //  5 hp
        100,                                                  //  6 max_hp
        (unit.currentMovement * 10).roundToInt(),             //  7 movement ×10
        unit.baseUnit.strength,                               //  8 strength
        unit.baseUnit.rangedStrength,                         //  9 ranged_strength
        0,                                                    // 10 population
        0,                                                    // 11 production_stock
        -1,                                                   // 12 build_queue_id
        0,                                                    // 13 is_capital
        0,                                                    // 14 is_garrisoned
        if (unit.isFortified()) 1 else 0,                    // 15 is_fortified
        if (unit.due && unit.currentMovement > 0f) 1 else 0, // 16 has_action_available
        terrainIdx,                                           // 17 terrain_type
        featureIdx,                                           // 18 feature_type
        resourceIdx,                                          // 19 resource_type
        improvIdx,                                            // 20 improvement_type
        roadLevel,                                            // 21 road_level
        0,                                                    // 22 city_defenses
        0,                                                    // 23 founded_turn
        0                                                     // 24 last_action_turn
    )
    return EntityObservation(features = features)
}

private fun encodeCity(
    city: City,
    observer: Civilization,
    civIdToIndex: Map<String, Int>,
    terrainList: List<String>,
    featureList: List<String>,
    resourceList: List<String>,
    improvementList: List<String>,
    productionList: List<String>
): EntityObservation {
    val tile = city.getCenterTile()
    val visible = tile.isVisible(observer) || tile.isExplored(observer)

    val ownerId = civIdToIndex[city.civ.civID] ?: -1
    val terrainIdx = terrainList.indexOf(tile.baseTerrain).coerceAtLeast(0)
    val featureIdx = tile.terrainFeatures.firstOrNull()?.let { f ->
        featureList.indexOf(f).let { if (it < 0) 0 else it }
    } ?: 0
    val resourceIdx = tile.resource?.let { r ->
        resourceList.indexOf(r).let { if (it < 0) 0 else it }
    } ?: 0
    val improvIdx = tile.improvement?.let { imp ->
        improvementList.indexOf(imp).let { if (it < 0) 0 else it }
    } ?: 0
    val roadLevel = when (tile.getUnpillagedRoad()) {
        RoadStatus.None -> 0; RoadStatus.Road -> 1; RoadStatus.Railroad -> 2
    }
    val currentConstructionName = city.cityConstructions.currentConstructionName()
    val buildQueueId = if (currentConstructionName.isEmpty()) -1
    else productionList.indexOf(currentConstructionName).let { if (it < 0) -1 else it }

    val productionProgress = if (currentConstructionName.isNotEmpty())
        city.cityConstructions.getWorkDone(currentConstructionName)
    else 0

    val features = listOf(
        1,                                              //  0 entity_type = city
        ownerId,                                        //  1 owner_id
        tile.position.x,                                //  2 x
        tile.position.y,                                //  3 y
        if (visible) 1 else 0,                          //  4 visible
        city.health,                                    //  5 hp (city defence HP)
        200 + city.cityConstructions.getBuiltBuildings().sumOf { it.cityHealth }, //  6 max_hp
        0,                                              //  7 movement
        0,                                              //  8 strength
        0,                                              //  9 ranged_strength
        city.population.population,                     // 10 population
        productionProgress,                             // 11 production_stock
        buildQueueId,                                   // 12 build_queue_id
        if (city.isCapital()) 1 else 0,                 // 13 is_capital
        if (tile.militaryUnit != null) 1 else 0,        // 14 is_garrisoned
        0,                                              // 15 is_fortified
        0,                                              // 16 has_action_available
        terrainIdx,                                     // 17 terrain_type
        featureIdx,                                     // 18 feature_type
        resourceIdx,                                    // 19 resource_type
        improvIdx,                                      // 20 improvement_type
        roadLevel,                                      // 21 road_level
        city.health,                                    // 22 city_defenses
        city.turnAcquired,                              // 23 founded_turn
        0                                               // 24 last_action_turn
    )
    return EntityObservation(features = features)
}

private fun buildMapPlanes(
    gameInfo: GameInfo,
    observer: Civilization,
    civIdToIndex: Map<String, Int>,
    terrainList: List<String>,
    featureList: List<String>,
    resourceList: List<String>,
    improvementList: List<String>
): MapPlanes {
    val tiles = gameInfo.tileMap.values.toList()
    val tileCount = tiles.size

    // Flat layout: channel × tileCount (tileCount = height, width = 1 for simplicity)
    val data = IntArray(N_MAP_CHANNELS * tileCount) { 0 }

    for (tile in tiles) {
        val i = tile.zeroBasedIndex
        val explored = tile.isExplored(observer)
        val visible = tile.isVisible(observer)

        val visibilityCode = when {
            visible -> 2
            explored -> 1
            else -> 0
        }

        val terrainIdx = terrainList.indexOf(tile.baseTerrain).coerceAtLeast(0)
        val featureIdx = if (explored) tile.terrainFeatures.firstOrNull()
            ?.let { f -> featureList.indexOf(f).let { if (it < 0) 0 else it } } ?: 0 else 0
        val resourceIdx = if (explored) tile.resource
            ?.let { r -> resourceList.indexOf(r).let { if (it < 0) 0 else it } } ?: 0 else 0

        val ownerIdx = if (explored)
            tile.owningCity?.civ?.civID?.let { civIdToIndex[it] } ?: -1
        else -1

        val cityPresence = if (visible && tile.isCityCenter()) 1 else 0
        val unitPresence = if (visible)
            if (tile.getUnits().any { it.civ.civID == observer.civID }) 1 else 0
        else 0

        val roadLevel = if (explored) when (tile.getUnpillagedRoad()) {
            RoadStatus.None -> 0; RoadStatus.Road -> 1; RoadStatus.Railroad -> 2
        } else 0
        val improvIdx = if (explored) tile.improvement
            ?.let { imp -> improvementList.indexOf(imp).let { if (it < 0) 0 else it } } ?: 0 else 0

        // ZoC: 1 if any adjacent tile has an enemy unit (and this tile is visible)
        val zoc = if (visible) tile.neighbors.any { nb ->
            nb.getUnits().any { u ->
                val dm = observer.getDiplomacyManager(u.civ)
                dm?.diplomaticStatus == DiplomaticStatus.War
            }
        }.let { if (it) 1 else 0 } else 0

        // Threat: distance-based proxy (0 = far/unknown, 4 = immediate danger)
        val threat = if (visible) {
            val dist = observer.threatManager.getDistanceToClosestEnemyUnit(tile, 5, false)
            when {
                dist <= 0 -> 4
                dist == 1 -> 3
                dist == 2 -> 2
                dist <= 4 -> 1
                else -> 0
            }
        } else 0

        data[0 * tileCount + i] = terrainIdx
        data[1 * tileCount + i] = featureIdx
        data[2 * tileCount + i] = resourceIdx
        data[3 * tileCount + i] = ownerIdx
        data[4 * tileCount + i] = visibilityCode
        data[5 * tileCount + i] = cityPresence
        data[6 * tileCount + i] = unitPresence
        data[7 * tileCount + i] = roadLevel
        data[8 * tileCount + i] = improvIdx
        data[9 * tileCount + i] = zoc
        data[10 * tileCount + i] = threat
    }

    return MapPlanes(
        channels = N_MAP_CHANNELS,
        height = tileCount,   // flattened: height = tileCount, width = 1
        width = 1,
        data = data.toList()
    )
}
