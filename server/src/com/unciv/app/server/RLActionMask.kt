package com.unciv.app.server

import com.unciv.logic.GameInfo
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.civilization.diplomacy.DiplomaticStatus
import com.unciv.models.ruleset.unique.GameContext
import com.unciv.models.ruleset.unique.UniqueType
import kotlinx.serialization.Serializable

/**
 * Per-action-dimension masks returned by GET /rl/action_mask/{gameId}.
 *
 * Every mask array has **true** for valid choices and **false** for illegal ones.
 * The 12 keys mirror the 12 keys in the Python action space exactly.
 */
@Serializable
data class RLActionMaskResponse(
    val gameId: String,
    val agentCivId: String,
    /** Which of the 10 macro actions are legal this turn. */
    val macroMask: List<Boolean>,
    /** Which entity indices are own moveable units (UNIT_MOVE / UNIT_ATTACK / UNIT_ABILITY). */
    val unitTargetMask: List<Boolean>,
    /** Which unit sub-actions are available for at least one unit. */
    val unitSubactionMask: List<Boolean>,
    /** Which entity indices are own cities (CITY_ACTION). */
    val cityTargetMask: List<Boolean>,
    /** Which city sub-actions are available for at least one city. */
    val citySubactionMask: List<Boolean>,
    /** Which production items are buildable in at least one city. */
    val productionTargetMask: List<Boolean>,
    /** Which tech indices are currently researchable (prereqs met, not yet researched). */
    val techTargetMask: List<Boolean>,
    /** Which policy indices are currently adoptable. */
    val policyTargetMask: List<Boolean>,
    /** Which agent indices (in allAgents) are valid diplomacy targets. */
    val diplomacyTargetMask: List<Boolean>,
    /** Which diplomacy sub-actions are available against at least one target. */
    val diplomacySubactionMask: List<Boolean>,
    /** Which tile indices are relevant (reachable / attackable / purchasable / buildable). */
    val tileTargetMask: List<Boolean>,
    /** Which improvement indices can be built by at least one worker. */
    val improvementTargetMask: List<Boolean>
)

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fun computeActionMask(
    gameInfo: GameInfo,
    agentCivId: String,
    allAgents: List<String>,
    entitySnapshot: List<EntitySnapshot>
): RLActionMaskResponse {

    val civ = gameInfo.getCivilization(agentCivId)
    val ruleset = gameInfo.ruleset
    val sortedTiles = gameInfo.tileMap.values.sortedBy { it.position.toString() }

    // ---- catalogues -------------------------------------------------------
    val techList = ruleset.technologies.keys.sorted()
    val policyList = ruleset.policies.keys.sorted()
    val productionList = (ruleset.buildings.keys + ruleset.units.keys).distinct().sorted()
    val improvList = ruleset.tileImprovements.keys.sorted()

    // ---- units this agent can act with ------------------------------------
    val actionableUnits = civ.units.getCivUnits()
        .filter { it.currentMovement > 0f || it.due }
        .toList()
    val workerUnits = actionableUnits.filter { it.baseUnit.hasUnique("Can build improvements on tiles") }
    val settlerUnits = actionableUnits.filter { it.baseUnit.hasUnique("Founds a new city") }
    val combatUnits = actionableUnits.filter { !it.baseUnit.isCivilian() }

    // ---- unit target mask ------------------------------------------------
    val unitTargetMask = BooleanArray(MAX_ENTITIES) { false }
    for (snap in entitySnapshot) {
        if (snap.entityType != 2) continue
        val unit = civ.units.getCivUnits().firstOrNull { u ->
            u.getTile().position == snap.position
        } ?: continue
        if (unit.currentMovement > 0f || unit.due)
            unitTargetMask[entitySnapshot.indexOf(snap)] = true
    }

    // ---- unit subaction mask (any of the actionable units) ---------------
    val unitSubactionMask = BooleanArray(7) { false }
    if (actionableUnits.isNotEmpty()) {
        unitSubactionMask[UnitSubAction.FORTIFY] = combatUnits.any { !it.isFortified() }
        unitSubactionMask[UnitSubAction.HEAL] = actionableUnits.any { it.health < 100 }
        unitSubactionMask[UnitSubAction.SKIP] = actionableUnits.isNotEmpty()
        unitSubactionMask[UnitSubAction.PROMOTE] = actionableUnits.any {
            it.promotions.getAvailablePromotions().any()
        }
        unitSubactionMask[UnitSubAction.PILLAGE] = combatUnits.any { it.canPillage() }
        unitSubactionMask[UnitSubAction.DISBAND] = actionableUnits.isNotEmpty()
        unitSubactionMask[UnitSubAction.FOUND_CITY] = settlerUnits.isNotEmpty()
    }

    // ---- city masks -------------------------------------------------------
    val cityTargetMask = BooleanArray(MAX_ENTITIES) { false }
    for (snap in entitySnapshot) {
        if (snap.entityType != 1) continue
        val city = civ.cities.firstOrNull { it.location == snap.position } ?: continue
        cityTargetMask[entitySnapshot.indexOf(snap)] = true
    }

    val citySubactionMask = BooleanArray(4) { false }
    if (civ.cities.isNotEmpty()) {
        citySubactionMask[CitySubAction.SET_PRODUCTION] = true
        citySubactionMask[CitySubAction.BUY_PRODUCTION] = civ.gold > 0
        citySubactionMask[CitySubAction.SELL_BUILDING] =
            civ.cities.any { it.cityConstructions.getBuiltBuildings().any() }
        citySubactionMask[CitySubAction.BUY_TILE] =
            civ.cities.any { city -> gameInfo.tileMap.values.any { city.expansion.canBuyTile(it) } }
    }

    // ---- production items ------------------------------------------------
    val productionTargetMask = BooleanArray(productionList.size) { false }
    for ((idx, name) in productionList.withIndex()) {
        if (civ.cities.any { it.cityConstructions.isQueueable(name) })
            productionTargetMask[idx] = true
    }

    // ---- tech mask -------------------------------------------------------
    val techTargetMask = BooleanArray(techList.size) { false }
    for ((idx, name) in techList.withIndex()) {
        val tech = ruleset.technologies[name] ?: continue
        if (!civ.tech.isResearched(name) &&
            tech.prerequisites.all { civ.tech.isResearched(it) })
            techTargetMask[idx] = true
    }

    // ---- policy mask -----------------------------------------------------
    val policyTargetMask = BooleanArray(policyList.size) { false }
    if (civ.policies.canAdoptPolicy()) {
        for ((idx, name) in policyList.withIndex()) {
            val policy = ruleset.policies[name] ?: continue
            if (civ.policies.isAdoptable(policy))
                policyTargetMask[idx] = true
        }
    }

    // ---- diplomacy masks -------------------------------------------------
    val diplomacyTargetMask = BooleanArray(allAgents.size) { false }
    for ((idx, id) in allAgents.withIndex()) {
        if (id != agentCivId && gameInfo.getCivilizationOrNull(id) != null)
            diplomacyTargetMask[idx] = true
    }

    val diplomacySubactionMask = BooleanArray(7) { false }
    val majorCivs = allAgents.filter { it != agentCivId }
        .mapNotNull { gameInfo.getCivilizationOrNull(it) }
    diplomacySubactionMask[DiplomacySubAction.DECLARE_WAR] =
        majorCivs.any { other ->
            civ.getDiplomacyManager(other)?.diplomaticStatus != DiplomaticStatus.War
        }
    diplomacySubactionMask[DiplomacySubAction.OFFER_PEACE] =
        majorCivs.any { other ->
            civ.getDiplomacyManager(other)?.diplomaticStatus == DiplomaticStatus.War
        }
    diplomacySubactionMask[DiplomacySubAction.OPEN_BORDERS] = majorCivs.isNotEmpty()
    diplomacySubactionMask[DiplomacySubAction.FRIENDSHIP] = majorCivs.isNotEmpty()
    diplomacySubactionMask[DiplomacySubAction.DENOUNCE] = majorCivs.isNotEmpty()
    diplomacySubactionMask[DiplomacySubAction.RESEARCH_AGREEMENT] = false
    diplomacySubactionMask[DiplomacySubAction.DEFENSIVE_PACT] = false

    // ---- tile target mask ------------------------------------------------
    val tileTargetMask = BooleanArray(maxOf(sortedTiles.size, MAX_TILES)) { false }
    // Reachable tiles for any unit
    for (unit in actionableUnits) {
        for (entry in unit.movement.getDistanceToTiles()) {
            val idx = sortedTiles.indexOf(entry.key)
            if (idx >= 0 && idx < tileTargetMask.size) tileTargetMask[idx] = true
        }
    }
    // Purchasable tiles for any city
    for (city in civ.cities) {
        for (tile in sortedTiles.indices) {
            val t = sortedTiles[tile]
            if (city.expansion.canBuyTile(t) && tile < tileTargetMask.size)
                tileTargetMask[tile] = true
        }
    }

    // ---- improvement target mask -----------------------------------------
    val improvementTargetMask = BooleanArray(improvList.size) { false }
    for ((idx, name) in improvList.withIndex()) {
        val impr = ruleset.tileImprovements[name] ?: continue
        if (workerUnits.any { worker ->
            gameInfo.tileMap.values.any { tile -> tile.canBuildImprovement(impr, civ) }
        }) improvementTargetMask[idx] = true
    }

    // ---- macro mask -------------------------------------------------------
    val macroMask = BooleanArray(10)
    macroMask[MacroAction.END_TURN] = true
    macroMask[MacroAction.UNIT_MOVE] = actionableUnits.any { !it.baseUnit.isCivilian() || it.baseUnit.isCivilian() }
    macroMask[MacroAction.UNIT_ATTACK] = combatUnits.isNotEmpty()
    macroMask[MacroAction.UNIT_ABILITY] = actionableUnits.isNotEmpty()
    macroMask[MacroAction.CITY_ACTION] = civ.cities.isNotEmpty()
    macroMask[MacroAction.TECH_RESEARCH] = techTargetMask.any { it }
    macroMask[MacroAction.POLICY_ADOPT] = policyTargetMask.any { it }
    macroMask[MacroAction.DIPLOMACY] = diplomacyTargetMask.any { it }
    macroMask[MacroAction.WORKER_BUILD] = workerUnits.isNotEmpty() && improvementTargetMask.any { it }
    macroMask[MacroAction.FOUNDER_SETTLE] = settlerUnits.isNotEmpty()

    return RLActionMaskResponse(
        gameId = gameInfo.gameId,
        agentCivId = agentCivId,
        macroMask = macroMask.toList(),
        unitTargetMask = unitTargetMask.toList(),
        unitSubactionMask = unitSubactionMask.toList(),
        cityTargetMask = cityTargetMask.toList(),
        citySubactionMask = citySubactionMask.toList(),
        productionTargetMask = productionTargetMask.toList(),
        techTargetMask = techTargetMask.toList(),
        policyTargetMask = policyTargetMask.toList(),
        diplomacyTargetMask = diplomacyTargetMask.toList(),
        diplomacySubactionMask = diplomacySubactionMask.toList(),
        tileTargetMask = tileTargetMask.toList(),
        improvementTargetMask = improvementTargetMask.toList()
    )
}
