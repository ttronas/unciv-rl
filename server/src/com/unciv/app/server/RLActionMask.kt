package com.unciv.app.server

import com.unciv.logic.GameInfo
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.civilization.PlayerType
import com.unciv.logic.civilization.diplomacy.DiplomaticStatus
import com.unciv.models.ruleset.unique.GameContext
import com.unciv.models.ruleset.unique.UniqueType
import kotlinx.serialization.Serializable

/**
 * Per-dimension legal-action masks returned by GET /rl/action_mask/{gameId}.
 * True = legal, False = illegal.  Mirrors the 12-key Python action space.
 */
@Serializable
data class RLActionMaskResponse(
    val gameId: String,
    val agentCivId: String,
    val macroMask: List<Boolean>,
    val unitTargetMask: List<Boolean>,
    val unitSubactionMask: List<Boolean>,
    val cityTargetMask: List<Boolean>,
    val citySubactionMask: List<Boolean>,
    val productionTargetMask: List<Boolean>,
    val techTargetMask: List<Boolean>,
    val policyTargetMask: List<Boolean>,
    val diplomacyTargetMask: List<Boolean>,
    val diplomacySubactionMask: List<Boolean>,
    val tileTargetMask: List<Boolean>,
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

    // ---- catalogues -------------------------------------------------------
    val techList = ruleset.technologies.keys.sorted()
    val policyList = ruleset.policies.keys.sorted()
    val productionList = (ruleset.buildings.keys + ruleset.units.keys).distinct().sorted()
    val improvList = ruleset.tileImprovements.keys.sorted()
    val allTiles = gameInfo.tileMap.values.toList()

    // ---- classify agent's units -------------------------------------------
    val actionableUnits = civ.units.getCivUnits()
        .filter { it.currentMovement > 0f || it.due }
        .toList()
    val combatUnits = actionableUnits.filter { !it.baseUnit.isCivilian() }
    val workerUnits = actionableUnits.filter { it.hasUnique(UniqueType.BuildImprovements) }
    val settlerUnits = actionableUnits.filter { it.baseUnit.isCityFounder() }

    // ---- unit target mask ------------------------------------------------
    // True at entity index i if snap[i] is an own unit that can still act
    val unitTargetMask = BooleanArray(MAX_ENTITIES) { false }
    for ((idx, snap) in entitySnapshot.withIndex()) {
        if (snap.entityType != 2) continue
        val u = snap.unit ?: continue
        if (u.civ == civ && (u.currentMovement > 0f || u.due))
            unitTargetMask[idx] = true
    }

    // ---- unit subaction mask ---------------------------------------------
    val unitSubactionMask = BooleanArray(7) { false }
    if (actionableUnits.isNotEmpty()) {
        unitSubactionMask[UnitSubAction.FORTIFY] = combatUnits.any { !it.isFortified() }
        unitSubactionMask[UnitSubAction.HEAL] = actionableUnits.any { it.health < 100 }
        unitSubactionMask[UnitSubAction.SKIP] = true
        unitSubactionMask[UnitSubAction.PROMOTE] =
            actionableUnits.any { it.promotions.getAvailablePromotions().any() }
        unitSubactionMask[UnitSubAction.PILLAGE] =
            combatUnits.any { it.getTile().canPillageTile() }
        unitSubactionMask[UnitSubAction.DISBAND] = true
        unitSubactionMask[UnitSubAction.FOUND_CITY] = settlerUnits.isNotEmpty()
    }

    // ---- city target mask ------------------------------------------------
    val cityTargetMask = BooleanArray(MAX_ENTITIES) { false }
    for ((idx, snap) in entitySnapshot.withIndex()) {
        if (snap.entityType != 1) continue
        val city = snap.city ?: continue
        if (city.civ == civ) cityTargetMask[idx] = true
    }

    // ---- city subaction mask ---------------------------------------------
    val citySubactionMask = BooleanArray(4) { false }
    if (civ.cities.isNotEmpty()) {
        citySubactionMask[CitySubAction.SET_PRODUCTION] = true
        citySubactionMask[CitySubAction.BUY_PRODUCTION] = civ.gold > 0
        citySubactionMask[CitySubAction.SELL_BUILDING] =
            civ.cities.any { it.cityConstructions.getBuiltBuildings().any() }
        citySubactionMask[CitySubAction.BUY_TILE] =
            civ.cities.any { city -> allTiles.any { city.expansion.canBuyTile(it) } }
    }

    // ---- production target mask -----------------------------------------
    val productionTargetMask = BooleanArray(productionList.size) { false }
    for ((idx, name) in productionList.withIndex()) {
        val construction = ruleset.buildings[name] ?: ruleset.units[name] ?: continue
        if (civ.cities.any { construction.isBuildable(it.cityConstructions) })
            productionTargetMask[idx] = true
    }

    // ---- tech target mask -----------------------------------------------
    val techTargetMask = BooleanArray(techList.size) { false }
    for ((idx, name) in techList.withIndex()) {
        val tech = ruleset.technologies[name] ?: continue
        if (!civ.tech.isResearched(name) &&
            tech.prerequisites.all { civ.tech.isResearched(it) })
            techTargetMask[idx] = true
    }

    // ---- policy target mask ---------------------------------------------
    val policyTargetMask = BooleanArray(policyList.size) { false }
    if (civ.policies.canAdoptPolicy()) {
        for ((idx, name) in policyList.withIndex()) {
            val policy = ruleset.policies[name] ?: continue
            if (civ.policies.isAdoptable(policy)) policyTargetMask[idx] = true
        }
    }

    // ---- diplomacy target mask ------------------------------------------
    val diplomacyTargetMask = BooleanArray(allAgents.size) { false }
    for ((idx, id) in allAgents.withIndex()) {
        if (id != agentCivId && gameInfo.getCivilizationOrNull(id) != null)
            diplomacyTargetMask[idx] = true
    }

    val diplomacySubactionMask = BooleanArray(7) { false }
    val peers = allAgents.filter { it != agentCivId }
        .mapNotNull { gameInfo.getCivilizationOrNull(it) }
    diplomacySubactionMask[DiplomacySubAction.DECLARE_WAR] =
        peers.any { civ.getDiplomacyManager(it)?.diplomaticStatus != DiplomaticStatus.War }
    diplomacySubactionMask[DiplomacySubAction.OFFER_PEACE] =
        peers.any { civ.getDiplomacyManager(it)?.diplomaticStatus == DiplomaticStatus.War }
    diplomacySubactionMask[DiplomacySubAction.OPEN_BORDERS] = peers.isNotEmpty()
    diplomacySubactionMask[DiplomacySubAction.FRIENDSHIP] = peers.isNotEmpty()
    diplomacySubactionMask[DiplomacySubAction.DENOUNCE] = peers.isNotEmpty()
    diplomacySubactionMask[DiplomacySubAction.RESEARCH_AGREEMENT] = false
    diplomacySubactionMask[DiplomacySubAction.DEFENSIVE_PACT] = false

    // ---- tile target mask -----------------------------------------------
    val tileTargetMask = BooleanArray(maxOf(allTiles.size, MAX_TILES)) { false }
    for (unit in actionableUnits) {
        for (tile in unit.movement.getDistanceToTiles().keys) {
            if (tile.zeroBasedIndex < tileTargetMask.size) tileTargetMask[tile.zeroBasedIndex] = true
        }
    }
    for (city in civ.cities) {
        for (tile in allTiles) {
            if (city.expansion.canBuyTile(tile) && tile.zeroBasedIndex < tileTargetMask.size)
                tileTargetMask[tile.zeroBasedIndex] = true
        }
    }

    // ---- improvement target mask ----------------------------------------
    val improvementTargetMask = BooleanArray(improvList.size) { false }
    if (workerUnits.isNotEmpty()) {
        for ((idx, name) in improvList.withIndex()) {
            val impr = ruleset.tileImprovements[name] ?: continue
            if (workerUnits.any { worker ->
                    val gc = GameContext(unit = worker)
                    allTiles.any { tile ->
                        tile.improvementFunctions.canBuildImprovement(impr, gc)
                    }
                }) improvementTargetMask[idx] = true
        }
    }

    // ---- macro mask -----------------------------------------------------
    val macroMask = BooleanArray(10)
    macroMask[MacroAction.END_TURN] = true
    macroMask[MacroAction.UNIT_MOVE] = actionableUnits.isNotEmpty()
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
