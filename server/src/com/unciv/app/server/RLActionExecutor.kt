package com.unciv.app.server

import com.unciv.logic.GameInfo
import com.unciv.logic.battle.Battle
import com.unciv.logic.battle.MapUnitCombatant
import com.unciv.logic.battle.TargetHelper
import com.unciv.logic.city.City
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.civilization.diplomacy.DeclareWarReason
import com.unciv.logic.civilization.diplomacy.WarType
import com.unciv.logic.map.mapunit.MapUnit
import kotlinx.serialization.Serializable

// ---------------------------------------------------------------------------
// Macro action codes
// ---------------------------------------------------------------------------

object MacroAction {
    const val END_TURN = 0
    /** Move unit to a tile: unitTarget + tileTarget */
    const val UNIT_MOVE = 1
    /** Attack a tile: unitTarget + tileTarget */
    const val UNIT_ATTACK = 2
    /** Non-move unit ability: unitTarget + unitSubaction */
    const val UNIT_ABILITY = 3
    /** City management: cityTarget + citySubaction + productionTarget / tileTarget */
    const val CITY_ACTION = 4
    /** Research a technology: techTarget */
    const val TECH_RESEARCH = 5
    /** Adopt a social policy: policyTarget */
    const val POLICY_ADOPT = 6
    /** Diplomatic action: diplomacyTarget + diplomacySubaction */
    const val DIPLOMACY = 7
    /** Build improvement: unitTarget (worker) + tileTarget + improvementTarget */
    const val WORKER_BUILD = 8
    /** Found city at settler's current tile: unitTarget */
    const val FOUNDER_SETTLE = 9
}

// ---------------------------------------------------------------------------
// Sub-action codes
// ---------------------------------------------------------------------------

object UnitSubAction {
    const val FORTIFY = 0
    const val HEAL = 1
    const val SKIP = 2
    const val PROMOTE = 3
    const val PILLAGE = 4
    const val DISBAND = 5
    const val FOUND_CITY = 6
}

object CitySubAction {
    const val SET_PRODUCTION = 0
    const val BUY_PRODUCTION = 1
    const val SELL_BUILDING = 2
    const val BUY_TILE = 3
}

object DiplomacySubAction {
    const val DECLARE_WAR = 0
    const val OFFER_PEACE = 1
    const val OPEN_BORDERS = 2
    const val FRIENDSHIP = 3
    const val DENOUNCE = 4
    const val RESEARCH_AGREEMENT = 5
    const val DEFENSIVE_PACT = 6
}

// ---------------------------------------------------------------------------
// Action DTO
// ---------------------------------------------------------------------------

/**
 * Semantic 12-field action submitted to POST /rl/action/{gameId}.
 *
 * | macro            | relevant fields                                         |
 * |------------------|---------------------------------------------------------|
 * | 0 END_TURN       | —                                                       |
 * | 1 UNIT_MOVE      | unitTarget, tileTarget                                  |
 * | 2 UNIT_ATTACK    | unitTarget, tileTarget                                  |
 * | 3 UNIT_ABILITY   | unitTarget, unitSubaction                               |
 * | 4 CITY_ACTION    | cityTarget, citySubaction, productionTarget / tileTarget|
 * | 5 TECH_RESEARCH  | techTarget                                              |
 * | 6 POLICY_ADOPT   | policyTarget                                            |
 * | 7 DIPLOMACY      | diplomacyTarget, diplomacySubaction                     |
 * | 8 WORKER_BUILD   | unitTarget (worker), tileTarget, improvementTarget      |
 * | 9 FOUNDER_SETTLE | unitTarget (settler)                                    |
 */
@Serializable
data class RLAction(
    val macro: Int,
    val unitTarget: Int = 0,
    val unitSubaction: Int = 0,
    val cityTarget: Int = 0,
    val citySubaction: Int = 0,
    val productionTarget: Int = 0,
    val techTarget: Int = 0,
    val policyTarget: Int = 0,
    val diplomacyTarget: Int = 0,
    val diplomacySubaction: Int = 0,
    val tileTarget: Int = 0,
    val improvementTarget: Int = 0
)

@Serializable
data class ActionResult(
    val success: Boolean,
    val message: String = "",
    /** Delta in total score since the last step for the acting agent. */
    val reward: Float = 0f
)

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/**
 * Routes an [RLAction] to Unciv game-logic.
 * Returns [ActionResult]; the caller handles turn advancement for END_TURN.
 */
fun executeAction(
    gameInfo: GameInfo,
    agentCivId: String,
    action: RLAction,
    allAgents: List<String>,
    entitySnapshot: List<EntitySnapshot>,
    previousScore: Int
): ActionResult {

    val civ = gameInfo.getCivilization(agentCivId)
    val ruleset = gameInfo.ruleset

    val result = when (action.macro) {
        MacroAction.END_TURN -> ActionResult(success = true, message = "end_turn")

        MacroAction.UNIT_MOVE -> executeUnitMove(civ, action, entitySnapshot, gameInfo)

        MacroAction.UNIT_ATTACK -> executeUnitAttack(civ, action, entitySnapshot, gameInfo)

        MacroAction.UNIT_ABILITY -> executeUnitAbility(civ, action, entitySnapshot, ruleset)

        MacroAction.CITY_ACTION -> executeCityAction(civ, action, entitySnapshot, gameInfo, ruleset)

        MacroAction.TECH_RESEARCH -> {
            val techList = ruleset.technologies.keys.sorted()
            if (action.techTarget !in techList.indices)
                return ActionResult(false, "techTarget ${action.techTarget} out of range")
            val techName = techList[action.techTarget]
            if (civ.tech.isResearched(techName))
                return ActionResult(false, "Tech $techName already researched")
            civ.tech.techsToResearch.remove(techName)
            civ.tech.techsToResearch.add(0, techName)
            ActionResult(true, "Researching $techName")
        }

        MacroAction.POLICY_ADOPT -> {
            val policyList = ruleset.policies.keys.sorted()
            if (action.policyTarget !in policyList.indices)
                return ActionResult(false, "policyTarget ${action.policyTarget} out of range")
            val policyName = policyList[action.policyTarget]
            val policy = ruleset.policies[policyName]
                ?: return ActionResult(false, "Policy $policyName not found")
            if (!civ.policies.isAdoptable(policy))
                return ActionResult(false, "Policy $policyName not adoptable")
            civ.policies.adopt(policy)
            ActionResult(true, "Adopted $policyName")
        }

        MacroAction.DIPLOMACY -> executeDiplomacy(civ, action, allAgents, gameInfo)

        MacroAction.WORKER_BUILD -> executeWorkerBuild(civ, action, entitySnapshot, gameInfo, ruleset)

        MacroAction.FOUNDER_SETTLE -> {
            val unit = entitySnapshot.getOrNull(action.unitTarget)?.let {
                findUnit(civ, it)
            } ?: return ActionResult(false, "No settler unit at index ${action.unitTarget}")
            if (!unit.baseUnit.hasUnique("Founds a new city")) // settler check
                return ActionResult(false, "Unit at ${action.unitTarget} cannot found a city")
            try {
                unit.foundCity()
                ActionResult(true, "City founded")
            } catch (e: Exception) {
                ActionResult(false, "Settle failed: ${e.message}")
            }
        }

        else -> ActionResult(false, "Unknown macro action: ${action.macro}")
    }

    val newScore = civ.calculateTotalScore().toInt()
    val reward = (newScore - previousScore).toFloat()
    return result.copy(reward = reward)
}

// ---------------------------------------------------------------------------
// Sub-executors
// ---------------------------------------------------------------------------

private fun executeUnitMove(
    civ: Civilization,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>,
    gameInfo: GameInfo
): ActionResult {
    val unit = entitySnapshot.getOrNull(action.unitTarget)?.let { findUnit(civ, it) }
        ?: return ActionResult(false, "No unit at entity index ${action.unitTarget}")
    val tiles = gameInfo.tileMap.values.sortedBy { it.position.toString() }
    val tile = tiles.getOrNull(action.tileTarget)
        ?: return ActionResult(false, "tileTarget ${action.tileTarget} out of range")
    return try {
        unit.movement.moveToTile(tile)
        ActionResult(true, "Unit moved to (${tile.position.x},${tile.position.y})")
    } catch (e: Exception) {
        ActionResult(false, "Move failed: ${e.message}")
    }
}

private fun executeUnitAttack(
    civ: Civilization,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>,
    gameInfo: GameInfo
): ActionResult {
    val unit = entitySnapshot.getOrNull(action.unitTarget)?.let { findUnit(civ, it) }
        ?: return ActionResult(false, "No unit at entity index ${action.unitTarget}")
    val tiles = gameInfo.tileMap.values.sortedBy { it.position.toString() }
    val tile = tiles.getOrNull(action.tileTarget)
        ?: return ActionResult(false, "tileTarget ${action.tileTarget} out of range")
    val attackable = TargetHelper.getAttackableEnemies(unit, unit.movement.getDistanceToTiles())
        .firstOrNull { it.tileToAttack == tile }
        ?: return ActionResult(false, "Tile not attackable")
    Battle.attack(MapUnitCombatant(unit), attackable)
    return ActionResult(true, "Attacked tile (${tile.position.x},${tile.position.y})")
}

private fun executeUnitAbility(
    civ: Civilization,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>,
    ruleset: com.unciv.models.ruleset.Ruleset
): ActionResult {
    val unit = entitySnapshot.getOrNull(action.unitTarget)?.let { findUnit(civ, it) }
        ?: return ActionResult(false, "No unit at entity index ${action.unitTarget}")
    return when (action.unitSubaction) {
        UnitSubAction.FORTIFY -> {
            unit.fortify()
            ActionResult(true, "Unit fortified")
        }
        UnitSubAction.HEAL -> {
            unit.due = false
            unit.currentMovement = 0f
            ActionResult(true, "Unit healing")
        }
        UnitSubAction.SKIP -> {
            unit.due = false
            ActionResult(true, "Unit skipped")
        }
        UnitSubAction.PROMOTE -> {
            val promotion = unit.promotions.getAvailablePromotions().firstOrNull()
                ?: return ActionResult(false, "No promotions available")
            unit.promotions.addPromotion(promotion.name)
            ActionResult(true, "Promoted: ${promotion.name}")
        }
        UnitSubAction.PILLAGE -> {
            if (!unit.canPillage())
                return ActionResult(false, "Cannot pillage here")
            unit.pillageCurrentTile()
            ActionResult(true, "Tile pillaged")
        }
        UnitSubAction.DISBAND -> {
            unit.destroy()
            ActionResult(true, "Unit disbanded")
        }
        UnitSubAction.FOUND_CITY -> {
            if (!unit.baseUnit.hasUnique("Founds a new city"))
                return ActionResult(false, "Unit cannot found a city")
            unit.foundCity()
            ActionResult(true, "City founded")
        }
        else -> ActionResult(false, "Unknown unit subaction: ${action.unitSubaction}")
    }
}

private fun executeCityAction(
    civ: Civilization,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>,
    gameInfo: GameInfo,
    ruleset: com.unciv.models.ruleset.Ruleset
): ActionResult {
    val city = entitySnapshot.getOrNull(action.cityTarget)?.let { snap ->
        civ.cities.firstOrNull { it.location == snap.position }
    } ?: return ActionResult(false, "No city at entity index ${action.cityTarget}")

    val productionList = (ruleset.buildings.keys + ruleset.units.keys).distinct().sorted()

    return when (action.citySubaction) {
        CitySubAction.SET_PRODUCTION -> {
            if (action.productionTarget !in productionList.indices)
                return ActionResult(false, "productionTarget out of range")
            val itemName = productionList[action.productionTarget]
            if (!city.cityConstructions.isQueueable(itemName))
                return ActionResult(false, "$itemName not buildable in this city")
            city.cityConstructions.addToQueue(itemName)
            ActionResult(true, "Production set to $itemName")
        }
        CitySubAction.BUY_PRODUCTION -> {
            if (action.productionTarget !in productionList.indices)
                return ActionResult(false, "productionTarget out of range")
            val itemName = productionList[action.productionTarget]
            val stat = com.unciv.models.stats.Stat.Gold
            if (!city.cityConstructions.isQueueable(itemName))
                return ActionResult(false, "$itemName not buyable")
            val cost = city.cityConstructions.getRemainingWork(itemName)
            if (civ.gold < cost)
                return ActionResult(false, "Not enough gold (need $cost, have ${civ.gold})")
            city.cityConstructions.purchaseConstruction(itemName, 0, true)
            ActionResult(true, "Bought $itemName")
        }
        CitySubAction.SELL_BUILDING -> {
            if (action.productionTarget !in productionList.indices)
                return ActionResult(false, "productionTarget out of range")
            val buildingName = productionList[action.productionTarget]
            if (buildingName !in city.cityConstructions.getBuiltBuildings().map { it.name })
                return ActionResult(false, "$buildingName not built in this city")
            city.cityConstructions.sellBuilding(buildingName)
            ActionResult(true, "Sold $buildingName")
        }
        CitySubAction.BUY_TILE -> {
            val tiles = gameInfo.tileMap.values.sortedBy { it.position.toString() }
            val tile = tiles.getOrNull(action.tileTarget)
                ?: return ActionResult(false, "tileTarget out of range")
            if (!city.expansion.canBuyTile(tile))
                return ActionResult(false, "Cannot buy tile at (${tile.position.x},${tile.position.y})")
            city.expansion.buyTile(tile)
            ActionResult(true, "Tile purchased")
        }
        else -> ActionResult(false, "Unknown city subaction: ${action.citySubaction}")
    }
}

private fun executeDiplomacy(
    civ: Civilization,
    action: RLAction,
    allAgents: List<String>,
    gameInfo: GameInfo
): ActionResult {
    if (action.diplomacyTarget !in allAgents.indices)
        return ActionResult(false, "diplomacyTarget ${action.diplomacyTarget} out of range")
    val targetCivId = allAgents[action.diplomacyTarget]
    if (targetCivId == civ.civID)
        return ActionResult(false, "Cannot target self in diplomacy")
    val targetCiv = gameInfo.getCivilizationOrNull(targetCivId)
        ?: return ActionResult(false, "Target civ $targetCivId not found")
    val dm = civ.getDiplomacyManager(targetCiv)
        ?: return ActionResult(false, "No diplomacy manager for $targetCivId")

    return when (action.diplomacySubaction) {
        DiplomacySubAction.DECLARE_WAR -> {
            if (dm.diplomaticStatus == com.unciv.logic.civilization.diplomacy.DiplomaticStatus.War)
                return ActionResult(false, "Already at war with $targetCivId")
            civ.getDiplomacyManager(targetCiv)!!.declareWar(DeclareWarReason(WarType.DirectWar, targetCiv))
            ActionResult(true, "Declared war on $targetCivId")
        }
        DiplomacySubAction.OFFER_PEACE -> {
            if (dm.diplomaticStatus != com.unciv.logic.civilization.diplomacy.DiplomaticStatus.War)
                return ActionResult(false, "Not at war with $targetCivId")
            // Simple white peace via negotiation — signal intent; full trade UI not available headless
            ActionResult(false, "Peace negotiations require trade UI (not implemented headless)")
        }
        DiplomacySubAction.OPEN_BORDERS -> {
            dm.signOpenBorders()
            ActionResult(true, "Signed open borders with $targetCivId")
        }
        DiplomacySubAction.FRIENDSHIP -> {
            dm.signDeclarationOfFriendship()
            ActionResult(true, "Signed declaration of friendship with $targetCivId")
        }
        DiplomacySubAction.DENOUNCE -> {
            dm.denounce()
            ActionResult(true, "Denounced $targetCivId")
        }
        DiplomacySubAction.RESEARCH_AGREEMENT -> {
            ActionResult(false, "Research agreement not yet implemented")
        }
        DiplomacySubAction.DEFENSIVE_PACT -> {
            ActionResult(false, "Defensive pact not yet implemented")
        }
        else -> ActionResult(false, "Unknown diplomacy subaction: ${action.diplomacySubaction}")
    }
}

private fun executeWorkerBuild(
    civ: Civilization,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>,
    gameInfo: GameInfo,
    ruleset: com.unciv.models.ruleset.Ruleset
): ActionResult {
    val worker = entitySnapshot.getOrNull(action.unitTarget)?.let { findUnit(civ, it) }
        ?: return ActionResult(false, "No worker at entity index ${action.unitTarget}")
    val tiles = gameInfo.tileMap.values.sortedBy { it.position.toString() }
    val tile = tiles.getOrNull(action.tileTarget)
        ?: return ActionResult(false, "tileTarget ${action.tileTarget} out of range")
    val improvList = ruleset.tileImprovements.keys.sorted()
    if (action.improvementTarget !in improvList.indices)
        return ActionResult(false, "improvementTarget out of range")
    val improvName = improvList[action.improvementTarget]
    val improvement = ruleset.tileImprovements[improvName]
        ?: return ActionResult(false, "Improvement $improvName not found")
    if (!tile.canBuildImprovement(improvement, civ))
        return ActionResult(false, "Cannot build $improvName on this tile")
    tile.startWorkingOnImprovement(improvement, civ, worker)
    return ActionResult(true, "Worker building $improvName")
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

private fun findUnit(civ: Civilization, snap: EntitySnapshot): MapUnit? =
    civ.units.getCivUnits().firstOrNull { u ->
        u.getTile().position == snap.position && snap.entityType == 2
    }
