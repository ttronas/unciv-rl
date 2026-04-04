package com.unciv.app.server

import com.unciv.logic.GameInfo
import com.unciv.logic.battle.Battle
import com.unciv.logic.battle.MapUnitCombatant
import com.unciv.logic.battle.TargetHelper
import com.unciv.logic.city.City
import com.unciv.logic.city.managers.CityFounder
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.civilization.PlayerType
import com.unciv.logic.civilization.diplomacy.DeclareWarReason
import com.unciv.logic.civilization.diplomacy.WarType
import com.unciv.logic.map.mapunit.MapUnit
import com.unciv.models.ruleset.unique.UniqueType
import kotlinx.serialization.Serializable

// ---------------------------------------------------------------------------
// Macro action codes
// ---------------------------------------------------------------------------

object MacroAction {
    const val END_TURN = 0
    /** Move unit to tile: unitTarget + tileTarget */
    const val UNIT_MOVE = 1
    /** Attack tile with unit: unitTarget + tileTarget */
    const val UNIT_ATTACK = 2
    /** Non-move unit ability: unitTarget + unitSubaction */
    const val UNIT_ABILITY = 3
    /** City management: cityTarget + citySubaction + productionTarget/tileTarget */
    const val CITY_ACTION = 4
    /** Research a technology: techTarget */
    const val TECH_RESEARCH = 5
    /** Adopt a social policy: policyTarget */
    const val POLICY_ADOPT = 6
    /** Diplomatic action: diplomacyTarget + diplomacySubaction */
    const val DIPLOMACY = 7
    /** Build tile improvement: unitTarget (worker) + tileTarget + improvementTarget */
    const val WORKER_BUILD = 8
    /** Settler founds city at current location: unitTarget */
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
// Action DTO (12 semantic fields)
// ---------------------------------------------------------------------------

/**
 * Semantic 12-field action submitted to POST /rl/action/{gameId}.
 *
 * | macro            | relevant fields                                           |
 * |------------------|-----------------------------------------------------------|
 * | 0 END_TURN       | —                                                         |
 * | 1 UNIT_MOVE      | unitTarget, tileTarget                                    |
 * | 2 UNIT_ATTACK    | unitTarget, tileTarget                                    |
 * | 3 UNIT_ABILITY   | unitTarget, unitSubaction                                 |
 * | 4 CITY_ACTION    | cityTarget, citySubaction, productionTarget / tileTarget  |
 * | 5 TECH_RESEARCH  | techTarget                                                |
 * | 6 POLICY_ADOPT   | policyTarget                                              |
 * | 7 DIPLOMACY      | diplomacyTarget, diplomacySubaction                       |
 * | 8 WORKER_BUILD   | unitTarget (worker), tileTarget, improvementTarget        |
 * | 9 FOUNDER_SETTLE | unitTarget (settler)                                      |
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
    val reward: Float = 0f
)

// ---------------------------------------------------------------------------
// Entity snapshot
// ---------------------------------------------------------------------------

/**
 * Lightweight snapshot mapping entity-list index → game object.
 * Units appear before cities, matching the observation builder order.
 */
data class EntitySnapshot(
    /** 0=padding, 1=city, 2=unit */
    val entityType: Int,
    val unit: MapUnit?,
    val city: City?
)

/** Build entity snapshots in the same order as the observation entity list. */
fun buildEntitySnapshot(gameInfo: GameInfo): List<EntitySnapshot> {
    val result = mutableListOf<EntitySnapshot>()
    val padding = EntitySnapshot(0, null, null)

    for (tile in gameInfo.tileMap.values) {
        for (unit in tile.getUnits()) {
            result += EntitySnapshot(2, unit, null)
        }
    }
    for (gameCiv in gameInfo.civilizations) {
        if (gameCiv.isDefeated()) continue
        for (city in gameCiv.cities) {
            result += EntitySnapshot(1, null, city)
        }
    }

    while (result.size < MAX_ENTITIES) result += padding
    return result.take(MAX_ENTITIES)
}

// ---------------------------------------------------------------------------
// Main executor
// ---------------------------------------------------------------------------

/**
 * Routes [action] to the appropriate game-logic handler.
 * Returns [ActionResult]; turn advancement is handled by the caller.
 */
fun executeAction(
    gameInfo: GameInfo,
    agentCivId: String,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val civ = gameInfo.getCivilization(agentCivId)

    return try {
        when (action.macro) {
            MacroAction.END_TURN -> ActionResult(success = true, message = "end_turn")
            MacroAction.UNIT_MOVE -> executeUnitMove(civ, gameInfo, action, entitySnapshot)
            MacroAction.UNIT_ATTACK -> executeUnitAttack(civ, gameInfo, action, entitySnapshot)
            MacroAction.UNIT_ABILITY -> executeUnitAbility(civ, action, entitySnapshot)
            MacroAction.CITY_ACTION -> executeCityAction(civ, gameInfo, action, entitySnapshot)
            MacroAction.TECH_RESEARCH -> executeTechResearch(civ, gameInfo, action)
            MacroAction.POLICY_ADOPT -> executePolicyAdopt(civ, gameInfo, action)
            MacroAction.DIPLOMACY -> executeDiplomacy(civ, gameInfo, action)
            MacroAction.WORKER_BUILD -> executeWorkerBuild(civ, gameInfo, action, entitySnapshot)
            MacroAction.FOUNDER_SETTLE -> executeFounderSettle(civ, gameInfo, action, entitySnapshot)
            else -> ActionResult(false, "Unknown macro action: ${action.macro}")
        }
    } catch (e: Exception) {
        ActionResult(false, e.message ?: "Unknown error")
    }
}

// ---------------------------------------------------------------------------
// Sub-executors
// ---------------------------------------------------------------------------

private fun executeUnitMove(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val unit = resolveUnit(civ, entitySnapshot, action.unitTarget)
        ?: return ActionResult(false, "No own unit at entity index ${action.unitTarget}")
    val destTile = gameInfo.tileMap.values.getOrNull(action.tileTarget)
        ?: return ActionResult(false, "Tile index ${action.tileTarget} out of range")
    if (!unit.movement.canReachInCurrentTurn(destTile))
        return ActionResult(false, "Unit cannot reach tile in current turn")
    unit.movement.moveToTile(destTile)
    return ActionResult(true, "Unit moved to (${destTile.position.x},${destTile.position.y})")
}

private fun executeUnitAttack(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val unit = resolveUnit(civ, entitySnapshot, action.unitTarget)
        ?: return ActionResult(false, "No own unit at entity index ${action.unitTarget}")
    val targetTile = gameInfo.tileMap.values.getOrNull(action.tileTarget)
        ?: return ActionResult(false, "Tile index ${action.tileTarget} out of range")
    val distToTiles = unit.movement.getDistanceToTiles()
    val attackable = TargetHelper.getAttackableEnemies(unit, distToTiles)
        .firstOrNull { it.tileToAttack == targetTile }
        ?: return ActionResult(false, "Tile is not attackable")
    Battle.moveAndAttack(MapUnitCombatant(unit), attackable)
    return ActionResult(true, "Attacked tile (${targetTile.position.x},${targetTile.position.y})")
}

private fun executeUnitAbility(
    civ: Civilization,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val unit = resolveUnit(civ, entitySnapshot, action.unitTarget)
        ?: return ActionResult(false, "No own unit at entity index ${action.unitTarget}")
    return when (action.unitSubaction) {
        UnitSubAction.FORTIFY -> {
            unit.fortifyIfCan()
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
            val available = unit.promotions.getAvailablePromotions().toList()
            if (available.isEmpty()) return ActionResult(false, "No promotions available")
            if (!unit.promotions.canBePromoted()) return ActionResult(false, "Unit cannot be promoted")
            unit.promotions.addPromotion(available[0].name)
            ActionResult(true, "Promoted: ${available[0].name}")
        }
        UnitSubAction.PILLAGE -> {
            val tile = unit.getTile()
            if (!tile.canPillageTile()) return ActionResult(false, "Cannot pillage this tile")
            tile.setPillaged()
            unit.currentMovement = 0f
            ActionResult(true, "Tile pillaged")
        }
        UnitSubAction.DISBAND -> {
            unit.disband()
            ActionResult(true, "Unit disbanded")
        }
        UnitSubAction.FOUND_CITY -> {
            if (!unit.baseUnit.isCityFounder())
                return ActionResult(false, "Unit is not a settler")
            val location = unit.getTile().position
            CityFounder().foundCity(civ, location, unit)
            unit.destroy()
            ActionResult(true, "City founded")
        }
        else -> ActionResult(false, "Unknown unit subaction: ${action.unitSubaction}")
    }
}

private fun executeCityAction(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val snap = entitySnapshot.getOrNull(action.cityTarget)
        ?: return ActionResult(false, "Entity index ${action.cityTarget} out of range")
    val city = snap.city ?: return ActionResult(false, "Entity ${action.cityTarget} is not a city")
    if (city.civ != civ) return ActionResult(false, "City does not belong to agent")

    val ruleset = gameInfo.ruleset
    val productionList = (ruleset.buildings.keys + ruleset.units.keys).distinct().sorted()

    return when (action.citySubaction) {
        CitySubAction.SET_PRODUCTION -> {
            val itemName = productionList.getOrNull(action.productionTarget)
                ?: return ActionResult(false, "productionTarget ${action.productionTarget} out of range")
            val construction = ruleset.buildings[itemName] ?: ruleset.units[itemName]
                ?: return ActionResult(false, "'$itemName' not found in ruleset")
            if (!construction.isBuildable(city.cityConstructions))
                return ActionResult(false, "'$itemName' not buildable in this city")
            city.cityConstructions.constructionQueue.clear()
            city.cityConstructions.addToQueue(itemName)
            ActionResult(true, "Production set to $itemName")
        }
        CitySubAction.BUY_PRODUCTION -> {
            val itemName = productionList.getOrNull(action.productionTarget)
                ?: return ActionResult(false, "productionTarget ${action.productionTarget} out of range")
            val purchasable = (ruleset.buildings[itemName] ?: ruleset.units[itemName])
                as? com.unciv.models.ruleset.INonPerpetualConstruction
                ?: return ActionResult(false, "'$itemName' not purchasable")
            if (!purchasable.canBePurchasedWithStat(city, com.unciv.models.stats.Stat.Gold))
                return ActionResult(false, "Cannot purchase $itemName with gold")
            val cost = purchasable.getStatBuyCost(city, com.unciv.models.stats.Stat.Gold)
                ?: return ActionResult(false, "No gold cost for $itemName")
            if (civ.gold < cost)
                return ActionResult(false, "Insufficient gold ($cost required, ${civ.gold} available)")
            city.cityConstructions.purchaseConstruction(itemName, 0, false)
            ActionResult(true, "Purchased $itemName for $cost gold")
        }
        CitySubAction.SELL_BUILDING -> {
            val itemName = productionList.getOrNull(action.productionTarget)
                ?: return ActionResult(false, "productionTarget ${action.productionTarget} out of range")
            val building = ruleset.buildings[itemName]
                ?: return ActionResult(false, "'$itemName' is not a building")
            if (!city.cityConstructions.isBuilt(itemName))
                return ActionResult(false, "$itemName not built in this city")
            if (city.hasSoldBuildingThisTurn)
                return ActionResult(false, "Already sold a building this turn")
            city.cityConstructions.removeBuilding(itemName)
            val sellGold = (building.cost / 10).coerceAtLeast(1)
            civ.addGold(sellGold)
            city.hasSoldBuildingThisTurn = true
            ActionResult(true, "Sold $itemName for $sellGold gold")
        }
        CitySubAction.BUY_TILE -> {
            val tile = gameInfo.tileMap.values.getOrNull(action.tileTarget)
                ?: return ActionResult(false, "tileTarget ${action.tileTarget} out of range")
            if (!city.expansion.canBuyTile(tile))
                return ActionResult(false, "Cannot buy tile at (${tile.position.x},${tile.position.y})")
            city.expansion.buyTile(tile)
            ActionResult(true, "Tile purchased")
        }
        else -> ActionResult(false, "Unknown city subaction: ${action.citySubaction}")
    }
}

private fun executeTechResearch(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction
): ActionResult {
    val techList = gameInfo.ruleset.technologies.keys.sorted()
    val techName = techList.getOrNull(action.techTarget)
        ?: return ActionResult(false, "techTarget ${action.techTarget} out of range")
    if (!civ.tech.canBeResearched(techName))
        return ActionResult(false, "'$techName' cannot be researched right now")
    civ.tech.techsToResearch.remove(techName)
    civ.tech.techsToResearch.add(0, techName)
    return ActionResult(true, "Now researching $techName")
}

private fun executePolicyAdopt(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction
): ActionResult {
    val policyList = gameInfo.ruleset.policies.keys.sorted()
    val policyName = policyList.getOrNull(action.policyTarget)
        ?: return ActionResult(false, "policyTarget ${action.policyTarget} out of range")
    val policy = gameInfo.ruleset.policies[policyName]
        ?: return ActionResult(false, "Policy '$policyName' not in ruleset")
    if (!civ.policies.isAdoptable(policy))
        return ActionResult(false, "'$policyName' is not adoptable right now")
    if (!civ.policies.canAdoptPolicy())
        return ActionResult(false, "Cannot adopt policy: no free policies or insufficient culture")
    civ.policies.adopt(policy)
    return ActionResult(true, "Adopted policy $policyName")
}

private fun executeDiplomacy(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction
): ActionResult {
    // Derive agent list from all human players (same order as observation)
    val allAgents = gameInfo.civilizations
        .filter { it.playerType == PlayerType.Human && !it.isSpectator() }
        .map { it.civID }
    val targetCivId = allAgents.getOrNull(action.diplomacyTarget)
        ?: gameInfo.civilizations.filter { it.isMajorCiv() }.getOrNull(action.diplomacyTarget)?.civID
        ?: return ActionResult(false, "diplomacyTarget ${action.diplomacyTarget} out of range")
    if (targetCivId == civ.civID)
        return ActionResult(false, "Cannot target self in diplomacy")
    val targetCiv = gameInfo.getCivilizationOrNull(targetCivId)
        ?: return ActionResult(false, "Target civ '$targetCivId' not found")
    val dm = civ.getDiplomacyManagerOrMeet(targetCiv)

    return when (action.diplomacySubaction) {
        DiplomacySubAction.DECLARE_WAR -> {
            if (dm.diplomaticStatus == com.unciv.logic.civilization.diplomacy.DiplomaticStatus.War)
                return ActionResult(false, "Already at war with ${targetCiv.civName}")
            dm.declareWar(DeclareWarReason(WarType.DirectWar))
            ActionResult(true, "Declared war on ${targetCiv.civName}")
        }
        DiplomacySubAction.OFFER_PEACE -> {
            if (dm.diplomaticStatus != com.unciv.logic.civilization.diplomacy.DiplomaticStatus.War)
                return ActionResult(false, "Not at war with ${targetCiv.civName}")
            dm.makePeace()
            ActionResult(true, "Made peace with ${targetCiv.civName}")
        }
        DiplomacySubAction.OPEN_BORDERS -> {
            if (dm.diplomaticStatus == com.unciv.logic.civilization.diplomacy.DiplomaticStatus.War)
                return ActionResult(false, "Cannot open borders while at war")
            val trade = com.unciv.logic.trade.Trade()
            val offer = com.unciv.logic.trade.TradeOffer(
                com.unciv.Constants.openBorders,
                com.unciv.logic.trade.TradeOfferType.Agreement, duration = 30)
            trade.ourOffers.add(offer); trade.theirOffers.add(offer)
            dm.trades.add(trade); dm.otherCivDiplomacy().trades.add(trade)
            dm.updateHasOpenBorders(); dm.otherCivDiplomacy().updateHasOpenBorders()
            ActionResult(true, "Opened borders with ${targetCiv.civName}")
        }
        DiplomacySubAction.FRIENDSHIP -> {
            dm.signDeclarationOfFriendship()
            ActionResult(true, "Signed declaration of friendship with ${targetCiv.civName}")
        }
        DiplomacySubAction.DENOUNCE -> {
            dm.denounce()
            ActionResult(true, "Denounced ${targetCiv.civName}")
        }
        else -> ActionResult(false, "Diplomacy subaction ${action.diplomacySubaction} not implemented")
    }
}

private fun executeWorkerBuild(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val worker = resolveUnit(civ, entitySnapshot, action.unitTarget)
        ?: return ActionResult(false, "No own unit at entity index ${action.unitTarget}")
    if (!worker.hasUnique(UniqueType.BuildImprovements))
        return ActionResult(false, "Unit cannot build improvements")
    val targetTile = gameInfo.tileMap.values.getOrNull(action.tileTarget)
        ?: return ActionResult(false, "tileTarget ${action.tileTarget} out of range")
    val improvNames = gameInfo.ruleset.tileImprovements.keys.sorted()
    val improvName = improvNames.getOrNull(action.improvementTarget)
        ?: return ActionResult(false, "improvementTarget ${action.improvementTarget} out of range")
    val improvement = gameInfo.ruleset.tileImprovements[improvName]
        ?: return ActionResult(false, "Improvement '$improvName' not in ruleset")
    val gameContext = com.unciv.models.ruleset.unique.GameContext(unit = worker)
    if (!targetTile.improvementFunctions.canBuildImprovement(improvement, gameContext))
        return ActionResult(false, "Cannot build '$improvName' on this tile")
    targetTile.queueImprovement(improvement, civ, worker)
    return ActionResult(true, "Worker queued building $improvName")
}

private fun executeFounderSettle(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val settler = resolveUnit(civ, entitySnapshot, action.unitTarget)
        ?: return ActionResult(false, "No own unit at entity index ${action.unitTarget}")
    if (!settler.baseUnit.isCityFounder())
        return ActionResult(false, "Unit is not a settler")
    val location = settler.getTile().position
    CityFounder().foundCity(civ, location, settler)
    settler.destroy()
    return ActionResult(true, "City founded at $location")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

private fun resolveUnit(
    civ: Civilization,
    entitySnapshot: List<EntitySnapshot>,
    index: Int
): MapUnit? {
    val snap = entitySnapshot.getOrNull(index) ?: return null
    if (snap.entityType != 2) return null
    val unit = snap.unit ?: return null
    if (unit.civ != civ) return null
    return unit
}

/** Extension: get tile by zeroBasedIndex. */
private fun Iterable<com.unciv.logic.map.tile.Tile>.getOrNull(index: Int): com.unciv.logic.map.tile.Tile? =
    this.firstOrNull { it.zeroBasedIndex == index }
