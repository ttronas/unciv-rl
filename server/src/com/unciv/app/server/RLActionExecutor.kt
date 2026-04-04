package com.unciv.app.server

import com.unciv.logic.GameInfo
import com.unciv.logic.battle.AttackableTile
import com.unciv.logic.battle.Battle
import com.unciv.logic.battle.MapUnitCombatant
import com.unciv.logic.battle.TargetHelper
import com.unciv.logic.city.City
import com.unciv.logic.city.managers.CityFounder
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.civilization.diplomacy.DeclareWarReason
import com.unciv.logic.civilization.diplomacy.WarType
import com.unciv.logic.map.mapunit.MapUnit
import com.unciv.models.ruleset.Policy
import kotlinx.serialization.Serializable

// ---------------------------------------------------------------------------
// Action constants
// ---------------------------------------------------------------------------

/** Macro action codes used in [RLAction.macro]. */
object MacroAction {
    const val END_TURN = 0
    const val UNIT_ACTION = 1
    const val CITY_ACTION = 2
    const val TECH_ACTION = 3
    const val DIPLOMACY_ACTION = 4
    const val POLICY_ACTION = 5
    const val SETTLER_ACTION = 6
    const val IMPROVE_TILE = 7
}

/** Unit sub-action codes used in [RLAction.subaction] when macro == UNIT_ACTION. */
object UnitSubAction {
    const val MOVE = 0
    const val ATTACK = 1
    const val FORTIFY = 2
    const val SKIP = 3
    const val PROMOTE = 4
    const val PILLAGE = 5
    const val DISBAND = 6
    const val FOUND_CITY = 7
}

/** City sub-action codes used in [RLAction.subaction] when macro == CITY_ACTION. */
object CitySubAction {
    const val SET_PRODUCTION = 0
    const val BUY_PRODUCTION = 1
    const val SELL_BUILDING = 2
}

/** Tech sub-action codes. */
object TechSubAction {
    const val RESEARCH = 0
}

/** Diplomacy sub-action codes. */
object DiplomacySubAction {
    const val DECLARE_WAR = 0
    const val OFFER_PEACE = 1
    const val OPEN_BORDERS = 2
}

/** Policy sub-action codes. */
object PolicySubAction {
    const val ADOPT = 0
}

// ---------------------------------------------------------------------------
// Action DTO
// ---------------------------------------------------------------------------

/**
 * Hierarchical action submitted to POST /rl/action/{gameId}.
 *
 * | macro | target meaning | subaction meaning | arg1 meaning | arg2 |
 * |-------|---------------|-------------------|-------------|------|
 * | 0 END_TURN | — | — | — | — |
 * | 1 UNIT_ACTION | entity index of the unit | UnitSubAction.* | destination tile index (MOVE), promotion index (PROMOTE) | — |
 * | 2 CITY_ACTION | entity index of the city | CitySubAction.* | production item index | — |
 * | 3 TECH_ACTION | tech index in catalogues.techNames | TechSubAction.RESEARCH | — | — |
 * | 4 DIPLOMACY_ACTION | civ index in allAgents | DiplomacySubAction.* | — | — |
 * | 5 POLICY_ACTION | policy index in catalogues.policyNames | PolicySubAction.ADOPT | — | — |
 * | 6 SETTLER_ACTION | entity index of the settler unit | UnitSubAction.FOUND_CITY | — | — |
 * | 7 IMPROVE_TILE | tile zeroBasedIndex | improve type (0=road,1=improvement,2=clear) | worker entity index | improvement name index |
 */
@Serializable
data class RLAction(
    val macro: Int,
    val target: Int = 0,
    val subaction: Int = 0,
    val arg1: Int = 0,
    val arg2: Int = 0
)

/** Result returned by [executeAction]. */
@Serializable
data class ActionResult(
    val success: Boolean,
    val message: String = "",
    /** Reward signal: delta in total score since the last step for the acting agent. */
    val reward: Float = 0f
)

// ---------------------------------------------------------------------------
// Action executor
// ---------------------------------------------------------------------------

/**
 * Routes a hierarchical [RLAction] to the appropriate Unciv game-logic methods.
 *
 * This function is the single entry point called from the HTTP route handler.
 * It does **not** advance the turn; the caller must call [RLGameManager.endCurrentAgentTurn]
 * after receiving [MacroAction.END_TURN] or handle the flow themselves.
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

            MacroAction.UNIT_ACTION -> executeUnitAction(civ, gameInfo, action, entitySnapshot)

            MacroAction.CITY_ACTION -> executeCityAction(civ, gameInfo, action, entitySnapshot)

            MacroAction.TECH_ACTION -> executeTechAction(civ, gameInfo, action)

            MacroAction.DIPLOMACY_ACTION -> executeDiplomacyAction(civ, gameInfo, action)

            MacroAction.POLICY_ACTION -> executePolicyAction(civ, gameInfo, action)

            MacroAction.SETTLER_ACTION -> executeSettlerAction(civ, gameInfo, action, entitySnapshot)

            MacroAction.IMPROVE_TILE -> executeImproveTileAction(civ, gameInfo, action, entitySnapshot)

            else -> ActionResult(success = false, message = "Unknown macro action ${action.macro}")
        }
    } catch (e: Exception) {
        ActionResult(success = false, message = e.message ?: "Unknown error")
    }
}

// ---------------------------------------------------------------------------
// Entity snapshot (maps entity index → game object)
// ---------------------------------------------------------------------------

/**
 * A lightweight snapshot of the entity list so that the action executor can
 * look up units and cities by their observation-list index without re-running
 * the full observation builder.
 */
data class EntitySnapshot(
    val entityType: Int,   // 1 = city, 2 = unit, 0 = padding
    val unit: MapUnit?,
    val city: City?
)

/**
 * Build a list of [EntitySnapshot]s in the same order as the entity list in the
 * observation (units first, then cities).  Padded with empty entries up to
 * [MAX_ENTITIES].
 */
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
// Sub-executors
// ---------------------------------------------------------------------------

private fun executeUnitAction(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val snap = entitySnapshot.getOrNull(action.target)
        ?: return ActionResult(false, "Entity index ${action.target} out of range")
    val unit = snap.unit
        ?: return ActionResult(false, "Entity at index ${action.target} is not a unit")
    if (unit.civ != civ)
        return ActionResult(false, "Unit does not belong to agent")

    return when (action.subaction) {
        UnitSubAction.MOVE -> {
            val destTile = gameInfo.tileMap.values.getOrNull(action.arg1)
                ?: return ActionResult(false, "Destination tile index ${action.arg1} out of range")
            if (!unit.movement.canReachInCurrentTurn(destTile))
                return ActionResult(false, "Unit cannot reach tile in current turn")
            unit.movement.moveToTile(destTile)
            ActionResult(true, "Unit moved")
        }

        UnitSubAction.ATTACK -> {
            val targetTile = gameInfo.tileMap.values.getOrNull(action.arg1)
                ?: return ActionResult(false, "Target tile index ${action.arg1} out of range")
            val distToTiles = unit.movement.getDistanceToTiles()
            val attackables = TargetHelper.getAttackableEnemies(unit, distToTiles)
            val attackableTile = attackables.firstOrNull { it.tileToAttack == targetTile }
                ?: return ActionResult(false, "Target tile is not attackable")
            Battle.moveAndAttack(MapUnitCombatant(unit), attackableTile)
            ActionResult(true, "Unit attacked")
        }

        UnitSubAction.FORTIFY -> {
            unit.fortifyIfCan()
            ActionResult(true, "Unit fortified")
        }

        UnitSubAction.SKIP -> {
            unit.due = false
            ActionResult(true, "Unit skipped")
        }

        UnitSubAction.PROMOTE -> {
            val availablePromotions = unit.promotions.getAvailablePromotions().toList()
            val promotion = availablePromotions.getOrNull(action.arg1)
                ?: return ActionResult(false, "Promotion index ${action.arg1} out of range")
            if (!unit.promotions.canBePromoted())
                return ActionResult(false, "Unit cannot be promoted")
            unit.promotions.addPromotion(promotion.name)
            ActionResult(true, "Unit promoted to ${promotion.name}")
        }

        UnitSubAction.PILLAGE -> {
            val tile = unit.getTile()
            if (!tile.canPillageTile())
                return ActionResult(false, "Cannot pillage this tile")
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

        else -> ActionResult(false, "Unknown unit subaction ${action.subaction}")
    }
}

private fun executeCityAction(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val snap = entitySnapshot.getOrNull(action.target)
        ?: return ActionResult(false, "Entity index ${action.target} out of range")
    val city = snap.city
        ?: return ActionResult(false, "Entity at index ${action.target} is not a city")
    if (city.civ != civ)
        return ActionResult(false, "City does not belong to agent")

    val ruleset = gameInfo.ruleset
    // Build production item list in the same order as in catalogues
    val productionList = (ruleset.buildings.keys + ruleset.units.keys).distinct().sorted()

    return when (action.subaction) {
        CitySubAction.SET_PRODUCTION -> {
            val itemName = productionList.getOrNull(action.arg1)
                ?: return ActionResult(false, "Production item index ${action.arg1} out of range")
            val construction = ruleset.buildings[itemName] ?: ruleset.units[itemName]
                ?: return ActionResult(false, "'$itemName' not found in ruleset")
            if (!construction.isBuildable(city.cityConstructions))
                return ActionResult(false, "'$itemName' is not buildable in this city right now")
            city.cityConstructions.constructionQueue.clear()
            city.cityConstructions.addToQueue(itemName)
            ActionResult(true, "Production set to $itemName")
        }

        CitySubAction.BUY_PRODUCTION -> {
            val itemName = productionList.getOrNull(action.arg1)
                ?: return ActionResult(false, "Production item index ${action.arg1} out of range")
            val construction = (ruleset.buildings[itemName] ?: ruleset.units[itemName])
                as? com.unciv.models.ruleset.INonPerpetualConstruction
                ?: return ActionResult(false, "'$itemName' not purchasable")
            if (!construction.canBePurchasedWithStat(city, com.unciv.models.stats.Stat.Gold))
                return ActionResult(false, "Cannot purchase $itemName with gold in this city")
            val goldCost = construction.getStatBuyCost(city, com.unciv.models.stats.Stat.Gold)
                ?: return ActionResult(false, "No gold cost for $itemName")
            if (civ.gold < goldCost)
                return ActionResult(false, "Insufficient gold ($goldCost required, ${civ.gold} available)")
            city.cityConstructions.purchaseConstruction(itemName, 0, false)
            ActionResult(true, "Purchased $itemName for $goldCost gold")
        }

        CitySubAction.SELL_BUILDING -> {
            val itemName = productionList.getOrNull(action.arg1)
                ?: return ActionResult(false, "Building index ${action.arg1} out of range")
            val building = ruleset.buildings[itemName]
                ?: return ActionResult(false, "'$itemName' is not a building")
            if (!city.cityConstructions.isBuilt(itemName))
                return ActionResult(false, "Building $itemName is not built in this city")
            if (city.hasSoldBuildingThisTurn)
                return ActionResult(false, "Already sold a building this turn")
            city.cityConstructions.removeBuilding(itemName)
            val sellGold = (building.cost / 10).coerceAtLeast(1)
            civ.addGold(sellGold)
            city.hasSoldBuildingThisTurn = true
            ActionResult(true, "Building $itemName sold for $sellGold gold")
        }

        else -> ActionResult(false, "Unknown city subaction ${action.subaction}")
    }
}

private fun executeTechAction(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction
): ActionResult {
    val techList = gameInfo.ruleset.technologies.keys.sorted()
    val techName = techList.getOrNull(action.target)
        ?: return ActionResult(false, "Tech index ${action.target} out of range")

    if (!civ.tech.canBeResearched(techName))
        return ActionResult(false, "'$techName' cannot be researched right now")

    // Set as the next technology to research (insert at front of queue)
    civ.tech.techsToResearch.remove(techName)
    civ.tech.techsToResearch.add(0, techName)
    return ActionResult(true, "Now researching $techName")
}

private fun executeDiplomacyAction(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction
): ActionResult {
    val allAgents = gameInfo.civilizations
        .filter { it.playerType == com.unciv.logic.civilization.PlayerType.Human && !it.isSpectator() }
        .map { it.civID }
    val targetCivId = allAgents.getOrNull(action.target)
        ?: // Fall back to any major civ by index
        gameInfo.civilizations.filter { it.isMajorCiv() }.getOrNull(action.target)?.civID
        ?: return ActionResult(false, "Civ index ${action.target} out of range")

    val targetCiv = gameInfo.getCivilizationOrNull(targetCivId)
        ?: return ActionResult(false, "Target civ '$targetCivId' not found")
    val dm = civ.getDiplomacyManagerOrMeet(targetCiv)

    return when (action.subaction) {
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
            // Open borders is handled through trades; we establish it via a direct flag
            // if allowed under the trade rules (simplified implementation)
            if (dm.diplomaticStatus == com.unciv.logic.civilization.diplomacy.DiplomaticStatus.War)
                return ActionResult(false, "Cannot open borders while at war with ${targetCiv.civName}")
            // Add an open-borders trade offer for a fixed duration of 30 turns
            val openBordersTrade = com.unciv.logic.trade.Trade()
            openBordersTrade.ourOffers.add(
                com.unciv.logic.trade.TradeOffer(
                    com.unciv.Constants.openBorders,
                    com.unciv.logic.trade.TradeOfferType.Agreement, 30))
            openBordersTrade.theirOffers.add(
                com.unciv.logic.trade.TradeOffer(
                    com.unciv.Constants.openBorders,
                    com.unciv.logic.trade.TradeOfferType.Agreement, 30))
            dm.trades.add(openBordersTrade)
            dm.otherCivDiplomacy().trades.add(openBordersTrade)
            dm.updateHasOpenBorders()
            dm.otherCivDiplomacy().updateHasOpenBorders()
            ActionResult(true, "Opened borders with ${targetCiv.civName}")
        }

        else -> ActionResult(false, "Unknown diplomacy subaction ${action.subaction}")
    }
}

private fun executePolicyAction(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction
): ActionResult {
    val policyList = gameInfo.ruleset.policies.keys.sorted()
    val policyName = policyList.getOrNull(action.target)
        ?: return ActionResult(false, "Policy index ${action.target} out of range")

    val policy = gameInfo.ruleset.policies[policyName]
        ?: return ActionResult(false, "Policy '$policyName' not in ruleset")

    if (!civ.policies.isAdoptable(policy))
        return ActionResult(false, "'$policyName' is not adoptable right now")
    if (!civ.policies.canAdoptPolicy())
        return ActionResult(false, "Cannot adopt policy: no free policies or insufficient culture")

    civ.policies.adopt(policy)
    return ActionResult(true, "Adopted policy $policyName")
}

private fun executeSettlerAction(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val snap = entitySnapshot.getOrNull(action.target)
        ?: return ActionResult(false, "Entity index ${action.target} out of range")
    val unit = snap.unit
        ?: return ActionResult(false, "Entity at index ${action.target} is not a unit")
    if (!unit.baseUnit.isCityFounder())
        return ActionResult(false, "Unit is not a settler")
    if (unit.civ != civ)
        return ActionResult(false, "Settler does not belong to agent")

    val location = unit.getTile().position
    CityFounder().foundCity(civ, location, unit)
    unit.destroy()
    return ActionResult(true, "City founded at $location")
}

private fun executeImproveTileAction(
    civ: Civilization,
    gameInfo: GameInfo,
    action: RLAction,
    entitySnapshot: List<EntitySnapshot>
): ActionResult {
    val targetTile = gameInfo.tileMap.values.getOrNull(action.target)
        ?: return ActionResult(false, "Tile index ${action.target} out of range")

    val workerSnap = entitySnapshot.getOrNull(action.arg1)
    val worker = workerSnap?.unit
        ?: return ActionResult(false, "Worker entity index ${action.arg1} is not a unit")
    if (worker.civ != civ)
        return ActionResult(false, "Worker does not belong to agent")
    if (!worker.hasUnique(com.unciv.models.ruleset.unique.UniqueType.BuildImprovements))
        return ActionResult(false, "Unit cannot build improvements")

    val gameContext = com.unciv.models.ruleset.unique.GameContext(unit = worker)
    val improvementNames = gameInfo.ruleset.tileImprovements.keys.sorted()

    return when (action.subaction) {
        0 -> { // build road
            val roadImprovement = gameInfo.ruleset.tileImprovements["Road"]
                ?: return ActionResult(false, "Road improvement not in ruleset")
            if (!targetTile.improvementFunctions.canBuildImprovement(roadImprovement, gameContext))
                return ActionResult(false, "Cannot build road on this tile")
            targetTile.queueImprovement(roadImprovement, civ, worker)
            ActionResult(true, "Worker queued road construction")
        }
        1 -> { // build improvement
            val improvName = improvementNames.getOrNull(action.arg2)
                ?: return ActionResult(false, "Improvement index ${action.arg2} out of range")
            val improvement = gameInfo.ruleset.tileImprovements[improvName]
                ?: return ActionResult(false, "Improvement '$improvName' not in ruleset")
            if (!targetTile.improvementFunctions.canBuildImprovement(improvement, gameContext))
                return ActionResult(false, "Cannot build '$improvName' on this tile")
            targetTile.queueImprovement(improvement, civ, worker)
            ActionResult(true, "Worker queued building $improvName")
        }
        2 -> { // clear feature
            val clearAction = targetTile.terrainFeatures.firstOrNull()?.let { feat ->
                gameInfo.ruleset.tileImprovements["Remove $feat"]
            } ?: return ActionResult(false, "No clearable feature on this tile")
            if (!targetTile.improvementFunctions.canBuildImprovement(clearAction, gameContext))
                return ActionResult(false, "Cannot clear feature on this tile")
            targetTile.queueImprovement(clearAction, civ, worker)
            ActionResult(true, "Worker queued feature clearing")
        }
        else -> ActionResult(false, "Unknown improve_tile subaction ${action.subaction}")
    }
}

// ---------------------------------------------------------------------------
// Small helper extensions
// ---------------------------------------------------------------------------

private fun Iterable<com.unciv.logic.map.tile.Tile>.getOrNull(index: Int): com.unciv.logic.map.tile.Tile? =
    this.firstOrNull { it.zeroBasedIndex == index }
