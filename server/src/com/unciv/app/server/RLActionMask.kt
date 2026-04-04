package com.unciv.app.server

import com.unciv.logic.GameInfo
import com.unciv.logic.civilization.Civilization
import com.unciv.logic.civilization.diplomacy.DiplomaticStatus
import com.unciv.models.ruleset.unique.GameContext
import com.unciv.models.ruleset.unique.UniqueType
import kotlinx.serialization.Serializable

/**
 * Per-level action masks returned by GET /rl/action_mask/{gameId}.
 *
 * The masks are intentionally hierarchical: [macroMask] tells the agent which macro
 * actions are available at all; the per-macro arrays then indicate which targets /
 * subactions / arguments are legal for each enabled macro.
 *
 * All mask arrays contain **true** for valid choices and **false** for illegal ones.
 */
@Serializable
data class RLActionMaskResponse(
    val gameId: String,
    val agentCivId: String,
    /** Length = number of macro actions (currently 8). */
    val macroMask: List<Boolean>,
    /**
     * For macro 1 (UNIT_ACTION): which entity-list indices are units belonging
     * to the current agent that can still act this turn.
     * Length = MAX_ENTITIES.
     */
    val unitTargetMask: List<Boolean>,
    /**
     * For macro 1 (UNIT_ACTION): which sub-actions are legal for ANY of the
     * moveable units.  If a sub-action is in this list it means at least one
     * unit can perform it; the agent should still check the per-unit validity
     * after selecting a target.
     * Length = number of unit sub-actions (8).
     */
    val unitSubactionMask: List<Boolean>,
    /**
     * For macro 2 (CITY_ACTION): which entity indices are cities belonging to
     * the agent.  Length = MAX_ENTITIES.
     */
    val cityTargetMask: List<Boolean>,
    /**
     * For macro 3 (TECH_ACTION): which tech indices are currently researchable.
     * Length = number of techs in the ruleset (variable; clients should use the
     * catalogues list to map indices).
     */
    val techTargetMask: List<Boolean>,
    /**
     * For macro 4 (DIPLOMACY_ACTION): which civ indices (in allAgents) are
     * valid targets.  Length = number of agents.
     */
    val diplomacyTargetMask: List<Boolean>,
    /**
     * For macro 4 (DIPLOMACY_ACTION): which sub-actions are legally available
     * against any of the valid targets.  Length = 3.
     */
    val diplomacySubactionMask: List<Boolean>,
    /**
     * For macro 5 (POLICY_ACTION): which policy indices are currently adoptable.
     * Length = number of policies in the ruleset.
     */
    val policyTargetMask: List<Boolean>,
    /**
     * For macro 6 (SETTLER_ACTION): which entity indices are settler units.
     * Length = MAX_ENTITIES.
     */
    val settlerTargetMask: List<Boolean>,
    /**
     * For macro 2 (CITY_ACTION): which production items are buildable across
     * at least one city.  Length = number of production items in catalogues.
     */
    val productionItemMask: List<Boolean>
)

// ---------------------------------------------------------------------------
// Mask builder
// ---------------------------------------------------------------------------

/**
 * Computes a complete [RLActionMaskResponse] for [agentCivId] in [gameInfo].
 */
fun computeActionMask(
    gameInfo: GameInfo,
    agentCivId: String,
    allAgents: List<String>,
    entitySnapshot: List<EntitySnapshot>
): RLActionMaskResponse {

    val civ = gameInfo.getCivilization(agentCivId)
    val ruleset = gameInfo.ruleset

    // ---- catalogues (same order as observations) --------------------------
    val techList = ruleset.technologies.keys.sorted()
    val policyList = ruleset.policies.keys.sorted()
    val productionList = (ruleset.buildings.keys + ruleset.units.keys).distinct().sorted()

    // ---- unit masks -------------------------------------------------------
    val unitTargetMask = BooleanArray(MAX_ENTITIES) { false }
    val unitSubactionSet = BooleanArray(8) { false }

    for ((idx, snap) in entitySnapshot.withIndex()) {
        if (snap.entityType != 2) continue       // not a unit
        val unit = snap.unit ?: continue
        if (unit.civ != civ) continue             // not ours
        if (!unit.due || unit.currentMovement <= 0f) continue  // no actions left

        unitTargetMask[idx] = true

        // Subaction 0: MOVE – the unit has movement points
        if (unit.currentMovement > 0f) unitSubactionSet[UnitSubAction.MOVE] = true

        // Subaction 1: ATTACK – any attackable enemy nearby
        val distToTiles = unit.movement.getDistanceToTiles()
        val attackables = com.unciv.logic.battle.TargetHelper
            .getAttackableEnemies(unit, distToTiles)
        if (attackables.isNotEmpty()) unitSubactionSet[UnitSubAction.ATTACK] = true

        // Subaction 2: FORTIFY
        if (!unit.isFortified()) unitSubactionSet[UnitSubAction.FORTIFY] = true

        // Subaction 3: SKIP always available while unit has due flag
        unitSubactionSet[UnitSubAction.SKIP] = true

        // Subaction 4: PROMOTE
        if (unit.promotions.canBePromoted()) unitSubactionSet[UnitSubAction.PROMOTE] = true

        // Subaction 5: PILLAGE
        if (unit.getTile().canPillageTile()) unitSubactionSet[UnitSubAction.PILLAGE] = true

        // Subaction 6: DISBAND always available if the unit exists
        unitSubactionSet[UnitSubAction.DISBAND] = true

        // Subaction 7: FOUND_CITY (settler)
        if (unit.baseUnit.isCityFounder()) unitSubactionSet[UnitSubAction.FOUND_CITY] = true
    }

    // ---- city masks -------------------------------------------------------
    val cityTargetMask = BooleanArray(MAX_ENTITIES) { false }
    for ((idx, snap) in entitySnapshot.withIndex()) {
        if (snap.entityType != 1) continue
        val city = snap.city ?: continue
        if (city.civ != civ) continue
        cityTargetMask[idx] = true
    }

    // ---- settler masks (cities can be founded on current tile) ------------
    val settlerTargetMask = BooleanArray(MAX_ENTITIES) { false }
    for ((idx, snap) in entitySnapshot.withIndex()) {
        if (snap.entityType != 2) continue
        val unit = snap.unit ?: continue
        if (unit.civ != civ) continue
        if (unit.baseUnit.isCityFounder() && unit.currentMovement > 0f)
            settlerTargetMask[idx] = true
    }

    // ---- tech mask --------------------------------------------------------
    val techTargetMask = BooleanArray(techList.size) { false }
    for ((idx, techName) in techList.withIndex()) {
        if (civ.tech.canBeResearched(techName)) techTargetMask[idx] = true
    }

    // ---- policy mask ------------------------------------------------------
    val policyTargetMask = BooleanArray(policyList.size) { false }
    if (civ.policies.canAdoptPolicy()) {
        for ((idx, policyName) in policyList.withIndex()) {
            val policy = ruleset.policies[policyName] ?: continue
            if (civ.policies.isAdoptable(policy)) policyTargetMask[idx] = true
        }
    }

    // ---- diplomacy masks --------------------------------------------------
    val diplomacyTargetMask = BooleanArray(allAgents.size) { false }
    val diplomacySubactionMask = BooleanArray(3) { false }
    for ((idx, otherId) in allAgents.withIndex()) {
        if (otherId == agentCivId) continue
        val otherCiv = gameInfo.getCivilizationOrNull(otherId) ?: continue
        if (otherCiv.isDefeated()) continue
        val dm = civ.getDiplomacyManager(otherCiv)
        if (dm != null) {
            diplomacyTargetMask[idx] = true
            if (dm.diplomaticStatus != DiplomaticStatus.War) {
                diplomacySubactionMask[DiplomacySubAction.DECLARE_WAR] = true
                diplomacySubactionMask[DiplomacySubAction.OPEN_BORDERS] = true
            } else {
                diplomacySubactionMask[DiplomacySubAction.OFFER_PEACE] = true
            }
        }
    }

    // ---- production item mask (across all cities) -------------------------
    val productionItemMask = BooleanArray(productionList.size) { false }
    for (agentCity in civ.cities) {
        for ((idx, itemName) in productionList.withIndex()) {
            if (productionItemMask[idx]) continue  // already enabled
            val construction = ruleset.buildings[itemName] ?: ruleset.units[itemName] ?: continue
            if (construction.isBuildable(agentCity.cityConstructions)) productionItemMask[idx] = true
        }
    }

    // ---- macro mask -------------------------------------------------------
    val macroMask = booleanArrayOf(
        true,                                                     // 0 END_TURN always valid
        unitTargetMask.any { it },                                // 1 UNIT_ACTION
        cityTargetMask.any { it },                                // 2 CITY_ACTION
        techTargetMask.any { it },                                // 3 TECH_ACTION
        diplomacyTargetMask.any { it },                           // 4 DIPLOMACY_ACTION
        policyTargetMask.any { it },                              // 5 POLICY_ACTION
        settlerTargetMask.any { it },                             // 6 SETTLER_ACTION
        unitTargetMask.any { it }                                 // 7 IMPROVE_TILE (any unit could be a worker)
    )

    return RLActionMaskResponse(
        gameId = gameInfo.gameId,
        agentCivId = agentCivId,
        macroMask = macroMask.toList(),
        unitTargetMask = unitTargetMask.toList(),
        unitSubactionMask = unitSubactionSet.toList(),
        cityTargetMask = cityTargetMask.toList(),
        techTargetMask = techTargetMask.toList(),
        diplomacyTargetMask = diplomacyTargetMask.toList(),
        diplomacySubactionMask = diplomacySubactionMask.toList(),
        policyTargetMask = policyTargetMask.toList(),
        settlerTargetMask = settlerTargetMask.toList(),
        productionItemMask = productionItemMask.toList()
    )
}
