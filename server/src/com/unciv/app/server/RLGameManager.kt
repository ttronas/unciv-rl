package com.unciv.app.server

import com.unciv.UncivGame
import com.unciv.logic.GameInfo
import com.unciv.logic.GameStarter
import com.unciv.logic.civilization.PlayerType
import com.unciv.logic.civilization.managers.TurnManager
import com.unciv.models.metadata.BaseRuleset
import com.unciv.models.metadata.GameParameters
import com.unciv.models.metadata.GameSetupInfo
import com.unciv.models.metadata.Player
import com.unciv.models.ruleset.RulesetCache
import com.unciv.models.metadata.GameSettings
import kotlinx.coroutines.sync.Mutex
import kotlinx.serialization.Serializable
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

// ---------------------------------------------------------------------------
// Request / response DTOs
// ---------------------------------------------------------------------------

/** Parameters for POST /rl/new_game */
@Serializable
data class NewGameRequest(
    /** Number of RL-controlled civilisations (players). Must be ≥ 1. */
    val numAgents: Int = 1,
    /** Number of AI opponents. Default: 3. */
    val numAI: Int = 3,
    /** Number of city-state civs. Default: 6. */
    val numCityStates: Int = 6,
    /** Difficulty level name (e.g. "Prince"). */
    val difficulty: String = "Prince",
    /** Whether to include the spatial map planes in observations. */
    val includeMapPlanes: Boolean = true,
    /** Whether to disable barbarians for faster/cleaner games. */
    val noBarbarians: Boolean = true,
    /** Random seed, 0 = random. */
    val seed: Long = 0L
)

/** Response for POST /rl/new_game */
@Serializable
data class NewGameResponse(
    val gameId: String,
    /** civID strings for each RL agent, in order. */
    val agentCivIds: List<String>,
    /** The civID of the agent whose turn it currently is. */
    val currentAgent: String,
    val observation: RLObservationResponse
)

/** Response for GET /rl/done/{gameId} */
@Serializable
data class DoneResponse(
    val gameId: String,
    val done: Boolean,
    val winner: String?,
    val scores: Map<String, Int>
)

/** Response for POST /rl/reset/{gameId} */
@Serializable
data class ResetResponse(
    val gameId: String,
    val agentCivIds: List<String>,
    val currentAgent: String,
    val observation: RLObservationResponse
)

// ---------------------------------------------------------------------------
// Internal game state tracking
// ---------------------------------------------------------------------------

/**
 * Container that pairs a running [GameInfo] with its RL-specific metadata.
 */
data class RLGameState(
    val gameInfo: GameInfo,
    /** civID strings for the RL-controlled players, in agent-index order. */
    val agentCivIds: List<String>,
    /** Original setup params, kept so we can reset the game. */
    val setupRequest: NewGameRequest,
    /** Whether to include map planes in observations for this game. */
    val includeMapPlanes: Boolean,
    /** Previous score per agent civID, used to compute per-step rewards. */
    var previousScores: Map<String, Int> = emptyMap()
)

// ---------------------------------------------------------------------------
// Manager singleton
// ---------------------------------------------------------------------------

/**
 * Manages a pool of in-progress RL game instances.
 *
 * Thread-safety: individual game instances are **not** thread-safe.  This
 * manager provides a per-game [Mutex] (see [getMutex]) that the HTTP route
 * layer uses to serialise concurrent requests that target the **same** game.
 * Requests for different games proceed concurrently without contention.
 */
object RLGameManager {

    private val games = ConcurrentHashMap<String, RLGameState>()

    /**
     * Per-game coroutine mutexes.  Each game gets one [Mutex] so that
     * concurrent HTTP requests targeting the **same** game are serialised
     * (important for correctness when RLlib retries or sends overlapping
     * requests).  Different games are fully independent and make progress in
     * parallel.
     */
    private val gameMutexes = ConcurrentHashMap<String, Mutex>()

    /**
     * Return the [Mutex] for [gameId], creating one on first access.
     *
     * Callers should wrap any read or write operation on a game inside
     * `getMutex(gameId).withLock { … }` to prevent data races.
     */
    fun getMutex(gameId: String): Mutex = gameMutexes.getOrPut(gameId) { Mutex() }

    // -----------------------------------------------------------------------
    // Lifecycle helpers
    // -----------------------------------------------------------------------

    /**
     * One-time initialisation that must be called before any game can be
     * started.  Safe to call more than once (subsequent calls are no-ops).
     *
     * Loads ruleset data using *console mode* (direct file-system access) so
     * that no LibGDX display is needed.  The current working directory must
     * contain the `jsons/` folder from `android/assets/jsons/` in the repo.
     */
    @Synchronized
    fun initialise() {
        if (isInitialised) return

        // Minimal UncivGame setup: no UI, no file persistence
        UncivGame.Current = UncivGame(isConsoleMode = true)
        UncivGame.Current.settings = GameSettings()

        // Load built-in rulesets from the filesystem (consoleMode = true).
        // Mods are skipped (noMods = true) to keep things deterministic.
        RulesetCache.loadRulesets(consoleMode = true, noMods = true)

        isInitialised = true
        println("[RLGameManager] Initialised — ${RulesetCache.size} ruleset(s) loaded.")
    }

    private var isInitialised = false

    // -----------------------------------------------------------------------
    // Game CRUD
    // -----------------------------------------------------------------------

    /** Create a new game according to [request] and return its [RLGameState]. */
    fun createGame(request: NewGameRequest): RLGameState {
        require(isInitialised) { "RLGameManager.initialise() must be called first." }
        require(request.numAgents in 1..8) { "numAgents must be 1..8" }

        val setup = buildSetup(request)
        val gameInfo = GameStarter.startNewGame(setup)

        // Identify the civIDs assigned to Human (RL-agent) players.
        // Spectators are excluded because they don't participate in the game
        // and should not receive RL observations or perform actions.
        val agentCivIds: List<String> = gameInfo.civilizations
            .filter { it.playerType == PlayerType.Human && !it.isSpectator() }
            .map { it.civID }

        check(agentCivIds.isNotEmpty()) { "No Human civs found after game creation." }

        // Begin the first turn so that the first human player's state is ready
        // to be observed (startTurn has been called on them).
        advanceToNextAgentTurn(gameInfo)

        val initialScores = agentCivIds.associateWith { id ->
            gameInfo.getCivilizationOrNull(id)?.calculateTotalScore()?.toInt() ?: 0
        }

        val state = RLGameState(
            gameInfo = gameInfo,
            agentCivIds = agentCivIds,
            setupRequest = request,
            includeMapPlanes = request.includeMapPlanes,
            previousScores = initialScores
        )
        games[gameInfo.gameId] = state
        return state
    }

    /** Retrieve an existing game state, or null if not found. */
    fun getGame(gameId: String): RLGameState? = games[gameId]

    /** Remove a game from the pool and discard its mutex.
     *
     * **Important ordering**: this must be called only while the caller already
     * holds the game's [Mutex] (i.e. inside a [getMutex] `withLock` block), or
     * after all coroutines that previously obtained a reference to the mutex via
     * [getMutex] have completed their `withLock` blocks.  Calling [getMutex]
     * *after* [removeGame] returns will create a new, unrelated [Mutex] object,
     * so the new caller will not wait for any prior lock holder.
     *
     * In practice, all RL route handlers call [getMutex] once, complete their
     * operation (which may call [removeGame]), and then release the lock.
     * Subsequent requests for a removed game receive a 404 from [getGame].
     */
    fun removeGame(gameId: String) {
        games.remove(gameId)
        gameMutexes.remove(gameId)
    }

    // -----------------------------------------------------------------------
    // Turn flow
    // -----------------------------------------------------------------------

    /**
     * End the current human (RL agent) player's turn and advance the game
     * until the next human player's turn begins.
     *
     * After this call, [GameInfo.currentPlayerCiv] is the next RL agent.
     */
    fun endCurrentAgentTurn(state: RLGameState) {
        val gameInfo = state.gameInfo
        // nextTurn() ends the current human's turn, processes all AI civs,
        // and starts the next human player's turn.
        gameInfo.nextTurn()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /** Advance the game to the first human-player turn, calling startTurn. */
    private fun advanceToNextAgentTurn(gameInfo: GameInfo) {
        // The game is freshly created.  nextTurn() will process any AI civs
        // that appear before the first human civ and stop when it finds one.
        gameInfo.nextTurn()
    }

    private fun buildSetup(request: NewGameRequest): GameSetupInfo {
        val params = GameParameters()
        params.difficulty = request.difficulty
        params.noBarbarians = request.noBarbarians
        params.numberOfCityStates = request.numCityStates

        // Build the player list: numAgents Human + numAI AI
        params.players = ArrayList<Player>().apply {
            repeat(request.numAgents) { add(Player(playerType = PlayerType.Human)) }
            repeat(request.numAI) { add(Player(playerType = PlayerType.AI)) }
        }

        val setup = GameSetupInfo(params)
        if (request.seed != 0L) setup.mapParameters.seed = request.seed

        return setup
    }
}
