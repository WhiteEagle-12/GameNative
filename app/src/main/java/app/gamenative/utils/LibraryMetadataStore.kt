package app.gamenative.utils

import app.gamenative.data.LibraryMetadata
import app.gamenative.db.dao.LibraryMetadataDao
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import timber.log.Timber

/**
 * Lightweight app-wide access point for library metadata.
 *
 * We keep this as a tiny singleton so the library screen, game detail screen,
 * and launch pipeline can all share the same metadata table without threading
 * extra dependencies through every composable.
 */
object LibraryMetadataStore {

    private val _initialized = MutableStateFlow(false)
    val initialized: StateFlow<Boolean> = _initialized.asStateFlow()

    @Volatile
    private var dao: LibraryMetadataDao? = null

    fun init(libraryMetadataDao: LibraryMetadataDao) {
        dao = libraryMetadataDao
        _initialized.value = true
    }

    private fun requireDao(): LibraryMetadataDao {
        return dao ?: throw IllegalStateException("LibraryMetadataStore was not initialized")
    }

    val metadataFlow: Flow<List<LibraryMetadata>>
        get() = requireDao().getAll()

    suspend fun get(appId: String): LibraryMetadata? = withContext(Dispatchers.IO) {
        requireDao().getByAppId(appId)
    }

    suspend fun ensure(appId: String): LibraryMetadata = withContext(Dispatchers.IO) {
        val current = requireDao().getByAppId(appId)
        if (current != null) {
            current
        } else {
            val created = LibraryMetadata(appId = appId)
            requireDao().upsert(created)
            created
        }
    }

    suspend fun setFavorite(appId: String, isFavorite: Boolean) = withContext(Dispatchers.IO) {
        requireDao().upsert(
            (requireDao().getByAppId(appId) ?: LibraryMetadata(appId)).copy(isFavorite = isFavorite)
        )
    }

    suspend fun toggleFavorite(appId: String): Boolean = withContext(Dispatchers.IO) {
        val current = ensure(appId)
        val newValue = !current.isFavorite
        requireDao().upsert(current.copy(isFavorite = newValue))
        newValue
    }

    suspend fun setTags(appId: String, tags: List<String>) = withContext(Dispatchers.IO) {
        val sanitized = tags.map { it.trim() }.filter { it.isNotBlank() }.distinct()
        requireDao().upsert(
            (requireDao().getByAppId(appId) ?: LibraryMetadata(appId)).copy(tags = sanitized)
        )
    }

    suspend fun touchLastPlayed(appId: String, timestampMs: Long = System.currentTimeMillis()) = withContext(Dispatchers.IO) {
        val current = ensure(appId)
        requireDao().upsert(current.copy(lastPlayedAt = timestampMs))
    }

    suspend fun delete(appId: String) = withContext(Dispatchers.IO) {
        try {
            requireDao().delete(appId)
        } catch (e: Exception) {
            Timber.w(e, "Failed to delete library metadata for $appId")
        }
    }
}
