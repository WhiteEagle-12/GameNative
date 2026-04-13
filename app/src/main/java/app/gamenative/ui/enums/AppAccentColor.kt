package app.gamenative.ui.enums

import androidx.annotation.StringRes
import app.gamenative.R

enum class AppAccentColor(
    val key: String,
    val argb: Long,
    @param:StringRes val labelRes: Int,
) {
    MAGENTA("magenta", 0xFFA21CAFL, R.string.accent_color_magenta),
    CYAN("cyan", 0xFF00D4FFL, R.string.accent_color_cyan),
    BLUE("blue", 0xFF3B82F6L, R.string.accent_color_blue),
    GREEN("green", 0xFF10B981L, R.string.accent_color_green),
    GOLD("gold", 0xFFF59E0BL, R.string.accent_color_gold),
    RED("red", 0xFFEF4444L, R.string.accent_color_red),
    WHITE("white", 0xFFE5E7EBL, R.string.accent_color_white),
    CUSTOM("custom", 0xFFA21CAFL, R.string.accent_color_custom),
    ;

    fun resolvedArgb(customArgb: Long): Long {
        return if (this == CUSTOM) customArgb else argb
    }

    companion object {
        fun fromKey(key: String): AppAccentColor {
            return entries.find { it.key == key } ?: MAGENTA
        }
    }
}
