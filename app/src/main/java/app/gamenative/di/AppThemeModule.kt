package app.gamenative.di

import app.gamenative.PrefManager
import app.gamenative.enums.AppTheme
import app.gamenative.ui.enums.AppAccentColor
import com.materialkolor.PaletteStyle
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton
import kotlin.properties.ReadWriteProperty
import kotlin.reflect.KProperty
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Referenced from https://github.com/fvilarino/App-Theme-Compose-Sample
 */

interface IAppTheme {
    val themeFlow: StateFlow<AppTheme>
    var currentTheme: AppTheme
    val paletteFlow: StateFlow<PaletteStyle>
    var currentPalette: PaletteStyle
    val accentColorFlow: StateFlow<AppAccentColor>
    var currentAccentColor: AppAccentColor
    val customAccentColorFlow: StateFlow<Long>
    var currentCustomAccentColorArgb: Long
}

class AppThemeImpl : IAppTheme {

    override val themeFlow: MutableStateFlow<AppTheme> = MutableStateFlow(PrefManager.appTheme)

    override var currentTheme: AppTheme by AppThemeDelegate()

    override val paletteFlow: MutableStateFlow<PaletteStyle> = MutableStateFlow(PrefManager.appThemePalette)

    override var currentPalette: PaletteStyle by AppPaletteDelegate()

    override val accentColorFlow: MutableStateFlow<AppAccentColor> = MutableStateFlow(PrefManager.appAccentColor)

    override var currentAccentColor: AppAccentColor by AppAccentColorDelegate()

    override val customAccentColorFlow: MutableStateFlow<Long> = MutableStateFlow(PrefManager.appCustomAccentColorArgb)

    override var currentCustomAccentColorArgb: Long by AppCustomAccentColorDelegate()

    inner class AppThemeDelegate : ReadWriteProperty<Any, AppTheme> {

        override fun getValue(thisRef: Any, property: KProperty<*>): AppTheme = PrefManager.appTheme

        override fun setValue(thisRef: Any, property: KProperty<*>, value: AppTheme) {
            themeFlow.value = value
            PrefManager.appTheme = value
        }
    }

    inner class AppPaletteDelegate : ReadWriteProperty<Any, PaletteStyle> {

        override fun getValue(thisRef: Any, property: KProperty<*>): PaletteStyle = PrefManager.appThemePalette

        override fun setValue(thisRef: Any, property: KProperty<*>, value: PaletteStyle) {
            paletteFlow.value = value
            PrefManager.appThemePalette = value
        }
    }

    inner class AppAccentColorDelegate : ReadWriteProperty<Any, AppAccentColor> {

        override fun getValue(thisRef: Any, property: KProperty<*>): AppAccentColor = PrefManager.appAccentColor

        override fun setValue(thisRef: Any, property: KProperty<*>, value: AppAccentColor) {
            accentColorFlow.value = value
            PrefManager.appAccentColor = value
        }
    }

    inner class AppCustomAccentColorDelegate : ReadWriteProperty<Any, Long> {

        override fun getValue(thisRef: Any, property: KProperty<*>): Long = PrefManager.appCustomAccentColorArgb

        override fun setValue(thisRef: Any, property: KProperty<*>, value: Long) {
            customAccentColorFlow.value = value
            PrefManager.appCustomAccentColorArgb = value
        }
    }
}

@InstallIn(SingletonComponent::class)
@Module
class AppThemeModule {
    @Provides
    @Singleton
    fun provideAppTheme(): IAppTheme = AppThemeImpl()
}
