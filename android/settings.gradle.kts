pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "BreathEngineAndroid"
include(":breathengine", ":sample")
// Local-only apps (not part of the public repository).
listOf("game", "lumen").filter { file(it).isDirectory }.forEach { include(":$it") }
