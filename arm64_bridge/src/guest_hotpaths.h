#pragma once

#include <cstdint>

class DynarmicHost;

// Installs checksum-pinned native implementations for guest functions proven
// hot by a sampled profile. The caller supplies the fixed libunity load base.
bool InstallUnityProfiledHotPaths(DynarmicHost& host, std::uint32_t unity_base);

// Skips the five authored intro logo cards at their single IntroSpawner call
// site while leaving AppSaveData's getter semantics intact everywhere else,
// then makes the now-obsolete splash-to-menu camera transition immediate.
bool ApplyIntroLogoSkipPatch(DynarmicHost& host, std::uint32_t il2cpp_base);

// Completes both stages of the main-menu television reveal on their first
// MoveNext calls: activate each image immediately, then preserve its authored
// final alpha while skipping the stagger delay and repeated alpha flashes.
bool ApplyMainMenuImageFadeSkipPatch(DynarmicHost& host, std::uint32_t il2cpp_base);
