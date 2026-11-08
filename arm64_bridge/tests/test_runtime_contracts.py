import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTests(unittest.TestCase):
    def test_guest_affinity_hint_does_not_pin_host_threads(self):
        source = (ROOT / "src" / "thunks_libc.cpp").read_text(encoding="utf-8")
        start = source.index('RegisterThunk("syscall"')
        end = source.index('RegisterThunk("pthread_detach"', start)
        syscall_thunk = source[start:end]

        self.assertNotIn("SYS_sched_setaffinity", syscall_thunk)
        self.assertIn("SYS_sched_getaffinity, pid, sizeof(allowed), allowed", syscall_thunk)
        self.assertIn("intersects_allowed_cpu", syscall_thunk)
        self.assertIn("jit.Regs()[0] = 0;", syscall_thunk)

    def test_arm64_exclusive_fast_path_keeps_host_cas(self):
        host = (ROOT / "src" / "dynarmic_host.cpp").read_text(encoding="utf-8")
        patcher = (ROOT / "cmake" / "patch_dynarmic_arm64.cmake").read_text(encoding="utf-8")
        extras = (ROOT / "cmake" / "patch_dynarmic_arm64_extras.cmake").read_text(encoding="utf-8")

        self.assertIn("config.unsafe_optimizations = true;", host)
        self.assertIn("OptimizationFlag::Unsafe_IgnoreGlobalMonitor", host)
        self.assertIn("__atomic_compare_exchange_n", host)
        self.assertIn("BRIDGE_ARM64_IGNORE_GLOBAL_MONITOR", patcher)
        self.assertIn("GetExclusiveMonitorValuePointer", patcher)
        self.assertIn("MemoryWriteExclusive", host)
        self.assertIn("config.fastmem_exclusive_access = true;", host)
        self.assertIn("BRIDGE_ARM64_FASTMEM_EXCLUSIVE", extras)
        self.assertIn("code.LDAXR", extras)
        self.assertIn("code.STLXR", extras)

    def test_dynarmic_audio_instruction_patch_is_reproducible(self):
        extras = (ROOT / "cmake" / "patch_dynarmic_arm64_extras.cmake").read_text(encoding="utf-8")

        self.assertIn("BRIDGE_A32_VADDHN", extras)
        self.assertIn("asimd_VADDHN", extras)
        self.assertIn("asimd_VRADDHN", extras)

    def test_frame_threads_use_adpf_without_device_specific_cpu_numbers(self):
        perf = (ROOT / "src" / "performance_hints.cpp").read_text(encoding="utf-8")
        jni = (ROOT / "src" / "jni_bridge.cpp").read_text(encoding="utf-8")
        libc = (ROOT / "src" / "thunks_libc.cpp").read_text(encoding="utf-8")

        self.assertIn("1'000'000'000LL / 120", perf)
        self.assertIn("kRefreshFrameDurationNs * 9 / 10", perf)
        self.assertIn("APerformanceHint_createSession", perf)
        self.assertIn("APerformanceHint_reportActualWorkDuration", perf)
        self.assertIn("bridge_perf::BeginRenderFrame", jni)
        self.assertIn("bridge_perf::NoteGraphicsThread", libc)
        self.assertIn("bridge_perf::NoteWorkerThread", libc)
        self.assertNotIn("sched_setaffinity(", perf)
        self.assertNotIn("CPU_SET(", perf)
        self.assertNotIn("CPU_SET(", jni)
        self.assertNotIn("CPU_SET(", libc)

    def test_guest_call_arguments_do_not_allocate_each_frame(self):
        header = (ROOT / "src" / "dynarmic_host.h").read_text(encoding="utf-8")
        source = (ROOT / "src" / "dynarmic_host.cpp").read_text(encoding="utf-8")

        self.assertIn("std::span<const uint32_t> args", header)
        self.assertIn("std::initializer_list<uint32_t> args", header)
        self.assertNotIn("const std::vector<uint32_t>& args", header)
        self.assertNotIn("const std::vector<uint32_t>& args", source)

    def test_unexpected_guest_halt_survives_release_log_filter(self):
        source = (ROOT / "src" / "dynarmic_host.cpp").read_text(encoding="utf-8")
        self.assertIn('LOGE("RunFunction(0x%08X) unexpected halt', source)
        self.assertNotIn('LOGI("RunFunction(0x%08X) unexpected halt', source)

    def test_bulk_memory_thunks_respect_protected_guest_null_page(self):
        header = (ROOT / "src" / "dynarmic_host.h").read_text(encoding="utf-8")
        libc = (ROOT / "src" / "thunks_libc.cpp").read_text(encoding="utf-8")
        opensles = (ROOT / "src" / "thunks_opensles.cpp").read_text(encoding="utf-8")

        self.assertIn("bool IsGuestMemoryRangeAccessible", header)
        self.assertIn("m_low_page_guard_size", header)
        self.assertIn('GuestBulkCopy(jit, host, "memcpy"', libc)
        self.assertIn('GuestBulkCopy(jit, host, "memmove"', libc)
        self.assertIn('GuestBulkSet(jit, host, "memset"', libc)
        self.assertIn('GuestBulkCopy(jit, host, "__aeabi_memcpy"', libc)
        self.assertIn("if (destination_valid) memset", libc)
        self.assertIn("host.IsGuestMemoryRangeAccessible(pBuffer", opensles)
        self.assertNotIn("GuestRangeValid(host.GetGuestMemorySize(), pBuffer", opensles)

    def test_dense_thunk_ids_use_direct_dispatch(self):
        header = (ROOT / "src" / "thunk_manager.h").read_text(encoding="utf-8")
        source = (ROOT / "src" / "thunk_manager.cpp").read_text(encoding="utf-8")

        self.assertIn("PublishedEntries<ThunkEntry, kMaxThunks> m_dispatch", header)
        dispatch = source[source.index("void ThunkManager::DispatchThunk"):]
        dispatch = dispatch[:dispatch.index("std::string ThunkManager::GetThunkName")]
        self.assertIn("m_dispatch.Read(thunk_id)", dispatch)
        self.assertIn("entry->handler(jit, m_host)", dispatch)
        self.assertNotIn("lock_guard", dispatch)
        self.assertNotIn(".find(", dispatch)

    def test_opensl_native_calls_hold_player_lifetime_and_api_lock(self):
        import re
        source = (ROOT / "src" / "thunks_opensles.cpp").read_text(encoding="utf-8")
        for chunk in source.split('manager.RegisterThunk(')[1:]:
            if "FindPlayer(" not in chunk:
                continue
            self.assertIn("auto player = FindPlayer(", chunk)
            self.assertIn("if (player) api_lock = std::unique_lock(player->api_mutex);", chunk)
            native = re.search(r'\(\*player->', chunk)
            if native:
                self.assertLess(chunk.index("api_lock ="), native.start())
        self.assertNotIn("MAX_TRACKED_AUDIO_PLAYERS", source)
        self.assertIn("std::erase(g_players, player)", source)
        self.assertIn("g_playerByGuestHandle.erase(handle)", source)
        self.assertIn("std::vector<uint8_t>().swap(storage)", source)

    def test_audio_pump_does_not_serialize_render_or_spin_when_paused(self):
        source = (ROOT / "src" / "thunks_opensles.cpp").read_text(encoding="utf-8")
        self.assertNotIn("host.GetMutex()", source)
        self.assertIn("!g_audioThreadRunning.load(std::memory_order_acquire)) PumpCallbacksOnCurrentThread", source)
        self.assertIn("!player->delivery.Current(ticket)", source)
        wait = source[source.index("g_audioCv.wait_for"):source.index("// Periodic health dump")]
        self.assertIn("!g_audioPaused.load(std::memory_order_acquire)", wait)
        self.assertIn("std::chrono::milliseconds(1)", wait)

    def test_opensl_millibel_output_is_16_bit(self):
        source = (ROOT / "src" / "thunks_opensles.cpp").read_text(encoding="utf-8")
        for name in ("SLVolume_GetVolumeLevel", "SLVolume_GetMaxVolumeLevel"):
            start = source.index(f'manager.RegisterThunk("{name}"')
            chunk = source[start:source.index("\n    });", start)]
            self.assertIn("pLevel, sizeof(SLmillibel)", chunk)
            self.assertIn("host.MemoryWrite16(pLevel", chunk)
            self.assertNotIn("host.MemoryWrite32(pLevel", chunk)

    def test_gl_mappings_follow_native_identity_and_context_share_group(self):
        core = (ROOT / "src" / "thunks_gl_gles2_core.cpp").read_text(encoding="utf-8")
        libc = (ROOT / "src" / "thunks_libc.cpp").read_text(encoding="utf-8")
        self.assertNotIn("thread_local std::unordered_map<GLuint, GuestMappedBuffer>", core)
        self.assertIn("std::unordered_map<void*, GuestMappedBuffer>", core)
        self.assertIn("find(NativeMappedPointer(target))", core)
        self.assertIn("mapping.group == group", core)
        self.assertIn("TrackGlContext(context, share)", libc)
        self.assertIn('manager.RegisterThunk("glGetBufferPointerv"', core)

    def test_no_log_build_skips_verbose_wrapper_and_frame_diagnostics(self):
        log = (ROOT / "src" / "bridge_log.h").read_text(encoding="utf-8")
        self.assertIn("#if !CMR_ANDROID_LOG_PRINT", log)
        self.assertIn("#define LOGI(...) ((void)0)", log)
        for file in ("jni_bridge.cpp", "thunks_libc.cpp", "thunks_opensles.cpp"):
            source = (ROOT / "src" / file).read_text(encoding="utf-8")
            self.assertIn("#if CMR_ANDROID_LOG_PRINT && !defined(BRIDGE_RELEASE_LOGS)", source)

    def test_profiled_hotpaths_are_checksum_pinned_and_non_fused(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        hotpaths = (ROOT / "src" / "guest_hotpaths.cpp").read_text(encoding="utf-8")
        jni = (ROOT / "src" / "jni_bridge.cpp").read_text(encoding="utf-8")

        self.assertIn("InstallUnityProfiledHotPaths(*g_host, cfg::kUnityBase)", jni)
        self.assertIn("0x9F2E28", hotpaths)
        self.assertIn("0xE1A0C00D", hotpaths)
        self.assertIn("0x8EF9F0", hotpaths)
        self.assertIn("0xE1B02F9F", hotpaths)
        self.assertIn("0xE1C020D0", hotpaths)
        self.assertIn("Keep CLREX, DMB ISH", hotpaths)
        self.assertIn("SkinningMathSelfCheck", hotpaths)
        self.assertIn("vst1q_f32(output + 3, normal)", hotpaths)
        self.assertIn("vst1q_f32(output + 6, tangent)", hotpaths)
        self.assertIn("kOutputStride = 10 * sizeof(float)", hotpaths)
        self.assertIn("-ffp-contract=off", cmake)

    def test_intro_logo_skip_is_callsite_scoped_and_checksum_pinned(self):
        hotpaths = (ROOT / "src" / "guest_hotpaths.cpp").read_text(encoding="utf-8")
        jni = (ROOT / "src" / "jni_bridge.cpp").read_text(encoding="utf-8")

        self.assertIn("ApplyIntroLogoSkipPatch(*g_host, cfg::kIl2cppBase)", jni)
        self.assertIn("0x8E2490", hotpaths)
        self.assertIn("0xE5D00008", hotpaths)
        self.assertIn("0xE12FFF1E", hotpaths)
        self.assertIn("0x5B5948", hotpaths)
        self.assertIn("0xEB0CB2D0", hotpaths)
        self.assertIn("0xE3500001", hotpaths)
        self.assertIn("0x1A000014", hotpaths)
        self.assertIn("host.MemoryWrite32(call, 0xE3A00000)", hotpaths)
        self.assertIn("HubspaceController.IsFirstRun", hotpaths)
        self.assertIn("0x8B7F20", hotpaths)
        self.assertIn("0xE30F1000", hotpaths)
        self.assertIn("0xE344359C", hotpaths)
        self.assertIn("0xE30C1000", hotpaths)
        self.assertIn("0xE344152D", hotpaths)
        self.assertIn("0xE3442402", hotpaths)
        self.assertIn("0xE3443554", hotpaths)
        self.assertIn("cameraMotion.zoomOutPosition", hotpaths)
        self.assertIn("0x8B7F84", hotpaths)
        self.assertIn("0xE285300C", hotpaths)
        self.assertIn("0xEB001BD7", hotpaths)
        self.assertIn("0x8B7F9C", hotpaths)
        self.assertIn("0xE3A005F0", hotpaths)
        self.assertIn("0xE3A015F0", hotpaths)
        self.assertIn("divide-by-zero", hotpaths)
        self.assertIn("timeToZoomOutFromSplashScreens", hotpaths)

    def test_main_menu_image_fade_skip_is_checksum_pinned_and_scope_limited(self):
        hotpaths = (ROOT / "src" / "guest_hotpaths.cpp").read_text(encoding="utf-8")
        header = (ROOT / "src" / "guest_hotpaths.h").read_text(encoding="utf-8")
        jni = (ROOT / "src" / "jni_bridge.cpp").read_text(encoding="utf-8")

        self.assertIn("ApplyMainMenuImageFadeSkipPatch", header)
        self.assertIn("ApplyMainMenuImageFadeSkipPatch(*g_host, cfg::kIl2cppBase)", jni)
        self.assertIn("UIMainMenuGameobjectFlashOn.<Flash>.MoveNext", hotpaths)
        self.assertIn("0xDC521C", hotpaths)
        self.assertIn("0xE92D48F0", hotpaths)
        self.assertIn("kObjectFinalActivationOffset = 0x1F8", hotpaths)
        self.assertIn("0xEBFFFF4C", hotpaths)
        self.assertIn("host.MemoryWrite32(object_bypass, 0xEA00006A)", hotpaths)
        self.assertIn("host.MemoryWrite32(object_after_activation, 0xEA000027)", hotpaths)
        self.assertIn("UIMainMenuSpriteTurnOn.<Flash>.MoveNext", hotpaths)
        self.assertIn("0x83289C", hotpaths)
        self.assertIn("0xE92D4C70", hotpaths)
        self.assertIn("0xE3E00000", hotpaths)
        self.assertIn("kSpriteFinalStateOffset = 0x43C", hotpaths)
        self.assertIn("0xED958A05", hotpaths)
        self.assertIn("0xE5840020", hotpaths)
        self.assertIn("host.MemoryWrite32(sprite_bypass, 0xEA000109)", hotpaths)
        self.assertIn("UIMainMenuWidgetTurnOn coroutine stays untouched", hotpaths)

    def test_android_log_print_gate_is_forced_and_disabled_by_default(self):
        gate = (ROOT / "src" / "android_log_gate.h").read_text(encoding="utf-8")
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build_arm64.ps1").read_text(encoding="utf-8")

        self.assertIn("#ifndef CMR_ANDROID_LOG_PRINT", gate)
        self.assertIn("# define CMR_ANDROID_LOG_PRINT 0", gate)
        self.assertIn("# define __android_log_print(...) ((void)0)", gate)
        self.assertIn('-include "${CMAKE_CURRENT_SOURCE_DIR}/src/android_log_gate.h"', cmake)
        self.assertIn("CMR_ANDROID_LOG_PRINT=$<BOOL:${CMR_ANDROID_LOG_PRINT}>", cmake)
        self.assertIn('"-DCMR_ANDROID_LOG_PRINT=$([bool]$AndroidLogPrint)"', build)

        libc = (ROOT / "src" / "thunks_libc.cpp").read_text(encoding="utf-8")
        for thunk in ("__android_log_write", "__android_log_print", "__android_log_vprint"):
            start = libc.index(f'RegisterThunk("{thunk}"')
            end = libc.index("});", start)
            self.assertIn("#if CMR_ANDROID_LOG_PRINT", libc[start:end])


if __name__ == "__main__":
    unittest.main()
