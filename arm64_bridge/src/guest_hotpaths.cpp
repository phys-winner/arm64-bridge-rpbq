#include "guest_hotpaths.h"

#include <algorithm>
#include <arm_neon.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#define LOG_TAG "Arm32Bridge_Hotpaths"
#include "bridge_log.h"
#include "dynarmic_host.h"
#include "thunk_manager.h"
#include "thunk_types.h"

namespace {

constexpr std::uint32_t kArmSvcMask = 0xFF000000u;
constexpr std::uint32_t kArmSvcOpcode = 0xEF000000u;

void ReturnToGuestCaller(Dynarmic::A32::Jit& jit) {
    auto& regs = jit.Regs();
    const std::uint32_t return_address = regs[14];
    jit.SetCpsr((jit.Cpsr() & ~0x20u) | ((return_address & 1u) ? 0x20u : 0u));
    regs[15] = return_address & ~1u;
}

template <std::size_t Size>
bool VerifyGuestWords(DynarmicHost& host, std::uint32_t entry,
                      const std::array<std::uint32_t, Size>& expected,
                      const char* label) {
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const std::uint32_t address = entry + static_cast<std::uint32_t>(index * 4);
        const std::uint32_t actual = host.MemoryRead32(address);
        if (actual != expected[index]) {
            LOGE("%s: unexpected opcode at 0x%08X: expected=0x%08X actual=0x%08X",
                 label, address, expected[index], actual);
            return false;
        }
    }
    return true;
}

bool ReplaceGuestFunction(DynarmicHost& host, std::uint32_t entry,
                          const char* thunk_name, ThunkHandler handler,
                          const char* label) {
    const std::uint32_t thunk = host.GetThunkManager().RegisterThunk(thunk_name, std::move(handler));
    const std::uint32_t svc = host.MemoryRead32(thunk);
    if ((svc & kArmSvcMask) != kArmSvcOpcode) {
        LOGE("%s: invalid thunk at 0x%08X", label, thunk);
        return false;
    }
    host.MemoryWrite32(entry, svc);
    host.InvalidateGuestCode(entry, sizeof(std::uint32_t));
    LOGI("%s installed at 0x%08X", label, entry);
    return true;
}

float32x4_t MultiplyAdd(float32x4_t accumulator, float32x4_t value, float scalar) {
    // Keep ARMv7 NEON VMLA's multiply-then-add rounding. This file is also
    // compiled with -ffp-contract=off so Clang cannot turn this into FMLA.
    return vaddq_f32(accumulator, vmulq_n_f32(value, scalar));
}

void SkinOneVertex(const float* const matrices[4], const float weights[4],
                   const float input[10], float output[10]) {
    float32x4_t blended[4];
    for (unsigned row = 0; row < 4; ++row) {
        blended[row] = vmulq_n_f32(vld1q_f32(matrices[0] + row * 4), weights[0]);
        blended[row] = MultiplyAdd(blended[row], vld1q_f32(matrices[1] + row * 4), weights[1]);
        blended[row] = MultiplyAdd(blended[row], vld1q_f32(matrices[2] + row * 4), weights[2]);
        blended[row] = MultiplyAdd(blended[row], vld1q_f32(matrices[3] + row * 4), weights[3]);
    }

    float32x4_t position = vmulq_n_f32(blended[0], input[0]);
    position = MultiplyAdd(position, blended[1], input[1]);
    position = MultiplyAdd(position, blended[2], input[2]);
    position = vaddq_f32(position, blended[3]);

    float32x4_t normal = vmulq_n_f32(blended[0], input[3]);
    normal = MultiplyAdd(normal, blended[1], input[4]);
    normal = MultiplyAdd(normal, blended[2], input[5]);

    float32x4_t tangent = vmulq_n_f32(blended[0], input[6]);
    tangent = MultiplyAdd(tangent, blended[1], input[7]);
    tangent = MultiplyAdd(tangent, blended[2], input[8]);
    tangent = vsetq_lane_f32(input[9], tangent, 3);

    // The ARM32 routine deliberately advances the first two 16-byte VST1s by
    // 12 bytes, then advances the last one by 16. The stores overlap their W
    // lanes and produce a packed 10-float record: position.xyz, normal.xyz,
    // tangent.xyzw. Reproducing that overlap is essential; a 12-float stride
    // corrupts every following vertex.
    vst1q_f32(output + 0, position);
    vst1q_f32(output + 3, normal);
    vst1q_f32(output + 6, tangent);
}

bool SkinningMathSelfCheck() {
    alignas(16) const float palette[4][16] = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {2, 0, 1, 3, 4, 2, 0, 1, 6, 4, 2, 0, 8, 6, 4, 2},
        {0, 1, 0, 1, 1, 0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 2},
        {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3},
    };
    const float* matrices[4] = {palette[0], palette[1], palette[2], palette[3]};
    const float weights[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float input[10] = {1.25f, -2.0f, 0.75f, 0.5f, 1.5f,
                             -0.25f, 2.0f, -1.0f, 0.125f, -1.0f};
    float actual[10]{};
    SkinOneVertex(matrices, weights, input, actual);

    float blended[4][4]{};
    for (unsigned row = 0; row < 4; ++row) {
        for (unsigned lane = 0; lane < 4; ++lane) {
            for (unsigned matrix = 0; matrix < 4; ++matrix) {
                blended[row][lane] += palette[matrix][row * 4 + lane] * weights[matrix];
            }
        }
    }
    float expected[10]{};
    for (unsigned lane = 0; lane < 3; ++lane) {
        expected[lane] = blended[0][lane] * input[0] + blended[1][lane] * input[1] +
                         blended[2][lane] * input[2] + blended[3][lane];
        expected[3 + lane] = blended[0][lane] * input[3] + blended[1][lane] * input[4] +
                             blended[2][lane] * input[5];
    }
    for (unsigned lane = 0; lane < 3; ++lane) {
        expected[6 + lane] = blended[0][lane] * input[6] + blended[1][lane] * input[7] +
                             blended[2][lane] * input[8];
    }
    expected[9] = input[9];

    for (unsigned index = 0; index < 10; ++index) {
        const float tolerance = 1.0e-5f * std::max(1.0f, std::fabs(expected[index]));
        if (std::fabs(actual[index] - expected[index]) > tolerance) {
            LOGE("Unity skinning self-check failed at lane %u: expected=%g actual=%g",
                 index, expected[index], actual[index]);
            return false;
        }
    }
    return true;
}

bool RunSkinningHotPath(DynarmicHost& host, std::uint32_t palette_address,
                        std::uint32_t source_address, std::uint32_t source_end,
                        std::uint32_t skin_address, std::uint32_t output_address,
                        std::size_t& vertex_count) {
    constexpr std::size_t kSourceStride = 10 * sizeof(float);
    constexpr std::size_t kSkinStride = 4 * sizeof(float) + 4 * sizeof(std::uint32_t);
    constexpr std::size_t kOutputStride = 10 * sizeof(float);

    if (source_end < source_address || ((source_end - source_address) % kSourceStride) != 0) {
        return false;
    }
    vertex_count = (source_end - source_address) / kSourceStride;
    if (!GuestRangeValid(host.GetGuestMemorySize(), source_address, vertex_count * kSourceStride) ||
        !GuestRangeValid(host.GetGuestMemorySize(), skin_address, vertex_count * kSkinStride) ||
        !GuestRangeValid(host.GetGuestMemorySize(), output_address, vertex_count * kOutputStride)) {
        return false;
    }

    std::uint8_t* const guest = host.GetGuestMemoryBase();
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        const std::uint8_t* const skin = guest + skin_address + vertex * kSkinStride;
        float weights[4];
        std::uint32_t indices[4];
        float input[10];
        std::memcpy(weights, skin, sizeof(weights));
        std::memcpy(indices, skin + sizeof(weights), sizeof(indices));
        std::memcpy(input, guest + source_address + vertex * kSourceStride, sizeof(input));

        const float* matrices[4];
        for (unsigned index = 0; index < 4; ++index) {
            const std::uint64_t matrix_address = static_cast<std::uint64_t>(palette_address) +
                                                 static_cast<std::uint64_t>(indices[index]) * 64;
            if (matrix_address + 64 > host.GetGuestMemorySize()) return false;
            matrices[index] = reinterpret_cast<const float*>(guest + matrix_address);
        }
        SkinOneVertex(matrices, weights, input,
                      reinterpret_cast<float*>(guest + output_address + vertex * kOutputStride));
    }
    return true;
}

bool InstallUnitySkinningHotPath(DynarmicHost& host, std::uint32_t unity_base) {
    constexpr std::uint32_t kRva = 0x9F2E28;
    constexpr std::array<std::uint32_t, 4> kExpectedWords = {
        0xE1A0C00D,  // mov ip, sp
        0xED2D8B0A,  // vpush {d8-d12}
        0xE92D01F0,  // push {r4-r8}
        0xE59C4000,  // ldr r4, [ip]
    };
    const std::uint32_t entry = unity_base + kRva;
    if (!VerifyGuestWords(host, entry, kExpectedWords, "Unity skinning hotpath") ||
        !SkinningMathSelfCheck()) {
        return false;
    }

    return ReplaceGuestFunction(
        host, entry, "unity_skinning_4weights_host",
        [](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            auto& regs = jit.Regs();
            std::size_t vertices = 0;
            bool valid = GuestRangeValid(h.GetGuestMemorySize(), regs[13], sizeof(std::uint32_t));
            const std::uint32_t output_address = valid ? h.MemoryRead32(regs[13]) : 0;
            if (valid) {
                valid = RunSkinningHotPath(h, regs[0], regs[1], regs[2], regs[3],
                                            output_address, vertices);
            }
            if (!valid) {
                static std::atomic<unsigned> errors{0};
                const unsigned error = errors.fetch_add(1, std::memory_order_relaxed) + 1;
                if (error <= 8) {
                    LOGE("Unity skinning hotpath invalid guest range: palette=0x%08X src=0x%08X "
                         "end=0x%08X skin=0x%08X out=0x%08X",
                         regs[0], regs[1], regs[2], regs[3], output_address);
                }
            }
#if !defined(BRIDGE_RELEASE_LOGS)
            thread_local std::uint64_t calls = 0;
            thread_local std::uint64_t total_vertices = 0;
            ++calls;
            total_vertices += vertices;
            if (calls == 1 || (calls % 8192) == 0) {
                LOGI("Unity skinning hotpath: guest_tid=%u calls=%llu vertices=%llu",
                     DynarmicHost::CurrentGuestThreadId(),
                     static_cast<unsigned long long>(calls),
                     static_cast<unsigned long long>(total_vertices));
            }
#endif
            ReturnToGuestCaller(jit);
        },
        "Unity skinning host hotpath");
}

bool ApplyUnityAtomicRead64GuestPatch(DynarmicHost& host, std::uint32_t unity_base) {
    // This helper takes an exclusive 64-bit snapshot and immediately abandons
    // the reservation. It never executes STREXD, so creating the reservation
    // has no observable purpose. On the ARM64 host an aligned LDR X is itself a
    // single-copy atomic 64-bit read; Dynarmic emits exactly that for A32 LDRD.
    // Keep CLREX, DMB ISH and the result/store sequence unchanged so any older
    // local reservation is still cleared exactly as in the original helper.
    constexpr std::uint32_t kRva = 0x8EF9F0;
    constexpr std::array<std::uint32_t, 6> kExpectedWords = {
        0xE1B02F9F,  // ldrexd r2, [r0]
        0xF57FF01F,  // clrex
        0xF57FF05B,  // dmb ish
        0xE1A00002,  // mov r0, r2
        0xE5813000,  // str r3, [r1]
        0xE12FFF1E,  // bx lr
    };
    const std::uint32_t entry = unity_base + kRva;
    if (!VerifyGuestWords(host, entry, kExpectedWords, "Unity atomic-read64 guest patch")) {
        return false;
    }

    host.MemoryWrite32(entry, 0xE1C020D0);  // ldrd r2, r3, [r0]
    host.InvalidateGuestCode(entry, sizeof(std::uint32_t));
    LOGI("Unity atomic-read64 guest patch installed at 0x%08X (LDREXD -> LDRD)", entry);
    return true;
}

bool ApplyIntroLogoSkipGuestPatch(DynarmicHost& host, std::uint32_t il2cpp_base) {
    // Il2CppDumper identifies AppSaveData.get_ShouldShowIntroSplashScreens at
    // RVA 0x8E2490.  It has two callers in this build.  Patch only the call in
    // IntroSpawner.Start so HubspaceController.IsFirstRun keeps the authored
    // save-data behavior.  Replacing BL getter with MOV r0,#0 selects the
    // existing InitializeMainMenu branch without changing any serialized data.
    constexpr std::uint32_t kGetterRva = 0x8E2490;
    constexpr std::array<std::uint32_t, 2> kExpectedGetterWords = {
        0xE5D00008,  // ldrb r0, [r0, #8]
        0xE12FFF1E,  // bx lr
    };
    constexpr std::uint32_t kIntroSpawnerCallRva = 0x5B5948;
    constexpr std::array<std::uint32_t, 3> kExpectedCallWords = {
        0xEB0CB2D0,  // bl AppSaveData.get_ShouldShowIntroSplashScreens
        0xE3500001,  // cmp r0, #1
        0x1A000014,  // bne InitializeMainMenu
    };
    // On Android, UIMainMenu.Initialize first places cameraMotion's Transform at
    // a splash close-up and then creates an iTween to its authored
    // zoomOutPosition.  With the splash cards disabled that close-up must never
    // reach the display: replace its Vector3 constants with the exact
    // zoomOutPosition constants initialized by UIMainMenuCameraMotion.Awake.
    // Keep the authored iTween/state-cleanup path, but reduce both copies of
    // timeToZoomOutFromSplashScreens to 1/128 second.  The no-op tween then
    // completes inside the first 120 Hz frame while remaining non-zero so
    // iTween never sees a divide-by-zero.
    constexpr std::uint32_t kSplashStartPositionRva = 0x8B7F20;
    constexpr std::array<std::uint32_t, 13> kExpectedSplashStartPositionWords = {
        0xE30F1000,  // movw r1, #0xF000 (splash close-up X low)
        0xE3002000,  // movw r2, #0x0000 (splash close-up Y low)
        0xE3043000,  // movw r3, #0x4000 (splash close-up Z low)
        0xE1A05000,  // mov r5, r0 (cameraMotion Transform)
        0xE28D0008,  // add r0, sp, #8 (temporary Vector3)
        0xE3441518,  // movt r1, #0x4518 (splash close-up X high)
        0xE344243E,  // movt r2, #0x443E (splash close-up Y high)
        0xE344359C,  // movt r3, #0x459C (splash close-up Z high)
        0xE58D6010,  // str r6, [sp, #16]
        0xE58D600C,  // str r6, [sp, #12]
        0xE58D6008,  // str r6, [sp, #8]
        0xE58D6000,  // str r6, [sp] (method)
        0xEB64EC11,  // bl Vector3.ctor
    };
    constexpr std::uint32_t kSplashZoomRva = 0x8B7F84;
    constexpr std::array<std::uint32_t, 6> kExpectedSplashZoomWords = {
        0xE285300C,  // add r3, r5, #12 (cameraMotion.zoomOutPosition)
        0xE3A005FF,  // mov r0, #1.5f
        0xE893000E,  // ldm r3, {r1, r2, r3}
        0xE58D0000,  // str r0, [sp] (duration)
        0xE1A00005,  // mov r0, r5 (cameraMotion)
        0xEB001BD7,  // bl UIMainMenuCameraMotion.StartMotionZoom
    };
    constexpr std::uint32_t kSplashTimeLockRva = 0x8B7F9C;
    constexpr std::array<std::uint32_t, 8> kExpectedSplashTimeLockWords = {
        0xE1A00004,  // mov r0, r4
        0xE3A015FF,  // mov r1, #1.5f
        0xEB0000DB,  // bl UIMainMenu.TimeLock
        0xE1A01000,  // mov r1, r0
        0xE1A00004,  // mov r0, r4
        0xE3A02000,  // mov r2, #0
        0xEB4C8CC6,  // bl MonoBehaviour.StartCoroutine
        0xEA000010,  // b continuation
    };
    const std::uint32_t getter = il2cpp_base + kGetterRva;
    const std::uint32_t call = il2cpp_base + kIntroSpawnerCallRva;
    const std::uint32_t splash_start_position = il2cpp_base + kSplashStartPositionRva;
    const std::uint32_t splash_zoom = il2cpp_base + kSplashZoomRva;
    const std::uint32_t splash_time_lock = il2cpp_base + kSplashTimeLockRva;
    if (!VerifyGuestWords(host, getter, kExpectedGetterWords,
                          "AppSaveData intro-logo getter") ||
        !VerifyGuestWords(host, call, kExpectedCallWords,
                          "IntroSpawner intro-logo call site") ||
        !VerifyGuestWords(host, splash_start_position, kExpectedSplashStartPositionWords,
                          "UIMainMenu splash start position") ||
        !VerifyGuestWords(host, splash_zoom, kExpectedSplashZoomWords,
                          "UIMainMenu splash zoom") ||
        !VerifyGuestWords(host, splash_time_lock, kExpectedSplashTimeLockWords,
                          "UIMainMenu splash time lock")) {
        return false;
    }

    host.MemoryWrite32(call, 0xE3A00000);  // mov r0, #0
    // Vector3(2780.0f, 520.0f, 3400.0f): cameraMotion.zoomOutPosition.
    host.MemoryWrite32(splash_start_position, 0xE30C1000);       // movw r1, #0xC000
    host.MemoryWrite32(splash_start_position + 8, 0xE3083000);   // movw r3, #0x8000
    host.MemoryWrite32(splash_start_position + 20, 0xE344152D);  // movt r1, #0x452D
    host.MemoryWrite32(splash_start_position + 24, 0xE3442402);  // movt r2, #0x4402
    host.MemoryWrite32(splash_start_position + 28, 0xE3443554);  // movt r3, #0x4554
    host.MemoryWrite32(splash_zoom + 4, 0xE3A005F0);  // mov r0, #0.0078125f
    host.MemoryWrite32(splash_time_lock + 4, 0xE3A015F0);  // mov r1, #0.0078125f
    host.InvalidateGuestCode(call, sizeof(std::uint32_t));
    host.InvalidateGuestCode(splash_start_position, 3 * sizeof(std::uint32_t));
    host.InvalidateGuestCode(splash_start_position + 20, 3 * sizeof(std::uint32_t));
    host.InvalidateGuestCode(splash_zoom + 4, sizeof(std::uint32_t));
    host.InvalidateGuestCode(splash_time_lock + 4, sizeof(std::uint32_t));
    LOGI("Intro logo and splash camera skip installed at 0x%08X/0x%08X/0x%08X",
         call, splash_start_position, splash_zoom);
    return true;
}

bool ApplyMainMenuImageFadeSkipGuestPatch(DynarmicHost& host, std::uint32_t il2cpp_base) {
    // Il2CppDumper maps the television-image reveal to two nested coroutines.
    // UIMainMenuGameobjectFlashOn.<Flash>.MoveNext (RVA 0xDC521C) waits for a
    // serialized per-image delay before activating the image GameObject, then
    // flashes it three times.  Its final state activates the target, decrements
    // flashCount and completes.  Jump to that activation on the first call and
    // skip the decrement because the bypass also skips the matching increment;
    // IsFlashing therefore truthfully remains false for an immediate reveal.
    //
    // Activating the target starts UIMainMenuSpriteTurnOn.<Flash>.MoveNext
    // (RVA 0x83289C).  That inner state machine alternates UISprite alpha
    // between cachedAlpha and zero seven times.  Jump to its authored final
    // state, which restores cachedAlpha and completes normally.  The separate
    // UIMainMenuWidgetTurnOn coroutine stays untouched, preserving navigation
    // animations unrelated to the television images.
    constexpr std::uint32_t kObjectMoveNextRva = 0xDC521C;
    constexpr std::uint32_t kObjectBypassOffset = 0x48;
    constexpr std::uint32_t kObjectFinalActivationOffset = 0x1F8;
    constexpr std::uint32_t kObjectAfterActivationOffset = 0x218;
    constexpr std::uint32_t kObjectCompletionOffset = 0x2BC;
    constexpr std::array<std::uint32_t, 3> kExpectedObjectEntryWords = {
        0xE92D48F0,  // push {r4-r7,r11,lr}
        0xE28DB010,  // add r11,sp,#16
        0xE1A04000,  // mov r4,r0 (iterator)
    };
    constexpr std::array<std::uint32_t, 4> kExpectedObjectBypassWords = {
        0xE594001C,  // ldr r0,[r4,#28] ($PC)
        0xE3E01000,  // mvn r1,#0
        0xE3A05000,  // mov r5,#0 (MoveNext result)
        0xE584101C,  // str r1,[r4,#28] (normal dispatch begins)
    };
    constexpr std::array<std::uint32_t, 8> kExpectedObjectFinalActivationWords = {
        0xE5946008,  // ldr r6,[r4,#8] (target)
        0xE5940010,  // ldr r0,[r4,#16] ($this)
        0xE3500000,  // cmp r0,#0
        0x1A000000,  // bne SetObjectStates call
        0xEB53CA18,  // bl null-check helper
        0xE1A01006,  // mov r1,r6 (target)
        0xE3A02001,  // mov r2,#1 (active)
        0xEBFFFF4C,  // bl UIMainMenuGameobjectFlashOn.SetObjectStates
    };
    constexpr std::array<std::uint32_t, 4> kExpectedObjectCompletionWords = {
        0xE3E00000,  // mvn r0,#0
        0xE584001C,  // str r0,[r4,#28] ($PC = -1)
        0xE1A00005,  // mov r0,r5 (false)
        0xE8BD88F0,  // pop {r4-r7,r11,pc}
    };

    constexpr std::uint32_t kSpriteMoveNextRva = 0x83289C;
    constexpr std::uint32_t kSpriteBypassOffset = 0x10;
    constexpr std::uint32_t kSpriteFinalStateOffset = 0x43C;
    constexpr std::uint32_t kSpriteCompletionOffset = 0x488;
    constexpr std::array<std::uint32_t, 5> kExpectedSpriteEntryWords = {
        0xE92D4C70,  // push {r4-r6,r10,r11,lr}
        0xE28DB010,  // add r11,sp,#16
        0xED2D8B02,  // vpush {d8}
        0xE1A04000,  // mov r4,r0 (iterator)
        0xE3E00000,  // mvn r0,#0 (normal state-machine dispatch begins)
    };
    constexpr std::array<std::uint32_t, 5> kExpectedSpriteFinalStateWords = {
        0xE5945014,  // ldr r5,[r4,#20] ($this)
        0xE3550000,  // cmp r5,#0
        0x0A000001,  // beq null-check path
        0xED958A05,  // vldr s16,[r5,#20] (cachedAlpha)
        0xEA000003,  // b load-target path
    };
    constexpr std::array<std::uint32_t, 5> kExpectedSpriteCompletionWords = {
        0xE3E00000,  // mvn r0,#0
        0xE5840020,  // str r0,[r4,#32] ($PC = -1)
        0xE3A00000,  // mov r0,#0 (MoveNext returns false)
        0xECBD8B02,  // vpop {d8}
        0xE8BD8C70,  // pop {r4-r6,r10,r11,pc}
    };

    const std::uint32_t object_move_next = il2cpp_base + kObjectMoveNextRva;
    const std::uint32_t object_bypass = object_move_next + kObjectBypassOffset;
    const std::uint32_t object_final_activation =
        object_move_next + kObjectFinalActivationOffset;
    const std::uint32_t object_after_activation =
        object_move_next + kObjectAfterActivationOffset;
    const std::uint32_t object_completion = object_move_next + kObjectCompletionOffset;
    const std::uint32_t sprite_move_next = il2cpp_base + kSpriteMoveNextRva;
    const std::uint32_t sprite_bypass = sprite_move_next + kSpriteBypassOffset;
    const std::uint32_t sprite_final_state = sprite_move_next + kSpriteFinalStateOffset;
    const std::uint32_t sprite_completion = sprite_move_next + kSpriteCompletionOffset;
    if (!VerifyGuestWords(host, object_move_next, kExpectedObjectEntryWords,
                          "UIMainMenuGameobjectFlashOn Flash MoveNext") ||
        !VerifyGuestWords(host, object_bypass - 0xC, kExpectedObjectBypassWords,
                          "UIMainMenuGameobjectFlashOn Flash dispatch") ||
        !VerifyGuestWords(host, object_final_activation,
                          kExpectedObjectFinalActivationWords,
                          "UIMainMenuGameobjectFlashOn Flash final activation") ||
        host.MemoryRead32(object_after_activation) != 0xE5946010 ||
        !VerifyGuestWords(host, object_completion, kExpectedObjectCompletionWords,
                          "UIMainMenuGameobjectFlashOn Flash completion") ||
        !VerifyGuestWords(host, sprite_move_next, kExpectedSpriteEntryWords,
                          "UIMainMenuSpriteTurnOn Flash MoveNext") ||
        !VerifyGuestWords(host, sprite_final_state, kExpectedSpriteFinalStateWords,
                          "UIMainMenuSpriteTurnOn Flash final state") ||
        !VerifyGuestWords(host, sprite_completion, kExpectedSpriteCompletionWords,
                          "UIMainMenuSpriteTurnOn Flash completion")) {
        return false;
    }

    // ARM B uses PC = instruction address + 8.
    host.MemoryWrite32(object_bypass, 0xEA00006A);  // b object_final_activation
    host.MemoryWrite32(object_after_activation, 0xEA000027);  // b object_completion
    host.MemoryWrite32(sprite_bypass, 0xEA000109);  // b sprite_final_state
    host.InvalidateGuestCode(object_bypass, sizeof(std::uint32_t));
    host.InvalidateGuestCode(object_after_activation, sizeof(std::uint32_t));
    host.InvalidateGuestCode(sprite_bypass, sizeof(std::uint32_t));
    LOGI("Main-menu image fade skip installed at 0x%08X/0x%08X/0x%08X",
         object_bypass, object_after_activation, sprite_bypass);
    return true;
}

}  // namespace

bool InstallUnityProfiledHotPaths(DynarmicHost& host, std::uint32_t unity_base) {
    const bool installed = InstallUnitySkinningHotPath(host, unity_base) &&
                           ApplyUnityAtomicRead64GuestPatch(host, unity_base);
    if (installed) LOGI("Profile-guided Unity hotpaths installed");
    return installed;
}

bool ApplyIntroLogoSkipPatch(DynarmicHost& host, std::uint32_t il2cpp_base) {
    return ApplyIntroLogoSkipGuestPatch(host, il2cpp_base);
}

bool ApplyMainMenuImageFadeSkipPatch(DynarmicHost& host, std::uint32_t il2cpp_base) {
    return ApplyMainMenuImageFadeSkipGuestPatch(host, il2cpp_base);
}
