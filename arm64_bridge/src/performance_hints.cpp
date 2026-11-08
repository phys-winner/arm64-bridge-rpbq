#include "performance_hints.h"

#include <android/performance_hint.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <signal.h>
#include <sys/syscall.h>
#include <mutex>

#define LOG_TAG "Arm32Bridge_Perf"
#include "bridge_log.h"

namespace bridge_perf {
namespace {

constexpr int64_t kRefreshFrameDurationNs = 1'000'000'000LL / 120;
// Reserve 10% of the 120 Hz period for scheduler wake-up, IRQ and compositor
// jitter. This is a CPU work budget; it does not alter Unity's 120 FPS target.
constexpr int64_t kTargetFrameDurationNs = kRefreshFrameDurationNs * 9 / 10;
constexpr size_t kMaxSessionThreads = 16;

using GetManagerFn = APerformanceHintManager* (*)();
using CreateSessionFn = APerformanceHintSession* (*)(
        APerformanceHintManager*, const int32_t*, size_t, int64_t);
using ReportDurationFn = int (*)(APerformanceHintSession*, int64_t);
using SetThreadsFn = int (*)(APerformanceHintSession*, const pid_t*, size_t);
using SetPowerEfficiencyFn = int (*)(APerformanceHintSession*, bool);
using CloseSessionFn = void (*)(APerformanceHintSession*);

struct State {
    std::mutex mutex;
    void* library = nullptr;
    GetManagerFn get_manager = nullptr;
    CreateSessionFn create_session = nullptr;
    ReportDurationFn report_duration = nullptr;
    SetThreadsFn set_threads = nullptr;
    SetPowerEfficiencyFn set_power_efficiency = nullptr;
    CloseSessionFn close_session = nullptr;
    std::atomic<APerformanceHintSession*> session{nullptr};
    pid_t render_tid = 0;
    std::array<pid_t, kMaxSessionThreads> auxiliary_tids{};
    size_t auxiliary_count = 0;
    bool load_attempted = false;
    std::chrono::steady_clock::time_point next_attempt{};
};

State& GetState() {
    static State state;
    return state;
}

void AddAuxiliaryThread(pid_t tid) {
    if (tid <= 0) return;
    State& state = GetState();
    std::lock_guard lock(state.mutex);
    const auto begin = state.auxiliary_tids.begin();
    const auto end = begin + state.auxiliary_count;
    if (std::find(begin, end, tid) != end || state.auxiliary_count == kMaxSessionThreads) return;
    state.auxiliary_tids[state.auxiliary_count++] = tid;

    APerformanceHintSession* session = state.session.load(std::memory_order_acquire);
    if (session && state.set_threads) {
        std::array<pid_t, kMaxSessionThreads + 1> tids{};
        tids[0] = state.render_tid;
        std::copy_n(state.auxiliary_tids.begin(), state.auxiliary_count, tids.begin() + 1);
        const int result = state.set_threads(session, tids.data(), state.auxiliary_count + 1);
        LOGI("ADPF session thread update: count=%zu result=%d", state.auxiliary_count + 1, result);
    }
}

bool LoadApi(State& state) {
    if (state.load_attempted) return state.get_manager && state.create_session &&
                                     state.report_duration && state.close_session;
    state.load_attempted = true;
    state.library = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
    if (!state.library) return false;
    state.get_manager = reinterpret_cast<GetManagerFn>(dlsym(state.library, "APerformanceHint_getManager"));
    state.create_session = reinterpret_cast<CreateSessionFn>(dlsym(state.library, "APerformanceHint_createSession"));
    state.report_duration = reinterpret_cast<ReportDurationFn>(
            dlsym(state.library, "APerformanceHint_reportActualWorkDuration"));
    state.set_threads = reinterpret_cast<SetThreadsFn>(dlsym(state.library, "APerformanceHint_setThreads"));
    state.set_power_efficiency = reinterpret_cast<SetPowerEfficiencyFn>(
            dlsym(state.library, "APerformanceHint_setPreferPowerEfficiency"));
    state.close_session = reinterpret_cast<CloseSessionFn>(
            dlsym(state.library, "APerformanceHint_closeSession"));
    return state.get_manager && state.create_session && state.report_duration && state.close_session;
}

}  // namespace

void NoteGraphicsThread(pid_t tid) {
    AddAuxiliaryThread(tid);
}

void NoteWorkerThread(pid_t tid) {
    AddAuxiliaryThread(tid);
}

void TuneRenderThread() {
    // Android's scheduler and ADPF own placement. Hard-pinning the render
    // thread also forced Unity workers away from the fastest CPUs on common
    // 8-core Qualcomm topologies, so it could turn a worker dependency into a
    // frame stall even when CPU capacity was otherwise available.
}

void TuneGraphicsThread() {
}

void TuneWorkerThread() {
}

void TuneAudioThread() {
}

void BeginRenderFrame() {
    State& state = GetState();
    if (state.session.load(std::memory_order_acquire)) return;
    std::lock_guard lock(state.mutex);
    if (state.session.load(std::memory_order_relaxed) || !LoadApi(state)) return;

    const auto now = std::chrono::steady_clock::now();
    if (now < state.next_attempt) return;
    // Unavailable/vendor-rejected sessions must not trigger a Binder call every frame.
    state.next_attempt = now + std::chrono::seconds(2);
    state.render_tid = gettid();
    auto begin = state.auxiliary_tids.begin();
    const auto end = std::remove_if(begin, begin + state.auxiliary_count, [&](pid_t tid) {
        return tid == state.render_tid ||
               (syscall(SYS_tgkill, getpid(), tid, 0) == -1 && errno == ESRCH);
    });
    state.auxiliary_count = static_cast<size_t>(end - begin);
    std::array<int32_t, kMaxSessionThreads + 1> tids{};
    tids[0] = state.render_tid;
    for (size_t i = 0; i < state.auxiliary_count; ++i) {
        tids[i + 1] = state.auxiliary_tids[i];
    }
    APerformanceHintManager* manager = state.get_manager();
    if (!manager) return;
    APerformanceHintSession* session = state.create_session(
            manager, tids.data(), state.auxiliary_count + 1, kTargetFrameDurationNs);
    if (!session) {
        LOGI("ADPF performance-hint session unavailable");
        return;
    }
    if (state.set_power_efficiency) state.set_power_efficiency(session, false);
    state.session.store(session, std::memory_order_release);
    LOGI("ADPF performance-hint session active: target_ns=%lld threads=%zu",
         static_cast<long long>(kTargetFrameDurationNs), state.auxiliary_count + 1);
}

void EndRenderFrame(int64_t actual_duration_ns) {
    if (actual_duration_ns <= 0) return;
    State& state = GetState();
    // Hold the same lock as Shutdown through the native report call.
    std::lock_guard lock(state.mutex);
    APerformanceHintSession* session = state.session.load(std::memory_order_acquire);
    if (!session || !state.report_duration) return;
    state.report_duration(session, actual_duration_ns);
}

void Shutdown() {
    State& state = GetState();
    std::lock_guard lock(state.mutex);
    APerformanceHintSession* session = state.session.exchange(nullptr, std::memory_order_acq_rel);
    if (session && state.close_session) state.close_session(session);
    state.render_tid = 0;
    state.next_attempt = {};
    state.auxiliary_tids.fill(0);
    state.auxiliary_count = 0;
}

}  // namespace bridge_perf
