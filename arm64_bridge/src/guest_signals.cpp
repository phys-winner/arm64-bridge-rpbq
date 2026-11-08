#include "guest_signals.h"

#include <android/log.h>
#include <signal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include <dynarmic/interface/A32/a32.h>

#include "dynarmic_host.h"

namespace {
constexpr int kMaxSignals = 64;
constexpr int kRealtimeFirst = 32;
constexpr uint64_t kUnmaskable = (uint64_t{1} << (SIGKILL - 1)) | (uint64_t{1} << (SIGSTOP - 1));
#if defined(SA_SIGINFO)
constexpr uint32_t kSaSiginfo = static_cast<uint32_t>(SA_SIGINFO);
#else
constexpr uint32_t kSaSiginfo = 4u;
#endif
#if defined(SA_NODEFER)
constexpr uint32_t kSaNodefer = static_cast<uint32_t>(SA_NODEFER);
#else
constexpr uint32_t kSaNodefer = 0x40000000u;
#endif

#define LOG_TAG "Arm32Bridge_Signals"

using ActionTable = std::array<GuestSignalAction, kMaxSignals + 1>;
std::mutex g_actions_mutex;
ActionTable g_actions{};
std::mutex g_threads_mutex;

struct SigThread {
    explicit SigThread(int id, Dynarmic::A32::Jit* jit) : tid(id), base(jit) {
        if (jit) jits.push_back(jit);
    }
    const int tid;
    Dynarmic::A32::Jit* const base;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::deque<int> pending;
    std::atomic<uint32_t> pending_count{0};
    std::mutex jits_mutex;
    std::vector<Dynarmic::A32::Jit*> jits;
    std::atomic<bool> retired{false};
};

std::map<int, std::shared_ptr<SigThread>> g_threads;
thread_local std::shared_ptr<SigThread> t_self;
thread_local uint64_t t_mask = 0;
thread_local uint64_t t_active_signals = 0;
std::atomic<unsigned> g_send_diagnostics{0};
std::atomic<unsigned> g_delivery_diagnostics{0};

ActionTable Actions() {
    std::lock_guard<std::mutex> lock(g_actions_mutex);
    return g_actions;
}

std::shared_ptr<SigThread> FindThread(int tid) {
    std::lock_guard<std::mutex> lock(g_threads_mutex);
    const auto it = g_threads.find(tid);
    return it == g_threads.end() ? nullptr : it->second;
}

void Retire(const std::shared_ptr<SigThread>& state) {
    {
        std::lock_guard<std::mutex> lock(state->jits_mutex);
        state->retired.store(true, std::memory_order_release);
        state->jits.clear();
    }
    {
        std::lock_guard<std::mutex> lock(state->queue_mutex);
        state->pending.clear();
        state->pending_count.store(0, std::memory_order_release);
    }
    state->cv.notify_all();
}

bool SoftwareHandler(const GuestSignalAction& action, int signum) {
    return action.handler > 1 &&
           ((action.flags & kSaSiginfo) == 0 || signum == 24 || signum == 30);
}

bool CanDeliver(const GuestSignalAction& action, int signum, uint64_t mask, uint64_t active) {
    const uint64_t bit = uint64_t{1} << (signum - 1);
    const bool nodefer = (action.flags & kSaNodefer) != 0;
    return SoftwareHandler(action, signum) && (mask & bit) == 0 &&
           (nodefer || (active & bit) == 0);
}

int SendResult(int tid, int signum, int result) {
#if defined(BRIDGE_RELEASE_LOGS)
    (void)tid;
    (void)signum;
    return result;
#else
    if (g_send_diagnostics.fetch_add(1, std::memory_order_relaxed) < 12) {
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                            "send tid=%d sig=%d result=%d", tid, signum, result);
    }
    return result;
#endif
}

}  // namespace

void GuestSignalsReset() {
    {
        std::lock_guard<std::mutex> lock(g_actions_mutex);
        for (auto& action : g_actions) action = {};
    }
    {
        std::lock_guard<std::mutex> lock(g_threads_mutex);
        for (const auto& entry : g_threads) Retire(entry.second);
        g_threads.clear();
    }
    t_self.reset();
    t_mask = 0;
    t_active_signals = 0;
    g_send_diagnostics.store(0, std::memory_order_relaxed);
    g_delivery_diagnostics.store(0, std::memory_order_relaxed);
}

int GuestSignalActionSet(int signum, const GuestSignalAction* action, GuestSignalAction* old) {
    if (signum < 1 || signum > kMaxSignals) return EINVAL;
    if (action && (signum == SIGKILL || signum == SIGSTOP)) return EINVAL;

    {
        std::lock_guard<std::mutex> lock(g_actions_mutex);
        if (old) *old = g_actions[signum];
        if (action) g_actions[signum] = *action;
    }
    return 0;
}

void GuestSignalRegisterThread(int tid, Dynarmic::A32::Jit* base) {
    auto state = std::make_shared<SigThread>(tid, base);
    {
        std::lock_guard<std::mutex> lock(g_threads_mutex);
        const auto it = g_threads.find(tid);
        if (it != g_threads.end()) Retire(it->second);
        g_threads[tid] = state;
    }
    // Keep the inherited mask: the parent sets it before registering this TLS state.
    t_self = std::move(state);
}

void GuestSignalAddThreadJit(Dynarmic::A32::Jit* jit) {
    if (!t_self || !jit) return;
    std::lock_guard<std::mutex> lock(t_self->jits_mutex);
    if (!t_self->retired.load(std::memory_order_acquire)) t_self->jits.push_back(jit);
}

void GuestSignalUnregisterThread(int tid, Dynarmic::A32::Jit* base) {
    std::shared_ptr<SigThread> state;
    {
        std::lock_guard<std::mutex> lock(g_threads_mutex);
        const auto it = g_threads.find(tid);
        if (it == g_threads.end() || it->second->base != base) return;
        state = it->second;
        Retire(state);
        g_threads.erase(it);
    }
    if (t_self == state) t_self.reset();
}

int GuestSignalSend(int tid, int signum) {
    auto state = FindThread(tid);
    if (!state) return SendResult(tid, signum, ESRCH);
    if (signum < 0 || signum > kMaxSignals) return SendResult(tid, signum, EINVAL);
    if (signum == 0) return SendResult(tid, signum, 0);

    const GuestSignalAction action = Actions()[signum];
    if (action.handler == 1) return SendResult(tid, signum, 0);  // SIG_IGN
    if (action.handler == 0) return SendResult(tid, signum, ENOTSUP);  // SIG_DFL
    if ((action.flags & kSaSiginfo) != 0 && signum != 24 && signum != 30)
        return SendResult(tid, signum, ENOTSUP);

    std::lock_guard<std::mutex> jit_lock(state->jits_mutex);
    if (state->retired.load(std::memory_order_acquire) || state->jits.empty())
        return SendResult(tid, signum, ESRCH);
    {
        std::lock_guard<std::mutex> queue_lock(state->queue_mutex);
        const bool ordinary = signum < kRealtimeFirst;
        if (!ordinary || std::find(state->pending.begin(), state->pending.end(), signum) == state->pending.end()) {
            state->pending.push_back(signum);
            state->pending_count.store(static_cast<uint32_t>(state->pending.size()),
                                       std::memory_order_release);
        }
    }
    state->cv.notify_all();
    for (auto* jit : state->jits) jit->HaltExecution(Dynarmic::HaltReason::UserDefined2);
    return SendResult(tid, signum, 0);
}

bool GuestSignalPending() {
    auto* const state = t_self.get();
    if (!state || state->retired.load(std::memory_order_acquire) ||
        state->pending_count.load(std::memory_order_acquire) == 0)
        return false;

    const ActionTable actions = Actions();
    std::lock_guard<std::mutex> lock(state->queue_mutex);
    if (state->retired.load(std::memory_order_acquire)) return false;
    for (const int signum : state->pending) {
        if (CanDeliver(actions[signum], signum, t_mask, t_active_signals)) return true;
    }
    return false;
}

void GuestSignalDeliverPending(DynarmicHost& host) {
    const auto state = t_self;
    if (!state) return;

    for (;;) {
        const ActionTable actions = Actions();
        int signum = 0;
        GuestSignalAction action{};
        {
            std::lock_guard<std::mutex> lock(state->queue_mutex);
            if (state->retired.load(std::memory_order_acquire)) return;
            for (auto it = state->pending.begin(); it != state->pending.end();) {
                const int candidate = *it;
                const GuestSignalAction& candidate_action = actions[candidate];
                if (candidate_action.handler <= 1 ||
                    ((candidate_action.flags & kSaSiginfo) != 0 && candidate != 24 && candidate != 30)) {
                    it = state->pending.erase(it);
                    state->pending_count.store(static_cast<uint32_t>(state->pending.size()),
                                               std::memory_order_release);
                    continue;
                }
                if (!CanDeliver(candidate_action, candidate, t_mask, t_active_signals)) {
                    ++it;
                    continue;
                }
                signum = candidate;
                action = candidate_action;
                state->pending.erase(it);
                state->pending_count.store(static_cast<uint32_t>(state->pending.size()),
                                           std::memory_order_release);
                break;
            }
        }
        if (signum == 0) return;

        const uint64_t bit = uint64_t{1} << (signum - 1);
        const uint64_t previous_mask = t_mask;
        const uint64_t previous_active = t_active_signals;
        t_active_signals |= bit;
        t_mask = (previous_mask | static_cast<uint64_t>(action.mask)) & ~kUnmaskable;
        if ((action.flags & kSaNodefer) == 0) t_mask |= bit;
#if defined(BRIDGE_RELEASE_LOGS)
        host.CallGuestNested(action.handler, {static_cast<uint32_t>(signum), 0u, 0u});
#else
        const bool trace = g_delivery_diagnostics.fetch_add(1, std::memory_order_relaxed) < 12;
        if (trace) {
            __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                                "deliver tid=%d sig=%d handler=0x%08X",
                                state->tid, signum, action.handler);
        }
        host.CallGuestNested(action.handler, {static_cast<uint32_t>(signum), 0u, 0u});
        if (trace) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "delivered tid=%d sig=%d", state->tid, signum);
#endif
        t_mask = previous_mask;
        t_active_signals = previous_active;
    }
}

void GuestSignalWait(DynarmicHost& host) {
    const auto state = t_self;
    if (!state) return;
    for (;;) {
        if (state->retired.load(std::memory_order_acquire)) return;
        if (GuestSignalPending()) {
            GuestSignalDeliverPending(host);
            return;
        }
        std::unique_lock<std::mutex> lock(state->queue_mutex);
        state->cv.wait_for(lock, std::chrono::milliseconds(20));
    }
}

uint64_t GuestSignalMask() {
    return t_mask;
}

void GuestSignalSetMask(uint64_t mask) {
    t_mask = mask & ~kUnmaskable;
    if (t_self) t_self->cv.notify_all();
}
