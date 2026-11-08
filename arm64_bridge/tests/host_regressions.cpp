#include "published_entries.h"
#include "audio_callback_delivery.h"
#include "mapped_buffer_copy.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define LOGI(...) ((void)0)
// PRODUCTION_FRAGMENTS

static unsigned checks = 0;
#define CHECK(expr) do { ++checks; if (!(expr)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); std::abort(); } } while (0)

static void RegistryConcurrency() {
    struct Entry { std::string name; std::function<int()> handler; };
    PublishedEntries<Entry, 8192> table;
    CHECK(!table.Read(0) && !table.Read(8193) && !table.Read(1));
    table.Publish(1, {"0", [] { return 0; }});
    const auto* old = table.Read(1);
    std::atomic<bool> done{false}, bad{false};
    std::atomic<uint64_t> reads{0};
    std::array<std::thread, 4> readers;
    std::promise<void> ready;
    auto gate = ready.get_future().share();
    for (auto& thread : readers) thread = std::thread([&] {
        gate.wait();
        do {
            const auto* entry = table.Read(1);
            if (!entry || std::stoi(entry->name) != entry->handler()) bad = true;
            reads.fetch_add(1, std::memory_order_relaxed);
        } while (!done.load(std::memory_order_acquire));
    });
    ready.set_value();
    for (int i = 1; i <= 20000; ++i) table.Publish(1, {std::to_string(i), [i] { return i; }});
    done.store(true, std::memory_order_release);
    for (auto& thread : readers) thread.join();
    CHECK(!bad && reads >= 4);
    CHECK(old->name == "0" && old->handler() == 0);
    for (unsigned i = 2; i <= 8192; ++i) table.Publish(i, {"last", [i] { return int(i); }});
    CHECK(table.Read(8192)->handler() == 8192);
    for (size_t bad_id : {size_t(0), size_t(8193), std::numeric_limits<size_t>::max()}) {
        bool threw = false;
        try { table.Publish(bad_id, {}); } catch (const std::out_of_range&) { threw = true; }
        CHECK(threw);
    }
    std::printf("PASS immutable dispatch: 20000 replacements, 8192 IDs, %llu concurrent reads\n",
                static_cast<unsigned long long>(reads.load()));
}

static void AudioLifetimeAndDelivery() {
    using namespace audio_test;
    std::weak_ptr<AudioPlayerState> previous;
    for (uint32_t i = 1; i <= 160; ++i) {
        CHECK(previous.expired());
        auto player = std::make_shared<AudioPlayerState>();
        previous = player;
        const uint32_t handle = 0x10000 + i * 32;
        player->guest_object = handle;
        player->guest_play = handle + 4;
        player->guest_queue = handle + 8;
        player->guest_volume = handle + 12;
        player->guest_configuration = handle + 16;
        {
            std::lock_guard lock(g_playersMutex);
            RegisterPlayerHandles(player);
            g_players.push_back(player);
        }
        std::weak_ptr<AudioPlayerState> weak = player;
        auto snapshot = SnapshotPlayers();
        CHECK(snapshot.size() == 1 && snapshot.front() == player);
        CHECK(FindPlayer(handle + 8) == player);
        auto& delivery = player->delivery;
        std::lock_guard lock(delivery.mutex);
        delivery.Register(0x777, 0x888);
        auto ticket = delivery.Capture();
        CHECK(ticket.callback == 0x777 && ticket.context == 0x888 && delivery.Current(ticket));
        HostBqPlayerCallback(nullptr, player.get());
        HostBqPlayerCallback(nullptr, player.get());
        CHECK(g_bqCallbackPending == 2 && player->pending_callbacks == 2);
        DiscardPendingCallbacks(*player);
        CHECK(!delivery.Current(ticket) && g_bqCallbackPending == 0 && player->pending_callbacks == 0);
        ticket = delivery.Capture();
        delivery.Register(0x999, 0xAAA);
        CHECK(!delivery.Current(ticket));
        ticket = delivery.Capture();
        CHECK(ticket.callback == 0x999 && ticket.context == 0xAAA);
        // A reentrant callback clears itself: later calls in its batch stop.
        unsigned calls = 0;
        for (int j = 0; j < 4 && delivery.Current(ticket); ++j) {
            std::lock_guard nested(delivery.mutex);
            ++calls;
            DiscardPendingCallbacks(*player);
        }
        CHECK(calls == 1);
        player->destroyed = true;
        HostBqPlayerCallback(nullptr, player.get());
        CHECK(g_bqCallbackPending == 0);
        ForgetPlayer(player);
        CHECK(SnapshotPlayers().empty());
        for (uint32_t h = handle; h <= handle + 16; h += 4) CHECK(!FindPlayer(h));
        player.reset();
        CHECK(!weak.expired());
        // Keep mutex owner alive until lock is destroyed at the end of scope.
        CHECK(snapshot.front()->destroyed);
    }
    CHECK(previous.expired());
    CHECK(g_players.empty() && g_playerByGuestHandle.empty());
    std::puts("PASS OpenSL production fragments: 160 player lifetimes, callback cancellation, handle retirement");
}

static void GlShareGroups() {
    using namespace gl_test;
    DynarmicHost host;
    TrackGlContext(1, EGL_NO_CONTEXT);
    TrackGlContext(2, 1);
    TrackGlContext(3, EGL_NO_CONTEXT);
    std::lock_guard lock_for_groups(s_guest_mapping_mutex);
    const auto shared = ContextGroupLocked(1);
    const auto separate = ContextGroupLocked(3);
    CHECK(ContextGroupLocked(2) == shared && shared != separate);
    // End this lock before calling the externally locking deletion helper.
}

static void GlMappingRetirement() {
    using namespace gl_test;
    DynarmicHost host;
    const auto shared = s_gl_context_groups.at(1), separate = s_gl_context_groups.at(3);
    std::array<uint8_t, 2> native{};
    s_guest_mapped_buffers[&native[0]] = {&native[0], 100, 1, 0, shared, 7};
    s_guest_mapped_buffers[&native[1]] = {&native[1], 200, 1, 0, separate, 7};
    const GLuint name = 7;
    current_context = 2; // Delete through the other member of the share group.
    ForgetGuestBufferMappings(host, 1, &name);
    CHECK(host.freed == std::vector<uint32_t>{100});
    CHECK(s_guest_mapped_buffers.size() == 1 && s_guest_mapped_buffers.contains(&native[1]));
    current_context = 3;
    ForgetGuestBufferMappings(host, 1, &name);
    CHECK(host.freed.size() == 2 && host.freed[1] == 200 && s_guest_mapped_buffers.empty());
    TrackGlContext(3, EGL_NO_CONTEXT); // Native EGL handle recycled for a new context.
    CHECK(s_gl_context_groups.at(3) != separate);
    std::puts("PASS GL production registry: shared/unshared name collisions, shared deletion, recycled EGL handles");
}

static void MappedCopies() {
    using namespace bridge_gl;
    std::array<uint8_t, 32> guest, native;
    guest.fill(7); native.fill(1);
    CopyOnFlush(native.data(), guest.data(), 32, kMapWrite | kMapFlushExplicit, 4, 8);
    for (size_t i = 0; i < native.size(); ++i) CHECK(native[i] == (i >= 4 && i < 12 ? 7 : 1));
    guest.fill(9);
    CopyOnUnmap(native.data(), guest.data(), 32, kMapWrite | kMapFlushExplicit);
    for (size_t i = 0; i < native.size(); ++i) CHECK(native[i] == (i >= 4 && i < 12 ? 7 : 1));
    const auto saved = native;
    for (auto [offset, length] : {std::pair<int64_t,int64_t>{-1,1}, {0,-1}, {0,0},
                                {31,2}, {33,1}, {INT64_MAX,INT64_MAX}, {1,INT64_MAX}}) {
        CopyOnFlush(native.data(), guest.data(), 32, kMapWrite | kMapFlushExplicit, offset, length);
        CHECK(native == saved);
    }
    for (uint32_t flags : {0u, 1u, kMapWrite, kMapFlushExplicit}) {
        CopyOnFlush(native.data(), guest.data(), 32, flags, 0, 32);
        CHECK(native == saved);
    }
    CopyOnUnmap(native.data(), guest.data(), 32, 1);
    CHECK(native == saved);
    CopyOnUnmap(native.data(), guest.data(), 32, kMapWrite);
    CHECK(native == guest);
    std::puts("PASS GL production copy helpers: explicit flush, read-only, full write, overflow and negative ranges");
}

static void PerformanceSessionRace() {
    using namespace perf_test;
    auto& state = GetState();
    static std::promise<void> report_entered, release_report;
    static auto release = release_report.get_future().share();
    static std::atomic<int> closed{0}, attempts{0};
    APerformanceHintSession session;
    state.load_attempted = true;
    state.create_session = [](APerformanceHintManager*, const int32_t*, size_t, int64_t) -> APerformanceHintSession* { return nullptr; };
    state.report_duration = [](APerformanceHintSession*, int64_t) { report_entered.set_value(); release.wait(); return 0; };
    state.close_session = [](APerformanceHintSession*) { ++closed; };
    CHECK(!LoadApi(state)); // Missing getManager must also fail after first load.
    state.get_manager = []() -> APerformanceHintManager* { ++attempts; return nullptr; };
    CHECK(LoadApi(state));
    state.auxiliary_tids[0] = 33; // Dead TID; fixture syscall returns ESRCH.
    state.auxiliary_tids[1] = 44;
    state.auxiliary_count = 2;
    for (int i = 0; i < 1000; ++i) BeginRenderFrame();
    CHECK(attempts == 1 && state.auxiliary_count == 1 && state.auxiliary_tids[0] == 44);
    state.next_attempt = {};
    BeginRenderFrame();
    CHECK(attempts == 2);
    state.session.store(&session);
    auto entered = report_entered.get_future();
    std::thread reporter([] { EndRenderFrame(100); });
    entered.wait();
    const bool unlocked_during_report = state.mutex.try_lock();
    if (unlocked_during_report) state.mutex.unlock();
    CHECK(!unlocked_during_report);
    std::thread shutdown([] { Shutdown(); });
    CHECK(closed == 0);
    release_report.set_value();
    reporter.join(); shutdown.join();
    CHECK(closed == 1 && state.session == nullptr && state.auxiliary_count == 0);
    EndRenderFrame(100); Shutdown();
    CHECK(closed == 1);
    std::puts("PASS ADPF production functions: report/close exclusion, partial API, retry backoff, stale TID pruning");
}

int main() {
    RegistryConcurrency();
    AudioLifetimeAndDelivery();
    MappedCopies();
    GlShareGroups();
    GlMappingRetirement();
    PerformanceSessionRace();
    std::printf("PASS %u checks; Android HAL/GLES/JIT/device execution NOT tested\n", checks);
}
