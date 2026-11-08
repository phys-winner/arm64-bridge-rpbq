#pragma once

#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <vector>
#include <string>
#include <optional>
#include <mutex>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <dynarmic/interface/A32/a32.h>
#include <dynarmic/interface/A32/config.h>
#include <dynarmic/interface/exclusive_monitor.h>

class ElfLoader;
class ThunkManager;

struct GuestModule {
    const char* name;
    const uint8_t* data;
    size_t size;
    uint32_t load_base;
};

class DynarmicHost : public Dynarmic::A32::UserCallbacks {
public:
    // The full 32-bit guest address space. Mapping all of it means fastmem can resolve any
    // guest address as base + zext32(vaddr) without ever faulting.
    static constexpr size_t GUEST_ADDRESS_SPACE = 0x100000000ULL; // 4 GiB
    // Fallback if the 4 GiB reservation is refused: no fastmem, bounds-checked callbacks.
    static constexpr size_t GUEST_FALLBACK_SIZE = 0x80000000ULL;  // 2 GiB

    DynarmicHost();
    ~DynarmicHost() override;

    bool Initialize();
    bool LoadGame(const std::string& elf_path);
    bool LoadGameFromMemory(const uint8_t* data, size_t size);

    bool LoadLibraries(const std::string& fmodex_path, const std::string& fmodevent_path, const std::string& game_path);
    bool LoadLibrariesFromMemory(const uint8_t* fmodex_data, size_t fmodex_size,
                                 const uint8_t* fmodevent_data, size_t fmodevent_size,
                                 const uint8_t* game_data, size_t game_size);
    bool LoadGuestModulesFromMemory(const std::vector<GuestModule>& modules);
    uint32_t RunFunction(uint32_t target_pc, std::span<const uint32_t> args);
    uint32_t RunFunction(uint32_t target_pc) {
        return RunFunction(target_pc, std::span<const uint32_t>{});
    }
    uint32_t RunFunction(uint32_t target_pc, std::initializer_list<uint32_t> args) {
        return RunFunction(target_pc, std::span<const uint32_t>{args.begin(), args.size()});
    }
    // Java audio callbacks need synchronous results while the render CPU is busy.
    uint32_t RunAudioFunction(uint32_t target_pc, std::span<const uint32_t> args);
    uint32_t RunAudioFunction(uint32_t target_pc, std::initializer_list<uint32_t> args) {
        return RunAudioFunction(target_pc, std::span<const uint32_t>{args.begin(), args.size()});
    }
    uint32_t CallGuestNested(uint32_t target_pc, std::span<const uint32_t> args);
    uint32_t CallGuestNested(uint32_t target_pc, std::initializer_list<uint32_t> args) {
        return CallGuestNested(target_pc, std::span<const uint32_t>{args.begin(), args.size()});
    }
    // Start a real host thread with its own A32 CPU context for guest pthread_create.
    uint32_t StartGuestThread(uint32_t start_pc, uint32_t arg);
    int JoinGuestThread(uint32_t id);
    static uint32_t CurrentGuestThreadId();

    uint8_t* GetGuestMemoryBase() const { return m_memory; }
    size_t GetGuestMemorySize() const { return m_memory_size; }
    // Unlike a plain arena-bounds check, this also excludes the protected guest
    // null page. Host-side thunks must use it before dereferencing base + gaddr;
    // Dynarmic's scalar callbacks already treat reads there as zero and ignore writes.
    bool IsGuestMemoryRangeAccessible(uint32_t gaddr, size_t size) const {
        if (size == 0) return true;
        if (!m_memory) return false;
        const size_t first_accessible = m_low_page_guard ? m_low_page_guard_size : 1;
        if (static_cast<size_t>(gaddr) < first_accessible || size > m_memory_size) return false;
        return static_cast<uint64_t>(gaddr) <= static_cast<uint64_t>(m_memory_size - size);
    }
    ElfLoader& GetElfLoader() { return *m_elf_loader; }
    ElfLoader* GetFmodExLoader() { return m_fmodex_loader.get(); }
    ElfLoader* GetFmodEventLoader() { return m_fmodevent_loader.get(); }
    ThunkManager& GetThunkManager() { return *m_thunk_manager; }
    std::recursive_mutex& GetMutex() { return m_mutex; }
    uint32_t FindSymbol(const std::string& name);
    uint32_t FindGuestExidx(uint32_t address, uint32_t& count) const;
    // Drop any cached translation of a guest range that was just rewritten.
    void InvalidateGuestCode(uint32_t addr, size_t size);
    uint32_t GuestAlloc(size_t size, size_t alignment = 8);
    void GuestFree(uint32_t addr);
    // Payload size of a live GuestAlloc block, 0 if addr did not come from here.
    uint32_t GuestAllocSize(uint32_t addr);
    // Address space for whole-file mappings (the mmap thunk), deliberately outside the
    // malloc heap: one expansion file can be hundreds of megabytes and would otherwise eat
    // the arena permanently -- a 345 MB .obb mapped twice is how "GuestAlloc out of guest
    // heap" showed up right after loading started. Bump-only, never freed: these mappings
    // live for the process and the address is never reused.
    uint32_t GuestAllocMapping(size_t size, size_t alignment);

    // Runaway-PC containment. The loaded modules and the trampoline page are
    // covered automatically; register anything else that legitimately holds guest
    // code (hand-built veneers, JIT-ed guest stubs) so it is not trapped.
    void AddExecutableRange(uint32_t base, uint32_t size);

    uint32_t GuestStackBase();
    uint32_t GuestStackSize();
    uint32_t GuestErrnoAddress();
    uint32_t CurrentJitSlot();

    // Dynarmic UserCallbacks interface
    std::optional<std::uint32_t> MemoryReadCode(std::uint32_t vaddr) override;

    std::uint8_t MemoryRead8(std::uint32_t vaddr) override;
    std::uint16_t MemoryRead16(std::uint32_t vaddr) override;
    std::uint32_t MemoryRead32(std::uint32_t vaddr) override;
    std::uint64_t MemoryRead64(std::uint32_t vaddr) override;

    void MemoryWrite8(std::uint32_t vaddr, std::uint8_t value) override;
    void MemoryWrite16(std::uint32_t vaddr, std::uint16_t value) override;
    void MemoryWrite32(std::uint32_t vaddr, std::uint32_t value) override;
    void MemoryWrite64(std::uint32_t vaddr, std::uint64_t value) override;

    // Needed now that global_monitor is set: the base-class defaults return false, which makes
    // every guest STREX report failure and spin its retry loop forever.
    bool MemoryWriteExclusive8(std::uint32_t vaddr, std::uint8_t value, std::uint8_t expected) override;
    bool MemoryWriteExclusive16(std::uint32_t vaddr, std::uint16_t value, std::uint16_t expected) override;
    bool MemoryWriteExclusive32(std::uint32_t vaddr, std::uint32_t value, std::uint32_t expected) override;
    bool MemoryWriteExclusive64(std::uint32_t vaddr, std::uint64_t value, std::uint64_t expected) override;

    void InterpreterFallback(std::uint32_t pc, std::size_t num_instructions) override;
    void CallSVC(std::uint32_t swi) override;
    void ExceptionRaised(std::uint32_t pc, Dynarmic::A32::Exception exception) override;

    void AddTicks(std::uint64_t ticks) override;
    std::uint64_t GetTicksRemaining() override;

    bool IsJitRunning() const;

private:
    uint8_t* m_memory = nullptr;
    size_t m_memory_size = 0;
    struct Cpu;
    struct Lifetime;
    std::shared_ptr<Lifetime> m_lifetime;
    std::shared_ptr<Dynarmic::ExclusiveMonitor> m_monitor;
    std::mutex m_cpus_mutex;
    std::unordered_map<int, std::weak_ptr<Cpu>> m_cpus;
    std::mutex m_jit_slots_mutex;
    std::vector<uint32_t> m_free_jit_slots;
    uint32_t m_next_jit_slot = 0;
    static constexpr uint32_t kMaxJits = 256;
    std::unique_ptr<ElfLoader> m_fmodex_loader;
    std::unique_ptr<ElfLoader> m_fmodevent_loader;
    std::unique_ptr<ElfLoader> m_elf_loader;
    std::vector<std::unique_ptr<ElfLoader>> m_extra_loaders;
    std::unique_ptr<ThunkManager> m_thunk_manager;
    std::recursive_mutex m_mutex;
    struct GuestThreadEntry {
        uint32_t id;
        std::thread thread;
    };
    std::mutex m_guest_threads_mutex;
    std::vector<GuestThreadEntry> m_guest_threads;

    std::vector<std::pair<uint32_t, uint32_t>> m_code_ranges;   // [base, end)

    // Diagnostic only: protect the low guest page to identify null-object stores.
    bool m_low_page_guard = false;
    size_t m_low_page_guard_size = 0;

    // Guest heap: bump pointer for fresh memory, free lists for reuse. Requests up
    // to kMaxClassSize round to a power-of-two size class; anything larger (asset
    // payloads, thread stacks) is sized exactly and recycled by exact capacity, so
    // repeated 8 MB stacks do not round up to 16 MB.
    static constexpr uint32_t kHeapBase  = 0x50000000;
    static constexpr uint32_t kHeapLimit = 0x7E000000;   // guest stacks live above this
    // File mappings sit above both guest stacks and below the kuser helper page.
    static constexpr uint32_t kMapBase   = 0xA0000000;
    static constexpr uint32_t kMapLimit  = 0xF0000000;
    static constexpr int kMinClassShift = 5;                        // 32 B smallest block
    static constexpr uint32_t kMaxClassSize = 1u << 20;             // 1 MB largest class
    static constexpr int kNumSizeClasses = 20 - kMinClassShift + 1;
    std::shared_ptr<Cpu> CurrentCpu();
    std::unique_ptr<Dynarmic::A32::Jit> CreateJit(uint32_t& slot);
    void ReleaseJitSlot(uint32_t slot);
    uint32_t RunJit(Dynarmic::A32::Jit& jit, uint32_t target_pc,
                    std::span<const uint32_t> args, uint32_t stack_top);
    void HeapSelfCheck();
    std::mutex m_heap_mutex;
    std::vector<uint32_t> m_free_small[kNumSizeClasses];             // class -> block bases
    std::unordered_map<uint32_t, std::vector<uint32_t>> m_free_huge; // capacity -> block bases
    uint32_t m_heap_current = kHeapBase;                             // guarded by m_heap_mutex
    uint32_t m_map_current  = kMapBase;                              // guarded by m_heap_mutex

    std::uint64_t m_ticks_remaining = 0;
};
