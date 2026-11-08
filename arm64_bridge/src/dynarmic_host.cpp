#include "dynarmic_host.h"
#include "guest_signals.h"
#include <cstdlib>
#include <unistd.h>
#include "elf_loader.h"
#include "thunk_manager.h"
#include <cstring>
#include <android/log.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#define LOG_TAG "Arm32Bridge_Host"
#include "bridge_log.h"

#define EXIT_TRAP_ARM   0x00000004
#define EXIT_TRAP_THUMB 0x00000008

thread_local Dynarmic::A32::Jit* g_active_guest_jit = nullptr;
static thread_local uint32_t g_guest_thread_id = 0;
static std::atomic<uint32_t> g_next_guest_thread_id{0x1000};

static bool LowPageLogAllowed() {
    static std::atomic<unsigned> count{0};
#if defined(BRIDGE_RELEASE_LOGS)
    return count.fetch_add(1, std::memory_order_relaxed) < 8;
#else
    return count.fetch_add(1, std::memory_order_relaxed) < 128;
#endif
}

struct DynarmicHost::Lifetime {
    std::recursive_mutex mutex;
    DynarmicHost* host = nullptr;
};

struct DynarmicHost::Cpu {
    std::weak_ptr<Lifetime> lifetime;
    std::weak_ptr<Cpu> self;
    std::shared_ptr<Dynarmic::ExclusiveMonitor> monitor;
    std::vector<std::pair<uint32_t, std::unique_ptr<Dynarmic::A32::Jit>>> jits;
    std::mutex jits_mutex;
    std::recursive_mutex execution;
    int tid = 0;
    uint32_t allocation = 0;
    uint32_t stack = 0;
    uint32_t stack_size = 1u << 20;
    unsigned depth = 0;
    bool running = false;

    ~Cpu() {
        if (!jits.empty()) GuestSignalUnregisterThread(tid, jits[0].second.get());
        for (auto& entry : jits) {
            entry.second.reset();
            monitor->ClearProcessor(entry.first);
        }
        if (auto owner = lifetime.lock()) {
            std::lock_guard<std::recursive_mutex> lock(owner->mutex);
            if (owner->host) {
                owner->host->GuestFree(allocation);
                for (auto& entry : jits) owner->host->ReleaseJitSlot(entry.first);
                std::lock_guard<std::mutex> table_lock(owner->host->m_cpus_mutex);
                auto found = owner->host->m_cpus.find(tid);
                if (found != owner->host->m_cpus.end() && !found->second.owner_before(self) &&
                    !self.owner_before(found->second)) owner->host->m_cpus.erase(found);
            }
        }
    }
};

DynarmicHost::DynarmicHost() : m_lifetime(std::make_shared<Lifetime>()) {
    m_lifetime->host = this;
}

DynarmicHost::~DynarmicHost() {
    std::vector<GuestThreadEntry> guest_threads;
    {
        std::lock_guard<std::mutex> lock(m_guest_threads_mutex);
        guest_threads.swap(m_guest_threads);
    }
    for (GuestThreadEntry& entry : guest_threads) {
        if (entry.thread.joinable()) entry.thread.join();
    }

    {
        std::lock_guard<std::recursive_mutex> lock(m_lifetime->mutex);
        m_lifetime->host = nullptr;
    }
    std::vector<std::shared_ptr<Cpu>> cpus;
    {
        std::lock_guard<std::mutex> lock(m_cpus_mutex);
        for (auto& [tid, weak] : m_cpus) if (auto cpu = weak.lock()) cpus.push_back(std::move(cpu));
    }
    std::vector<std::unique_lock<std::recursive_mutex>> idle;
    for (auto& cpu : cpus) {
        idle.emplace_back(cpu->execution);
        if (cpu->running) { LOGE("Host destruction inside an active guest frame"); std::abort(); }
        GuestSignalUnregisterThread(cpu->tid, cpu->jits[0].second.get());
    }

    ResetGlSyncRegistry();

    if (m_memory) {
#if defined(_WIN32)
        VirtualFree(m_memory, 0, MEM_RELEASE);
#else
        munmap(m_memory, m_memory_size);
#endif
        m_memory = nullptr;
    }
}

// Reserve guest memory. MAP_NORESERVE keeps the 4 GiB reservation virtual: untouched pages
// cost nothing and only written pages are ever committed.
static uint8_t* MapGuestMemory(size_t size) {
#if defined(_WIN32)
    return reinterpret_cast<uint8_t*>(VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : reinterpret_cast<uint8_t*>(p);
#endif
}

bool DynarmicHost::Initialize() {
    // Mapping the whole 32-bit space lets the JIT emit a single `ldr Wt, [Xfastmem, Waddr, uxtw]`
    // per guest access instead of a call into MemoryRead*/MemoryWrite*.
    m_memory_size = GUEST_ADDRESS_SPACE;
    m_memory = MapGuestMemory(m_memory_size);

    bool fastmem = m_memory != nullptr;
    if (m_memory == nullptr) {
        LOGE("Could not reserve %zu bytes for the guest address space; "
             "falling back to %zu bytes without fastmem.", GUEST_ADDRESS_SPACE, GUEST_FALLBACK_SIZE);
        m_memory_size = GUEST_FALLBACK_SIZE;
        m_memory = MapGuestMemory(m_memory_size);
        if (!m_memory) {
            LOGE("Failed to allocate guest memory (%zu bytes)", m_memory_size);
            m_memory_size = 0;
            return false;
        }
    }

    HeapSelfCheck();

    *reinterpret_cast<uint32_t*>(m_memory) = 0xEFFFFFFF; // Null call: not a normal return.
    // ARM exit trap at 0x00000004: SVC 0 (0xEF000000)
    *reinterpret_cast<uint32_t*>(m_memory + EXIT_TRAP_ARM) = 0xEF000000;

    // Thumb exit trap at 0x00000008: SVC 0 (0xDF00, 0xDF00)
    *reinterpret_cast<uint32_t*>(m_memory + EXIT_TRAP_THUMB) = 0xDF00DF00;

    m_thunk_manager = std::make_unique<ThunkManager>(*this);
    m_thunk_manager->RegisterAllThunks();

    // Diagnostic only: make null-page stores observable through the callbacks.
    m_low_page_guard_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (!m_low_page_guard_size || mprotect(m_memory, m_low_page_guard_size, PROT_NONE) != 0) {
        LOGE("low-page diagnostic mprotect failed: %s", strerror(errno));
        return false;
    }
    m_low_page_guard = true;
    LOGI("low-page diagnostic guard enabled: size=%zu", m_low_page_guard_size);

    // The JIT dereferences global_monitor unconditionally for LDREX/STREX, so it must exist
    // even though we only run one guest core at a time per JIT context.
    m_monitor = std::make_shared<Dynarmic::ExclusiveMonitor>(kMaxJits);
    GuestSignalsReset();
    CurrentCpu();
    LOGI("Dynarmic Host initialized: %zu bytes guest space, fastmem %s",
         m_memory_size, fastmem ? "ON" : "OFF");
    return true;
}

std::unique_ptr<Dynarmic::A32::Jit> DynarmicHost::CreateJit(uint32_t& slot) {
    uint32_t high_water = 0;
    {
        std::lock_guard<std::mutex> lock(m_jit_slots_mutex);
        if (!m_free_jit_slots.empty()) {
            slot = m_free_jit_slots.back();
            m_free_jit_slots.pop_back();
        } else {
            if (m_next_jit_slot == kMaxJits) { LOGE("Concurrent guest CPU limit reached"); std::abort(); }
            slot = m_next_jit_slot++;
        }
        high_water = m_next_jit_slot;
    }
    Dynarmic::A32::UserConfig config;
    config.processor_id = slot;
    config.callbacks = this;
    config.global_monitor = m_monitor.get();
    config.arch_version = Dynarmic::A32::ArchVersion::v7;  // guest libs are ARMv7-A + NEON
    config.enable_cycle_counting = false;                  // no timing model; skips AddTicks emission
    config.always_little_endian = true;                    // guest never uses SETEND; drops CPSR.E handling
    config.define_unpredictable_behaviour = true;          // define rather than halt on odd encodings
    // The ARM64 backend patch mirrors Dynarmic's x64 fast path: reservations
    // remain per-JIT while the real host CAS below provides inter-thread
    // exclusion.  Avoiding one process-wide monitor is essential for Unity's
    // heavily contended job queues.
    config.unsafe_optimizations = true;
    config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_IgnoreGlobalMonitor;
    if (m_memory_size == GUEST_ADDRESS_SPACE) {
        config.fastmem_pointer = reinterpret_cast<uintptr_t>(m_memory);
        config.fastmem_exclusive_access = true;
    }
    LOGI("Created guest JIT slot=%u high_water=%u/%u", slot, high_water, kMaxJits);
    return std::make_unique<Dynarmic::A32::Jit>(config);
}

void DynarmicHost::ReleaseJitSlot(uint32_t slot) {
    std::lock_guard<std::mutex> lock(m_jit_slots_mutex);
    m_free_jit_slots.push_back(slot);
}

std::shared_ptr<DynarmicHost::Cpu> DynarmicHost::CurrentCpu() {
    struct Binding {
        std::shared_ptr<Cpu> cpu;
        ~Binding() {
            if (cpu) GuestSignalUnregisterThread(cpu->tid, cpu->jits[0].second.get());
        }
    };
    static thread_local Binding binding;
    if (binding.cpu && binding.cpu->lifetime.lock() == m_lifetime) return binding.cpu;
    if (binding.cpu) {
        GuestSignalUnregisterThread(binding.cpu->tid, binding.cpu->jits[0].second.get());
        binding.cpu.reset();
    }
    std::lock_guard<std::recursive_mutex> alive(m_lifetime->mutex);
    if (m_lifetime->host != this) return {};
    auto cpu = std::make_shared<Cpu>();
    cpu->lifetime = m_lifetime;
    cpu->self = cpu;
    cpu->monitor = m_monitor;
    cpu->tid = gettid();
    cpu->allocation = GuestAlloc(cpu->stack_size + 16, 16);
    if (!cpu->allocation) { LOGE("No guest stack for host thread %d", cpu->tid); std::abort(); }
    cpu->stack = cpu->allocation + 16; // Per-thread errno lives before the stack, not inside it.
    uint32_t slot;
    auto jit = CreateJit(slot);
    cpu->jits.emplace_back(slot, std::move(jit));
    {
        std::lock_guard<std::mutex> lock(m_cpus_mutex);
        m_cpus[cpu->tid] = cpu;
    }
    GuestSignalRegisterThread(cpu->tid, cpu->jits[0].second.get());
    LOGI("Guest CPU tid=%d slot=%u stack=0x%08X size=%u", cpu->tid, slot, cpu->stack, cpu->stack_size);
    binding.cpu = cpu;
    return cpu;
}

uint32_t DynarmicHost::GuestStackBase() { return CurrentCpu()->stack; }
uint32_t DynarmicHost::GuestStackSize() { return CurrentCpu()->stack_size; }
uint32_t DynarmicHost::GuestErrnoAddress() { return CurrentCpu()->allocation; }
uint32_t DynarmicHost::CurrentJitSlot() { auto cpu = CurrentCpu(); return cpu->jits[cpu->depth].first; }
bool DynarmicHost::IsJitRunning() const { return g_active_guest_jit != nullptr; }

bool DynarmicHost::LoadGame(const std::string& elf_path) {
    LOGI("Loading main game ELF from file: %s", elf_path.c_str());
    m_elf_loader = std::make_unique<ElfLoader>(m_memory, 0x40000000);
    if (!m_elf_loader->Load(elf_path)) {
        LOGE("Failed to load ELF: %s", elf_path.c_str());
        return false;
    }

    auto resolver = [this](const std::string& name) -> uint32_t {
        if (m_fmodevent_loader) {
            uint32_t addr = m_fmodevent_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (m_fmodex_loader) {
            uint32_t addr = m_fmodex_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (m_elf_loader) {
            uint32_t addr = m_elf_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        return m_thunk_manager->GetOrCreateThunkAddress(name);
    };

    if (!m_elf_loader->Relocate(resolver)) {
        LOGE("Failed to relocate ELF.");
        return false;
    }

    LOGI("Running %zu static initializers in .init_array...", m_elf_loader->GetInitArray().size());
    for (uint32_t init_func : m_elf_loader->GetInitArray()) {
        RunFunction(init_func);
    }

    LOGI("Game ELF loaded and initialized successfully!");
    return true;
}

bool DynarmicHost::LoadGameFromMemory(const uint8_t* data, size_t size) {
    LOGI("Loading main game ELF from memory (%zu bytes)...", size);
    m_elf_loader = std::make_unique<ElfLoader>(m_memory, 0x40000000);
    if (!m_elf_loader->LoadFromMemory(data, size)) {
        LOGE("Failed to load ELF from memory buffer");
        return false;
    }

    auto resolver = [this](const std::string& name) -> uint32_t {
        if (m_fmodevent_loader) {
            uint32_t addr = m_fmodevent_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (m_fmodex_loader) {
            uint32_t addr = m_fmodex_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (m_elf_loader) {
            uint32_t addr = m_elf_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        return m_thunk_manager->GetOrCreateThunkAddress(name);
    };

    if (!m_elf_loader->Relocate(resolver)) {
        LOGE("Failed to relocate ELF.");
        return false;
    }

    LOGI("Running %zu static initializers in .init_array...", m_elf_loader->GetInitArray().size());
    for (uint32_t init_func : m_elf_loader->GetInitArray()) {
        RunFunction(init_func);
    }

    LOGI("Game ELF loaded from memory and initialized successfully!");
    return true;
}

bool DynarmicHost::LoadLibraries(const std::string& fmodex_path, const std::string& fmodevent_path, const std::string& game_path) {
    LOGI("Loading libraries: fmodex='%s', fmodevent='%s', game='%s'", fmodex_path.c_str(), fmodevent_path.c_str(), game_path.c_str());

    m_fmodex_loader = std::make_unique<ElfLoader>(m_memory, 0x40400000);
    if (!m_fmodex_loader->Load(fmodex_path)) {
        LOGE("Failed to load libfmodex.so from %s", fmodex_path.c_str());
        return false;
    }

    m_fmodevent_loader = std::make_unique<ElfLoader>(m_memory, 0x40600000);
    if (!m_fmodevent_loader->Load(fmodevent_path)) {
        LOGE("Failed to load libfmodevent.so from %s", fmodevent_path.c_str());
        return false;
    }

    m_elf_loader = std::make_unique<ElfLoader>(m_memory, 0x40000000);
    if (!m_elf_loader->Load(game_path)) {
        LOGE("Failed to load the guest library from %s", game_path.c_str());
        return false;
    }

    auto resolver = [this](const std::string& name) -> uint32_t {
        if (m_fmodevent_loader) {
            uint32_t addr = m_fmodevent_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (m_fmodex_loader) {
            uint32_t addr = m_fmodex_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (m_elf_loader) {
            uint32_t addr = m_elf_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        return m_thunk_manager->GetOrCreateThunkAddress(name);
    };

    if (!m_fmodex_loader->Relocate(resolver)) {
        LOGE("Failed to relocate libfmodex.so");
        return false;
    }

    if (!m_fmodevent_loader->Relocate(resolver)) {
        LOGE("Failed to relocate libfmodevent.so");
        return false;
    }

    if (!m_elf_loader->Relocate(resolver)) {
        LOGE("Failed to relocate the guest library");
        return false;
    }

    LOGI("Running initializers for libfmodex.so (%zu funcs)...", m_fmodex_loader->GetInitArray().size());
    for (uint32_t init_func : m_fmodex_loader->GetInitArray()) {
        RunFunction(init_func);
    }

    LOGI("Running initializers for libfmodevent.so (%zu funcs)...", m_fmodevent_loader->GetInitArray().size());
    for (uint32_t init_func : m_fmodevent_loader->GetInitArray()) {
        RunFunction(init_func);
    }

    LOGI("Running initializers for the guest library (%zu funcs)...", m_elf_loader->GetInitArray().size());
    for (uint32_t init_func : m_elf_loader->GetInitArray()) {
        RunFunction(init_func);
    }

    LOGI("All libraries loaded, relocated, and initialized successfully with sound!");
    return true;
}

bool DynarmicHost::LoadLibrariesFromMemory(const uint8_t* fmodex_data, size_t fmodex_size,
                                           const uint8_t* fmodevent_data, size_t fmodevent_size,
                                           const uint8_t* game_data, size_t game_size) {
    LOGI("Loading libraries from memory: fmodex=%zu bytes, fmodevent=%zu bytes, game=%zu bytes",
         fmodex_size, fmodevent_size, game_size);

    m_fmodex_loader = std::make_unique<ElfLoader>(m_memory, 0x40400000);
    if (!m_fmodex_loader->LoadFromMemory(fmodex_data, fmodex_size)) {
        LOGE("Failed to load libfmodex.so from memory");
        return false;
    }

    m_fmodevent_loader = std::make_unique<ElfLoader>(m_memory, 0x40600000);
    if (!m_fmodevent_loader->LoadFromMemory(fmodevent_data, fmodevent_size)) {
        LOGE("Failed to load libfmodevent.so from memory");
        return false;
    }

    m_elf_loader = std::make_unique<ElfLoader>(m_memory, 0x40000000);
    if (!m_elf_loader->LoadFromMemory(game_data, game_size)) {
        LOGE("Failed to load the guest library from memory");
        return false;
    }

    auto resolver = [this](const std::string& name) -> uint32_t {
        if (m_fmodevent_loader) {
            uint32_t addr = m_fmodevent_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (m_fmodex_loader) {
            uint32_t addr = m_fmodex_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (m_elf_loader) {
            uint32_t addr = m_elf_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        return m_thunk_manager->GetOrCreateThunkAddress(name);
    };

    if (!m_fmodex_loader->Relocate(resolver)) {
        LOGE("Failed to relocate libfmodex.so");
        return false;
    }

    if (!m_fmodevent_loader->Relocate(resolver)) {
        LOGE("Failed to relocate libfmodevent.so");
        return false;
    }

    if (!m_elf_loader->Relocate(resolver)) {
        LOGE("Failed to relocate the guest library");
        return false;
    }

    LOGI("Running initializers for libfmodex.so (%zu funcs)...", m_fmodex_loader->GetInitArray().size());
    for (uint32_t init_func : m_fmodex_loader->GetInitArray()) {
        RunFunction(init_func);
    }

    LOGI("Running initializers for libfmodevent.so (%zu funcs)...", m_fmodevent_loader->GetInitArray().size());
    for (uint32_t init_func : m_fmodevent_loader->GetInitArray()) {
        RunFunction(init_func);
    }

    LOGI("Running initializers for the guest library (%zu funcs)...", m_elf_loader->GetInitArray().size());
    for (uint32_t init_func : m_elf_loader->GetInitArray()) {
        RunFunction(init_func);
    }

    LOGI("All libraries loaded from memory and initialized successfully with sound!");
    return true;
}

bool DynarmicHost::LoadGuestModulesFromMemory(const std::vector<GuestModule>& modules) {
    if (modules.empty()) return false;

    m_elf_loader.reset();
    m_fmodex_loader.reset();
    m_fmodevent_loader.reset();
    m_extra_loaders.clear();

    for (size_t i = 0; i < modules.size(); ++i) {
        const GuestModule& module = modules[i];
        auto loader = std::make_unique<ElfLoader>(m_memory, module.load_base);
        if (!loader->LoadFromMemory(module.data, module.size)) {
            LOGE("Failed to load guest module %s", module.name ? module.name : "<unnamed>");
            return false;
        }
        LOGI("Loaded guest module %s at 0x%08X", module.name ? module.name : "<unnamed>",
             module.load_base);
        if (i == 0) {
            m_elf_loader = std::move(loader);
        } else if (i == 1) {
            m_fmodex_loader = std::move(loader);
        } else if (i == 2) {
            m_fmodevent_loader = std::move(loader);
        } else {
            m_extra_loaders.push_back(std::move(loader));
        }
    }

    auto resolve = [this](const std::string& name) -> uint32_t {
        if (m_elf_loader) {
            uint32_t addr = m_elf_loader->GetSymbolAddress(name);
            if (addr) return addr;
        }
        if (m_fmodex_loader) {
            uint32_t addr = m_fmodex_loader->GetSymbolAddress(name);
            if (addr) return addr;
        }
        if (m_fmodevent_loader) {
            uint32_t addr = m_fmodevent_loader->GetSymbolAddress(name);
            if (addr) return addr;
        }
        for (const auto& loader : m_extra_loaders) {
            uint32_t addr = loader->GetSymbolAddress(name);
            if (addr) return addr;
        }
        return m_thunk_manager->GetOrCreateThunkAddress(name);
    };

    auto relocate = [&](ElfLoader* loader, const char* name) {
        if (!loader || loader->Relocate(resolve)) return true;
        LOGE("Failed to relocate guest module %s", name);
        return false;
    };
    for (size_t i = 0; i < modules.size(); ++i) {
        ElfLoader* loader = i == 0 ? m_elf_loader.get()
                         : i == 1 ? m_fmodex_loader.get()
                         : i == 2 ? m_fmodevent_loader.get()
                                   : m_extra_loaders[i - 3].get();
        if (!relocate(loader, modules[i].name)) return false;
    }

    for (size_t i = 0; i < modules.size(); ++i) {
        ElfLoader* loader = i == 0 ? m_elf_loader.get()
                         : i == 1 ? m_fmodex_loader.get()
                         : i == 2 ? m_fmodevent_loader.get()
                                   : m_extra_loaders[i - 3].get();
        LOGI("Running initializers for %s (%zu funcs)...", modules[i].name,
             loader->GetInitArray().size());
        for (uint32_t init_func : loader->GetInitArray()) RunFunction(init_func);
    }
    LOGI("All guest modules loaded, relocated, and initialized successfully");
    return true;
}

uint32_t DynarmicHost::FindSymbol(const std::string& name) {
    if (m_elf_loader) {
        uint32_t addr = m_elf_loader->GetSymbolAddress(name);
        if (addr) return addr;
    }
    if (m_fmodex_loader) {
        uint32_t addr = m_fmodex_loader->GetSymbolAddress(name);
        if (addr) return addr;
    }
    if (m_fmodevent_loader) {
        uint32_t addr = m_fmodevent_loader->GetSymbolAddress(name);
        if (addr) return addr;
    }
    for (const auto& loader : m_extra_loaders) {
        uint32_t addr = loader->GetSymbolAddress(name);
        if (addr) return addr;
    }
    return 0;
}

uint32_t DynarmicHost::FindGuestExidx(uint32_t address, uint32_t& count) const {
    count = 0;
    address &= ~1u;
    const auto lookup = [&address, &count](const ElfLoader* loader) {
        if (!loader || !loader->ContainsExecutableAddress(address)) return uint32_t{};
        const uint32_t table = loader->GetExidxAddress();
        if (!table) return uint32_t{};
        count = loader->GetExidxCount();
        return table;
    };
    if (uint32_t table = lookup(m_elf_loader.get())) return table;
    if (uint32_t table = lookup(m_fmodex_loader.get())) return table;
    if (uint32_t table = lookup(m_fmodevent_loader.get())) return table;
    for (const auto& loader : m_extra_loaders) {
        if (uint32_t table = lookup(loader.get())) return table;
    }
    return 0;
}

uint32_t DynarmicHost::RunJit(Dynarmic::A32::Jit& jit, uint32_t target_pc,
                               std::span<const uint32_t> args, uint32_t stack_top) {
    for (size_t i = 0; i < args.size() && i < 4; ++i) {
        jit.Regs()[i] = args[i];
    }

    jit.Regs()[13] = stack_top;
    if (args.size() > 4) {
        const size_t extra_args = args.size() - 4;
        const uint32_t sp = (stack_top - static_cast<uint32_t>(extra_args * 4)) & ~7u;
        jit.Regs()[13] = sp;
        for (size_t i = 0; i < extra_args; ++i) {
            *reinterpret_cast<uint32_t*>(m_memory + sp + i * 4) = args[4 + i];
        }
    }

    const bool is_thumb = (target_pc & 1) != 0;
    jit.Regs()[14] = is_thumb ? (EXIT_TRAP_THUMB | 1) : EXIT_TRAP_ARM;
    jit.Regs()[15] = target_pc & ~1u;
    jit.SetCpsr((jit.Cpsr() & ~0x20u) | (is_thumb ? 0x20u : 0u));

    extern thread_local Dynarmic::A32::Jit* g_active_guest_jit;
    Dynarmic::A32::Jit* previous_jit = g_active_guest_jit;
    g_active_guest_jit = &jit;
    jit.ClearHalt();
    if (GuestSignalPending()) GuestSignalDeliverPending(*this);
    Dynarmic::HaltReason hr;
    for (;;) {
        hr = jit.Run();
        const auto resumable = Dynarmic::HaltReason::CacheInvalidation | Dynarmic::HaltReason::UserDefined2;
        if (static_cast<uint32_t>(hr & ~resumable) != 0 || !hr) break;
        jit.ClearHalt(Dynarmic::HaltReason::UserDefined2);
        if (GuestSignalPending()) GuestSignalDeliverPending(*this);
    }
    g_active_guest_jit = previous_jit;

    const uint32_t final_pc = jit.Regs()[15];
    if (final_pc != EXIT_TRAP_ARM && final_pc != (EXIT_TRAP_ARM + 4) &&
        final_pc != (EXIT_TRAP_ARM + 8) && final_pc != (EXIT_TRAP_THUMB & ~1u) &&
        final_pc != ((EXIT_TRAP_THUMB & ~1u) + 2)) {
        LOGE("RunFunction(0x%08X) unexpected halt with reason=0x%X, PC=0x%08X, LR=0x%08X, R0=0x%08X R1=0x%08X R2=0x%08X R3=0x%08X R4=0x%08X R5=0x%08X R6=0x%08X R7=0x%08X R8=0x%08X R9=0x%08X R10=0x%08X R11=0x%08X R12=0x%08X, SP=0x%08X",
             target_pc, static_cast<unsigned>(hr), final_pc, jit.Regs()[14],
             jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3],
             jit.Regs()[4], jit.Regs()[5], jit.Regs()[6], jit.Regs()[7],
             jit.Regs()[8], jit.Regs()[9], jit.Regs()[10], jit.Regs()[11],
             jit.Regs()[12], jit.Regs()[13]);
        std::abort(); // A guest guard/fault is not a successful native return.
    }
    return jit.Regs()[0];
}

uint32_t DynarmicHost::RunFunction(uint32_t target_pc, std::span<const uint32_t> args) {
    const auto lifetime = m_lifetime;
    auto cpu = CurrentCpu();
    if (!cpu) return 0;
    std::lock_guard<std::recursive_mutex> execution(cpu->execution);
    {
        std::lock_guard<std::recursive_mutex> alive(lifetime->mutex);
        if (lifetime->host != this) return 0;
    }
    if (cpu->running) return CallGuestNested(target_pc, args);
    cpu->running = true;
    struct RunningGuard { bool& running; ~RunningGuard() { running = false; } } guard{cpu->running};
    return RunJit(*cpu->jits[0].second, target_pc, args, cpu->stack + cpu->stack_size);
}

uint32_t DynarmicHost::RunAudioFunction(uint32_t target_pc, std::span<const uint32_t> args) {
    return RunFunction(target_pc, args);
}

uint32_t DynarmicHost::CallGuestNested(uint32_t target_pc, std::span<const uint32_t> args) {
    auto cpu = CurrentCpu();
    Dynarmic::A32::Jit* outer = g_active_guest_jit;
    if (!cpu || !outer || cpu->depth >= 8 || outer->Regs()[13] < cpu->stack + 1024 ||
        outer->Regs()[13] > cpu->stack + cpu->stack_size) {
        LOGE("Invalid nested guest frame at 0x%08X", target_pc);
        std::abort();
    }
    const unsigned index = cpu->depth + 1;
    {
        std::lock_guard<std::mutex> lock(cpu->jits_mutex);
        if (cpu->jits.size() <= index) {
            uint32_t slot;
            auto jit = CreateJit(slot);
            cpu->jits.emplace_back(slot, std::move(jit));
            GuestSignalAddThreadJit(cpu->jits.back().second.get());
        }
    }
    auto& nested = *cpu->jits[index].second;
    nested.Regs() = outer->Regs();
    nested.ExtRegs() = outer->ExtRegs();
    nested.SetCpsr(outer->Cpsr());
    nested.SetFpscr(outer->Fpscr());
    // Each nested invocation is a fresh logical frame; reservations belong to
    // the previous use of this JIT slot and must not leak into the new call.
    nested.ClearExclusiveState();
    m_monitor->ClearProcessor(cpu->jits[index].first);
    // Like an ARM signal frame: preserve all interrupted register roots in the
    // real guest stack, above the handler's SP, where Boehm actually scans.
    const uint32_t sp = (outer->Regs()[13] - 512) & ~15u;
    for (size_t i = 0; i < outer->Regs().size(); ++i) MemoryWrite32(sp + i * 4, outer->Regs()[i]);
    for (size_t i = 0; i < outer->ExtRegs().size(); ++i) MemoryWrite32(sp + 64 + i * 4, outer->ExtRegs()[i]);
    MemoryWrite32(sp + 320, outer->Cpsr());
    MemoryWrite32(sp + 324, outer->Fpscr());
    outer->ClearExclusiveState();
    m_monitor->ClearProcessor(cpu->jits[cpu->depth].first);
    ++cpu->depth;
    const uint32_t result = RunJit(nested, target_pc, args, sp);
    --cpu->depth;
    return result;
}

uint32_t DynarmicHost::CurrentGuestThreadId() {
    if (!g_guest_thread_id) g_guest_thread_id = g_next_guest_thread_id.fetch_add(1);
    return g_guest_thread_id;
}

uint32_t DynarmicHost::StartGuestThread(uint32_t start_pc, uint32_t arg) {
    const uint32_t id = g_next_guest_thread_id.fetch_add(1);
    const uint64_t mask = GuestSignalMask();
    try {
        std::thread worker([this, start_pc, arg, id, mask] {
            g_guest_thread_id = id;
            GuestSignalSetMask(mask);
            LOGI("Guest pthread started: id=0x%08X func=0x%08X arg=0x%08X", id, start_pc, arg);
            RunFunction(start_pc, {arg});
            LOGI("Guest pthread finished: id=0x%08X func=0x%08X", id, start_pc);
        });
        std::lock_guard<std::mutex> lock(m_guest_threads_mutex);
        m_guest_threads.push_back({id, std::move(worker)});
        return id;
    } catch (const std::system_error& error) {
        LOGE("pthread_create failed: %s", error.what());
        return 0;
    }
}

int DynarmicHost::JoinGuestThread(uint32_t id) {
    if (!id || id == CurrentGuestThreadId()) return EDEADLK;
    std::thread thread;
    {
        std::lock_guard<std::mutex> lock(m_guest_threads_mutex);
        for (auto it = m_guest_threads.begin(); it != m_guest_threads.end(); ++it) {
            if (it->id == id) {
                thread = std::move(it->thread);
                m_guest_threads.erase(it);
                break;
            }
        }
    }
    if (!thread.joinable()) return ESRCH;
    thread.join();
    return 0;
}

void DynarmicHost::InvalidateGuestCode(uint32_t addr, size_t size) {
    std::vector<std::shared_ptr<Cpu>> cpus;
    {
        std::lock_guard<std::mutex> lock(m_cpus_mutex);
        for (auto& [tid, weak] : m_cpus) if (auto cpu = weak.lock()) cpus.push_back(std::move(cpu));
    }
    for (auto& cpu : cpus) {
        std::lock_guard<std::mutex> lock(cpu->jits_mutex);
        for (auto& entry : cpu->jits) entry.second->InvalidateCacheRange(addr, size);
    }
}

// -------------------------------------------------------------
// Guest heap
//
// Every block carries a 16-byte header immediately before its payload:
//   [-16] magic  [-12] block base  [-8] capacity  [-4] payload size
// The magic is what makes GuestFree safe: a pointer that does not carry it is
// not ours (module data, an interior pointer, a stack top, something the guest
// invented) and is ignored -- exactly as a no-op free ignored everything.
// Freeing clears the magic, so a double free is ignored too.
//
// A plain bump allocator with an empty GuestFree is the usual first version and
// it is a guaranteed leak: the guest does call free(), nobody listens. Measured
// on a Scaleform title that was ~1 MB/s during gameplay.
// -------------------------------------------------------------
namespace {
constexpr uint32_t kBlockMagic = 0x41524D31; // 'ARM1' - live block
constexpr uint32_t kFreedMagic = 0x41524D46; // 'ARMF' - already on a free list
constexpr uint32_t kHeaderSize = 16;
}

uint32_t DynarmicHost::GuestAlloc(size_t size, size_t alignment) {
    if (size == 0) size = 1;
    if (alignment < 8) alignment = 8;
    // Worst case layout: header, then up to a full alignment step of padding.
    uint64_t need = static_cast<uint64_t>(size) + alignment + kHeaderSize;

    uint32_t base = 0, capacity = 0;
    bool recycled = false;
    {
        std::lock_guard<std::mutex> lk(m_heap_mutex);
        if (need <= kMaxClassSize) {
            int cls = 0;
            while ((1u << (cls + kMinClassShift)) < need) ++cls;
            capacity = 1u << (cls + kMinClassShift);
            auto& fl = m_free_small[cls];
            if (!fl.empty()) { base = fl.back(); fl.pop_back(); recycled = true; }
        } else {
            capacity = static_cast<uint32_t>((need + 15) & ~15ull);
            auto it = m_free_huge.find(capacity);
            if (it != m_free_huge.end() && !it->second.empty()) {
                base = it->second.back();
                it->second.pop_back();
                recycled = true;
            }
        }
        if (!recycled) {
            // Stop before the heap walks into the guest stacks instead of silently
            // handing out addresses that overlap them.
            if (static_cast<uint64_t>(m_heap_current) + capacity > kHeapLimit) {
                static std::atomic<bool> exhausted_logged{false};
                if (!exhausted_logged.exchange(true, std::memory_order_relaxed)) {
                    const uint32_t pc = g_active_guest_jit ? g_active_guest_jit->Regs()[15] : 0;
                    const uint32_t lr = g_active_guest_jit ? g_active_guest_jit->Regs()[14] : 0;
                    const uint32_t sp = g_active_guest_jit ? g_active_guest_jit->Regs()[13] : 0;
                    const uint32_t svc = (pc >= 4 && m_thunk_manager) ?
                        (MemoryRead32(pc - 4) & 0x00FFFFFFu) : 0;
                    const std::string thunk = m_thunk_manager ? m_thunk_manager->GetThunkName(svc) : std::string();
                    LOGE("GuestAlloc out of guest heap: at 0x%08X, want %u, limit 0x%08X, PC=0x%08X LR=0x%08X SP=0x%08X thunk=%s",
                         m_heap_current, capacity, kHeapLimit, pc, lr, sp,
                         thunk.empty() ? "<unknown>" : thunk.c_str());
                }
                return 0;
            }
            base = m_heap_current;
            m_heap_current = base + capacity;
        }
    }

    uint32_t payload = static_cast<uint32_t>(
        (static_cast<uint64_t>(base) + kHeaderSize + alignment - 1) & ~(static_cast<uint64_t>(alignment) - 1));
    if (recycled && static_cast<uint64_t>(payload) + size <= m_memory_size) {
        // Fresh arena memory has always been zero, and guest code that leaned on
        // that would break in confusing ways the first time it saw a recycled
        // block. Cheap insurance at the observed churn.
        memset(m_memory + payload, 0, size);
    }
    MemoryWrite32(payload - 16, kBlockMagic);
    MemoryWrite32(payload - 12, base);
    MemoryWrite32(payload - 8, capacity);
    MemoryWrite32(payload - 4, static_cast<uint32_t>(size));
    return payload;
}

void DynarmicHost::GuestFree(uint32_t addr) {
    if (addr < kHeapBase + kHeaderSize) return;         // cannot be one of ours
    if (MemoryRead32(addr - 16) != kBlockMagic) return; // foreign pointer, or already freed
    uint32_t base = MemoryRead32(addr - 12);
    uint32_t capacity = MemoryRead32(addr - 8);
    MemoryWrite32(addr - 16, kFreedMagic);

    std::lock_guard<std::mutex> lk(m_heap_mutex);
    if (capacity <= kMaxClassSize) {
        int cls = 0;
        while ((1u << (cls + kMinClassShift)) < capacity) ++cls;
        m_free_small[cls].push_back(base);
    } else {
        m_free_huge[capacity].push_back(base);
    }
}

uint32_t DynarmicHost::GuestAllocSize(uint32_t addr) {
    if (addr < kHeapBase + kHeaderSize) return 0;
    if (MemoryRead32(addr - 16) != kBlockMagic) return 0;
    return MemoryRead32(addr - 4);
}

// Address space for the mmap thunk. Separate from the malloc heap on purpose -- see the
// declaration. No free list: a file mapping is handed to the engine for the life of the
// process, and the mmap thunk reuses an identical (file, offset, length) mapping instead of
// making a second one, which is what keeps a re-opened expansion file from costing the arena
// twice.
uint32_t DynarmicHost::GuestAllocMapping(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(m_heap_mutex);
    const uint32_t aligned = (m_map_current + static_cast<uint32_t>(alignment - 1)) &
                             ~static_cast<uint32_t>(alignment - 1);
    if (static_cast<uint64_t>(aligned) + size > kMapLimit) {
        LOGE("GuestAllocMapping out of mapping space: at 0x%08X, want %zu", aligned, size);
        return 0;
    }
    m_map_current = aligned + static_cast<uint32_t>(size);
    return aligned;
}

// Runs once at startup. A silently wrong allocator corrupts the guest far away
// from the cause, so pay a few microseconds to fail loudly at the source.
void DynarmicHost::HeapSelfCheck() {
#define HEAP_CHECK(cond) do { if (!(cond)) { LOGE("HeapSelfCheck FAILED: %s", #cond); ok = false; } } while (0)
    bool ok = true;

    uint32_t a = GuestAlloc(100, 8);
    uint32_t b = GuestAlloc(100, 8);
    HEAP_CHECK(a != 0 && b != 0 && a != b);
    HEAP_CHECK((a & 7) == 0 && (b & 7) == 0);
    HEAP_CHECK(a + 100 <= b - kHeaderSize || b + 100 <= a - kHeaderSize); // no overlap
    HEAP_CHECK(GuestAllocSize(a) == 100);

    uint32_t aligned = GuestAlloc(64, 0x1000);
    HEAP_CHECK(aligned != 0 && (aligned & 0xFFF) == 0);

    // Freed blocks come back, and the payload is handed over zeroed.
    MemoryWrite32(a, 0xDEADBEEF);
    GuestFree(a);
    HEAP_CHECK(GuestAllocSize(a) == 0);        // magic cleared
    GuestFree(a);                              // double free: must be a no-op
    uint32_t reused = GuestAlloc(100, 8);
    HEAP_CHECK(reused == a);
    HEAP_CHECK(MemoryRead32(reused) == 0);

    // Pointers we did not hand out must be ignored, not recycled.
    GuestFree(0x40000000);
    GuestFree(kHeapBase + 4);
    GuestFree(b + 8);                          // interior pointer
    HEAP_CHECK(GuestAllocSize(b) == 100);      // b survived all of that

    // Large blocks recycle by exact capacity instead of rounding to a power of two.
    uint32_t big = GuestAlloc(8 * 1024 * 1024, 0x10000);
    HEAP_CHECK(big != 0 && (big & 0xFFFF) == 0);
    GuestFree(big);
    uint32_t big2 = GuestAlloc(8 * 1024 * 1024, 0x10000);
    HEAP_CHECK(big2 == big);
    GuestFree(big2);

    GuestFree(b);
    GuestFree(reused);
    GuestFree(aligned);
    LOGI("HeapSelfCheck %s", ok ? "OK" : "FAILED");
#undef HEAP_CHECK
}

void DynarmicHost::AddExecutableRange(uint32_t base, uint32_t size) {
    if (!size) return;
    m_code_ranges.emplace_back(base, base + size);
    LOGI("Executable range: 0x%08X..0x%08X", base, base + size);
}

std::optional<std::uint32_t> DynarmicHost::MemoryReadCode(std::uint32_t vaddr) {
    if (m_low_page_guard && vaddr < m_low_page_guard_size) {
        if (vaddr == 0) return 0xEFFFFFFFu;
        if (vaddr == EXIT_TRAP_ARM) return 0xEF000000u;
        if (vaddr == EXIT_TRAP_THUMB) return 0xDF00DF00u;
        LOGE("[LOWPAGE] code fetch 0x%08X pc=0x%08X lr=0x%08X", vaddr,
             g_active_guest_jit ? g_active_guest_jit->Regs()[15] : 0,
             g_active_guest_jit ? g_active_guest_jit->Regs()[14] : 0);
        return 0xE7FEDEFFu;
    }

    // Runaway PC containment. The arena is zero-filled and zeros decode as a valid
    // `andeq r0,r0,r0`, so a jump through a garbage pointer makes the JIT translate
    // a fresh block every 0x8000 bytes forever: RSS runs away and the LMK kills the
    // process, a symptom that reads exactly like a memory leak. Feed the JIT a
    // permanently-undefined instruction instead -- ExceptionRaised halts it and
    // prints where the jump came from.
    if (m_elf_loader) {
        auto in_module = [vaddr](const ElfLoader* l) {
            return l && vaddr >= l->GetBaseAddress() &&
                   vaddr < l->GetBaseAddress() + l->GetImageSize();
        };
        // Below kTrampolineEnd: the exit traps and the SVC trampolines.
        constexpr uint32_t kTrampolineEnd = 0x00020000;
        bool in_code = vaddr < kTrampolineEnd ||
                       in_module(m_elf_loader.get()) ||
                       in_module(m_fmodex_loader.get()) ||
                       in_module(m_fmodevent_loader.get());
        for (const auto& loader : m_extra_loaders) {
            if (in_module(loader.get())) {
                in_code = true;
                break;
            }
        }
        for (const auto& r : m_code_ranges) {
            if (in_code) break;
            if (vaddr >= r.first && vaddr < r.second) in_code = true;
        }
        if (!in_code) {
            const auto* jit = g_active_guest_jit;
            LOGE("[RUNAWAY] code fetch 0x%08X pc=0x%08X lr=0x%08X sp=0x%08X r0=0x%08X r1=0x%08X r2=0x%08X r3=0x%08X r4=0x%08X r5=0x%08X r6=0x%08X r7=0x%08X r8=0x%08X r9=0x%08X r10=0x%08X r11=0x%08X r12=0x%08X",
                 vaddr, jit ? jit->Regs()[15] : 0, jit ? jit->Regs()[14] : 0,
                 jit ? jit->Regs()[13] : 0, jit ? jit->Regs()[0] : 0,
                 jit ? jit->Regs()[1] : 0, jit ? jit->Regs()[2] : 0,
                 jit ? jit->Regs()[3] : 0, jit ? jit->Regs()[4] : 0,
                 jit ? jit->Regs()[5] : 0, jit ? jit->Regs()[6] : 0,
                 jit ? jit->Regs()[7] : 0, jit ? jit->Regs()[8] : 0,
                 jit ? jit->Regs()[9] : 0, jit ? jit->Regs()[10] : 0,
                 jit ? jit->Regs()[11] : 0, jit ? jit->Regs()[12] : 0);
            return 0xE7FEDEFFu; // permanently undefined
        }
    }
    if (static_cast<uint64_t>(vaddr) + 4 <= m_memory_size) {
        return *reinterpret_cast<uint32_t*>(m_memory + vaddr);
    }
    return std::nullopt;
}

std::uint8_t DynarmicHost::MemoryRead8(std::uint32_t vaddr) {
    if (m_low_page_guard && vaddr < m_low_page_guard_size) {
        if (LowPageLogAllowed()) LOGE("[LOWPAGE] read8 addr=0x%08X pc=0x%08X lr=0x%08X sp=0x%08X",
            vaddr, g_active_guest_jit ? g_active_guest_jit->Regs()[15] : 0,
            g_active_guest_jit ? g_active_guest_jit->Regs()[14] : 0,
            g_active_guest_jit ? g_active_guest_jit->Regs()[13] : 0);
        return 0;
    }
    if (static_cast<uint64_t>(vaddr) < m_memory_size) return m_memory[vaddr];
    return 0;
}

std::uint16_t DynarmicHost::MemoryRead16(std::uint32_t vaddr) {
    if (m_low_page_guard && vaddr < m_low_page_guard_size) {
        if (LowPageLogAllowed()) LOGE("[LOWPAGE] read16 addr=0x%08X pc=0x%08X lr=0x%08X sp=0x%08X",
            vaddr, g_active_guest_jit ? g_active_guest_jit->Regs()[15] : 0,
            g_active_guest_jit ? g_active_guest_jit->Regs()[14] : 0,
            g_active_guest_jit ? g_active_guest_jit->Regs()[13] : 0);
        return 0;
    }
    if (static_cast<uint64_t>(vaddr) + 2 <= m_memory_size) return *reinterpret_cast<uint16_t*>(m_memory + vaddr);
    return 0;
}

std::uint32_t DynarmicHost::MemoryRead32(std::uint32_t vaddr) {
    if (m_low_page_guard && vaddr < m_low_page_guard_size) {
        if (LowPageLogAllowed()) LOGE("[LOWPAGE] read32 addr=0x%08X pc=0x%08X lr=0x%08X sp=0x%08X",
            vaddr, g_active_guest_jit ? g_active_guest_jit->Regs()[15] : 0,
            g_active_guest_jit ? g_active_guest_jit->Regs()[14] : 0,
            g_active_guest_jit ? g_active_guest_jit->Regs()[13] : 0);
        return 0;
    }
    if (static_cast<uint64_t>(vaddr) + 4 <= m_memory_size) return *reinterpret_cast<uint32_t*>(m_memory + vaddr);
    return 0;
}

std::uint64_t DynarmicHost::MemoryRead64(std::uint32_t vaddr) {
    if (m_low_page_guard && vaddr < m_low_page_guard_size) {
        if (LowPageLogAllowed()) LOGE("[LOWPAGE] read64 addr=0x%08X pc=0x%08X lr=0x%08X sp=0x%08X",
            vaddr, g_active_guest_jit ? g_active_guest_jit->Regs()[15] : 0,
            g_active_guest_jit ? g_active_guest_jit->Regs()[14] : 0,
            g_active_guest_jit ? g_active_guest_jit->Regs()[13] : 0);
        return 0;
    }
    if (static_cast<uint64_t>(vaddr) + 8 <= m_memory_size) return *reinterpret_cast<uint64_t*>(m_memory + vaddr);
    return 0;
}

void DynarmicHost::MemoryWrite8(std::uint32_t vaddr, std::uint8_t value) {
    if (m_low_page_guard && vaddr < m_low_page_guard_size) {
        const uint32_t sp = g_active_guest_jit ? g_active_guest_jit->Regs()[13] : 0;
        LOGE("[LOWPAGE] write8 addr=0x%08X value=0x%02X pc=0x%08X lr=0x%08X sp=0x%08X saved_lr4=0x%08X saved_lr9=0x%08X",
             vaddr, value, g_active_guest_jit ? g_active_guest_jit->Regs()[15] : 0,
             g_active_guest_jit ? g_active_guest_jit->Regs()[14] : 0, sp,
             sp && static_cast<uint64_t>(sp) + 16 <= m_memory_size ? MemoryRead32(sp + 12) : 0,
             sp && static_cast<uint64_t>(sp) + 36 <= m_memory_size ? MemoryRead32(sp + 32) : 0);
        return;
    }
    if (static_cast<uint64_t>(vaddr) < m_memory_size) m_memory[vaddr] = value;
}

void DynarmicHost::MemoryWrite16(std::uint32_t vaddr, std::uint16_t value) {
    if (m_low_page_guard && vaddr < m_low_page_guard_size) {
        const uint32_t sp = g_active_guest_jit ? g_active_guest_jit->Regs()[13] : 0;
        LOGE("[LOWPAGE] write16 addr=0x%08X value=0x%04X pc=0x%08X lr=0x%08X sp=0x%08X saved_lr4=0x%08X saved_lr9=0x%08X",
             vaddr, value, g_active_guest_jit ? g_active_guest_jit->Regs()[15] : 0,
             g_active_guest_jit ? g_active_guest_jit->Regs()[14] : 0, sp,
             sp && static_cast<uint64_t>(sp) + 16 <= m_memory_size ? MemoryRead32(sp + 12) : 0,
             sp && static_cast<uint64_t>(sp) + 36 <= m_memory_size ? MemoryRead32(sp + 32) : 0);
        return;
    }
    if (static_cast<uint64_t>(vaddr) + 2 <= m_memory_size) *reinterpret_cast<uint16_t*>(m_memory + vaddr) = value;
}

void DynarmicHost::MemoryWrite32(std::uint32_t vaddr, std::uint32_t value) {
    if (m_low_page_guard && vaddr < m_low_page_guard_size) {
        const uint32_t sp = g_active_guest_jit ? g_active_guest_jit->Regs()[13] : 0;
        const auto* jit = g_active_guest_jit;
        LOGE("[LOWPAGE] write32 addr=0x%08X value=0x%08X pc=0x%08X lr=0x%08X sp=0x%08X r0=0x%08X r1=0x%08X r2=0x%08X r3=0x%08X r4=0x%08X r5=0x%08X r6=0x%08X r7=0x%08X r8=0x%08X r9=0x%08X r10=0x%08X r11=0x%08X r12=0x%08X saved_lr4=0x%08X saved_lr9=0x%08X",
             vaddr, value, jit ? jit->Regs()[15] : 0, jit ? jit->Regs()[14] : 0, sp,
             jit ? jit->Regs()[0] : 0, jit ? jit->Regs()[1] : 0, jit ? jit->Regs()[2] : 0,
             jit ? jit->Regs()[3] : 0, jit ? jit->Regs()[4] : 0, jit ? jit->Regs()[5] : 0,
             jit ? jit->Regs()[6] : 0, jit ? jit->Regs()[7] : 0, jit ? jit->Regs()[8] : 0,
             jit ? jit->Regs()[9] : 0, jit ? jit->Regs()[10] : 0, jit ? jit->Regs()[11] : 0,
             jit ? jit->Regs()[12] : 0,
             sp && static_cast<uint64_t>(sp) + 16 <= m_memory_size ? MemoryRead32(sp + 12) : 0,
             sp && static_cast<uint64_t>(sp) + 36 <= m_memory_size ? MemoryRead32(sp + 32) : 0);
        return;
    }
    if (static_cast<uint64_t>(vaddr) + 4 <= m_memory_size) *reinterpret_cast<uint32_t*>(m_memory + vaddr) = value;
}

void DynarmicHost::MemoryWrite64(std::uint32_t vaddr, std::uint64_t value) {
    if (m_low_page_guard && vaddr < m_low_page_guard_size) {
        const uint32_t sp = g_active_guest_jit ? g_active_guest_jit->Regs()[13] : 0;
        LOGE("[LOWPAGE] write64 addr=0x%08X value=0x%016llX pc=0x%08X lr=0x%08X sp=0x%08X saved_lr4=0x%08X saved_lr9=0x%08X",
             vaddr, static_cast<unsigned long long>(value),
             g_active_guest_jit ? g_active_guest_jit->Regs()[15] : 0,
             g_active_guest_jit ? g_active_guest_jit->Regs()[14] : 0, sp,
             sp && static_cast<uint64_t>(sp) + 16 <= m_memory_size ? MemoryRead32(sp + 12) : 0,
             sp && static_cast<uint64_t>(sp) + 36 <= m_memory_size ? MemoryRead32(sp + 32) : 0);
        return;
    }
    if (static_cast<uint64_t>(vaddr) + 8 <= m_memory_size) *reinterpret_cast<uint64_t*>(m_memory + vaddr) = value;
}

void DynarmicHost::InterpreterFallback(std::uint32_t pc, std::size_t num_instructions) {
    LOGE("Interpreter Fallback at 0x%08X", pc);
    if (g_active_guest_jit) g_active_guest_jit->HaltExecution(Dynarmic::HaltReason::UserDefined1);
}

void DynarmicHost::CallSVC(std::uint32_t swi) {
    Dynarmic::A32::Jit* jit = g_active_guest_jit;
    if (!jit || swi == 0x00FFFFFF) {
        LOGE("Null/invalid guest SVC target swi=0x%08X pc=0x%08X lr=0x%08X r0=0x%08X r1=0x%08X",
             swi, jit ? jit->Regs()[15] : 0, jit ? jit->Regs()[14] : 0,
             jit ? jit->Regs()[0] : 0, jit ? jit->Regs()[1] : 0);
        std::abort();
    }
    if (swi == 0 || jit->Regs()[15] <= 0x40) {
        jit->HaltExecution(Dynarmic::HaltReason::UserDefined1);
        return;
    }
    m_thunk_manager->DispatchThunk(swi, *jit);
    if (GuestSignalPending()) jit->HaltExecution(Dynarmic::HaltReason::UserDefined2);
}

void DynarmicHost::ExceptionRaised(std::uint32_t pc, Dynarmic::A32::Exception exception) {
    LOGE("Exception at PC: 0x%08X (type %d)", pc, static_cast<int>(exception));
    if (g_active_guest_jit) g_active_guest_jit->HaltExecution(Dynarmic::HaltReason::UserDefined1);
}

void DynarmicHost::AddTicks(std::uint64_t ticks) {
    if (m_ticks_remaining > ticks) m_ticks_remaining -= ticks;
    else m_ticks_remaining = 0;
}

std::uint64_t DynarmicHost::GetTicksRemaining() {
    return m_ticks_remaining;
}

// STREX: atomically compare-and-set against guest memory.  The ExclusiveMonitor serializes
// exclusive operations with each other, but ordinary guest STR instructions use fastmem and
// therefore do not take that monitor lock.  A split load/store here can otherwise overwrite a
// concurrent unlock (for example Boehm GC's plain STR 0), leaving a lock permanently set with
// no owner.  Use a real host CAS so exclusive and ordinary guest stores are coherent.
template<typename T>
static bool GuestCompareAndSwap(uint8_t* mem, size_t mem_size, uint32_t vaddr, T value, T expected) {
    if (static_cast<uint64_t>(vaddr) + sizeof(T) > mem_size) return false;
    T* p = reinterpret_cast<T*>(mem + vaddr);
    return __atomic_compare_exchange_n(p, &expected, value, false,
                                       __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

bool DynarmicHost::MemoryWriteExclusive8(std::uint32_t vaddr, std::uint8_t value, std::uint8_t expected) {
    return GuestCompareAndSwap(m_memory, m_memory_size, vaddr, value, expected);
}

bool DynarmicHost::MemoryWriteExclusive16(std::uint32_t vaddr, std::uint16_t value, std::uint16_t expected) {
    return GuestCompareAndSwap(m_memory, m_memory_size, vaddr, value, expected);
}

bool DynarmicHost::MemoryWriteExclusive32(std::uint32_t vaddr, std::uint32_t value, std::uint32_t expected) {
    return GuestCompareAndSwap(m_memory, m_memory_size, vaddr, value, expected);
}

bool DynarmicHost::MemoryWriteExclusive64(std::uint32_t vaddr, std::uint64_t value, std::uint64_t expected) {
    return GuestCompareAndSwap(m_memory, m_memory_size, vaddr, value, expected);
}
