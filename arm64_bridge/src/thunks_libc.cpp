#include "thunk_manager.h"
#include "dynarmic_host.h"
#include "guest_signals.h"
#include "thunk_types.h"
#include "thunks_opensles.h"
#include "elf_loader.h"
#include "jni_fake_env.h"
#include "performance_hints.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/prctl.h>
#include <sys/utsname.h>
#include <sys/auxv.h>
#include <sys/system_properties.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cwctype>
#include <locale.h>
#include <signal.h>
#include <setjmp.h>
#include <semaphore.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <dirent.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <thread>
#include <cerrno>
#include <zlib.h>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cinttypes>

#define LOG_TAG "Arm32Bridge_Libc"
#include "bridge_log.h"

static bool GuestBulkLogAllowed() {
    static std::atomic<unsigned> count{0};
    return count.fetch_add(1, std::memory_order_relaxed) < 16;
}

static void LogRejectedGuestBulk(Dynarmic::A32::Jit& jit, const char* operation,
                                 uint32_t destination, uint32_t source, size_t size,
                                 bool destination_valid, bool source_valid) {
    if (!GuestBulkLogAllowed()) return;
    LOGE("[GUESTMEM] %s rejected dst=0x%08X(%s) src=0x%08X(%s) size=%zu "
         "pc=0x%08X lr=0x%08X",
         operation, destination, destination_valid ? "ok" : "invalid",
         source, source_valid ? "ok" : "invalid", size,
         jit.Regs()[15], jit.Regs()[14]);
}

static void GuestBulkSet(Dynarmic::A32::Jit& jit, DynarmicHost& host,
                         const char* operation, uint32_t destination,
                         int value, size_t size) {
    if (size == 0) return;
    const bool destination_valid = host.IsGuestMemoryRangeAccessible(destination, size);
    if (!destination_valid) {
        LogRejectedGuestBulk(jit, operation, destination, 0, size, false, true);
        return;
    }
    memset(host.GetGuestMemoryBase() + destination, value, size);
}

static void GuestBulkCopy(Dynarmic::A32::Jit& jit, DynarmicHost& host,
                          const char* operation, uint32_t destination,
                          uint32_t source, size_t size, bool may_overlap) {
    if (size == 0) return;
    const bool destination_valid = host.IsGuestMemoryRangeAccessible(destination, size);
    const bool source_valid = host.IsGuestMemoryRangeAccessible(source, size);
    if (!destination_valid || !source_valid) {
        LogRejectedGuestBulk(jit, operation, destination, source, size,
                             destination_valid, source_valid);
        // Scalar reads from the protected/null or out-of-arena guest range yield
        // zero. Mirror that fail-soft contract for a valid bulk destination so
        // an invalid audio source becomes silence instead of stale/noisy samples.
        if (destination_valid) memset(host.GetGuestMemoryBase() + destination, 0, size);
        return;
    }
    void* host_destination = host.GetGuestMemoryBase() + destination;
    const void* host_source = host.GetGuestMemoryBase() + source;
    if (may_overlap) memmove(host_destination, host_source, size);
    else memcpy(host_destination, host_source, size);
}

static int GuestBulkCompare(Dynarmic::A32::Jit& jit, DynarmicHost& host,
                            uint32_t lhs, uint32_t rhs, size_t size) {
    if (size == 0) return 0;
    const bool lhs_valid = host.IsGuestMemoryRangeAccessible(lhs, size);
    const bool rhs_valid = host.IsGuestMemoryRangeAccessible(rhs, size);
    if (!lhs_valid || !rhs_valid) {
        LogRejectedGuestBulk(jit, "memcmp", lhs, rhs, size, lhs_valid, rhs_valid);
        return 0;
    }
    return memcmp(host.GetGuestMemoryBase() + lhs, host.GetGuestMemoryBase() + rhs, size);
}

// Implemented in jni_bridge.cpp once the guest modules are loaded; resolves a
// guest code address to its owning library name for dladdr-style queries.
const char* GuestModuleNameForAddress(uint32_t address, uint32_t* base_out);

extern "C" FILE *funopen(const void *cookie,
                         int (*readfn)(void *, char *, int),
                         int (*writefn)(void *, const char *, int),
                         fpos_t (*seekfn)(void *, fpos_t, int),
                         int (*closefn)(void *));

struct ApkEntry {
    char name[256];
    off_t data_offset;
    off_t file_size;
    int apk_fd;
    int method;
};

static ApkEntry g_apk_entries[1024];
static int g_num_apk_entries = 0;
static std::vector<std::string> g_apk_indexed_paths;
static std::vector<int> g_apk_indexed_fds;

static std::vector<GuestThreadWorker> g_thread_workers;

const std::vector<GuestThreadWorker>& GetGuestThreadWorkers() {
    return g_thread_workers;
}

void AddGuestThreadWorker(uint32_t start_func, uint32_t arg) {
    g_thread_workers.push_back({start_func, arg});
}

struct ApkCookie {
    int fd;
    off_t base_offset;
    off_t file_size;
    off_t current_offset;
};

static int ApkRead(void* cookie, char* buf, int size) {
    ApkCookie* c = static_cast<ApkCookie*>(cookie);
    if (size <= 0 || !c || c->fd < 0) return 0;

    if (c->current_offset + size > c->file_size) {
        size = static_cast<int>(c->file_size - c->current_offset);
    }
    if (size <= 0) return 0;

    ssize_t n = pread(c->fd, buf, size, c->base_offset + c->current_offset);
    if (n > 0) {
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 96) {
            LOGI("ApkRead fd=%d size=%d -> %zd at %lld", c->fd, size, n,
                 static_cast<long long>(c->base_offset + c->current_offset));
        }
        c->current_offset += n;
        return static_cast<int>(n);
    }
    LOGI("ApkRead: size=%d, offset=%lld, n=%zd", size, static_cast<long long>(c->current_offset), n);
    return 0;
}

static fpos_t ApkSeek(void* cookie, fpos_t offset, int whence) {
    ApkCookie* c = static_cast<ApkCookie*>(cookie);
    if (!c) return -1;
    off_t target = 0;

    if (whence == SEEK_SET) target = offset;
    else if (whence == SEEK_CUR) target = c->current_offset + offset;
    else if (whence == SEEK_END) target = c->file_size + offset;
    else return -1;

    if (target < 0) return -1;
    if (target > c->file_size) target = c->file_size;
    static std::atomic<unsigned> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 96) {
        LOGI("ApkSeek fd=%d off=%lld whence=%d -> %lld", c->fd,
             static_cast<long long>(offset), whence, static_cast<long long>(target));
    }
    c->current_offset = target;
    return target;
}

static int ApkClose(void* cookie) {
    ApkCookie* c = static_cast<ApkCookie*>(cookie);
    if (c) {
        if (c->fd >= 0) close(c->fd);
        free(c);
    }
    return 0;
}

// Expansion payloads (main/patch OBB) ship inside split APKs on a split-only
// install. CRIWare and Unity drive those archives with raw-fd semantics (mmap,
// FIONREAD, shared positions) that a funopen stream cannot fully provide, so on
// first launch they are extracted to the app's canonical OBB directory as real
// files. Subsequent launches find them and skip the copy.
static std::vector<uint8_t>& ExtractionBuffer() {
    static std::vector<uint8_t> buffer(4 << 20);
    return buffer;
}

extern "C" void ExtractExpansionFilesFromApks() {
    static const char kObbDir[] = "/storage/emulated/0/Android/obb/com.camouflaj.republique";
    ::mkdir(kObbDir, 0775);
    for (int i = 0; i < g_num_apk_entries; ++i) {
        const char* name = g_apk_entries[i].name;
        const char* base = strrchr(name, '/');
        base = base ? base + 1 : name;
        const bool is_expansion = (strncmp(base, "main.", 5) == 0 || strncmp(base, "patch.", 6) == 0) &&
                                  strstr(base, ".obb") != nullptr;
        if (!is_expansion || g_apk_entries[i].apk_fd < 0 || g_apk_entries[i].method != 0) continue;

        const std::string dest = std::string(kObbDir) + "/" + base;
        struct stat st{};
        if (stat(dest.c_str(), &st) == 0 &&
            static_cast<uint64_t>(st.st_size) == g_apk_entries[i].file_size) {
            LOGI("Expansion '%s' already extracted (%lld bytes)", dest.c_str(),
                 static_cast<long long>(st.st_size));
            continue;
        }
        LOGI("Extracting '%s' -> '%s' (%lld bytes, one-time)", name, dest.c_str(),
             static_cast<long long>(g_apk_entries[i].file_size));
        const int out = open(dest.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (out < 0) {
            LOGE("Extraction open '%s' failed errno=%d", dest.c_str(), errno);
            continue;
        }
        constexpr off_t kChunk = 4 << 20;
        off64_t done = 0;
        bool failed = false;
        while (done < g_apk_entries[i].file_size) {
            const size_t want = static_cast<size_t>(
                (g_apk_entries[i].file_size - done) < kChunk ? (g_apk_entries[i].file_size - done) : kChunk);
            ssize_t n = pread(g_apk_entries[i].apk_fd, ExtractionBuffer().data(), want,
                              g_apk_entries[i].data_offset + done);
            if (n <= 0 || ::write(out, ExtractionBuffer().data(), static_cast<size_t>(n)) != n) {
                LOGE("Extraction '%s' failed at %lld errno=%d", dest.c_str(),
                     static_cast<long long>(done), errno);
                failed = true;
                break;
            }
            done += n;
        }
        close(out);
        if (failed) {
            ::remove(dest.c_str());
        } else {
            LOGI("Extracted '%s' (%lld bytes)", dest.c_str(), static_cast<long long>(done));
        }
    }
}

extern "C" int IndexApk(const char* apk_path) {
    if (!apk_path) return -1;
    if (std::find(g_apk_indexed_paths.begin(), g_apk_indexed_paths.end(), apk_path) !=
        g_apk_indexed_paths.end()) {
        return 0;
    }

    const int fd = open(apk_path, O_RDONLY);
    if (fd < 0) {
        LOGE("IndexApk: Failed to open APK: %s (errno %d)", apk_path, errno);
        return -1;
    }
    off_t file_len = lseek(fd, 0, SEEK_END);
    if (file_len < 22) { close(fd); return -1; }

    off_t search_len = file_len < 65536 + 22 ? file_len : 65536 + 22;
    off_t search_start = file_len - search_len;
    std::vector<uint8_t> buf(static_cast<size_t>(search_len));
    if (pread(fd, buf.data(), buf.size(), search_start) != static_cast<ssize_t>(buf.size())) {
        close(fd);
        return -1;
    }

    off_t eocd_offset = -1;
    for (ssize_t i = static_cast<ssize_t>(buf.size()) - 22; i >= 0; --i) {
        if (buf[i] == 0x50 && buf[i + 1] == 0x4b && buf[i + 2] == 0x05 && buf[i + 3] == 0x06) {
            eocd_offset = search_start + i;
            break;
        }
    }
    if (eocd_offset < 0) {
        LOGE("IndexApk: EOCD not found in APK: %s", apk_path);
        close(fd);
        return -1;
    }

    uint8_t eocd[22];
    if (pread(fd, eocd, sizeof(eocd), eocd_offset) != sizeof(eocd)) {
        close(fd);
        return -1;
    }
    const uint16_t num_entries = *reinterpret_cast<uint16_t*>(eocd + 10);
    const uint32_t cd_size = *reinterpret_cast<uint32_t*>(eocd + 12);
    const uint32_t cd_offset = *reinterpret_cast<uint32_t*>(eocd + 16);
    if (num_entries == UINT16_MAX || cd_size == UINT32_MAX || cd_offset == UINT32_MAX ||
        static_cast<uint64_t>(cd_offset) + cd_size > static_cast<uint64_t>(file_len)) {
        LOGE("IndexApk: ZIP64 or invalid central directory: %s", apk_path);
        close(fd);
        return -1;
    }

    std::vector<uint8_t> cd(cd_size);
    if (pread(fd, cd.data(), cd.size(), cd_offset) != static_cast<ssize_t>(cd.size())) {
        close(fd);
        return -1;
    }

    uint32_t pos = 0;
    for (uint16_t i = 0; i < num_entries && pos + 46 <= cd_size; ++i) {
        if (*reinterpret_cast<uint32_t*>(cd.data() + pos) != 0x02014b50) break;
        const uint16_t method = *reinterpret_cast<uint16_t*>(cd.data() + pos + 10);
        const uint32_t usize = *reinterpret_cast<uint32_t*>(cd.data() + pos + 24);
        const uint16_t fn_len = *reinterpret_cast<uint16_t*>(cd.data() + pos + 28);
        const uint16_t extra_len = *reinterpret_cast<uint16_t*>(cd.data() + pos + 30);
        const uint16_t comm_len = *reinterpret_cast<uint16_t*>(cd.data() + pos + 32);
        const uint32_t local_hdr_offset = *reinterpret_cast<uint32_t*>(cd.data() + pos + 42);
        if (pos + 46u + fn_len + extra_len + comm_len > cd_size) break;

        char fn[256] = {0};
        const int cpy = fn_len < 255 ? fn_len : 255;
        memcpy(fn, cd.data() + pos + 46, cpy);
        const bool relevant = strstr(fn, "FMOD") != nullptr || strstr(fn, ".fsb") != nullptr ||
                              strstr(fn, ".fev") != nullptr || strstr(fn, "assets/data/audio") != nullptr ||
                              strstr(fn, "assets/bin/Data/") != nullptr || strstr(fn, ".obb") != nullptr ||
                              strcmp(fn, "assets/com.camouflaj.republique") == 0;
        if (relevant && method == 0 && g_num_apk_entries < 1024) {
            uint8_t local_hdr[30];
            if (pread(fd, local_hdr, sizeof(local_hdr), local_hdr_offset) != sizeof(local_hdr) ||
                *reinterpret_cast<uint32_t*>(local_hdr) != 0x04034b50) {
                close(fd);
                return -1;
            }
            const uint16_t l_fn_len = *reinterpret_cast<uint16_t*>(local_hdr + 26);
            const uint16_t l_extra_len = *reinterpret_cast<uint16_t*>(local_hdr + 28);
            const off_t raw_data_offset = local_hdr_offset + 30 + l_fn_len + l_extra_len;
            if (raw_data_offset < 0 || static_cast<uint64_t>(raw_data_offset) + usize >
                static_cast<uint64_t>(file_len)) {
                close(fd);
                return -1;
            }
            ApkEntry& entry = g_apk_entries[g_num_apk_entries];
            strncpy(entry.name, fn, sizeof(entry.name) - 1);
            entry.data_offset = raw_data_offset;
            entry.file_size = usize;
            entry.apk_fd = fd;
            entry.method = method;
            LOGI("IndexApk entry [%d]: %s (method=%d, size=%u, offset=%lld)",
                 g_num_apk_entries, fn, method, usize, static_cast<long long>(raw_data_offset));
            ++g_num_apk_entries;
        }
        pos += 46u + fn_len + extra_len + comm_len;
    }

    g_apk_indexed_paths.emplace_back(apk_path);
    g_apk_indexed_fds.push_back(fd);
    LOGI("IndexApk: indexed %d direct entries from %s", g_num_apk_entries, apk_path);
    return 0;
}

static ApkEntry* FindApkEntry(const char* path) {
    if (!path || g_num_apk_entries == 0) return nullptr;
    const char* slash = strrchr(path, '/');
    const char* base = slash ? slash + 1 : path;

    // The guest may pass either an APK-relative asset path or an absolute path
    // ending in <base.apk>/assets/.... Preserve the full asset suffix; matching
    // only the basename maps global-metadata.dat and other duplicate names wrong.
    std::string exact_asset;
    if (const char* apk = strstr(path, ".apk/")) {
        exact_asset.assign(apk + 5);
    } else if (strncmp(path, "assets/", 7) == 0) {
        exact_asset = path;
    }

    char preferred[256];
    snprintf(preferred, sizeof(preferred), "assets/data/audio/%s", base);
    char in_assets[256];
    snprintf(in_assets, sizeof(in_assets), "assets/%s", base);
    for (int i = 0; i < g_num_apk_entries; ++i) {
        if ((!exact_asset.empty() && exact_asset == g_apk_entries[i].name) ||
            strcmp(g_apk_entries[i].name, preferred) == 0 ||
            strcmp(g_apk_entries[i].name, path) == 0 ||
            strcmp(g_apk_entries[i].name, in_assets) == 0) {
            return &g_apk_entries[i];
        }
        // République's CRI split stores the patch CPK under this opaque asset name.
        if (strncmp(base, "patch.", 6) == 0 &&
            strcmp(g_apk_entries[i].name, "assets/com.camouflaj.republique") == 0) {
            return &g_apk_entries[i];
        }
    }

    // 3. Language suffix fallback (e.g. VOICE.fsb -> VOICE_eng.fsb)
    std::string base_str(base);
    if (base_str.size() > 4 && base_str.substr(base_str.size() - 4) == ".fsb") {
        std::string stem = base_str.substr(0, base_str.size() - 4);
        std::string eng_variant = "assets/data/audio/" + stem + "_eng.fsb";
        for (int i = 0; i < g_num_apk_entries; ++i) {
            if (g_apk_entries[i].name == eng_variant || strstr(g_apk_entries[i].name, (stem + "_eng").c_str()) != nullptr) {
                return &g_apk_entries[i];
            }
        }
        for (int i = 0; i < g_num_apk_entries; ++i) {
            if (strstr(g_apk_entries[i].name, stem.c_str()) != nullptr) {
                return &g_apk_entries[i];
            }
        }
    }
    return nullptr;
}

static std::string ApkRelativeAssetName(const char* path) {
    if (!path) return {};
    if (const char* apk = strstr(path, ".apk/")) return std::string(apk + 5);
    if (strncmp(path, "assets/", 7) == 0) return path;
    return {};
}

static bool HasApkDirectory(const char* path) {
    std::string asset = ApkRelativeAssetName(path);
    if (asset.empty()) return false;
    if (asset.back() != '/') asset.push_back('/');
    for (int i = 0; i < g_num_apk_entries; ++i) {
        if (strncmp(g_apk_entries[i].name, asset.c_str(), asset.size()) == 0) return true;
    }
    return false;
}

// A file region already laid into the guest arena. The mmap thunk consults this before
// making a new mapping: the arena is bump-allocated, so mapping the same 345 MB expansion
// file a second time is a second permanent hole. Engines do exactly that -- a "reinstall"
// or "reload paks" path closes every archive and reopens it.
struct GuestFileMapping {
    dev_t  dev;
    ino_t  ino;
    off_t  off;      // page-aligned file offset
    size_t span;     // page-aligned length
    uint32_t va;     // guest address
};
static std::vector<GuestFileMapping> g_guest_file_maps;
static std::mutex g_mapping_mutex;

struct VirtualFd {
    int real_apk_fd;
    size_t file_size;
    off_t file_offset; // offset within APK
    off_t current_pos; // 0 .. file_size
};
static std::unordered_map<int, VirtualFd> g_vfd_map;
static int g_next_vfd = 1000;
static std::mutex g_vfd_mutex;

struct GuestZStream {
    z_stream host{};
    uint32_t handle = 0;
    bool initialized = false;
};
static std::unordered_map<uint32_t, GuestZStream> g_zstreams;
static std::mutex g_zstream_mutex;

static bool SyncGuestZStreamToHost(DynarmicHost& host, uint32_t guest_addr, z_stream& stream) {
    if (!GuestRangeValid(host.GetGuestMemorySize(), guest_addr, 56)) return false;
    const uint32_t next_in = host.MemoryRead32(guest_addr + 0);
    const uint32_t next_out = host.MemoryRead32(guest_addr + 12);
    stream.next_in = next_in ? static_cast<Bytef*>(GuestToHost(host.GetGuestMemoryBase(), next_in)) : nullptr;
    stream.avail_in = host.MemoryRead32(guest_addr + 4);
    stream.next_out = next_out ? static_cast<Bytef*>(GuestToHost(host.GetGuestMemoryBase(), next_out)) : nullptr;
    stream.avail_out = host.MemoryRead32(guest_addr + 16);
    return true;
}

static void SyncHostZStreamToGuest(DynarmicHost& host, uint32_t guest_addr, const z_stream& stream,
                                   uint32_t state_handle) {
    if (!GuestRangeValid(host.GetGuestMemorySize(), guest_addr, 56)) return;
    host.MemoryWrite32(guest_addr + 0, stream.next_in ? HostToGuest(host.GetGuestMemoryBase(), stream.next_in) : 0);
    host.MemoryWrite32(guest_addr + 4, stream.avail_in);
    host.MemoryWrite32(guest_addr + 8, static_cast<uint32_t>(stream.total_in));
    host.MemoryWrite32(guest_addr + 12, stream.next_out ? HostToGuest(host.GetGuestMemoryBase(), stream.next_out) : 0);
    host.MemoryWrite32(guest_addr + 16, stream.avail_out);
    host.MemoryWrite32(guest_addr + 20, static_cast<uint32_t>(stream.total_out));
    host.MemoryWrite32(guest_addr + 24, 0);
    host.MemoryWrite32(guest_addr + 28, state_handle);
    host.MemoryWrite32(guest_addr + 32, 0);
    host.MemoryWrite32(guest_addr + 36, 0);
    host.MemoryWrite32(guest_addr + 40, 0);
    host.MemoryWrite32(guest_addr + 44, static_cast<uint32_t>(stream.data_type));
    host.MemoryWrite32(guest_addr + 48, static_cast<uint32_t>(stream.adler));
    host.MemoryWrite32(guest_addr + 52, static_cast<uint32_t>(stream.reserved));
}

// Guest address of the real archive-open function, resolved by
// InstallAssetOverrideHook and tail-called from the veneer.
static uint32_t g_asset_override_real_fn = 0;

static std::unordered_map<uint32_t, FILE*> g_file_map;
static std::unordered_map<uint32_t, int> g_file_vfd_map;

// Android's 32-bit stdio ABI exposes the beginning of __sFILE to native code.
// CRI Ware intentionally reads _flags at +12 and the signed 16-bit _file at +14
// before issuing raw lseek64/fstat calls. A numeric token is therefore not a
// sufficient FILE*: give the guest a real, private ABI-shaped object while the
// map keeps ownership of the AArch64 host FILE.
static constexpr size_t kArm32FileBytes = 84;
static constexpr uint16_t kArm32FileEof = 0x0020;
static constexpr uint16_t kArm32FileError = 0x0040;

static uint32_t AllocateArm32File(DynarmicHost& host, FILE* stream, int exposed_fd,
                                  const char* mode) {
    if (!stream || exposed_fd < INT16_MIN || exposed_fd > INT16_MAX) {
        LOGE("guest FILE allocation rejected stream=%p fd=%d", stream, exposed_fd);
        return 0;
    }
    const uint32_t guest_file = host.GuestAlloc(kArm32FileBytes, 4);
    if (!guest_file) return 0;
    memset(host.GetGuestMemoryBase() + guest_file, 0, kArm32FileBytes);

    uint16_t flags = 0;
    if (mode && strchr(mode, 'r')) flags |= 0x0004;  // __SRD
    if (mode && (strchr(mode, 'w') || strchr(mode, 'a'))) flags |= 0x0008;  // __SWR
    if (mode && strchr(mode, '+')) flags |= 0x0080;  // __SRW
    host.MemoryWrite16(guest_file + 12, flags);
    host.MemoryWrite16(guest_file + 14, static_cast<uint16_t>(static_cast<int16_t>(exposed_fd)));
    return guest_file;
}

// A split-entry FILE* (funopen) and its exposed virtual fd each carry their own
// position: the cookie for stdio, VirtualFd::current_pos for raw read/lseek/
// FIONREAD. Bionic callers treat fileno() as the SAME file description, so the
// two trackers must stay coherent or fseek(FILE)+read(fd) reads stale offsets.
static void SyncVfdFromFile(uint32_t guest_file, FILE* stream) {
    if (!stream) return;
    std::lock_guard<std::mutex> lock(g_vfd_mutex);
    auto vfd_it = g_file_vfd_map.find(guest_file);
    if (vfd_it == g_file_vfd_map.end()) return;
    auto pos_it = g_vfd_map.find(vfd_it->second);
    if (pos_it == g_vfd_map.end()) return;
    pos_it->second.current_pos = ftell(stream);
}

static bool GuestVfdExists(int fd) {
    std::lock_guard<std::mutex> lock(g_vfd_mutex);
    return g_vfd_map.find(fd) != g_vfd_map.end();
}

static void SyncFileFromVfd(int vfd_number) {
    // Raw lseek moved the virtual fd; mirror it onto the stdio stream bound to
    // it (in practice one: each funopen FILE owns a private vfd).
    std::lock_guard<std::mutex> lock(g_vfd_mutex);
    auto pos_it = g_vfd_map.find(vfd_number);
    if (pos_it == g_vfd_map.end()) return;
    const off_t position = pos_it->second.current_pos;
    for (auto& entry : g_file_vfd_map) {
        if (entry.second != vfd_number) continue;
        auto stream_it = g_file_map.find(entry.first);
        if (stream_it != g_file_map.end() && stream_it->second) {
            fseek(stream_it->second, static_cast<long>(position), SEEK_SET);
        }
    }
}

static void SyncArm32FileStatus(DynarmicHost& host, uint32_t guest_file, FILE* stream) {
    if (!stream || !GuestRangeValid(host.GetGuestMemorySize(), guest_file, 16)) return;
    uint16_t flags = host.MemoryRead16(guest_file + 12);
    flags &= static_cast<uint16_t>(~(kArm32FileEof | kArm32FileError));
    if (feof(stream)) flags |= kArm32FileEof;
    if (ferror(stream)) flags |= kArm32FileError;
    host.MemoryWrite16(guest_file + 12, flags);
    SyncVfdFromFile(guest_file, stream);
}

struct GuestDirState {
    DIR* native = nullptr;
    std::vector<std::string> names;
    size_t index = 0;
    uint32_t dirent = 0;
    // Names already returned from the native listing; synthesized overlay names
    // skip them so a split install with extracted OBB files does not enumerate
    // the same archive twice.
    std::vector<std::string> emitted;
};
static std::unordered_map<uint32_t, GuestDirState> g_guest_dirs;
static uint32_t g_next_dir_handle = 0x24000000;
static std::mutex g_dir_mutex;

static std::unordered_map<uint32_t, AAssetManager*> g_asset_managers;
static std::unordered_map<uint32_t, AAsset*> g_assets;
static uint32_t g_next_asset_handle = 0x20000000;

static std::unordered_map<uint32_t, ANativeWindow*> g_native_windows;
static uint32_t g_next_native_window = 0x30000000;
static std::unordered_map<uint32_t, EGLDisplay> g_egl_displays;
static std::unordered_map<uint32_t, EGLConfig> g_egl_configs;
static std::unordered_map<uint32_t, EGLContext> g_egl_contexts;
static std::unordered_map<uint32_t, EGLSurface> g_egl_surfaces;
static uint32_t g_next_egl_handle = 0x31000000;
static std::mutex g_egl_mutex;
void RefreshGlContextBindings();
void TrackGlContext(EGLContext context, EGLContext share);

static thread_local std::unordered_map<uint32_t, void*> g_tls_map;
static thread_local bool g_dlerror_pending = false;
static std::atomic<uint32_t> g_next_tls_key{1};

struct GuestPthreadAttr {
    uint32_t stack_base;
    uint32_t stack_size;
};
static std::unordered_map<uint32_t, GuestPthreadAttr> g_pthread_attrs;
static std::mutex g_pthread_attr_mutex;

struct GuestMutexState {
    std::mutex gate;
    std::condition_variable available;
    std::thread::id owner;
    uint64_t recursion = 0;

    void lock(DynarmicHost& host) {
        std::unique_lock<std::mutex> lock(gate);
        const std::thread::id self = std::this_thread::get_id();
        if (owner == self) { ++recursion; return; }
        while (recursion != 0) {
            available.wait_for(lock, std::chrono::milliseconds(20));
            if (GuestSignalPending()) {
                lock.unlock();
                GuestSignalDeliverPending(host);
                lock.lock();
            }
        }
        owner = self;
        recursion = 1;
    }

    bool try_lock() {
        std::lock_guard<std::mutex> lock(gate);
        const std::thread::id self = std::this_thread::get_id();
        if (owner == self) {
            ++recursion;
            return true;
        }
        if (recursion != 0) return false;
        owner = self;
        recursion = 1;
        return true;
    }

    void unlock() {
        std::unique_lock<std::mutex> lock(gate);
        if (owner != std::this_thread::get_id() || recursion == 0) return;
        if (--recursion == 0) {
            owner = {};
            lock.unlock();
            available.notify_one();
        }
    }
};

struct GuestCondState {
    std::mutex gate;
    std::condition_variable condition;
    std::atomic<uint64_t> generation{0};
};

struct GuestOnceState {
    std::mutex gate;
    std::condition_variable condition;
    bool running = false;
    bool done = false;
    std::thread::id owner;
};

static std::unordered_map<uint32_t, std::shared_ptr<GuestOnceState>> g_guest_onces;

static bool WaitGuestCondition(GuestCondState& cond, GuestMutexState& mutex,
                                DynarmicHost& host, std::chrono::steady_clock::time_point deadline) {
    const uint64_t generation = cond.generation.load();
    mutex.unlock();
    bool signalled = false;
    for (;;) {
        if (cond.generation.load() != generation) { signalled = true; break; }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) break;
        {
            std::unique_lock<std::mutex> lock(cond.gate);
            cond.condition.wait_until(lock, std::min(deadline, now + std::chrono::milliseconds(20)),
                [&] { return cond.generation.load() != generation; });
        }
        // No registry, condition, or guest mutex is held while its handler runs.
        if (GuestSignalPending()) GuestSignalDeliverPending(host);
    }
    mutex.lock(host);
    return signalled;
}

static void NotifyGuestCondition(GuestCondState& cond, bool all) {
    {
        std::lock_guard<std::mutex> lock(cond.gate);
        cond.generation.fetch_add(1);
    }
    if (all) cond.condition.notify_all();
    else cond.condition.notify_one();
}
static std::unordered_map<uint32_t, std::shared_ptr<GuestMutexState>> g_guest_mutexes;
static std::unordered_map<uint32_t, std::shared_ptr<GuestCondState>> g_guest_conds;
static std::mutex g_guest_sync_mutex;
static std::unordered_map<uint32_t, std::shared_ptr<sem_t>> g_guest_semaphores;

struct GuestEnvValue {
    DynarmicHost* host = nullptr;
    uint32_t guest_addr = 0;
};
static std::unordered_map<std::string, GuestEnvValue> g_guest_env;
static std::mutex g_guest_env_mutex;

static std::unordered_map<uint32_t, wctype_t> g_wctype_handles;
static uint32_t g_next_wctype_handle = 0x22000000;

#undef st_atime
#undef st_mtime
#undef st_ctime
#undef st_atime_nsec
#undef st_mtime_nsec
#undef st_ctime_nsec

// This guest uses a 104-byte ARM32 large-file stat layout: mode is at offset
// 16, size at 48, and the 64-bit inode mirror at 96. Write fields by ABI offset
// so the AArch64 host's struct stat padding cannot leak into the guest view.
static void WriteArm32Stat(uint8_t* guest_base, uint32_t gaddr, const struct stat& st) {
    uint8_t* arm_st = static_cast<uint8_t*>(GuestToHost(guest_base, gaddr));
    if (!arm_st) return;
    memset(arm_st, 0, 104);

    const auto put32 = [arm_st](size_t offset, uint32_t value) {
        memcpy(arm_st + offset, &value, sizeof(value));
    };
    const auto put64 = [arm_st](size_t offset, uint64_t value) {
        memcpy(arm_st + offset, &value, sizeof(value));
    };

    put64(0, static_cast<uint64_t>(st.st_dev));
    put32(12, static_cast<uint32_t>(st.st_ino));
    put32(16, static_cast<uint32_t>(st.st_mode));
    put32(20, static_cast<uint32_t>(st.st_nlink));
    put32(24, static_cast<uint32_t>(st.st_uid));
    put32(28, static_cast<uint32_t>(st.st_gid));
    put64(32, static_cast<uint64_t>(st.st_rdev));
    put64(48, static_cast<uint64_t>(st.st_size));
    put32(56, static_cast<uint32_t>(st.st_blksize));
    put64(64, static_cast<uint64_t>(st.st_blocks));
    put32(72, static_cast<uint32_t>(st.st_atim.tv_sec));
    put32(76, static_cast<uint32_t>(st.st_atim.tv_nsec));
    put32(80, static_cast<uint32_t>(st.st_mtim.tv_sec));
    put32(84, static_cast<uint32_t>(st.st_mtim.tv_nsec));
    put32(88, static_cast<uint32_t>(st.st_ctim.tv_sec));
    put32(92, static_cast<uint32_t>(st.st_ctim.tv_nsec));
    put64(96, static_cast<uint64_t>(st.st_ino));
}

// Guest printf formatter
static std::string FormatGuestString(Dynarmic::A32::Jit& jit, DynarmicHost& host, const char* fmt, int first_arg_reg) {
    if (!fmt) return "";
    std::string result;
    int reg_idx = first_arg_reg;
    uint32_t sp = jit.Regs()[13];
    int stack_offset = 0;

    auto get_next_arg = [&]() -> uint32_t {
        if (reg_idx <= 3) {
            return jit.Regs()[reg_idx++];
        } else {
            uint32_t val = host.MemoryRead32(sp + stack_offset);
            stack_offset += 4;
            return val;
        }
    };

    const char* p = fmt;
    while (*p) {
        if (*p == '%' && *(p + 1)) {
            p++;
            if (*p == '%') {
                result += '%';
                p++;
                continue;
            }
            // Skip width/modifiers
            while (*p && (*p == '-' || *p == '+' || *p == ' ' || *p == '#' || *p == '0' ||
                          isdigit(*p) || *p == '.' || *p == 'l' || *p == 'h' || *p == 'z')) {
                p++;
            }
            if (*p == 's') {
                uint32_t gstr = get_next_arg();
                const char* str = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), gstr));
                result += (str ? str : "(null)");
            } else if (*p == 'd' || *p == 'i') {
                int32_t val = static_cast<int32_t>(get_next_arg());
                result += std::to_string(val);
            } else if (*p == 'u') {
                uint32_t val = get_next_arg();
                result += std::to_string(val);
            } else if (*p == 'x' || *p == 'X' || *p == 'p') {
                uint32_t val = get_next_arg();
                char buf[32];
                snprintf(buf, sizeof(buf), "0x%X", val);
                result += buf;
            } else if (*p == 'f' || *p == 'g') {
                uint32_t val = get_next_arg();
                float f = *reinterpret_cast<float*>(&val);
                char buf[32];
                snprintf(buf, sizeof(buf), "%f", f);
                result += buf;
            } else if (*p == 'c') {
                char c = static_cast<char>(get_next_arg());
                result += c;
            } else {
                result += *p;
            }
            if (*p) p++;
        } else {
            result += *p++;
        }
    }
    return result;
}

static std::string FormatGuestVaList(DynarmicHost& host, const char* fmt, uint32_t va_list_guest) {
    if (!fmt || !va_list_guest) return fmt ? fmt : "";
    uint32_t cursor = va_list_guest;
    auto next_word = [&]() {
        const uint32_t value = host.MemoryRead32(cursor);
        cursor += 4;
        return value;
    };
    std::string result;
    for (const char* p = fmt; *p; ++p) {
        if (*p != '%' || !p[1]) { result += *p; continue; }
        ++p;
        if (*p == '%') { result += '%'; continue; }
        while (*p && (*p == '-' || *p == '+' || *p == ' ' || *p == '#' || *p == '0' ||
                      isdigit(*p) || *p == '.' || *p == 'l' || *p == 'h' || *p == 'z')) ++p;
        switch (*p) {
            case 's': {
                const char* value = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), next_word()));
                result += value ? value : "(null)";
                break;
            }
            case 'd': case 'i': result += std::to_string(static_cast<int32_t>(next_word())); break;
            case 'u': result += std::to_string(next_word()); break;
            case 'x': case 'X': case 'p': {
                char value[32]; snprintf(value, sizeof(value), "0x%X", next_word()); result += value; break;
            }
            case 'f': case 'g': {
                cursor = (cursor + 7) & ~7u;
                uint64_t raw = static_cast<uint64_t>(next_word()) | (static_cast<uint64_t>(next_word()) << 32);
                double value; memcpy(&value, &raw, sizeof(value));
                char text[64]; snprintf(text, sizeof(text), "%f", value); result += text; break;
            }
            case 'c': result += static_cast<char>(next_word()); break;
            default: result += *p; break;
        }
    }
    return result;
}

static uint32_t PutNativeWindow(ANativeWindow* window) {
    if (!window) return 0;
    for (const auto& [handle, known] : g_native_windows) {
        if (known == window) return handle;
    }
    const uint32_t handle = g_next_native_window++;
    g_native_windows[handle] = window;
    return handle;
}

static ANativeWindow* GetNativeWindow(uint32_t handle) {
    auto it = g_native_windows.find(handle);
    return it == g_native_windows.end() ? nullptr : it->second;
}

static uint32_t PutEglDisplay(EGLDisplay display) {
    if (display == EGL_NO_DISPLAY) return 0;
    std::lock_guard<std::mutex> lock(g_egl_mutex);
    for (const auto& [handle, known] : g_egl_displays) {
        if (known == display) return handle;
    }
    const uint32_t handle = g_next_egl_handle++;
    g_egl_displays[handle] = display;
    return handle;
}

static EGLDisplay GetEglDisplay(uint32_t handle) {
    std::lock_guard<std::mutex> lock(g_egl_mutex);
    auto it = g_egl_displays.find(handle);
    return it == g_egl_displays.end() ? EGL_NO_DISPLAY : it->second;
}

static uint32_t PutEglConfig(EGLConfig config) {
    if (!config) return 0;
    std::lock_guard<std::mutex> lock(g_egl_mutex);
    for (const auto& [handle, known] : g_egl_configs) {
        if (known == config) return handle;
    }
    const uint32_t handle = g_next_egl_handle++;
    g_egl_configs[handle] = config;
    return handle;
}

static EGLConfig GetEglConfig(uint32_t handle) {
    std::lock_guard<std::mutex> lock(g_egl_mutex);
    auto it = g_egl_configs.find(handle);
    return it == g_egl_configs.end() ? nullptr : it->second;
}

static uint32_t PutEglContext(EGLContext context) {
    if (context == EGL_NO_CONTEXT) return 0;
    std::lock_guard<std::mutex> lock(g_egl_mutex);
    for (const auto& [handle, known] : g_egl_contexts) {
        if (known == context) return handle;
    }
    const uint32_t handle = g_next_egl_handle++;
    g_egl_contexts[handle] = context;
    return handle;
}

static EGLContext GetEglContext(uint32_t handle) {
    std::lock_guard<std::mutex> lock(g_egl_mutex);
    auto it = g_egl_contexts.find(handle);
    return it == g_egl_contexts.end() ? EGL_NO_CONTEXT : it->second;
}

static uint32_t PutEglSurface(EGLSurface surface) {
    if (surface == EGL_NO_SURFACE) return 0;
    std::lock_guard<std::mutex> lock(g_egl_mutex);
    for (const auto& [handle, known] : g_egl_surfaces) {
        if (known == surface) return handle;
    }
    const uint32_t handle = g_next_egl_handle++;
    g_egl_surfaces[handle] = surface;
    return handle;
}

static EGLSurface GetEglSurface(uint32_t handle) {
    std::lock_guard<std::mutex> lock(g_egl_mutex);
    auto it = g_egl_surfaces.find(handle);
    return it == g_egl_surfaces.end() ? EGL_NO_SURFACE : it->second;
}

static bool ReadEglAttribs(DynarmicHost& host, uint32_t address, std::vector<EGLint>& out) {
    out.clear();
    if (!address) { out.push_back(EGL_NONE); return true; }
    for (size_t i = 0; i < 256; ++i) {
        if (!GuestRangeValid(host.GetGuestMemorySize(), address + static_cast<uint32_t>(i * 4), 4)) {
            return false;
        }
        const EGLint value = static_cast<EGLint>(host.MemoryRead32(address + static_cast<uint32_t>(i * 4)));
        out.push_back(value);
        if (value == EGL_NONE) return true;
    }
    return false;
}

void ThunkManager::RegisterLibcThunks() {
    uint8_t* mem = m_host.GetGuestMemoryBase();
    struct EglStringCache {
        std::mutex mutex;
        std::unordered_map<std::string, uint32_t> values;
    };
    auto egl_strings = std::make_shared<EglStringCache>();
    {
        std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
        g_guest_onces.clear();
    }

    *reinterpret_cast<uint32_t*>(mem + 0x100) = 0xDEADBEEF;
    *reinterpret_cast<uint32_t*>(mem + 0x300) = 0;

    RegisterSymbol("__stack_chk_guard", 0x100);
    RegisterSymbol("__sF", 0x200);
    const uint32_t ctype_table = m_host.GuestAlloc(384, 4);
    RegisterSymbol("_ctype_", ctype_table);
    RegisterSymbol("_tolower_tab_", ctype_table);
    RegisterSymbol("_toupper_tab_", ctype_table);
    const uint32_t environ_table = m_host.GuestAlloc(4, 4);
    RegisterSymbol("environ", environ_table);

    RegisterThunk("__errno", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = host.GuestErrnoAddress();
    });

    // -------------------------------------------------------------
    // Additional POSIX & Math Thunks for FMOD
    // -------------------------------------------------------------

    auto read_double_01 = [](const Dynarmic::A32::Jit& jit) -> double {
        uint64_t raw = (static_cast<uint64_t>(jit.Regs()[1]) << 32) | jit.Regs()[0];
        double val;
        memcpy(&val, &raw, sizeof(double));
        return val;
    };
    auto write_double_01 = [](Dynarmic::A32::Jit& jit, double val) {
        uint64_t raw;
        memcpy(&raw, &val, sizeof(double));
        jit.Regs()[0] = static_cast<uint32_t>(raw & 0xFFFFFFFF);
        jit.Regs()[1] = static_cast<uint32_t>((raw >> 32) & 0xFFFFFFFF);
    };

    RegisterThunk("sin", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        write_double_01(jit, sin(read_double_01(jit)));
    });
    RegisterThunk("cos", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        write_double_01(jit, cos(read_double_01(jit)));
    });
    RegisterThunk("tan", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        write_double_01(jit, tan(read_double_01(jit)));
    });
    RegisterThunk("acos", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        write_double_01(jit, acos(read_double_01(jit)));
    });
    RegisterThunk("atan2", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        double y = read_double_01(jit);
        uint64_t raw_x = (static_cast<uint64_t>(jit.Regs()[3]) << 32) | jit.Regs()[2];
        double x;
        memcpy(&x, &raw_x, sizeof(double));
        write_double_01(jit, atan2(y, x));
    });
    RegisterThunk("exp", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        write_double_01(jit, exp(read_double_01(jit)));
    });
    RegisterThunk("log", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        write_double_01(jit, log(read_double_01(jit)));
    });
    RegisterThunk("log10", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        write_double_01(jit, log10(read_double_01(jit)));
    });
    RegisterThunk("log10f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float val = *reinterpret_cast<float*>(&jit.Regs()[0]);
        float res = log10f(val);
        jit.Regs()[0] = *reinterpret_cast<uint32_t*>(&res);
    });
    RegisterThunk("pow", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        double x = read_double_01(jit);
        uint64_t raw_y = (static_cast<uint64_t>(jit.Regs()[3]) << 32) | jit.Regs()[2];
        double y;
        memcpy(&y, &raw_y, sizeof(double));
        write_double_01(jit, pow(x, y));
    });
    RegisterThunk("sqrt", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        write_double_01(jit, sqrt(read_double_01(jit)));
    });
    RegisterThunk("floor", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        write_double_01(jit, floor(read_double_01(jit)));
    });
    RegisterThunk("rint", [=](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        write_double_01(jit, rint(read_double_01(jit)));
    });
    RegisterThunk("lrintf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float val = *reinterpret_cast<float*>(&jit.Regs()[0]);
        jit.Regs()[0] = static_cast<uint32_t>(lrintf(val));
    });
    RegisterThunk("frexp", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint64_t raw = (static_cast<uint64_t>(jit.Regs()[1]) << 32) | jit.Regs()[0];
        double val;
        memcpy(&val, &raw, sizeof(double));
        uint32_t exp_ptr = jit.Regs()[2];
        int exp_val = 0;
        double res = frexp(val, &exp_val);
        if (exp_ptr) host.MemoryWrite32(exp_ptr, static_cast<uint32_t>(exp_val));
        uint64_t res_raw;
        memcpy(&res_raw, &res, sizeof(double));
        jit.Regs()[0] = static_cast<uint32_t>(res_raw & 0xFFFFFFFF);
        jit.Regs()[1] = static_cast<uint32_t>((res_raw >> 32) & 0xFFFFFFFF);
    });
    RegisterThunk("ldexp", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint64_t raw = (static_cast<uint64_t>(jit.Regs()[1]) << 32) | jit.Regs()[0];
        double val;
        memcpy(&val, &raw, sizeof(double));
        int exp_val = static_cast<int>(jit.Regs()[2]);
        double res = ldexp(val, exp_val);
        uint64_t res_raw;
        memcpy(&res_raw, &res, sizeof(double));
        jit.Regs()[0] = static_cast<uint32_t>(res_raw & 0xFFFFFFFF);
        jit.Regs()[1] = static_cast<uint32_t>((res_raw >> 32) & 0xFFFFFFFF);
    });

    // Memory & Strings
    RegisterThunk("getenv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* name = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        if (!name || !*name) { jit.Regs()[0] = 0; return; }

        const char* host_value = std::getenv(name);
        std::string fallback;
        if (!host_value) {
            if (strcmp(name, "ANDROID_ROOT") == 0) fallback = "/system";
            else if (strcmp(name, "ANDROID_DATA") == 0) fallback = "/data";
            else if (strcmp(name, "ANDROID_STORAGE") == 0) fallback = "/storage";
            else if (strcmp(name, "HOME") == 0) fallback = "/data/user/0/com.camouflaj.republique";
            else if (strcmp(name, "TMPDIR") == 0) fallback = "/data/user/0/com.camouflaj.republique/cache";
            else if (strcmp(name, "PATH") == 0) fallback = "/system/bin";
            else { jit.Regs()[0] = 0; return; }
            host_value = fallback.c_str();
        }

        std::lock_guard<std::mutex> lock(g_guest_env_mutex);
        GuestEnvValue& cached = g_guest_env[name];
        if (cached.host != &host || cached.guest_addr == 0) {
            cached.host = &host;
            cached.guest_addr = host.GuestAlloc(strlen(host_value) + 1, 1);
            if (cached.guest_addr) {
                memcpy(GuestToHost(host.GetGuestMemoryBase(), cached.guest_addr),
                       host_value, strlen(host_value) + 1);
            }
        }
        jit.Regs()[0] = cached.guest_addr;
        LOGI("getenv('%s') -> %s", name, cached.guest_addr ? host_value : "<null>");
    });
    RegisterThunk("memmove", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t dst = jit.Regs()[0];
        uint32_t src = jit.Regs()[1];
        uint32_t n = jit.Regs()[2];
        GuestBulkCopy(jit, host, "memmove", dst, src, n, true);
        jit.Regs()[0] = dst;
    });
    RegisterThunk("memmem", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t haystack = jit.Regs()[0];
        uint32_t hlen = jit.Regs()[1];
        uint32_t needle = jit.Regs()[2];
        uint32_t nlen = jit.Regs()[3];
        const void* h_hay = GuestToHost(host.GetGuestMemoryBase(), haystack);
        const void* h_nee = GuestToHost(host.GetGuestMemoryBase(), needle);
        if (h_hay && h_nee && hlen >= nlen) {
            const void* found = memmem(h_hay, hlen, h_nee, nlen);
            if (found) {
                uint32_t offset = static_cast<uint32_t>(reinterpret_cast<const uint8_t*>(found) - reinterpret_cast<const uint8_t*>(h_hay));
                jit.Regs()[0] = haystack + offset;
                return;
            }
        }
        jit.Regs()[0] = 0;
    });
    RegisterThunk("strcat", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t dst = jit.Regs()[0];
        uint32_t src = jit.Regs()[1];
        char* h_dst = reinterpret_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), dst));
        const char* h_src = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), src));
        if (h_dst && h_src) strcat(h_dst, h_src);
        jit.Regs()[0] = dst;
    });
    RegisterThunk("strncpy", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t dst = jit.Regs()[0];
        uint32_t src = jit.Regs()[1];
        uint32_t n = jit.Regs()[2];
        char* h_dst = reinterpret_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), dst));
        const char* h_src = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), src));
        if (h_dst && h_src && n > 0) strncpy(h_dst, h_src, n);
        jit.Regs()[0] = dst;
    });
    RegisterThunk("strncasecmp", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t s1 = jit.Regs()[0];
        uint32_t s2 = jit.Regs()[1];
        uint32_t n = jit.Regs()[2];
        const char* h_s1 = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), s1));
        const char* h_s2 = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), s2));
        jit.Regs()[0] = (h_s1 && h_s2) ? static_cast<uint32_t>(strncasecmp(h_s1, h_s2, n)) : 0;
    });
    RegisterThunk("strtol", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t nptr = jit.Regs()[0];
        uint32_t endptr = jit.Regs()[1];
        int base = static_cast<int>(jit.Regs()[2]);
        const char* h_nptr = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), nptr));
        if (!h_nptr) { jit.Regs()[0] = 0; return; }
        char* h_end = nullptr;
        long val = strtol(h_nptr, &h_end, base);
        if (endptr && h_end) {
            uint32_t offset = static_cast<uint32_t>(h_end - h_nptr);
            host.MemoryWrite32(endptr, nptr + offset);
        }
        jit.Regs()[0] = static_cast<uint32_t>(val);
    });
    RegisterThunk("puts", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t s = jit.Regs()[0];
        const char* str = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), s));
        if (str) LOGI("[puts] %s", str);
        jit.Regs()[0] = 1;
    });
    RegisterThunk("vsnprintf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t buf = jit.Regs()[0];
        uint32_t size = jit.Regs()[1];
        uint32_t fmt = jit.Regs()[2];
        char* h_buf = reinterpret_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), buf));
        const char* h_fmt = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), fmt));
        if (size && !GuestRangeValid(host.GetGuestMemorySize(), buf, size)) {
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        const std::string formatted = FormatGuestVaList(host, h_fmt, jit.Regs()[3]);
        jit.Regs()[0] = static_cast<uint32_t>(snprintf(h_buf, size, "%s", formatted.c_str()));
    });

    // Time & Threading
    RegisterThunk("clock_gettime", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        struct timespec ts{};
        const int result = clock_gettime(static_cast<clockid_t>(jit.Regs()[0]), &ts);
        if (result == 0) {
            host.MemoryWrite32(jit.Regs()[1], static_cast<uint32_t>(ts.tv_sec));
            host.MemoryWrite32(jit.Regs()[1] + 4, static_cast<uint32_t>(ts.tv_nsec));
        }
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    RegisterThunk("clock_getres", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        struct timespec ts{};
        const int result = clock_getres(static_cast<clockid_t>(jit.Regs()[0]), &ts);
        if (result == 0 && jit.Regs()[1]) {
            host.MemoryWrite32(jit.Regs()[1], static_cast<uint32_t>(ts.tv_sec));
            host.MemoryWrite32(jit.Regs()[1] + 4, static_cast<uint32_t>(ts.tv_nsec));
        }
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    RegisterThunk("sysconf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const long result = sysconf(static_cast<int>(jit.Regs()[0]));
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    RegisterThunk("gettid", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = static_cast<uint32_t>(syscall(SYS_gettid));
    });
    RegisterThunk("getpriority", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        errno = 0;
        jit.Regs()[0] = static_cast<uint32_t>(getpriority(static_cast<int>(jit.Regs()[0]),
                                                          static_cast<id_t>(jit.Regs()[1])));
    });
    RegisterThunk("setpriority", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = static_cast<uint32_t>(setpriority(static_cast<int>(jit.Regs()[0]),
                                                          static_cast<id_t>(jit.Regs()[1]),
                                                          static_cast<int>(jit.Regs()[2])));
    });
    RegisterThunk("strchr", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* source = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* result = source ? strchr(source, static_cast<int>(jit.Regs()[1])) : nullptr;
        jit.Regs()[0] = result ? HostToGuest(host.GetGuestMemoryBase(), const_cast<char*>(result)) : 0;
    });
    RegisterThunk("strstr", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* haystack = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* needle = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        const char* result = haystack && needle ? strstr(haystack, needle) : nullptr;
        jit.Regs()[0] = result ? HostToGuest(host.GetGuestMemoryBase(), const_cast<char*>(result)) : 0;
    });
    RegisterThunk("strsep", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t stringp = jit.Regs()[0];
        const char* delimiters = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        const uint32_t current_guest = stringp ? host.MemoryRead32(stringp) : 0;
        char* current = reinterpret_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), current_guest));
        if (!stringp || !current || !delimiters) { jit.Regs()[0] = 0; return; }
        char* result = current;
        current += strcspn(current, delimiters);
        if (*current) {
            *current++ = '\0';
            host.MemoryWrite32(stringp, HostToGuest(host.GetGuestMemoryBase(), current));
        } else {
            host.MemoryWrite32(stringp, 0);
        }
        jit.Regs()[0] = HostToGuest(host.GetGuestMemoryBase(), result);
    });
    RegisterThunk("strncat", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        char* dst = reinterpret_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* src = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        if (dst && src) strncat(dst, src, jit.Regs()[2]);
        jit.Regs()[0] = HostToGuest(host.GetGuestMemoryBase(), dst);
    });
    RegisterThunk("strlcpy", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        char* dst = reinterpret_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* src = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        const size_t size = jit.Regs()[2];
        if (!dst || !src) { jit.Regs()[0] = 0; return; }
        const size_t length = strlen(src);
        if (size) {
            const size_t count = std::min(length, size - 1);
            memcpy(dst, src, count);
            dst[count] = '\0';
        }
        jit.Regs()[0] = static_cast<uint32_t>(length);
    });
    RegisterThunk("strcoll", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* a = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* b = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        jit.Regs()[0] = static_cast<uint32_t>((a && b) ? strcoll(a, b) : 0);
    });
    RegisterThunk("strtod", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* source = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        char* end = nullptr;
        const double value = source ? strtod(source, &end) : 0.0;
        if (jit.Regs()[1]) host.MemoryWrite32(jit.Regs()[1],
            end ? HostToGuest(host.GetGuestMemoryBase(), end) : jit.Regs()[0]);
        WriteDF(jit, 0, value);
    });
    RegisterThunk("strtoull", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* source = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const unsigned long long value = source ? strtoull(source, nullptr, static_cast<int>(jit.Regs()[2])) : 0;
        jit.Regs()[0] = static_cast<uint32_t>(value);
        jit.Regs()[1] = static_cast<uint32_t>(value >> 32);
    });
    RegisterThunk("div", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t result = jit.Regs()[0];
        const int numerator = static_cast<int32_t>(jit.Regs()[1]);
        const int denominator = static_cast<int32_t>(jit.Regs()[2]);
        if (!GuestRangeValid(host.GetGuestMemorySize(), result, sizeof(div_t)) ||
            denominator == 0 || (numerator == INT_MIN && denominator == -1)) {
            LOGE("Invalid guest div result=0x%08X numerator=%d denominator=%d", result,
                 numerator, denominator);
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
            return;
        }
        const div_t value = std::div(numerator, denominator);
        host.MemoryWrite32(result, static_cast<uint32_t>(value.quot));
        host.MemoryWrite32(result + 4, static_cast<uint32_t>(value.rem));
        jit.Regs()[0] = result;
    });
    RegisterThunk("gettimeofday", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t tv_ptr = jit.Regs()[0];
        if (tv_ptr) {
            struct timeval tv;
            gettimeofday(&tv, nullptr);
            host.MemoryWrite32(tv_ptr + 0, static_cast<uint32_t>(tv.tv_sec));
            host.MemoryWrite32(tv_ptr + 4, static_cast<uint32_t>(tv.tv_usec));
        }
        jit.Regs()[0] = 0;
    });
    RegisterThunk("usleep", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const useconds_t duration_us = static_cast<useconds_t>(jit.Regs()[0]);
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOGI("usleep duration_us=%u lr=0x%08X", duration_us, jit.Regs()[14]);
        }
        const int result = ::usleep(duration_us);
        if (result != 0) host.MemoryWrite32(host.GuestErrnoAddress(), errno);
        if (GuestSignalPending()) GuestSignalDeliverPending(host);
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    RegisterThunk("lrand48", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = static_cast<uint32_t>(lrand48());
    });
    RegisterThunk("srand48", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        srand48(static_cast<long>(jit.Regs()[0]));
        jit.Regs()[0] = 0;
    });
    RegisterThunk("utime", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = 0;
    });
    RegisterThunk("unlink", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* path = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        jit.Regs()[0] = path ? static_cast<uint32_t>(::unlink(path)) : static_cast<uint32_t>(-1);
    });
    RegisterThunk("chown", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = 0;
    });
    RegisterThunk("fcntl", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = 0;
    });

    // Host semaphores stay in stable storage: the ARM32 sem_t is only four bytes.
    RegisterThunk("sem_init", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t address = jit.Regs()[0];
        if (!GuestRangeValid(host.GetGuestMemorySize(), address, 4) || jit.Regs()[1] != 0) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EINVAL);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        auto* native = new sem_t{};
        const int result = ::sem_init(native, 0, jit.Regs()[2]);
        if (result == 0) {
            std::shared_ptr<sem_t> state(native, [](sem_t* p) { ::sem_destroy(p); delete p; });
            std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
            g_guest_semaphores[address] = std::move(state);
            host.MemoryWrite32(address, 0);
        } else {
            delete native;
            host.MemoryWrite32(host.GuestErrnoAddress(), errno);
        }
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    RegisterThunk("sem_destroy", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
        const bool removed = g_guest_semaphores.erase(jit.Regs()[0]) != 0;
        if (!removed) host.MemoryWrite32(host.GuestErrnoAddress(), EINVAL);
        jit.Regs()[0] = removed ? 0 : static_cast<uint32_t>(-1);
    });
    const auto find_guest_semaphore = [](uint32_t address) {
        std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
        auto it = g_guest_semaphores.find(address);
        return it == g_guest_semaphores.end() ? std::shared_ptr<sem_t>() : it->second;
    };
    const auto register_semaphore_op = [&](const char* name, int (*operation)(sem_t*)) {
        RegisterThunk(name, [operation, find_guest_semaphore](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
            auto state = find_guest_semaphore(jit.Regs()[0]);
            const int result = state ? operation(state.get()) : -1;
            if (result < 0) host.MemoryWrite32(host.GuestErrnoAddress(), state ? errno : EINVAL);
            jit.Regs()[0] = static_cast<uint32_t>(result);
        });
    };
    RegisterThunk("sem_wait", [find_guest_semaphore](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        auto state = find_guest_semaphore(jit.Regs()[0]);
        if (!state) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EINVAL);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        // Native posts wake immediately; the short timeout is only a guest-signal safepoint.
        for (;;) {
            timespec deadline{};
            clock_gettime(CLOCK_REALTIME, &deadline);
            deadline.tv_nsec += 20000000;
            if (deadline.tv_nsec >= 1000000000) { ++deadline.tv_sec; deadline.tv_nsec -= 1000000000; }
            const int result = ::sem_timedwait(state.get(), &deadline);
            const int error = errno;
            if (GuestSignalPending()) GuestSignalDeliverPending(host);
            if (result == 0) { jit.Regs()[0] = 0; return; }
            if (error == ETIMEDOUT) continue;
            host.MemoryWrite32(host.GuestErrnoAddress(), error);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
    });
    register_semaphore_op("sem_trywait", ::sem_trywait);
    register_semaphore_op("sem_post", ::sem_post);
    RegisterThunk("pthread_self", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = DynarmicHost::CurrentGuestThreadId();
    });
    RegisterThunk("pthread_equal", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = jit.Regs()[0] == jit.Regs()[1];
    });
    RegisterThunk("pthread_join", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t id = jit.Regs()[0];
        const int result = host.JoinGuestThread(id);
        if (result != 0) LOGE("pthread_join id=0x%08X failed: %d", id, result);
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    RegisterThunk("sched_yield", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = static_cast<uint32_t>(::sched_yield());
    });
    RegisterThunk("nanosleep", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t request = jit.Regs()[0];
        const uint32_t remaining = jit.Regs()[1];
        if (!GuestRangeValid(host.GetGuestMemorySize(), request, 8)) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EFAULT);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        timespec duration{
            static_cast<time_t>(static_cast<int32_t>(host.MemoryRead32(request))),
            static_cast<long>(host.MemoryRead32(request + 4))};
        if (duration.tv_sec < 0 || duration.tv_nsec < 0 || duration.tv_nsec >= 1000000000L) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EINVAL);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        timespec leftover{};
        const int result = ::nanosleep(&duration, remaining ? &leftover : nullptr);
        if (result != 0) host.MemoryWrite32(host.GuestErrnoAddress(), errno);
        if (result != 0 && remaining) {
            host.MemoryWrite32(remaining, static_cast<uint32_t>(leftover.tv_sec));
            host.MemoryWrite32(remaining + 4, static_cast<uint32_t>(leftover.tv_nsec));
        }
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    RegisterThunk("pthread_attr_init", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t attr = jit.Regs()[0];
        if (attr) {
            std::lock_guard<std::mutex> lock(g_pthread_attr_mutex);
            g_pthread_attrs[attr] = {host.GuestStackBase(), host.GuestStackSize()};
        }
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_attr_destroy", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        std::lock_guard<std::mutex> lock(g_pthread_attr_mutex);
        g_pthread_attrs.erase(jit.Regs()[0]);
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_attr_setdetachstate", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_attr_setstacksize", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t attr = jit.Regs()[0];
        if (attr) {
            std::lock_guard<std::mutex> lock(g_pthread_attr_mutex);
            g_pthread_attrs[attr] = {host.GuestStackBase(), jit.Regs()[1]};
        }
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_getattr_np", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t attr = jit.Regs()[1];
        if (attr) {
            std::lock_guard<std::mutex> lock(g_pthread_attr_mutex);
            g_pthread_attrs[attr] = {host.GuestStackBase(), host.GuestStackSize()};
            LOGI("pthread_getattr_np attr=0x%08X stack=0x%08X/%u", attr,
                 host.GuestStackBase(), host.GuestStackSize());
        }
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_attr_getstack", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t attr = jit.Regs()[0];
        GuestPthreadAttr value{host.GuestStackBase(), host.GuestStackSize()};
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(g_pthread_attr_mutex);
            auto it = g_pthread_attrs.find(attr);
            if (it != g_pthread_attrs.end()) { value = it->second; found = true; }
        }
        if (jit.Regs()[1]) host.MemoryWrite32(jit.Regs()[1], value.stack_base);
        if (jit.Regs()[2]) host.MemoryWrite32(jit.Regs()[2], value.stack_size);
        LOGI("pthread_attr_getstack attr=0x%08X found=%d stack=0x%08X/%u out=0x%08X size=0x%08X values=0x%08X/%u", attr,
             found ? 1 : 0, value.stack_base, value.stack_size, jit.Regs()[1], jit.Regs()[2],
             jit.Regs()[1] ? host.MemoryRead32(jit.Regs()[1]) : 0,
             jit.Regs()[2] ? host.MemoryRead32(jit.Regs()[2]) : 0);
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_mutex_init", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t mtx = jit.Regs()[0];
        if (mtx) {
            std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
            g_guest_mutexes[mtx] = std::make_shared<GuestMutexState>();
            host.MemoryWrite32(mtx, 0);
        }
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_mutex_destroy", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
        g_guest_mutexes.erase(jit.Regs()[0]);
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_mutexattr_init", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_mutexattr_settype", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_create", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t thread_out = jit.Regs()[0];
        uint32_t start_func = jit.Regs()[2];
        uint32_t arg = jit.Regs()[3];
        LOGI("pthread_create: thread_out=0x%08X, func=0x%08X, arg=0x%08X", thread_out, start_func, arg);
        const uint32_t id = host.StartGuestThread(start_func, arg);
        if (!id) { jit.Regs()[0] = EAGAIN; return; }
        // pthread_create owns only thread_out, never the opaque caller-owned argument.
        if (start_func == 0x40CEFF10 && GuestRangeValid(host.GetGuestMemorySize(), arg, 48)) {
            LOGI("Unity worker role: id=0x%08X func=0x%08X arg=0x%08X", id,
                 host.MemoryRead32(arg + 28), host.MemoryRead32(arg + 24));
        }
        if (thread_out) host.MemoryWrite32(thread_out, id);
        jit.Regs()[0] = 0; // Return OK
    });

    // Sockets & Net (Stubs)
    RegisterThunk("socket", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0xFFFFFFFF; });
    RegisterThunk("bind", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0xFFFFFFFF; });
    RegisterThunk("listen", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0xFFFFFFFF; });
    RegisterThunk("accept", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0xFFFFFFFF; });
    RegisterThunk("connect", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0xFFFFFFFF; });
    RegisterThunk("send", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0xFFFFFFFF; });
    RegisterThunk("recv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0xFFFFFFFF; });
    RegisterThunk("select", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("setsockopt", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0xFFFFFFFF; });
    RegisterThunk("inet_addr", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("gethostbyname", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });

    RegisterThunk("ANativeWindow_fromSurface", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        JNIEnv* env = FakeJniEnv::Instance().GetEnv();
        jobject surface = FakeJniEnv::Instance().ResolveObject(jit.Regs()[1]);
        jit.Regs()[0] = PutNativeWindow((env && surface) ? ANativeWindow_fromSurface(env, surface) : nullptr);
    });
    RegisterThunk("ANativeWindow_acquire", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        if (ANativeWindow* window = GetNativeWindow(jit.Regs()[0])) ANativeWindow_acquire(window);
    });
    RegisterThunk("ANativeWindow_release", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        if (ANativeWindow* window = GetNativeWindow(jit.Regs()[0])) ANativeWindow_release(window);
    });
    RegisterThunk("ANativeWindow_getWidth", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        ANativeWindow* window = GetNativeWindow(jit.Regs()[0]);
        jit.Regs()[0] = window ? static_cast<uint32_t>(ANativeWindow_getWidth(window)) : 0;
    });
    RegisterThunk("ANativeWindow_getHeight", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        ANativeWindow* window = GetNativeWindow(jit.Regs()[0]);
        jit.Regs()[0] = window ? static_cast<uint32_t>(ANativeWindow_getHeight(window)) : 0;
    });
    RegisterThunk("ANativeWindow_setBuffersGeometry", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        ANativeWindow* window = GetNativeWindow(jit.Regs()[0]);
        jit.Regs()[0] = window ? static_cast<uint32_t>(ANativeWindow_setBuffersGeometry(
            window, static_cast<int32_t>(jit.Regs()[1]), static_cast<int32_t>(jit.Regs()[2]),
            static_cast<int32_t>(jit.Regs()[3]))) : static_cast<uint32_t>(-1);
    });

    RegisterThunk("eglGetDisplay", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = PutEglDisplay(eglGetDisplay(EGL_DEFAULT_DISPLAY));
    });
    RegisterThunk("eglInitialize", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        EGLDisplay display = GetEglDisplay(jit.Regs()[0]);
        EGLint major = 0, minor = 0;
        const EGLBoolean ok = display != EGL_NO_DISPLAY ? eglInitialize(display, &major, &minor) : EGL_FALSE;
        if (jit.Regs()[1]) host.MemoryWrite32(jit.Regs()[1], static_cast<uint32_t>(major));
        if (jit.Regs()[2]) host.MemoryWrite32(jit.Regs()[2], static_cast<uint32_t>(minor));
        jit.Regs()[0] = ok;
    });
    RegisterThunk("eglChooseConfig", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        std::vector<EGLint> attribs;
        EGLDisplay display = GetEglDisplay(jit.Regs()[0]);
        const GLsizei capacity = static_cast<GLsizei>(jit.Regs()[3]);
        const uint32_t count_address = host.MemoryRead32(jit.Regs()[13]);
        if (display == EGL_NO_DISPLAY || !ReadEglAttribs(host, jit.Regs()[1], attribs) || capacity < 0) {
            jit.Regs()[0] = EGL_FALSE;
            return;
        }
        std::vector<EGLConfig> configs(static_cast<size_t>(capacity));
        EGLint count = 0;
        const EGLBoolean ok = eglChooseConfig(display, attribs.data(), configs.data(), capacity, &count);
        for (EGLint i = 0; i < count && jit.Regs()[2]; ++i) {
            host.MemoryWrite32(jit.Regs()[2] + static_cast<uint32_t>(i * 4), PutEglConfig(configs[i]));
        }
        if (count_address) host.MemoryWrite32(count_address, static_cast<uint32_t>(count));
        jit.Regs()[0] = ok;
    });
    RegisterThunk("eglCreateContext", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        std::vector<EGLint> attribs;
        EGLDisplay display = GetEglDisplay(jit.Regs()[0]);
        EGLConfig config = GetEglConfig(jit.Regs()[1]);
        EGLContext share = GetEglContext(jit.Regs()[2]);
        if (display == EGL_NO_DISPLAY || !config || !ReadEglAttribs(host, jit.Regs()[3], attribs)) {
            jit.Regs()[0] = 0;
            return;
        }
        const EGLContext context = eglCreateContext(display, config, share, attribs.data());
        TrackGlContext(context, share);
        jit.Regs()[0] = PutEglContext(context);
    });
    RegisterThunk("eglCreateWindowSurface", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        std::vector<EGLint> attribs;
        EGLDisplay display = GetEglDisplay(jit.Regs()[0]);
        EGLConfig config = GetEglConfig(jit.Regs()[1]);
        ANativeWindow* window = GetNativeWindow(jit.Regs()[2]);
        if (display == EGL_NO_DISPLAY || !config || !window || !ReadEglAttribs(host, jit.Regs()[3], attribs)) {
            jit.Regs()[0] = 0;
            return;
        }
        jit.Regs()[0] = PutEglSurface(eglCreateWindowSurface(display, config, window, attribs.data()));
    });
    RegisterThunk("eglCreatePbufferSurface", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        std::vector<EGLint> attribs;
        EGLDisplay display = GetEglDisplay(jit.Regs()[0]);
        EGLConfig config = GetEglConfig(jit.Regs()[1]);
        if (display == EGL_NO_DISPLAY || !config || !ReadEglAttribs(host, jit.Regs()[2], attribs)) {
            jit.Regs()[0] = 0;
            return;
        }
        jit.Regs()[0] = PutEglSurface(eglCreatePbufferSurface(display, config, attribs.data()));
    });
    RegisterThunk("eglMakeCurrent", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const uint32_t display = jit.Regs()[0], draw = jit.Regs()[1];
        const uint32_t read = jit.Regs()[2], context = jit.Regs()[3];
        jit.Regs()[0] = eglMakeCurrent(GetEglDisplay(display), GetEglSurface(draw),
                                       GetEglSurface(read), GetEglContext(context));
        if (jit.Regs()[0]) RefreshGlContextBindings();
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1) < 12)
            LOGI("eglMakeCurrent display=0x%08X draw=0x%08X read=0x%08X context=0x%08X -> %u current=0x%08X",
                 display, draw, read, context, jit.Regs()[0], PutEglContext(eglGetCurrentContext()));
    });
    RegisterThunk("eglSwapBuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
#if CMR_ANDROID_LOG_PRINT && !defined(BRIDGE_RELEASE_LOGS)
        const auto start = std::chrono::steady_clock::now();
#endif
        jit.Regs()[0] = eglSwapBuffers(GetEglDisplay(jit.Regs()[0]), GetEglSurface(jit.Regs()[1]));
#if CMR_ANDROID_LOG_PRINT && !defined(BRIDGE_RELEASE_LOGS)
        const uint64_t elapsed_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count());
        static std::atomic<uint64_t> calls{0};
        static std::atomic<uint64_t> sum_us{0};
        static std::atomic<uint64_t> max_us{0};
        const uint64_t call = calls.fetch_add(1, std::memory_order_relaxed) + 1;
        sum_us.fetch_add(elapsed_us, std::memory_order_relaxed);
        uint64_t prior_max = max_us.load(std::memory_order_relaxed);
        while (prior_max < elapsed_us &&
               !max_us.compare_exchange_weak(prior_max, elapsed_us, std::memory_order_relaxed)) {}
        if (call <= 4) {
            LOGI("eglSwapBuffers call=%" PRIu64 " elapsed_us=%" PRIu64 " result=%u",
                 call, elapsed_us, jit.Regs()[0]);
        } else if (call % 120 == 0) {
            const uint64_t window_sum = sum_us.exchange(0, std::memory_order_relaxed);
            const uint64_t window_max = max_us.exchange(0, std::memory_order_relaxed);
            LOGI("eglSwapBuffers call=%" PRIu64 " avg_us=%" PRIu64 " max_us=%" PRIu64 " result=%u",
                 call, window_sum / 120, window_max, jit.Regs()[0]);
        }
#endif
    });
    RegisterThunk("eglSwapInterval", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const int32_t requested = static_cast<int32_t>(jit.Regs()[1]);
        jit.Regs()[0] = eglSwapInterval(GetEglDisplay(jit.Regs()[0]), static_cast<EGLint>(requested));
#if CMR_ANDROID_LOG_PRINT && !defined(BRIDGE_RELEASE_LOGS)
        static std::atomic<unsigned> calls{0};
        const unsigned call = calls.fetch_add(1, std::memory_order_relaxed) + 1;
        LOGI("eglSwapInterval call=%u requested=%d result=%u", call, requested, jit.Regs()[0]);
#endif
    });
    RegisterThunk("eglDestroyContext", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        EGLDisplay display = GetEglDisplay(jit.Regs()[0]);
        EGLContext context = GetEglContext(jit.Regs()[1]);
        if (display != EGL_NO_DISPLAY && context != EGL_NO_CONTEXT &&
            context == eglGetCurrentContext()) {
            ResetGlSyncRegistry();
        }
        jit.Regs()[0] = display != EGL_NO_DISPLAY ? eglDestroyContext(display, context) : EGL_FALSE;
    });
    RegisterThunk("eglDestroySurface", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        EGLDisplay display = GetEglDisplay(jit.Regs()[0]);
        EGLSurface surface = GetEglSurface(jit.Regs()[1]);
        jit.Regs()[0] = display != EGL_NO_DISPLAY ? eglDestroySurface(display, surface) : EGL_FALSE;
    });
    RegisterThunk("eglTerminate", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        EGLDisplay display = GetEglDisplay(jit.Regs()[0]);
        if (display != EGL_NO_DISPLAY && eglGetCurrentContext() != EGL_NO_CONTEXT) {
            ResetGlSyncRegistry();
        }
        jit.Regs()[0] = eglTerminate(display);
    });
    RegisterThunk("eglGetError", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = eglGetError();
    });
    RegisterThunk("eglGetCurrentContext", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = PutEglContext(eglGetCurrentContext());
    });
    RegisterThunk("eglGetCurrentSurface", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = PutEglSurface(eglGetCurrentSurface(static_cast<EGLint>(jit.Regs()[0])));
    });
    RegisterThunk("eglGetConfigAttrib", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        EGLint value = 0;
        EGLDisplay display = GetEglDisplay(jit.Regs()[0]);
        EGLConfig config = GetEglConfig(jit.Regs()[1]);
        EGLBoolean ok = (display != EGL_NO_DISPLAY && config) ?
            eglGetConfigAttrib(display, config, static_cast<EGLint>(jit.Regs()[2]), &value) : EGL_FALSE;
        if (ok && jit.Regs()[3]) host.MemoryWrite32(jit.Regs()[3], static_cast<uint32_t>(value));
        jit.Regs()[0] = ok;
    });
    RegisterThunk("eglQuerySurface", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        EGLint value = 0;
        EGLDisplay display = GetEglDisplay(jit.Regs()[0]);
        EGLSurface surface = GetEglSurface(jit.Regs()[1]);
        EGLBoolean ok = (display != EGL_NO_DISPLAY && surface) ?
            eglQuerySurface(display, surface, static_cast<EGLint>(jit.Regs()[2]), &value) : EGL_FALSE;
        if (ok && jit.Regs()[3]) host.MemoryWrite32(jit.Regs()[3], static_cast<uint32_t>(value));
        jit.Regs()[0] = ok;
    });
    // A query made before eglInitialize/eglMakeCurrent may legitimately return
    // NULL. Never cache that transient failure, and keep guest pointers scoped
    // to this bridge instance instead of sharing them across guest memories.
    RegisterThunk("eglQueryString", [egl_strings](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t display_handle = jit.Regs()[0];
        const uint32_t name = jit.Regs()[1];
        const char* value = eglQueryString(GetEglDisplay(display_handle), static_cast<EGLint>(name));
        if (!value) {
            jit.Regs()[0] = 0;
            return;
        }

        const std::string text(value);
        std::lock_guard<std::mutex> lock(egl_strings->mutex);
        auto found = egl_strings->values.find(text);
        if (found == egl_strings->values.end()) {
            const uint32_t guest_addr = host.GuestAlloc(text.size() + 1, 1);
            if (!guest_addr) {
                jit.Regs()[0] = 0;
                return;
            }
            memcpy(host.GetGuestMemoryBase() + guest_addr, text.c_str(), text.size() + 1);
            found = egl_strings->values.emplace(text, guest_addr).first;
        }
        jit.Regs()[0] = found->second;
    });
    RegisterThunk("eglGetProcAddress", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* name = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        if (!name) { jit.Regs()[0] = 0; return; }
        jit.Regs()[0] = host.GetThunkManager().FindRegisteredSymbol(name);
    });

    // Android asset manager handles. Host pointers never cross the 32-bit guest ABI.
    RegisterThunk("AAssetManager_fromJava", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        JNIEnv* env = FakeJniEnv::Instance().GetEnv();
        jobject manager_obj = FakeJniEnv::Instance().ResolveObject(jit.Regs()[1]);
        AAssetManager* manager = (env && manager_obj) ? AAssetManager_fromJava(env, manager_obj) : nullptr;
        if (!manager) { jit.Regs()[0] = 0; return; }
        for (const auto& [handle, known] : g_asset_managers) {
            if (known == manager) { jit.Regs()[0] = handle; return; }
        }
        const uint32_t handle = g_next_asset_handle++;
        g_asset_managers[handle] = manager;
        jit.Regs()[0] = handle;
    });

    RegisterThunk("AAssetManager_open", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        auto manager_it = g_asset_managers.find(jit.Regs()[0]);
        const char* filename = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        if (manager_it == g_asset_managers.end() || !filename) { jit.Regs()[0] = 0; return; }
        AAsset* asset = AAssetManager_open(manager_it->second, filename, static_cast<int>(jit.Regs()[2]));
        if (!asset) { jit.Regs()[0] = 0; return; }
        const uint32_t handle = g_next_asset_handle++;
        g_assets[handle] = asset;
        jit.Regs()[0] = handle;
    });

    RegisterThunk("AAsset_close", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        auto it = g_assets.find(jit.Regs()[0]);
        if (it != g_assets.end()) {
            AAsset_close(it->second);
            g_assets.erase(it);
        }
    });

    RegisterThunk("AAsset_getLength", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        auto it = g_assets.find(jit.Regs()[0]);
        jit.Regs()[0] = it != g_assets.end() ? static_cast<uint32_t>(AAsset_getLength(it->second)) : 0;
    });

    RegisterThunk("AAsset_read", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        auto it = g_assets.find(jit.Regs()[0]);
        void* buffer = GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]);
        const size_t count = jit.Regs()[2];
        jit.Regs()[0] = (it != g_assets.end() && buffer) ?
            static_cast<uint32_t>(AAsset_read(it->second, buffer, count)) : static_cast<uint32_t>(-1);
    });

    RegisterThunk("AAsset_seek", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        auto it = g_assets.find(jit.Regs()[0]);
        const off_t offset = static_cast<off_t>(static_cast<int32_t>(jit.Regs()[1]));
        const int whence = static_cast<int>(jit.Regs()[2]);
        jit.Regs()[0] = it != g_assets.end() ? static_cast<uint32_t>(AAsset_seek(it->second, offset, whence)) : static_cast<uint32_t>(-1);
    });

    // Android & Runtime
    RegisterThunk("__android_log_write", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
#if CMR_ANDROID_LOG_PRINT
        int prio = static_cast<int>(jit.Regs()[0]);
        uint32_t tag_ptr = jit.Regs()[1];
        uint32_t msg_ptr = jit.Regs()[2];
        const char* tag = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), tag_ptr));
        const char* msg = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), msg_ptr));
        if (tag && msg && BRIDGE_GUEST_LOG_PRIORITY_OK(prio)) __android_log_write(prio, tag, msg);
#else
        (void)host;
#endif
        jit.Regs()[0] = 1;
    });
    RegisterThunk("__aeabi_atexit", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("__cxa_pure_virtual", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        LOGE("__cxa_pure_virtual called!");
        jit.Regs()[0] = 0;
    });
    RegisterThunk("__cxa_type_match", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("__cxa_begin_cleanup", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("__cxa_call_unexpected", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("__gnu_Unwind_Find_exidx", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t pc = jit.Regs()[0];
        const uint32_t count_out = jit.Regs()[1];
        uint32_t count = 0;
        const uint32_t table = host.FindGuestExidx(pc, count);
        if (count_out && !GuestRangeValid(host.GetGuestMemorySize(), count_out, 4)) {
            LOGE("Invalid guest __gnu_Unwind_Find_exidx count pointer 0x%08X", count_out);
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
            return;
        }
        if (count_out) host.MemoryWrite32(count_out, table ? count : 0u);
        jit.Regs()[0] = table;
        static std::atomic<unsigned> logged{0};
        if (!table || logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOGI("__gnu_Unwind_Find_exidx pc=0x%08X table=0x%08X count=%u",
                 pc, table, table ? count : 0u);
        }
    });
    RegisterThunk("dlopen", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t name_ptr = jit.Regs()[0];
        const char* name = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), name_ptr));
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOGI("dlopen('%s', %d)", name ? name : "NULL", jit.Regs()[1]);
        }
        jit.Regs()[0] = 0x50005000; // Non-null handle
    });
    RegisterThunk("dlclose", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    constexpr char dl_message[] = "Symbol is not provided by the ARM32 bridge";
    const uint32_t dl_message_address = m_host.GuestAlloc(sizeof(dl_message), 1);
    if (!dl_message_address) { LOGE("No storage for dlerror"); std::abort(); }
    memcpy(mem + dl_message_address, dl_message, sizeof(dl_message));
    g_dlerror_pending = false;
    RegisterThunk("dlerror", [dl_message_address](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = g_dlerror_pending ? dl_message_address : 0;
        g_dlerror_pending = false;
    });
    RegisterThunk("dlsym", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sym_ptr = jit.Regs()[1];
        const char* sym = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), sym_ptr));
        static std::atomic<unsigned> logged{0};
        const bool log_lookup = logged.fetch_add(1, std::memory_order_relaxed) < 64;
        if (log_lookup) LOGI("dlsym(handle, '%s')", sym ? sym : "NULL");
        if (sym) {
            if (strncmp(sym, "SL_IID_", 7) == 0) {
                uint32_t iid_ptr = GetOpenSLESInterfacePointer(sym);
                if (iid_ptr != 0) {
                    jit.Regs()[0] = iid_ptr;
                    return;
                }
            }
            if (strcmp(sym, "slCreateEngine") == 0) {
                jit.Regs()[0] = host.GetThunkManager().GetOrCreateThunkAddress("slCreateEngine");
                return;
            }
            const uint32_t registered = host.GetThunkManager().FindRegisteredSymbol(sym);
            if (registered) {
                jit.Regs()[0] = registered;
                return;
            }
            uint32_t addr = host.FindSymbol(sym);
            if (addr != 0) {
                if (log_lookup && strncmp(sym, "il2cpp_", 7) == 0)
                    LOGI("dlsym resolved %s -> 0x%08X", sym, addr);
                jit.Regs()[0] = addr;
                return;
            }
            // P/Invoke probes decorated names (including W) before the real export.
            // A missing symbol must be null, not a callable success-looking stub.
            g_dlerror_pending = true;
            jit.Regs()[0] = 0;
            return;
        }
        g_dlerror_pending = true;
        jit.Regs()[0] = 0;
    });

    RegisterThunk("getauxval", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const unsigned long key = jit.Regs()[0];
        unsigned long value = getauxval(key);
#if defined(__aarch64__)
        // AArch64 names ASIMD where the 32-bit guest expects NEON.
        if (key == AT_HWCAP && (value & (1UL << 1))) value |= (1UL << 12);
#endif
        jit.Regs()[0] = static_cast<uint32_t>(value);
    });

    RegisterThunk("__system_property_get", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* name = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        char* out = reinterpret_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        LOGI("__system_property_get('%s')", name ? name : "null");
        if (!name || !out) { jit.Regs()[0] = 0; return; }
        const char* value = nullptr;
        if (strcmp(name, "ro.product.cpu.abi") == 0) value = "armeabi-v7a";
        else if (strcmp(name, "ro.product.cpu.abi2") == 0) value = "armeabi";
        else if (strcmp(name, "ro.product.cpu.abilist") == 0) value = "armeabi-v7a,armeabi";
        if (value) {
            strcpy(out, value);
            jit.Regs()[0] = static_cast<uint32_t>(strlen(value));
            return;
        }
        const int result = __system_property_get(name, out);
        jit.Regs()[0] = result > 0 ? static_cast<uint32_t>(result) : 0;
    });

    // C++ Memory Allocation operators
    RegisterThunk("_Znaj", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        size_t size = jit.Regs()[0];
        jit.Regs()[0] = host.GuestAlloc(size, 8);
    });

    RegisterThunk("_Znwj", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        size_t size = jit.Regs()[0];
        jit.Regs()[0] = host.GuestAlloc(size, 8);
    });

    RegisterThunk("_ZnajRKSt9nothrow_t", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        size_t size = jit.Regs()[0];
        jit.Regs()[0] = host.GuestAlloc(size, 8);
    });

    RegisterThunk("_ZnwjRKSt9nothrow_t", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        size_t size = jit.Regs()[0];
        jit.Regs()[0] = host.GuestAlloc(size, 8);
    });

    RegisterThunk("_ZdaPv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        host.GuestFree(jit.Regs()[0]);
    });

    RegisterThunk("_ZdlPv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        host.GuestFree(jit.Regs()[0]);
    });

    RegisterThunk("_ZNSt8ios_base4InitC1Ev", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {});
    RegisterThunk("_ZNSt8ios_base4InitD1Ev", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {});

    // Memory Allocation inside guest space
    RegisterThunk("malloc", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        size_t size = jit.Regs()[0];
        jit.Regs()[0] = host.GuestAlloc(size, 8);
    });

    RegisterThunk("inflateInit2_", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t guest_addr = jit.Regs()[0];
        const int window_bits = static_cast<int>(jit.Regs()[1]);
        const char* version = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2]));
        if (!GuestRangeValid(host.GetGuestMemorySize(), guest_addr, 56)) {
            jit.Regs()[0] = Z_STREAM_ERROR;
            return;
        }
        std::lock_guard<std::mutex> lock(g_zstream_mutex);
        auto it = g_zstreams.find(guest_addr);
        if (it != g_zstreams.end()) {
            if (it->second.initialized) ::inflateEnd(&it->second.host);
            g_zstreams.erase(it);
        }
        auto [entry, inserted] = g_zstreams.emplace(guest_addr, GuestZStream{});
        GuestZStream& state = entry->second;
        state.handle = guest_addr;
        const int result = ::inflateInit2_(&state.host, window_bits,
                                           version ? version : ZLIB_VERSION, sizeof(z_stream));
        if (result == Z_OK) {
            state.initialized = true;
            SyncHostZStreamToGuest(host, guest_addr, state.host, guest_addr);
        } else {
            LOGE("guest inflateInit2 failed stream=0x%08X result=%d", guest_addr, result);
            g_zstreams.erase(entry);
        }
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    RegisterThunk("inflate", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t guest_addr = jit.Regs()[0];
        const int flush = static_cast<int>(jit.Regs()[1]);
        std::lock_guard<std::mutex> lock(g_zstream_mutex);
        auto it = g_zstreams.find(guest_addr);
        if (it == g_zstreams.end() || !it->second.initialized ||
            !SyncGuestZStreamToHost(host, guest_addr, it->second.host)) {
            jit.Regs()[0] = Z_STREAM_ERROR;
            return;
        }
        GuestZStream& state = it->second;
        const int result = ::inflate(&state.host, flush);
        SyncHostZStreamToGuest(host, guest_addr, state.host, state.handle);
        if (result < 0) {
            LOGE("guest inflate failed stream=0x%08X result=%d message=%s",
                 guest_addr, result, state.host.msg ? state.host.msg : "<none>");
        }
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    RegisterThunk("inflateEnd", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const uint32_t guest_addr = jit.Regs()[0];
        std::lock_guard<std::mutex> lock(g_zstream_mutex);
        auto it = g_zstreams.find(guest_addr);
        if (it == g_zstreams.end()) {
            jit.Regs()[0] = Z_STREAM_ERROR;
            return;
        }
        const int result = it->second.initialized ? ::inflateEnd(&it->second.host) : Z_STREAM_ERROR;
        if (result != Z_OK) {
            LOGE("guest inflateEnd failed stream=0x%08X result=%d", guest_addr, result);
        }
        g_zstreams.erase(it);
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });

    RegisterThunk("metadata_store_watch", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t slot = jit.Regs()[0] + (jit.Regs()[4] << 2);
        const uint32_t value = jit.Regs()[5];
        host.MemoryWrite32(slot, value);
        static std::atomic<unsigned> count{0};
        const unsigned sample = count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (sample <= 12 || sample % 100000 == 0) {
            LOGI("metadata store: table=0x%08X index=0x%08X addr=0x%08X value=0x%08X readback=0x%08X count=%u",
                 jit.Regs()[0], jit.Regs()[4], slot, value, host.MemoryRead32(slot), sample);
        }
    });

    RegisterThunk("calloc", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const size_t num = jit.Regs()[0];
        const size_t size = jit.Regs()[1];
        static uint64_t count = 0;
        const uint64_t call = ++count;
        if (call <= 12 || (call % 100000) == 0) {
            LOGI("calloc #%llu num=%zu size=%zu PC=0x%08X LR=0x%08X R4=0x%08X R5=0x%08X R6=0x%08X",
                 static_cast<unsigned long long>(call), num, size,
                 jit.Regs()[15], jit.Regs()[14], jit.Regs()[4], jit.Regs()[5], jit.Regs()[6]);
        }
        uint32_t addr = host.GuestAlloc(num * size, 8);
        if (addr) {
            memset(GuestToHost(host.GetGuestMemoryBase(), addr), 0, num * size);
        }
        if (jit.Regs()[14] == 0x4428983C &&
            (call <= 12 || (call % 100000) == 0)) {
            constexpr uint32_t kMetadataCacheSlot = 0x44A6C4C8;
            constexpr uint32_t kMetadataCacheStoreSlot = 0x44A6C4CC;
            constexpr uint32_t kMetadataCacheCheckSlot = 0x44964B44;
            const uint32_t cache = host.MemoryRead32(kMetadataCacheSlot);
            const uint32_t store_cache = host.MemoryRead32(kMetadataCacheStoreSlot);
            const uint32_t check_cache = host.MemoryRead32(kMetadataCacheCheckSlot);
            const uint32_t index = jit.Regs()[4];
            const uint32_t cache_item = cache ? host.MemoryRead32(cache + index * 4) : 0;
            const uint32_t store_item = store_cache ? host.MemoryRead32(store_cache + index * 4) : 0;
            const uint32_t check_item = check_cache ? host.MemoryRead32(check_cache + index * 4) : 0;
            LOGI("metadata cache: index=0x%08X alloc=0x%08X cache=0x%08X item=0x%08X store_cache=0x%08X store_item=0x%08X check_cache=0x%08X check_item=0x%08X",
                 index, addr, cache, cache_item, store_cache, store_item, check_cache, check_item);
        }
        jit.Regs()[0] = addr;
    });

    RegisterThunk("realloc", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t old_addr = jit.Regs()[0];
        const size_t new_size = jit.Regs()[1];
        if (!old_addr) {
            jit.Regs()[0] = host.GuestAlloc(new_size, 8);
            return;
        }
        if (new_size == 0) {
            host.GuestFree(old_addr);
            jit.Regs()[0] = 0;
            return;
        }
        const uint32_t new_addr = host.GuestAlloc(new_size, 8);
        if (!new_addr) {
            jit.Regs()[0] = 0;
            return;
        }
        const uint32_t old_size = host.GuestAllocSize(old_addr);
        const size_t copy_size = old_size ? std::min<size_t>(old_size, new_size) : new_size;
        memcpy(GuestToHost(host.GetGuestMemoryBase(), new_addr),
               GuestToHost(host.GetGuestMemoryBase(), old_addr), copy_size);
        host.GuestFree(old_addr);
        jit.Regs()[0] = new_addr;
    });

    RegisterThunk("free", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        host.GuestFree(jit.Regs()[0]);
    });

    RegisterThunk("memalign", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        size_t alignment = jit.Regs()[0];
        size_t size = jit.Regs()[1];
        jit.Regs()[0] = host.GuestAlloc(size, alignment);
    });

    // Logging & Formatted output
    RegisterThunk("__android_log_print", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
#if CMR_ANDROID_LOG_PRINT
        int prio = static_cast<int>(jit.Regs()[0]);
        const char* tag = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        const char* fmt = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2]));
        std::string formatted = FormatGuestString(jit, host, fmt, 3);
        if (BRIDGE_GUEST_LOG_PRIORITY_OK(prio))
            __android_log_print(prio, tag ? tag : "Arm32Bridge_Guest", "[GUEST] %s", formatted.c_str());
#else
        (void)host;
#endif
        jit.Regs()[0] = 0;
    });

    RegisterThunk("printf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* fmt = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        std::string formatted = FormatGuestString(jit, host, fmt, 1);
        LOGI("[printf] %s", formatted.c_str());
        jit.Regs()[0] = formatted.length();
    });

    RegisterThunk("fprintf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* fmt = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        std::string formatted = FormatGuestString(jit, host, fmt, 2);
        LOGI("[fprintf] %s", formatted.c_str());
        jit.Regs()[0] = formatted.length();
    });

    RegisterThunk("sprintf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        char* str = reinterpret_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* fmt = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        std::string formatted = FormatGuestString(jit, host, fmt, 2);
        if (str) {
            strcpy(str, formatted.c_str());
            jit.Regs()[0] = formatted.length();
        } else {
            jit.Regs()[0] = 0;
        }
    });

    RegisterThunk("snprintf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        char* str = reinterpret_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        size_t size = jit.Regs()[1];
        const char* fmt = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2]));
        std::string formatted = FormatGuestString(jit, host, fmt, 3);
        if (str && size > 0) {
            strncpy(str, formatted.c_str(), size - 1);
            str[size - 1] = '\0';
            jit.Regs()[0] = formatted.length();
        } else {
            jit.Regs()[0] = 0;
        }
    });

    RegisterThunk("sscanf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("vfprintf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("vasprintf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("putc", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = jit.Regs()[0]; });
    RegisterThunk("fputc", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = jit.Regs()[0]; });

    // fputs(const char *s, FILE *stream). Lesson 2.16: on Android fd 1 and 2 go nowhere, and
    // that is exactly where a C++ runtime writes the reason it is giving up. A stream the
    // guest really opened is written for real; anything else goes to the log instead of being
    // swallowed. In Angry Birds Go this was the last default stub still firing on a cold
    // start, and the first run with a real thunk printed what had been thrown away for
    // weeks: "io::IOException".
    RegisterThunk("fputs", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* s = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        if (!s) { jit.Regs()[0] = static_cast<uint32_t>(-1); return; }
        auto it = g_file_map.find(jit.Regs()[1]);
        if (it != g_file_map.end()) {
            jit.Regs()[0] = static_cast<uint32_t>(fputs(s, it->second));
            return;
        }
        LOGI("[guest fputs] %s", s);
        jit.Regs()[0] = static_cast<uint32_t>(strlen(s));
    });

    RegisterThunk("btowc", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = static_cast<uint32_t>(btowc(static_cast<int>(jit.Regs()[0])));
    });
    RegisterThunk("wctob", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = static_cast<uint32_t>(wctob(static_cast<wint_t>(jit.Regs()[0])));
    });
    RegisterThunk("wctype", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* name = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const wctype_t type = name ? wctype(name) : 0;
        if (!type) { jit.Regs()[0] = 0; return; }
        const uint32_t handle = g_next_wctype_handle++;
        g_wctype_handles[handle] = type;
        jit.Regs()[0] = handle;
    });
    RegisterThunk("iswctype", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        auto it = g_wctype_handles.find(jit.Regs()[1]);
        jit.Regs()[0] = it != g_wctype_handles.end() ?
            static_cast<uint32_t>(iswctype(static_cast<wint_t>(jit.Regs()[0]), it->second)) : 0;
    });
    RegisterThunk("towlower", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = static_cast<uint32_t>(towlower(static_cast<wint_t>(jit.Regs()[0])));
    });
    RegisterThunk("towupper", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = static_cast<uint32_t>(towupper(static_cast<wint_t>(jit.Regs()[0])));
    });

    // Math functions
    RegisterThunk("asin", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        WriteDF(jit, 0, asin(ReadDF(jit, 0)));
    });
    RegisterThunk("atan", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        WriteDF(jit, 0, atan(ReadDF(jit, 0)));
    });
    RegisterThunk("ceil", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        WriteDF(jit, 0, ceil(ReadDF(jit, 0)));
    });
    RegisterThunk("exp2", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        WriteDF(jit, 0, exp2(ReadDF(jit, 0)));
    });
    RegisterThunk("fmod", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        WriteDF(jit, 0, fmod(ReadDF(jit, 0), ReadDF(jit, 1)));
    });
    RegisterThunk("modf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        double integral = 0.0;
        const double result = modf(ReadDF(jit, 0), &integral);
        const uint32_t output = jit.Regs()[2];
        if (output) {
            uint64_t raw;
            memcpy(&raw, &integral, sizeof(raw));
            host.MemoryWrite32(output, static_cast<uint32_t>(raw));
            host.MemoryWrite32(output + 4, static_cast<uint32_t>(raw >> 32));
        }
        WriteDF(jit, 0, result);
    });
    RegisterThunk("remainder", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        WriteDF(jit, 0, remainder(ReadDF(jit, 0), ReadDF(jit, 1)));
    });
    RegisterThunk("asinf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        float value = ReadSF(jit, 0);
        WriteSF(jit, 0, asinf(value));
    });
    RegisterThunk("atan2f", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, atan2f(ReadSF(jit, 0), ReadSF(jit, 1)));
    });
    RegisterThunk("ceilf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, ceilf(ReadSF(jit, 0)));
    });
    RegisterThunk("exp2f", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, exp2f(ReadSF(jit, 0)));
    });
    RegisterThunk("expf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, expf(ReadSF(jit, 0)));
    });
    RegisterThunk("fabsf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, fabsf(ReadSF(jit, 0)));
    });
    RegisterThunk("fmodf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, fmodf(ReadSF(jit, 0), ReadSF(jit, 1)));
    });
    RegisterThunk("floorf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, floorf(ReadSF(jit, 0)));
    });
    RegisterThunk("ldexpf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, ldexpf(ReadSF(jit, 0), static_cast<int>(jit.Regs()[1])));
    });
    RegisterThunk("logf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, logf(ReadSF(jit, 0)));
    });
    RegisterThunk("modff", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float integral = 0.0f;
        const float result = modff(ReadSF(jit, 0), &integral);
        if (jit.Regs()[1]) {
            uint32_t raw;
            memcpy(&raw, &integral, sizeof(raw));
            host.MemoryWrite32(jit.Regs()[1], raw);
        }
        WriteSF(jit, 0, result);
    });
    RegisterThunk("rintf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, rintf(ReadSF(jit, 0)));
    });
    RegisterThunk("tanf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, tanf(ReadSF(jit, 0)));
    });
    RegisterThunk("truncf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        WriteSF(jit, 0, truncf(ReadSF(jit, 0)));
    });
    RegisterThunk("sinf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float val = *reinterpret_cast<float*>(&jit.Regs()[0]);
        float res = sinf(val);
        jit.Regs()[0] = *reinterpret_cast<uint32_t*>(&res);
    });

    RegisterThunk("cosf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float val = *reinterpret_cast<float*>(&jit.Regs()[0]);
        float res = cosf(val);
        jit.Regs()[0] = *reinterpret_cast<uint32_t*>(&res);
    });

    RegisterThunk("sqrtf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float val = *reinterpret_cast<float*>(&jit.Regs()[0]);
        float res = sqrtf(val);
        jit.Regs()[0] = *reinterpret_cast<uint32_t*>(&res);
    });

    RegisterThunk("atanf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float val = *reinterpret_cast<float*>(&jit.Regs()[0]);
        float res = atanf(val);
        jit.Regs()[0] = *reinterpret_cast<uint32_t*>(&res);
    });

    RegisterThunk("acosf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float val = *reinterpret_cast<float*>(&jit.Regs()[0]);
        float res = acosf(val);
        jit.Regs()[0] = *reinterpret_cast<uint32_t*>(&res);
    });

    RegisterThunk("powf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float base = *reinterpret_cast<float*>(&jit.Regs()[0]);
        float exp = *reinterpret_cast<float*>(&jit.Regs()[1]);
        float res = powf(base, exp);
        jit.Regs()[0] = *reinterpret_cast<uint32_t*>(&res);
    });

    // String & Memory functions
    RegisterThunk("strcmp", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* s1 = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* s2 = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        jit.Regs()[0] = (s1 && s2) ? strcmp(s1, s2) : 0;
    });

    RegisterThunk("strncmp", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* s1 = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* s2 = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        size_t n = jit.Regs()[2];
        jit.Regs()[0] = (s1 && s2) ? strncmp(s1, s2, n) : 0;
    });

    RegisterThunk("strcasecmp", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* s1 = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* s2 = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        jit.Regs()[0] = (s1 && s2) ? strcasecmp(s1, s2) : 0;
    });

    RegisterThunk("strlen", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* s = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        jit.Regs()[0] = s ? static_cast<uint32_t>(strlen(s)) : 0;
    });

    RegisterThunk("strcpy", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        char* dest = reinterpret_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* src = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        if (dest && src) strcpy(dest, src);
    });

    RegisterThunk("strdup", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* src = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        if (!src) { jit.Regs()[0] = 0; return; }
        size_t len = strlen(src) + 1;
        uint32_t dest_addr = host.GuestAlloc(len);
        memcpy(GuestToHost(host.GetGuestMemoryBase(), dest_addr), src, len);
        jit.Regs()[0] = dest_addr;
    });

    RegisterThunk("strrchr", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* s = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        int c = static_cast<int>(jit.Regs()[1]);
        if (!s) { jit.Regs()[0] = 0; return; }
        const char* res = strrchr(s, c);
        jit.Regs()[0] = res ? static_cast<uint32_t>(reinterpret_cast<uintptr_t>(res) - reinterpret_cast<uintptr_t>(host.GetGuestMemoryBase())) : 0;
    });

    // strerror hands back storage the caller does not own and never frees, so a
    // fresh block per call is a block per failed open - and a guest that probes
    // for files that are not there does that constantly. One block per errno.
    RegisterThunk("strerror", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        static std::mutex m;
        static std::unordered_map<int, uint32_t> cache;
        int err_no = static_cast<int>(jit.Regs()[0]);
        std::lock_guard<std::mutex> lock(m);
        auto it = cache.find(err_no);
        if (it == cache.end()) {
            const char* err = strerror(err_no);
            size_t len = strlen(err) + 1;
            uint32_t dest = host.GuestAlloc(len);
            if (dest) memcpy(GuestToHost(host.GetGuestMemoryBase(), dest), err, len);
            it = cache.emplace(err_no, dest).first;
        }
        jit.Regs()[0] = it->second;
    });

    RegisterThunk("strtoul", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* nptr = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        int base = static_cast<int>(jit.Regs()[2]);
        jit.Regs()[0] = nptr ? static_cast<uint32_t>(strtoul(nptr, nullptr, base)) : 0;
    });

    RegisterThunk("atoi", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* str = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        jit.Regs()[0] = str ? atoi(str) : 0;
    });

    RegisterThunk("islower", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = islower(static_cast<int>(jit.Regs()[0]));
    });

    RegisterThunk("isxdigit", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = isxdigit(static_cast<int>(jit.Regs()[0]));
    });

    RegisterThunk("memset", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t destination = jit.Regs()[0];
        int c = static_cast<int>(jit.Regs()[1]);
        size_t n = jit.Regs()[2];
        GuestBulkSet(jit, host, "memset", destination, c, n);
    });

    RegisterThunk("memcpy", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t destination = jit.Regs()[0];
        const uint32_t source = jit.Regs()[1];
        size_t n = jit.Regs()[2];
        GuestBulkCopy(jit, host, "memcpy", destination, source, n, false);
    });

    RegisterThunk("memcmp", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t lhs = jit.Regs()[0];
        const uint32_t rhs = jit.Regs()[1];
        const size_t n = jit.Regs()[2];
        jit.Regs()[0] = static_cast<uint32_t>(GuestBulkCompare(jit, host, lhs, rhs, n));
    });

    RegisterThunk("memchr", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t source = jit.Regs()[0];
        int c = static_cast<int>(jit.Regs()[1]);
        size_t n = jit.Regs()[2];
        if (!host.IsGuestMemoryRangeAccessible(source, n)) {
            if (n != 0) LogRejectedGuestBulk(jit, "memchr", 0, source, n, true, false);
            jit.Regs()[0] = 0;
            return;
        }
        const void* s = host.GetGuestMemoryBase() + source;
        void* res = const_cast<void*>(memchr(s, c, n));
        jit.Regs()[0] = res ? HostToGuest(host.GetGuestMemoryBase(), res) : 0;
    });

    RegisterThunk("qsort", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {});

    // ARM AEABI Helpers
    RegisterThunk("__aeabi_idiv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int32_t num = static_cast<int32_t>(jit.Regs()[0]);
        int32_t den = static_cast<int32_t>(jit.Regs()[1]);
        jit.Regs()[0] = den ? (num / den) : 0;
    });

    RegisterThunk("__aeabi_uidiv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t num = jit.Regs()[0];
        uint32_t den = jit.Regs()[1];
        jit.Regs()[0] = den ? (num / den) : 0;
    });

    RegisterThunk("__aeabi_idivmod", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int32_t num = static_cast<int32_t>(jit.Regs()[0]);
        int32_t den = static_cast<int32_t>(jit.Regs()[1]);
        if (den != 0) {
            jit.Regs()[0] = num / den;
            jit.Regs()[1] = num % den;
        } else {
            jit.Regs()[0] = 0;
            jit.Regs()[1] = 0;
        }
    });

    RegisterThunk("__aeabi_uidivmod", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t num = jit.Regs()[0];
        uint32_t den = jit.Regs()[1];
        if (den != 0) {
            jit.Regs()[0] = num / den;
            jit.Regs()[1] = num % den;
        } else {
            jit.Regs()[0] = 0;
            jit.Regs()[1] = 0;
        }
    });

    RegisterThunk("__aeabi_uldivmod", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint64_t num = static_cast<uint64_t>(jit.Regs()[0]) | (static_cast<uint64_t>(jit.Regs()[1]) << 32);
        uint64_t den = static_cast<uint64_t>(jit.Regs()[2]) | (static_cast<uint64_t>(jit.Regs()[3]) << 32);
        if (den != 0) {
            uint64_t q = num / den;
            uint64_t r = num % den;
            jit.Regs()[0] = static_cast<uint32_t>(q & 0xFFFFFFFF);
            jit.Regs()[1] = static_cast<uint32_t>(q >> 32);
            jit.Regs()[2] = static_cast<uint32_t>(r & 0xFFFFFFFF);
            jit.Regs()[3] = static_cast<uint32_t>(r >> 32);
        }
    });

    RegisterThunk("__gnu_ldivmod_helper", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int64_t num = static_cast<int64_t>(jit.Regs()[0]) | (static_cast<int64_t>(jit.Regs()[1]) << 32);
        int64_t den = static_cast<int64_t>(jit.Regs()[2]) | (static_cast<int64_t>(jit.Regs()[3]) << 32);
        if (den != 0) {
            int64_t q = num / den;
            int64_t r = num % den;
            jit.Regs()[0] = static_cast<uint32_t>(q & 0xFFFFFFFF);
            jit.Regs()[1] = static_cast<uint32_t>(q >> 32);
            jit.Regs()[2] = static_cast<uint32_t>(r & 0xFFFFFFFF);
            jit.Regs()[3] = static_cast<uint32_t>(r >> 32);
        }
    });

    RegisterThunk("__aeabi_f2lz", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float f = *reinterpret_cast<float*>(&jit.Regs()[0]);
        int64_t i = static_cast<int64_t>(f);
        jit.Regs()[0] = static_cast<uint32_t>(i & 0xFFFFFFFF);
        jit.Regs()[1] = static_cast<uint32_t>(i >> 32);
    });

    RegisterThunk("__aeabi_f2ulz", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float f = *reinterpret_cast<float*>(&jit.Regs()[0]);
        uint64_t u = static_cast<uint64_t>(f);
        jit.Regs()[0] = static_cast<uint32_t>(u & 0xFFFFFFFF);
        jit.Regs()[1] = static_cast<uint32_t>(u >> 32);
    });

    RegisterThunk("__aeabi_l2f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int64_t i = static_cast<int64_t>(jit.Regs()[0]) | (static_cast<int64_t>(jit.Regs()[1]) << 32);
        float f = static_cast<float>(i);
        jit.Regs()[0] = *reinterpret_cast<uint32_t*>(&f);
    });

    RegisterThunk("__aeabi_ul2f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint64_t u = static_cast<uint64_t>(jit.Regs()[0]) | (static_cast<uint64_t>(jit.Regs()[1]) << 32);
        float f = static_cast<float>(u);
        jit.Regs()[0] = *reinterpret_cast<uint32_t*>(&f);
    });

    RegisterThunk("__aeabi_l2d", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int64_t i = static_cast<int64_t>(jit.Regs()[0]) | (static_cast<int64_t>(jit.Regs()[1]) << 32);
        double d = static_cast<double>(i);
        uint64_t u = *reinterpret_cast<uint64_t*>(&d);
        jit.Regs()[0] = static_cast<uint32_t>(u & 0xFFFFFFFF);
        jit.Regs()[1] = static_cast<uint32_t>(u >> 32);
    });

    // AEABI memory functions
    auto aeabi_memclr_handler = [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t dest = jit.Regs()[0];
        uint32_t n = jit.Regs()[1];
        GuestBulkSet(jit, host, "__aeabi_memclr", dest, 0, n);
    };
    RegisterThunk("__aeabi_memclr", aeabi_memclr_handler);
    RegisterThunk("__aeabi_memclr4", aeabi_memclr_handler);
    RegisterThunk("__aeabi_memclr8", aeabi_memclr_handler);

    auto aeabi_memset_handler = [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t dest = jit.Regs()[0];
        uint32_t n = jit.Regs()[1];
        int c = static_cast<int>(jit.Regs()[2]);
        GuestBulkSet(jit, host, "__aeabi_memset", dest, c, n);
    };
    RegisterThunk("__aeabi_memset", aeabi_memset_handler);
    RegisterThunk("__aeabi_memset4", aeabi_memset_handler);
    RegisterThunk("__aeabi_memset8", aeabi_memset_handler);

    auto aeabi_memcpy_handler = [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t dest = jit.Regs()[0];
        uint32_t src = jit.Regs()[1];
        uint32_t n = jit.Regs()[2];
        GuestBulkCopy(jit, host, "__aeabi_memcpy", dest, src, n, false);
    };
    RegisterThunk("__aeabi_memcpy", aeabi_memcpy_handler);
    RegisterThunk("__aeabi_memcpy4", aeabi_memcpy_handler);
    RegisterThunk("__aeabi_memcpy8", aeabi_memcpy_handler);

    auto aeabi_memmove_handler = [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t dest = jit.Regs()[0];
        uint32_t src = jit.Regs()[1];
        uint32_t n = jit.Regs()[2];
        GuestBulkCopy(jit, host, "__aeabi_memmove", dest, src, n, true);
    };
    RegisterThunk("__aeabi_memmove", aeabi_memmove_handler);
    RegisterThunk("__aeabi_memmove4", aeabi_memmove_handler);
    RegisterThunk("__aeabi_memmove8", aeabi_memmove_handler);

    // Pthreads
    const auto find_guest_mutex = [](uint32_t addr) {
        std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
        auto& state = g_guest_mutexes[addr];
        if (!state) state = std::make_shared<GuestMutexState>();
        return state;
    };
    const auto find_guest_cond = [](uint32_t addr) {
        std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
        auto& state = g_guest_conds[addr];
        if (!state) state = std::make_shared<GuestCondState>();
        return state;
    };
    RegisterThunk("pthread_mutex_lock", [find_guest_mutex](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t addr = jit.Regs()[0];
        if (!addr) { jit.Regs()[0] = EINVAL; return; }
        find_guest_mutex(addr)->lock(host);
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_mutex_unlock", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t addr = jit.Regs()[0];
        std::shared_ptr<GuestMutexState> state;
        {
            std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
            auto it = g_guest_mutexes.find(addr);
            if (it != g_guest_mutexes.end()) state = it->second;
        }
        if (state) state->unlock();
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_mutex_trylock", [find_guest_mutex](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const uint32_t addr = jit.Regs()[0];
        jit.Regs()[0] = (addr && find_guest_mutex(addr)->try_lock()) ? 0 : EBUSY;
    });
    RegisterThunk("pthread_cond_wait", [find_guest_mutex, find_guest_cond](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t cond_addr = jit.Regs()[0];
        const uint32_t mutex_addr = jit.Regs()[1];
        if (!cond_addr || !mutex_addr) { jit.Regs()[0] = EINVAL; return; }
        auto cond = find_guest_cond(cond_addr);
        auto mutex = find_guest_mutex(mutex_addr);
        WaitGuestCondition(*cond, *mutex, host, std::chrono::steady_clock::time_point::max());
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_cond_timedwait", [find_guest_mutex, find_guest_cond](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t cond_addr = jit.Regs()[0];
        const uint32_t mutex_addr = jit.Regs()[1];
        const uint32_t timespec_addr = jit.Regs()[2];
        if (!cond_addr || !mutex_addr || !timespec_addr) { jit.Regs()[0] = EINVAL; return; }
        const auto sec = static_cast<time_t>(static_cast<int32_t>(host.MemoryRead32(timespec_addr)));
        const auto nsec = static_cast<long>(host.MemoryRead32(timespec_addr + 4));
        auto cond = find_guest_cond(cond_addr);
        auto mutex = find_guest_mutex(mutex_addr);
        if (nsec < 0 || nsec >= 1000000000L) { jit.Regs()[0] = EINVAL; return; }
        const auto remaining = std::chrono::system_clock::time_point(std::chrono::seconds(sec)) +
                               std::chrono::nanoseconds(nsec) - std::chrono::system_clock::now();
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(remaining);
        const bool signalled = WaitGuestCondition(*cond, *mutex, host, deadline);
        jit.Regs()[0] = signalled ? 0 : ETIMEDOUT;
    });
    RegisterThunk("pthread_cond_timedwait_relative_np", [find_guest_mutex, find_guest_cond](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t cond_addr = jit.Regs()[0];
        const uint32_t mutex_addr = jit.Regs()[1];
        const uint32_t timespec_addr = jit.Regs()[2];
        if (!cond_addr || !mutex_addr ||
            !GuestRangeValid(host.GetGuestMemorySize(), timespec_addr, 8)) {
            jit.Regs()[0] = EINVAL;
            return;
        }
        const int64_t sec = static_cast<int32_t>(host.MemoryRead32(timespec_addr));
        const int64_t nsec = static_cast<int32_t>(host.MemoryRead32(timespec_addr + 4));
        if (sec < 0 || nsec < 0 || nsec >= 1000000000LL) {
            jit.Regs()[0] = EINVAL;
            return;
        }
        const int64_t duration_ns = sec * 1000000000LL + nsec;
        const bool signalled = WaitGuestCondition(
            *find_guest_cond(cond_addr), *find_guest_mutex(mutex_addr), host,
            std::chrono::steady_clock::now() + std::chrono::nanoseconds(duration_ns));
        jit.Regs()[0] = signalled ? 0 : ETIMEDOUT;
    });
    RegisterThunk("pthread_cond_broadcast", [find_guest_cond](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const uint32_t addr = jit.Regs()[0];
        if (addr) NotifyGuestCondition(*find_guest_cond(addr), true);
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_cond_init", [find_guest_cond](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t addr = jit.Regs()[0];
        if (addr) {
            find_guest_cond(addr);
            // ARM32 pthread_cond_t is one word, not the host's 48 bytes.
            host.MemoryWrite32(addr, 0);
        }
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_cond_destroy", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
        g_guest_conds.erase(jit.Regs()[0]);
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_cond_signal", [find_guest_cond](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const uint32_t addr = jit.Regs()[0];
        if (addr) NotifyGuestCondition(*find_guest_cond(addr), false);
        jit.Regs()[0] = 0;
    });
    RegisterThunk("pthread_mutexattr_destroy", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = 0;
    });

    RegisterThunk("pthread_once", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t control_addr = jit.Regs()[0];
        const uint32_t init_routine = jit.Regs()[1];
        if (!control_addr || !init_routine) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EINVAL);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        std::shared_ptr<GuestOnceState> state;
        {
            std::lock_guard<std::mutex> lock(g_guest_sync_mutex);
            auto& entry = g_guest_onces[control_addr];
            if (!entry) entry = std::make_shared<GuestOnceState>();
            state = entry;
        }
        const std::thread::id self = std::this_thread::get_id();
        std::unique_lock<std::mutex> lock(state->gate);
        while (state->running && state->owner != self) {
            state->condition.wait_for(lock, std::chrono::milliseconds(20));
            if (GuestSignalPending()) {
                lock.unlock();
                GuestSignalDeliverPending(host);
                lock.lock();
            }
        }
        if (state->done || host.MemoryRead32(control_addr) != 0) {
            state->done = true;
            lock.unlock();
            jit.Regs()[0] = 0;
            return;
        }
        if (state->running) {
            LOGE("pthread_once recursive initialization at 0x%08X", control_addr);
            lock.unlock();
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
            return;
        }
        state->running = true;
        state->owner = self;
        lock.unlock();

        host.CallGuestNested(init_routine, {});

        lock.lock();
        state->running = false;
        state->done = true;
        state->owner = {};
        lock.unlock();
        state->condition.notify_all();
        host.MemoryWrite32(control_addr, 1);
        jit.Regs()[0] = 0;
    });

    RegisterThunk("pthread_key_create", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t key_ptr = jit.Regs()[0];
        const uint32_t key = key_ptr ? g_next_tls_key++ : 0;
        if (key_ptr) host.MemoryWrite32(key_ptr, key);
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 64)
            LOGI("pthread_key_create out=0x%08X key=%u destructor=0x%08X lr=0x%08X",
                 key_ptr, key, jit.Regs()[1], jit.Regs()[14]);
        jit.Regs()[0] = 0;
    });

    RegisterThunk("pthread_setspecific", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t key = jit.Regs()[0];
        const uint32_t val = jit.Regs()[1];
        g_tls_map[key] = reinterpret_cast<void*>(static_cast<uintptr_t>(val));
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 128)
            LOGI("pthread_setspecific key=%u value=0x%08X lr=0x%08X", key, val, jit.Regs()[14]);
        jit.Regs()[0] = 0;
    });

    RegisterThunk("pthread_getspecific", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t key = jit.Regs()[0];
        auto it = g_tls_map.find(key);
        const uint32_t value = (it != g_tls_map.end())
            ? static_cast<uint32_t>(reinterpret_cast<uintptr_t>(it->second)) : 0;
        static std::atomic<unsigned> logged{0};
        const unsigned call = logged.fetch_add(1, std::memory_order_relaxed);
        if (call < 64)
            LOGI("pthread_getspecific key=%u -> 0x%08X lr=0x%08X", key, value, jit.Regs()[14]);
        jit.Regs()[0] = value;
    });

    // The expansion-file directory is empty on a split-APK install, but the
    // managed scanners still enumerate it to discover main/patch archives. The
    // file layer serves those names from the indexed split entries, so list the
    // logical OBB names here regardless of the real directory contents.
    const auto is_package_obb_dir = [](const char* p) {
        if (!p) return false;
        static const char kMarker[] = "/Android/obb/com.camouflaj.republique";
        const size_t marker_length = sizeof(kMarker) - 1;
        std::string trimmed = p;
        while (!trimmed.empty() && trimmed.back() == '/') trimmed.pop_back();
        if (trimmed.size() <= marker_length) return false;
        return trimmed.compare(trimmed.size() - marker_length, marker_length, kMarker) == 0;
    };
    const auto add_logical_obb_names = [](GuestDirState& state) {
        for (int i = 0; i < g_num_apk_entries; ++i) {
            std::string logical;
            const char* name = g_apk_entries[i].name;
            if (strncmp(name, "assets/main.", 12) == 0 || strncmp(name, "assets/patch.", 13) == 0) {
                logical = name + 7;
            } else if (strcmp(name, "assets/com.camouflaj.republique") == 0) {
                // République's CRI patch split stores the CPK under this opaque name.
                logical = "patch.949671.com.camouflaj.republique.obb";
            } else {
                continue;
            }
            if (std::find(state.names.begin(), state.names.end(), logical) == state.names.end()) {
                state.names.push_back(logical);
            }
        }
    };
    static_cast<void>(is_package_obb_dir);
    static_cast<void>(add_logical_obb_names);

    RegisterThunk("opendir", [is_package_obb_dir, add_logical_obb_names](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* path = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        if (!path) { jit.Regs()[0] = 0; return; }

        GuestDirState state;
        state.native = ::opendir(path);
        if (!state.native) {
            const std::string asset = ApkRelativeAssetName(path);
            if (asset.empty()) { jit.Regs()[0] = 0; return; }
            const std::string prefix = asset.back() == '/' ? asset : asset + '/';
            for (int i = 0; i < g_num_apk_entries; ++i) {
                const char* child = g_apk_entries[i].name;
                if (strncmp(child, prefix.c_str(), prefix.size()) != 0) continue;
                const char* name = child + prefix.size();
                const char* slash = strchr(name, '/');
                const size_t length = slash ? static_cast<size_t>(slash - name) : strlen(name);
                if (length == 0) continue;
                const std::string child_name(name, length);
                if (std::find(state.names.begin(), state.names.end(), child_name) == state.names.end()) {
                    state.names.push_back(child_name);
                }
            }
            if (state.names.empty()) { jit.Regs()[0] = 0; return; }
        } else if (is_package_obb_dir(path)) {
            add_logical_obb_names(state);
        }

        state.dirent = host.GuestAlloc(280, 4);
        if (!state.dirent) {
            if (state.native) ::closedir(state.native);
            jit.Regs()[0] = 0;
            return;
        }
        const uint32_t handle = g_next_dir_handle++;
        {
            std::lock_guard<std::mutex> lock(g_dir_mutex);
            g_guest_dirs.emplace(handle, std::move(state));
        }
        LOGI("opendir guest '%s' -> handle=0x%08X", path, handle);
        jit.Regs()[0] = handle;
    });

    RegisterThunk("readdir", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t handle = jit.Regs()[0];
        std::lock_guard<std::mutex> lock(g_dir_mutex);
        auto it = g_guest_dirs.find(handle);
        if (it == g_guest_dirs.end()) { jit.Regs()[0] = 0; return; }
        GuestDirState& state = it->second;
        std::string name;
        while (true) {
            bool from_overlay = false;
            if (state.native) {
                if (struct dirent* entry = ::readdir(state.native)) {
                    name = entry->d_name;
                    state.emitted.push_back(name);
                } else if (state.index < state.names.size()) {
                    // Native listing exhausted; synthesized overlay names follow.
                    name = state.names[state.index++];
                    from_overlay = true;
                } else {
                    name.clear();
                }
            } else if (state.index < state.names.size()) {
                name = state.names[state.index++];
                from_overlay = true;
            } else {
                name.clear();
            }
            if (name.empty()) break;
            // A synthesized name duplicating a real entry would enumerate the
            // same archive twice and confuse the managed scanners.
            if (from_overlay &&
                std::find(state.emitted.begin(), state.emitted.end(), name) != state.emitted.end()) {
                continue;
            }
            break;
        }
        if (name.empty()) { jit.Regs()[0] = 0; return; }
        uint8_t* record = static_cast<uint8_t*>(GuestToHost(host.GetGuestMemoryBase(), state.dirent));
        if (!record) { jit.Regs()[0] = 0; return; }
        memset(record, 0, 280);
        record[18] = 4; // DT_DIR/DT_REG is advisory; the name is what Unity consumes.
        strncpy(reinterpret_cast<char*>(record + 19), name.c_str(), 255);
        jit.Regs()[0] = state.dirent;
    });

    RegisterThunk("closedir", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t handle = jit.Regs()[0];
        GuestDirState state;
        {
            std::lock_guard<std::mutex> lock(g_dir_mutex);
            auto it = g_guest_dirs.find(handle);
            if (it == g_guest_dirs.end()) { jit.Regs()[0] = -1; return; }
            state = std::move(it->second);
            g_guest_dirs.erase(it);
        }
        if (state.native) ::closedir(state.native);
        host.GuestFree(state.dirent);
        jit.Regs()[0] = 0;
    });

    RegisterThunk("fopen", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* path = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* mode = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        if (!path) { jit.Regs()[0] = 0; return; }

        FILE* f = fopen(path, mode ? mode : "r");
        if (f && strstr(path, ".obb")) {
            struct stat st{};
            const int native_fd = fileno(f);
            LOGI("fopen guest external '%s' -> fd=%d size=%lld", path, native_fd,
                 native_fd >= 0 && fstat(native_fd, &st) == 0 ? static_cast<long long>(st.st_size) : -1LL);
        }
        int stream_vfd = -1;
        if (!f) {
            ApkEntry* entry = FindApkEntry(path);
            if (entry && entry->apk_fd >= 0) {
                const int dup_fd = dup(entry->apk_fd);
                if (dup_fd >= 0) {
                    ApkCookie* cookie = static_cast<ApkCookie*>(malloc(sizeof(ApkCookie)));
                    if (cookie) {
                        cookie->fd = dup_fd;
                        cookie->base_offset = entry->data_offset;
                        cookie->file_size = entry->file_size;
                        cookie->current_offset = 0;
                        f = funopen(cookie, ApkRead, nullptr, ApkSeek, ApkClose);
                        if (f) {
                            std::lock_guard<std::mutex> lock(g_vfd_mutex);
                            stream_vfd = g_next_vfd++;
                            g_vfd_map[stream_vfd] = {dup_fd, static_cast<size_t>(entry->file_size),
                                                     entry->data_offset, 0};
                            LOGI("fopen: APK direct stream opened '%s' -> '%s' (size %lld, offset %lld, vfd %d)",
                                 path, entry->name, static_cast<long long>(entry->file_size),
                                 static_cast<long long>(entry->data_offset), stream_vfd);
                        } else {
                            close(dup_fd);
                            free(cookie);
                        }
                    } else {
                        close(dup_fd);
                    }
                }
            }
        }

        if (f) {
            const int exposed_fd = stream_vfd >= 0 ? stream_vfd : fileno(f);
            const uint32_t handle = AllocateArm32File(host, f, exposed_fd, mode);
            if (!handle) {
                fclose(f);
                if (stream_vfd >= 0) {
                    std::lock_guard<std::mutex> lock(g_vfd_mutex);
                    g_vfd_map.erase(stream_vfd);
                }
                jit.Regs()[0] = 0;
                return;
            }
            g_file_map[handle] = f;
            if (stream_vfd >= 0) g_file_vfd_map[handle] = stream_vfd;
            if (strstr(path, ".obb") || strstr(path, "base.apk") || strstr(path, "Managed")) {
                LOGI("fopen guest tracked '%s' -> handle=0x%08X guest_fd=%d vfd=%d",
                     path, handle, exposed_fd, stream_vfd);
            }
            jit.Regs()[0] = handle;
        } else {
            jit.Regs()[0] = 0;
        }
    });

    RegisterThunk("fdopen", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int fd = static_cast<int>(jit.Regs()[0]);
        const char* mode = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        FILE* f = fdopen(fd, mode);
        if (f) {
            const uint32_t handle = AllocateArm32File(host, f, fd, mode);
            if (!handle) {
                fclose(f);
                jit.Regs()[0] = 0;
                return;
            }
            g_file_map[handle] = f;
            jit.Regs()[0] = handle;
        } else {
            jit.Regs()[0] = 0;
        }
    });

    RegisterThunk("fclose", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t handle = jit.Regs()[0];
        auto it = g_file_map.find(handle);
        if (it != g_file_map.end()) {
            fclose(it->second);
            g_file_map.erase(it);
            auto file_vfd = g_file_vfd_map.find(handle);
            if (file_vfd != g_file_vfd_map.end()) {
                std::lock_guard<std::mutex> lock(g_vfd_mutex);
                g_vfd_map.erase(file_vfd->second);
                g_file_vfd_map.erase(file_vfd);
            }
            host.GuestFree(handle);
            jit.Regs()[0] = 0;
        } else {
            jit.Regs()[0] = -1;
        }
    });

    RegisterThunk("fread", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        void* ptr = GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]);
        size_t size = jit.Regs()[1];
        size_t count = jit.Regs()[2];
        uint32_t handle = jit.Regs()[3];
        auto it = g_file_map.find(handle);
        size_t res = (it != g_file_map.end() && ptr) ? fread(ptr, size, count, it->second) : 0;
        if (it != g_file_map.end()) {
            SyncArm32FileStatus(host, handle, it->second);
            if (g_file_vfd_map.count(handle)) {
                static std::atomic<unsigned> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 96) {
                    LOGI("fread vfd-file handle=0x%08X size=%zu count=%zu -> %zu pos=%ld",
                         handle, size, count, res, it != g_file_map.end() ? ftell(it->second) : -1L);
                }
            }
        }
        jit.Regs()[0] = res;
    });

    RegisterThunk("fwrite", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const void* ptr = GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]);
        size_t size = jit.Regs()[1];
        size_t count = jit.Regs()[2];
        uint32_t handle = jit.Regs()[3];
        auto it = g_file_map.find(handle);
        const size_t result = (it != g_file_map.end() && ptr) ? fwrite(ptr, size, count, it->second) : 0;
        if (it != g_file_map.end()) SyncArm32FileStatus(host, handle, it->second);
        jit.Regs()[0] = result;
    });

    RegisterThunk("fseek", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t handle = jit.Regs()[0];
        long offset = static_cast<long>(static_cast<int32_t>(jit.Regs()[1]));
        int whence = static_cast<int>(jit.Regs()[2]);
        auto it = g_file_map.find(handle);
        int res = (it != g_file_map.end()) ? fseek(it->second, offset, whence) : -1;
        if (it != g_file_map.end()) {
            SyncArm32FileStatus(host, handle, it->second);
            if (g_file_vfd_map.count(handle)) {
                static std::atomic<unsigned> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 96) {
                    LOGI("fseek vfd-file handle=0x%08X off=%ld whence=%d -> %d pos=%ld",
                         handle, offset, whence, res, ftell(it->second));
                }
            }
        }
        jit.Regs()[0] = res;
    });

    RegisterThunk("fseeko", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t handle = jit.Regs()[0];
        off_t offset = static_cast<off_t>(static_cast<int32_t>(jit.Regs()[1]));
        int whence = static_cast<int>(jit.Regs()[2]);
        auto it = g_file_map.find(handle);
        int res = (it != g_file_map.end()) ? fseeko(it->second, offset, whence) : -1;
        if (it != g_file_map.end()) SyncArm32FileStatus(host, handle, it->second);
        jit.Regs()[0] = res;
    });

    RegisterThunk("ftell", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t handle = jit.Regs()[0];
        auto it = g_file_map.find(handle);
        jit.Regs()[0] = (it != g_file_map.end()) ? static_cast<uint32_t>(ftell(it->second)) : 0;
    });

    RegisterThunk("ftello", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t handle = jit.Regs()[0];
        auto it = g_file_map.find(handle);
        off_t res = (it != g_file_map.end()) ? ftello(it->second) : -1;
        jit.Regs()[0] = static_cast<uint32_t>(res);
    });

    RegisterThunk("fileno", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t handle = jit.Regs()[0];
        auto vfd = g_file_vfd_map.find(handle);
        if (vfd != g_file_vfd_map.end()) {
            jit.Regs()[0] = static_cast<uint32_t>(vfd->second);
            return;
        }
        auto it = g_file_map.find(handle);
        jit.Regs()[0] = (it != g_file_map.end()) ? fileno(it->second) : -1;
    });

    RegisterThunk("clearerr", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t handle = jit.Regs()[0];
        auto it = g_file_map.find(handle);
        if (it != g_file_map.end()) {
            clearerr(it->second);
            SyncArm32FileStatus(host, handle, it->second);
        }
    });

    RegisterThunk("ferror", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t handle = jit.Regs()[0];
        auto it = g_file_map.find(handle);
        if (it != g_file_map.end()) SyncArm32FileStatus(host, handle, it->second);
        jit.Regs()[0] = (it != g_file_map.end()) ? ferror(it->second) : 0;
    });

    RegisterThunk("open", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* path = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        int flags = static_cast<int>(jit.Regs()[1]);
        int mode = static_cast<int>(jit.Regs()[2]);

        int fd = -1;
        if (path && (flags & O_CREAT) == 0 && (flags & O_WRONLY) == 0 && (flags & O_RDWR) == 0) {
            ApkEntry* entry = FindApkEntry(path);
            if (entry && entry->apk_fd >= 0) {
                int dup_fd = dup(entry->apk_fd);
                if (dup_fd >= 0) {
                    std::lock_guard<std::mutex> lock(g_vfd_mutex);
                    int vfd = g_next_vfd++;
                    g_vfd_map[vfd] = { dup_fd, static_cast<size_t>(entry->file_size), entry->data_offset, 0 };
                    LOGI("open: APK direct stream opened '%s' -> '%s' (vfd %d, size %zu, offset %lld)",
                         path, entry->name, vfd, entry->file_size, static_cast<long long>(entry->data_offset));
                    jit.Regs()[0] = static_cast<uint32_t>(vfd);
                    return;
                }
            }
        }

        fd = open(path ? path : "", flags, mode);
        if (path && (strstr(path, ".obb") || strstr(path, "assets/"))) {
            struct stat st{};
            LOGI("open guest '%s' -> fd=%d size=%lld", path, fd,
                 fd >= 0 && fstat(fd, &st) == 0 ? static_cast<long long>(st.st_size) : -1LL);
        }
        jit.Regs()[0] = (fd >= 0) ? static_cast<uint32_t>(fd) : static_cast<uint32_t>(-1);
    });

    RegisterThunk("close", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int fd = static_cast<int>(jit.Regs()[0]);
        {
            std::lock_guard<std::mutex> lock(g_vfd_mutex);
            auto it = g_vfd_map.find(fd);
            if (it != g_vfd_map.end()) {
                close(it->second.real_apk_fd);
                g_vfd_map.erase(it);
                jit.Regs()[0] = 0;
                return;
            }
        }
        jit.Regs()[0] = (fd >= 0) ? static_cast<uint32_t>(close(fd)) : static_cast<uint32_t>(-1);
    });

    RegisterThunk("read", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int fd = static_cast<int>(jit.Regs()[0]);
        void* buf = GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]);
        size_t count = jit.Regs()[2];
        if (!buf) { jit.Regs()[0] = static_cast<uint32_t>(-1); return; }

        {
            std::lock_guard<std::mutex> lock(g_vfd_mutex);
            auto it = g_vfd_map.find(fd);
            if (it != g_vfd_map.end()) {
                VirtualFd& vfd = it->second;
                if (vfd.current_pos >= static_cast<off_t>(vfd.file_size)) {
                    jit.Regs()[0] = 0; // EOF
                    return;
                }
                size_t avail = vfd.file_size - vfd.current_pos;
                size_t to_read = (count < avail) ? count : avail;
                ssize_t nread = pread(vfd.real_apk_fd, buf, to_read, vfd.file_offset + vfd.current_pos);
                if (nread > 0) {
                    vfd.current_pos += nread;
                }
                jit.Regs()[0] = (nread >= 0) ? static_cast<uint32_t>(nread) : static_cast<uint32_t>(-1);
            }
        }
        // The vfd branch above handled the request; mirror the advanced position
        // onto the bound stdio stream (lock released) and skip the raw read.
        if (GuestVfdExists(fd)) {
            SyncFileFromVfd(fd);
            return;
        }

        ssize_t res = (fd >= 0) ? read(fd, buf, count) : -1;
        static unsigned real_read_log_count = 0;
        if (real_read_log_count++ < 80) {
            const uint8_t* bytes = static_cast<const uint8_t*>(buf);
            const off_t pos = fd >= 0 ? lseek(fd, 0, SEEK_CUR) : -1;
            LOGI("read fd=%d count=%zu -> %zd pos=%lld first=%02X%02X%02X%02X",
                 fd, count, res, static_cast<long long>(pos),
                 res > 0 ? bytes[0] : 0, res > 1 ? bytes[1] : 0,
                 res > 2 ? bytes[2] : 0, res > 3 ? bytes[3] : 0);
        }
        jit.Regs()[0] = static_cast<uint32_t>(res);
    });

    RegisterThunk("write", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int fd = static_cast<int>(jit.Regs()[0]);
        const void* buf = GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]);
        size_t count = jit.Regs()[2];
        ssize_t res = (buf && fd >= 0) ? write(fd, buf, count) : -1;
        jit.Regs()[0] = static_cast<uint32_t>(res);
    });

    RegisterThunk("lseek", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int fd = static_cast<int>(jit.Regs()[0]);
        off_t offset = static_cast<off_t>(static_cast<int32_t>(jit.Regs()[1]));
        int whence = static_cast<int>(jit.Regs()[2]);
        bool moved_virtual = false;

        {
            std::lock_guard<std::mutex> lock(g_vfd_mutex);
            auto it = g_vfd_map.find(fd);
            if (it != g_vfd_map.end()) {
                VirtualFd& vfd = it->second;
                off_t new_pos = vfd.current_pos;
                if (whence == SEEK_SET) new_pos = offset;
                else if (whence == SEEK_CUR) new_pos += offset;
                else if (whence == SEEK_END) new_pos = static_cast<off_t>(vfd.file_size) + offset;
                else { jit.Regs()[0] = static_cast<uint32_t>(-1); return; }

                if (new_pos < 0) { jit.Regs()[0] = static_cast<uint32_t>(-1); return; }
                vfd.current_pos = new_pos;
                jit.Regs()[0] = static_cast<uint32_t>(new_pos);
                moved_virtual = true;
            }
        }
        if (moved_virtual) {
            // Outside the vfd lock: this re-acquires it while mirroring the new
            // position onto the bound stdio stream.
            SyncFileFromVfd(fd);
            return;
        }

        off_t res = (fd >= 0) ? lseek(fd, offset, whence) : -1;
        jit.Regs()[0] = static_cast<uint32_t>(res);
    });

    RegisterThunk("lseek64", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int fd = static_cast<int>(jit.Regs()[0]);
        int64_t offset = static_cast<int64_t>(jit.Regs()[2]) | (static_cast<int64_t>(jit.Regs()[3]) << 32);
        int whence = static_cast<int>(host.MemoryRead32(jit.Regs()[13]));
        bool moved_virtual = false;

        {
            std::lock_guard<std::mutex> lock(g_vfd_mutex);
            auto it = g_vfd_map.find(fd);
            if (it != g_vfd_map.end()) {
                VirtualFd& vfd = it->second;
                off_t new_pos = vfd.current_pos;
                if (whence == SEEK_SET) new_pos = offset;
                else if (whence == SEEK_CUR) new_pos += offset;
                else if (whence == SEEK_END) new_pos = static_cast<off_t>(vfd.file_size) + offset;
                else { jit.Regs()[0] = static_cast<uint32_t>(-1); jit.Regs()[1] = static_cast<uint32_t>(-1); return; }

                if (new_pos < 0) { jit.Regs()[0] = static_cast<uint32_t>(-1); jit.Regs()[1] = static_cast<uint32_t>(-1); return; }
                vfd.current_pos = new_pos;
                jit.Regs()[0] = static_cast<uint32_t>(new_pos & 0xFFFFFFFF);
                jit.Regs()[1] = static_cast<uint32_t>(new_pos >> 32);
                moved_virtual = true;
            }
        }
        if (moved_virtual) {
            // Outside the vfd lock: this re-acquires it while mirroring the new
            // position onto the bound stdio stream.
            SyncFileFromVfd(fd);
            return;
        }

        off64_t res = (fd >= 0) ? lseek64(fd, offset, whence) : -1;
        jit.Regs()[0] = static_cast<uint32_t>(res & 0xFFFFFFFF);
        jit.Regs()[1] = static_cast<uint32_t>(res >> 32);
    });

    RegisterThunk("access", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* path = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        int mode = static_cast<int>(jit.Regs()[1]);
        int res = access(path ? path : "", mode);
        if (res != 0 && (FindApkEntry(path) || HasApkDirectory(path))) {
            res = 0;
        }
        jit.Regs()[0] = res;
    });

    const auto stat_handler = [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* path = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const uint32_t buf_gaddr = jit.Regs()[1];
        struct stat st{};
        int res = stat(path ? path : "", &st);
        const int stat_errno = errno;
        if (res != 0 && HasApkDirectory(path)) {
            memset(&st, 0, sizeof(st));
            st.st_mode = S_IFDIR | 0755;
            st.st_nlink = 1;
            res = 0;
        }
        if (res != 0) {
            ApkEntry* entry = FindApkEntry(path);
            if (entry) {
                memset(&st, 0, sizeof(st));
                st.st_size = entry->file_size;
                st.st_mode = S_IFREG | 0644;
                st.st_nlink = 1;
                res = 0;
            }
        }
        if (res == 0) WriteArm32Stat(host.GetGuestMemoryBase(), buf_gaddr, st);
        else host.MemoryWrite32(host.GuestErrnoAddress(), static_cast<uint32_t>(stat_errno));
        static unsigned log_count = 0;
        if (path && log_count++ < 120) {
            LOGI("stat guest '%s' -> res=%d mode=0x%o size=%lld apk=%s", path, res,
                 res == 0 ? static_cast<unsigned>(st.st_mode) : 0u,
                 res == 0 ? static_cast<long long>(st.st_size) : -1LL,
                 FindApkEntry(path) ? "yes" : HasApkDirectory(path) ? "dir" : "no");
        }
        jit.Regs()[0] = res;
    };
    RegisterThunk("stat", stat_handler);
    RegisterThunk("lstat", stat_handler);
    RegisterThunk("mkdir", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* path = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        jit.Regs()[0] = path ? static_cast<uint32_t>(::mkdir(path, static_cast<mode_t>(jit.Regs()[1]))) : static_cast<uint32_t>(-1);
    });
    RegisterThunk("rmdir", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* path = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        jit.Regs()[0] = path ? static_cast<uint32_t>(::rmdir(path)) : static_cast<uint32_t>(-1);
    });

    RegisterThunk("feof", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t handle = jit.Regs()[0];
        auto it = g_file_map.find(handle);
        if (it != g_file_map.end()) SyncArm32FileStatus(host, handle, it->second);
        jit.Regs()[0] = (it != g_file_map.end()) ? feof(it->second) : 1;
    });

    RegisterThunk("fflush", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t handle = jit.Regs()[0];
        auto it = g_file_map.find(handle);
        jit.Regs()[0] = (it != g_file_map.end()) ? fflush(it->second) : 0;
    });

    RegisterThunk("setvbuf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t handle = jit.Regs()[0];
        char* buffer = static_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        const int mode = static_cast<int>(jit.Regs()[2]);
        const size_t size = jit.Regs()[3];
        auto it = g_file_map.find(handle);
        jit.Regs()[0] = it != g_file_map.end() ? setvbuf(it->second, buffer, mode, size) : -1;
    });

    RegisterThunk("setbuf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t handle = jit.Regs()[0];
        char* buffer = static_cast<char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        auto it = g_file_map.find(handle);
        if (it != g_file_map.end()) setbuf(it->second, buffer);
        jit.Regs()[0] = 0;
    });

    RegisterThunk("rewind", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t handle = jit.Regs()[0];
        auto it = g_file_map.find(handle);
        if (it != g_file_map.end()) {
            rewind(it->second);
            SyncArm32FileStatus(host, handle, it->second);
        }
        jit.Regs()[0] = 0;
    });

    // mmap/munmap. Do NOT stub these as "hand out a fresh heap block": a caller that checks
    // the returned address (Boehm does, see lesson 2.15) will abort, and an engine that maps
    // its archive gets 345 MB of zeroes -- which reads as "the pak has no entries", not as an
    // error, and parks the boot on a download screen.
    //
    // The guest arena is one anonymous MAP_PRIVATE reservation, so a file mapping can be laid
    // straight into it with MAP_FIXED and the guest reads it at base + vaddr like any other
    // memory: no copy, no RSS. MAP_PRIVATE, never MAP_SHARED, so a stray guest write cannot
    // reach the file on disk.
    RegisterThunk("mmap", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t len   = jit.Regs()[1];
        const uint32_t flags = jit.Regs()[3];
        const uint32_t sp    = jit.Regs()[13];         // ARM32 mmap: fd at sp, 32-bit off_t at sp+4
        int            fd    = static_cast<int>(host.MemoryRead32(sp));
        off_t          off   = static_cast<int32_t>(host.MemoryRead32(sp + 4));
        static unsigned mmap_log_count = 0;
        if (mmap_log_count++ < 40 || fd >= 1000) {
            LOGI("mmap request len=%u prot=0x%08X flags=0x%08X fd=%d off=%lld",
                 len, jit.Regs()[2], flags, fd, static_cast<long long>(off));
        }

        constexpr uint32_t kMapAnonymous = 0x20;       // Linux MAP_ANONYMOUS on arm
        constexpr uint32_t kMapFailed    = 0xFFFFFFFF;
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));

        if (len == 0) { jit.Regs()[0] = kMapFailed; return; }

        if (fd < 0 || (flags & kMapAnonymous)) {
            // Anonymous: fresh mapping space, which the arena zero-fills for us.
            const uint32_t va = host.GuestAllocMapping((len + page - 1) & ~(page - 1), page);
            jit.Regs()[0] = va ? va : kMapFailed;
            return;
        }

        // A virtual fd is an entry inside the APK: map the real APK descriptor at the entry's
        // offset instead. This is what lets an expansion file ship inside the APK -- the
        // engine fopen()s the external path, the APK fallback answers, and the mmap that
        // follows lands on the same bytes without anything being unpacked.
        {
            std::lock_guard<std::mutex> lock(g_vfd_mutex);
            auto it = g_vfd_map.find(fd);
            if (it != g_vfd_map.end()) {
                off += it->second.file_offset;
                fd = it->second.real_apk_fd;
            }
        }
        if (fd < 0) { jit.Regs()[0] = kMapFailed; return; }

        // mmap needs a page-aligned file offset; keep the remainder as a bias on the result.
        const off_t  aligned = off & ~static_cast<off_t>(page - 1);
        const size_t bias    = static_cast<size_t>(off - aligned);
        const size_t span    = (len + bias + page - 1) & ~(page - 1);

        struct stat fst{};
        if (fstat(fd, &fst) == 0) {
            std::lock_guard<std::mutex> lock(g_mapping_mutex);
            for (const auto& m : g_guest_file_maps) {
                if (m.dev == fst.st_dev && m.ino == fst.st_ino && m.off == aligned && m.span == span) {
                    jit.Regs()[0] = m.va + static_cast<uint32_t>(bias);
                    return;
                }
            }
        }

        const uint32_t va = host.GuestAllocMapping(span, page);
        if (!va) { LOGE("mmap: no guest arena for %zu bytes", span); jit.Regs()[0] = kMapFailed; return; }
        if (static_cast<uint64_t>(va) + span > host.GetGuestMemorySize()) {
            LOGE("mmap: 0x%08X + %zu would run past the guest arena", va, span);
            jit.Regs()[0] = kMapFailed;
            return;
        }
        void* want = host.GetGuestMemoryBase() + va;
        void* got = ::mmap(want, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fd, aligned);
        if (got == MAP_FAILED) {
            LOGE("mmap(fd=%d, len=%u, off=%lld) failed: %s", fd, len,
                 static_cast<long long>(off), strerror(errno));
            jit.Regs()[0] = kMapFailed;
            return;
        }
        {
            std::lock_guard<std::mutex> lock(g_mapping_mutex);
            g_guest_file_maps.push_back({fst.st_dev, fst.st_ino, aligned, span, va});
        }
        LOGI("mmap: fd=%d len=%u off=%lld -> guest 0x%08X", fd, len,
             static_cast<long long>(off), va + static_cast<uint32_t>(bias));
        jit.Regs()[0] = va + static_cast<uint32_t>(bias);
    });

    RegisterThunk("munmap", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t addr = jit.Regs()[0];
        const uint32_t len  = jit.Regs()[1];
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        if (!addr || !len) { jit.Regs()[0] = 0; return; }

        const uint32_t base_va = addr & ~static_cast<uint32_t>(page - 1);
        const size_t span = (static_cast<uint64_t>(len) + (addr - base_va) + page - 1) & ~(page - 1);

        // A cached file mapping stays: another filesystem may still be reading through it,
        // and the arena address is never reused anyway.
        {
            std::lock_guard<std::mutex> lock(g_mapping_mutex);
            for (const auto& m : g_guest_file_maps) {
                if (base_va >= m.va && base_va < m.va + m.span) { jit.Regs()[0] = 0; return; }
            }
        }
        // MAP_FIXED writes to *host* address space, and both the address and the length come
        // straight from the guest -- an out-of-range pair would unmap whatever the host keeps
        // after the arena, and that surfaces half a second later as a SIGSEGV in an unrelated
        // free(). Bounds-check before touching anything.
        if (static_cast<uint64_t>(base_va) + span > host.GetGuestMemorySize()) {
            LOGE("munmap(0x%08X, %u) out of arena, ignored", addr, len);
            jit.Regs()[0] = 0;
            return;
        }
        // Put anonymous pages back rather than leaving a hole: the arena has to stay one
        // continuous mapping or the next access through base + vaddr faults.
        ::mmap(host.GetGuestMemoryBase() + base_va, span, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        jit.Regs()[0] = 0;
    });

    RegisterThunk("fstat", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        int fd = static_cast<int>(jit.Regs()[0]);
        uint32_t buf_gaddr = jit.Regs()[1];
        struct stat st;
        int res = -1;

        {
            std::lock_guard<std::mutex> lock(g_vfd_mutex);
            auto it = g_vfd_map.find(fd);
            if (it != g_vfd_map.end()) {
                memset(&st, 0, sizeof(st));
                st.st_size = it->second.file_size;
                st.st_mode = S_IFREG | 0644;
                st.st_nlink = 1;
                res = 0;
            }
        }

        if (res != 0) {
            res = fstat(fd, &st);
        }

        if (res == 0) {
            WriteArm32Stat(host.GetGuestMemoryBase(), buf_gaddr, st);
        }
        jit.Regs()[0] = res;
    });

    RegisterThunk("remove", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* path = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        jit.Regs()[0] = remove(path ? path : "");
    });

    RegisterThunk("rename", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* oldp = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* newp = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        jit.Regs()[0] = rename(oldp ? oldp : "", newp ? newp : "");
    });

    RegisterThunk("chmod", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("umask", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("mkstemp", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = -1; });
    RegisterThunk("getpid", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = getpid(); });

    RegisterThunk("time", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = static_cast<uint32_t>(time(nullptr));
    });

    // Same contract as strerror: the caller never frees it. Real localtime
    // overwrites one static buffer per call; this does the same, per thread, so
    // two threads formatting a timestamp cannot scribble on each other.
    RegisterThunk("localtime", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        static thread_local uint32_t gtm = 0;
        time_t t = static_cast<time_t>(host.MemoryRead32(jit.Regs()[0]));
        struct tm* tm_res = localtime(&t);
        if (!gtm) gtm = host.GuestAlloc(sizeof(struct tm));
        if (tm_res && gtm) {
            memcpy(GuestToHost(host.GetGuestMemoryBase(), gtm), tm_res, sizeof(struct tm));
            jit.Regs()[0] = gtm;
        } else {
            jit.Regs()[0] = 0;
        }
    });

    RegisterThunk("mktime", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        struct tm* tm_ptr = reinterpret_cast<struct tm*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        jit.Regs()[0] = tm_ptr ? static_cast<uint32_t>(mktime(tm_ptr)) : 0;
    });

    RegisterThunk("raise", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });

    // ARM32 signal actions are guest data, never native function pointers for ART.
    RegisterThunk("sigaction", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const int signal = static_cast<int>(jit.Regs()[0]);
        const uint32_t action_address = jit.Regs()[1], old_address = jit.Regs()[2];
        GuestSignalAction action{}, old{};
        if ((action_address && !GuestRangeValid(host.GetGuestMemorySize(), action_address, 16)) ||
            (old_address && !GuestRangeValid(host.GetGuestMemorySize(), old_address, 16))) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EFAULT);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        if (action_address) memcpy(&action, host.GetGuestMemoryBase() + action_address, 16);
        const int error = GuestSignalActionSet(signal, action_address ? &action : nullptr, old_address ? &old : nullptr);
        if (!error && old_address) memcpy(host.GetGuestMemoryBase() + old_address, &old, 16);
        if (error) host.MemoryWrite32(host.GuestErrnoAddress(), error);
        if (!error && action_address && (signal == 24 || signal == 30))
            LOGI("guest sigaction sig=%d handler=0x%08X flags=0x%08X mask=0x%08X", signal, action.handler, action.flags, action.mask);
        jit.Regs()[0] = error ? static_cast<uint32_t>(-1) : 0;
    });
    RegisterThunk("tgkill", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const int error = jit.Regs()[0] == static_cast<uint32_t>(getpid()) ?
            GuestSignalSend(static_cast<int>(jit.Regs()[1]), static_cast<int>(jit.Regs()[2])) : ESRCH;
        if (error) host.MemoryWrite32(host.GuestErrnoAddress(), error);
        jit.Regs()[0] = error ? static_cast<uint32_t>(-1) : 0;
    });
    RegisterThunk("sigsuspend", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        if (!GuestRangeValid(host.GetGuestMemorySize(), jit.Regs()[0], 4)) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EFAULT);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        const uint64_t previous = GuestSignalMask();
        GuestSignalSetMask(host.MemoryRead32(jit.Regs()[0]));
        GuestSignalWait(host);
        GuestSignalSetMask(previous);
        host.MemoryWrite32(host.GuestErrnoAddress(), EINTR);
        jit.Regs()[0] = static_cast<uint32_t>(-1);
    });
    const auto save_jump = [](Dynarmic::A32::Jit& jit, DynarmicHost& host, bool save_mask) {
        const uint32_t buffer = jit.Regs()[0];
        if (!GuestRangeValid(host.GetGuestMemorySize(), buffer, 256)) {
            LOGE("Invalid guest jmp_buf 0x%08X", buffer);
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
            return;
        }
        for (unsigned i = 0; i < 16; ++i) host.MemoryWrite32(buffer + i * 4, jit.Regs()[i]);
        host.MemoryWrite32(buffer + 64, (jit.Cpsr() & ~0x20u) | ((jit.Regs()[14] & 1) ? 0x20u : 0));
        host.MemoryWrite32(buffer + 68, jit.Fpscr());
        for (unsigned i = 0; i < 16; ++i) host.MemoryWrite32(buffer + 72 + i * 4, jit.ExtRegs()[16 + i]);
        host.MemoryWrite32(buffer + 136, save_mask);
        const uint64_t mask = GuestSignalMask();
        host.MemoryWrite32(buffer + 140, static_cast<uint32_t>(mask));
        host.MemoryWrite32(buffer + 144, static_cast<uint32_t>(mask >> 32));
        host.MemoryWrite32(buffer + 148, host.CurrentJitSlot());
        host.MemoryWrite32(buffer + 152, 0x4A4D5032);
        host.MemoryWrite32(buffer + 156, host.CurrentGuestThreadId());
        jit.Regs()[0] = 0;
    };
    RegisterThunk("setjmp", [save_jump](Dynarmic::A32::Jit& jit, DynarmicHost& host) { save_jump(jit, host, true); });
    RegisterThunk("_setjmp", [save_jump](Dynarmic::A32::Jit& jit, DynarmicHost& host) { save_jump(jit, host, false); });
    RegisterThunk("sigsetjmp", [save_jump](Dynarmic::A32::Jit& jit, DynarmicHost& host) { save_jump(jit, host, jit.Regs()[1] != 0); });
    const auto restore_jump = [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t buffer = jit.Regs()[0], value = jit.Regs()[1];
        if (!GuestRangeValid(host.GetGuestMemorySize(), buffer, 256) ||
            host.MemoryRead32(buffer + 152) != 0x4A4D5032 ||
            host.MemoryRead32(buffer + 148) != host.CurrentJitSlot() ||
            host.MemoryRead32(buffer + 156) != host.CurrentGuestThreadId()) {
            LOGE("Invalid or cross-context guest longjmp 0x%08X", buffer);
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
            return;
        }
        for (unsigned i = 0; i < 15; ++i) jit.Regs()[i] = host.MemoryRead32(buffer + i * 4);
        jit.SetCpsr(host.MemoryRead32(buffer + 64));
        jit.SetFpscr(host.MemoryRead32(buffer + 68));
        for (unsigned i = 0; i < 16; ++i) jit.ExtRegs()[16 + i] = host.MemoryRead32(buffer + 72 + i * 4);
        if (host.MemoryRead32(buffer + 136)) GuestSignalSetMask(static_cast<uint64_t>(host.MemoryRead32(buffer + 140)) |
            (static_cast<uint64_t>(host.MemoryRead32(buffer + 144)) << 32));
        jit.Regs()[0] = value ? value : 1;
        jit.Regs()[15] = jit.Regs()[14] & ~1u;
    };
    RegisterThunk("longjmp", restore_jump);
    RegisterThunk("_longjmp", restore_jump);
    RegisterThunk("siglongjmp", restore_jump);
    RegisterThunk("pthread_exit", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[15] = 4; // End this worker; never return into its noreturn caller.
        jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
    });

    // Runtime guards & hooks
    RegisterThunk("__stack_chk_fail", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        LOGE("Guest stack check failed at PC: 0x%08X", jit.Regs()[15]);
        jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
    });

    RegisterThunk("abort", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        LOGE("Guest called abort() at PC 0x%08X", jit.Regs()[15]);
        std::string stack;
        for (unsigned i = 0; i < 48; ++i) {
            const uint32_t value = host.MemoryRead32(jit.Regs()[13] + i * 4);
            if (value >= 0x40000000u && value < 0x48000000u) {
                char text[24];
                snprintf(text, sizeof(text), " [%u]=0x%08X", i, value);
                stack += text;
            }
        }
        if (!stack.empty()) LOGE("abort guest stack:%s", stack.c_str());
        jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
    });

    RegisterThunk("__assert2", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* file = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        int line = static_cast<int>(jit.Regs()[1]);
        const char* func = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2]));
        const char* expr = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3]));
        LOGE("Assertion failed: %s at %s:%d (%s)", expr ? expr : "", file ? file : "", line, func ? func : "");
        jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
    });

    RegisterThunk("__cxa_atexit", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("__cxa_finalize", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("__aeabi_unwind_cpp_pr0", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {});
    RegisterThunk("__aeabi_unwind_cpp_pr1", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {});
    RegisterThunk("_Unwind_Resume", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {});
    RegisterThunk("_Unwind_RaiseException", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("_Unwind_DeleteException", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {});
    RegisterThunk("_Unwind_GetLanguageSpecificData", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("_Unwind_GetRegionStart", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("_Unwind_VRS_Get", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("_Unwind_VRS_Set", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });
    RegisterThunk("__gnu_unwind_frame", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) { jit.Regs()[0] = 0; });

    RegisterThunk("__android_log_vprint", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
#if CMR_ANDROID_LOG_PRINT
        const int priority = static_cast<int>(jit.Regs()[0]);
        const char* tag = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        const char* fmt = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2]));
        const std::string message = FormatGuestVaList(host, fmt, jit.Regs()[3]);
        if (BRIDGE_GUEST_LOG_PRIORITY_OK(priority))
            __android_log_write(priority, tag ? tag : "Unity", message.c_str());
#else
        (void)host;
#endif
        jit.Regs()[0] = 0;
    });
    RegisterThunk("__google_potentially_blocking_region_begin", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = 0;
    });
    RegisterThunk("__google_potentially_blocking_region_end", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = 0;
    });

    // Imports the guest really exercises at startup/gameplay, promoted from the
    // unsupported stub table to faithful implementations.
        RegisterThunk("uname", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t buffer = jit.Regs()[0];
        utsname host_name{};
        if (::uname(&host_name) != 0 || !GuestRangeValid(host.GetGuestMemorySize(), buffer, 5 * 65)) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EFAULT);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        // Bionic utsname: five 65-byte fields. The guest is a 32-bit ARM
        // process; report the compatibility machine name it expects.
        const char* fields[5] = {host_name.sysname, host_name.nodename, host_name.release,
                                 host_name.version, "armv8l"};
        for (unsigned i = 0; i < 5; ++i) {
            snprintf(reinterpret_cast<char*>(host.GetGuestMemoryBase() + buffer + i * 65), 65,
                     "%s", fields[i] ? fields[i] : "");
        }
        jit.Regs()[0] = 0;
    });
        RegisterThunk("readlink", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* path = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const uint32_t buffer = jit.Regs()[1];
        const uint32_t size = jit.Regs()[2];
        if (!path || !buffer || size == 0 ||
            !GuestRangeValid(host.GetGuestMemorySize(), buffer, size)) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EFAULT);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        const ssize_t result = ::readlink(path, reinterpret_cast<char*>(host.GetGuestMemoryBase() + buffer), size);
        if (result < 0) host.MemoryWrite32(host.GuestErrnoAddress(), errno);
        jit.Regs()[0] = static_cast<uint32_t>(static_cast<int32_t>(result));
    });
        RegisterThunk("getuid", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = static_cast<uint32_t>(::getuid());
    });
        RegisterThunk("setlocale", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        // Never touch the host locale: this bridge formats floats for the guest
        // with host printf/snprintf, and a locale with a comma decimal point
        // silently corrupts every number crossing the printf thunks. The guest
        // runs its own static libc; report the neutral "C" locale.
        static std::mutex mutex;
        static uint32_t cached_c = 0;
        std::lock_guard<std::mutex> lock(mutex);
        if (!cached_c) {
            cached_c = host.GuestAlloc(2, 1);
            if (cached_c) memcpy(host.GetGuestMemoryBase() + cached_c, "C", 2);
        }
        jit.Regs()[0] = cached_c;
    });
        RegisterThunk("setenv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* name = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        const char* value = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        if (!name || !value || !*name || strchr(name, '=')) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EINVAL);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        const int result = ::setenv(name, value, jit.Regs()[2] != 0);
        if (result != 0) host.MemoryWrite32(host.GuestErrnoAddress(), errno);
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    RegisterThunk("unsetenv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* name = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        if (!name || !*name || strchr(name, '=')) {
            host.MemoryWrite32(host.GuestErrnoAddress(), EINVAL);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        const int result = ::unsetenv(name);
        if (result != 0) host.MemoryWrite32(host.GuestErrnoAddress(), errno);
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
        RegisterThunk("prctl", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const int option = static_cast<int>(jit.Regs()[0]);
        if (option == PR_SET_NAME) {
            const char* name = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
            char safe[16]{};
            if (name) {
                strncpy(safe, name, sizeof(safe) - 1);
                jit.Regs()[0] = static_cast<uint32_t>(::prctl(PR_SET_NAME, safe, 0, 0, 0));
#if defined(__ANDROID__)
                if (strcmp(safe, "UnityGfxDeviceW") == 0) {
                    bridge_perf::NoteGraphicsThread(gettid());
                    bridge_perf::TuneGraphicsThread();
                    const int priority_result = setpriority(PRIO_PROCESS, 0, -4);
                    LOGI("Gfx thread priority: nice=-4 result=%d errno=%d",
                         priority_result, errno);
                } else if (strcmp(safe, "Worker Thread") == 0) {
                    bridge_perf::NoteWorkerThread(gettid());
                    bridge_perf::TuneWorkerThread();
                } else if (strncmp(safe, "FMOD ", 5) == 0) {
                    bridge_perf::TuneAudioThread();
                }
#endif
                return;
            }
        }
        jit.Regs()[0] = 0;
    });
        RegisterThunk("syscall", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        // ARM32 EABI syscall numbers as passed to bionic syscall(). Only a
        // whitelist is translated; anything else fails closed with ENOSYS.
        constexpr uint32_t kArm32SysGettid = 224;
        constexpr uint32_t kArm32SysSchedSetaffinity = 241;
        constexpr uint32_t kArm32SysSchedGetaffinity = 242;
        const uint32_t number = jit.Regs()[0];
        const auto fail = [&](int error) {
            host.MemoryWrite32(host.GuestErrnoAddress(), error);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
        };
        switch (number) {
            case kArm32SysGettid:
                jit.Regs()[0] = static_cast<uint32_t>(gettid());
                return;
            case kArm32SysSchedSetaffinity:
            case kArm32SysSchedGetaffinity: {
                const pid_t pid = static_cast<pid_t>(static_cast<int32_t>(jit.Regs()[1]));
                const uint32_t size = jit.Regs()[2];
                const uint32_t mask_ptr = jit.Regs()[3];
                if (size == 0 || size > 128 ||
                    !GuestRangeValid(host.GetGuestMemorySize(), mask_ptr, size)) {
                    fail(EINVAL);
                    return;
                }
                if (number == kArm32SysSchedGetaffinity) {
                    uint8_t mask[128]{};
                    const long result = ::syscall(SYS_sched_getaffinity, pid, size, mask);
                    if (result < 0) {
                        fail(errno);
                        return;
                    }
                    memcpy(host.GetGuestMemoryBase() + mask_ptr, mask, size);
                    jit.Regs()[0] = static_cast<uint32_t>(result);
                    return;
                }

                uint8_t requested[128]{};
                memcpy(requested, host.GetGuestMemoryBase() + mask_ptr, size);

                // CPU indexes chosen by the ARM32 guest describe the source
                // device's topology, not this ARM64 host. Applying that mask
                // literally also makes newly-created host threads (including
                // AudioTrack callbacks) inherit the restriction. Validate the
                // target and mask like sched_setaffinity(), but leave placement
                // to Android's scheduler.
                uint8_t allowed[128]{};
                const long affinity_size = ::syscall(
                    SYS_sched_getaffinity, pid, sizeof(allowed), allowed);
                if (affinity_size < 0) {
                    fail(errno);
                    return;
                }
                bool intersects_allowed_cpu = false;
                const size_t comparable_size = std::min<size_t>(
                    size, static_cast<size_t>(affinity_size));
                for (size_t i = 0; i < comparable_size; ++i) {
                    if ((requested[i] & allowed[i]) != 0) {
                        intersects_allowed_cpu = true;
                        break;
                    }
                }
                if (!intersects_allowed_cpu) {
                    fail(EINVAL);
                    return;
                }

                static std::atomic<unsigned> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                    unsigned long long requested_low = 0;
                    unsigned long long allowed_low = 0;
                    memcpy(&requested_low, requested, std::min(sizeof(requested_low), sizeof(requested)));
                    memcpy(&allowed_low, allowed, std::min(sizeof(allowed_low), sizeof(allowed)));
                    LOGI("Ignoring guest sched_setaffinity pid=%d size=%u requested=0x%llX allowed=0x%llX",
                         pid, size, requested_low, allowed_low);
                }
                jit.Regs()[0] = 0;
                return;
            }
            default:
                LOGE("Unsupported guest syscall number=%u r1=0x%08X", number, jit.Regs()[1]);
                fail(ENOSYS);
                return;
        }
    });
        RegisterThunk("pthread_detach", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
        RegisterThunk("pthread_key_delete", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
        RegisterThunk("pthread_getschedparam", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        // Guest pthread_t values are bridge-assigned ids, not host handles; the
        // host scheduler view of the current worker thread is what callers want.
        int policy = 0;
        sched_param param{};
        const int result = ::pthread_getschedparam(::pthread_self(), &policy, &param);
        if (result == 0) {
            if (jit.Regs()[1]) host.MemoryWrite32(jit.Regs()[1], static_cast<uint32_t>(policy));
            if (jit.Regs()[2]) host.MemoryWrite32(jit.Regs()[2], static_cast<uint32_t>(param.sched_priority));
        }
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
        RegisterThunk("pthread_attr_setschedparam", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
        RegisterThunk("pthread_attr_setschedpolicy", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
        RegisterThunk("dladdr", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        // Guest addresses do not exist in the host link map; report
        // symbol-not-found, which callers handle like an unresolved symbol.
        jit.Regs()[0] = 0;
    });
    // Sensors: report a valid manager with no sensors. Unity then skips its
    // accelerometer subsystem entirely instead of walking a fake queue.
    constexpr uint32_t kFakeSensorManager = 0x5E150001;
    RegisterThunk("ASensorManager_getInstance", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = kFakeSensorManager;
    });
    RegisterThunk("ASensorManager_getDefaultSensor", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = 0;
    });
    RegisterThunk("ASensorManager_getSensorList", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        if (jit.Regs()[1]) host.MemoryWrite32(jit.Regs()[1], 0);
        jit.Regs()[0] = 0;
    });
    RegisterThunk("ASensorManager_createEventQueue", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = 0;
    });
    RegisterThunk("ASensorManager_destroyEventQueue", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = 0;
    });
    RegisterThunk("ASensorEventQueue_enableSensor", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
    RegisterThunk("ASensorEventQueue_disableSensor", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
    RegisterThunk("ASensorEventQueue_setEventRate", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
    RegisterThunk("ASensorEventQueue_hasEvents", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
    RegisterThunk("ASensorEventQueue_getEvents", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
    RegisterThunk("ASensor_getName", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
    RegisterThunk("ASensor_getVendor", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
    RegisterThunk("ASensor_getType", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
    RegisterThunk("ASensor_getResolution", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });
    RegisterThunk("ASensor_getMinDelay", [](Dynarmic::A32::Jit& jit, DynarmicHost&) { jit.Regs()[0] = 0; });

    // fscanf forwards to the host FILE* behind the ABI-shaped guest stream.
    // Guest output pointers cannot be handed to the host scanf directly, so
    // each conversion writes through a shadow slot copied back afterwards. The
    // format is rewritten because guest long is 32-bit while the host's is
    // 64-bit ('l' drops, 'll' stays). Supported: the integer/float/string
    // conversions Unity uses on /proc files; anything else stays a no-op.
    RegisterThunk("fscanf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const auto stream_it = g_file_map.find(jit.Regs()[0]);
        const char* fmt = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        if (stream_it == g_file_map.end() || !fmt) { jit.Regs()[0] = 0; return; }
        FILE* stream = stream_it->second;

        std::vector<uint32_t> args{jit.Regs()[2], jit.Regs()[3]};
        for (unsigned i = 0; i < 16; ++i) args.push_back(host.MemoryRead32(jit.Regs()[13] + i * 4));

        std::string host_fmt;
        struct Out { uint32_t guest; int kind; int size; };  // 0:i32 1:i64 2:float 3:double 4:string
        std::vector<Out> outputs;
        bool malformed = false;
        for (const char* p = fmt; *p; ++p) {
            if (*p != '%') { host_fmt.push_back(*p); continue; }
            host_fmt.push_back('%');
            ++p;
            if (*p == '%') { host_fmt.push_back('%'); continue; }
            bool suppressed = false;
            if (*p == '*') { suppressed = true; host_fmt.push_back(*p++); }
            int width = 0;
            while (isdigit(static_cast<unsigned char>(*p))) {
                width = width * 10 + (*p - '0');
                host_fmt.push_back(*p++);
            }
            if (*p == 'h') { host_fmt.push_back(*p++); if (*p == 'h') host_fmt.push_back(*p++); }
            bool has_l = false, has_ll = false;
            if (*p == 'l') {
                ++p;
                if (*p == 'l') { has_ll = true; ++p; } else has_l = true;
            } else if (*p == 'z' || *p == 'j' || *p == 't' || *p == 'L') {
                ++p;
            }
            const char conversion = *p;
            if (!conversion) { malformed = true; break; }
            const bool integer = conversion == 'd' || conversion == 'i' || conversion == 'u' ||
                                 conversion == 'o' || conversion == 'x' || conversion == 'X' ||
                                 conversion == 'n';
            const bool floating = conversion == 'f' || conversion == 'e' || conversion == 'E' ||
                                  conversion == 'g' || conversion == 'G' || conversion == 'a' ||
                                  conversion == 'A';
            if (integer && has_ll) { host_fmt.push_back('l'); host_fmt.push_back('l'); }
            else if (floating && has_l) host_fmt.push_back('l');
            host_fmt.push_back(conversion);
            if (suppressed) continue;
            if (integer) outputs.push_back({0, has_ll ? 1 : 0, 4});
            else if (floating) outputs.push_back({0, has_l ? 3 : 2, has_l ? 8 : 4});
            else if (conversion == 's') outputs.push_back({0, 4, 512});
            else { malformed = true; break; }
        }
        if (malformed || outputs.size() > 8) {
            LOGE("fscanf: unsupported format '%s'", fmt);
            jit.Regs()[0] = 0;
            return;
        }

        struct Shadow { long long integer; double real; std::vector<char> text; };
        std::vector<Shadow> shadow(outputs.size());
        std::vector<void*> pointers(outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i].guest = args[i];
            if (outputs[i].kind == 2 || outputs[i].kind == 3) pointers[i] = &shadow[i].real;
            else if (outputs[i].kind == 4) {
                shadow[i].text.assign(static_cast<size_t>(outputs[i].size) + 1, '\0');
                pointers[i] = shadow[i].text.data();
            } else pointers[i] = &shadow[i].integer;
        }

        int result = 0;
        {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
            switch (outputs.size()) {
                case 0: result = fscanf(stream, host_fmt.c_str()); break;
                case 1: result = fscanf(stream, host_fmt.c_str(), pointers[0]); break;
                case 2: result = fscanf(stream, host_fmt.c_str(), pointers[0], pointers[1]); break;
                case 3: result = fscanf(stream, host_fmt.c_str(), pointers[0], pointers[1], pointers[2]); break;
                case 4: result = fscanf(stream, host_fmt.c_str(), pointers[0], pointers[1], pointers[2], pointers[3]); break;
                default: result = fscanf(stream, host_fmt.c_str(), pointers[0], pointers[1], pointers[2], pointers[3],
                                         pointers[4], pointers[5], pointers[6], pointers[7]); break;
            }
#pragma clang diagnostic pop
        }
        if (result > 0) {
            for (size_t i = 0; i < outputs.size() && i < static_cast<size_t>(result); ++i) {
                const uint32_t out = outputs[i].guest;
                if (!out || !GuestRangeValid(host.GetGuestMemorySize(), out, outputs[i].size)) continue;
                if (outputs[i].kind == 1) {
                    host.MemoryWrite32(out, static_cast<uint32_t>(static_cast<uint64_t>(shadow[i].integer) & 0xFFFFFFFFu));
                    host.MemoryWrite32(out + 4, static_cast<uint32_t>(static_cast<uint64_t>(shadow[i].integer) >> 32));
                } else if (outputs[i].kind == 2) {
                    uint32_t bits = 0;
                    memcpy(&bits, &shadow[i].real, 4);
                    host.MemoryWrite32(out, bits);
                } else if (outputs[i].kind == 3) {
                    uint64_t bits = 0;
                    memcpy(&bits, &shadow[i].real, 8);
                    host.MemoryWrite32(out, static_cast<uint32_t>(bits & 0xFFFFFFFFu));
                    host.MemoryWrite32(out + 4, static_cast<uint32_t>(bits >> 32));
                } else if (outputs[i].kind == 4) {
                    memcpy(host.GetGuestMemoryBase() + out, shadow[i].text.data(),
                           strnlen(shadow[i].text.data(), static_cast<size_t>(outputs[i].size)) + 1);
                } else {
                    host.MemoryWrite32(out, static_cast<uint32_t>(shadow[i].integer));
                }
            }
        }
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });



    // Explicitly surfaced unsupported imports. They are not silent default thunks: the
    // first call identifies a real missing contract, while cold paths can safely observe zero.
    auto register_unsupported = [&](const char* name) {
        RegisterThunk(name, [name](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
            static std::mutex log_mutex;
            static std::unordered_map<std::string, unsigned> logged;
            std::lock_guard<std::mutex> lock(log_mutex);
            const unsigned limit = strcmp(name, "tgkill") == 0 || strcmp(name, "syscall") == 0 ||
                strcmp(name, "sigaction") == 0 ? 12 : 1;
            if (logged[name]++ < limit) {
                LOGE("Unsupported guest import: %s r0=0x%08X r1=0x%08X r2=0x%08X r3=0x%08X LR=0x%08X stack0=0x%08X stack1=0x%08X",
                     name, jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3], jit.Regs()[14],
                     host.MemoryRead32(jit.Regs()[13]), host.MemoryRead32(jit.Regs()[13] + 4));
            }
            jit.Regs()[0] = 0;
        });
    };
    register_unsupported("ALooper_forThread");
    register_unsupported("ALooper_prepare");
    register_unsupported("atol");
    register_unsupported("bsd_signal");
    register_unsupported("bsearch");
    register_unsupported("clock");
    register_unsupported("clock_nanosleep");
    register_unsupported("difftime");
    register_unsupported("dup2");
    register_unsupported("execl");
    register_unsupported("exit");
    register_unsupported("fgets");
    register_unsupported("flock");
    register_unsupported("fork");
    register_unsupported("freeaddrinfo");
    register_unsupported("ftruncate");
    register_unsupported("gai_strerror");
    register_unsupported("getaddrinfo");
    register_unsupported("getc");
    register_unsupported("getcwd");
    register_unsupported("gethostname");
    register_unsupported("getnameinfo");
    register_unsupported("getpeername");
    register_unsupported("getpwuid");
    register_unsupported("getsockname");
    register_unsupported("getsockopt");
    register_unsupported("getwc");
    register_unsupported("gmtime");
    register_unsupported("inet_aton");
    register_unsupported("inet_ntoa");
    register_unsupported("inet_ntop");
    register_unsupported("inet_pton");
    // CRIWare asks FIONREAD on expansion-file streams to size its next read; a
    // fake success with an untouched buffer reads as "zero bytes available" and
    // aborts the CPK bind. Answer it from the virtual fd bookkeeping, and fail
    // unknown requests on a virtual fd honestly instead of returning garbage.
        // CRIWare asks FIONREAD on expansion-file streams to size its next read; a
    // fake success with an untouched buffer reads as "zero bytes available" and
    // stalls the CPK bind in a data-wait retry loop. Answer it from the virtual
    // fd bookkeeping, and fail unknown requests on a virtual fd honestly.
    RegisterThunk("ioctl", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const int fd = static_cast<int>(jit.Regs()[0]);
        const uint32_t request = jit.Regs()[1];
        const uint32_t arg = jit.Regs()[2];
        constexpr uint32_t kFionread = 0x541B;
        bool is_virtual = false;
        {
            std::lock_guard<std::mutex> lock(g_vfd_mutex);
            is_virtual = g_vfd_map.find(fd) != g_vfd_map.end();
        }
        if (is_virtual) {
            if (request == kFionread && arg && GuestRangeValid(host.GetGuestMemorySize(), arg, 4)) {
                std::lock_guard<std::mutex> lock(g_vfd_mutex);
                auto it = g_vfd_map.find(fd);
                if (it != g_vfd_map.end()) {
                    const long long available = static_cast<long long>(it->second.file_size) -
                                               static_cast<long long>(it->second.current_pos);
                    host.MemoryWrite32(arg, static_cast<uint32_t>(available > 0 ? available : 0));
                }
                jit.Regs()[0] = 0;
                return;
            }
            host.MemoryWrite32(host.GuestErrnoAddress(), ENOTTY);
            jit.Regs()[0] = static_cast<uint32_t>(-1);
            return;
        }
        void* host_arg = GuestToHost(host.GetGuestMemoryBase(), arg);
        const int result = ::ioctl(fd, request, host_arg);
        if (result < 0) host.MemoryWrite32(host.GuestErrnoAddress(), errno);
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    register_unsupported("kill");
    register_unsupported("mbrtowc");
    register_unsupported("mprotect");
    register_unsupported("perror");
    register_unsupported("pipe");
    register_unsupported("poll");
    register_unsupported("pthread_setname_np");
    register_unsupported("ptrace");
    register_unsupported("putchar");
    register_unsupported("putwc");
    register_unsupported("realpath");
    register_unsupported("recvfrom");
    register_unsupported("sem_getvalue");
    register_unsupported("sendto");
    register_unsupported("shutdown");
    register_unsupported("sleep");
    register_unsupported("statfs");
    register_unsupported("strftime");
    register_unsupported("strxfrm");
    register_unsupported("truncate");
    register_unsupported("ungetc");
    register_unsupported("ungetwc");
    register_unsupported("vprintf");
    register_unsupported("vsprintf");
    register_unsupported("waitpid");
    register_unsupported("wcrtomb");
    register_unsupported("wcscoll");
    register_unsupported("wcsftime");
    register_unsupported("wcslen");
    register_unsupported("wcsxfrm");
    register_unsupported("wmemchr");
    register_unsupported("wmemcmp");
    register_unsupported("wmemcpy");
    register_unsupported("wmemmove");
    register_unsupported("wmemset");
    register_unsupported("writev");

    // Asset override: see InstallAssetOverrideHook below. R1 is the archive-relative
    // entry name; returning ip=0 makes the veneer return NULL, so the engine's own
    // "not in the archive" path takes over and opens the plain file instead.
    RegisterThunk("asset_override_archive_open", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* name = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        jit.Regs()[12] = HasAssetOverride(name) ? 0u : g_asset_override_real_fn;
    });
}

// ---------------------------------------------------------------------------
// Loading modified assets from <externalFilesDir>/...
//
// Engines of this era resolve a file through a chain: archive -> optional second
// archive -> plain fopen(rootDir + path). Making the first archive report "not
// found" for any file the user dropped into the override root turns that existing
// fallback into a full mod mechanism, with no archive-format code and no risk to
// the normal path. Point InstallAssetOverrideHook at the guest's archive-open PLT
// slot; everything below is engine-agnostic.
// ---------------------------------------------------------------------------

static std::string g_override_root;   // e.g. /sdcard/Android/data/<pkg>/files/
static bool g_override_enabled = false;

void SetAssetOverrideRoot(const std::string& root) {
    g_override_root = root;
    if (!g_override_root.empty() && g_override_root.back() != '/') g_override_root += '/';
    struct stat st;
    g_override_enabled = !g_override_root.empty() &&
                         stat(g_override_root.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    LOGI("Asset override root: '%s' (%s)", g_override_root.c_str(), g_override_enabled ? "active" : "inactive");
}

bool HasAssetOverride(const char* entry_name) {
    if (!g_override_enabled || !entry_name) return false;
    // Entry names carry the archive layout, e.g. "assets/data/textures/x.bin" or
    // "<expansion-prefix>/assets/data/textures/x.bin"; the override root mirrors
    // everything below "assets/". Adjust this split if the guest archives differ.
    const char* rel = strstr(entry_name, "assets/");
    if (!rel) return false;
    rel += 7;
    if (!*rel) return false;
    std::string path = g_override_root + rel;
    if (access(path.c_str(), R_OK) != 0) return false;
    LOGI("Asset override: '%s' -> '%s'", entry_name, path.c_str());
    return true;
}

void InstallAssetOverrideHook(DynarmicHost& host, uint32_t plt_slot, const char* real_symbol) {
    if (!g_override_enabled || !plt_slot || !real_symbol) return;

    static bool s_installed = false;
    if (s_installed) return;

    g_asset_override_real_fn = host.GetElfLoader().GetSymbolAddress(real_symbol);
    if (!g_asset_override_real_fn) {
        LOGE("InstallAssetOverrideHook: '%s' not found in the guest", real_symbol);
        return;
    }

    uint32_t hook = host.GetThunkManager().GetOrCreateThunkAddress("asset_override_archive_open");
    uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000) != 0xEF000000) {
        LOGE("InstallAssetOverrideHook: unexpected trampoline at 0x%08X", hook);
        return;
    }

    // ARM veneer: run the hook, then either tail-call the real archive-open (ip != 0)
    // or return NULL. BX reaches the whole address space and handles the Thumb bit,
    // so unlike a B-based trampoline there is no +-32 MB limit.
    uint32_t veneer = host.GuestAlloc(32, 4);
    host.MemoryWrite32(veneer + 0,  svc);         // svc  #id
    host.MemoryWrite32(veneer + 4,  0xE35C0000);  // cmp  ip, #0
    host.MemoryWrite32(veneer + 8,  0x112FFF1C);  // bxne ip
    host.MemoryWrite32(veneer + 12, 0xE3A00000);  // mov  r0, #0
    host.MemoryWrite32(veneer + 16, 0xE12FFF1E);  // bx   lr
    host.InvalidateGuestCode(veneer, 32);

    host.MemoryWrite32(plt_slot + 0, 0xE51FF004);  // ldr pc, [pc, #-4]
    host.MemoryWrite32(plt_slot + 4, veneer);
    host.InvalidateGuestCode(plt_slot, 12);
    // The veneer is executable guest code living in the heap, so it must be
    // exempt from runaway-PC containment.
    host.AddExecutableRange(veneer, 32);

    s_installed = true;
    LOGI("Installed asset-override hook on '%s' (plt 0x%08X, veneer 0x%08X)",
         real_symbol, plt_slot, veneer);
}
