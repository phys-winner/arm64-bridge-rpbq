#include <jni.h>
#include <android/log.h>
#include <android/native_window_jni.h>
#include <string>
#include <vector>
#include <memory>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <cinttypes>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <zlib.h>
#include <lz4.h>

#include "dynarmic_host.h"
#include "elf_loader.h"
#include "thunk_manager.h"
#include "thunk_types.h"
#include "jni_fake_env.h"
#include "thunks_opensles.h"
#include "guest_signals.h"
#include "guest_hotpaths.h"
#include "performance_hints.h"
#include <future>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <signal.h>

#define LOG_TAG "Arm32Bridge_JNI"
#include "bridge_log.h"

namespace cfg {
constexpr const char* kGuestUnity = "libunity.so";
constexpr const char* kGuestIl2cpp = "libil2cpp.so";
constexpr const char* kGuestCri = "libcri_ware_unity.so";
constexpr const char* kGuestMain = "libmain.so";
constexpr const char* kGuestOvr = "libovrplatformloader.so";

constexpr const char* kNativeLoaderClass = "com/unity3d/player/NativeLoader";
constexpr const char* kUnityPlayerClass = "com/unity3d/player/UnityPlayer";
constexpr const char* kReflectionClass = "com/unity3d/player/ReflectionHelper";
constexpr const char* kWebRequestClass = "com/unity3d/player/UnityWebRequest";
constexpr const char* kVrClass = "com/unity3d/player/GoogleVrProxy";
constexpr const char* kArClass = "com/unity3d/player/GoogleARCoreApi";
constexpr const char* kArProxyClass = "com/unity3d/player/GoogleARProxy";
constexpr const char* kFmodClass = "org/fmod/FMODAudioDevice";
constexpr const char* kJniBridgeClass = "bitter/jnibridge/JNIBridge";

// RVAs come from the checksum-pinned libunity.so in the Phase 4/5 canonical APK.
constexpr uint32_t kUnityJniOnLoad = 0x2e47e8;
constexpr uint32_t kUnityJniOnUnload = 0x2e49a4;
constexpr uint32_t kCriJniOnLoad = 0x106ed | 1u;
constexpr uint32_t kOvrJniOnLoad = 0x6c49d | 1u;
constexpr uint32_t kInitJni = 0x2e3de4;
constexpr uint32_t kNativeDone = 0x2e3e50;
constexpr uint32_t kNativePause = 0x2e3e90;
constexpr uint32_t kNativeRecreateGfxState = 0x2e4014;
constexpr uint32_t kNativeRender = 0x2e4070;
constexpr uint32_t kNativeResume = 0x2e3ee4;
constexpr uint32_t kNativeLowMemory = 0x2e3f48;
constexpr uint32_t kNativeFocusChanged = 0x2e3f8c;
constexpr uint32_t kNativeSetInputString = 0x2e4248;
constexpr uint32_t kNativeSoftInputClosed = 0x2e450c;
constexpr uint32_t kNativeSoftInputCanceled = 0x2e443c;
constexpr uint32_t kNativeSoftInputLostFocus = 0x2e44a4;
constexpr uint32_t kNativeInitWebRequest = 0x2e4598;
constexpr uint32_t kNativeInjectEvent = 0x2e40c0;
constexpr uint32_t kNativeUnitySendMessage = 0x2e45e4;
constexpr uint32_t kNativeIsAutorotationOn = 0x2e46e8;
constexpr uint32_t kNativeMuteMasterAudio = 0x2e4744;
constexpr uint32_t kNativeRestartActivityIndicator = 0x2e47a8;

constexpr uint32_t kReflectionInvoke = 0x107b80;
constexpr uint32_t kReflectionFinalize = 0x107c4c;
constexpr uint32_t kJniBridgeInvoke = 0xa7fd50;
constexpr uint32_t kVrInit = 0x2dcb04;
constexpr uint32_t kVrSetVideoTransform = 0x2dc880;
constexpr uint32_t kVrIsQuitting = 0x2dcb64;
constexpr uint32_t kArInitialize = 0x2e1e1c;
constexpr uint32_t kArPause = 0x2e1e90;
constexpr uint32_t kArResume = 0x2e1ee4;
// libunity's original JNINativeMethod table at RVA 0x10da344.
constexpr uint32_t kFmodGetInfo = 0xb151a4;
constexpr uint32_t kFmodProcess = 0xb15288;
constexpr uint32_t kFmodProcessMicData = 0xb15314;

constexpr uint32_t kUnityBase = 0x40400000;
constexpr uint32_t kIl2cppBase = 0x42000000;
constexpr uint32_t kCriBase = 0x45000000;
constexpr uint32_t kMainBase = 0x40000000;
constexpr uint32_t kOvrBase = 0x45400000;
}  // namespace cfg

std::unique_ptr<DynarmicHost> g_host;
JavaVM* g_jvm = nullptr;
std::string g_saved_apk_path;
std::string g_saved_files_dir;
std::vector<std::string> g_apk_paths;
bool g_registered_natives = false;
static std::atomic<uint32_t> g_android_jni_invoke_method{0};
static std::atomic<bool> g_android_jni_helper_initializing{false};
static std::atomic<bool> g_android_jni_helper_ready{false};
static std::atomic<bool> g_android_jni_helper_wait_cancelled{false};
static std::mutex g_android_jni_helper_ready_mutex;
static std::condition_variable g_android_jni_helper_ready_cv;
static thread_local unsigned g_android_jni_helper_init_depth = 0;

namespace {
constexpr size_t kActivatorDiagRingSize = 8;
constexpr size_t kActivatorDiagStackWords = 16;
constexpr size_t kActivatorDiagTypeWords = 8;
constexpr size_t kActivatorDiagTextSize = 128;

struct ActivatorDiagSnapshot {
    unsigned call = 0;
    uint32_t raw[4] = {};
    uint32_t type_arg = 0;
    uint32_t non_public = 0;
    uint32_t pre_lr = 0;
    uint32_t frame_sp = 0;
    uint32_t saved_lr = 0;
    uint32_t cpsr = 0;
    uint32_t host_tid = 0;
    uint32_t guest_tid = 0;
    bool stack_valid = false;
    std::array<uint32_t, kActivatorDiagStackWords> stack = {};
    bool type_valid = false;
    std::array<uint32_t, kActivatorDiagTypeWords> type_words = {};
    uint32_t type_object_klass = 0;
    uint32_t type_handle = 0;
    uint32_t type_data = 0;
    uint32_t type_bits = 0;
    uint32_t target_klass = 0;
    bool target_name_valid = false;
    bool target_namespace_valid = false;
    char target_name[kActivatorDiagTextSize] = {};
    char target_namespace[kActivatorDiagTextSize] = {};
};

struct ActivatorDiagRing {
    std::array<ActivatorDiagSnapshot, kActivatorDiagRingSize> entries = {};
    size_t next = 0;
    size_t count = 0;

    void Push(const ActivatorDiagSnapshot& snapshot) {
        entries[next] = snapshot;
        next = (next + 1) % entries.size();
        if (count < entries.size()) ++count;
    }

    const ActivatorDiagSnapshot* Find(uint32_t frame_sp) const {
        if (!frame_sp) return nullptr;
        for (size_t i = 0; i < count; ++i) {
            if (entries[i].frame_sp == frame_sp) return &entries[i];
        }
        return nullptr;
    }
};

thread_local ActivatorDiagRing g_activator_diag_ring;

struct ActivatorValueTypeDiagSnapshot {
    uint32_t call = 0;
    uint32_t type_object = 0;
    uint32_t object_klass = 0;
    uint32_t type_handle = 0;
    uint32_t type_data = 0;
    uint32_t type_bits = 0;
    uint32_t target = 0;
    uint32_t method_info = 0;
};

thread_local ActivatorValueTypeDiagSnapshot g_activator_value_type_diag;

struct ActivatorIsPublicDiagSnapshot {
    unsigned call = 0;
    uint32_t object = 0;
    uint32_t klass = 0;
    uint32_t mhandle = 0;
    uint32_t reflected_type = 0;
    uint32_t type_handle = 0;
    uint32_t type_data = 0;
};

thread_local ActivatorIsPublicDiagSnapshot g_activator_is_public_diag;

static bool ReadActivatorWord(DynarmicHost& host, uint32_t base, size_t offset,
                              uint32_t& value) {
    const uint64_t address = static_cast<uint64_t>(base) + offset;
    if (address > 0xFFFFFFFFULL) return false;
    const uint32_t guest_address = static_cast<uint32_t>(address);
    if (guest_address < 0x1000 ||
        !GuestRangeValid(host.GetGuestMemorySize(), guest_address, sizeof(uint32_t))) {
        return false;
    }
    value = host.MemoryRead32(guest_address);
    return true;
}

static bool CopyActivatorText(DynarmicHost& host, uint32_t address, char* out,
                              size_t capacity) {
    if (!out || capacity < 2) return false;
    out[0] = '\0';
    if (address < 0x1000 || !GuestRangeValid(host.GetGuestMemorySize(), address, 1)) {
        return false;
    }
    const size_t available = host.GetGuestMemorySize() - static_cast<size_t>(address);
    const size_t probe = std::min(capacity, available);
    const char* text = reinterpret_cast<const char*>(host.GetGuestMemoryBase() + address);
    const size_t length = strnlen(text, probe);
    if (length >= probe || length >= capacity) return false;
    std::memcpy(out, text, length);
    out[length] = '\0';
    return true;
}

static const char* ActivatorCallerLabel(uint32_t raw_lr) {
    switch (raw_lr & ~1u) {
    case cfg::kIl2cppBase + 0x1633A0:
        return "System.Net.WebRequest.AddPrefix";
    case cfg::kIl2cppBase + 0x1929F40:
        return "System.Xml.Serialization.XmlReflectionImporter.CreateTypeMapping";
    case cfg::kIl2cppBase + 0x193CEB4:
        return "System.Xml.Serialization.XmlSerializationReader.ReadXmlSerializableElement";
    case cfg::kIl2cppBase + 0x193D218:
        return "System.Xml.Serialization.XmlSerializationReader.ReadClassInstance";
    case cfg::kIl2cppBase + 0x193DD28:
        return "System.Xml.Serialization.XmlSerializationReader.AddListValue";
    case cfg::kIl2cppBase + 0x193E660:
        return "System.Xml.Serialization.XmlSerializationReader.ReadObjectElement";
    default:
        return "unknown/tail-or-outer";
    }
}

static bool CopyActivatorManagedText(DynarmicHost& host, uint32_t address, char* out,
                                      size_t capacity) {
    if (!out || capacity < 2) return false;
    out[0] = '\0';
    if (address < 0x1000 || !GuestRangeValid(host.GetGuestMemorySize(), address, 12)) {
        return false;
    }
    const uint32_t length = std::min(host.MemoryRead32(address + 8),
                                     static_cast<uint32_t>(capacity - 1));
    if (static_cast<uint64_t>(address) + 12 + static_cast<uint64_t>(length) * 2 >
        host.GetGuestMemorySize()) {
        return false;
    }
    for (uint32_t i = 0; i < length; ++i) {
        out[i] = static_cast<char>(host.MemoryRead16(address + 12 + i * 2) & 0xFF);
    }
    out[length] = '\0';
    return true;
}

static void DecodeActivatorType(DynarmicHost& host, ActivatorDiagSnapshot& snapshot) {
    if (!snapshot.type_valid) return;
    snapshot.type_object_klass = snapshot.type_words[0];
    snapshot.type_handle = snapshot.type_words[2];
    if (!ReadActivatorWord(host, snapshot.type_handle, 0, snapshot.type_data) ||
        !ReadActivatorWord(host, snapshot.type_handle, 4, snapshot.type_bits)) {
        snapshot.type_data = 0;
        snapshot.type_bits = 0;
    }

    const auto try_klass = [&host, &snapshot](uint32_t klass) {
        uint32_t name = 0;
        uint32_t name_space = 0;
        if (!klass || !ReadActivatorWord(host, klass, 8, name) ||
            !ReadActivatorWord(host, klass, 12, name_space)) {
            return false;
        }
        char candidate_name[kActivatorDiagTextSize] = {};
        if (!CopyActivatorText(host, name, candidate_name, sizeof(candidate_name))) {
            return false;
        }
        char candidate_namespace[kActivatorDiagTextSize] = {};
        const bool namespace_valid =
            CopyActivatorText(host, name_space, candidate_namespace, sizeof(candidate_namespace));
        snapshot.target_klass = klass;
        snapshot.target_name_valid = true;
        snapshot.target_namespace_valid = namespace_valid;
        std::memcpy(snapshot.target_name, candidate_name, sizeof(snapshot.target_name));
        if (namespace_valid) {
            std::memcpy(snapshot.target_namespace, candidate_namespace,
                        sizeof(snapshot.target_namespace));
        }
        return true;
    };

    if (try_klass(snapshot.type_data)) return;
    (void)try_klass(snapshot.type_handle);
}

static void LogActivatorSnapshot(DynarmicHost&, const ActivatorDiagSnapshot& snapshot,
                                 bool failure) {
    const int priority = failure ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO;
    __android_log_print(
        priority, LOG_TAG,
        "activator diag %s call=%u static_this=0x%08X type_arg=0x%08X nonPublic=0x%08X methodInfo=0x%08X pre_lr=0x%08X lr_norm=0x%08X caller=%s frame_sp=0x%08X saved_lr=0x%08X cpsr=0x%08X host_tid=%u guest_tid=0x%08X stack_valid=%d type_valid=%d object_klass=0x%08X type_handle=0x%08X type_data=0x%08X type_bits=0x%08X target_klass=0x%08X type_name=%s type_namespace=%s",
        failure ? "matched-raise" : "entry", snapshot.call, snapshot.raw[0],
        snapshot.type_arg, snapshot.non_public, snapshot.raw[3], snapshot.pre_lr,
        snapshot.pre_lr & ~1u, ActivatorCallerLabel(snapshot.pre_lr), snapshot.frame_sp,
        snapshot.saved_lr, snapshot.cpsr, snapshot.host_tid, snapshot.guest_tid,
        snapshot.stack_valid, snapshot.type_valid, snapshot.type_object_klass,
        snapshot.type_handle, snapshot.type_data, snapshot.type_bits, snapshot.target_klass,
        snapshot.target_name_valid ? snapshot.target_name : "<invalid>",
        snapshot.target_namespace_valid ? snapshot.target_namespace : "<invalid>");
    if (snapshot.type_valid) {
        __android_log_print(
            priority, LOG_TAG,
            "activator diag type_words w0=0x%08X w1=0x%08X w2=0x%08X w3=0x%08X w4=0x%08X w5=0x%08X w6=0x%08X w7=0x%08X",
            snapshot.type_words[0], snapshot.type_words[1], snapshot.type_words[2],
            snapshot.type_words[3], snapshot.type_words[4], snapshot.type_words[5],
            snapshot.type_words[6], snapshot.type_words[7]);
    }
    if (snapshot.stack_valid) {
        for (size_t i = 0; i < snapshot.stack.size(); i += 4) {
            __android_log_print(
                priority, LOG_TAG,
                "activator diag stack sp+0x%02zX=0x%08X sp+0x%02zX=0x%08X sp+0x%02zX=0x%08X sp+0x%02zX=0x%08X",
                i * 4, snapshot.stack[i], (i + 1) * 4, snapshot.stack[i + 1],
                (i + 2) * 4, snapshot.stack[i + 2], (i + 3) * 4, snapshot.stack[i + 3]);
        }
    }
}
}  // namespace

extern "C" int IndexApk(const char* apk_path);
extern "C" void ExtractExpansionFilesFromApks();

static uint32_t GuestUnity(uint32_t rva) {
    return cfg::kUnityBase + rva;
}

static bool ReadAndroidJniHelperReady(DynarmicHost& host, uint32_t& table,
                                      uint32_t& create, uint32_t& invoke) {
    table = host.MemoryRead32(cfg::kUnityBase + 0x112464C);
    if (!table || !GuestRangeValid(host.GetGuestMemorySize(), table, 12)) return false;
    create = host.MemoryRead32(table + 4);
    invoke = host.MemoryRead32(table + 8);
    return create && invoke &&
           GuestRangeValid(host.GetGuestMemorySize(), create, 4) &&
           GuestRangeValid(host.GetGuestMemorySize(), invoke, 4);
}

static uint32_t GuestObject(JNIEnv* env, jobject object) {
    return object ? FakeJniEnv::Instance().AddObjectRef(object) : 0;
}

static bool CopyString(JNIEnv* env, jstring value, std::string& out) {
    if (!value) return false;
    const char* text = env->GetStringUTFChars(value, nullptr);
    if (!text) return false;
    out.assign(text);
    env->ReleaseStringUTFChars(value, text);
    return true;
}

static bool ExtractFromApk(const std::string& apk_path,
                           const std::vector<std::string>& candidates,
                           std::vector<uint8_t>& out_data) {
    if (apk_path.empty()) return false;
    const int fd = open(apk_path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    const off_t file_len = lseek(fd, 0, SEEK_END);
    if (file_len < 22) { close(fd); return false; }
    const off_t search_len = file_len < 65536 + 22 ? file_len : 65536 + 22;
    const off_t search_start = file_len - search_len;
    std::vector<uint8_t> tail(static_cast<size_t>(search_len));
    if (pread(fd, tail.data(), tail.size(), search_start) != static_cast<ssize_t>(tail.size())) {
        close(fd);
        return false;
    }

    off_t eocd_offset = -1;
    for (ssize_t i = static_cast<ssize_t>(tail.size()) - 22; i >= 0; --i) {
        if (tail[i] == 0x50 && tail[i + 1] == 0x4b && tail[i + 2] == 0x05 && tail[i + 3] == 0x06) {
            eocd_offset = search_start + i;
            break;
        }
    }
    if (eocd_offset < 0) { close(fd); return false; }

    uint8_t eocd[22];
    if (pread(fd, eocd, sizeof(eocd), eocd_offset) != sizeof(eocd)) {
        close(fd);
        return false;
    }
    const uint16_t entries = *reinterpret_cast<uint16_t*>(eocd + 10);
    const uint32_t cd_size = *reinterpret_cast<uint32_t*>(eocd + 12);
    const uint32_t cd_offset = *reinterpret_cast<uint32_t*>(eocd + 16);
    if (entries == UINT16_MAX || cd_size == UINT32_MAX || cd_offset == UINT32_MAX ||
        static_cast<uint64_t>(cd_offset) + cd_size > static_cast<uint64_t>(file_len)) {
        close(fd);
        return false;
    }

    std::vector<uint8_t> cd(cd_size);
    if (pread(fd, cd.data(), cd.size(), cd_offset) != static_cast<ssize_t>(cd.size())) {
        close(fd);
        return false;
    }

    uint32_t pos = 0;
    for (uint16_t i = 0; i < entries && pos + 46 <= cd_size; ++i) {
        if (*reinterpret_cast<uint32_t*>(cd.data() + pos) != 0x02014b50) break;
        const uint16_t method = *reinterpret_cast<uint16_t*>(cd.data() + pos + 10);
        const uint32_t csize = *reinterpret_cast<uint32_t*>(cd.data() + pos + 20);
        const uint32_t usize = *reinterpret_cast<uint32_t*>(cd.data() + pos + 24);
        const uint16_t fn_len = *reinterpret_cast<uint16_t*>(cd.data() + pos + 28);
        const uint16_t extra_len = *reinterpret_cast<uint16_t*>(cd.data() + pos + 30);
        const uint16_t comment_len = *reinterpret_cast<uint16_t*>(cd.data() + pos + 32);
        const uint32_t local_offset = *reinterpret_cast<uint32_t*>(cd.data() + pos + 42);
        if (pos + 46u + fn_len + extra_len + comment_len > cd_size) break;

        std::string name(reinterpret_cast<char*>(cd.data() + pos + 46), fn_len);
        bool wanted = false;
        for (const std::string& candidate : candidates) {
            if (name == candidate) { wanted = true; break; }
        }
        if (wanted) {
            uint8_t local[30];
            if (pread(fd, local, sizeof(local), local_offset) != sizeof(local) ||
                *reinterpret_cast<uint32_t*>(local) != 0x04034b50) {
                close(fd);
                return false;
            }
            const uint16_t local_fn_len = *reinterpret_cast<uint16_t*>(local + 26);
            const uint16_t local_extra_len = *reinterpret_cast<uint16_t*>(local + 28);
            const off_t data_offset = local_offset + 30 + local_fn_len + local_extra_len;
            if (method == 0) {
                out_data.resize(usize);
                const bool ok = pread(fd, out_data.data(), usize, data_offset) == static_cast<ssize_t>(usize);
                close(fd);
                return ok;
            }
            if (method == 8) {
                std::vector<uint8_t> compressed(csize);
                if (pread(fd, compressed.data(), compressed.size(), data_offset) !=
                    static_cast<ssize_t>(compressed.size())) {
                    close(fd);
                    return false;
                }
                out_data.resize(usize);
                z_stream stream{};
                if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) { close(fd); return false; }
                stream.next_in = compressed.data();
                stream.avail_in = csize;
                stream.next_out = out_data.data();
                stream.avail_out = usize;
                const int result = inflate(&stream, Z_FINISH);
                inflateEnd(&stream);
                close(fd);
                return result == Z_STREAM_END;
            }
        }
        pos += 46u + fn_len + extra_len + comment_len;
    }
    close(fd);
    return false;
}

static bool ReadApplicationPaths(JNIEnv* env, jobject context) {
    g_apk_paths.clear();
    g_saved_apk_path.clear();
    g_saved_files_dir.clear();
    if (!context) return false;

    jclass context_class = env->GetObjectClass(context);
    jmethodID get_info = env->GetMethodID(
        context_class, "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;");
    jobject info = get_info ? env->CallObjectMethod(context, get_info) : nullptr;
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (!info) {
        env->DeleteLocalRef(context_class);
        return false;
    }

    jclass info_class = env->GetObjectClass(info);
    jfieldID source_field = env->GetFieldID(
        info_class, "sourceDir", "Ljava/lang/String;");
    jstring source = source_field ? static_cast<jstring>(env->GetObjectField(info, source_field)) : nullptr;
    if (source) CopyString(env, source, g_saved_apk_path);
    if (source) env->DeleteLocalRef(source);
    if (!g_saved_apk_path.empty()) g_apk_paths.push_back(g_saved_apk_path);

    jfieldID split_field = env->GetFieldID(
        info_class, "splitSourceDirs", "[Ljava/lang/String;");
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (split_field) {
        jobjectArray splits = static_cast<jobjectArray>(env->GetObjectField(info, split_field));
        if (env->ExceptionCheck()) env->ExceptionClear();
        if (splits) {
            const jsize count = env->GetArrayLength(splits);
            for (jsize i = 0; i < count; ++i) {
                jstring split = static_cast<jstring>(env->GetObjectArrayElement(splits, i));
                std::string path;
                if (CopyString(env, split, path) && !path.empty()) g_apk_paths.push_back(path);
                if (split) env->DeleteLocalRef(split);
            }
            env->DeleteLocalRef(splits);
        }
    }

    jmethodID get_external = env->GetMethodID(
        context_class, "getExternalFilesDir", "(Ljava/lang/String;)Ljava/io/File;");
    jobject file = get_external ? env->CallObjectMethod(context, get_external, nullptr) : nullptr;
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (file) {
        jclass file_class = env->GetObjectClass(file);
        jmethodID absolute = env->GetMethodID(file_class, "getAbsolutePath", "()Ljava/lang/String;");
        jstring path = absolute ? static_cast<jstring>(env->CallObjectMethod(file, absolute)) : nullptr;
        if (env->ExceptionCheck()) env->ExceptionClear();
        if (path) CopyString(env, path, g_saved_files_dir);
        if (path) env->DeleteLocalRef(path);
        env->DeleteLocalRef(file_class);
        env->DeleteLocalRef(file);
    }
    if (g_saved_files_dir.empty()) {
        jmethodID get_files = env->GetMethodID(
            context_class, "getFilesDir", "()Ljava/io/File;");
        jobject files = get_files ? env->CallObjectMethod(context, get_files) : nullptr;
        if (env->ExceptionCheck()) env->ExceptionClear();
        if (files) {
            jclass file_class = env->GetObjectClass(files);
            jmethodID absolute = env->GetMethodID(file_class, "getAbsolutePath", "()Ljava/lang/String;");
            jstring path = absolute ? static_cast<jstring>(env->CallObjectMethod(files, absolute)) : nullptr;
            if (env->ExceptionCheck()) env->ExceptionClear();
            if (path) CopyString(env, path, g_saved_files_dir);
            if (path) env->DeleteLocalRef(path);
            env->DeleteLocalRef(file_class);
            env->DeleteLocalRef(files);
        }
    }

    env->DeleteLocalRef(info_class);
    env->DeleteLocalRef(info);
    env->DeleteLocalRef(context_class);
    return !g_saved_apk_path.empty();
}

static bool RegisterClassNatives(JNIEnv* env, const char* name,
                                 const JNINativeMethod* methods, jint count,
                                 bool required) {
    jclass cls = env->FindClass(name);
    if (!cls) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        LOGE("Native class not found: %s", name);
        return !required;
    }
    const bool ok = env->RegisterNatives(cls, methods, count) == JNI_OK;
    if (!ok && env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
    env->DeleteLocalRef(cls);
    if (!ok) LOGE("RegisterNatives failed for %s", name);
    return ok;
}

static bool EnsureGameLoaded(JNIEnv* env, jobject context);

static void Native_InitJni(JNIEnv* env, jobject self, jobject context) {
    if (!EnsureGameLoaded(env, context) || !g_host) return;
    FakeJniEnv::Instance().AttachHostEnv(env);
    g_host->RunFunction(GuestUnity(cfg::kInitJni), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self), GuestObject(env, context)
    });
    LOGI("Guest initJni completed");
}

static void Native_Done(JNIEnv* env, jobject self) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeDone), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
}

static jboolean Native_Pause(JNIEnv* env, jobject self) {
    if (!g_host) return JNI_FALSE;
    PauseAudioPlayback();
    const uint32_t result = g_host->RunFunction(GuestUnity(cfg::kNativePause), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
    return result ? JNI_TRUE : JNI_FALSE;
}

static void Native_Resume(JNIEnv* env, jobject self) {
    if (!g_host) return;
    ResumeAudioPlayback();
    g_host->RunFunction(GuestUnity(cfg::kNativeResume), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
}

static void Native_RecreateGfxState(JNIEnv* env, jobject self, jint mode, jobject surface) {
    if (!g_host) return;
    if (surface) {
        ANativeWindow* window = ANativeWindow_fromSurface(env, surface);
        if (window) {
            const int width = ANativeWindow_getWidth(window);
            const int height = ANativeWindow_getHeight(window);
            if (width > 0 && height > 0) {
                ThunkManager::s_screenWidth = static_cast<uint32_t>(width);
                ThunkManager::s_screenHeight = static_cast<uint32_t>(height);
            }
            ANativeWindow_release(window);
        }
    }
    g_host->RunFunction(GuestUnity(cfg::kNativeRecreateGfxState), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self),
        static_cast<uint32_t>(mode), GuestObject(env, surface)
    });
}

static jboolean Native_Render(JNIEnv* env, jobject self) {
    if (!g_host) return JNI_FALSE;
    const auto perf_frame_start = std::chrono::steady_clock::now();
    bridge_perf::BeginRenderFrame();
#if defined(__ANDROID__)
    static thread_local uint64_t render_tuning_calls = 0;
    if ((render_tuning_calls++ % 120) == 0) bridge_perf::TuneRenderThread();
    static std::once_flag render_thread_priority;
    std::call_once(render_thread_priority, []() {
        const int priority_result = setpriority(PRIO_PROCESS, 0, -4);
        LOGI("Render thread priority: nice=-4 result=%d errno=%d", priority_result, errno);
    });
#endif
#if CMR_ANDROID_LOG_PRINT && !defined(BRIDGE_RELEASE_LOGS)
    using Clock = std::chrono::steady_clock;
    const auto frame_start = Clock::now();
#endif
    PumpAudioWorkers(*g_host);
    PumpOpenSLESCallbacks(*g_host);
#if CMR_ANDROID_LOG_PRINT && !defined(BRIDGE_RELEASE_LOGS)
    const auto guest_start = Clock::now();
#endif
    const uint32_t result = g_host->RunFunction(GuestUnity(cfg::kNativeRender), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
    const auto reported_frame_end = std::chrono::steady_clock::now();
    if ((render_tuning_calls % 120) == 1) bridge_perf::TuneRenderThread();
    bridge_perf::EndRenderFrame(std::chrono::duration_cast<std::chrono::nanoseconds>(
            reported_frame_end - perf_frame_start).count());
#if CMR_ANDROID_LOG_PRINT && !defined(BRIDGE_RELEASE_LOGS)
    const auto frame_end = Clock::now();
    const uint64_t pump_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(guest_start - frame_start).count());
    const uint64_t guest_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(frame_end - guest_start).count());
    const uint64_t total_us = pump_us + guest_us;
    static uint64_t frame_count = 0;
    static uint64_t pump_sum_us = 0;
    static uint64_t guest_sum_us = 0;
    static uint64_t max_total_us = 0;
    ++frame_count;
    pump_sum_us += pump_us;
    guest_sum_us += guest_us;
    max_total_us = std::max(max_total_us, total_us);
    if (frame_count <= 4 || frame_count % 120 == 0) {
        const uint64_t window = frame_count <= 4 ? frame_count : 120;
        LOGI("Render timing frame=%" PRIu64 " avg_pump_us=%" PRIu64
             " avg_guest_us=%" PRIu64 " max_total_us=%" PRIu64,
             frame_count, pump_sum_us / window, guest_sum_us / window, max_total_us);
        if (frame_count > 4) {
            pump_sum_us = 0;
            guest_sum_us = 0;
            max_total_us = 0;
        }
    }
#endif
    return result ? JNI_TRUE : JNI_FALSE;
}

static void Native_LowMemory(JNIEnv* env, jobject self) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeLowMemory), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
}

static void Native_FocusChanged(JNIEnv* env, jobject self, jboolean focused) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeFocusChanged), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self), focused ? 1u : 0u
    });
}

static void Native_SetInputString(JNIEnv* env, jobject self, jstring value) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeSetInputString), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self), GuestObject(env, value)
    });
}

static void Native_SoftInputClosed(JNIEnv* env, jobject self) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeSoftInputClosed), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
}

static void Native_SoftInputCanceled(JNIEnv* env, jobject self) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeSoftInputCanceled), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
}

static void Native_SoftInputLostFocus(JNIEnv* env, jobject self) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeSoftInputLostFocus), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
}

static void Native_InitWebRequest(JNIEnv* env, jobject self, jclass web_request) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeInitWebRequest), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self), GuestObject(env, web_request)
    });
}

static jboolean Native_InjectEvent(JNIEnv* env, jobject self, jobject event) {
    if (!g_host) return JNI_FALSE;
    const uint32_t result = g_host->RunFunction(GuestUnity(cfg::kNativeInjectEvent), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self), GuestObject(env, event)
    });
    return result ? JNI_TRUE : JNI_FALSE;
}

static void Native_UnitySendMessage(JNIEnv* env, jclass clazz,
                                    jstring object_name, jstring method, jstring message) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeUnitySendMessage), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, clazz),
        GuestObject(env, object_name), GuestObject(env, method), GuestObject(env, message)
    });
}

static jboolean Native_IsAutorotationOn(JNIEnv* env, jobject self) {
    if (!g_host) return JNI_FALSE;
    const uint32_t result = g_host->RunFunction(GuestUnity(cfg::kNativeIsAutorotationOn), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
    return result ? JNI_TRUE : JNI_FALSE;
}

static void Native_MuteMasterAudio(JNIEnv* env, jobject self, jboolean mute) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeMuteMasterAudio), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self), mute ? 1u : 0u
    });
}

static void Native_RestartActivityIndicator(JNIEnv* env, jobject self) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kNativeRestartActivityIndicator), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
}

static jobject Native_ReflectionInvoke(JNIEnv* env, jclass clazz, jint proxy,
                                       jstring method, jobjectArray args) {
    if (!g_host) return nullptr;
    const uint32_t result = g_host->RunFunction(cfg::kUnityBase + cfg::kReflectionInvoke, {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, clazz),
        static_cast<uint32_t>(proxy), GuestObject(env, method), GuestObject(env, args)
    });
    return FakeJniEnv::Instance().ResolveObject(result);
}

static void Native_ReflectionFinalize(JNIEnv* env, jclass clazz, jint proxy) {
    if (g_host) g_host->RunFunction(cfg::kUnityBase + cfg::kReflectionFinalize, {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, clazz), static_cast<uint32_t>(proxy)
    });
}

static void Native_VrInit(JNIEnv* env, jobject self) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kVrInit), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
}

static void Native_VrSetVideoTransform(JNIEnv* env, jobject self, jobjectArray transform) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kVrSetVideoTransform), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self), GuestObject(env, transform)
    });
}

static jboolean Native_VrIsQuitting(JNIEnv* env, jobject self) {
    if (!g_host) return JNI_FALSE;
    return g_host->RunFunction(GuestUnity(cfg::kVrIsQuitting), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    }) ? JNI_TRUE : JNI_FALSE;
}

static void Native_ArInitialize(JNIEnv* env, jobject self, jobject activity) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kArInitialize), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self), GuestObject(env, activity)
    });
}

static void Native_ArPause(JNIEnv* env, jobject self) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kArPause), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
}

static void Native_ArResume(JNIEnv* env, jobject self) {
    if (g_host) g_host->RunFunction(GuestUnity(cfg::kArResume), {
        FakeJniEnv::Instance().EnvTableGuestAddr(), GuestObject(env, self)
    });
}

static void Native_ArProxyOnCreate(JNIEnv*, jobject, jobject) {}
static void Native_ArProxyOnServiceConnected(JNIEnv*, jobject, jobject) {}
static void Native_ArProxyOnStop(JNIEnv*, jobject) {}

static void Native_WebContentLength(JNIEnv*, jclass, jlong, jint) {}
static jboolean Native_WebDownload(JNIEnv*, jclass, jlong, jobject, jint) { return JNI_FALSE; }
static void Native_WebError(JNIEnv*, jclass, jlong, jint, jstring) {}
static void Native_WebHeader(JNIEnv*, jclass, jlong, jstring, jstring) {}
static void Native_WebResponseCode(JNIEnv*, jclass, jlong, jint) {}
static jint Native_WebUpload(JNIEnv*, jclass, jlong, jobject) { return 0; }

static void Native_FmodDelete(JNIEnv*, jclass, jlong) {}
static jobject Native_FmodInvoke(JNIEnv* env, jclass clazz, jlong proxy,
                                 jclass declaring_class, jobject method, jobjectArray args) {
    if (!g_host || !method) return nullptr;
    jclass method_class = env->GetObjectClass(method);
    jmethodID get_name = method_class ? env->GetMethodID(method_class, "getName", "()Ljava/lang/String;") : nullptr;
    jstring name = get_name ? static_cast<jstring>(env->CallObjectMethod(method, get_name)) : nullptr;
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (!name) {
        if (method_class) env->DeleteLocalRef(method_class);
        return nullptr;
    }
    FakeJniEnv::Instance().AttachHostEnv(env);
    const char* method_text = env->GetStringUTFChars(name, nullptr);
    const bool defer_run = method_text && strcmp(method_text, "run") == 0;
    const bool is_do_frame = method_text && strcmp(method_text, "doFrame") == 0;
    static std::atomic<unsigned> do_frame_logs{0};
    const bool trace_invoke = !is_do_frame ||
        do_frame_logs.fetch_add(1, std::memory_order_relaxed) < 4;
    if (trace_invoke) {
        LOGI("JNI bridge invoke method=%s proxy=0x%08X args=%p", method_text ? method_text : "<invalid>",
             static_cast<uint32_t>(proxy), args);
    }
    if (method_text) env->ReleaseStringUTFChars(name, method_text);
    const uint32_t proxy32 = static_cast<uint32_t>(proxy);
    if (trace_invoke) {
        std::string proxy_words;
        for (unsigned i = 0; i < 8; ++i) {
            char word[24];
            snprintf(word, sizeof(word), " +%u=0x%08X", i * 4, g_host->MemoryRead32(proxy32 + i * 4));
            proxy_words += word;
        }
        LOGI("JNI bridge proxy=0x%08X%s", proxy32, proxy_words.c_str());
    }

    bool cancelled = false;
    if (defer_run && !g_android_jni_helper_ready.load(std::memory_order_acquire)) {
        LOGI("JNI bridge invoke run waiting for AndroidJNIHelper readiness");
        std::unique_lock<std::mutex> lock(g_android_jni_helper_ready_mutex);
        g_android_jni_helper_ready_cv.wait(lock, [] {
            return g_android_jni_helper_ready.load(std::memory_order_acquire) ||
                   g_android_jni_helper_wait_cancelled.load(std::memory_order_acquire);
        });
        cancelled = g_android_jni_helper_wait_cancelled.load(std::memory_order_acquire);
        LOGI("JNI bridge invoke run readiness wait ended ready=%d cancelled=%d",
             g_android_jni_helper_ready.load(std::memory_order_acquire) ? 1 : 0,
             cancelled ? 1 : 0);
    }

    if (trace_invoke) {
        LOGI("JNI bridge callbacks current=0x%08X invoke=0x%08X invokeA=0x%08X gc_register=%u",
             g_host->MemoryRead32(cfg::kUnityBase + 0x115AF60),
             g_host->MemoryRead32(cfg::kUnityBase + 0x115AFF8),
             g_host->MemoryRead32(cfg::kUnityBase + 0x115AFFC),
             g_host->MemoryRead32(cfg::kIl2cppBase + 0x2A6CBC8));
    }
    uint32_t result_handle = 0;
    if (!cancelled &&
        !g_android_jni_helper_wait_cancelled.load(std::memory_order_acquire) && g_host) {
        const uint64_t proxy_bits = static_cast<uint64_t>(proxy);
        auto& jni = FakeJniEnv::Instance();
        result_handle = g_host->RunFunction(GuestUnity(cfg::kJniBridgeInvoke), {
            jni.EnvTableGuestAddr(), GuestObject(env, clazz),
            static_cast<uint32_t>(proxy_bits), static_cast<uint32_t>(proxy_bits >> 32),
            GuestObject(env, declaring_class), GuestObject(env, method), GuestObject(env, args)
        });
    }
    const jobject result = FakeJniEnv::Instance().ResolveObject(result_handle);
    if (trace_invoke) {
        LOGI("JNI bridge invoke proxy=0x%llX method=%p result_handle=0x%08X result=%p",
             static_cast<unsigned long long>(proxy), method, result_handle, result);
    }
    env->DeleteLocalRef(name);
    if (method_class) env->DeleteLocalRef(method_class);
    return result;
}
static std::mutex g_fmod_callbacks;

static jint Native_FmodGetInfo(JNIEnv* env, jobject self, jint selector) {
    std::lock_guard<std::mutex> lock(g_fmod_callbacks);
    if (!g_host) return -1;
    auto& jni = FakeJniEnv::Instance();
    jni.AttachHostEnv(env);
    const jint result = static_cast<jint>(g_host->RunAudioFunction(GuestUnity(cfg::kFmodGetInfo), {
        jni.EnvTableGuestAddr(), GuestObject(env, self), static_cast<uint32_t>(selector)}));
    static unsigned logged = 0;
    if (logged++ < 8) LOGI("guest FMOD info selector=%d result=%d", selector, result);
    return result;
}

static jint Native_FmodProcess(JNIEnv* env, jobject self, jobject buffer) {
    std::lock_guard<std::mutex> lock(g_fmod_callbacks);
    if (!g_host) return -1;
    auto& jni = FakeJniEnv::Instance();
    jni.AttachHostEnv(env);
    const uint32_t handle = jni.RegisterDirectBuffer(buffer);
    if (!handle) return -1;
    static unsigned input_logged = 0;
    if (input_logged++ < 4) {
        LOGI("FMOD process input env=0x%08X slot920=0x%08X handle=0x%08X capacity=%lld",
             jni.EnvTableGuestAddr(), g_host->MemoryRead32(jni.EnvTableGuestAddr() + 920), handle,
             static_cast<long long>(jni.DirectBufferCapacity(handle)));
    }
    const jint result = static_cast<jint>(g_host->RunAudioFunction(GuestUnity(cfg::kFmodProcess), {
        jni.EnvTableGuestAddr(), GuestObject(env, self), handle}));
    jni.CommitDirectBuffer(handle);
    static unsigned logged = 0;
    if (logged++ < 4) LOGI("guest FMOD process result=%d capacity=%lld", result,
                           static_cast<long long>(jni.DirectBufferCapacity(handle)));
    return result;
}

static jint Native_FmodProcessMicData(JNIEnv* env, jobject self, jobject buffer, jint length) {
    std::lock_guard<std::mutex> lock(g_fmod_callbacks);
    if (!g_host || length < 0) return -1;
    auto& jni = FakeJniEnv::Instance();
    jni.AttachHostEnv(env);
    const uint32_t handle = jni.RegisterDirectBuffer(buffer);
    if (!handle || length > jni.DirectBufferCapacity(handle)) return -1;
    return static_cast<jint>(g_host->RunAudioFunction(GuestUnity(cfg::kFmodProcessMicData), {
        jni.EnvTableGuestAddr(), GuestObject(env, self), handle, static_cast<uint32_t>(length)}));
}

static const JNINativeMethod kUnityMethods[] = {
    {"initJni", "(Landroid/content/Context;)V", reinterpret_cast<void*>(Native_InitJni)},
    {"nativeDone", "()V", reinterpret_cast<void*>(Native_Done)},
    {"nativePause", "()Z", reinterpret_cast<void*>(Native_Pause)},
    {"nativeRecreateGfxState", "(ILandroid/view/Surface;)V", reinterpret_cast<void*>(Native_RecreateGfxState)},
    {"nativeRender", "()Z", reinterpret_cast<void*>(Native_Render)},
    {"nativeResume", "()V", reinterpret_cast<void*>(Native_Resume)},
    {"nativeLowMemory", "()V", reinterpret_cast<void*>(Native_LowMemory)},
    {"nativeFocusChanged", "(Z)V", reinterpret_cast<void*>(Native_FocusChanged)},
    {"nativeSetInputString", "(Ljava/lang/String;)V", reinterpret_cast<void*>(Native_SetInputString)},
    {"nativeSoftInputClosed", "()V", reinterpret_cast<void*>(Native_SoftInputClosed)},
    {"nativeSoftInputCanceled", "()V", reinterpret_cast<void*>(Native_SoftInputCanceled)},
    {"nativeSoftInputLostFocus", "()V", reinterpret_cast<void*>(Native_SoftInputLostFocus)},
    {"nativeInitWebRequest", "(Ljava/lang/Class;)V", reinterpret_cast<void*>(Native_InitWebRequest)},
    {"nativeInjectEvent", "(Landroid/view/InputEvent;)Z", reinterpret_cast<void*>(Native_InjectEvent)},
    {"nativeUnitySendMessage", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V", reinterpret_cast<void*>(Native_UnitySendMessage)},
    {"nativeIsAutorotationOn", "()Z", reinterpret_cast<void*>(Native_IsAutorotationOn)},
    {"nativeMuteMasterAudio", "(Z)V", reinterpret_cast<void*>(Native_MuteMasterAudio)},
    {"nativeRestartActivityIndicator", "()V", reinterpret_cast<void*>(Native_RestartActivityIndicator)},
};

static const JNINativeMethod kReflectionMethods[] = {
    {"nativeProxyInvoke", "(ILjava/lang/String;[Ljava/lang/Object;)Ljava/lang/Object;", reinterpret_cast<void*>(Native_ReflectionInvoke)},
    {"nativeProxyFinalize", "(I)V", reinterpret_cast<void*>(Native_ReflectionFinalize)},
};

static const JNINativeMethod kWebRequestMethods[] = {
    {"contentLengthCallback", "(JI)V", reinterpret_cast<void*>(Native_WebContentLength)},
    {"downloadCallback", "(JLjava/nio/ByteBuffer;I)Z", reinterpret_cast<void*>(Native_WebDownload)},
    {"errorCallback", "(JILjava/lang/String;)V", reinterpret_cast<void*>(Native_WebError)},
    {"headerCallback", "(JLjava/lang/String;Ljava/lang/String;)V", reinterpret_cast<void*>(Native_WebHeader)},
    {"responseCodeCallback", "(JI)V", reinterpret_cast<void*>(Native_WebResponseCode)},
    {"uploadCallback", "(JLjava/nio/ByteBuffer;)I", reinterpret_cast<void*>(Native_WebUpload)},
};

static const JNINativeMethod kArProxyMethods[] = {
    {"tangoOnCreate", "(Landroid/app/Activity;)V", reinterpret_cast<void*>(Native_ArProxyOnCreate)},
    {"tangoOnServiceConnected", "(Landroid/os/IBinder;)V", reinterpret_cast<void*>(Native_ArProxyOnServiceConnected)},
    {"tangoOnStop", "()V", reinterpret_cast<void*>(Native_ArProxyOnStop)},
};

static const JNINativeMethod kFmodMethods[] = {
    {"fmodGetInfo", "(I)I", reinterpret_cast<void*>(Native_FmodGetInfo)},
    {"fmodProcess", "(Ljava/nio/ByteBuffer;)I", reinterpret_cast<void*>(Native_FmodProcess)},
    {"fmodProcessMicData", "(Ljava/nio/ByteBuffer;I)I", reinterpret_cast<void*>(Native_FmodProcessMicData)},
};

static const JNINativeMethod kJniBridgeMethods[] = {
    {"delete", "(J)V", reinterpret_cast<void*>(Native_FmodDelete)},
    {"invoke", "(JLjava/lang/Class;Ljava/lang/reflect/Method;[Ljava/lang/Object;)Ljava/lang/Object;", reinterpret_cast<void*>(Native_FmodInvoke)},
};

static const JNINativeMethod kVrMethods[] = {
    {"initVrJni", "()V", reinterpret_cast<void*>(Native_VrInit)},
    {"setVrVideoTransform", "([[F)V", reinterpret_cast<void*>(Native_VrSetVideoTransform)},
    {"isQuiting", "()Z", reinterpret_cast<void*>(Native_VrIsQuitting)},
};

static const JNINativeMethod kArMethods[] = {
    {"initializeARCore", "(Landroid/app/Activity;)V", reinterpret_cast<void*>(Native_ArInitialize)},
    {"pauseARCore", "()V", reinterpret_cast<void*>(Native_ArPause)},
    {"resumeARCore", "()V", reinterpret_cast<void*>(Native_ArResume)},
};

static bool RegisterBridgeNatives(JNIEnv* env) {
    if (g_registered_natives) return true;
    if (!RegisterClassNatives(env, cfg::kUnityPlayerClass, kUnityMethods,
                              sizeof(kUnityMethods) / sizeof(kUnityMethods[0]), true)) return false;
    if (!RegisterClassNatives(env, cfg::kReflectionClass, kReflectionMethods,
                              sizeof(kReflectionMethods) / sizeof(kReflectionMethods[0]), true)) return false;
    if (!RegisterClassNatives(env, cfg::kWebRequestClass, kWebRequestMethods,
                              sizeof(kWebRequestMethods) / sizeof(kWebRequestMethods[0]), true)) return false;
    if (!RegisterClassNatives(env, cfg::kJniBridgeClass, kJniBridgeMethods,
                              sizeof(kJniBridgeMethods) / sizeof(kJniBridgeMethods[0]), true)) return false;
    RegisterClassNatives(env, cfg::kFmodClass, kFmodMethods,
                         sizeof(kFmodMethods) / sizeof(kFmodMethods[0]), false);
    RegisterClassNatives(env, cfg::kVrClass, kVrMethods,
                         sizeof(kVrMethods) / sizeof(kVrMethods[0]), false);
    RegisterClassNatives(env, cfg::kArClass, kArMethods,
                         sizeof(kArMethods) / sizeof(kArMethods[0]), false);
    RegisterClassNatives(env, cfg::kArProxyClass, kArProxyMethods,
                         sizeof(kArProxyMethods) / sizeof(kArProxyMethods[0]), false);
    g_registered_natives = true;
    LOGI("Registered République UnityPlayer JNI bridge methods");
    return true;
}

static jboolean NativeLoader_Load(JNIEnv* env, jclass, jstring native_library_dir) {
    g_android_jni_invoke_method.store(0, std::memory_order_release);
    g_android_jni_helper_initializing.store(false, std::memory_order_release);
    g_android_jni_helper_ready.store(false, std::memory_order_release);
    g_android_jni_helper_wait_cancelled.store(false, std::memory_order_release);
    g_android_jni_helper_init_depth = 0;
    std::string path;
    CopyString(env, native_library_dir, path);
    LOGI("NativeLoader.load('%s')", path.c_str());
    return RegisterBridgeNatives(env) ? JNI_TRUE : JNI_FALSE;
}

static jboolean NativeLoader_Unload(JNIEnv*, jclass) {
    g_android_jni_helper_wait_cancelled.store(true, std::memory_order_release);
    g_android_jni_helper_ready_cv.notify_all();
    g_android_jni_helper_initializing.store(false, std::memory_order_release);
    StopAudioThread();
    bridge_perf::Shutdown();
    if (g_host) {
        g_host->RunFunction(GuestUnity(cfg::kUnityJniOnUnload), {
            FakeJniEnv::Instance().VmTableGuestAddr(), 0
        });
    }
    g_host.reset();
    g_android_jni_invoke_method.store(0, std::memory_order_release);
    g_android_jni_helper_initializing.store(false, std::memory_order_release);
    g_android_jni_helper_ready.store(false, std::memory_order_release);
    g_android_jni_helper_init_depth = 0;
    g_registered_natives = false;
    LOGI("NativeLoader.unload()");
    return JNI_TRUE;
}

static const JNINativeMethod kLoaderMethods[] = {
    {"load", "(Ljava/lang/String;)Z", reinterpret_cast<void*>(NativeLoader_Load)},
    {"unload", "()Z", reinterpret_cast<void*>(NativeLoader_Unload)},
};

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    g_jvm = vm;
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) {
        return JNI_ERR;
    }
    if (!RegisterClassNatives(env, cfg::kNativeLoaderClass, kLoaderMethods,
                              sizeof(kLoaderMethods) / sizeof(kLoaderMethods[0]), true)) {
        return JNI_ERR;
    }
    LOGI("JNI_OnLoad: ARM64 host for République ARM32 guest");
    return JNI_VERSION_1_6;
}

static bool InstallCriErrorWatch(DynarmicHost& host) {
    // sub_1164D8 is CRI's central error dispatcher. Hook its second instruction:
    // PUSH has already preserved the frame, and we replay only `add r11, sp, #0x18`.
    constexpr uint32_t kWatchPc = cfg::kCriBase + 0x1164DC;
    constexpr uint32_t kExpectedAdd = 0xE28DB018;
    if (host.MemoryRead32(kWatchPc) != kExpectedAdd) {
        LOGE("CRI error watch: unexpected opcode at 0x%08X: 0x%08X",
             kWatchPc, host.MemoryRead32(kWatchPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_cri_error_watch", [kWatchPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t severity = jit.Regs()[0];
            const uint32_t message_address = jit.Regs()[1];
            const char* message = GuestRangeValid(h.GetGuestMemorySize(), message_address, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + message_address)
                : "<invalid guest string>";
            static std::atomic<unsigned> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 128) {
                LOGE("CRI internal error severity=%u message=%s", severity, message);
            }
            jit.Regs()[11] = jit.Regs()[13] + 0x18;
            jit.Regs()[15] = kWatchPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("CRI error watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kWatchPc, svc);
    host.InvalidateGuestCode(kWatchPc, sizeof(uint32_t));
    LOGI("CRI error watch installed at 0x%08X", kWatchPc);
    return true;
}

static bool InstallUnityLz4Hook(DynarmicHost& host) {
    // libunity's LZ4_decompress_safe is a compact ARM32 implementation with
    // byte/word copy loops.  UnityFS calls it once for every ~128 KiB asset
    // block, so executing it through Dynarmic dominates episode loading even
    // though the outer OBB itself is ZIP_STORED.  Replace only this checksum-
    // pinned entry point; its caller and all UnityFS validation remain guest
    // code.
    constexpr uint32_t kEntryPc = cfg::kUnityBase + 0x79A84;
    constexpr std::array<uint32_t, 4> kExpectedWords = {
        0xE92D4FF0,  // push {r4-r11, lr}
        0xE24DD01C,  // sub sp, sp, #0x1c
        0xE1A04001,  // mov r4, r1
        0xE3530000,  // cmp r3, #0
    };
    for (size_t index = 0; index < kExpectedWords.size(); ++index) {
        const uint32_t address = kEntryPc + static_cast<uint32_t>(index * sizeof(uint32_t));
        const uint32_t actual = host.MemoryRead32(address);
        if (actual != kExpectedWords[index]) {
            LOGE("Unity LZ4 hook: unexpected opcode at 0x%08X: expected=0x%08X actual=0x%08X",
                 address, kExpectedWords[index], actual);
            return false;
        }
    }

    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "unity_lz4_decompress_safe_host", [](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            auto& regs = jit.Regs();
            const uint32_t source_address = regs[0];
            const uint32_t destination_address = regs[1];
            const int32_t compressed_size = static_cast<int32_t>(regs[2]);
            const int32_t destination_capacity = static_cast<int32_t>(regs[3]);

            const uint64_t started_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            int result = -1;
            const bool valid = compressed_size >= 0 && destination_capacity >= 0 &&
                GuestRangeValid(h.GetGuestMemorySize(), source_address,
                                static_cast<size_t>(compressed_size)) &&
                GuestRangeValid(h.GetGuestMemorySize(), destination_address,
                                static_cast<size_t>(destination_capacity));
            if (valid) {
                result = LZ4_decompress_safe(
                    reinterpret_cast<const char*>(h.GetGuestMemoryBase() + source_address),
                    reinterpret_cast<char*>(h.GetGuestMemoryBase() + destination_address),
                    compressed_size, destination_capacity);
            }
            const uint64_t finished_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());

            static std::atomic<uint64_t> calls{0};
            static std::atomic<uint64_t> input_bytes{0};
            static std::atomic<uint64_t> output_bytes{0};
            static std::atomic<uint64_t> elapsed_ns{0};
            static std::atomic<uint64_t> errors{0};
            const uint64_t call = calls.fetch_add(1, std::memory_order_relaxed) + 1;
            const uint64_t input_total = input_bytes.fetch_add(
                valid ? static_cast<uint64_t>(compressed_size) : 0,
                std::memory_order_relaxed) +
                (valid ? static_cast<uint64_t>(compressed_size) : 0);
            const uint64_t output_delta = result > 0 ? static_cast<uint64_t>(result) : 0;
            const uint64_t output_total = output_bytes.fetch_add(
                output_delta, std::memory_order_relaxed) + output_delta;
            const uint64_t elapsed_total = elapsed_ns.fetch_add(
                finished_ns - started_ns, std::memory_order_relaxed) +
                (finished_ns - started_ns);
            const uint64_t error_total = result < 0
                ? errors.fetch_add(1, std::memory_order_relaxed) + 1
                : errors.load(std::memory_order_relaxed);
            if (call == 1 || (call % 1024) == 0 || (result < 0 && error_total <= 8)) {
                LOGI("Unity LZ4 host: calls=%" PRIu64 " input=%" PRIu64
                     " output=%" PRIu64 " time_ms=%.3f avg_us=%.3f errors=%" PRIu64,
                     call, input_total, output_total,
                     static_cast<double>(elapsed_total) / 1000000.0,
                     static_cast<double>(elapsed_total) / (1000.0 * static_cast<double>(call)),
                     error_total);
            }

            regs[0] = static_cast<uint32_t>(static_cast<int32_t>(result));
            const uint32_t return_address = regs[14];
            jit.SetCpsr((jit.Cpsr() & ~0x20u) | ((return_address & 1u) ? 0x20u : 0u));
            regs[15] = return_address & ~1u;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("Unity LZ4 hook: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kEntryPc, svc);
    host.InvalidateGuestCode(kEntryPc, sizeof(uint32_t));
    LOGI("Unity LZ4 host hook installed at 0x%08X", kEntryPc);
    return true;
}

static bool ApplyGameOptionsFrameRatePolicyPatch(DynarmicHost& host) {
    // GameOptionsSaveData::.ctor initializes _batterySaverModeOn to true and
    // immediately applies the 15 FPS saver target.  On the original runtime
    // deserialization replaces that default before gameplay.  A reset/default
    // options object can remain authoritative across the ARM32 bridge, though,
    // which re-applies 15 FPS when an episode starts even after the user chose
    // Battery Saver OFF.  Default to OFF so the constructor's existing policy
    // path selects 120 FPS.  OnDeserialized, SetBatterySaverState and
    // SetGraphicsQuality still apply the saved/live ON=15 and OFF=120 values.
    constexpr uint32_t kBatterySaverDefaultPc = cfg::kIl2cppBase + 0x563858;
    constexpr uint32_t kExpectedDefaultOn = 0xE5C4504C;  // strb r5, [r4, #0x4c]
    constexpr uint32_t kDefaultOff = 0xE5C4604C;         // strb r6, [r4, #0x4c]
    const uint32_t actual = host.MemoryRead32(kBatterySaverDefaultPc);
    if (actual != kExpectedDefaultOn) {
        LOGE("GameOptions FPS policy: unexpected opcode at 0x%08X: 0x%08X",
             kBatterySaverDefaultPc, actual);
        return false;
    }
    host.MemoryWrite32(kBatterySaverDefaultPc, kDefaultOff);
    host.InvalidateGuestCode(kBatterySaverDefaultPc, sizeof(uint32_t));
    LOGI("GameOptions FPS policy: default Battery Saver OFF (120 FPS)");
    return true;
}

static bool InstallTargetFrameRateWatch(DynarmicHost& host) {
#if defined(BRIDGE_RELEASE_LOGS)
    (void)host;
    return true;
#else
    // Application::set_targetFrameRate(int) receives the managed value in r1.
    // Replace its `mov r4, r1`, log every write, then replay that instruction.
    constexpr uint32_t kWatchPc = cfg::kIl2cppBase + 0x1D4AFF8;
    constexpr uint32_t kExpectedMov = 0xE1A04001;
    if (host.MemoryRead32(kWatchPc) != kExpectedMov) {
        LOGE("targetFrameRate watch: unexpected opcode at 0x%08X: 0x%08X",
             kWatchPc, host.MemoryRead32(kWatchPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_target_frame_rate_watch", [kWatchPc](Dynarmic::A32::Jit& jit, DynarmicHost&) {
            static std::atomic<unsigned> writes{0};
            const unsigned write = writes.fetch_add(1, std::memory_order_relaxed) + 1;
            LOGI("targetFrameRate write=%u value=%d caller=0x%08X",
                 write, static_cast<int32_t>(jit.Regs()[1]), jit.Regs()[14]);
            jit.Regs()[4] = jit.Regs()[1];
            jit.Regs()[15] = kWatchPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("targetFrameRate watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kWatchPc, svc);
    host.InvalidateGuestCode(kWatchPc, sizeof(uint32_t));
    LOGI("targetFrameRate watch installed at 0x%08X", kWatchPc);
    return true;
#endif
}

static bool InstallMetadataStoreWatch(DynarmicHost& host) {
    constexpr uint32_t kStorePc = cfg::kIl2cppBase + 0x2289B70;
    constexpr uint32_t kExpectedStore = 0xE7805104;
    if (host.MemoryRead32(kStorePc) != kExpectedStore) {
        LOGE("metadata store watch: unexpected opcode at 0x%08X: 0x%08X",
             kStorePc, host.MemoryRead32(kStorePc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().GetOrCreateThunkAddress("metadata_store_watch");
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("metadata store watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kStorePc, svc);
    host.InvalidateGuestCode(kStorePc, sizeof(uint32_t));
    LOGI("metadata store watch installed at 0x%08X (replaced 0x%08X)",
         kStorePc, kExpectedStore);
    return true;
}

static bool InstallNullReferenceWatch(DynarmicHost& host) {
    constexpr uint32_t kNullCheckPc = cfg::kIl2cppBase + 0x22b7ca8;
    constexpr uint32_t kExpectedMovLrPc = 0xE1A0E00F;
    if (host.MemoryRead32(kNullCheckPc) != kExpectedMovLrPc) {
        LOGE("null-reference watch: unexpected opcode at 0x%08X: 0x%08X",
             kNullCheckPc, host.MemoryRead32(kNullCheckPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_null_reference_watch", [kNullCheckPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const auto regs = jit.Regs();
            const uint32_t sp = regs[13];
            std::array<uint32_t, 16> stack{};
            const bool stack_valid = GuestRangeValid(h.GetGuestMemorySize(), sp, sizeof(stack));
            if (stack_valid) {
                for (size_t i = 0; i < stack.size(); ++i) stack[i] = h.MemoryRead32(sp + i * 4);
            }
            LOGE("guest null check host_tid=%d guest_tid=0x%08X pre_lr=0x%08X sp=0x%08X cpsr=0x%08X stack_valid=%d",
                 gettid(), DynarmicHost::CurrentGuestThreadId(), regs[14], sp, jit.Cpsr(), stack_valid);
            LOGE("guest null regs r0=0x%08X r1=0x%08X r2=0x%08X r3=0x%08X r4=0x%08X r5=0x%08X r6=0x%08X r7=0x%08X",
                 regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7]);
            LOGE("guest null regs r8=0x%08X r9=0x%08X r10=0x%08X r11=0x%08X r12=0x%08X",
                 regs[8], regs[9], regs[10], regs[11], regs[12]);
            for (size_t i = 0; i < stack.size(); i += 4) {
                LOGE("guest null stack sp+0x%02zX=0x%08X sp+0x%02zX=0x%08X sp+0x%02zX=0x%08X sp+0x%02zX=0x%08X",
                     i * 4, stack[i], (i + 1) * 4, stack[i + 1],
                     (i + 2) * 4, stack[i + 2], (i + 3) * 4, stack[i + 3]);
            }
            jit.Regs()[14] = kNullCheckPc + 8;
            jit.Regs()[15] = kNullCheckPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("null-reference watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kNullCheckPc, svc);
    host.InvalidateGuestCode(kNullCheckPc, sizeof(uint32_t));
    LOGI("null-reference watch installed at 0x%08X", kNullCheckPc);
    return true;
}

static bool InstallThrowWatch(DynarmicHost& host) {
    constexpr uint32_t kAdrlSecond = cfg::kIl2cppBase + 0x37acc4;
    constexpr uint32_t kExpectedAdd = 0xE28CCADC;
    if (host.MemoryRead32(kAdrlSecond) != kExpectedAdd) {
        LOGE("throw watch: unexpected opcode at 0x%08X: 0x%08X",
             kAdrlSecond, host.MemoryRead32(kAdrlSecond));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_throw_watch", [kAdrlSecond](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t object = jit.Regs()[0];
            const uint32_t typeinfo = jit.Regs()[1];
            const bool valid_object = object >= 0x1000 &&
                GuestRangeValid(h.GetGuestMemorySize(), object, 4);
            const bool valid_typeinfo = typeinfo >= 0x1000 &&
                GuestRangeValid(h.GetGuestMemorySize(), typeinfo, 8);
            const uint32_t name = valid_typeinfo ? h.MemoryRead32(typeinfo + 4) : 0;
            const char* text = GuestRangeValid(h.GetGuestMemorySize(), name, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + name) : "<invalid>";
            LOGE("guest __cxa_throw pass-through object=0x%08X typeinfo=0x%08X name=%s dtor=0x%08X caller=0x%08X valid=%d",
                 object, typeinfo, text, jit.Regs()[2], jit.Regs()[14],
                 valid_object && valid_typeinfo ? 1 : 0);
            if (!valid_object || !valid_typeinfo) {
                LOGE("guest __cxa_throw rejected invalid guest state");
                jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
                return;
            }
            // The replaced PLT instruction is exactly `add r12, r12, #0xADC`.
            jit.Regs()[12] += 0xADCu;
            jit.Regs()[15] = kAdrlSecond + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("throw watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kAdrlSecond, svc);
    host.InvalidateGuestCode(kAdrlSecond, sizeof(uint32_t));
    LOGI("throw watch installed at 0x%08X with guest pass-through", kAdrlSecond);
    return true;
}

static bool InstallIl2cppInitCompletionWatch(DynarmicHost& host) {
    constexpr uint32_t kAfterInitPc = cfg::kUnityBase + 0x2DFE04;
    constexpr uint32_t kExpectedLdr = 0xE59D0014;
    if (host.MemoryRead32(kAfterInitPc) != kExpectedLdr) {
        LOGE("IL2CPP init completion watch: unexpected opcode at 0x%08X: 0x%08X",
             kAfterInitPc, host.MemoryRead32(kAfterInitPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_il2cpp_init_complete", [kAfterInitPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t helper_global = cfg::kUnityBase + 0x112464C;
            const uint32_t old_table = h.MemoryRead32(helper_global);
            if (old_table) {
                LOGI("Clearing pre-IL2CPP AndroidJNIHelper table 0x%08X", old_table);
            }
            g_android_jni_helper_ready.store(false, std::memory_order_release);
            h.MemoryWrite32(helper_global, 0);
            jit.Regs()[0] = h.MemoryRead32(jit.Regs()[13] + 0x14);
            jit.Regs()[15] = kAfterInitPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("IL2CPP init completion watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kAfterInitPc, svc);
    host.InvalidateGuestCode(kAfterInitPc, sizeof(uint32_t));
    LOGI("IL2CPP init completion watch installed at 0x%08X", kAfterInitPc);
    return true;
}

static bool InstallAndroidJniHelperAccessWatch(DynarmicHost& host) {
    constexpr uint32_t kHelperAccessPc = cfg::kUnityBase + 0x134E28;
    constexpr uint32_t kHelperReturnPc = cfg::kUnityBase + 0x134E30;
    constexpr uint32_t kExpectedPush = 0xE92D4800;
    constexpr uint32_t kExpectedPop = 0xE8BD8800;
    if (host.MemoryRead32(kHelperAccessPc) != kExpectedPush ||
        host.MemoryRead32(kHelperReturnPc) != kExpectedPop) {
        LOGE("AndroidJNIHelper access watch: unexpected opcodes at 0x%08X/0x%08X: 0x%08X/0x%08X",
             kHelperAccessPc, kHelperReturnPc, host.MemoryRead32(kHelperAccessPc),
             host.MemoryRead32(kHelperReturnPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_androidjni_helper_access", [kHelperAccessPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t table = h.MemoryRead32(cfg::kUnityBase + 0x112464C);
            if (g_android_jni_helper_init_depth != 0) {
                ++g_android_jni_helper_init_depth;
            } else if (!table && !g_android_jni_helper_initializing.exchange(true, std::memory_order_acq_rel)) {
                g_android_jni_helper_init_depth = 1;
                LOGI("AndroidJNIHelper initialization owner acquired");
            } else if (g_android_jni_helper_initializing.load(std::memory_order_acquire)) {
                unsigned wait = 0;
                while (g_android_jni_helper_initializing.load(std::memory_order_acquire) && wait++ < 30000) {
                    usleep(1000);
                }
                if (g_android_jni_helper_initializing.load(std::memory_order_acquire)) {
                    LOGE("AndroidJNIHelper initialization wait timed out");
                }
            }
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 8;
            h.MemoryWrite32(new_sp, jit.Regs()[11]);
            h.MemoryWrite32(new_sp + 4, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            static std::atomic<unsigned> access_count{0};
            const unsigned access = access_count.fetch_add(1, std::memory_order_relaxed);
            if (access < 64 || (!table && (access & 0xFFu) == 0)) {
                LOGI("AndroidJNIHelper accessor global=0x%08X table=0x%08X count=%u",
                     cfg::kUnityBase + 0x112464C,
                     h.MemoryRead32(cfg::kUnityBase + 0x112464C), access + 1);
            }
            jit.Regs()[15] = kHelperAccessPc + 4;
        });
    const uint32_t return_hook = host.GetThunkManager().RegisterThunk(
        "guest_androidjni_helper_return", [kHelperReturnPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t lr = h.MemoryRead32(old_sp + 4);
            jit.Regs()[11] = h.MemoryRead32(old_sp);
            jit.Regs()[13] = old_sp + 8;
            bool initialization_complete = false;
            if (g_android_jni_helper_init_depth != 0 && --g_android_jni_helper_init_depth == 0) {
                g_android_jni_helper_initializing.store(false, std::memory_order_release);
                initialization_complete = true;
            }
            if (initialization_complete &&
                !g_android_jni_helper_ready.load(std::memory_order_acquire)) {
                uint32_t table = 0, create = 0, invoke = 0;
                if (ReadAndroidJniHelperReady(h, table, create, invoke) &&
                    !g_android_jni_helper_ready.exchange(true, std::memory_order_acq_rel)) {
                    LOGI("AndroidJNIHelper ready after return table=0x%08X create=0x%08X invoke=0x%08X",
                         table, create, invoke);
                    g_android_jni_helper_ready_cv.notify_all();
                }
            }
            jit.Regs()[15] = lr & ~1u;
            jit.SetCpsr((jit.Cpsr() & ~0x20u) | (lr & 1 ? 0x20u : 0u));
            (void)kHelperReturnPc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    const uint32_t return_svc = host.MemoryRead32(return_hook);
    if ((svc & 0xFF000000u) != 0xEF000000u ||
        (return_svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("AndroidJNIHelper access watch: invalid thunk");
        return false;
    }
    host.MemoryWrite32(kHelperAccessPc, svc);
    host.MemoryWrite32(kHelperReturnPc, return_svc);
    host.InvalidateGuestCode(kHelperAccessPc, sizeof(uint32_t));
    host.InvalidateGuestCode(kHelperReturnPc, sizeof(uint32_t));
    LOGI("AndroidJNIHelper access watch installed at 0x%08X/return 0x%08X",
         kHelperAccessPc, kHelperReturnPc);
    return true;
}

static bool InstallAndroidJniHelperResultWatches(DynarmicHost& host) {
    constexpr uint32_t kCreateResultPc = cfg::kUnityBase + 0x134F08;
    constexpr uint32_t kInvokeResultPc = cfg::kUnityBase + 0x134F28;
    constexpr uint32_t kCreateLdr = 0xE59F1F8C;
    constexpr uint32_t kInvokeLdr = 0xE59F1F70;
    if (host.MemoryRead32(kCreateResultPc) != kCreateLdr ||
        host.MemoryRead32(kInvokeResultPc) != kInvokeLdr) {
        LOGE("AndroidJNIHelper result watch: unexpected opcodes at 0x%08X/0x%08X",
             kCreateResultPc, kInvokeResultPc);
        return false;
    }
    const uint32_t create_hook = host.GetThunkManager().RegisterThunk(
        "guest_androidjni_helper_create_result", [kCreateResultPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            LOGI("AndroidJNIHelper CreateJavaProxy lookup result=0x%08X", jit.Regs()[0]);
            // The original LDR loads (string RVA - next PC), not an absolute address.
            jit.Regs()[1] = 0x00E5936E;
            jit.Regs()[15] = kCreateResultPc + 4;
        });
    const uint32_t invoke_hook = host.GetThunkManager().RegisterThunk(
        "guest_androidjni_helper_invoke_result", [kInvokeResultPc](Dynarmic::A32::Jit& jit, DynarmicHost&) {
            uint32_t result = jit.Regs()[0];
            if (!result) {
                const uint32_t replacement = g_android_jni_invoke_method.load(std::memory_order_acquire);
                if (replacement) {
                    result = replacement;
                    jit.Regs()[0] = result;
                    LOGI("AndroidJNIHelper lookup fallback to method=0x%08X", result);
                }
            }
            LOGI("AndroidJNIHelper InvokeJavaProxyMethod lookup result=0x%08X", result);
            // The original LDR loads (string RVA - next PC), not an absolute address.
            jit.Regs()[1] = 0x00E59368;
            jit.Regs()[15] = kInvokeResultPc + 4;
        });
    const uint32_t create_svc = host.MemoryRead32(create_hook);
    const uint32_t invoke_svc = host.MemoryRead32(invoke_hook);
    if ((create_svc & 0xFF000000u) != 0xEF000000u ||
        (invoke_svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("AndroidJNIHelper result watch: invalid thunk");
        return false;
    }
    host.MemoryWrite32(kCreateResultPc, create_svc);
    host.MemoryWrite32(kInvokeResultPc, invoke_svc);
    host.InvalidateGuestCode(cfg::kUnityBase + 0x134E80, 0x1530);
    LOGI("AndroidJNIHelper result watches installed at 0x%08X/0x%08X",
         kCreateResultPc, kInvokeResultPc);
    return true;
}

static bool InstallMethodNameWatch(DynarmicHost& host) {
    constexpr uint32_t kMethodNamePc = cfg::kIl2cppBase + 0x22CAEA0;
    constexpr uint32_t kExpectedLoad = 0xE5900008;
    if (host.MemoryRead32(kMethodNamePc) != kExpectedLoad) {
        LOGE("method-name watch: unexpected opcode at 0x%08X: 0x%08X",
             kMethodNamePc, host.MemoryRead32(kMethodNamePc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_method_name_watch", [kMethodNamePc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t method = jit.Regs()[0];
            const uint32_t name = GuestRangeValid(h.GetGuestMemorySize(), method, 12)
                ? h.MemoryRead32(method + 8) : 0;
            const auto text = [&h](uint32_t address) {
                if (!GuestRangeValid(h.GetGuestMemorySize(), address, 1)) return std::string("<invalid>");
                const char* value = reinterpret_cast<const char*>(h.GetGuestMemoryBase() + address);
                const size_t length = strnlen(value, h.GetGuestMemorySize() - address);
                return length < h.GetGuestMemorySize() - address ? std::string(value, length) : std::string("<invalid>");
            };
            const std::string method_text = text(name);
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (method_text == "InvokeJavaProxyMethod") {
                g_android_jni_invoke_method.store(method, std::memory_order_release);
            }
            if (call <= 32 || method_text == "InvokeJavaProxyMethod" || method_text == "CreateJavaProxy") {
                LOGI("guest method name call=%u method=0x%08X flags=0x%02X name=%s",
                     call, method, GuestRangeValid(h.GetGuestMemorySize(), method + 47, 1)
                         ? (h.MemoryRead32(method + 44) >> 24) & 0xFF : 0, method_text.c_str());
            }
            jit.Regs()[0] = name;
            const uint32_t lr = jit.Regs()[14];
            jit.Regs()[15] = lr & ~1u;
            jit.SetCpsr((jit.Cpsr() & ~0x20u) | (lr & 1 ? 0x20u : 0u));
            (void)kMethodNamePc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("method-name watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kMethodNamePc, svc);
    host.InvalidateGuestCode(cfg::kIl2cppBase + 0x22CAEA0, 8);
    LOGI("method-name watch installed at 0x%08X", kMethodNamePc);
    return true;
}

static bool InstallMethodFlagWatches(DynarmicHost& host) {
    constexpr uint32_t kGenericPc = cfg::kIl2cppBase + 0x22CAEA8;
    constexpr uint32_t kInflatedPc = cfg::kIl2cppBase + 0x22CAEB4;
    constexpr uint32_t kExpectedLdrb = 0xE5D0002F;
    if (host.MemoryRead32(kGenericPc) != kExpectedLdrb ||
        host.MemoryRead32(kInflatedPc) != kExpectedLdrb) {
        LOGE("method-flag watch: unexpected opcode at 0x%08X/0x%08X",
             kGenericPc, kInflatedPc);
        return false;
    }
    const uint32_t generic_hook = host.GetThunkManager().RegisterThunk(
        "guest_method_is_generic_watch", [kGenericPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t method = jit.Regs()[0];
            const uint32_t flags = GuestRangeValid(h.GetGuestMemorySize(), method, 48)
                ? (h.MemoryRead32(method + 44) >> 24) & 0xFF : 0;
            const uint32_t name = GuestRangeValid(h.GetGuestMemorySize(), method, 12)
                ? h.MemoryRead32(method + 8) : 0;
            const char* text = GuestRangeValid(h.GetGuestMemorySize(), name, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + name) : "<invalid>";
            const uint32_t result = flags & 1;
            if (strcmp(text, "InvokeJavaProxyMethod") == 0 || strcmp(text, "CreateJavaProxy") == 0) {
                LOGI("method_is_generic name=%s flags=0x%02X result=%u", text, flags, result);
            }
            jit.Regs()[0] = result;
            const uint32_t lr = jit.Regs()[14];
            jit.Regs()[15] = lr & ~1u;
            jit.SetCpsr((jit.Cpsr() & ~0x20u) | (lr & 1 ? 0x20u : 0u));
            (void)kGenericPc;
        });
    const uint32_t inflated_hook = host.GetThunkManager().RegisterThunk(
        "guest_method_is_inflated_watch", [kInflatedPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t method = jit.Regs()[0];
            const uint32_t flags = GuestRangeValid(h.GetGuestMemorySize(), method, 48)
                ? (h.MemoryRead32(method + 44) >> 24) & 0xFF : 0;
            const uint32_t name = GuestRangeValid(h.GetGuestMemorySize(), method, 12)
                ? h.MemoryRead32(method + 8) : 0;
            const char* text = GuestRangeValid(h.GetGuestMemorySize(), name, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + name) : "<invalid>";
            const uint32_t result = (flags >> 1) & 1;
            if (strcmp(text, "InvokeJavaProxyMethod") == 0 || strcmp(text, "CreateJavaProxy") == 0) {
                LOGI("method_is_inflated name=%s flags=0x%02X result=%u", text, flags, result);
            }
            jit.Regs()[0] = result;
            const uint32_t lr = jit.Regs()[14];
            jit.Regs()[15] = lr & ~1u;
            jit.SetCpsr((jit.Cpsr() & ~0x20u) | (lr & 1 ? 0x20u : 0u));
            (void)kInflatedPc;
        });
    const uint32_t generic_svc = host.MemoryRead32(generic_hook);
    const uint32_t inflated_svc = host.MemoryRead32(inflated_hook);
    if ((generic_svc & 0xFF000000u) != 0xEF000000u ||
        (inflated_svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("method-flag watch: invalid thunk");
        return false;
    }
    host.MemoryWrite32(kGenericPc, generic_svc);
    host.MemoryWrite32(kInflatedPc, inflated_svc);
    host.InvalidateGuestCode(kGenericPc, 8);
    host.InvalidateGuestCode(kInflatedPc, 8);
    LOGI("method-flag watches installed at 0x%08X/0x%08X", kGenericPc, kInflatedPc);
    return true;
}

static bool InstallMethodFilterWatches(DynarmicHost& host) {
    constexpr uint32_t kGenericBranchPc = cfg::kUnityBase + 0x87E6D4;
    constexpr uint32_t kInflatedBranchPc = cfg::kUnityBase + 0x87E6EC;
    constexpr uint32_t kExpectedBneGeneric = 0x1A000005;
    constexpr uint32_t kExpectedBneInflated = 0x1A000017;
    if (host.MemoryRead32(kGenericBranchPc) != kExpectedBneGeneric ||
        host.MemoryRead32(kInflatedBranchPc) != kExpectedBneInflated) {
        LOGE("method-filter watch: unexpected opcode at 0x%08X/0x%08X",
             kGenericBranchPc, kInflatedBranchPc);
        return false;
    }
    const auto log_filter = [](Dynarmic::A32::Jit& jit, DynarmicHost& h, const char* kind) {
        const uint32_t method = jit.Regs()[4];
        const uint32_t name = GuestRangeValid(h.GetGuestMemorySize(), method, 12)
            ? h.MemoryRead32(method + 8) : 0;
        const char* text = GuestRangeValid(h.GetGuestMemorySize(), name, 1)
            ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + name) : "<invalid>";
        if (strcmp(text, "InvokeJavaProxyMethod") == 0 || strcmp(text, "CreateJavaProxy") == 0) {
            LOGI("method filter %s name=%s value=%u branch=%s", kind, text, jit.Regs()[0],
                 (jit.Cpsr() & (1u << 30)) ? "not-taken" : "taken");
        }
    };
    const uint32_t generic_hook = host.GetThunkManager().RegisterThunk(
        "guest_method_generic_filter", [kGenericBranchPc, log_filter](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            log_filter(jit, h, "generic");
            jit.Regs()[15] = (jit.Cpsr() & (1u << 30)) ? kGenericBranchPc + 4 : cfg::kUnityBase + 0x87E6F0;
        });
    const uint32_t inflated_hook = host.GetThunkManager().RegisterThunk(
        "guest_method_inflated_filter", [kInflatedBranchPc, log_filter](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            log_filter(jit, h, "inflated");
            jit.Regs()[15] = (jit.Cpsr() & (1u << 30)) ? kInflatedBranchPc + 4 : cfg::kUnityBase + 0x87E750;
        });
    const uint32_t generic_svc = host.MemoryRead32(generic_hook);
    const uint32_t inflated_svc = host.MemoryRead32(inflated_hook);
    if ((generic_svc & 0xFF000000u) != 0xEF000000u ||
        (inflated_svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("method-filter watch: invalid thunk");
        return false;
    }
    host.MemoryWrite32(kGenericBranchPc, generic_svc);
    host.MemoryWrite32(kInflatedBranchPc, inflated_svc);
    host.InvalidateGuestCode(cfg::kUnityBase + 0x87E5EC, 0x1D8);
    LOGI("method-filter watches installed at 0x%08X/0x%08X", kGenericBranchPc, kInflatedBranchPc);
    return true;
}

static bool InstallMethodSearchResultWatch(DynarmicHost& host) {
    constexpr uint32_t kSearchResultPc = cfg::kUnityBase + 0x87E7B8;
    constexpr uint32_t kExpectedMov = 0xE1A00001;
    if (host.MemoryRead32(kSearchResultPc) != kExpectedMov) {
        LOGE("method-search result watch: unexpected opcode at 0x%08X: 0x%08X",
             kSearchResultPc, host.MemoryRead32(kSearchResultPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_method_search_result", [kSearchResultPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t result = jit.Regs()[1];
            const uint32_t method_name = jit.Regs()[8];
            const char* text = GuestRangeValid(h.GetGuestMemorySize(), method_name, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + method_name) : "<invalid>";
            if (strcmp(text, "CreateJavaProxy") == 0 || strcmp(text, "InvokeJavaProxyMethod") == 0) {
                LOGI("method search name=%s result=0x%08X", text, result);
            }
            jit.Regs()[0] = result;
            jit.Regs()[15] = kSearchResultPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("method-search result watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kSearchResultPc, svc);
    host.InvalidateGuestCode(cfg::kUnityBase + 0x87E5EC, 0x1D8);
    LOGI("method-search result watch installed at 0x%08X", kSearchResultPc);
    return true;
}

static bool InstallReflectionInvokeEntryWatch(DynarmicHost& host) {
    constexpr uint32_t kEntryPc = cfg::kUnityBase + 0x8796CC;
    constexpr uint32_t kExpectedPush = 0xE92D4070;
    if (host.MemoryRead32(kEntryPc) != kExpectedPush) {
        LOGE("reflection invoke-entry watch: unexpected opcode at 0x%08X: 0x%08X",
             kEntryPc, host.MemoryRead32(kEntryPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_reflection_invoke_entry", [kEntryPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t invocation = jit.Regs()[0];
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32) {
                LOGI("reflection invoke entry call=%u invocation=0x%08X method=0x%08X fallback=0x%08X object=0x%08X exc=0x%08X",
                     call, invocation,
                     GuestRangeValid(h.GetGuestMemorySize(), invocation + 200, 4) ? h.MemoryRead32(invocation + 200) : 0,
                     GuestRangeValid(h.GetGuestMemorySize(), invocation + 204, 4) ? h.MemoryRead32(invocation + 204) : 0,
                     GuestRangeValid(h.GetGuestMemorySize(), invocation + 208, 4) ? h.MemoryRead32(invocation + 208) : 0,
                     jit.Regs()[1]);
            }
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 16;
            h.MemoryWrite32(new_sp + 0, jit.Regs()[4]);
            h.MemoryWrite32(new_sp + 4, jit.Regs()[5]);
            h.MemoryWrite32(new_sp + 8, jit.Regs()[6]);
            h.MemoryWrite32(new_sp + 12, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[15] = kEntryPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("reflection invoke-entry watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kEntryPc, svc);
    host.InvalidateGuestCode(kEntryPc, sizeof(uint32_t));
    LOGI("reflection invoke-entry watch installed at 0x%08X", kEntryPc);
    return true;
}

static bool InstallReflectionInvokeDispatchWatch(DynarmicHost& host) {
    constexpr uint32_t kDispatchPc = cfg::kUnityBase + 0x87E7FC;
    constexpr uint32_t kExpectedPush = 0xE92D41F0;
    if (host.MemoryRead32(kDispatchPc) != kExpectedPush) {
        LOGE("reflection invoke-dispatch watch: unexpected opcode at 0x%08X: 0x%08X",
             kDispatchPc, host.MemoryRead32(kDispatchPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_reflection_invoke_dispatch", [kDispatchPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t method = jit.Regs()[0];
            const uint32_t name = GuestRangeValid(h.GetGuestMemorySize(), method, 12)
                ? h.MemoryRead32(method + 8) : 0;
            const char* text = GuestRangeValid(h.GetGuestMemorySize(), name, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + name) : "<invalid>";
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32 || strcmp(text, "InvokeJavaProxyMethod") == 0) {
                LOGI("reflection invoke dispatch call=%u method=0x%08X ptr=0x%08X name=%s object=0x%08X invocation=0x%08X exc=0x%08X",
                     call, method, GuestRangeValid(h.GetGuestMemorySize(), method, 4) ? h.MemoryRead32(method) : 0,
                     text, jit.Regs()[1], jit.Regs()[2], jit.Regs()[3]);
            }
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 24;
            for (unsigned i = 0; i < 5; ++i) h.MemoryWrite32(new_sp + i * 4, jit.Regs()[4 + i]);
            h.MemoryWrite32(new_sp + 20, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[15] = kDispatchPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("reflection invoke-dispatch watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kDispatchPc, svc);
    host.InvalidateGuestCode(kDispatchPc, sizeof(uint32_t));
    LOGI("reflection invoke-dispatch watch installed at 0x%08X", kDispatchPc);
    return true;
}

static bool InstallReflectionInvokeAfterParamWatch(DynarmicHost& host) {
    constexpr uint32_t kAfterParamPc = cfg::kUnityBase + 0x87E834;
    constexpr uint32_t kExpectedLdr = 0xE59D0020;
    if (host.MemoryRead32(kAfterParamPc) != kExpectedLdr) {
        LOGE("reflection invoke after-param watch: unexpected opcode at 0x%08X: 0x%08X",
             kAfterParamPc, host.MemoryRead32(kAfterParamPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_reflection_invoke_after_param", [kAfterParamPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32) {
                LOGI("reflection invoke after method-param callback call=%u callback_result=0x%08X method=0x%08X",
                     call, jit.Regs()[0], jit.Regs()[6]);
            }
            jit.Regs()[0] = h.MemoryRead32(jit.Regs()[13] + 0x20);
            jit.Regs()[15] = kAfterParamPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("reflection invoke after-param watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kAfterParamPc, svc);
    host.InvalidateGuestCode(kAfterParamPc, sizeof(uint32_t));
    LOGI("reflection invoke after-param watch installed at 0x%08X", kAfterParamPc);
    return true;
}

static bool InstallReflectionInvokeAfterReturnWatch(DynarmicHost& host) {
    constexpr uint32_t kAfterReturnPc = cfg::kUnityBase + 0x87E884;
    constexpr uint32_t kExpectedAdd = 0xE28DD008;
    if (host.MemoryRead32(kAfterReturnPc) != kExpectedAdd) {
        LOGE("reflection invoke after-return watch: unexpected opcode at 0x%08X: 0x%08X",
             kAfterReturnPc, host.MemoryRead32(kAfterReturnPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_reflection_invoke_after_return", [kAfterReturnPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t method = jit.Regs()[6];
            const uint32_t name = GuestRangeValid(h.GetGuestMemorySize(), method, 12)
                ? h.MemoryRead32(method + 8) : 0;
            const char* text = GuestRangeValid(h.GetGuestMemorySize(), name, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + name) : "<invalid>";
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32 || strcmp(text, "InvokeJavaProxyMethod") == 0) {
                LOGI("reflection invoke after return call=%u method=0x%08X name=%s result=0x%08X exc=0x%08X",
                     call, method, text, jit.Regs()[0], jit.Regs()[4]);
            }
            jit.Regs()[13] += 8;
            jit.Regs()[15] = kAfterReturnPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("reflection invoke after-return watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kAfterReturnPc, svc);
    host.InvalidateGuestCode(kAfterReturnPc, sizeof(uint32_t));
    LOGI("reflection invoke after-return watch installed at 0x%08X", kAfterReturnPc);
    return true;
}

static bool InstallIl2cppRuntimeInvokeWatch(DynarmicHost& host) {
    constexpr uint32_t kInvokePc = cfg::kIl2cppBase + 0x22AF0E4;
    constexpr uint32_t kExpectedLdr = 0xE590C00C;
    if (host.MemoryRead32(kInvokePc) != kExpectedLdr) {
        LOGE("il2cpp runtime-invoke watch: unexpected opcode at 0x%08X: 0x%08X",
             kInvokePc, host.MemoryRead32(kInvokePc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_il2cpp_runtime_invoke", [kInvokePc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t method = jit.Regs()[0];
            const uint32_t method_ptr = GuestRangeValid(h.GetGuestMemorySize(), method, 4)
                ? h.MemoryRead32(method) : 0;
            const uint32_t name = GuestRangeValid(h.GetGuestMemorySize(), method, 12)
                ? h.MemoryRead32(method + 8) : 0;
            const char* text = GuestRangeValid(h.GetGuestMemorySize(), name, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + name) : "<invalid>";
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32 || strcmp(text, "InvokeJavaProxyMethod") == 0) {
                LOGI("il2cpp_runtime_invoke call=%u method=0x%08X ptr=0x%08X name=%s object=0x%08X args=0x%08X exc=0x%08X klass=0x%08X",
                     call, method, method_ptr, text, jit.Regs()[1], jit.Regs()[2],
                     jit.Regs()[3], GuestRangeValid(h.GetGuestMemorySize(), method, 16)
                         ? h.MemoryRead32(method + 12) : 0);
            }
            jit.Regs()[12] = GuestRangeValid(h.GetGuestMemorySize(), method, 16)
                ? h.MemoryRead32(method + 12) : 0;
            jit.Regs()[15] = kInvokePc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("il2cpp runtime-invoke watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kInvokePc, svc);
    host.InvalidateGuestCode(kInvokePc, sizeof(uint32_t));
    LOGI("il2cpp runtime-invoke watch installed at 0x%08X", kInvokePc);
    return true;
}

static bool InstallAndroidJniMethodEntryWatch(DynarmicHost& host) {
    constexpr uint32_t kMethodPc = cfg::kIl2cppBase + 0x1D39B8C;
    constexpr uint32_t kExpectedPush = 0xE92D4FF0;
    if (host.MemoryRead32(kMethodPc) != kExpectedPush) {
        LOGE("AndroidJNI method-entry watch: unexpected opcode at 0x%08X: 0x%08X",
             kMethodPc, host.MemoryRead32(kMethodPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_androidjni_method_entry", [kMethodPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t helper_class = h.MemoryRead32(cfg::kIl2cppBase + 0x2A30D8C);
            const uint32_t helper_class_2 = h.MemoryRead32(cfg::kIl2cppBase + 0x2A30D90);
            const uint32_t helper_table = h.MemoryRead32(cfg::kUnityBase + 0x112464C);
            LOGI("AndroidJNI InvokeJavaProxyMethod entry r0=0x%08X r1=0x%08X r2=0x%08X r3=0x%08X class=0x%08X class2=0x%08X table=0x%08X byte=0x%02X lr=0x%08X",
                 jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3],
                 helper_class, helper_class_2, helper_table,
                 h.MemoryRead8(cfg::kIl2cppBase + 0x2A69346), jit.Regs()[14]);
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 36;
            for (unsigned i = 0; i < 8; ++i) h.MemoryWrite32(new_sp + i * 4, jit.Regs()[4 + i]);
            h.MemoryWrite32(new_sp + 32, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[11] = new_sp + 28;
            jit.Regs()[15] = kMethodPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("AndroidJNI method-entry watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kMethodPc, svc);
    host.InvalidateGuestCode(kMethodPc, sizeof(uint32_t));
    LOGI("AndroidJNI method-entry watch installed at 0x%08X", kMethodPc);
    return true;
}

static bool InstallAndroidJniMetadataWatch(DynarmicHost& host) {
    constexpr uint32_t kMetadataInitPc = cfg::kIl2cppBase + 0x1D39BD4;
    constexpr uint32_t kExpectedMov = 0xE3A00000;
    if (host.MemoryRead32(kMetadataInitPc) != kExpectedMov) {
        LOGE("AndroidJNI metadata watch: unexpected opcode at 0x%08X: 0x%08X",
             kMetadataInitPc, host.MemoryRead32(kMetadataInitPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_androidjni_metadata_init", [kMetadataInitPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t slot = cfg::kIl2cppBase + 0x2A30D8C;
            LOGI("AndroidJNI metadata after initialize_runtime_metadata slot=0x%08X value=0x%08X",
                 slot, h.MemoryRead32(slot));
            jit.Regs()[0] = 0;
            jit.Regs()[15] = kMetadataInitPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("AndroidJNI metadata watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kMetadataInitPc, svc);
    host.InvalidateGuestCode(cfg::kIl2cppBase + 0x1D39B8C, 0x270);
    LOGI("AndroidJNI metadata watch installed at 0x%08X", kMetadataInitPc);
    return true;
}

static bool InstallAndroidJniRuntimeMetadataWatch(DynarmicHost& host) {
    constexpr uint32_t kMetadataPc = cfg::kIl2cppBase + 0x1D39BC8;
    constexpr uint32_t kExpectedLdr = 0xE59F024C;
    if (host.MemoryRead32(kMetadataPc) != kExpectedLdr) {
        LOGE("AndroidJNI runtime metadata watch: unexpected opcode at 0x%08X: 0x%08X",
             kMetadataPc, host.MemoryRead32(kMetadataPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_androidjni_runtime_metadata", [kMetadataPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t slot = cfg::kIl2cppBase + 0x2A30D8C;
            LOGI("AndroidJNI runtime metadata slot=0x%08X value=0x%08X reg0=0x%08X reg4=0x%08X regBC=0x%08X",
                 slot, h.MemoryRead32(slot), h.MemoryRead32(cfg::kIl2cppBase + 0x2A6C4B0),
                 h.MemoryRead32(cfg::kIl2cppBase + 0x2A6C4BC), h.MemoryRead32(cfg::kIl2cppBase + 0x2A6C4C0));
            // The original LDR loads (byte flag RVA - next PC), not an absolute address.
            jit.Regs()[0] = 0x00D2F76E;
            jit.Regs()[15] = kMetadataPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("AndroidJNI runtime metadata watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kMetadataPc, svc);
    host.InvalidateGuestCode(cfg::kIl2cppBase + 0x1D39B8C, 0x270);
    LOGI("AndroidJNI runtime metadata watch installed at 0x%08X", kMetadataPc);
    return true;
}

static bool InstallAndroidJniProxyClassWatch(DynarmicHost& host) {
    constexpr uint32_t kProxyClassPc = cfg::kIl2cppBase + 0x1D39C0C;
    constexpr uint32_t kExpectedLoad = 0xE5904000;
    if (host.MemoryRead32(kProxyClassPc) != kExpectedLoad) {
        LOGE("AndroidJavaProxy class watch: unexpected opcode at 0x%08X: 0x%08X",
             kProxyClassPc, host.MemoryRead32(kProxyClassPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_android_proxy_class", [kProxyClassPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t slot = jit.Regs()[0];
            const uint32_t value = GuestRangeValid(h.GetGuestMemorySize(), slot, 4)
                ? h.MemoryRead32(slot) : 0;
            LOGI("AndroidJavaProxy class slot=0x%08X value=0x%08X", slot, value);
            jit.Regs()[4] = value;
            jit.Regs()[15] = kProxyClassPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("AndroidJavaProxy class watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kProxyClassPc, svc);
    host.InvalidateGuestCode(cfg::kIl2cppBase + 0x1D39B8C, 0x270);
    LOGI("AndroidJavaProxy class watch installed at 0x%08X", kProxyClassPc);
    return true;
}

static bool InstallMethodLookupResultWatch(DynarmicHost& host) {
    constexpr uint32_t kLookupResultPc = cfg::kUnityBase + 0x86D524;
    constexpr uint32_t kExpectedMov = 0xE1A08000;
    if (host.MemoryRead32(kLookupResultPc) != kExpectedMov) {
        LOGE("method lookup result watch: unexpected opcode at 0x%08X: 0x%08X",
             kLookupResultPc, host.MemoryRead32(kLookupResultPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_method_lookup_result", [kLookupResultPc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const auto text = [&h](uint32_t address) {
                if (!GuestRangeValid(h.GetGuestMemorySize(), address, 1)) return std::string("<invalid>");
                const char* value = reinterpret_cast<const char*>(h.GetGuestMemoryBase() + address);
                const size_t length = strnlen(value, h.GetGuestMemorySize() - address);
                return length < h.GetGuestMemorySize() - address ? std::string(value, length) : std::string("<invalid>");
            };
            const uint32_t result = jit.Regs()[0];
            const std::string method = text(jit.Regs()[5]);
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32 || method == "CreateJavaProxy" || method == "InvokeJavaProxyMethod") {
                LOGI("guest method lookup result=%u result=0x%08X class=%s.%s method=%s",
                     call, result, text(jit.Regs()[4]).c_str(), text(jit.Regs()[7]).c_str(), method.c_str());
            }
            jit.Regs()[8] = result;
            jit.Regs()[15] = kLookupResultPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("method lookup result watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kLookupResultPc, svc);
    host.InvalidateGuestCode(cfg::kUnityBase + 0x86D4F4, 0xD4);
    LOGI("method lookup result watch installed at 0x%08X", kLookupResultPc);
    return true;
}

static bool InstallMethodLookupWatch(DynarmicHost& host) {
    constexpr uint32_t kLookupPc = cfg::kUnityBase + 0x86D4F4;
    constexpr uint32_t kExpectedPush = 0xE92D41F0;
    if (host.MemoryRead32(kLookupPc) != kExpectedPush) {
        LOGE("method lookup watch: unexpected opcode at 0x%08X: 0x%08X",
             kLookupPc, host.MemoryRead32(kLookupPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_method_lookup_watch", [](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t assembly = jit.Regs()[0];
            const uint32_t name_space = jit.Regs()[1];
            const uint32_t class_name = jit.Regs()[2];
            const uint32_t method_name = jit.Regs()[3];
            const auto text = [&h](uint32_t address) {
                return GuestRangeValid(h.GetGuestMemorySize(), address, 1) ?
                    std::string(reinterpret_cast<const char*>(h.GetGuestMemoryBase() + address)) :
                    std::string("<invalid>");
            };
            const std::string method_text = text(method_name);
            static std::atomic<unsigned> lookup_count{0};
            const unsigned call = lookup_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32 || method_text == "CreateJavaProxy" ||
                method_text == "InvokeJavaProxyMethod") {
                LOGI("guest method lookup call=%u assembly=0x%08X class=%s.%s method=%s",
                     call, assembly, text(name_space).c_str(), text(class_name).c_str(),
                     method_text.c_str());
            }
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 24;
            h.MemoryWrite32(new_sp + 0, jit.Regs()[4]);
            h.MemoryWrite32(new_sp + 4, jit.Regs()[5]);
            h.MemoryWrite32(new_sp + 8, jit.Regs()[6]);
            h.MemoryWrite32(new_sp + 12, jit.Regs()[7]);
            h.MemoryWrite32(new_sp + 16, jit.Regs()[8]);
            h.MemoryWrite32(new_sp + 20, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[15] = cfg::kUnityBase + 0x86D4F4 + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("method lookup watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kLookupPc, svc);
    host.InvalidateGuestCode(kLookupPc, sizeof(uint32_t));
    LOGI("method lookup watch installed at 0x%08X", kLookupPc);
    return true;
}

static bool InstallUnityAssemblyConstructionWatches(DynarmicHost& host) {
    constexpr uint32_t kAssemblyCtorPc = cfg::kUnityBase + 0x876D00;
    constexpr uint32_t kAssemblyInitPc = cfg::kUnityBase + 0x875E94;
    constexpr uint32_t kExpectedPush = 0xE92D4070;
    if (host.MemoryRead32(kAssemblyCtorPc) != kExpectedPush ||
        host.MemoryRead32(kAssemblyInitPc) != kExpectedPush) {
        LOGE("Unity assembly construction watches: unexpected opcodes at 0x%08X/0x%08X: 0x%08X/0x%08X",
             kAssemblyCtorPc, kAssemblyInitPc, host.MemoryRead32(kAssemblyCtorPc),
             host.MemoryRead32(kAssemblyInitPc));
        return false;
    }
    const auto read_word = [](DynarmicHost& h, uint32_t address) {
        return address >= 0x1000 && GuestRangeValid(h.GetGuestMemorySize(), address, 4)
            ? h.MemoryRead32(address) : 0;
    };
    const uint32_t ctor_hook = host.GetThunkManager().RegisterThunk(
        "guest_unity_assembly_constructor", [kAssemblyCtorPc, read_word](Dynarmic::A32::Jit& jit,
                                                                            DynarmicHost& h) {
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            const uint32_t slots = cfg::kUnityBase + 0x1140EC0;
            LOGI("Unity assembly constructor call=%u r0=0x%08X r1=0x%08X r2=0x%08X r3=0x%08X slot5_before=0x%08X slot0=0x%08X slot1=0x%08X slot4=0x%08X slot5=0x%08X lr=0x%08X sp=0x%08X",
                 call, jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3],
                 read_word(h, slots + 20), read_word(h, slots), read_word(h, slots + 4),
                 read_word(h, slots + 16), read_word(h, slots + 20), jit.Regs()[14], jit.Regs()[13]);
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 16;
            h.MemoryWrite32(new_sp + 0, jit.Regs()[4]);
            h.MemoryWrite32(new_sp + 4, jit.Regs()[5]);
            h.MemoryWrite32(new_sp + 8, jit.Regs()[6]);
            h.MemoryWrite32(new_sp + 12, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[15] = kAssemblyCtorPc + 4;
        });
    const uint32_t init_hook = host.GetThunkManager().RegisterThunk(
        "guest_unity_assembly_initializer", [kAssemblyInitPc, read_word](Dynarmic::A32::Jit& jit,
                                                                            DynarmicHost& h) {
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            const uint32_t slots = cfg::kUnityBase + 0x1140EC0;
            LOGI("Unity assembly initializer call=%u object=0x%08X slot5_before=0x%08X object108=0x%08X object10c=0x%08X lr=0x%08X sp=0x%08X",
                 call, jit.Regs()[0], read_word(h, slots + 20), read_word(h, jit.Regs()[0] + 0x108),
                 read_word(h, jit.Regs()[0] + 0x10C), jit.Regs()[14], jit.Regs()[13]);
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 16;
            h.MemoryWrite32(new_sp + 0, jit.Regs()[4]);
            h.MemoryWrite32(new_sp + 4, jit.Regs()[5]);
            h.MemoryWrite32(new_sp + 8, jit.Regs()[6]);
            h.MemoryWrite32(new_sp + 12, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[15] = kAssemblyInitPc + 4;
        });
    const uint32_t ctor_svc = host.MemoryRead32(ctor_hook);
    const uint32_t init_svc = host.MemoryRead32(init_hook);
    if ((ctor_svc & 0xFF000000u) != 0xEF000000u ||
        (init_svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("Unity assembly construction watches: invalid thunk");
        return false;
    }
    host.MemoryWrite32(kAssemblyCtorPc, ctor_svc);
    host.MemoryWrite32(kAssemblyInitPc, init_svc);
    host.InvalidateGuestCode(kAssemblyCtorPc, sizeof(uint32_t));
    host.InvalidateGuestCode(kAssemblyInitPc, sizeof(uint32_t));
    LOGI("Unity assembly construction watches installed at 0x%08X/0x%08X",
         kAssemblyCtorPc, kAssemblyInitPc);
    return true;
}

static bool InstallUnityHashProbeWatches(DynarmicHost& host) {
    constexpr uint32_t kLookupEntryPc = cfg::kUnityBase + 0x876868;
    constexpr uint32_t kProbePc = cfg::kUnityBase + 0x87BA80;
    constexpr uint32_t kExpectedLookupPush = 0xE92D47F0;
    constexpr uint32_t kExpectedProbePush = 0xE92D4FF0;
    if (host.MemoryRead32(kLookupEntryPc) != kExpectedLookupPush ||
        host.MemoryRead32(kProbePc) != kExpectedProbePush) {
        LOGE("Unity hash-probe watches: unexpected opcodes at 0x%08X/0x%08X: 0x%08X/0x%08X",
             kLookupEntryPc, kProbePc, host.MemoryRead32(kLookupEntryPc),
             host.MemoryRead32(kProbePc));
        return false;
    }
    const auto read_word = [](DynarmicHost& h, uint32_t address) {
        return address >= 0x1000 && GuestRangeValid(h.GetGuestMemorySize(), address, 4)
            ? h.MemoryRead32(address) : 0;
    };
    const auto text = [](DynarmicHost& h, uint32_t address) {
        if (address < 0x1000 || !GuestRangeValid(h.GetGuestMemorySize(), address, 1))
            return std::string("<invalid>");
        const char* value = reinterpret_cast<const char*>(h.GetGuestMemoryBase() + address);
        const size_t limit = h.GetGuestMemorySize() - address;
        const size_t length = strnlen(value, limit);
        return length < limit ? std::string(value, length) : std::string("<invalid>");
    };
    const uint32_t lookup_hook = host.GetThunkManager().RegisterThunk(
        "guest_unity_assembly_lookup_entry", [kLookupEntryPc, read_word, text](Dynarmic::A32::Jit& jit,
                                                                                  DynarmicHost& h) {
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            const uint32_t original_assembly = jit.Regs()[0];
            const uint32_t name = jit.Regs()[1];
            const uint32_t slots = cfg::kUnityBase + 0x1140EC0;
            // Observe the original producer/consumer order without waiting or substituting state.
            const uint32_t assembly = original_assembly;
            constexpr unsigned waited_ms = 0;
            const uint32_t descriptor = assembly + 0x108;
            const uint32_t table = read_word(h, descriptor);
            const uint32_t mask = read_word(h, descriptor + 4);
            LOGE("Unity assembly lookup entry call=%u r0_before=0x%08X r0_after=0x%08X waited_ms=%u r1=0x%08X(%s) r2=0x%08X r3=0x%08X desc=0x%08X table=0x%08X mask=0x%08X fields_fc=0x%08X fields_108=0x%08X fields_10c=0x%08X lr=0x%08X sp=0x%08X",
                 call, original_assembly, assembly, waited_ms, name, text(h, name).c_str(), jit.Regs()[2],
                 jit.Regs()[3], descriptor, table, mask, read_word(h, assembly + 0xFC),
                 read_word(h, assembly + 0x108), read_word(h, assembly + 0x10C), jit.Regs()[14], jit.Regs()[13]);
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 32;
            for (unsigned i = 0; i < 7; ++i) h.MemoryWrite32(new_sp + i * 4, jit.Regs()[4 + i]);
            h.MemoryWrite32(new_sp + 28, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[15] = kLookupEntryPc + 4;
        });
    const uint32_t probe_hook = host.GetThunkManager().RegisterThunk(
        "guest_unity_hash_probe_entry", [kProbePc, read_word, text](Dynarmic::A32::Jit& jit,
                                                                       DynarmicHost& h) {
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            const uint32_t descriptor = jit.Regs()[0];
            const uint32_t key = jit.Regs()[1];
            const uint32_t key_ptr = read_word(h, key);
            const uint32_t table = read_word(h, descriptor);
            const uint32_t mask = read_word(h, descriptor + 4);
            const bool interesting = !table || call <= 32;
            if (interesting) {
                LOGE("Unity hash probe entry call=%u r0=0x%08X table=0x%08X mask=0x%08X r1=0x%08X key0=0x%08X key4=0x%08X key8=0x%08X keyc=0x%08X key10=0x%08X key14=0x%08X key_text=%s lr=0x%08X sp=0x%08X",
                     call, descriptor, table, mask, key, read_word(h, key), read_word(h, key + 4),
                     read_word(h, key + 8), read_word(h, key + 12), read_word(h, key + 16),
                     read_word(h, key + 20), text(h, key_ptr).c_str(), jit.Regs()[14], jit.Regs()[13]);
            }
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 36;
            for (unsigned i = 0; i < 8; ++i) h.MemoryWrite32(new_sp + i * 4, jit.Regs()[4 + i]);
            h.MemoryWrite32(new_sp + 32, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[15] = kProbePc + 4;
        });
    const uint32_t lookup_svc = host.MemoryRead32(lookup_hook);
    const uint32_t probe_svc = host.MemoryRead32(probe_hook);
    if ((lookup_svc & 0xFF000000u) != 0xEF000000u ||
        (probe_svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("Unity hash-probe watches: invalid thunk");
        return false;
    }
    host.MemoryWrite32(kLookupEntryPc, lookup_svc);
    host.MemoryWrite32(kProbePc, probe_svc);
    host.InvalidateGuestCode(kLookupEntryPc, sizeof(uint32_t));
    host.InvalidateGuestCode(kProbePc, sizeof(uint32_t));
    LOGI("Unity hash-probe watches installed at 0x%08X/0x%08X", kLookupEntryPc, kProbePc);
    return true;
}

static bool InstallDefaultExecutionOrderWatches(DynarmicHost& host) {
    constexpr uint32_t kDefaultOrderPc = cfg::kIl2cppBase + 0x1D4E1A0;
    constexpr uint32_t kGenericHelperPc = cfg::kIl2cppBase + 0x15CF980;
    constexpr uint32_t kExpectedDefaultPush = 0xE92D4830;
    constexpr uint32_t kExpectedHelperPush = 0xE92D48F0;
    if (host.MemoryRead32(kDefaultOrderPc) != kExpectedDefaultPush ||
        host.MemoryRead32(kGenericHelperPc) != kExpectedHelperPush) {
        LOGE("GetDefaultExecutionOrder watches: unexpected opcodes at 0x%08X/0x%08X: 0x%08X/0x%08X",
             kDefaultOrderPc, kGenericHelperPc, host.MemoryRead32(kDefaultOrderPc),
             host.MemoryRead32(kGenericHelperPc));
        return false;
    }
    const auto read_word = [](DynarmicHost& h, uint32_t address) {
        return address >= 0x1000 && GuestRangeValid(h.GetGuestMemorySize(), address, 4)
            ? h.MemoryRead32(address) : 0;
    };
    const uint32_t default_hook = host.GetThunkManager().RegisterThunk(
        "guest_default_execution_order_entry", [kDefaultOrderPc, read_word](Dynarmic::A32::Jit& jit,
                                                                               DynarmicHost& h) {
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            const uint32_t slot_address = cfg::kIl2cppBase + 0x2A39E2C;
            const uint32_t method = read_word(h, slot_address);
            const uint32_t rgctx = read_word(h, method + 24);
            LOGE("GetDefaultExecutionOrderFor entry call=%u r0=0x%08X r1=0x%08X r2=0x%08X r3=0x%08X method_slot=0x%08X method=0x%08X rgctx=0x%08X rgctx0=0x%08X r1_0=0x%08X r1_8=0x%08X lr=0x%08X sp=0x%08X",
                 call, jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3], slot_address,
                 method, rgctx, read_word(h, rgctx), read_word(h, jit.Regs()[1]),
                 read_word(h, jit.Regs()[1] + 8), jit.Regs()[14], jit.Regs()[13]);
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 16;
            h.MemoryWrite32(new_sp + 0, jit.Regs()[4]);
            h.MemoryWrite32(new_sp + 4, jit.Regs()[5]);
            h.MemoryWrite32(new_sp + 8, jit.Regs()[11]);
            h.MemoryWrite32(new_sp + 12, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[15] = kDefaultOrderPc + 4;
        });
    const uint32_t helper_hook = host.GetThunkManager().RegisterThunk(
        "guest_default_execution_order_helper", [kGenericHelperPc, read_word](Dynarmic::A32::Jit& jit,
                                                                                 DynarmicHost& h) {
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            const uint32_t method = jit.Regs()[2];
            LOGE("GetDefaultExecutionOrderFor helper call=%u r0=0x%08X r1=0x%08X r2=0x%08X slot=0x%08X method24=0x%08X method28=0x%08X method2c=0x%08X method30=0x%08X method34=0x%08X lr=0x%08X sp=0x%08X",
                 call, jit.Regs()[0], jit.Regs()[1], method, cfg::kIl2cppBase + 0x2A39E2C,
                 read_word(h, method + 24), read_word(h, method + 28), read_word(h, method + 44),
                 read_word(h, method + 48), read_word(h, method + 52), jit.Regs()[14], jit.Regs()[13]);
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 24;
            for (unsigned i = 0; i < 4; ++i) h.MemoryWrite32(new_sp + i * 4, jit.Regs()[4 + i]);
            h.MemoryWrite32(new_sp + 16, jit.Regs()[11]);
            h.MemoryWrite32(new_sp + 20, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[15] = kGenericHelperPc + 4;
        });
    const uint32_t default_svc = host.MemoryRead32(default_hook);
    const uint32_t helper_svc = host.MemoryRead32(helper_hook);
    if ((default_svc & 0xFF000000u) != 0xEF000000u ||
        (helper_svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("GetDefaultExecutionOrder watches: invalid thunk");
        return false;
    }
    host.MemoryWrite32(kDefaultOrderPc, default_svc);
    host.MemoryWrite32(kGenericHelperPc, helper_svc);
    host.InvalidateGuestCode(kDefaultOrderPc, sizeof(uint32_t));
    host.InvalidateGuestCode(kGenericHelperPc, sizeof(uint32_t));
    LOGI("GetDefaultExecutionOrder watches installed at 0x%08X/0x%08X",
         kDefaultOrderPc, kGenericHelperPc);
    return true;
}

static bool InstallReflectionBuilderMethodWatch(DynarmicHost& host) {
    constexpr uint32_t kHelperMethodPc = cfg::kUnityBase + 0x107BE0;
    constexpr uint32_t kExpectedLoad = 0xE5901008;
    if (host.MemoryRead32(kHelperMethodPc) != kExpectedLoad) {
        LOGE("reflection builder helper watch: unexpected opcode at 0x%08X: 0x%08X",
             kHelperMethodPc, host.MemoryRead32(kHelperMethodPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_reflection_builder_helper_method", [kHelperMethodPc](Dynarmic::A32::Jit& jit,
                                                                      DynarmicHost& h) {
            uint32_t table = jit.Regs()[0];
            uint32_t invoke = GuestRangeValid(h.GetGuestMemorySize(), table, 12)
                ? h.MemoryRead32(table + 8) : 0;
            if (!invoke) {
                const uint32_t retry = h.CallGuestNested(cfg::kUnityBase + 0x134E28, {});
                table = retry;
                invoke = GuestRangeValid(h.GetGuestMemorySize(), table, 12)
                    ? h.MemoryRead32(table + 8) : 0;
                LOGI("reflection builder helper retry table=0x%08X invoke=0x%08X", table, invoke);
            }
            LOGI("reflection builder helper table=0x%08X fields0=0x%08X fields4=0x%08X invoke=0x%08X fieldsC=0x%08X",
                 table, GuestRangeValid(h.GetGuestMemorySize(), table, 4) ? h.MemoryRead32(table) : 0,
                 GuestRangeValid(h.GetGuestMemorySize(), table + 4, 4) ? h.MemoryRead32(table + 4) : 0,
                 invoke, GuestRangeValid(h.GetGuestMemorySize(), table + 12, 4) ? h.MemoryRead32(table + 12) : 0);
            jit.Regs()[1] = invoke;
            jit.Regs()[15] = kHelperMethodPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("reflection builder helper watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kHelperMethodPc, svc);
    host.InvalidateGuestCode(kHelperMethodPc, sizeof(uint32_t));
    LOGI("reflection builder helper watch installed at 0x%08X", kHelperMethodPc);
    return true;
}

static bool InstallInvokeObjectWatch(DynarmicHost& host) {
    constexpr uint32_t kInvokePc = cfg::kIl2cppBase + 0x229F094;
    constexpr uint32_t kExpectedPush = 0xE92D4C70;
    if (host.MemoryRead32(kInvokePc) != kExpectedPush) {
        LOGE("invoke-object watch: unexpected opcode at 0x%08X: 0x%08X",
             kInvokePc, host.MemoryRead32(kInvokePc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_invoke_object_watch", [](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t object = jit.Regs()[0];
            const uint32_t old_sp = jit.Regs()[13];
            if (!object) {
                LOGE("guest reflection Invoke received null object caller=0x%08X sp=0x%08X",
                     jit.Regs()[14], old_sp);
                jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
                return;
            }
            const uint32_t new_sp = old_sp - 24;
            h.MemoryWrite32(new_sp + 0, jit.Regs()[4]);
            h.MemoryWrite32(new_sp + 4, jit.Regs()[5]);
            h.MemoryWrite32(new_sp + 8, jit.Regs()[6]);
            h.MemoryWrite32(new_sp + 12, jit.Regs()[10]);
            h.MemoryWrite32(new_sp + 16, jit.Regs()[11]);
            h.MemoryWrite32(new_sp + 20, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[11] = new_sp + 16;
            jit.Regs()[6] = object;
            jit.Regs()[5] = jit.Regs()[1];
            jit.Regs()[4] = jit.Regs()[2];
            jit.Regs()[15] = kInvokePc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("invoke-object watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kInvokePc, svc);
    host.InvalidateGuestCode(kInvokePc, sizeof(uint32_t));
    LOGI("invoke-object watch installed at 0x%08X", kInvokePc);
    return true;
}

static bool InstallAotMethodWatch(DynarmicHost& host) {
    constexpr uint32_t kAotPc = cfg::kIl2cppBase + 0x229F200;
    constexpr uint32_t kExpectedPush = 0xE92D4C10;
    if (host.MemoryRead32(kAotPc) != kExpectedPush) {
        LOGE("AOT method watch: unexpected opcode at 0x%08X: 0x%08X",
             kAotPc, host.MemoryRead32(kAotPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_aot_method_watch", [](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            constexpr uint32_t kAotPc = cfg::kIl2cppBase + 0x229F200;
            const uint32_t method = jit.Regs()[0];
            const uint32_t method_ptr = GuestRangeValid(h.GetGuestMemorySize(), method, 4)
                ? h.MemoryRead32(method) : 0;
            const uint32_t method_name = GuestRangeValid(h.GetGuestMemorySize(), method, 12)
                ? h.MemoryRead32(method + 8) : 0;
            const char* method_text = GuestRangeValid(h.GetGuestMemorySize(), method_name, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + method_name) : "<invalid>";
            if (method_ptr && (strcmp(method_text, "InvokeJavaProxyMethod") == 0 ||
                               strcmp(method_text, ".cctor") == 0)) {
                LOGI("guest AOT check method=0x%08X ptr=0x%08X name=%s invoker=0x%08X klass=0x%08X caller=0x%08X",
                     method, method_ptr, method_text, h.MemoryRead32(method + 4),
                     h.MemoryRead32(method + 12), jit.Regs()[14]);
            }
            if (method_ptr != 0) {
                const uint32_t old_sp = jit.Regs()[13];
                const uint32_t new_sp = old_sp - 16;
                h.MemoryWrite32(new_sp + 0, jit.Regs()[4]);
                h.MemoryWrite32(new_sp + 4, jit.Regs()[10]);
                h.MemoryWrite32(new_sp + 8, jit.Regs()[11]);
                h.MemoryWrite32(new_sp + 12, jit.Regs()[14]);
                jit.Regs()[13] = new_sp;
                jit.Regs()[15] = kAotPc + 4;
                return;
            }
            const uint32_t invoker = GuestRangeValid(h.GetGuestMemorySize(), method, 8)
                ? h.MemoryRead32(method + 4) : 0;
            const uint32_t name = GuestRangeValid(h.GetGuestMemorySize(), method, 12)
                ? h.MemoryRead32(method + 8) : 0;
            const uint32_t klass = GuestRangeValid(h.GetGuestMemorySize(), method, 16)
                ? h.MemoryRead32(method + 12) : 0;
            const uint32_t class_name = GuestRangeValid(h.GetGuestMemorySize(), klass, 12)
                ? h.MemoryRead32(klass + 8) : 0;
            const uint32_t class_space = GuestRangeValid(h.GetGuestMemorySize(), klass, 16)
                ? h.MemoryRead32(klass + 12) : 0;
            const char* text = GuestRangeValid(h.GetGuestMemorySize(), name, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + name) : "<invalid>";
            const char* name_text = GuestRangeValid(h.GetGuestMemorySize(), class_name, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + class_name) : "<invalid>";
            const char* space_text = GuestRangeValid(h.GetGuestMemorySize(), class_space, 1)
                ? reinterpret_cast<const char*>(h.GetGuestMemoryBase() + class_space) : "<invalid>";
            const uint32_t helper_table = h.MemoryRead32(cfg::kUnityBase + 0x112464C);
            const uint32_t helper_create = GuestRangeValid(h.GetGuestMemorySize(), helper_table, 12)
                ? h.MemoryRead32(helper_table + 4) : 0;
            const uint32_t helper_invoke = GuestRangeValid(h.GetGuestMemorySize(), helper_table, 12)
                ? h.MemoryRead32(helper_table + 8) : 0;
            LOGE("guest missing AOT method=0x%08X ptr=0x%08X name=%s klass=0x%08X class=%s.%s invoker=0x%08X methoddef=0x%08X generic=0x%08X custom=0x%08X token=0x%08X flags=0x%04X iflags=0x%04X slot=%u params=%u bits=0x%02X helper_table=0x%08X helper_create=0x%08X helper_invoke=0x%08X caller=0x%08X outer_lr=0x%08X sp=0x%08X",
                 method, method_ptr, text, klass, space_text, name_text, invoker,
                 h.MemoryRead32(method + 24), h.MemoryRead32(method + 28), h.MemoryRead32(method + 32),
                 h.MemoryRead32(method + 36), h.MemoryRead32(method + 40) & 0xFFFF,
                 h.MemoryRead32(method + 42) & 0xFFFF, h.MemoryRead32(method + 44) & 0xFFFF,
                 h.MemoryRead32(method + 46) & 0xFF, h.MemoryRead32(method + 47) & 0xFF,
                 helper_table, helper_create, helper_invoke, jit.Regs()[14],
                 h.MemoryRead32(jit.Regs()[13] + 28), jit.Regs()[13]);
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("AOT method watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kAotPc, svc);
    host.InvalidateGuestCode(kAotPc, sizeof(uint32_t));
    LOGI("AOT method watch installed at 0x%08X", kAotPc);
    return true;
}

static bool InstallActivatorCreateInstanceWatch(DynarmicHost& host) {
    constexpr uint32_t kEntryPc = cfg::kIl2cppBase + 0x1872AD0;
    constexpr uint32_t kHookPc = kEntryPc + 0x0C;
    constexpr uint32_t kLiteralPc = cfg::kIl2cppBase + 0x1872DF4;
    constexpr uint32_t kExpectedLiteral = 0x011F3A9B;
    const std::array<uint32_t, 6> addresses = {
        kEntryPc, kEntryPc + 4, kEntryPc + 8, kHookPc, kEntryPc + 0x10, kEntryPc + 0x14};
    const std::array<uint32_t, 6> expected = {
        0xE92D4C70, 0xE28DB010, 0xE24DD010, 0xE59F0310, 0xE1A05002, 0xE1A04001};
    for (size_t i = 0; i < addresses.size(); ++i) {
        uint32_t actual = 0;
        if (!ReadActivatorWord(host, addresses[i], 0, actual) || actual != expected[i]) {
            LOGE("activator watch: unexpected opcode at 0x%08X: got 0x%08X expected 0x%08X",
                 addresses[i], actual, expected[i]);
            return false;
        }
    }
    uint32_t literal = 0;
    if (!ReadActivatorWord(host, kLiteralPc, 0, literal) || literal != kExpectedLiteral) {
        LOGE("activator watch: unexpected literal at 0x%08X: got 0x%08X expected 0x%08X",
             kLiteralPc, literal, kExpectedLiteral);
        return false;
    }

    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_activator_create_instance_watch", [kHookPc, kLiteralPc](Dynarmic::A32::Jit& jit,
                                                                         DynarmicHost& h) {
            constexpr uint32_t kExpectedLiteral = 0x011F3A9B;
            ActivatorDiagSnapshot snapshot{};
            static std::atomic<unsigned> count{0};
            snapshot.call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            for (unsigned i = 0; i < 4; ++i) snapshot.raw[i] = jit.Regs()[i];
            snapshot.type_arg = snapshot.raw[1];
            snapshot.non_public = snapshot.raw[2];
            snapshot.pre_lr = jit.Regs()[14];
            snapshot.frame_sp = jit.Regs()[13];
            snapshot.cpsr = jit.Cpsr();
            snapshot.host_tid = static_cast<uint32_t>(gettid());
            snapshot.guest_tid = DynarmicHost::CurrentGuestThreadId();
            (void)ReadActivatorWord(h, snapshot.frame_sp, 36, snapshot.saved_lr);

            snapshot.stack_valid = snapshot.frame_sp >= 0x1000 &&
                                   GuestRangeValid(h.GetGuestMemorySize(), snapshot.frame_sp,
                                                   sizeof(snapshot.stack));
            if (snapshot.stack_valid) {
                for (size_t i = 0; i < snapshot.stack.size(); ++i) {
                    if (!ReadActivatorWord(h, snapshot.frame_sp, i * sizeof(uint32_t),
                                            snapshot.stack[i])) {
                        snapshot.stack_valid = false;
                        break;
                    }
                }
            }

            snapshot.type_valid = snapshot.type_arg >= 0x1000 &&
                                  GuestRangeValid(h.GetGuestMemorySize(), snapshot.type_arg,
                                                  sizeof(snapshot.type_words));
            if (snapshot.type_valid) {
                for (size_t i = 0; i < snapshot.type_words.size(); ++i) {
                    if (!ReadActivatorWord(h, snapshot.type_arg, i * sizeof(uint32_t),
                                            snapshot.type_words[i])) {
                        snapshot.type_valid = false;
                        break;
                    }
                }
                DecodeActivatorType(h, snapshot);
            }
            g_activator_diag_ring.Push(snapshot);

            const bool invalid_type = !snapshot.type_valid || !snapshot.type_arg;
            if (snapshot.call <= 32 || (snapshot.call & 0xFFu) == 0 || invalid_type ||
                snapshot.type_data == 0x250D) {
                LogActivatorSnapshot(h, snapshot, false);
            }

            uint32_t original_literal = 0;
            if (!ReadActivatorWord(h, kLiteralPc, 0, original_literal) ||
                original_literal != kExpectedLiteral) {
                LOGE("activator watch: literal changed/unreadable at 0x%08X: got 0x%08X",
                     kLiteralPc, original_literal);
                jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
                return;
            }
            // The replaced instruction is `ldr r0, [pc, #0x310]` in ARM state.
            jit.Regs()[0] = original_literal;
            jit.Regs()[15] = kHookPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("activator watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kHookPc, svc);
    host.InvalidateGuestCode(kHookPc, sizeof(uint32_t));
    LOGI("activator CreateInstance watch installed at 0x%08X", kHookPc);
    return true;
}

static bool InstallActivatorValueTypeWatch(DynarmicHost& host) {
    constexpr uint32_t kTypeMethodPc = cfg::kIl2cppBase + 0x1CBCC1C;
    constexpr uint32_t kTypeResultPc = cfg::kIl2cppBase + 0x1872C68;
    constexpr uint32_t kExpectedLoadClass = 0xE5901000;
    constexpr uint32_t kExpectedLoadMethod = 0xE591234C;
    constexpr uint32_t kExpectedLoadInfo = 0xE5911350;
    constexpr uint32_t kExpectedBx = 0xE12FFF12;
    constexpr uint32_t kExpectedCmp = 0xE3500001;
    if (host.MemoryRead32(kTypeMethodPc) != kExpectedLoadClass ||
        host.MemoryRead32(kTypeMethodPc + 4) != kExpectedLoadMethod ||
        host.MemoryRead32(kTypeMethodPc + 8) != kExpectedLoadInfo ||
        host.MemoryRead32(kTypeMethodPc + 12) != kExpectedBx ||
        host.MemoryRead32(kTypeResultPc) != kExpectedCmp) {
        LOGE("Activator value-type watch: unexpected opcode sequence");
        return false;
    }

    const uint32_t entry_hook = host.GetThunkManager().RegisterThunk(
        "guest_activator_isvaluetype_entry", [](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            ActivatorValueTypeDiagSnapshot& snapshot = g_activator_value_type_diag;
            static std::atomic<unsigned> count{0};
            snapshot = {};
            snapshot.call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            snapshot.type_object = jit.Regs()[0];
            if (GuestRangeValid(h.GetGuestMemorySize(), snapshot.type_object, 12)) {
                snapshot.object_klass = h.MemoryRead32(snapshot.type_object);
                snapshot.type_handle = h.MemoryRead32(snapshot.type_object + 8);
            }
            if (GuestRangeValid(h.GetGuestMemorySize(), snapshot.type_handle, 8)) {
                snapshot.type_data = h.MemoryRead32(snapshot.type_handle);
                snapshot.type_bits = h.MemoryRead32(snapshot.type_handle + 4);
            }
            if (GuestRangeValid(h.GetGuestMemorySize(), snapshot.object_klass + 0x350, 4)) {
                snapshot.target = h.MemoryRead32(snapshot.object_klass + 0x34C);
                snapshot.method_info = h.MemoryRead32(snapshot.object_klass + 0x350);
            }
            if (snapshot.call <= 64 || snapshot.type_data == 0x2489 || !snapshot.target) {
                LOGI("activator IsValueType dispatch call=%u type=0x%08X klass=0x%08X il2cpp_type=0x%08X data=0x%08X bits=0x%08X target=0x%08X methodInfo=0x%08X",
                     snapshot.call, snapshot.type_object, snapshot.object_klass, snapshot.type_handle,
                     snapshot.type_data, snapshot.type_bits, snapshot.target, snapshot.method_info);
            }
            if (!snapshot.target ||
                !GuestRangeValid(h.GetGuestMemorySize(), snapshot.target & ~1u, 4)) {
                LOGE("activator IsValueType dispatch invalid target type=0x%08X target=0x%08X",
                     snapshot.type_object, snapshot.target);
                jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
                return;
            }
            jit.Regs()[1] = snapshot.method_info;
            jit.Regs()[15] = snapshot.target & ~1u;
            jit.SetCpsr((jit.Cpsr() & ~0x20u) | (snapshot.target & 1 ? 0x20u : 0u));
        });
    const uint32_t result_hook = host.GetThunkManager().RegisterThunk(
        "guest_activator_isvaluetype_result", [kTypeResultPc](Dynarmic::A32::Jit& jit, DynarmicHost&) {
            ActivatorValueTypeDiagSnapshot& snapshot = g_activator_value_type_diag;
            const uint32_t result = jit.Regs()[0];
            if (snapshot.call <= 64 || snapshot.type_data == 0x2489 || result != 1) {
                LOGI("activator IsValueType result call=%u type=0x%08X data=0x%08X bits=0x%08X target=0x%08X result=%u",
                     snapshot.call, snapshot.type_object, snapshot.type_data, snapshot.type_bits,
                     snapshot.target, result);
            }
            const uint32_t diff = result - 1u;
            uint32_t cpsr = jit.Cpsr() & ~(0xFu << 28);
            if (diff & 0x80000000u) cpsr |= 1u << 31;
            if (diff == 0) cpsr |= 1u << 30;
            if (result >= 1) cpsr |= 1u << 29;
            if (((result ^ 1u) & (result ^ diff) & 0x80000000u) != 0) cpsr |= 1u << 28;
            jit.SetCpsr(cpsr);
            jit.Regs()[15] = kTypeResultPc + 4;
        });
    const uint32_t entry_svc = host.MemoryRead32(entry_hook);
    const uint32_t result_svc = host.MemoryRead32(result_hook);
    if ((entry_svc & 0xFF000000u) != 0xEF000000u ||
        (result_svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("Activator value-type watch: invalid thunk");
        return false;
    }
    host.MemoryWrite32(kTypeMethodPc, entry_svc);
    host.MemoryWrite32(kTypeResultPc, result_svc);
    host.InvalidateGuestCode(kTypeMethodPc, sizeof(uint32_t));
    host.InvalidateGuestCode(kTypeResultPc, sizeof(uint32_t));
    LOGI("Activator value-type watch installed at 0x%08X/0x%08X",
         kTypeMethodPc, kTypeResultPc);
    return true;
}

static bool InstallConstructorMetadataWatch(DynarmicHost& host) {
    constexpr uint32_t kWatchPc = cfg::kIl2cppBase + 0x151C074;
    constexpr uint32_t kResumePc = cfg::kIl2cppBase + 0x151C078;
    constexpr uint32_t kExpectedMov = 0xE1A06000;
    constexpr uint32_t kExpectedAdd = 0xE2869010;
    constexpr size_t kMaxConstructors = 32;
    if (host.MemoryRead32(kWatchPc) != kExpectedMov ||
        host.MemoryRead32(kResumePc) != kExpectedAdd) {
        LOGE("constructor metadata watch: unexpected opcodes at 0x%08X/0x%08X: "
             "0x%08X/0x%08X", kWatchPc, kResumePc, host.MemoryRead32(kWatchPc),
             host.MemoryRead32(kResumePc));
        return false;
    }

    const auto read_word = [](DynarmicHost& h, uint32_t base, size_t offset,
                              uint32_t& value) {
        return ReadActivatorWord(h, base, offset, value);
    };
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_mono_type_constructor_metadata", [kWatchPc, kResumePc, read_word,
                                                  kMaxConstructors](Dynarmic::A32::Jit& jit,
                                                                    DynarmicHost& h) {
            const uint32_t type_object = jit.Regs()[4];
            const uint32_t constructor_array = jit.Regs()[0];
            uint32_t type_info = 0;
            uint32_t full_name_object = 0;
            uint32_t default_constructor = 0;
            uint32_t type_handle = 0;
            uint32_t type_data = 0;
            uint32_t type_bits = 0;
            const bool type_info_valid = read_word(h, type_object, 0xC, type_info);
            const bool type_handle_valid = read_word(h, type_object, 0x8, type_handle);
            const bool type_data_valid = read_word(h, type_handle, 0, type_data);
            const bool type_bits_valid = read_word(h, type_handle, 4, type_bits);
            const bool target_type = type_data == 0x1A09 || type_data == 0x2489;
            bool full_name_field_valid = false;
            bool default_constructor_field_valid = false;
            if (target_type) {
                full_name_field_valid = read_word(h, type_info, 0x8, full_name_object);
                default_constructor_field_valid = read_word(h, type_info, 0xC, default_constructor);
            }
            char full_name[kActivatorDiagTextSize] = {};
            const bool full_name_valid = target_type && CopyActivatorManagedText(
                h, full_name_object, full_name, sizeof(full_name));

            uint32_t constructor_count = 0;
            const bool constructor_array_valid =
                constructor_array >= 0x1000 &&
                GuestRangeValid(h.GetGuestMemorySize(), constructor_array, 16);
            const bool constructor_count_valid =
                constructor_array_valid && read_word(h, constructor_array, 0xC,
                                                      constructor_count);
            const uint32_t binding_flags = jit.Regs()[5];
            const uint32_t call_convention = jit.Regs()[7];
            uint32_t types = 0;
            uint32_t modifiers = 0;
            uint32_t method_info = 0;
            if (target_type) {
                read_word(h, jit.Regs()[11], 0x8, types);
                read_word(h, jit.Regs()[11], 0xC, modifiers);
                read_word(h, jit.Regs()[11], 0x10, method_info);
            }

            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            const bool known_type = target_type;
            const bool log_summary = known_type;
            if (log_summary) {
                LOGI("constructor metadata diag call=%u type=0x%08X type_info=0x%08X "
                     "type_info_valid=%d full_name_obj=0x%08X full_name_valid=%d "
                     "full_name=%s default_ctor=0x%08X default_ctor_field_valid=%d "
                     "type_handle=0x%08X type_handle_valid=%d data=0x%08X "
                     "data_valid=%d bits=0x%08X bits_valid=%d array=0x%08X "
                     "array_valid=%d count=%u count_valid=%d binding=0x%08X "
                     "callconv=0x%08X types=0x%08X modifiers=0x%08X methodInfo=0x%08X "
                     "lr=0x%08X sp=0x%08X",
                     call, type_object, type_info, type_info_valid ? 1 : 0,
                     full_name_object, full_name_valid ? 1 : 0,
                     full_name_valid ? full_name : "<invalid>", default_constructor,
                     default_constructor_field_valid ? 1 : 0, type_handle,
                     type_handle_valid ? 1 : 0, type_data, type_data_valid ? 1 : 0,
                     type_bits, type_bits_valid ? 1 : 0, constructor_array,
                     constructor_array_valid ? 1 : 0, constructor_count,
                     constructor_count_valid ? 1 : 0, binding_flags, call_convention,
                     types, modifiers, method_info, jit.Regs()[14], jit.Regs()[13]);
            }

            const bool log_constructors = known_type;
            if (log_constructors && default_constructor) {
                uint32_t constructor_class = 0;
                uint32_t constructor_method = 0;
                uint32_t constructor_name_object = 0;
                uint32_t constructor_reflected_type = 0;
                read_word(h, default_constructor, 0, constructor_class);
                read_word(h, default_constructor, 8, constructor_method);
                read_word(h, default_constructor, 12, constructor_name_object);
                read_word(h, default_constructor, 16, constructor_reflected_type);
                char constructor_name[kActivatorDiagTextSize] = {};
                const bool constructor_name_valid = CopyActivatorManagedText(
                    h, constructor_name_object, constructor_name, sizeof(constructor_name));
                LOGI("constructor metadata ctor kind=default index=0 object=0x%08X "
                     "klass=0x%08X method=0x%08X name_obj=0x%08X name_valid=%d name=%s "
                     "reflected_type=0x%08X",
                     default_constructor, constructor_class, constructor_method,
                     constructor_name_object, constructor_name_valid ? 1 : 0,
                     constructor_name_valid ? constructor_name : "<invalid>",
                     constructor_reflected_type);
            }
            if (log_constructors && constructor_count_valid) {
                const uint32_t dump_count = std::min<uint32_t>(
                    constructor_count, static_cast<uint32_t>(kMaxConstructors));
                for (uint32_t index = 0; index < dump_count; ++index) {
                    uint32_t constructor = 0;
                    if (!read_word(h, constructor_array, 0x10 + index * 4, constructor)) {
                        LOGI("constructor metadata ctor index=%u object=<unreadable>", index);
                        continue;
                    }
                    uint32_t constructor_class = 0;
                    uint32_t constructor_method = 0;
                    uint32_t constructor_name_object = 0;
                    uint32_t constructor_reflected_type = 0;
                    read_word(h, constructor, 0, constructor_class);
                    read_word(h, constructor, 8, constructor_method);
                    read_word(h, constructor, 12, constructor_name_object);
                    read_word(h, constructor, 16, constructor_reflected_type);
                    char constructor_name[kActivatorDiagTextSize] = {};
                    const bool constructor_name_valid = CopyActivatorManagedText(
                        h, constructor_name_object, constructor_name, sizeof(constructor_name));
                    char class_name[kActivatorDiagTextSize] = {};
                    uint32_t class_name_object = 0;
                    const bool class_name_field_valid =
                        read_word(h, constructor_class, 8, class_name_object);
                    const bool class_name_valid = class_name_field_valid &&
                        CopyActivatorText(h, class_name_object, class_name, sizeof(class_name));
                    LOGI("constructor metadata ctor kind=array index=%u object=0x%08X "
                         "klass=0x%08X method=0x%08X name_obj=0x%08X name_valid=%d name=%s "
                         "reflected_type=0x%08X class_name=%s",
                         index, constructor, constructor_class, constructor_method,
                         constructor_name_object, constructor_name_valid ? 1 : 0,
                         constructor_name_valid ? constructor_name : "<invalid>",
                         constructor_reflected_type,
                         class_name_valid ? class_name : "<invalid>");
                }
                if (constructor_count > dump_count) {
                    LOGI("constructor metadata ctor array truncated count=%u limit=%u",
                         constructor_count, dump_count);
                }
            }

            // The replaced instruction is exactly `mov r6, r0`; do not alter any
            // other register or returned pointer, and resume at the next instruction.
            jit.Regs()[6] = constructor_array;
            jit.Regs()[15] = kResumePc;
            (void)kWatchPc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("constructor metadata watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kWatchPc, svc);
    host.InvalidateGuestCode(kWatchPc, sizeof(uint32_t));
    LOGI("constructor metadata watch installed at 0x%08X resume=0x%08X",
         kWatchPc, kResumePc);
    return true;
}

static bool InstallMethodCallingConventionWatch(DynarmicHost& host) {
    constexpr uint32_t kLoadPc = cfg::kIl2cppBase + 0x1240488;
    constexpr uint32_t kResumePc = cfg::kIl2cppBase + 0x124048c;
    constexpr uint32_t kExpectedLoad = 0xE59D0010;
    constexpr uint32_t kExpectedResume = 0xE24BD008;
    if (host.MemoryRead32(kLoadPc) != kExpectedLoad ||
        host.MemoryRead32(kResumePc) != kExpectedResume) {
        LOGE("method callconv watch: unexpected opcodes at 0x%08X/0x%08X: "
             "0x%08X/0x%08X", kLoadPc, kResumePc, host.MemoryRead32(kLoadPc),
             host.MemoryRead32(kResumePc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_method_calling_convention_watch", [kResumePc](Dynarmic::A32::Jit& jit,
                                                               DynarmicHost& h) {
            const uint32_t sp = jit.Regs()[13];
            const uint32_t method = jit.Regs()[4];
            uint32_t words[6] = {};
            const bool stack_valid = GuestRangeValid(h.GetGuestMemorySize(), sp, sizeof(words));
            if (stack_valid) {
                for (size_t i = 0; i < std::size(words); ++i) {
                    words[i] = h.MemoryRead32(sp + static_cast<uint32_t>(i * 4));
                }
            }
            uint32_t method_attrs = 0;
            uint32_t method_iattrs = 0;
            const bool method_valid = GuestRangeValid(h.GetGuestMemorySize(), method, 44);
            if (method_valid) {
                method_attrs = h.MemoryRead16(method + 40);
                method_iattrs = h.MemoryRead16(method + 42);
            }
            LOGI("method callconv diag method=0x%08X method_valid=%d sp=0x%08X "
                 "stack_valid=%d info_parent=0x%08X info_ret=0x%08X info_attrs=0x%08X "
                 "info_iattrs=0x%08X callconv=0x%08X neighbor=0x%08X "
                 "method_attrs=0x%08X method_iattrs=0x%08X",
                 method, method_valid ? 1 : 0, sp, stack_valid ? 1 : 0,
                 words[0], words[1], words[2], words[3], words[4], words[5],
                 method_attrs, method_iattrs);
            jit.Regs()[0] = words[4];
            jit.Regs()[15] = kResumePc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("method callconv watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kLoadPc, svc);
    host.InvalidateGuestCode(kLoadPc, sizeof(uint32_t));
    LOGI("method callconv watch installed at 0x%08X resume=0x%08X",
         kLoadPc, kResumePc);
    return true;
}

static bool InstallMethodFilterDispatchWatch(DynarmicHost& host) {
    constexpr uint32_t kLoadPc = cfg::kUnityBase + 0x87E6C0;
    constexpr uint32_t kResumePc = cfg::kUnityBase + 0x87E6C4;
    constexpr uint32_t kExpectedLoad = 0xE59D0014;
    constexpr uint32_t kExpectedResume = 0xE5901000;
    if (host.MemoryRead32(kLoadPc) != kExpectedLoad ||
        host.MemoryRead32(kResumePc) != kExpectedResume) {
        LOGE("method filter dispatch watch: unexpected opcodes at 0x%08X/0x%08X: "
             "0x%08X/0x%08X", kLoadPc, kResumePc, host.MemoryRead32(kLoadPc),
             host.MemoryRead32(kResumePc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_method_filter_dispatch_watch", [kResumePc](Dynarmic::A32::Jit& jit,
                                                            DynarmicHost& h) {
            const uint32_t sp = jit.Regs()[13];
            const uint32_t candidate = jit.Regs()[4];
            const bool stack_valid = GuestRangeValid(h.GetGuestMemorySize(), sp, 0x18);
            const uint32_t slot_pointer = stack_valid ? h.MemoryRead32(sp + 0x14) : 0;
            const uint32_t target = slot_pointer &&
                GuestRangeValid(h.GetGuestMemorySize(), slot_pointer, 4)
                ? h.MemoryRead32(slot_pointer) : 0;
            uint32_t klass = 0;
            if (GuestRangeValid(h.GetGuestMemorySize(), candidate, 4)) {
                klass = h.MemoryRead32(candidate);
            }
            LOGI("method filter dispatch candidate=0x%08X klass=0x%08X sp=0x%08X "
                 "stack_valid=%d slot_ptr=0x%08X target=0x%08X expected=0x%08X",
                 candidate, klass, sp, stack_valid ? 1 : 0, slot_pointer, target,
                 cfg::kIl2cppBase + 0x124045C);
            jit.Regs()[0] = slot_pointer;
            jit.Regs()[15] = kResumePc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("method filter dispatch watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kLoadPc, svc);
    host.InvalidateGuestCode(kLoadPc, sizeof(uint32_t));
    LOGI("method filter dispatch watch installed at 0x%08X resume=0x%08X",
         kLoadPc, kResumePc);
    return true;
}

static bool InstallConstructorFilterDispatchWatch(DynarmicHost& host) {
    constexpr uint32_t kLoadPc = cfg::kIl2cppBase + 0x151C0D8;
    constexpr uint32_t kResumePc = cfg::kIl2cppBase + 0x151C0DC;
    constexpr uint32_t kExpectedLoad = 0xE5950000;
    constexpr uint32_t kExpectedResume = 0xE5902164;
    if (host.MemoryRead32(kLoadPc) != kExpectedLoad ||
        host.MemoryRead32(kResumePc) != kExpectedResume) {
        LOGE("constructor filter dispatch watch: unexpected opcodes at 0x%08X/0x%08X: "
             "0x%08X/0x%08X", kLoadPc, kResumePc, host.MemoryRead32(kLoadPc),
             host.MemoryRead32(kResumePc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_constructor_filter_dispatch_watch", [kResumePc](Dynarmic::A32::Jit& jit,
                                                                  DynarmicHost& h) {
            const uint32_t candidate = jit.Regs()[5];
            const uint32_t klass = GuestRangeValid(h.GetGuestMemorySize(), candidate, 4)
                ? h.MemoryRead32(candidate) : 0;
            const uint32_t vtable = GuestRangeValid(h.GetGuestMemorySize(), klass, 0x16C)
                ? h.MemoryRead32(klass) : 0;
            const uint32_t callconv_target = GuestRangeValid(
                h.GetGuestMemorySize(), vtable + 0x164, 4) ? h.MemoryRead32(vtable + 0x164) : 0;
            const uint32_t alt_target = GuestRangeValid(
                h.GetGuestMemorySize(), vtable + 0x168, 4) ? h.MemoryRead32(vtable + 0x168) : 0;
            LOGI("constructor filter dispatch candidate=0x%08X klass=0x%08X "
                 "vtable=0x%08X slot164=0x%08X slot168=0x%08X expected=0x%08X",
                 candidate, klass, vtable, callconv_target, alt_target,
                 cfg::kIl2cppBase + 0x124045C);
            jit.Regs()[0] = klass;
            jit.Regs()[15] = kResumePc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("constructor filter dispatch watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kLoadPc, svc);
    host.InvalidateGuestCode(kLoadPc, sizeof(uint32_t));
    LOGI("constructor filter dispatch watch installed at 0x%08X resume=0x%08X",
         kLoadPc, kResumePc);
    return true;
}

static bool InstallConstructorBinderSelectionWatch(DynarmicHost& host) {
    constexpr uint32_t kWatchPc = cfg::kIl2cppBase + 0x151C17C;
    constexpr uint32_t kResumePc = cfg::kIl2cppBase + 0x151C180;
    constexpr uint32_t kExpectedMov = 0xE1A01004;
    constexpr uint32_t kExpectedCmp = 0xE3500000;
    if (host.MemoryRead32(kWatchPc) != kExpectedMov ||
        host.MemoryRead32(kResumePc) != kExpectedCmp) {
        LOGE("constructor binder watch: unexpected opcodes at 0x%08X/0x%08X: "
             "0x%08X/0x%08X", kWatchPc, kResumePc, host.MemoryRead32(kWatchPc),
             host.MemoryRead32(kResumePc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_constructor_binder_selection_watch", [kResumePc](Dynarmic::A32::Jit& jit,
                                                                  DynarmicHost& h) {
            const uint32_t binder_result = jit.Regs()[0];
            const uint32_t candidate = jit.Regs()[10];
            const uint32_t array = jit.Regs()[4];
            const uint32_t frame = jit.Regs()[11];
            const uint32_t types = GuestRangeValid(h.GetGuestMemorySize(), frame + 8, 4)
                ? h.MemoryRead32(frame + 8) : 0;
            const uint32_t modifiers = GuestRangeValid(h.GetGuestMemorySize(), frame + 12, 4)
                ? h.MemoryRead32(frame + 12) : 0;
            const uint32_t binder = GuestRangeValid(h.GetGuestMemorySize(), frame + 16, 4)
                ? h.MemoryRead32(frame + 16) : 0;
            LOGI("constructor binder diag result=0x%08X candidate=0x%08X array=0x%08X "
                 "frame=0x%08X types=0x%08X modifiers=0x%08X binder=0x%08X "
                 "binding=0x%08X callconv=0x%08X",
                 binder_result, candidate, array, frame, types, modifiers, binder,
                 jit.Regs()[5], jit.Regs()[7]);
            jit.Regs()[1] = array;
            jit.Regs()[15] = kResumePc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("constructor binder watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kWatchPc, svc);
    host.InvalidateGuestCode(kWatchPc, sizeof(uint32_t));
    LOGI("constructor binder watch installed at 0x%08X resume=0x%08X",
         kWatchPc, kResumePc);
    return true;
}

static bool InstallConstructorInvokeDispatchWatch(DynarmicHost& host) {
    constexpr uint32_t kWatchPc = cfg::kIl2cppBase + 0x152FB30;
    constexpr uint32_t kResumePc = cfg::kIl2cppBase + 0x152FB34;
    constexpr uint32_t kExpectedMov = 0xE1A00006;
    constexpr uint32_t kExpectedBlx = 0xE12FFF34;
    if (host.MemoryRead32(kWatchPc) != kExpectedMov ||
        host.MemoryRead32(kResumePc) != kExpectedBlx) {
        LOGE("constructor invoke watch: unexpected opcodes at 0x%08X/0x%08X: "
             "0x%08X/0x%08X", kWatchPc, kResumePc, host.MemoryRead32(kWatchPc),
             host.MemoryRead32(kResumePc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_constructor_invoke_dispatch_watch", [kResumePc](Dynarmic::A32::Jit& jit,
                                                                  DynarmicHost& h) {
            const auto read_word = [&h](uint32_t base, size_t offset) {
                const uint64_t address = static_cast<uint64_t>(base) + offset;
                if (address > UINT32_MAX || base < 0x1000 ||
                    !GuestRangeValid(h.GetGuestMemorySize(), static_cast<uint32_t>(address), 4)) {
                    return uint32_t{};
                }
                return h.MemoryRead32(static_cast<uint32_t>(address));
            };
            const uint32_t object = jit.Regs()[6];
            const uint32_t klass = read_word(object, 0);
            const uint32_t mhandle = read_word(object, 8);
            const uint32_t name = read_word(object, 12);
            const uint32_t reflected_type = read_word(object, 16);
            const uint32_t type_handle = read_word(reflected_type, 8);
            const uint32_t type_data = read_word(type_handle, 0);
            const uint32_t target = jit.Regs()[4];
            static std::atomic<unsigned> probe_count{0};
            if (type_data == 0x1A09 || type_data == 0x2489) {
                const unsigned probe = probe_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (probe <= 64) {
                    LOGE("constructor invoke probe=%u object=0x%08X klass=0x%08X "
                         "mhandle=0x%08X method_ptr=0x%08X invoker=0x%08X "
                         "method_name=0x%08X declaring=0x%08X params=0x%08X "
                         "rgctx=0x%08X flags=0x%04X iflags=0x%04X slot=%u "
                         "param_count=%u bits=0x%02X reflected_type=0x%08X "
                         "type_handle=0x%08X type_data=0x%08X "
                         "vslot134=0x%08X vslot138=0x%08X vslot14c=0x%08X "
                         "vslot150=0x%08X target=0x%08X r1=0x%08X r2=0x%08X r3=0x%08X",
                         probe, object, klass, mhandle, read_word(mhandle, 0),
                         read_word(mhandle, 4), read_word(mhandle, 8), read_word(mhandle, 12),
                         read_word(mhandle, 20), read_word(mhandle, 24), read_word(mhandle, 40) & 0xFFFF,
                         read_word(mhandle, 42) & 0xFFFF, read_word(mhandle, 44) & 0xFFFF,
                         read_word(mhandle, 46) & 0xFF, read_word(mhandle, 47) & 0xFF,
                         reflected_type, type_handle, type_data, read_word(klass, 0x134),
                         read_word(klass, 0x138), read_word(klass, 0x14C), read_word(klass, 0x150),
                         target, jit.Regs()[1], jit.Regs()[2], jit.Regs()[3]);
                }
            }
            LOGI("constructor invoke diag object=0x%08X klass=0x%08X mhandle=0x%08X "
                 "name=0x%08X target=0x%08X r1=0x%08X r2=0x%08X r3=0x%08X",
                 object, klass, mhandle, name, target, jit.Regs()[1], jit.Regs()[2],
                 jit.Regs()[3]);
            jit.Regs()[0] = object;
            jit.Regs()[15] = kResumePc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("constructor invoke watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kWatchPc, svc);
    host.InvalidateGuestCode(kWatchPc, sizeof(uint32_t));
    LOGI("constructor invoke watch installed at 0x%08X resume=0x%08X",
         kWatchPc, kResumePc);
    return true;
}

static bool InstallConstructorInvokeReturnWatch(DynarmicHost& host) {
    constexpr uint32_t kWatchPc = cfg::kIl2cppBase + 0x152FB38;
    constexpr uint32_t kResumePc = cfg::kIl2cppBase + 0x152FB3C;
    constexpr uint32_t kExpectedSub = 0xE24BD010;
    if (host.MemoryRead32(kWatchPc) != kExpectedSub) {
        LOGE("constructor invoke return watch: unexpected opcode at 0x%08X: 0x%08X",
             kWatchPc, host.MemoryRead32(kWatchPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_constructor_invoke_return_watch", [kResumePc](Dynarmic::A32::Jit& jit,
                                                                DynarmicHost& h) {
            const auto read_word = [&h](uint32_t base, size_t offset) {
                uint32_t value = 0;
                return ReadActivatorWord(h, base, offset, value) ? value : 0u;
            };
            const uint32_t object = jit.Regs()[6];
            const uint32_t reflected_type = read_word(object, 16);
            const uint32_t type_handle = read_word(reflected_type, 8);
            const uint32_t type_data = read_word(type_handle, 0);
            if (type_data == 0x1A09 || type_data == 0x2489) {
                const uint32_t result = jit.Regs()[0];
                const uint32_t result_klass = read_word(result, 0);
                char result_name[96] = {};
                char result_namespace[96] = {};
                const bool name_ok = CopyActivatorText(h, read_word(result_klass, 8),
                                                        result_name, sizeof(result_name));
                const bool namespace_ok = CopyActivatorText(
                    h, read_word(result_klass, 12), result_namespace, sizeof(result_namespace));
                LOGE("constructor invoke return object=0x%08X klass=0x%08X mhandle=0x%08X "
                     "type_data=0x%08X result=0x%08X result_klass=0x%08X "
                     "result_name=%s result_namespace=%s result_name_ok=%u "
                     "result_namespace_ok=%u r1=0x%08X r2=0x%08X r3=0x%08X "
                     "r4=0x%08X r5=0x%08X r6=0x%08X r11=0x%08X lr=0x%08X sp=0x%08X",
                     object, read_word(object, 0), read_word(object, 8), type_data, result,
                     result_klass, name_ok ? result_name : "<invalid>",
                     namespace_ok ? result_namespace : "<invalid>", name_ok, namespace_ok,
                     jit.Regs()[1], jit.Regs()[2], jit.Regs()[3], jit.Regs()[4],
                     jit.Regs()[5], jit.Regs()[6], jit.Regs()[11], jit.Regs()[14],
                     jit.Regs()[13]);
            }
            // The replaced instruction is exactly `sub sp, r11, #16`.
            jit.Regs()[13] = jit.Regs()[11] - 16;
            jit.Regs()[15] = kResumePc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("constructor invoke return watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kWatchPc, svc);
    host.InvalidateGuestCode(kWatchPc, sizeof(uint32_t));
    LOGI("constructor invoke return watch installed at 0x%08X resume=0x%08X",
         kWatchPc, kResumePc);
    return true;
}

static bool InstallActivatorConstructorResultWatch(DynarmicHost& host) {
    constexpr uint32_t kWatchPc = cfg::kIl2cppBase + 0x1872BF8;
    constexpr uint32_t kResumePc = cfg::kIl2cppBase + 0x1872BFC;
    constexpr uint32_t kExpectedMov = 0xE1A06000;
    if (host.MemoryRead32(kWatchPc) != kExpectedMov) {
        LOGE("activator constructor-result watch: unexpected opcode at 0x%08X: 0x%08X",
             kWatchPc, host.MemoryRead32(kWatchPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_activator_constructor_result_watch", [kResumePc](Dynarmic::A32::Jit& jit,
                                                                   DynarmicHost& h) {
            const auto read_word = [&h](uint32_t base, size_t offset) {
                const uint64_t address = static_cast<uint64_t>(base) + offset;
                if (address > UINT32_MAX || base < 0x1000 ||
                    !GuestRangeValid(h.GetGuestMemorySize(), static_cast<uint32_t>(address), 4)) {
                    return uint32_t{};
                }
                return h.MemoryRead32(static_cast<uint32_t>(address));
            };
            const uint32_t type = jit.Regs()[4];
            const uint32_t type_handle = read_word(type, 8);
            const uint32_t type_data = read_word(type_handle, 0);
            const uint32_t result = jit.Regs()[0];
            static std::atomic<unsigned> result_count{0};
            const unsigned call = result_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32 || type_data == 0x1A09 || type_data == 0x2489) {
                LOGE("activator constructor result call=%u type=0x%08X type_handle=0x%08X "
                     "type_data=0x%08X result=0x%08X result_klass=0x%08X "
                     "result_mhandle=0x%08X result_name=0x%08X result_reftype=0x%08X "
                     "lr=0x%08X sp=0x%08X r5=0x%08X",
                     call, type, type_handle, type_data, result, read_word(result, 0),
                     read_word(result, 8), read_word(result, 12), read_word(result, 16),
                     jit.Regs()[14], jit.Regs()[13], jit.Regs()[5]);
            }
            // The replaced instruction is exactly `mov r6, r0`.
            jit.Regs()[6] = jit.Regs()[0];
            jit.Regs()[15] = kResumePc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("activator constructor-result watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kWatchPc, svc);
    host.InvalidateGuestCode(kWatchPc, sizeof(uint32_t));
    LOGI("activator constructor-result watch installed at 0x%08X resume=0x%08X",
         kWatchPc, kResumePc);
    return true;
}

static bool InstallActivatorAlternateConstructorResultWatch(DynarmicHost& host) {
    constexpr uint32_t kWatchPc = cfg::kIl2cppBase + 0x1872C28;
    constexpr uint32_t kResumePc = cfg::kIl2cppBase + 0x1872C2C;
    constexpr uint32_t kExpectedMov = 0xE1A06000;
    if (host.MemoryRead32(kWatchPc) != kExpectedMov) {
        LOGE("activator alternate constructor-result watch: unexpected opcode at 0x%08X: 0x%08X",
             kWatchPc, host.MemoryRead32(kWatchPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_activator_alt_constructor_result_watch", [kResumePc](Dynarmic::A32::Jit& jit,
                                                                       DynarmicHost& h) {
            const auto read_word = [&h](uint32_t base, size_t offset) {
                const uint64_t address = static_cast<uint64_t>(base) + offset;
                if (address > UINT32_MAX || base < 0x1000 ||
                    !GuestRangeValid(h.GetGuestMemorySize(), static_cast<uint32_t>(address), 4)) {
                    return uint32_t{};
                }
                return h.MemoryRead32(static_cast<uint32_t>(address));
            };
            const uint32_t type = jit.Regs()[4];
            const uint32_t type_handle = read_word(type, 8);
            const uint32_t type_data = read_word(type_handle, 0);
            const uint32_t result = jit.Regs()[0];
            static std::atomic<unsigned> result_count{0};
            const unsigned call = result_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32 || type_data == 0x1A09 || type_data == 0x2489) {
                LOGE("activator alternate constructor result call=%u type=0x%08X "
                     "type_handle=0x%08X type_data=0x%08X result=0x%08X "
                     "result_klass=0x%08X result_mhandle=0x%08X result_name=0x%08X "
                     "result_reftype=0x%08X lr=0x%08X sp=0x%08X r5=0x%08X",
                     call, type, type_handle, type_data, result, read_word(result, 0),
                     read_word(result, 8), read_word(result, 12), read_word(result, 16),
                     jit.Regs()[14], jit.Regs()[13], jit.Regs()[5]);
            }
            // The replaced instruction is exactly `mov r6, r0`.
            jit.Regs()[6] = result;
            jit.Regs()[15] = kResumePc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("activator alternate constructor-result watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kWatchPc, svc);
    host.InvalidateGuestCode(kWatchPc, sizeof(uint32_t));
    LOGI("activator alternate constructor-result watch installed at 0x%08X resume=0x%08X",
         kWatchPc, kResumePc);
    return true;
}

static bool InstallActivatorIsPublicWatch(DynarmicHost& host) {
    constexpr uint32_t kEntryPc = cfg::kIl2cppBase + 0x123EE90;
    constexpr uint32_t kResultPc = cfg::kIl2cppBase + 0x123EEA8;
    constexpr uint32_t kResumeEntry = kEntryPc + 4;
    constexpr uint32_t kResumeResult = kResultPc + 4;
    if (host.MemoryRead32(kEntryPc) != 0xE92D4800 ||
        host.MemoryRead32(kResultPc) != 0xE2001007) {
        LOGE("activator IsPublic watch: unexpected opcodes at 0x%08X/0x%08X: "
             "0x%08X/0x%08X", kEntryPc, kResultPc, host.MemoryRead32(kEntryPc),
             host.MemoryRead32(kResultPc));
        return false;
    }
    const uint32_t entry_hook = host.GetThunkManager().RegisterThunk(
        "guest_activator_ispublic_entry_watch", [kResumeEntry](Dynarmic::A32::Jit& jit,
                                                                  DynarmicHost& h) {
            const auto read_word = [&h](uint32_t base, size_t offset) {
                uint32_t value = 0;
                return ReadActivatorWord(h, base, offset, value) ? value : 0u;
            };
            ActivatorIsPublicDiagSnapshot& snapshot = g_activator_is_public_diag;
            snapshot = {};
            static std::atomic<unsigned> count{0};
            snapshot.call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            snapshot.object = jit.Regs()[0];
            snapshot.klass = read_word(snapshot.object, 0);
            snapshot.mhandle = read_word(snapshot.object, 8);
            snapshot.reflected_type = read_word(snapshot.object, 16);
            snapshot.type_handle = read_word(snapshot.reflected_type, 8);
            snapshot.type_data = read_word(snapshot.type_handle, 0);
            if (snapshot.call <= 16 || snapshot.type_data == 0x1A09 ||
                snapshot.type_data == 0x2489) {
                LOGE("activator IsPublic entry call=%u object=0x%08X klass=0x%08X "
                     "mhandle=0x%08X reflected_type=0x%08X type_handle=0x%08X "
                     "type_data=0x%08X lr=0x%08X sp=0x%08X",
                     snapshot.call, snapshot.object, snapshot.klass, snapshot.mhandle,
                     snapshot.reflected_type, snapshot.type_handle, snapshot.type_data,
                     jit.Regs()[14], jit.Regs()[13]);
            }
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 8;
            h.MemoryWrite32(new_sp, jit.Regs()[11]);
            h.MemoryWrite32(new_sp + 4, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            jit.Regs()[11] = new_sp;
            jit.Regs()[15] = kResumeEntry;
        });
    const uint32_t result_hook = host.GetThunkManager().RegisterThunk(
        "guest_activator_ispublic_result_watch", [kResumeResult](Dynarmic::A32::Jit& jit,
                                                                   DynarmicHost&) {
            const ActivatorIsPublicDiagSnapshot& snapshot = g_activator_is_public_diag;
            if (snapshot.call <= 16 || snapshot.type_data == 0x1A09 ||
                snapshot.type_data == 0x2489) {
                LOGE("activator IsPublic result call=%u type_data=0x%08X object=0x%08X "
                     "mhandle=0x%08X attributes=0x%08X public=%u lr=0x%08X sp=0x%08X",
                     snapshot.call, snapshot.type_data, snapshot.object, snapshot.mhandle,
                     jit.Regs()[0], jit.Regs()[0] & 7u, jit.Regs()[14], jit.Regs()[13]);
            }
            // The replaced instruction is exactly `and r1, r0, #7`.
            jit.Regs()[1] = jit.Regs()[0] & 7u;
            jit.Regs()[15] = kResumeResult;
        });
    const uint32_t entry_svc = host.MemoryRead32(entry_hook);
    const uint32_t result_svc = host.MemoryRead32(result_hook);
    if ((entry_svc & 0xFF000000u) != 0xEF000000u ||
        (result_svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("activator IsPublic watch: invalid thunk");
        return false;
    }
    host.MemoryWrite32(kEntryPc, entry_svc);
    host.MemoryWrite32(kResultPc, result_svc);
    host.InvalidateGuestCode(kEntryPc, sizeof(uint32_t));
    host.InvalidateGuestCode(kResultPc, sizeof(uint32_t));
    LOGI("activator IsPublic watch installed at 0x%08X/0x%08X", kEntryPc, kResultPc);
    return true;
}

static bool InstallActivatorIsPublicBranchWatch(DynarmicHost& host) {
    constexpr uint32_t kBranchPc = cfg::kIl2cppBase + 0x1872C4C;
    constexpr uint32_t kTargetPc = cfg::kIl2cppBase + 0x1872BFC;
    constexpr uint32_t kFallthroughPc = kBranchPc + 4;
    constexpr uint32_t kExpectedBne = 0x1AFFFFEA;
    if (host.MemoryRead32(kBranchPc) != kExpectedBne) {
        LOGE("activator IsPublic branch watch: unexpected opcode at 0x%08X: 0x%08X",
             kBranchPc, host.MemoryRead32(kBranchPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_activator_ispublic_branch_watch", [kTargetPc, kFallthroughPc](
                                                        Dynarmic::A32::Jit& jit,
                                                        DynarmicHost&) {
            const ActivatorIsPublicDiagSnapshot& snapshot = g_activator_is_public_diag;
            if (snapshot.call <= 16 || snapshot.type_data == 0x1A09 ||
                snapshot.type_data == 0x2489) {
                LOGE("activator IsPublic branch call=%u type_data=0x%08X r0=0x%08X "
                     "r6=0x%08X cpsr=0x%08X z=%u branch=%s lr=0x%08X sp=0x%08X",
                     snapshot.call, snapshot.type_data, jit.Regs()[0], jit.Regs()[6],
                     jit.Cpsr(), (jit.Cpsr() >> 30) & 1u,
                     (jit.Cpsr() & (1u << 30)) ? "fallthrough" : "invoke",
                     jit.Regs()[14], jit.Regs()[13]);
            }
            // The replaced instruction is exactly `bne 0x1872BFC`.
            jit.Regs()[15] = (jit.Cpsr() & (1u << 30)) ? kFallthroughPc : kTargetPc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("activator IsPublic branch watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kBranchPc, svc);
    host.InvalidateGuestCode(kBranchPc, sizeof(uint32_t));
    LOGI("activator IsPublic branch watch installed at 0x%08X target=0x%08X",
         kBranchPc, kTargetPc);
    return true;
}

static bool InstallActivatorMissingCandidatePathWatch(DynarmicHost& host) {
    constexpr uint32_t kWatchPc = cfg::kIl2cppBase + 0x1872C5C;
    constexpr uint32_t kResumePc = kWatchPc + 4;
    constexpr uint32_t kExpectedMov = 0xE1A00004;
    if (host.MemoryRead32(kWatchPc) != kExpectedMov) {
        LOGE("activator missing-candidate watch: unexpected opcode at 0x%08X: 0x%08X",
             kWatchPc, host.MemoryRead32(kWatchPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_activator_missing_candidate_watch", [kResumePc](Dynarmic::A32::Jit& jit,
                                                                  DynarmicHost& h) {
            const auto read_word = [&h](uint32_t base, size_t offset) {
                uint32_t value = 0;
                return ReadActivatorWord(h, base, offset, value) ? value : 0u;
            };
            const uint32_t type = jit.Regs()[4];
            const uint32_t type_handle = read_word(type, 8);
            const uint32_t type_data = read_word(type_handle, 0);
            const uint32_t candidate = jit.Regs()[6];
            const uint32_t candidate_klass = read_word(candidate, 0);
            const uint32_t mhandle = read_word(candidate, 8);
            const uint32_t method_name = read_word(mhandle, 8);
            char method_name_text[96] = {};
            const bool method_name_ok = CopyActivatorText(h, method_name, method_name_text,
                                                           sizeof(method_name_text));
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32 || type_data == 0x1A09 || type_data == 0x2489) {
                LOGE("activator missing-candidate call=%u type=0x%08X type_handle=0x%08X "
                     "type_data=0x%08X candidate=0x%08X candidate_klass=0x%08X "
                     "mhandle=0x%08X method_ptr=0x%08X invoker=0x%08X method_name=%s "
                     "declaring=0x%08X flags=0x%04X iflags=0x%04X slot=%u params=%u "
                     "r5=0x%08X r0=0x%08X r1=0x%08X r2=0x%08X r3=0x%08X "
                     "lr=0x%08X sp=0x%08X cpsr=0x%08X",
                     call, type, type_handle, type_data, candidate, candidate_klass, mhandle,
                     read_word(mhandle, 0), read_word(mhandle, 4),
                     method_name_ok ? method_name_text : "<invalid>", read_word(mhandle, 12),
                     read_word(mhandle, 40) & 0xFFFF, read_word(mhandle, 42) & 0xFFFF,
                     read_word(mhandle, 44) & 0xFFFF, read_word(mhandle, 46) & 0xFF,
                     jit.Regs()[5], jit.Regs()[0], jit.Regs()[1], jit.Regs()[2],
                     jit.Regs()[3], jit.Regs()[14], jit.Regs()[13], jit.Cpsr());
            }
            // The replaced instruction is exactly `mov r0, r4`.
            jit.Regs()[0] = jit.Regs()[4];
            jit.Regs()[15] = kResumePc;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("activator missing-candidate watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kWatchPc, svc);
    host.InvalidateGuestCode(kWatchPc, sizeof(uint32_t));
    LOGI("activator missing-candidate watch installed at 0x%08X resume=0x%08X",
         kWatchPc, kResumePc);
    return true;
}

static bool InstallRaiseWatch(DynarmicHost& host) {
    constexpr uint32_t kRaisePc = cfg::kIl2cppBase + 0x22b7b4c;
    constexpr uint32_t kExpectedPush = 0xE92D4BF0;
    if (host.MemoryRead32(kRaisePc) != kExpectedPush) {
        LOGE("raise watch: unexpected opcode at 0x%08X: 0x%08X",
             kRaisePc, host.MemoryRead32(kRaisePc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_raise_watch", [kRaisePc](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            static std::atomic<unsigned> diagnostic_count{0};
            const uint32_t caller = jit.Regs()[14];
            const uint32_t old_sp = jit.Regs()[13];
            uint32_t exception = jit.Regs()[0];
            const bool exception_valid = exception >= 0x1000 &&
                GuestRangeValid(h.GetGuestMemorySize(), exception, 20);
            const uint32_t klass = exception_valid ? h.MemoryRead32(exception) : 0;
            if (!klass || !GuestRangeValid(h.GetGuestMemorySize(), klass, 16)) {
                LOGE("guest raise rejected invalid exception object=0x%08X caller=0x%08X sp=0x%08X",
                     exception, caller, old_sp);
                jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
                return;
            }
            const uint32_t new_sp = old_sp - 32;
            for (unsigned i = 0; i < 6; ++i) h.MemoryWrite32(new_sp + i * 4, jit.Regs()[4 + i]);
            h.MemoryWrite32(new_sp + 24, jit.Regs()[11]);
            h.MemoryWrite32(new_sp + 28, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            const bool log_diagnostics =
                diagnostic_count.fetch_add(1, std::memory_order_relaxed) < 32;
            auto guest_text = [&h](uint32_t address) {
                if (!GuestRangeValid(h.GetGuestMemorySize(), address, 1)) return std::string();
                const char* value = reinterpret_cast<const char*>(h.GetGuestMemoryBase() + address);
                const size_t length = strnlen(value, h.GetGuestMemorySize() - address);
                return length < h.GetGuestMemorySize() - address ? std::string(value, length) : std::string();
            };
            auto managed_text = [&h](uint32_t address) {
                if (!GuestRangeValid(h.GetGuestMemorySize(), address, 12)) return std::string();
                const uint32_t length = std::min(h.MemoryRead32(address + 8), 256u);
                if (!GuestRangeValid(h.GetGuestMemorySize(), address + 12, length * 2u)) return std::string();
                std::string value;
                value.reserve(length);
                for (uint32_t i = 0; i < length; ++i)
                    value.push_back(static_cast<char>(h.MemoryRead32(address + 12 + i * 2) & 0xFF));
                return value;
            };
            for (unsigned depth = 0; log_diagnostics && exception && depth < 3; ++depth) {
                const uint32_t name = h.MemoryRead32(klass + 8);
                const uint32_t space = h.MemoryRead32(klass + 12);
                const uint32_t message = h.MemoryRead32(exception + 8);
                const std::string class_text = guest_text(name);
                const std::string namespace_text = guest_text(space);
                LOGE("guest raise pass-through depth=%u object=0x%08X class=%s.%s message=%s inner=0x%08X caller=0x%08X sp=0x%08X",
                     depth, exception, namespace_text.c_str(), class_text.c_str(),
                     managed_text(message).c_str(), h.MemoryRead32(exception + 16), caller, old_sp);
                if (depth == 0 && namespace_text == "System" &&
                    class_text == "MissingMethodException") {
                    const ActivatorDiagSnapshot* snapshot = g_activator_diag_ring.Find(old_sp);
                    if (snapshot) LogActivatorSnapshot(h, *snapshot, true);
                }
                exception = h.MemoryRead32(exception + 16);
                if (exception && !GuestRangeValid(h.GetGuestMemorySize(), exception, 20)) break;
            }
            // The replaced instruction is exactly `push {r4-r9,r11,lr}`.
            jit.Regs()[15] = kRaisePc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("raise watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kRaisePc, svc);
    host.InvalidateGuestCode(kRaisePc, sizeof(uint32_t));
    LOGI("raise watch installed at 0x%08X with catchable pass-through", kRaisePc);
    return true;
}

static bool InstallUnhandledExceptionWatch(DynarmicHost& host) {
    constexpr uint32_t kUnhandledPc = cfg::kIl2cppBase + 0x22AEF84;
    constexpr uint32_t kExpectedBranch = 0xEAFFC256;
    if (host.MemoryRead32(kUnhandledPc) != kExpectedBranch) {
        LOGE("unhandled-exception watch: unexpected opcode at 0x%08X: 0x%08X",
             kUnhandledPc, host.MemoryRead32(kUnhandledPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_unhandled_exception_watch", [](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t exception = jit.Regs()[0];
            const uint32_t caller = jit.Regs()[14];
            const bool valid = exception >= 0x1000 &&
                GuestRangeValid(h.GetGuestMemorySize(), exception, 20);
            LOGE("guest unhandled exception object=0x%08X valid=%d caller=0x%08X sp=0x%08X",
                 exception, valid ? 1 : 0, caller, jit.Regs()[13]);
            // Unhandled guest exceptions remain fail-closed; do not return as success.
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("unhandled-exception watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kUnhandledPc, svc);
    host.InvalidateGuestCode(kUnhandledPc, sizeof(uint32_t));
    LOGI("unhandled-exception watch installed at 0x%08X", kUnhandledPc);
    return true;
}

static bool InstallUnityCallbackFieldWatch(DynarmicHost& host) {
    constexpr uint32_t kLoadPc = cfg::kUnityBase + 0x9B93EC;
    constexpr uint32_t kExpectedLoad = 0xE59B72D4;
    if (host.MemoryRead32(kLoadPc) != kExpectedLoad) {
        LOGE("Unity callback-field watch: unexpected opcode at 0x%08X: 0x%08X",
             kLoadPc, host.MemoryRead32(kLoadPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_unity_callback_field_watch", [kLoadPc](Dynarmic::A32::Jit& jit,
                                                        DynarmicHost& h) {
            const uint32_t owner = jit.Regs()[11];
            const auto read_word = [&h, owner](uint32_t offset) {
                const uint64_t address = static_cast<uint64_t>(owner) + offset;
                return address <= UINT32_MAX && owner >= 0x1000 &&
                    GuestRangeValid(h.GetGuestMemorySize(), static_cast<uint32_t>(address), 4)
                    ? h.MemoryRead32(static_cast<uint32_t>(address)) : 0u;
            };
            const uint32_t vtable = read_word(0);
            const uint32_t callback_source = read_word(0x2D0);
            const uint32_t callback = read_word(0x2D4);
            const uint32_t vtable_instruction =
                vtable >= 0x10000 && GuestRangeValid(h.GetGuestMemorySize(), vtable, 4)
                    ? h.MemoryRead32(vtable) : 0;
            const uint32_t vtable_thunk_id =
                (vtable_instruction & 0xFF000000u) == 0xEF000000u
                    ? vtable_instruction & 0x00FFFFFFu : 0;
            const std::string vtable_thunk = vtable_thunk_id
                ? h.GetThunkManager().GetThunkName(vtable_thunk_id) : std::string();
            static std::atomic<unsigned> count{0};
            const unsigned call = count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (call <= 32 || !callback) {
                LOGI("Unity callback-field call=%u owner=0x%08X vtable=0x%08X "
                     "vtable_instr=0x%08X thunk_id=%u thunk=%s source=0x%08X "
                     "callback=0x%08X selected=0x%08X r0=0x%08X r1=0x%08X "
                     "r2=0x%08X r3=0x%08X lr=0x%08X",
                     call, owner, vtable, vtable_instruction, vtable_thunk_id,
                     vtable_thunk.empty() ? "<none>" : vtable_thunk.c_str(),
                     callback_source, callback, jit.Regs()[5], jit.Regs()[0],
                     jit.Regs()[1], jit.Regs()[2], jit.Regs()[3], jit.Regs()[14]);
            }
            // The replaced instruction is exactly `ldr r7, [r11, #0x2D4]`.
            jit.Regs()[7] = callback;
            jit.Regs()[15] = kLoadPc + 4;
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("Unity callback-field watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kLoadPc, svc);
    host.InvalidateGuestCode(kLoadPc, sizeof(uint32_t));
    LOGI("Unity callback-field watch installed at 0x%08X", kLoadPc);
    return true;
}

static bool InstallGcRegisterWatch(DynarmicHost& host) {
    constexpr uint32_t kRegisterPc = cfg::kIl2cppBase + 0x22e2a98;
    constexpr uint32_t kExpectedPush = 0xE92D4830;
    if (host.MemoryRead32(kRegisterPc) != kExpectedPush) {
        LOGE("GC register watch: unexpected opcode at 0x%08X: 0x%08X",
             kRegisterPc, host.MemoryRead32(kRegisterPc));
        return false;
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_gc_register_watch", [](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t old_sp = jit.Regs()[13];
            const uint32_t new_sp = old_sp - 16;
            h.MemoryWrite32(new_sp + 0, jit.Regs()[4]);
            h.MemoryWrite32(new_sp + 4, jit.Regs()[5]);
            h.MemoryWrite32(new_sp + 8, jit.Regs()[11]);
            h.MemoryWrite32(new_sp + 12, jit.Regs()[14]);
            jit.Regs()[13] = new_sp;
            LOGE("GC_register_my_thread a1=0x%08X value=0x%08X tid=%u",
                 jit.Regs()[0], h.MemoryRead32(jit.Regs()[0]), DynarmicHost::CurrentGuestThreadId());
        });
    const uint32_t svc = host.MemoryRead32(hook);
    if ((svc & 0xFF000000u) != 0xEF000000u) {
        LOGE("GC register watch: invalid thunk at 0x%08X", hook);
        return false;
    }
    host.MemoryWrite32(kRegisterPc, svc);
    host.InvalidateGuestCode(kRegisterPc, sizeof(uint32_t));
    LOGI("GC register watch installed at 0x%08X", kRegisterPc);
    return true;
}

static bool InstallGuestAndroidJniCalls(DynarmicHost& host) {
    constexpr uint32_t kGetStringUtf = cfg::kIl2cppBase + 0x2A693F4;
    static std::mutex strings_mutex;
    static std::unordered_map<uint32_t, uint32_t> strings;
    {
        std::lock_guard<std::mutex> lock(strings_mutex);
        strings.clear();
    }
    const uint32_t hook = host.GetThunkManager().RegisterThunk(
        "guest_androidjni_get_string_utf", [](Dynarmic::A32::Jit& jit, DynarmicHost& h) {
            const uint32_t handle = jit.Regs()[0];
            JNIEnv* env = FakeJniEnv::Instance().GetEnv();
            jstring value = static_cast<jstring>(FakeJniEnv::Instance().ResolveObject(handle));
            if (!env || !value) { jit.Regs()[0] = 0; return; }
            const char* chars = env->GetStringUTFChars(value, nullptr);
            if (!chars) { jit.Regs()[0] = 0; return; }
            std::string text(chars);
            env->ReleaseStringUTFChars(value, chars);
            std::lock_guard<std::mutex> lock(strings_mutex);
            auto found = strings.find(handle);
            if (found == strings.end()) {
                const uint32_t guest = h.GuestAlloc(text.size() + 1, 1);
                if (!guest) { jit.Regs()[0] = 0; return; }
                memcpy(h.GetGuestMemoryBase() + guest, text.c_str(), text.size() + 1);
                found = strings.emplace(handle, guest).first;
            }
            jit.Regs()[0] = found->second;
        });
    host.MemoryWrite32(kGetStringUtf, hook);
    LOGI("guest AndroidJNI::GetStringUTFChars patched at 0x%08X", kGetStringUtf);
    return true;
}

static bool CheckGuestRuntimeAbi(DynarmicHost& host) {
    const uint32_t memory = host.GuestAlloc(12, 4);
    if (!memory) return false;
    const uint32_t saved_errno = host.MemoryRead32(host.GuestErrnoAddress());
    host.MemoryWrite32(memory, 0xA5A5A5A5);
    host.MemoryWrite32(memory + 8, 0xA5A5A5A5);
    const auto call = [&](const char* name, std::vector<uint32_t> args) {
        return static_cast<int32_t>(host.RunFunction(host.GetThunkManager().GetOrCreateThunkAddress(name), args));
    };
    bool ok = call("sem_init", {memory + 4, 0, 1}) == 0;
    ok = ok && host.MemoryRead32(memory) == 0xA5A5A5A5 && host.MemoryRead32(memory + 8) == 0xA5A5A5A5;
    ok = ok && call("sem_trywait", {memory + 4}) == 0;
    ok = ok && call("sem_trywait", {memory + 4}) == -1 && host.MemoryRead32(host.GuestErrnoAddress()) == EAGAIN;
    ok = ok && call("sem_post", {memory + 4}) == 0 && call("sem_wait", {memory + 4}) == 0;
    const bool destroyed = call("sem_destroy", {memory + 4}) == 0;
    LOGI("Guest semaphore ABI self-check %s", ok && destroyed ? "OK" : "FAILED");

    const uint32_t division = host.GuestAlloc(16, 4);
    bool div_ok = division != 0;
    if (div_ok) {
        host.MemoryWrite32(division, 0xA5A5A5A5);
        host.MemoryWrite32(division + 12, 0xA5A5A5A5);
        div_ok = call("div", {division + 4, static_cast<uint32_t>(-7), 3}) ==
                     static_cast<int32_t>(division + 4) &&
                 static_cast<int32_t>(host.MemoryRead32(division + 4)) == -2 &&
                 static_cast<int32_t>(host.MemoryRead32(division + 8)) == -1 &&
                 host.MemoryRead32(division) == 0xA5A5A5A5 &&
                 host.MemoryRead32(division + 12) == 0xA5A5A5A5;
        host.GuestFree(division);
    }
    LOGI("Guest div ABI self-check %s", div_ok ? "OK" : "FAILED");

    const uint32_t sync = host.GuestAlloc(24, 4);
    bool cond_ok = sync != 0;
    if (cond_ok) {
        constexpr uint32_t kSentinel = 0xA5A5A5A5;
        const uint32_t mutex = sync + 4;
        const uint32_t cond = sync + 8;
        const uint32_t timeout = sync + 12;
        host.MemoryWrite32(sync, kSentinel);
        host.MemoryWrite32(sync + 20, kSentinel);
        host.MemoryWrite32(timeout, 2);
        host.MemoryWrite32(timeout + 4, 0);
        cond_ok = call("pthread_mutex_init", {mutex, 0}) == 0 &&
                  call("pthread_cond_init", {cond, 0}) == 0 &&
                  call("pthread_mutex_lock", {mutex}) == 0;
        if (cond_ok) {
            const uint32_t lock = host.GetThunkManager().GetOrCreateThunkAddress("pthread_mutex_lock");
            const uint32_t signal = host.GetThunkManager().GetOrCreateThunkAddress("pthread_cond_signal");
            const uint32_t unlock = host.GetThunkManager().GetOrCreateThunkAddress("pthread_mutex_unlock");
            std::atomic<bool> signalled{false};
            std::thread signaler([&] {
                if (host.RunAudioFunction(lock, {mutex}) == 0) {
                    signalled.store(host.RunAudioFunction(signal, {cond}) == 0,
                                    std::memory_order_release);
                    host.RunAudioFunction(unlock, {mutex});
                }
            });
            const int32_t wait_result = call(
                "pthread_cond_timedwait_relative_np", {cond, mutex, timeout});
            const bool signalled_before_return = signalled.load(std::memory_order_acquire);
            const bool unlocked = call("pthread_mutex_unlock", {mutex}) == 0;
            signaler.join();
            cond_ok = wait_result == 0 && signalled_before_return && unlocked;
        }
        if (cond_ok) {
            host.MemoryWrite32(timeout, 0);
            host.MemoryWrite32(timeout + 4, 1000000);
            const bool locked = call("pthread_mutex_lock", {mutex}) == 0;
            const int32_t wait_result = locked ? call(
                "pthread_cond_timedwait_relative_np", {cond, mutex, timeout}) : EINVAL;
            const bool unlocked = !locked || call("pthread_mutex_unlock", {mutex}) == 0;
            cond_ok = locked && wait_result == ETIMEDOUT && unlocked;
        }
        const bool cond_destroyed = call("pthread_cond_destroy", {cond}) == 0;
        const bool mutex_destroyed = call("pthread_mutex_destroy", {mutex}) == 0;
        cond_ok = cond_ok && cond_destroyed && mutex_destroyed &&
                  host.MemoryRead32(sync) == kSentinel &&
                  host.MemoryRead32(sync + 20) == kSentinel;
        host.GuestFree(sync);
    }
    LOGI("Guest relative condition wait self-check %s", cond_ok ? "OK" : "FAILED");

    bool tls_ok = call("pthread_key_create", {memory + 4, 0}) == 0;
    const uint32_t key = host.MemoryRead32(memory + 4);
    tls_ok = tls_ok && call("pthread_setspecific", {key, 0x1111}) == 0;
    uint32_t child_before = 0xFFFFFFFF, child_after = 0, child_id = 0;
    const uint32_t self = host.GetThunkManager().GetOrCreateThunkAddress("pthread_self");
    const uint32_t parent_id = host.RunFunction(self, {});
    const uint32_t get_tls = host.GetThunkManager().GetOrCreateThunkAddress("pthread_getspecific");
    const uint32_t set_tls = host.GetThunkManager().GetOrCreateThunkAddress("pthread_setspecific");
    std::thread child([&] {
        child_id = host.RunAudioFunction(self, {});
        child_before = host.RunAudioFunction(get_tls, {key});
        host.RunAudioFunction(set_tls, {key, 0x2222});
        child_after = host.RunAudioFunction(get_tls, {key});
    });
    child.join();
    tls_ok = tls_ok && child_before == 0 && child_after == 0x2222 &&
        call("pthread_getspecific", {key}) == 0x1111;
    call("pthread_setspecific", {key, 0});
    host.GuestFree(memory);
    host.MemoryWrite32(host.GuestErrnoAddress(), saved_errno);
    LOGI("Guest TLS isolation self-check %s", tls_ok ? "OK" : "FAILED");
    const bool ids_ok = parent_id && child_id && parent_id != child_id &&
        host.RunFunction(self, {}) == parent_id && call("pthread_equal", {parent_id, parent_id}) == 1 &&
        call("pthread_equal", {parent_id, child_id}) == 0;
    LOGI("Guest pthread identity self-check %s", ids_ok ? "OK" : "FAILED");
    const uint32_t jump = host.GuestAlloc(256, 8);
    const bool jump_ok = jump && call("setjmp", {jump}) == 0 && call("longjmp", {jump, 7}) == 7 &&
        call("longjmp", {jump, 0}) == 1;
    host.GuestFree(jump);
    LOGI("Guest jump state self-check %s", jump_ok ? "OK" : "FAILED");
    const uint32_t names = host.GuestAlloc(80, 1);
    if (!names) return false;
    strcpy(reinterpret_cast<char*>(host.GetGuestMemoryBase() + names), "__bridge_missing_symbol__");
    strcpy(reinterpret_cast<char*>(host.GetGuestMemoryBase() + names + 40), "memcpy");
    const bool symbols_ok = call("dlsym", {0, names}) == 0 && call("dlerror", {}) != 0 &&
        call("dlerror", {}) == 0 && call("dlsym", {0, names + 40}) != 0 && call("dlerror", {}) == 0;
    host.GuestFree(names);
    LOGI("Guest dlsym failure self-check %s", symbols_ok ? "OK" : "FAILED");
    return ok && destroyed && cond_ok && tls_ok && ids_ok && jump_ok && symbols_ok;
}

static bool CheckGuestSignalRoundTrip(DynarmicHost& host) {
    // NDK-assembled work/phase7_signal_fixture.S. Executable bytes remain for the
    // host lifetime; never recycle a block still in the executable-range list.
    const uint32_t fixture = host.GuestAlloc(128, 16);
    if (!fixture) return false;
    const uint32_t saved_errno = host.MemoryRead32(host.GuestErrnoAddress());
    const uint32_t flag = fixture + 40, ack = fixture + 48, work = fixture + 52;
    const uint32_t action = fixture + 64, old_action = fixture + 80;
    const auto call = [&](const char* name, std::vector<uint32_t> args) {
        return static_cast<int32_t>(host.RunFunction(host.GetThunkManager().GetOrCreateThunkAddress(name), args));
    };
    const uint32_t post = host.GetThunkManager().GetOrCreateThunkAddress("sem_post");
    const uint32_t wait = host.GetThunkManager().GetOrCreateThunkAddress("sem_wait");
    const uint32_t code[] = {0xE92D4010, 0xE59F4010, 0xE584D000, 0xE59F000C, 0xE59F300C,
                             0xE12FFF33, 0xE8BD8010, flag, ack, post};
    memcpy(host.GetGuestMemoryBase() + fixture, code, sizeof(code));
    host.AddExecutableRange(fixture, sizeof(code));
    host.InvalidateGuestCode(fixture, sizeof(code));
    host.MemoryWrite32(action, fixture);
    host.MemoryWrite32(action + 8, SA_RESTART);
    if (call("sem_init", {ack, 0, 0}) || call("sem_init", {work, 0, 0}) ||
        call("sigaction", {SIGUSR1, action, old_action})) return false;
    std::promise<std::array<uint32_t, 3>> ready;
    auto future = ready.get_future();
    std::thread receiver([&] {
        const uint32_t tid = host.RunFunction(host.GetThunkManager().GetOrCreateThunkAddress("gettid"), {});
        ready.set_value({tid, host.GuestStackBase(), host.GuestStackSize()});
        host.RunFunction(wait, {work});
    });
    const auto target = future.get();
    bool acknowledged = false;
    const int sent = GuestSignalSend(target[0], SIGUSR1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (sent == 0 && std::chrono::steady_clock::now() < deadline) {
        if (call("sem_trywait", {ack}) == 0) { acknowledged = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    call("sem_post", {work});
    receiver.join();
    const uint32_t sp = host.MemoryRead32(flag);
    const bool ok = acknowledged && sp >= target[1] && sp < target[1] + target[2];
    call("sigaction", {SIGUSR1, old_action, 0});
    call("sem_destroy", {ack});
    call("sem_destroy", {work});
    host.MemoryWrite32(host.GuestErrnoAddress(), saved_errno);
    LOGI("Guest signal round-trip self-check %s tid=%u sp=0x%08X stack=0x%08X/%u", ok ? "OK" : "FAILED",
         target[0], sp, target[1], target[2]);
    return ok;
}

static bool CheckReflectedMethod(JNIEnv* env, DynarmicHost& host, FakeJniEnv& fake) {
    if (env->PushLocalFrame(16) != JNI_OK) return false;
    struct LocalFrame { JNIEnv* env; ~LocalFrame() { env->PopLocalFrame(nullptr); } } frame{env};
    jclass string_class = env->FindClass("java/lang/String");
    jmethodID substring = string_class ? env->GetMethodID(string_class, "substring", "(II)Ljava/lang/String;") : nullptr;
    if (!substring || env->ExceptionCheck()) return false;
    jobject method = env->ToReflectedMethod(string_class, substring, JNI_FALSE);
    const uint32_t method_object = fake.AddObjectRef(method);
    const uint32_t from_reflection = host.MemoryRead32(fake.EnvTableGuestAddr() + 28);
    const uint32_t method_id = host.RunFunction(from_reflection, {fake.EnvTableGuestAddr(), method_object});
    jstring source = env->NewStringUTF("bridge");
    const uint32_t source_handle = fake.AddObjectRef(source);
    const uint32_t arguments = host.GuestAlloc(16, 8);
    if (!method_id || !source_handle || !arguments || env->ExceptionCheck()) return false;
    host.MemoryWrite32(arguments, 1);
    host.MemoryWrite32(arguments + 8, 4);
    const uint32_t call_object_a = host.MemoryRead32(fake.EnvTableGuestAddr() + 144);
    const uint32_t result = host.RunFunction(call_object_a,
        {fake.EnvTableGuestAddr(), source_handle, method_id, arguments});
    jobject value = fake.ResolveObject(result);
    bool ok = false;
    if (value && !env->ExceptionCheck()) {
        const char* text = env->GetStringUTFChars(static_cast<jstring>(value), nullptr);
        if (text) { ok = strcmp(text, "rid") == 0; env->ReleaseStringUTFChars(static_cast<jstring>(value), text); }
    }
    host.GuestFree(arguments);
    fake.DemoteObjectRef(method_object);
    fake.DemoteObjectRef(source_handle);
    fake.DemoteObjectRef(result);
    LOGI("Guest reflected method self-check %s", ok ? "OK" : "FAILED");
    return ok;
}

struct GuestImageInfo {
    uint32_t base;
    uint32_t size;
    const char* name;
};
static std::vector<GuestImageInfo> g_guest_images;
static std::mutex g_guest_images_mutex;

void RegisterGuestImages(const std::vector<GuestImageInfo>& images) {
    std::lock_guard<std::mutex> lock(g_guest_images_mutex);
    g_guest_images = images;
}

const char* GuestModuleNameForAddress(uint32_t address, uint32_t* base_out) {
    std::lock_guard<std::mutex> lock(g_guest_images_mutex);
    for (const GuestImageInfo& image : g_guest_images) {
        if (address >= image.base && address < image.base + image.size) {
            if (base_out) *base_out = image.base;
            return image.name;
        }
    }
    return nullptr;
}

static bool EnsureGameLoaded(JNIEnv* env, jobject context) {
    if (g_host) return true;
    if (!ReadApplicationPaths(env, context)) {
        LOGE("Could not resolve République APK path");
        return false;
    }

    std::vector<uint8_t> unity, il2cpp, cri, main, ovr;
    const auto guest_candidates = [](const char* name) {
        return std::vector<std::string>{std::string("assets/guestlib/") + name};
    };
    if (!ExtractFromApk(g_saved_apk_path, guest_candidates(cfg::kGuestUnity), unity) ||
        !ExtractFromApk(g_saved_apk_path, guest_candidates(cfg::kGuestIl2cpp), il2cpp) ||
        !ExtractFromApk(g_saved_apk_path, guest_candidates(cfg::kGuestCri), cri) ||
        !ExtractFromApk(g_saved_apk_path, guest_candidates(cfg::kGuestMain), main) ||
        !ExtractFromApk(g_saved_apk_path, guest_candidates(cfg::kGuestOvr), ovr)) {
        LOGE("One or more guest libraries are missing from assets/guestlib");
        return false;
    }

    g_host = std::make_unique<DynarmicHost>();
    if (!g_host->Initialize() || !GlSyncRegistrySelfCheck() ||
        !CheckGuestRuntimeAbi(*g_host) || !CheckGuestSignalRoundTrip(*g_host)) {
        g_host.reset();
        return false;
    }

    FakeJniEnv& fake = FakeJniEnv::Instance();
    fake.SetJavaVM(g_jvm);
    fake.Init(*g_host);
    fake.AttachHostEnv(env);
    if (!CheckReflectedMethod(env, *g_host, fake)) { g_host.reset(); return false; }

    for (const std::string& apk : g_apk_paths) {
        if (IndexApk(apk.c_str()) != 0) LOGE("IndexApk failed: %s", apk.c_str());
    }

    const std::vector<GuestModule> modules = {
        {cfg::kGuestMain, main.data(), main.size(), cfg::kMainBase},
        {cfg::kGuestUnity, unity.data(), unity.size(), cfg::kUnityBase},
        {cfg::kGuestIl2cpp, il2cpp.data(), il2cpp.size(), cfg::kIl2cppBase},
        {cfg::kGuestCri, cri.data(), cri.size(), cfg::kCriBase},
        {cfg::kGuestOvr, ovr.data(), ovr.size(), cfg::kOvrBase},
    };
    if (!g_host->LoadGuestModulesFromMemory(modules)) {
        LOGE("Guest ELF module load failed");
        g_host.reset();
        return false;
    }
    RegisterGuestImages({
        {cfg::kMainBase, static_cast<uint32_t>(main.size()), cfg::kGuestMain},
        {cfg::kUnityBase, static_cast<uint32_t>(unity.size()), cfg::kGuestUnity},
        {cfg::kIl2cppBase, static_cast<uint32_t>(il2cpp.size()), cfg::kGuestIl2cpp},
        {cfg::kCriBase, static_cast<uint32_t>(cri.size()), cfg::kGuestCri},
        {cfg::kOvrBase, static_cast<uint32_t>(ovr.size()), cfg::kGuestOvr},
    });
    if (!InstallUnityLz4Hook(*g_host) || !InstallUnityProfiledHotPaths(*g_host, cfg::kUnityBase) ||
        !ApplyIntroLogoSkipPatch(*g_host, cfg::kIl2cppBase) ||
        !ApplyMainMenuImageFadeSkipPatch(*g_host, cfg::kIl2cppBase) ||
        !ApplyGameOptionsFrameRatePolicyPatch(*g_host) ||
        !InstallTargetFrameRateWatch(*g_host) ||
        !InstallIl2cppInitCompletionWatch(*g_host) || !InstallCriErrorWatch(*g_host)) {
        g_host.reset();
        return false;
    }

    // Production route: retain only fail-closed exception guards and the proven
    // AndroidJNIHelper lifecycle/fallback fix. The constructor, Activator, AOT,
    // callback-field and InvokeObject probes were diagnostic-only hot-path hooks.
    if (!InstallNullReferenceWatch(*g_host) || !InstallRaiseWatch(*g_host) ||
        !InstallUnhandledExceptionWatch(*g_host) ||
        !InstallAndroidJniHelperAccessWatch(*g_host) ||
        !InstallAndroidJniHelperResultWatches(*g_host)) {
        g_host.reset();
        return false;
    }

    const uint32_t onload_result = g_host->RunFunction(GuestUnity(cfg::kUnityJniOnLoad), {
        fake.VmTableGuestAddr(), 0
    });
    LOGI("Guest libunity JNI_OnLoad returned 0x%08X", onload_result);
    if (onload_result != JNI_VERSION_1_6) {
        LOGE("Guest libunity JNI_OnLoad failed");
        g_host.reset();
        return false;
    }
    const uint32_t cri_result = g_host->RunFunction(cfg::kCriBase + cfg::kCriJniOnLoad, {
        fake.VmTableGuestAddr(), 0
    });
    const uint32_t ovr_result = g_host->RunFunction(cfg::kOvrBase + cfg::kOvrJniOnLoad, {
        fake.VmTableGuestAddr(), 0
    });
    LOGI("Guest JNI_OnLoad results: cri=0x%08X ovr=0x%08X", cri_result, ovr_result);
    const bool cri_version_ok = cri_result == JNI_VERSION_1_2 || cri_result == JNI_VERSION_1_4 || cri_result == JNI_VERSION_1_6;
    const bool ovr_version_ok = ovr_result == JNI_VERSION_1_2 || ovr_result == JNI_VERSION_1_4 || ovr_result == JNI_VERSION_1_6;
    if (!cri_version_ok || !ovr_version_ok) {
        LOGE("Guest CRI/OVR JNI_OnLoad failed");
        g_host.reset();
        return false;
    }
    StartAudioThread(g_host.get());
    LOGI("First guest instruction completed; PC path is active");
    return true;
}
