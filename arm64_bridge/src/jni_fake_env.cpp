#include "jni_fake_env.h"
#include "dynarmic_host.h"
#include "thunk_manager.h"
#include "thunk_types.h"
#include <android/log.h>
#include <pthread.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdarg>

#define LOG_TAG "Arm32Bridge_FakeJni"
#include "bridge_log.h"

// JNIEnv table slot count (through GetObjectRefType)
static thread_local JNIEnv* thread_local_env = nullptr;

static const uint32_t kEnvSlots = 233;

static const uint32_t kEnvTableBytes = kEnvSlots * 4;
// JavaVM (JNIInvokeInterface) slots: 0..7
static const uint32_t kVmSlots = 8;

FakeJniEnv& FakeJniEnv::Instance() {
    static FakeJniEnv inst;
    return inst;
}

void FakeJniEnv::Init(DynarmicHost& host) {
    if (m_host == &host) return;
    // Host was recreated (first load attempt failed before the Activity
    // existed) � old tables pointed into freed guest memory.
    m_host = &host;
    m_env_table = 0;
    m_vm_table = 0;
    m_next_obj_handle = 0x00020000;
    m_next_mid_handle = 0x00030000;
    m_next_fid_handle = 0x00040000;
    m_objects.clear();
    m_objects_by_hash.clear();
    m_lru.clear();
    m_methods.clear();
    m_fields.clear();
    m_method_ids.clear();
    m_field_ids.clear();
    m_direct_bufs.clear();
    m_array_elements.clear();
    m_jstr_utf.clear();
    ThunkManager& tm = host.GetThunkManager();

    // Guest memory for tables
    m_env_table = host.GuestAlloc(kEnvTableBytes, 4);
    m_vm_table = host.GuestAlloc(kVmSlots * 4, 4);

    uint8_t* mem = host.GetGuestMemoryBase();
    auto register_slot = [&](uint32_t table, uint32_t offset, bool is_vm, const char* name) {
        uint32_t thunk = tm.RegisterThunk(
            std::string("jni_") + (is_vm ? "vm" : "env") + "_" + std::to_string(offset),
            [this, offset, is_vm](Dynarmic::A32::Jit& jit, DynarmicHost&) {
                if (is_vm) DispatchVmSlot(offset, jit);
                else DispatchEnvSlot(offset, jit);
            });
        host.MemoryWrite32(table + offset, thunk);
    };

    // Register all env slots: known offsets get real handlers (via DispatchEnvSlot
    // switch), everything else logs-and-zeroes on first use.
    for (uint32_t off = 0; off < kEnvSlots; ++off) {
        register_slot(m_env_table, off * 4, false, nullptr);
    }
    for (uint32_t off = 0; off < kVmSlots; ++off) {
        register_slot(m_vm_table, off * 4, true, nullptr);
    }

    // JNI ABI: guest code does `ldr r, [env]; ldr r, [r, #off]; blx r` �
    // i.e. [env] must point to the function-pointer table itself. Slot 0 of
    // both tables is reserved by the spec, so make it a self-pointer.
    host.MemoryWrite32(m_env_table, m_env_table);
    host.MemoryWrite32(m_vm_table, m_vm_table);

    LOGI("Fake JNI tables: env=0x%08X (%u bytes), vm=0x%08X", m_env_table, kEnvTableBytes, m_vm_table);
}

void FakeJniEnv::AttachHostEnv(JNIEnv* env) {
    thread_local_env = env;
}

// A thread that attaches itself and then exits still attached leaves the
// runtime's Thread object behind, and a checked build aborts the process over
// it. Guest worker threads (audio, streaming, a JIT's own helpers) come and go
// and none of them knows it is talking to a bridge, so the detach hangs off a
// TLS destructor instead of being left to whoever created the thread. Only
// threads attached here get the key set - detaching one the VM owns is worse
// than leaking it.
static pthread_key_t s_detach_key;
static JavaVM* s_detach_vm = nullptr;

static void DetachOnThreadExit(void*) {
    if (s_detach_vm) s_detach_vm->DetachCurrentThread();
    thread_local_env = nullptr;
}

JNIEnv* FakeJniEnv::GetEnv() {
    if (thread_local_env) return thread_local_env;
    if (!m_vm) return nullptr;
    JNIEnv* env = nullptr;
    if (m_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK) {
        thread_local_env = env;
        return env;
    }
    if (m_vm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
        thread_local_env = env;
        static std::once_flag once;
        std::call_once(once, [this] {
            s_detach_vm = m_vm;
            pthread_key_create(&s_detach_key, DetachOnThreadExit);
        });
        pthread_setspecific(s_detach_key, reinterpret_cast<void*>(1));
    }
    return env;
}

// -------------------------------------------------------------
// Handles
// -------------------------------------------------------------

// Cap on live global refs. ART's own limit is 51200 and hitting it kills the
// process outright, so the table has to stay well under it and reclaim rather
// than stop caching - a "cache full, hand out a ref and forget it" branch is the
// original leak with a delay on it.
static const size_t kMaxObjectRefs = 8192;

int32_t FakeJniEnv::IdentityHash(JNIEnv* env, jobject o) {
    static jclass system = nullptr;
    static jmethodID identity_hash = nullptr;
    if (!identity_hash) {
        jclass local = env->FindClass("java/lang/System");
        if (!local) { env->ExceptionClear(); return 0; }
        if (!system) system = static_cast<jclass>(env->NewGlobalRef(local));
        env->DeleteLocalRef(local);
        identity_hash = env->GetStaticMethodID(system, "identityHashCode", "(Ljava/lang/Object;)I");
        if (!identity_hash) { env->ExceptionClear(); return 0; }
    }
    int32_t h = env->CallStaticIntMethod(system, identity_hash, o);
    if (env->ExceptionCheck()) { env->ExceptionClear(); return 0; }
    return h;
}

void FakeJniEnv::TouchRef(uint32_t handle) const {
    auto it = m_objects.find(handle);
    if (it != m_objects.end() && it->second.pins == 0) {
        m_lru.splice(m_lru.begin(), m_lru, it->second.lru);
    }
}

void FakeJniEnv::EvictOneRef(JNIEnv* env) {
    if (m_lru.empty()) return;  // everything pinned; the table grows past the cap
    uint32_t handle = m_lru.back();
    m_lru.pop_back();

    auto it = m_objects.find(handle);
    if (it == m_objects.end()) return;

    auto bucket = m_objects_by_hash.find(it->second.hash);
    if (bucket != m_objects_by_hash.end()) {
        auto& v = bucket->second;
        v.erase(std::remove(v.begin(), v.end(), handle), v.end());
        if (v.empty()) m_objects_by_hash.erase(bucket);
    }
    // The UTF-8 copy of a jstring lives as long as the jstring's handle does;
    // leaving it behind strands the arena block and, worse, keeps a dead handle
    // mapped to it.
    auto utf = m_jstr_utf.find(handle);
    if (utf != m_jstr_utf.end()) {
        if (m_host) m_host->GuestFree(utf->second);
        m_jstr_utf.erase(utf);
    }
    env->DeleteGlobalRef(it->second.global);
    m_objects.erase(it);
}

uint32_t FakeJniEnv::AddObjectRef(jobject o) {
    if (!o) return 0;
    JNIEnv* env = GetEnv();
    if (!env) return 0;
    int32_t hash = IdentityHash(env, o);

    std::lock_guard<std::mutex> lk(m_mutex);
    for (uint32_t h : m_objects_by_hash[hash]) {
        auto it = m_objects.find(h);
        if (it != m_objects.end() && env->IsSameObject(it->second.global, o)) {
            TouchRef(h);
            return h;
        }
    }
    // Eviction rehashes both maps, so nothing looked up above survives it.
    if (m_objects.size() >= kMaxObjectRefs) EvictOneRef(env);

    jobject global = env->NewGlobalRef(o);
    if (!global) return 0;
    uint32_t h = m_next_obj_handle;
    m_next_obj_handle += 4;

    ObjRef& ref = m_objects[h];
    ref.global = global;
    ref.hash = hash;
    ref.pins = 0;
    m_lru.push_front(h);
    ref.lru = m_lru.begin();
    m_objects_by_hash[hash].push_back(h);
    return h;
}

jobject FakeJniEnv::ResolveObject(uint32_t handle) const {
    if (!handle) return nullptr;
    std::lock_guard<std::mutex> lk(m_mutex);
    auto it = m_objects.find(handle);
    if (it == m_objects.end()) return nullptr;
    // Resolving counts as using: an object a slot is part way through working
    // with must not be the one another thread picks to evict.
    TouchRef(handle);
    return it->second.global;
}

void FakeJniEnv::PinObjectRef(uint32_t handle, int delta) {
    std::lock_guard<std::mutex> lk(m_mutex);
    auto it = m_objects.find(handle);
    if (it == m_objects.end()) return;
    ObjRef& ref = it->second;
    const bool was = ref.pins > 0;
    ref.pins += delta;
    if (ref.pins < 0) ref.pins = 0;
    const bool now = ref.pins > 0;
    if (now == was) return;
    if (now) {
        m_lru.erase(ref.lru);
    } else {
        m_lru.push_front(handle);
        ref.lru = m_lru.begin();
    }
}

void FakeJniEnv::DemoteObjectRef(uint32_t handle) {
    std::lock_guard<std::mutex> lk(m_mutex);
    auto it = m_objects.find(handle);
    if (it != m_objects.end() && it->second.pins == 0) {
        m_lru.splice(m_lru.end(), m_lru, it->second.lru);
    }
}

// One id per jmethodID. Guests that reach Java through reflection ask for the
// same method on every call; appending unconditionally grew this table, and the
// signature string in each entry, for the life of the process.
//
// Dedup by mid alone is only correct because handles are interned: clsHandle for
// a given class is now stable, so the first registration's copy stays right.
uint32_t FakeJniEnv::AddMethod(jmethodID mid, bool isStatic, uint32_t clsHandle,
                               const std::string& name, const std::string& sig) {
    if (!mid) return 0;
    std::lock_guard<std::mutex> lk(m_mutex);
    auto known = m_method_ids.find(mid);
    if (known != m_method_ids.end()) return known->second;

    uint32_t h = m_next_mid_handle;
    m_next_mid_handle += 4;
    MethodInfo mi;
    mi.mid = mid;
    mi.isStatic = isStatic;
    mi.clsHandle = clsHandle;
    mi.name = name;
    mi.sig = sig;
    if (name == "loadLibrary" || name == "load") {
        LOGI("[JNI] cached method name=%s sig=%s static=%d", name.c_str(), sig.c_str(), isStatic ? 1 : 0);
    }
    m_methods[h] = mi;
    m_method_ids[mid] = h;
    return h;
}

uint32_t FakeJniEnv::AddField(jfieldID fid) {
    if (!fid) return 0;
    std::lock_guard<std::mutex> lk(m_mutex);
    auto known = m_field_ids.find(fid);
    if (known != m_field_ids.end()) return known->second;

    uint32_t h = m_next_fid_handle;
    m_next_fid_handle += 4;
    m_fields[h] = fid;
    m_field_ids[fid] = h;
    return h;
}

// -------------------------------------------------------------
// Signature parsing & guest arg reading
// -------------------------------------------------------------

std::vector<char> FakeJniEnv::ParseParams(const std::string& sig) const {
    std::vector<char> out;
    size_t i = sig.find('(');
    if (i == std::string::npos) return out;
    for (++i; i < sig.size() && sig[i] != ')'; ++i) {
        char c = sig[i];
        if (c == 'L') {
            while (sig[i] != ';') ++i;
            out.push_back('L');
        } else if (c == '[') {
            while (sig[i] == '[') ++i;
            if (sig[i] == 'L') while (sig[i] != ';') ++i;
            out.push_back('L');
        } else {
            out.push_back(c);
        }
    }
    return out;
}

// A-variant: r3 points to an array of jvalue (8 bytes each on ARM32)
std::vector<FakeJniEnv::GuestArg> FakeJniEnv::ReadArgsA(Dynarmic::A32::Jit& jit,
                                                        const std::vector<char>& params,
                                                        uint32_t jvalue_guest) {
    std::vector<GuestArg> args;
    uint8_t* mem = m_host->GetGuestMemoryBase();
    uint32_t p = jvalue_guest;
    for (char t : params) {
        GuestArg a;
        a.type = t;
        switch (t) {
            case 'L':
                a.obj = ResolveObject(*reinterpret_cast<uint32_t*>(mem + p));
                break;
            case 'J':
                a.v = *reinterpret_cast<uint64_t*>(mem + p);
                break;
            case 'D':
                a.v = *reinterpret_cast<uint64_t*>(mem + p);
                break;
            case 'F': {
                float f; memcpy(&f, mem + p, 4);
                a.v = *reinterpret_cast<uint32_t*>(mem + p);
                break;
            }
            default:
                a.v = static_cast<uint32_t>(*reinterpret_cast<uint32_t*>(mem + p));
                break;
        }
        args.push_back(a);
        p += 8;
    }
    return args;
}

// V-variant: r3 is a va_list (guest stack pointer); 64-bit slots aligned to 8
std::vector<FakeJniEnv::GuestArg> FakeJniEnv::ReadArgsV(Dynarmic::A32::Jit& jit,
                                                        const std::vector<char>& params,
                                                        uint32_t va_list_guest) {
    std::vector<GuestArg> args;
    uint8_t* mem = m_host->GetGuestMemoryBase();
    uint32_t p = va_list_guest;
    for (char t : params) {
        GuestArg a;
        a.type = t;
        if (t == 'J' || t == 'D') {
            p = (p + 7) & ~7u;
            a.v = *reinterpret_cast<uint64_t*>(mem + p);
            p += 8;
        } else {
            uint32_t raw = *reinterpret_cast<uint32_t*>(mem + p);
            if (t == 'L') a.obj = ResolveObject(raw);
            else a.v = raw;
            p += 4;
        }
        args.push_back(a);
    }
    return args;
}

// Plain vararg: first vararg in r3, then guest stack
std::vector<FakeJniEnv::GuestArg> FakeJniEnv::ReadArgsVar(Dynarmic::A32::Jit& jit,
                                                          const std::vector<char>& params) {
    std::vector<GuestArg> args;
    uint8_t* mem = m_host->GetGuestMemoryBase();
    uint32_t p = jit.Regs()[13];
    bool first = true;
    for (char t : params) {
        GuestArg a;
        a.type = t;
        uint32_t raw;
        if (first) { raw = jit.Regs()[3]; first = false; }
        else {
            raw = *reinterpret_cast<uint32_t*>(mem + p);
            p += 4;
        }
        if (t == 'J' || t == 'D') {
            uint32_t hi;
            if (first) { hi = *reinterpret_cast<uint32_t*>(mem + p); p += 4; first = false; }
            else { hi = *reinterpret_cast<uint32_t*>(mem + p); p += 4; }
            a.v = static_cast<uint64_t>(raw) | (static_cast<uint64_t>(hi) << 32);
        } else if (t == 'L') {
            a.obj = ResolveObject(raw);
        } else {
            a.v = raw;
        }
        args.push_back(a);
    }
    return args;
}

// -------------------------------------------------------------
// Host-side call marshaling
// -------------------------------------------------------------

static void LogAndPreserveHostException(JNIEnv* env, const char* context) {
    if (!env || !env->ExceptionCheck()) return;
    jthrowable thrown = env->ExceptionOccurred();
    env->ExceptionClear();
    std::string text;
    jclass throwable = env->FindClass("java/lang/Throwable");
    jmethodID to_string = throwable ? env->GetMethodID(throwable, "toString", "()Ljava/lang/String;") : nullptr;
    jstring value = (thrown && to_string) ? static_cast<jstring>(env->CallObjectMethod(thrown, to_string)) : nullptr;
    if (value && !env->ExceptionCheck()) {
        const char* utf = env->GetStringUTFChars(value, nullptr);
        if (utf) {
            text = utf;
            env->ReleaseStringUTFChars(value, utf);
        }
    }
    if (value) env->DeleteLocalRef(value);
    if (throwable) env->DeleteLocalRef(throwable);
    if (env->ExceptionCheck()) env->ExceptionClear();
    if (thrown) {
        env->Throw(thrown);
        env->DeleteLocalRef(thrown);
    }
    LOGE("[JNI] host exception after %s: %s", context, text.empty() ? "<unknown>" : text.c_str());
}

void FakeJniEnv::CallMethod(Dynarmic::A32::Jit& jit, uint32_t mid_handle, uint32_t obj_handle,
                            char retType, const std::vector<GuestArg>& args, bool isStatic) {
    MethodInfo mi;
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        auto mi_it = m_methods.find(mid_handle);
        if (mi_it == m_methods.end()) {
            LOGE("CallMethod: unknown method handle 0x%08X", mid_handle);
            jit.Regs()[0] = 0;
            jit.Regs()[1] = 0;
            return;
        }
        mi = mi_it->second; // copy; host JNI call below runs without the lock
    }
    if (isStatic && mi.sig == "(Ljava/lang/String;)V") {
        LOGI("[JNI] static String call name=%s", mi.name.c_str());
    }
    if ((mi.name == "loadLibrary" || mi.name == "load") &&
        (mi.sig == "(Ljava/lang/String;)V" || mi.sig == "(Ljava/lang/String;)Z")) {
        // libunity asks Java to load libil2cpp after the bridge has already loaded
        // that guest ELF from assets/guestlib.
        LOGI("[JNI] Suppressed guest library load; guest library is already mapped");
        jit.Regs()[0] = mi.sig.back() == 'Z' ? 1u : 0u;
        return;
    }
    jobject obj = ResolveObject(obj_handle);
    if (!obj) {
        jit.Regs()[0] = 0;
        jit.Regs()[1] = 0;
        return;
    }
    {
    }

    std::vector<jvalue> jvs(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        jvalue& jv = jvs[i];
        const GuestArg& a = args[i];
        switch (a.type) {
            case 'Z': jv.z = static_cast<jboolean>(a.v); break;
            case 'B': jv.b = static_cast<jbyte>(a.v); break;
            case 'C': jv.c = static_cast<jchar>(a.v); break;
            case 'S': jv.s = static_cast<jshort>(a.v); break;
            case 'I': jv.i = static_cast<jint>(a.v); break;
            case 'J': jv.j = static_cast<jlong>(a.v); break;
            case 'F': { float f; memcpy(&f, &a.v, 4); jv.f = f; break; }
            case 'D': { double d; memcpy(&d, &a.v, 8); jv.d = d; break; }
            case 'L': jv.l = a.obj; break;
        }
    }

    JNIEnv* env = GetEnv();
    bool pending = env->ExceptionCheck();
    if (pending) { env->ExceptionClear(); }

    switch (retType) {
        case 'V':
            if (isStatic) env->CallStaticVoidMethodA(static_cast<jclass>(obj), mi.mid, jvs.data());
            else env->CallVoidMethodA(obj, mi.mid, jvs.data());
            break;
        case 'Z': {
            jboolean r = isStatic ? env->CallStaticBooleanMethodA(static_cast<jclass>(obj), mi.mid, jvs.data())
                                  : env->CallBooleanMethodA(obj, mi.mid, jvs.data());
            jit.Regs()[0] = r ? 1 : 0;
            break;
        }
        case 'B': {
            jbyte r = isStatic ? env->CallStaticByteMethodA(static_cast<jclass>(obj), mi.mid, jvs.data())
                              : env->CallByteMethodA(obj, mi.mid, jvs.data());
            jit.Regs()[0] = static_cast<uint32_t>(static_cast<int32_t>(r));
            break;
        }
        case 'C': {
            jchar r = isStatic ? env->CallStaticCharMethodA(static_cast<jclass>(obj), mi.mid, jvs.data())
                              : env->CallCharMethodA(obj, mi.mid, jvs.data());
            jit.Regs()[0] = r;
            break;
        }
        case 'S': {
            jshort r = isStatic ? env->CallStaticShortMethodA(static_cast<jclass>(obj), mi.mid, jvs.data())
                               : env->CallShortMethodA(obj, mi.mid, jvs.data());
            jit.Regs()[0] = static_cast<uint32_t>(static_cast<int32_t>(r));
            break;
        }
        case 'I': {
            jint r = isStatic ? env->CallStaticIntMethodA(static_cast<jclass>(obj), mi.mid, jvs.data())
                              : env->CallIntMethodA(obj, mi.mid, jvs.data());
            jit.Regs()[0] = static_cast<uint32_t>(r);
            break;
        }
        case 'J': {
            jlong r = isStatic ? env->CallStaticLongMethodA(static_cast<jclass>(obj), mi.mid, jvs.data())
                               : env->CallLongMethodA(obj, mi.mid, jvs.data());
            jit.Regs()[0] = static_cast<uint32_t>(r & 0xFFFFFFFFu);
            jit.Regs()[1] = static_cast<uint32_t>(static_cast<uint64_t>(r) >> 32);
            break;
        }
        case 'F': {
            jfloat r = isStatic ? env->CallStaticFloatMethodA(static_cast<jclass>(obj), mi.mid, jvs.data())
                                : env->CallFloatMethodA(obj, mi.mid, jvs.data());
            WriteSF(jit, 0, r);
            break;
        }
        case 'D': {
            jdouble r = isStatic ? env->CallStaticDoubleMethodA(static_cast<jclass>(obj), mi.mid, jvs.data())
                                 : env->CallDoubleMethodA(obj, mi.mid, jvs.data());
            WriteDF(jit, 0, r);
            break;
        }
        case 'L': {
            jobject r = isStatic ? env->CallStaticObjectMethodA(static_cast<jclass>(obj), mi.mid, jvs.data())
                                 : env->CallObjectMethodA(obj, mi.mid, jvs.data());
            jit.Regs()[0] = AddObjectRef(r);
            break;
        }
        default:
            jit.Regs()[0] = 0;
            break;
    }
    if (mi.name == "forName" && mi.sig == "(Ljava/lang/String;)Ljava/lang/Class;" &&
        args.size() == 1 && args[0].obj) {
        const char* requested = env->GetStringUTFChars(static_cast<jstring>(args[0].obj), nullptr);
        const std::string requested_name = requested ? requested : "";
        const bool optional_immersive = requested_name == "com.darkwindmedia.immersive.Immersive";
        if (requested) env->ReleaseStringUTFChars(static_cast<jstring>(args[0].obj), requested);
        if (optional_immersive && env->ExceptionCheck()) {
            env->ExceptionClear();
            LOGI("[JNI] suppressed expected optional class probe: %s", requested_name.c_str());
        }
    }
    if (env->ExceptionCheck()) {
        char context[256];
        snprintf(context, sizeof(context), "%s%s ret=%c caller_lr=0x%08X",
                 mi.name.c_str(), mi.sig.c_str(), retType, jit.Regs()[14]);
        LogAndPreserveHostException(env, context);
    }
}

uint32_t FakeJniEnv::InternReflectedMethod(jobject reflected) {
    JNIEnv* e = GetEnv();
    if (!e || !reflected) return 0;
    const jmethodID method = e->FromReflectedMethod(reflected);
    if (!method || e->ExceptionCheck()) return 0;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto known = m_method_ids.find(method);
        if (known != m_method_ids.end()) return known->second;
    }
    if (e->PushLocalFrame(32) != JNI_OK) return 0;
    struct LocalFrame { JNIEnv* env; ~LocalFrame() { env->PopLocalFrame(nullptr); } } frame{e};
    const jclass reflected_class = e->GetObjectClass(reflected);
    const jclass constructor_class = e->FindClass("java/lang/reflect/Constructor");
    const jclass class_class = e->FindClass("java/lang/Class");
    if (!reflected_class || !constructor_class || !class_class || e->ExceptionCheck()) return 0;
    const bool constructor = e->IsInstanceOf(reflected, constructor_class);
    const auto get_declaring = e->GetMethodID(reflected_class, "getDeclaringClass", "()Ljava/lang/Class;");
    const auto get_parameters = e->GetMethodID(reflected_class, "getParameterTypes", "()[Ljava/lang/Class;");
    const auto get_modifiers = e->GetMethodID(reflected_class, "getModifiers", "()I");
    const auto get_name = e->GetMethodID(reflected_class, "getName", "()Ljava/lang/String;");
    const auto get_class_name = e->GetMethodID(class_class, "getName", "()Ljava/lang/String;");
    if (!get_declaring || !get_parameters || !get_modifiers || !get_name || !get_class_name || e->ExceptionCheck()) return 0;
    const auto utf = [e](jstring text) {
        std::string value;
        if (text) {
            const char* chars = e->GetStringUTFChars(text, nullptr);
            if (chars) { value = chars; e->ReleaseStringUTFChars(text, chars); }
            e->DeleteLocalRef(text);
        }
        return value;
    };
    const auto descriptor = [&](jobject type) -> std::string {
        auto text = static_cast<jstring>(e->CallObjectMethod(type, get_class_name));
        if (e->ExceptionCheck()) return {};
        std::string name = utf(text);
        if (name.empty()) return {};
        if (name == "void") return "V";
        if (name == "boolean") return "Z";
        if (name == "byte") return "B";
        if (name == "char") return "C";
        if (name == "short") return "S";
        if (name == "int") return "I";
        if (name == "long") return "J";
        if (name == "float") return "F";
        if (name == "double") return "D";
        std::replace(name.begin(), name.end(), '.', '/');
        return name.front() == '[' ? name : "L" + name + ";";
    };
    const jobject declaring = e->CallObjectMethod(reflected, get_declaring);
    if (!declaring || e->ExceptionCheck()) return 0;
    const auto parameters = static_cast<jobjectArray>(e->CallObjectMethod(reflected, get_parameters));
    if (!parameters || e->ExceptionCheck()) return 0;
    const jint modifiers = e->CallIntMethod(reflected, get_modifiers);
    if (e->ExceptionCheck()) return 0;
    std::string name = "<init>";
    if (!constructor) {
        auto text = static_cast<jstring>(e->CallObjectMethod(reflected, get_name));
        if (e->ExceptionCheck()) return 0;
        name = utf(text);
    }
    std::string signature = "(";
    for (jsize i = 0, count = e->GetArrayLength(parameters); i < count; ++i) {
        jobject type = e->GetObjectArrayElement(parameters, i);
        if (!type || e->ExceptionCheck()) return 0;
        std::string part = descriptor(type);
        e->DeleteLocalRef(type);
        if (part.empty() || e->ExceptionCheck()) return 0;
        signature += part;
    }
    signature += ")";
    if (constructor) signature += "V";
    else {
        const auto get_return = e->GetMethodID(reflected_class, "getReturnType", "()Ljava/lang/Class;");
        if (!get_return) return 0;
        jobject type = e->CallObjectMethod(reflected, get_return);
        if (!type || e->ExceptionCheck()) return 0;
        std::string part = descriptor(type);
        if (part.empty() || e->ExceptionCheck()) return 0;
        signature += part;
    }
    const uint32_t declaring_handle = AddObjectRef(declaring);
    if (!declaring_handle || name.empty() || e->ExceptionCheck()) return 0;
    const uint32_t result = AddMethod(method, (modifiers & 8) != 0, declaring_handle, name, signature);
    static std::atomic<unsigned> logged{0};
    if (logged.fetch_add(1) < 24) LOGI("[JNI] reflected method %s%s -> 0x%08X", name.c_str(), signature.c_str(), result);
    return result;
}

// -------------------------------------------------------------
// Direct ByteBuffer bridging
// -------------------------------------------------------------

uint32_t FakeJniEnv::RegisterDirectBuffer(jobject buf) {
    JNIEnv* env = GetEnv();
    if (!env || !buf) return 0;
    std::lock_guard<std::mutex> lock(m_mutex);
    // fmodProcess hands us the same ByteBuffer on every audio callback (~125/s).
    // A fresh entry per call burned a permanent JNI global ref and a
    // capacity-sized arena block each time. One entry per live buffer, so the
    // scan is over one or two elements.
    for (const auto& [h, db] : m_direct_bufs) {
        if (env->IsSameObject(db.jobj, buf)) {
            memcpy(m_host->GetGuestMemoryBase() + db.scratch, db.hostPtr, static_cast<size_t>(db.capacity));
            return h;
        }
    }
    void* p = env->GetDirectBufferAddress(buf);
    jlong cap = env->GetDirectBufferCapacity(buf);
    if (!p || cap <= 0 || static_cast<uint64_t>(cap) > UINT32_MAX) return 0;
    uint32_t handle = m_next_obj_handle;
    m_next_obj_handle += 4;
    DirectBuf db;
    db.jobj = env->NewGlobalRef(buf);
    if (!db.jobj) return 0;
    db.hostPtr = p;
    db.capacity = cap;
    db.scratch = m_host->GuestAlloc(static_cast<size_t>(cap), 16);
    if (!db.scratch) { env->DeleteGlobalRef(db.jobj); return 0; }
    memcpy(m_host->GetGuestMemoryBase() + db.scratch, p, static_cast<size_t>(cap));
    m_direct_bufs[handle] = db;
    return handle;
}

void FakeJniEnv::CommitDirectBuffer(uint32_t handle) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_direct_bufs.find(handle);
    if (it == m_direct_bufs.end()) return;
    DirectBuf& db = it->second;
    memcpy(db.hostPtr, m_host->GetGuestMemoryBase() + db.scratch, static_cast<size_t>(db.capacity));
}

jlong FakeJniEnv::DirectBufferCapacity(uint32_t handle) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_direct_bufs.find(handle);
    return (it != m_direct_bufs.end()) ? it->second.capacity : 0;
}

// -------------------------------------------------------------
// Dispatch
// -------------------------------------------------------------

// Helpers
static const char* ReadGuestStr(DynarmicHost& host, uint32_t gaddr) {
    if (!gaddr) return nullptr;
    return reinterpret_cast<const char*>(host.GetGuestMemoryBase() + gaddr);
}

void FakeJniEnv::DispatchVmSlot(uint32_t offset, Dynarmic::A32::Jit& jit) {
    static std::atomic<unsigned> logged{0};
    if (logged.fetch_add(1) < 32)
        LOGI("[JNI-VM] slot %u (off 0x%X) r1=0x%08X", offset / 4, offset, jit.Regs()[1]);
    switch (offset) {
        case 16: // AttachCurrentThread(vm, penv, args)
        case 24: // GetEnv(vm, penv, version)
        case 28: // AttachCurrentThreadAsDaemon(vm, penv, args)
        {
            uint32_t out = jit.Regs()[1];
            if (out) m_host->MemoryWrite32(out, m_env_table);
            jit.Regs()[0] = 0; // JNI_OK
            break;
        }
        case 20: // DetachCurrentThread(vm)
        {
            jit.Regs()[0] = 0; // JNI_OK
            break;
        }
        default:
            static bool warned_vm[64];
            if (offset / 4 < 64 && !warned_vm[offset / 4]) {
                warned_vm[offset / 4] = true;
                LOGI("[JNI-VM] unhandled slot %u (off 0x%X) at PC 0x%08X", offset / 4, offset, jit.Regs()[15]);
            }
            jit.Regs()[0] = 0;
            break;
    }
}

void FakeJniEnv::DispatchEnvSlot(uint32_t offset, Dynarmic::A32::Jit& jit) {
    JNIEnv* e = GetEnv();
    uint8_t* mem = m_host->GetGuestMemoryBase();

    if (!e) {
        LOGE("[JNI] no host env attached (slot %u)", offset / 4);
        jit.Regs()[0] = 0;
        return;
    }

    const uint32_t slot = offset / 4;
    const uint32_t call_base = slot >= 34 && slot <= 63 ? 34 : slot >= 114 && slot <= 143 ? 114 : 0;
    if (call_base) {
        const uint32_t method = jit.Regs()[2], object = jit.Regs()[1];
        std::vector<char> params;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto found = m_methods.find(method);
            if (found == m_methods.end()) {
                LOGE("Unknown JNI method 0x%08X at slot %u", method, slot);
                jit.Regs()[0] = jit.Regs()[1] = 0;
                return;
            }
            params = ParseParams(found->second.sig);
        }
        const unsigned variant = (slot - call_base) % 3;
        const auto args = variant == 0 ? ReadArgsVar(jit, params) :
            variant == 1 ? ReadArgsV(jit, params, jit.Regs()[3]) : ReadArgsA(jit, params, jit.Regs()[3]);
        CallMethod(jit, method, object, "LZBCSIJFDV"[(slot - call_base) / 3], args, call_base == 114);
        return;
    }

    switch (offset) {
        case 860: // RegisterNatives: host registration is performed by JNI_OnLoad.
            // The guest's JNI_OnLoad still needs the real success code so its own
            // initialization can continue; host wrappers are registered separately.
            jit.Regs()[0] = JNI_OK;
            break;
        case 76: // PushLocalFrame
            jit.Regs()[0] = 0;
            break;
        case 80: // PopLocalFrame
            jit.Regs()[0] = jit.Regs()[1];
            break;
        case 104: // EnsureLocalCapacity
            jit.Regs()[0] = 0;
            break;

        // ---- class / object basics ----
        case 28: { // FromReflectedMethod
            jit.Regs()[0] = InternReflectedMethod(ResolveObject(jit.Regs()[1]));
            break;
        }
        case 32: { // FromReflectedField
            jobject reflected = ResolveObject(jit.Regs()[1]);
            const jfieldID field = reflected ? e->FromReflectedField(reflected) : nullptr;
            jit.Regs()[0] = field ? AddField(field) : 0;
            break;
        }
        case 24: { // FindClass
            const char* name = ReadGuestStr(*m_host, jit.Regs()[1]);
            jclass c = e->FindClass(name);
            if (!c) { e->ExceptionClear(); jit.Regs()[0] = 0; }
            else jit.Regs()[0] = AddObjectRef(c);
            break;
        }
        case 40: { // GetSuperclass
            jobject o = ResolveObject(jit.Regs()[1]);
            jclass c = e->GetSuperclass(static_cast<jclass>(o));
            jit.Regs()[0] = c ? AddObjectRef(c) : 0;
            break;
        }
        // Handles are shared, so dropping the host ref in DeleteGlobalRef/
        // DeleteLocalRef would invalidate any copy the guest still holds. What
        // the three calls do instead is set how willing the table is to reclaim
        // the entry - which is as close to the real lifetime as a shared handle
        // gets, and it is what keeps the table off ART's 51200 limit.
        case 84: { // NewGlobalRef -> same handle, plus a pin
            jobject o = ResolveObject(jit.Regs()[1]);
            if (!o) { jit.Regs()[0] = 0; break; }
            PinObjectRef(jit.Regs()[1], +1);
            jit.Regs()[0] = jit.Regs()[1];
            break;
        }
        case 88: // DeleteGlobalRef -> drop the pin
            PinObjectRef(jit.Regs()[1], -1);
            jit.Regs()[0] = 0;
            break;
        case 92: // DeleteLocalRef -> next in line to be evicted
            DemoteObjectRef(jit.Regs()[1]);
            jit.Regs()[0] = 0;
            break;
        case 100: { // NewLocalRef
            jobject o = ResolveObject(jit.Regs()[1]);
            jit.Regs()[0] = AddObjectRef(o);
            break;
        }
        case 124: { // GetObjectClass
            jobject o = ResolveObject(jit.Regs()[1]);
            if (!o) { jit.Regs()[0] = 0; break; } // guest expects null, ART would abort
            jclass c = e->GetObjectClass(o);
            jit.Regs()[0] = c ? AddObjectRef(c) : 0;
            break;
        }
        case 128: { // IsInstanceOf
            jobject o = ResolveObject(jit.Regs()[1]);
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[2]));
            jit.Regs()[0] = e->IsInstanceOf(o, c) ? 1 : 0;
            break;
        }

        // ---- method/field IDs ----
        case 132: { // GetMethodID
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[1]));
            const char* name = ReadGuestStr(*m_host, jit.Regs()[2]);
            const char* sig = ReadGuestStr(*m_host, jit.Regs()[3]);
            if (!c) { jit.Regs()[0] = 0; break; }
            jmethodID mid = e->GetMethodID(c, name, sig);
            if (!mid) {
                e->ExceptionClear();
                LOGE("[JNI] GetMethodID failed: %s%s (cls=0x%08X)", name, sig, jit.Regs()[1]);
                jit.Regs()[0] = 0;
            } else {
                jit.Regs()[0] = AddMethod(mid, false, jit.Regs()[1], name, sig);
            }
            break;
        }
        case 376: { // GetFieldID
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[1]));
            const char* name = ReadGuestStr(*m_host, jit.Regs()[2]);
            const char* sig = ReadGuestStr(*m_host, jit.Regs()[3]);
            jfieldID fid = e->GetFieldID(c, name, sig);
            if (!fid) {
                e->ExceptionClear();
                LOGE("[JNI] GetFieldID failed: %s%s", name, sig);
                jit.Regs()[0] = 0;
            } else {
                jit.Regs()[0] = AddField(fid);
            }
            break;
        }
        case 452: { // GetStaticMethodID
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[1]));
            const char* name = ReadGuestStr(*m_host, jit.Regs()[2]);
            const char* sig = ReadGuestStr(*m_host, jit.Regs()[3]);
            if (!c) { jit.Regs()[0] = 0; break; }
            jmethodID mid = e->GetStaticMethodID(c, name, sig);
            if (!mid) {
                e->ExceptionClear();
                LOGE("[JNI] GetStaticMethodID failed: %s%s (cls=0x%08X)", name, sig, jit.Regs()[1]);
                jit.Regs()[0] = 0;
            } else {
                jit.Regs()[0] = AddMethod(mid, true, jit.Regs()[1], name, sig);
            }
            break;
        }

        // ---- object creation ----
        case 112: // NewObject (varargs)
        case 116: // NewObjectV
        case 120: { // NewObjectA
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[1]));
            uint32_t mid_handle = jit.Regs()[2];
            auto mi_it = m_methods.find(mid_handle);
            if (mi_it == m_methods.end()) { jit.Regs()[0] = 0; break; }
            MethodInfo& mi = mi_it->second;
            std::vector<char> params = ParseParams(mi.sig);
            std::vector<GuestArg> args = offset == 112 ? ReadArgsVar(jit, params) :
                offset == 116 ? ReadArgsV(jit, params, jit.Regs()[3]) :
                                 ReadArgsA(jit, params, jit.Regs()[3]);
            // Build jvalues and call NewObjectA
            std::vector<jvalue> jvs(args.size());
            for (size_t i = 0; i < args.size(); ++i) {
                const GuestArg& a = args[i];
                switch (a.type) {
                    case 'Z': jvs[i].z = static_cast<jboolean>(a.v); break;
                    case 'I': jvs[i].i = static_cast<jint>(a.v); break;
                    case 'J': jvs[i].j = static_cast<jlong>(a.v); break;
                    case 'F': { float f; memcpy(&f, &a.v, 4); jvs[i].f = f; break; }
                    case 'L': jvs[i].l = a.obj; break;
                    default: jvs[i].i = static_cast<jint>(a.v); break;
                }
            }
            jobject o = e->NewObjectA(c, mi.mid, jvs.data());
            if (!o) { e->ExceptionClear(); LOGE("[JNI] NewObject failed"); jit.Regs()[0] = 0; }
            else jit.Regs()[0] = AddObjectRef(o);
            break;
        }

        // ---- field access ----
        case 400: { // GetIntField
            jobject o = ResolveObject(jit.Regs()[1]);
            auto it = m_fields.find(jit.Regs()[2]);
            if (!o || it == m_fields.end()) { jit.Regs()[0] = 0; break; }
            jit.Regs()[0] = static_cast<uint32_t>(e->GetIntField(o, it->second));
            break;
        }
        case 380: { // GetObjectField
            jobject o = ResolveObject(jit.Regs()[1]);
            auto it = m_fields.find(jit.Regs()[2]);
            if (!o || it == m_fields.end()) { jit.Regs()[0] = 0; break; }
            jobject r = e->GetObjectField(o, it->second);
            jit.Regs()[0] = r ? AddObjectRef(r) : 0;
            break;
        }

        // ---- static fields ----
        case 576: { // GetStaticFieldID
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[1]));
            const char* name = ReadGuestStr(*m_host, jit.Regs()[2]);
            const char* sig = ReadGuestStr(*m_host, jit.Regs()[3]);
            jfieldID fid = c ? e->GetStaticFieldID(c, name, sig) : nullptr;
            if (!fid) {
                e->ExceptionClear();
                jit.Regs()[0] = 0;
            } else {
                jit.Regs()[0] = AddField(fid);
            }
            break;
        }
        case 580: { // GetStaticObjectField
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[1]));
            auto it = m_fields.find(jit.Regs()[2]);
            jobject value = (c && it != m_fields.end()) ? e->GetStaticObjectField(c, it->second) : nullptr;
            jit.Regs()[0] = value ? AddObjectRef(value) : 0;
            break;
        }
        case 584: { // GetStaticBooleanField
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[1]));
            auto it = m_fields.find(jit.Regs()[2]);
            jit.Regs()[0] = (c && it != m_fields.end()) ? e->GetStaticBooleanField(c, it->second) : 0;
            break;
        }
        case 600: { // GetStaticIntField
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[1]));
            auto it = m_fields.find(jit.Regs()[2]);
            jit.Regs()[0] = (c && it != m_fields.end()) ? static_cast<uint32_t>(e->GetStaticIntField(c, it->second)) : 0;
            break;
        }
        case 604: { // GetStaticLongField
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[1]));
            auto it = m_fields.find(jit.Regs()[2]);
            const jlong value = (c && it != m_fields.end()) ? e->GetStaticLongField(c, it->second) : 0;
            jit.Regs()[0] = static_cast<uint32_t>(value);
            jit.Regs()[1] = static_cast<uint32_t>(static_cast<uint64_t>(value) >> 32);
            break;
        }

        // ---- arrays ----
        case 684: { // GetArrayLength
            jarray array = static_cast<jarray>(ResolveObject(jit.Regs()[1]));
            jit.Regs()[0] = array ? static_cast<uint32_t>(e->GetArrayLength(array)) : 0;
            break;
        }
        case 688: { // NewObjectArray
            const jsize length = static_cast<jsize>(jit.Regs()[1]);
            jclass element_class = static_cast<jclass>(ResolveObject(jit.Regs()[2]));
            jobject initial = ResolveObject(jit.Regs()[3]);
            jobjectArray array = (length >= 0 && element_class) ?
                e->NewObjectArray(length, element_class, initial) : nullptr;
            if (!array) e->ExceptionClear();
            jit.Regs()[0] = array ? AddObjectRef(array) : 0;
            break;
        }
        case 692: { // GetObjectArrayElement
            jobjectArray array = static_cast<jobjectArray>(ResolveObject(jit.Regs()[1]));
            jobject value = array ? e->GetObjectArrayElement(array, static_cast<jsize>(jit.Regs()[2])) : nullptr;
            jit.Regs()[0] = value ? AddObjectRef(value) : 0;
            break;
        }
        case 696: { // SetObjectArrayElement
            jobjectArray array = static_cast<jobjectArray>(ResolveObject(jit.Regs()[1]));
            if (array) e->SetObjectArrayElement(array, static_cast<jsize>(jit.Regs()[2]), ResolveObject(jit.Regs()[3]));
            jit.Regs()[0] = 0;
            break;
        }
        case 700: { // NewBooleanArray
            jbooleanArray array = e->NewBooleanArray(static_cast<jsize>(jit.Regs()[1]));
            if (!array) e->ExceptionClear();
            jit.Regs()[0] = array ? AddObjectRef(array) : 0;
            break;
        }
        case 704: { // NewByteArray
            jbyteArray array = e->NewByteArray(static_cast<jsize>(jit.Regs()[1]));
            if (!array) e->ExceptionClear();
            jit.Regs()[0] = array ? AddObjectRef(array) : 0;
            break;
        }
        case 724: { // NewFloatArray
            const jsize length = static_cast<jsize>(jit.Regs()[1]);
            jfloatArray array = length >= 0 ? e->NewFloatArray(length) : nullptr;
            if (!array) e->ExceptionClear();
            jit.Regs()[0] = array ? AddObjectRef(array) : 0;
            if (array) e->DeleteLocalRef(array);
            break;
        }
        case 748: { // GetIntArrayElements
            const uint32_t handle = jit.Regs()[1];
            jintArray array = static_cast<jintArray>(ResolveObject(handle));
            if (!array) { jit.Regs()[0] = 0; break; }
            auto known = m_array_elements.find(handle);
            if (known != m_array_elements.end()) {
                if (jit.Regs()[2]) m_host->MemoryWrite8(jit.Regs()[2], JNI_TRUE);
                jit.Regs()[0] = known->second.guest;
                break;
            }
            jint* values = e->GetIntArrayElements(array, nullptr);
            const jsize length = e->GetArrayLength(array);
            const size_t bytes = length >= 0 ? static_cast<size_t>(length) * sizeof(jint) : 0;
            const uint32_t guest = (values && length >= 0) ? m_host->GuestAlloc(bytes, alignof(jint)) : 0;
            if (guest) memcpy(m_host->GetGuestMemoryBase() + guest, values, bytes);
            if (!guest) {
                if (values) e->ReleaseIntArrayElements(array, values, JNI_ABORT);
                jit.Regs()[0] = 0;
                break;
            }
            PinObjectRef(handle, +1);
            m_array_elements[handle] = {array, handle, values, guest, length, sizeof(jint), 'I'};
            if (jit.Regs()[2]) m_host->MemoryWrite8(jit.Regs()[2], JNI_TRUE);
            jit.Regs()[0] = guest;
            break;
        }

        // ---- strings ----
        case 668: { // NewStringUTF
            const char* s = ReadGuestStr(*m_host, jit.Regs()[1]);
            jstring js = e->NewStringUTF(s ? s : "");
            jit.Regs()[0] = AddObjectRef(js);
            break;
        }
        case 656: { // GetStringLength
            jstring s = static_cast<jstring>(ResolveObject(jit.Regs()[1]));
            jit.Regs()[0] = s ? static_cast<uint32_t>(e->GetStringLength(s)) : 0;
            break;
        }
        case 660: { // GetStringChars (UTF-16)
            jstring s = static_cast<jstring>(ResolveObject(jit.Regs()[1]));
            if (!s) { jit.Regs()[0] = 0; break; }
            const jchar* chars = e->GetStringChars(s, nullptr);
            jsize len = e->GetStringLength(s);
            uint32_t guest = m_host->GuestAlloc((len + 1) * 2, 4);
            memcpy(mem + guest, chars, len * 2);
            memset(mem + guest + len * 2, 0, 2);
            e->ReleaseStringChars(s, chars);
            jit.Regs()[0] = guest;
            break;
        }
        case 664: // ReleaseStringChars -> the copy is ours, and this is the only
                  // call that says when it stops being needed
            m_host->GuestFree(jit.Regs()[2]);
            jit.Regs()[0] = 0;
            break;
        case 676: { // GetStringUTFChars
            jstring s = static_cast<jstring>(ResolveObject(jit.Regs()[1]));
            uint32_t isCopyOut = jit.Regs()[2];
            if (!s) { jit.Regs()[0] = 0; break; }
            auto cached = m_jstr_utf.find(jit.Regs()[1]);
            if (cached != m_jstr_utf.end()) {
                if (isCopyOut) m_host->MemoryWrite8(isCopyOut, 0);
                jit.Regs()[0] = cached->second;
                break;
            }
            const char* utf = e->GetStringUTFChars(s, nullptr);
            size_t len = utf ? strlen(utf) + 1 : 1;
            uint32_t guest = m_host->GuestAlloc(len, 4);
            memcpy(mem + guest, utf ? utf : "", len);
            e->ReleaseStringUTFChars(s, utf);
            m_jstr_utf[jit.Regs()[1]] = guest;
            if (isCopyOut) m_host->MemoryWrite8(isCopyOut, 0); // JNI_FALSE
            jit.Regs()[0] = guest;
            break;
        }
        case 680: // ReleaseStringUTFChars -> no-op on purpose: unlike
                  // GetStringChars this copy is cached in m_jstr_utf and handed
                  // back on the next call, so it lives exactly as long as the
                  // jstring's handle and is freed when that handle is evicted
            jit.Regs()[0] = 0;
            break;

        // ---- exceptions ----
        case 60: { // ExceptionOccurred
            jthrowable t = e->ExceptionOccurred();
            jit.Regs()[0] = t ? AddObjectRef(t) : 0;
            static std::atomic<unsigned> logged{0};
            if (logged.fetch_add(1) < 16) LOGI("[JNI] ExceptionOccurred -> 0x%08X", jit.Regs()[0]);
            break;
        }
        case 64: // ExceptionDescribe
            e->ExceptionDescribe();
            jit.Regs()[0] = 0;
            break;
        case 68: // ExceptionClear
            e->ExceptionClear();
            static std::atomic<unsigned> logged{0};
            if (logged.fetch_add(1) < 16) LOGI("[JNI] ExceptionClear");
            jit.Regs()[0] = 0;
            break;
        case 912: { // ExceptionCheck
            jit.Regs()[0] = e->ExceptionCheck() ? 1 : 0;
            static std::atomic<unsigned> logged{0};
            if (logged.fetch_add(1) < 16) LOGI("[JNI] ExceptionCheck -> %u", jit.Regs()[0]);
            break;
        }
        case 56: { // ThrowNew
            jclass c = static_cast<jclass>(ResolveObject(jit.Regs()[1]));
            const char* msg = ReadGuestStr(*m_host, jit.Regs()[2]);
            if (c) e->ThrowNew(c, msg);
            jit.Regs()[0] = 0;
            break;
        }
        case 72: { // FatalError
            const char* msg = ReadGuestStr(*m_host, jit.Regs()[1]);
            LOGE("[JNI] FatalError: %s", msg ? msg : "");
            jit.Regs()[0] = 0;
            break;
        }

        // ---- primitive array mirrors ----
        case 736: { // GetByteArrayElements
            const uint32_t handle = jit.Regs()[1];
            jbyteArray array = static_cast<jbyteArray>(ResolveObject(handle));
            if (!array) { jit.Regs()[0] = 0; break; }
            auto known = m_array_elements.find(handle);
            if (known != m_array_elements.end()) {
                if (jit.Regs()[2]) m_host->MemoryWrite8(jit.Regs()[2], JNI_TRUE);
                jit.Regs()[0] = known->second.guest;
                break;
            }
            jbyte* values = e->GetByteArrayElements(array, nullptr);
            const jsize length = e->GetArrayLength(array);
            const uint32_t guest = (values && length >= 0) ? m_host->GuestAlloc(static_cast<size_t>(length), 1) : 0;
            if (guest) memcpy(m_host->GetGuestMemoryBase() + guest, values, static_cast<size_t>(length));
            if (!guest) {
                if (values) e->ReleaseByteArrayElements(array, values, JNI_ABORT);
                jit.Regs()[0] = 0;
                break;
            }
            PinObjectRef(handle, +1);
            m_array_elements[handle] = {array, handle, values, guest, length, sizeof(jbyte), 'B'};
            if (jit.Regs()[2]) m_host->MemoryWrite8(jit.Regs()[2], JNI_TRUE);
            jit.Regs()[0] = guest;
            break;
        }
        case 768: { // ReleaseByteArrayElements
            const uint32_t handle = jit.Regs()[1];
            auto it = m_array_elements.find(handle);
            if (it != m_array_elements.end()) {
                jbyteArray array = static_cast<jbyteArray>(it->second.array);
                jbyte* values = static_cast<jbyte*>(it->second.hostPtr);
                if (array && values && jit.Regs()[3] != JNI_ABORT) {
                    memcpy(values, m_host->GetGuestMemoryBase() + it->second.guest,
                           static_cast<size_t>(it->second.length));
                }
                if (array && values) e->ReleaseByteArrayElements(array, values, static_cast<jint>(jit.Regs()[3]));
                PinObjectRef(it->second.owner_handle, -1);
                m_host->GuestFree(it->second.guest);
                m_array_elements.erase(it);
            }
            jit.Regs()[0] = 0;
            break;
        }
        case 780: { // ReleaseIntArrayElements
            const uint32_t handle = jit.Regs()[1];
            auto it = m_array_elements.find(handle);
            if (it != m_array_elements.end() && it->second.kind == 'I') {
                jintArray array = static_cast<jintArray>(it->second.array);
                jint* values = static_cast<jint*>(it->second.hostPtr);
                const size_t bytes = static_cast<size_t>(it->second.length) * sizeof(jint);
                if (array && values && jit.Regs()[3] != JNI_ABORT)
                    memcpy(values, m_host->GetGuestMemoryBase() + it->second.guest, bytes);
                if (array && values) e->ReleaseIntArrayElements(array, values, static_cast<jint>(jit.Regs()[3]));
                PinObjectRef(it->second.owner_handle, -1);
                m_host->GuestFree(it->second.guest);
                m_array_elements.erase(it);
            }
            jit.Regs()[0] = 0;
            break;
        }

        // ---- direct buffers ----
        case 920: { // GetDirectBufferAddress
            uint32_t handle = jit.Regs()[1];
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = m_direct_bufs.find(handle);
                static std::atomic<unsigned> logged{0};
                if (logged.fetch_add(1) < 12)
                    LOGI("[JNI] GetDirectBufferAddress handle=0x%08X known=%d", handle,
                         it != m_direct_bufs.end() ? 1 : 0);
                if (it != m_direct_bufs.end()) {
                    jit.Regs()[0] = it->second.scratch;
                    break;
                }
            }
            jobject buf = ResolveObject(handle);
            if (!buf) { jit.Regs()[0] = 0; break; }
            void* p = e->GetDirectBufferAddress(buf);
            if (!p) { jit.Regs()[0] = 0; break; }
            // Unknown buffer: mirror host content into guest scratch (read-only view)
            uint32_t newHandle = RegisterDirectBuffer(buf);
            std::lock_guard<std::mutex> lock(m_mutex);
            jit.Regs()[0] = newHandle ? m_direct_bufs.at(newHandle).scratch : 0;
            break;
        }
        case 924: { // GetDirectBufferCapacity returns jlong in r0:r1.
            uint32_t handle = jit.Regs()[1];
            jlong capacity = DirectBufferCapacity(handle);
            if (!capacity) {
                jobject buf = ResolveObject(handle);
                capacity = buf ? e->GetDirectBufferCapacity(buf) : -1;
            }
            jit.Regs()[0] = static_cast<uint32_t>(capacity);
            jit.Regs()[1] = static_cast<uint32_t>(static_cast<uint64_t>(capacity) >> 32);
            break;
        }

        // ---- misc ----
        case 876: { // GetJavaVM
            uint32_t out = jit.Regs()[1];
            if (out) m_host->MemoryWrite32(out, m_vm_table);
            jit.Regs()[0] = 0;
            break;
        }
        case 96: { // IsSameObject
            jobject a = ResolveObject(jit.Regs()[1]);
            jobject b = ResolveObject(jit.Regs()[2]);
            jit.Regs()[0] = e->IsSameObject(a, b) ? 1 : 0;
            break;
        }
        case 52: { // Throw
            jobject t = ResolveObject(jit.Regs()[1]);
            if (t) e->Throw(static_cast<jthrowable>(t));
            jit.Regs()[0] = 0;
            break;
        }
        case 416: case 436: { // SetObjectField / SetIntField (unlikely)
            jobject o = ResolveObject(jit.Regs()[1]);
            auto it = m_fields.find(jit.Regs()[2]);
            if (o && it != m_fields.end()) {
                if (offset == 416) e->SetObjectField(o, it->second, ResolveObject(jit.Regs()[3]));
                else e->SetIntField(o, it->second, static_cast<jint>(jit.Regs()[3]));
            }
            jit.Regs()[0] = 0;
            break;
        }
        case 404: { // GetLongField
            jobject o = ResolveObject(jit.Regs()[1]);
            auto it = m_fields.find(jit.Regs()[2]);
            jlong v = (o && it != m_fields.end()) ? e->GetLongField(o, it->second) : 0;
            jit.Regs()[0] = static_cast<uint32_t>(v & 0xFFFFFFFFu);
            jit.Regs()[1] = static_cast<uint32_t>(static_cast<uint64_t>(v) >> 32);
            break;
        }

        default: {
            static bool warned[240];
            uint32_t slot = offset / 4;
            if (slot < 240 && !warned[slot]) {
                warned[slot] = true;
                LOGE("[JNI] unhandled env slot %u (off 0x%X) at PC 0x%08X, r1=0x%08X r2=0x%08X r3=0x%08X",
                     slot, offset, jit.Regs()[15], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3]);
            }
            jit.Regs()[0] = 0;
            jit.Regs()[1] = 0;
            break;
        }
    }
}
