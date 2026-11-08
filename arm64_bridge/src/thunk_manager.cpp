#include "thunk_manager.h"
#include "dynarmic_host.h"
#include "elf_loader.h"
#include "thunk_types.h"
#include <android/log.h>
#include <cstdlib>
#include <stdexcept>

#define LOG_TAG "Arm32Bridge_Thunks"
#include "bridge_log.h"

uint32_t ThunkManager::s_screenWidth = 1213;
uint32_t ThunkManager::s_screenHeight = 540;
uint32_t ThunkManager::s_virtualWidth = 0;
uint32_t ThunkManager::s_virtualHeight = 0;
uint32_t ThunkManager::s_forced_dpi = 0;
uint32_t ThunkManager::s_ui_inset = 0;

ThunkManager::ThunkManager(DynarmicHost& host)
        : m_host(host) {}

ThunkManager::~ThunkManager() = default;

#include "thunks_opensles.h"

void ThunkManager::RegisterAllThunks() {
    RegisterGLThunks();
    RegisterOpenSLESThunks(*this, m_host);
    RegisterLibcThunks();
    // The guest-visible JNIEnv/JavaVM tables live in FakeJniEnv; jni_bridge.cpp
    // calls FakeJniEnv::Instance().Init(host) once the host JavaVM is known.
}

#include <jni.h>
#include <sys/system_properties.h>
#include <cmath>
#include <cctype>
#include <cstring>

JavaVM* GetHostJavaVM();

std::string QueryDeviceLanguage() {
    std::string lang = "en";
    JavaVM* jvm = GetHostJavaVM();
    if (jvm) {
        JNIEnv* env = nullptr;
        bool attached = false;
        if (jvm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
            if (jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            }
        }
        if (env) {
            jclass locCls = env->FindClass("java/util/Locale");
            if (locCls) {
                jmethodID getDefault = env->GetStaticMethodID(locCls, "getDefault", "()Ljava/util/Locale;");
                if (getDefault) {
                    jobject locObj = env->CallStaticObjectMethod(locCls, getDefault);
                    if (locObj) {
                        jmethodID getLang = env->GetMethodID(locCls, "getLanguage", "()Ljava/lang/String;");
                        if (getLang) {
                            jstring jStr = (jstring)env->CallObjectMethod(locObj, getLang);
                            if (jStr) {
                                const char* utf = env->GetStringUTFChars(jStr, nullptr);
                                if (utf && strlen(utf) > 0) {
                                    lang = utf;
                                }
                                if (utf) env->ReleaseStringUTFChars(jStr, utf);
                            }
                        }
                    }
                }
            }
            if (env->ExceptionCheck()) env->ExceptionClear();
            if (attached) jvm->DetachCurrentThread();
        }
    }
    if (lang.empty() || lang == "en") {
        char prop[PROP_VALUE_MAX] = {0};
        if (__system_property_get("persist.sys.locale", prop) > 0 ||
            __system_property_get("ro.product.locale", prop) > 0 ||
            __system_property_get("persist.sys.language", prop) > 0) {
            char* dash = strchr(prop, '-');
            if (dash) *dash = '\0';
            char* under = strchr(prop, '_');
            if (under) *under = '\0';
            if (strlen(prop) > 0) lang = prop;
        }
    }
    for (char& c : lang) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    return lang;
}

std::string QueryDeviceCountryCode() {
    std::string country = "US";
    JavaVM* jvm = GetHostJavaVM();
    if (jvm) {
        JNIEnv* env = nullptr;
        bool attached = false;
        if (jvm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
            if (jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            }
        }
        if (env) {
            jclass locCls = env->FindClass("java/util/Locale");
            if (locCls) {
                jmethodID getDefault = env->GetStaticMethodID(locCls, "getDefault", "()Ljava/util/Locale;");
                if (getDefault) {
                    jobject locObj = env->CallStaticObjectMethod(locCls, getDefault);
                    if (locObj) {
                        jmethodID getCountry = env->GetMethodID(locCls, "getCountry", "()Ljava/lang/String;");
                        if (getCountry) {
                            jstring jStr = (jstring)env->CallObjectMethod(locObj, getCountry);
                            if (jStr) {
                                const char* utf = env->GetStringUTFChars(jStr, nullptr);
                                if (utf && strlen(utf) > 0) {
                                    country = utf;
                                }
                                if (utf) env->ReleaseStringUTFChars(jStr, utf);
                            }
                        }
                    }
                }
            }
            if (env->ExceptionCheck()) env->ExceptionClear();
            if (attached) jvm->DetachCurrentThread();
        }
    }
    if (country.empty() || country == "US") {
        char prop[PROP_VALUE_MAX] = {0};
        if (__system_property_get("persist.sys.country", prop) > 0 ||
            __system_property_get("ro.product.locale.region", prop) > 0) {
            if (strlen(prop) > 0) country = prop;
        } else if (__system_property_get("persist.sys.locale", prop) > 0 ||
                   __system_property_get("ro.product.locale", prop) > 0) {
            char* dash = strchr(prop, '-');
            if (dash && strlen(dash + 1) >= 2) {
                country = std::string(dash + 1, 2);
            } else {
                char* under = strchr(prop, '_');
                if (under && strlen(under + 1) >= 2) {
                    country = std::string(under + 1, 2);
                }
            }
        }
    }
    for (char& c : country) c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
    return country;
}

uint32_t QueryDeviceDensity() {
    if (ThunkManager::s_forced_dpi > 0) {
        return ThunkManager::s_forced_dpi;
    }
    uint32_t density = 0;
    JavaVM* jvm = GetHostJavaVM();
    if (jvm) {
        JNIEnv* env = nullptr;
        bool attached = false;
        if (jvm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
            if (jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            }
        }
        if (env) {
            jclass resCls = env->FindClass("android/content/res/Resources");
            if (resCls) {
                jmethodID getSys = env->GetStaticMethodID(resCls, "getSystem", "()Landroid/content/res/Resources;");
                if (getSys) {
                    jobject resObj = env->CallStaticObjectMethod(resCls, getSys);
                    if (resObj) {
                        jmethodID getDM = env->GetMethodID(resCls, "getDisplayMetrics", "()Landroid/util/DisplayMetrics;");
                        if (getDM) {
                            jobject dmObj = env->CallObjectMethod(resObj, getDM);
                            if (dmObj) {
                                jclass dmCls = env->FindClass("android/util/DisplayMetrics");
                                jfieldID fid = env->GetFieldID(dmCls, "densityDpi", "I");
                                if (fid) {
                                    density = static_cast<uint32_t>(env->GetIntField(dmObj, fid));
                                }
                            }
                        }
                    }
                }
            }
            if (env->ExceptionCheck()) env->ExceptionClear();
            if (attached) jvm->DetachCurrentThread();
        }
    }
    if (density == 0) {
        char prop[PROP_VALUE_MAX] = {0};
        if (__system_property_get("ro.sf.lcd_density", prop) > 0 ||
            __system_property_get("qemu.sf.lcd_density", prop) > 0) {
            density = static_cast<uint32_t>(atoi(prop));
        }
    }
    if (density == 0) {
        density = 320;
    }
    return density;
}

uint32_t QueryDeviceIsPad() {
    JavaVM* jvm = GetHostJavaVM();
    if (jvm) {
        JNIEnv* env = nullptr;
        bool attached = false;
        if (jvm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
            if (jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            }
        }
        if (env) {
            // Configuration.screenLayout is the platform answer; the diagonal
            // fallback below covers devices that report it wrong.
            jclass resCls = env->FindClass("android/content/res/Resources");
            if (resCls) {
                jmethodID getSys = env->GetStaticMethodID(resCls, "getSystem", "()Landroid/content/res/Resources;");
                if (getSys) {
                    jobject resObj = env->CallStaticObjectMethod(resCls, getSys);
                    if (resObj) {
                        jmethodID getCfg = env->GetMethodID(resCls, "getConfiguration", "()Landroid/content/res/Configuration;");
                        if (getCfg) {
                            jobject cfgObj = env->CallObjectMethod(resObj, getCfg);
                            if (cfgObj) {
                                jclass cfgCls = env->FindClass("android/content/res/Configuration");
                                jfieldID slFid = env->GetFieldID(cfgCls, "screenLayout", "I");
                                if (slFid) {
                                    jint screenLayout = env->GetIntField(cfgObj, slFid);
                                    if ((screenLayout & 0x0F) >= 3) { // SCREENLAYOUT_SIZE_LARGE or XLARGE
                                        if (env->ExceptionCheck()) env->ExceptionClear();
                                        if (attached) jvm->DetachCurrentThread();
                                        return 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (env->ExceptionCheck()) env->ExceptionClear();
            if (attached) jvm->DetachCurrentThread();
        }
    }

    uint32_t w = ThunkManager::s_screenWidth;
    uint32_t h = ThunkManager::s_screenHeight;
    uint32_t dpi = QueryDeviceDensity();
    if (w > 0 && h > 0 && dpi > 0) {
        double w_in = static_cast<double>(w) / static_cast<double>(dpi);
        double h_in = static_cast<double>(h) / static_cast<double>(dpi);
        double diag = sqrt(w_in * w_in + h_in * h_in);
        if (diag >= 6.5) return 1;
    }
    return 0;
}


uint32_t ThunkManager::RegisterThunk(const std::string& name, ThunkHandler handler) {
    return RegisterThunkImpl(name, std::move(handler), true);
}

uint32_t ThunkManager::RegisterThunkImpl(const std::string& name, ThunkHandler handler, bool replace) {
    uint32_t address;
    bool created = false;
    {
        std::lock_guard lock(m_registry_mutex);
        if (name.empty() || !handler) throw std::invalid_argument("Invalid thunk registration");
        if (!replace) {
            const auto found = m_symbol_to_address.find(name);
            if (found != m_symbol_to_address.end()) return found->second;
        }
        const auto existing = m_function_ids.find(name);
        uint32_t id;
        if (existing != m_function_ids.end()) {
            id = existing->second;
            address = m_symbol_to_address.at(name);
        } else {
            if (m_symbol_to_address.contains(name))
                throw std::logic_error("Cannot replace data with a function: " + name);
            if (m_function_ids.size() >= kMaxThunks)
                throw std::overflow_error("Guest trampoline region exhausted");
            id = static_cast<uint32_t>(m_function_ids.size() + 1);
            address = m_next_thunk_addr;
            m_host.MemoryWrite32(address, 0xEF000000 | id);
            m_host.MemoryWrite32(address + 4, 0xE12FFF1E);
            m_function_ids.emplace(name, id);
            m_symbol_to_address.emplace(name, address);
            m_next_thunk_addr += 8;
            created = true;
        }
        m_dispatch.Publish(id, ThunkEntry{name, std::move(handler)});
    }
    // Never hold the registry lock while acquiring JIT/cache locks.
    if (created) m_host.InvalidateGuestCode(address, 8);
    return address;
}

void ThunkManager::RegisterSymbol(const std::string& name, uint32_t address) {
    std::lock_guard lock(m_registry_mutex);
    if (name.empty() || !address || address >= m_host.GetGuestMemorySize() || m_function_ids.contains(name))
        throw std::invalid_argument("Invalid guest data symbol: " + name);
    m_symbol_to_address[name] = address;
}

uint32_t ThunkManager::FindRegisteredSymbol(const std::string& name) const {
    std::lock_guard lock(m_registry_mutex);
    const auto it = m_symbol_to_address.find(name);
    return it == m_symbol_to_address.end() ? 0 : it->second;
}

uint32_t ThunkManager::GetOrCreateThunkAddress(const std::string& name) {
    if (const uint32_t address = FindRegisteredSymbol(name)) return address;
    // Preserve this game's existing fallback policy. Audit missing imports
    // separately before changing its loader's unresolved-symbol contract.
    auto logged = std::make_shared<std::atomic_flag>();
    return RegisterThunkImpl(name, [name, logged](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        if (!logged->test_and_set(std::memory_order_relaxed)) {
            LOGI("[THUNK] Default stub executed for: %s", name.c_str());
        }
        jit.Regs()[0] = 0;
    }, false);
}

void ThunkManager::DispatchThunk(uint32_t thunk_id, Dynarmic::A32::Jit& jit) {
    const auto* entry = m_dispatch.Read(thunk_id);
    if (!entry) {
        LOGE("[THUNK] Unknown thunk ID: %u at PC: 0x%08X", thunk_id, jit.Regs()[15]);
        jit.Regs()[0] = 0;
        return;
    }
    try {
        entry->handler(jit, m_host);
    } catch (const std::exception& error) {
        LOGE("[THUNK] Host exception in %s: %s", entry->name.c_str(), error.what());
        std::abort(); // C++ exceptions must never unwind through generated JIT frames.
    } catch (...) {
        LOGE("[THUNK] Unknown host exception in %s", entry->name.c_str());
        std::abort();
    }
}

std::string ThunkManager::GetThunkName(uint32_t thunk_id) const {
    const auto* entry = m_dispatch.Read(thunk_id);
    return entry ? entry->name : std::string();
}
