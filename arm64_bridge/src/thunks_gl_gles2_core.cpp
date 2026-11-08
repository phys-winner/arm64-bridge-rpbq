#include "thunk_manager.h"
#include "dynarmic_host.h"
#include "thunk_types.h"
#include "mapped_buffer_copy.h"

#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <EGL/egl.h>
#include <android/log.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

#define LOG_TAG "Arm32Bridge_GLES2Core"
#include "bridge_log.h"

namespace {

bool CheckedArrayBytes(size_t count, size_t components, size_t element_size,
                       size_t* bytes) {
    const uint64_t total = static_cast<uint64_t>(count) *
                           static_cast<uint64_t>(components) *
                           static_cast<uint64_t>(element_size);
    if (total > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *bytes = static_cast<size_t>(total);
    return true;
}

template <typename T>
const T* GuestInputArray(DynarmicHost& host, uint32_t address, size_t count,
                         size_t components, const char* function_name) {
    size_t bytes = 0;
    if (!CheckedArrayBytes(count, components, sizeof(T), &bytes) ||
        bytes == 0 ||
        !host.IsGuestMemoryRangeAccessible(address, bytes)) {
        LOGE("%s rejected guest input: address=0x%08X count=%zu components=%zu",
             function_name, address, count, components);
        return nullptr;
    }
    return reinterpret_cast<const T*>(
        GuestSpan(host.GetGuestMemoryBase(), host.GetGuestMemorySize(), address, bytes));
}

template <typename T>
T* GuestOutputArray(DynarmicHost& host, uint32_t address, size_t count,
                    size_t components, const char* function_name) {
    size_t bytes = 0;
    if (!CheckedArrayBytes(count, components, sizeof(T), &bytes) ||
        bytes == 0 ||
        !host.IsGuestMemoryRangeAccessible(address, bytes)) {
        LOGE("%s rejected guest output: address=0x%08X count=%zu components=%zu",
             function_name, address, count, components);
        return nullptr;
    }
    return reinterpret_cast<T*>(
        GuestSpan(host.GetGuestMemoryBase(), host.GetGuestMemorySize(), address, bytes));
}

template <typename T>
T* OptionalGuestOutput(DynarmicHost& host, uint32_t address, size_t count,
                       const char* function_name, bool* valid) {
    if (address == 0) {
        *valid = true;
        return nullptr;
    }
    T* output = GuestOutputArray<T>(host, address, count, 1, function_name);
    *valid = output != nullptr;
    return output;
}

const void* CountedGuestBytes(DynarmicHost& host, uint32_t address,
                              GLsizei byte_count, const char* function_name,
                              bool* valid) {
    if (byte_count <= 0) {
        *valid = true;
        return nullptr;
    }
    const void* input = GuestInputArray<uint8_t>(
        host, address, static_cast<size_t>(byte_count), 1, function_name);
    *valid = input != nullptr;
    return input;
}

template <typename T>
const T* CountedGuestValues(DynarmicHost& host, uint32_t address,
                            GLsizei count, size_t components,
                            const char* function_name, bool* valid) {
    if (count <= 0) {
        *valid = true;
        return nullptr;
    }
    const T* input = GuestInputArray<T>(
        host, address, static_cast<size_t>(count), components, function_name);
    *valid = input != nullptr;
    return input;
}

size_t GetFloatStateValueCount(GLenum pname) {
    switch (pname) {
        case GL_ALIASED_LINE_WIDTH_RANGE:
        case GL_ALIASED_POINT_SIZE_RANGE:
        case GL_DEPTH_RANGE:
        case GL_MAX_VIEWPORT_DIMS:
            return 2;
        case GL_BLEND_COLOR:
        case GL_COLOR_CLEAR_VALUE:
        case GL_COLOR_WRITEMASK:
        case GL_SCISSOR_BOX:
        case GL_VIEWPORT:
            return 4;
        case GL_COMPRESSED_TEXTURE_FORMATS: {
            GLint count = 0;
            glGetIntegerv(GL_NUM_COMPRESSED_TEXTURE_FORMATS, &count);
            return count > 0 ? static_cast<size_t>(count) : 1;
        }
        case GL_SHADER_BINARY_FORMATS: {
            GLint count = 0;
            glGetIntegerv(GL_NUM_SHADER_BINARY_FORMATS, &count);
            return count > 0 ? static_cast<size_t>(count) : 1;
        }
        default:
            return 1;
    }
}

size_t GetVertexAttribValueCount(GLenum pname) {
    return pname == GL_CURRENT_VERTEX_ATTRIB ? 4 : 1;
}

size_t GetTextureParameterValueCount(GLenum pname) {
    // Core GLES texture parameter names are scalar. The border-color extension
    // reuses glTexParameter*v and is the one vector-valued exception exposed by
    // the Android GLES headers.
    return pname == GL_TEXTURE_BORDER_COLOR_EXT ? 4 : 1;
}

uint32_t HostVertexPointerToGuest(DynarmicHost& host, const void* pointer,
                                  GLint array_buffer_binding) {
    if (!pointer) return 0;

    const uintptr_t value = reinterpret_cast<uintptr_t>(pointer);
    if (array_buffer_binding != 0) {
        return value <= std::numeric_limits<uint32_t>::max()
            ? static_cast<uint32_t>(value)
            : 0;
    }

    const uintptr_t base = reinterpret_cast<uintptr_t>(host.GetGuestMemoryBase());
    const uintptr_t end = base + host.GetGuestMemorySize();
    if (value < base || value >= end) return 0;
    const uintptr_t offset = value - base;
    return offset <= std::numeric_limits<uint32_t>::max()
        ? static_cast<uint32_t>(offset)
        : 0;
}

const char* GuestGlName(DynarmicHost& host, uint32_t address) {
    if (!GuestRangeValid(host.GetGuestMemorySize(), address, 1)) return nullptr;
    const auto* text = reinterpret_cast<const char*>(host.GetGuestMemoryBase() + address);
    const size_t limit = host.GetGuestMemorySize() - address;
    return strnlen(text, limit) < limit ? text : nullptr;
}

bool GuestGlNameArray(DynarmicHost& host, uint32_t address, GLsizei count,
                      const char* function_name,
                      std::vector<const GLchar*>* names) {
    names->clear();
    if (count <= 0) return true;
    const auto* addresses = GuestInputArray<uint32_t>(
        host, address, static_cast<size_t>(count), 1, function_name);
    if (!addresses) return false;
    names->reserve(static_cast<size_t>(count));
    for (GLsizei i = 0; i < count; ++i) {
        const char* name = GuestGlName(host, addresses[i]);
        if (!name) {
            LOGE("%s rejected guest string[%d]: address=0x%08X",
                 function_name, static_cast<int>(i), addresses[i]);
            names->clear();
            return false;
        }
        names->push_back(name);
    }
    return true;
}

struct GuestMappedBuffer {
    void* host_pointer = nullptr;
    uint32_t guest_address = 0;
    size_t length = 0;
    GLbitfield access = 0;
    uint64_t group = 0;
    GLuint buffer = 0;
};

// Native mapping identity survives context migration and shared-context use.
// Names alone collide between unshared contexts; TLS does not follow a context.
std::mutex s_guest_mapping_mutex;
std::unordered_map<void*, GuestMappedBuffer> s_guest_mapped_buffers;
std::unordered_map<EGLContext, uint64_t> s_gl_context_groups;
uint64_t s_next_gl_group = 1;

uint64_t ContextGroupLocked(EGLContext context) {
    if (context == EGL_NO_CONTEXT) return 0;
    const auto it = s_gl_context_groups.find(context);
    if (it != s_gl_context_groups.end()) return it->second;
    const uint64_t group = s_next_gl_group++;
    s_gl_context_groups.emplace(context, group);
    return group;
}

void* NativeMappedPointer(GLenum target) {
    void* pointer = nullptr;
    glGetBufferPointerv(target, GL_BUFFER_MAP_POINTER, &pointer);
    return pointer;
}

GLuint NativeBoundBuffer(GLenum target) {
    GLenum binding = 0;
    switch (target) {
        case GL_ARRAY_BUFFER: binding = GL_ARRAY_BUFFER_BINDING; break;
        case GL_ELEMENT_ARRAY_BUFFER: binding = GL_ELEMENT_ARRAY_BUFFER_BINDING; break;
        case GL_COPY_READ_BUFFER: binding = GL_COPY_READ_BUFFER_BINDING; break;
        case GL_COPY_WRITE_BUFFER: binding = GL_COPY_WRITE_BUFFER_BINDING; break;
        case GL_PIXEL_PACK_BUFFER: binding = GL_PIXEL_PACK_BUFFER_BINDING; break;
        case GL_PIXEL_UNPACK_BUFFER: binding = GL_PIXEL_UNPACK_BUFFER_BINDING; break;
        case GL_UNIFORM_BUFFER: binding = GL_UNIFORM_BUFFER_BINDING; break;
        case GL_TRANSFORM_FEEDBACK_BUFFER: binding = GL_TRANSFORM_FEEDBACK_BUFFER_BINDING; break;
        case GL_ATOMIC_COUNTER_BUFFER: binding = GL_ATOMIC_COUNTER_BUFFER_BINDING; break;
        case GL_SHADER_STORAGE_BUFFER: binding = GL_SHADER_STORAGE_BUFFER_BINDING; break;
        case GL_DRAW_INDIRECT_BUFFER: binding = GL_DRAW_INDIRECT_BUFFER_BINDING; break;
        case GL_DISPATCH_INDIRECT_BUFFER: binding = GL_DISPATCH_INDIRECT_BUFFER_BINDING; break;
        default: return 0;
    }
    GLint buffer = 0;
    glGetIntegerv(binding, &buffer);
    return static_cast<GLuint>(buffer);
}

static_assert(bridge_gl::kMapWrite == GL_MAP_WRITE_BIT);
static_assert(bridge_gl::kMapFlushExplicit == GL_MAP_FLUSH_EXPLICIT_BIT);

constexpr std::optional<uint64_t> PixelUnpackBytes(uint32_t w, uint32_t h, uint32_t d,
                                                  uint32_t bpp, std::array<int, 6> u,
                                                  uint64_t limit) {
    if (!w || !h || !d) return 0;
    if (!bpp || u[0] < 1 || u[0] > 8 || (u[0] & (u[0] - 1))) return {};
    for (int value : u) if (value < 0) return {};
    const uint64_t row = ((uint64_t(u[1] ? u[1] : w) * bpp + u[0] - 1) / u[0]) * u[0];
    if (row > limit) return {};
    const uint64_t image = row * uint64_t(u[2] ? u[2] : h);
    if (image > limit) return {};
    uint64_t bytes = 0;
    for (uint64_t term : {uint64_t(u[5]) * image, uint64_t(u[4]) * row,
                          uint64_t(u[3]) * bpp, uint64_t(d - 1) * image,
                          uint64_t(h - 1) * row, uint64_t(w) * bpp}) {
        if (term > limit - bytes) return {};
        bytes += term;
    }
    return bytes;
}
static_assert(PixelUnpackBytes(1, 1, 1, 4, {4, 0, 0, 0, 0, 0}, UINT32_MAX) == 4);
static_assert(PixelUnpackBytes(3, 2, 2, 1, {4, 0, 0, 0, 0, 0}, UINT32_MAX) == 15);
static_assert(PixelUnpackBytes(2, 2, 2, 4, {8, 4, 3, 1, 1, 1}, 140) == 140);
static_assert(!PixelUnpackBytes(2, 2, 2, 4, {8, 4, 3, 1, 1, 1}, 139));
static_assert(!PixelUnpackBytes(INT32_MAX, INT32_MAX, INT32_MAX, 16, {8, 0, 0, 0, 0, 0}, UINT32_MAX));

struct GlStringCache {
    std::mutex mutex;
    std::unordered_map<std::string, uint32_t> values;
};

constexpr size_t kMaxGuestGlSyncs = 1024;
constexpr uint64_t kFirstGuestGlSyncHandle = 0x32000000ULL;

struct GuestGlSyncSlot {
    uint32_t guest_handle = 0;
    GLsync host_sync = nullptr;
};

std::array<GuestGlSyncSlot, kMaxGuestGlSyncs> g_guest_gl_syncs{};
std::mutex g_guest_gl_sync_mutex;
size_t g_guest_gl_sync_count = 0;
uint64_t g_next_guest_gl_sync_handle = kFirstGuestGlSyncHandle;

// ponytail: fixed 1024-slot linear scan; raise the cap only if measured sync volume needs it.
GuestGlSyncSlot* FindGuestGlSyncLocked(uint32_t handle) {
    if (!handle) return nullptr;
    for (GuestGlSyncSlot& slot : g_guest_gl_syncs) {
        if (slot.guest_handle == handle) return &slot;
    }
    return nullptr;
}

GuestGlSyncSlot* FindEmptyGuestGlSyncSlotLocked() {
    for (GuestGlSyncSlot& slot : g_guest_gl_syncs) {
        if (!slot.guest_handle) return &slot;
    }
    return nullptr;
}

uint32_t AllocateGuestGlSyncHandleLocked() {
    if (g_next_guest_gl_sync_handle > std::numeric_limits<uint32_t>::max()) return 0;
    return static_cast<uint32_t>(g_next_guest_gl_sync_handle++);
}

}  // namespace

static_assert(sizeof(GLuint64) == 8);
static_assert(sizeof(GLenum) == 4);
static_assert(std::is_pointer_v<GLsync>);
static_assert(sizeof(GLsync) == sizeof(void*));

bool GlSyncRegistrySelfCheck() {
    std::lock_guard<std::mutex> lock(g_guest_gl_sync_mutex);
    size_t occupied = 0;
    bool unique = true;
    bool slots_valid = true;
    for (size_t i = 0; i < g_guest_gl_syncs.size(); ++i) {
        const GuestGlSyncSlot& slot = g_guest_gl_syncs[i];
        if (slot.guest_handle) {
            ++occupied;
            slots_valid = slots_valid && slot.host_sync != nullptr;
            for (size_t j = i + 1; j < g_guest_gl_syncs.size(); ++j) {
                unique = unique && slot.guest_handle != g_guest_gl_syncs[j].guest_handle;
            }
        } else {
            slots_valid = slots_valid && slot.host_sync == nullptr;
        }
    }
    const bool ok = occupied == g_guest_gl_sync_count && slots_valid && unique &&
                    FindGuestGlSyncLocked(0) == nullptr &&
                    FindGuestGlSyncLocked(static_cast<uint32_t>(kFirstGuestGlSyncHandle - 1)) == nullptr &&
                    g_next_guest_gl_sync_handle != 0 &&
                    g_next_guest_gl_sync_handle <=
                        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    LOGE("GLsync registry self-check %s", ok ? "OK" : "FAILED");
    return ok;
}

void ResetGlSyncRegistry() {
    std::lock_guard<std::mutex> lock(g_guest_gl_sync_mutex);
    const bool can_delete = eglGetCurrentContext() != EGL_NO_CONTEXT;
    size_t deleted = 0;
    size_t dropped = 0;
    for (GuestGlSyncSlot& slot : g_guest_gl_syncs) {
        if (slot.host_sync) {
            if (can_delete) {
                glDeleteSync(slot.host_sync);
                ++deleted;
            } else {
                ++dropped;
            }
        }
        slot = {};
    }
    g_guest_gl_sync_count = 0;
    LOGE("GLsync registry reset deleted=%zu dropped_without_context=%zu", deleted, dropped);
}

// Shared by 2D and 3D uploads. Only a PIXEL_UNPACK_BUFFER binding makes data an offset.
bool ResolveGlPixels(DynarmicHost& host, int width, int height, int depth,
                     uint32_t format, uint32_t type, uint32_t address, const void** pixels, bool volume) {
    GLint pbo = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &pbo);
    if (pbo) {
        *pixels = reinterpret_cast<const void*>(static_cast<uintptr_t>(address));
        return true;
    }
    if (width < 0 || height < 0 || depth < 0) return false;
    if (!width || !height || !depth) { *pixels = nullptr; return true; }
    uint32_t bpp = 0, component_bytes = 0;
    switch (type) {
        case GL_UNSIGNED_SHORT_5_6_5: case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_5_5_5_1: bpp = 2; break;
        case GL_UNSIGNED_INT_2_10_10_10_REV: case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV: case GL_UNSIGNED_INT_24_8: bpp = 4; break;
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV: bpp = 8; break;
        case GL_BYTE: case GL_UNSIGNED_BYTE: component_bytes = 1; break;
        case GL_SHORT: case GL_UNSIGNED_SHORT: case GL_HALF_FLOAT: case GL_HALF_FLOAT_OES:
            component_bytes = 2;
            break;
        case GL_INT: case GL_UNSIGNED_INT: case GL_FLOAT: component_bytes = 4; break;
        default: return false;
    }
    if (!bpp) {
        uint32_t components = 0;
        switch (format) {
            case GL_RED: case GL_RED_INTEGER: case GL_ALPHA: case GL_LUMINANCE:
            case GL_DEPTH_COMPONENT: components = 1; break;
            case GL_RG: case GL_RG_INTEGER: case GL_LUMINANCE_ALPHA: components = 2; break;
            case GL_RGB: case GL_RGB_INTEGER: case GL_SRGB_EXT: components = 3; break;
            case GL_RGBA: case GL_RGBA_INTEGER: case GL_BGRA_EXT: case GL_SRGB_ALPHA_EXT:
                components = 4;
                break;
            case GL_DEPTH_STENCIL:
                bpp = 4;
                break;
            default: return false;
        }
        if (!bpp) bpp = components * component_bytes;
    }
    std::array<int, 6> unpack{4, 0, 0, 0, 0, 0};
    constexpr GLenum names[] = {GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH, GL_UNPACK_IMAGE_HEIGHT,
                               GL_UNPACK_SKIP_PIXELS, GL_UNPACK_SKIP_ROWS, GL_UNPACK_SKIP_IMAGES};
    for (size_t i = 0; i < unpack.size(); ++i) glGetIntegerv(names[i], &unpack[i]);
    if (!volume) { unpack[2] = 0; unpack[5] = 0; }
    const auto bytes = PixelUnpackBytes(width, height, depth, bpp, unpack, host.GetGuestMemorySize());
    if (!bytes || !GuestRangeValid(host.GetGuestMemorySize(), address, *bytes)) {
        LOGE("Rejected pixel span address=0x%08X dimensions=%dx%dx%d type=0x%X format=0x%X",
             address, width, height, depth, type, format);
        return false;
    }
    *pixels = GuestSpan(host.GetGuestMemoryBase(), host.GetGuestMemorySize(), address, *bytes);
    return true;
}

void TrackGlContext(EGLContext context, EGLContext share) {
    if (context == EGL_NO_CONTEXT) return;
    std::lock_guard lock(s_guest_mapping_mutex);
    // A driver can recycle an EGLContext address after destruction.
    const uint64_t group = share == EGL_NO_CONTEXT ? s_next_gl_group++ : ContextGroupLocked(share);
    s_gl_context_groups[context] = group;
}

void ForgetGuestBufferMappings(DynarmicHost& host, GLsizei count, const GLuint* buffers) {
    if (count <= 0 || !buffers) return;
    std::lock_guard lock(s_guest_mapping_mutex);
    const uint64_t group = ContextGroupLocked(eglGetCurrentContext());
    for (auto it = s_guest_mapped_buffers.begin(); it != s_guest_mapped_buffers.end();) {
        const auto& mapping = it->second;
        if (mapping.group == group && std::find(buffers, buffers + count, mapping.buffer) != buffers + count) {
            host.GuestFree(mapping.guest_address);
            it = s_guest_mapped_buffers.erase(it);
        } else ++it;
    }
}

void RefreshGlBindings();
void TrackGlBufferBinding(GLenum target, GLuint buffer);
GLuint GetBoundGlBuffer(GLenum target);

static bool ResolveDrawElementIndices(DynarmicHost& host, GLsizei count,
                                      GLenum type, uint32_t address,
                                      const void** indices,
                                      const char* function_name) {
    if (GetBoundGlBuffer(GL_ELEMENT_ARRAY_BUFFER) != 0) {
        *indices = reinterpret_cast<const void*>(static_cast<uintptr_t>(address));
        return true;
    }
    if (count <= 0) {
        *indices = nullptr;
        return true;
    }

    size_t element_size = 0;
    switch (type) {
        case GL_UNSIGNED_BYTE: element_size = sizeof(GLubyte); break;
        case GL_UNSIGNED_SHORT: element_size = sizeof(GLushort); break;
        case GL_UNSIGNED_INT: element_size = sizeof(GLuint); break;
        default:
            // Let the driver report GL_INVALID_ENUM, but never expose an unchecked
            // guest address to the host for an element type it cannot consume.
            *indices = nullptr;
            return true;
    }

    size_t bytes = 0;
    if (!CheckedArrayBytes(static_cast<size_t>(count), 1, element_size, &bytes) ||
        !GuestRangeValid(host.GetGuestMemorySize(), address, bytes)) {
        LOGE("%s rejected guest indices: address=0x%08X count=%d type=0x%X",
             function_name, address, static_cast<int>(count), type);
        return false;
    }
    *indices = GuestSpan(host.GetGuestMemoryBase(), host.GetGuestMemorySize(),
                         address, bytes);
    return true;
}

template <typename Proc>
static Proc ResolveBaseVertexProc(std::atomic<Proc>& cached,
                                  const char* first, const char* second,
                                  const char* core) {
    Proc function = cached.load(std::memory_order_acquire);
    if (function) return function;

    function = reinterpret_cast<Proc>(eglGetProcAddress(first));
    if (!function) function = reinterpret_cast<Proc>(eglGetProcAddress(second));
    if (!function) function = reinterpret_cast<Proc>(eglGetProcAddress(core));
    // A lookup before an EGL context exists may legitimately fail. Cache only a
    // real entry point so that a later draw can retry after eglMakeCurrent.
    if (function) cached.store(function, std::memory_order_release);
    return function;
}

static PFNGLDRAWELEMENTSBASEVERTEXOESPROC
ResolveDrawElementsBaseVertexProc(bool prefer_ext) {
    static std::atomic<PFNGLDRAWELEMENTSBASEVERTEXOESPROC> oes_cached{nullptr};
    static std::atomic<PFNGLDRAWELEMENTSBASEVERTEXOESPROC> ext_cached{nullptr};
    return prefer_ext
        ? ResolveBaseVertexProc(ext_cached, "glDrawElementsBaseVertexEXT",
                                "glDrawElementsBaseVertexOES",
                                "glDrawElementsBaseVertex")
        : ResolveBaseVertexProc(oes_cached, "glDrawElementsBaseVertexOES",
                                "glDrawElementsBaseVertexEXT",
                                "glDrawElementsBaseVertex");
}

static PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXOESPROC
ResolveDrawElementsInstancedBaseVertexProc(bool prefer_ext) {
    static std::atomic<PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXOESPROC> oes_cached{nullptr};
    static std::atomic<PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXOESPROC> ext_cached{nullptr};
    return prefer_ext
        ? ResolveBaseVertexProc(ext_cached,
                                "glDrawElementsInstancedBaseVertexEXT",
                                "glDrawElementsInstancedBaseVertexOES",
                                "glDrawElementsInstancedBaseVertex")
        : ResolveBaseVertexProc(oes_cached,
                                "glDrawElementsInstancedBaseVertexOES",
                                "glDrawElementsInstancedBaseVertexEXT",
                                "glDrawElementsInstancedBaseVertex");
}

void RegisterGles2CoreThunks(ThunkManager& manager) {
    manager.RegisterThunk("glBlendColor", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLfloat red = args.F32();
        const GLfloat green = args.F32();
        const GLfloat blue = args.F32();
        const GLfloat alpha = args.F32();
        glBlendColor(red, green, blue, alpha);
    });
    manager.RegisterThunk("glBlendEquation", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        glBlendEquation(static_cast<GLenum>(args.U32()));
    });
    manager.RegisterThunk("glBlendEquationSeparate", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum rgb = static_cast<GLenum>(args.U32());
        const GLenum alpha = static_cast<GLenum>(args.U32());
        glBlendEquationSeparate(rgb, alpha);
    });
    manager.RegisterThunk("glClearDepthf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        glClearDepthf(args.F32());
    });
    manager.RegisterThunk("glClearStencil", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        glClearStencil(static_cast<GLint>(args.I32()));
    });
    manager.RegisterThunk("glCompressedTexSubImage2D", [](Dynarmic::A32::Jit& jit,
                                                     DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLint level = static_cast<GLint>(args.I32());
        const GLint xoffset = static_cast<GLint>(args.I32());
        const GLint yoffset = static_cast<GLint>(args.I32());
        const GLsizei width = static_cast<GLsizei>(args.I32());
        const GLsizei height = static_cast<GLsizei>(args.I32());
        const GLenum format = static_cast<GLenum>(args.U32());
        const GLsizei image_size = static_cast<GLsizei>(args.I32());
        const uint32_t data_address = args.Ptr();
        GLint pbo = 0;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &pbo);
        const void* data = nullptr;
        if (pbo != 0) {
            data = reinterpret_cast<const void*>(static_cast<uintptr_t>(data_address));
        } else {
            bool valid = false;
            data = CountedGuestBytes(
                host, data_address, image_size, "glCompressedTexSubImage2D", &valid);
            if (!valid) return;
        }
        glCompressedTexSubImage2D(target, level, xoffset, yoffset, width, height,
                                  format, image_size, data);
    });
    manager.RegisterThunk("glCopyTexImage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLint level = static_cast<GLint>(args.I32());
        const GLenum internal_format = static_cast<GLenum>(args.U32());
        const GLint x = static_cast<GLint>(args.I32());
        const GLint y = static_cast<GLint>(args.I32());
        const GLsizei width = static_cast<GLsizei>(args.I32());
        const GLsizei height = static_cast<GLsizei>(args.I32());
        const GLint border = static_cast<GLint>(args.I32());
        glCopyTexImage2D(target, level, internal_format, x, y, width, height, border);
    });
    manager.RegisterThunk("glCopyTexSubImage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLint level = static_cast<GLint>(args.I32());
        const GLint xoffset = static_cast<GLint>(args.I32());
        const GLint yoffset = static_cast<GLint>(args.I32());
        const GLint x = static_cast<GLint>(args.I32());
        const GLint y = static_cast<GLint>(args.I32());
        const GLsizei width = static_cast<GLsizei>(args.I32());
        const GLsizei height = static_cast<GLsizei>(args.I32());
        glCopyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height);
    });
    manager.RegisterThunk("glDetachShader", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const GLuint shader = static_cast<GLuint>(args.U32());
        glDetachShader(program, shader);
    });

    manager.RegisterThunk("glGetAttachedShaders", [](Dynarmic::A32::Jit& jit,
                                                DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const GLsizei max_count = static_cast<GLsizei>(args.I32());
        const uint32_t count_address = args.Ptr();
        const uint32_t shaders_address = args.Ptr();
        bool count_valid = false;
        GLsizei* count = OptionalGuestOutput<GLsizei>(
            host, count_address, 1, "glGetAttachedShaders", &count_valid);
        if (!count_valid) return;
        GLuint* shaders = nullptr;
        if (max_count > 0) {
            shaders = GuestOutputArray<GLuint>(
                host, shaders_address, static_cast<size_t>(max_count), 1,
                "glGetAttachedShaders");
            if (!shaders) return;
        }
        glGetAttachedShaders(program, max_count, count, shaders);
    });
    manager.RegisterThunk("glGetBufferParameteriv", [](Dynarmic::A32::Jit& jit,
                                                  DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        GLint* params = GuestOutputArray<GLint>(
            host, args.Ptr(), 1, 1, "glGetBufferParameteriv");
        if (!params) return;
        glGetBufferParameteriv(target, pname, params);
    });
    manager.RegisterThunk("glGetFloatv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum pname = static_cast<GLenum>(args.U32());
        const size_t count = GetFloatStateValueCount(pname);
        GLfloat* data = GuestOutputArray<GLfloat>(
            host, args.Ptr(), count, 1, "glGetFloatv");
        if (!data) return;
        glGetFloatv(pname, data);
    });
    manager.RegisterThunk("glGetShaderPrecisionFormat", [](Dynarmic::A32::Jit& jit,
                                                      DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum shader_type = static_cast<GLenum>(args.U32());
        const GLenum precision_type = static_cast<GLenum>(args.U32());
        GLint* range = GuestOutputArray<GLint>(
            host, args.Ptr(), 2, 1, "glGetShaderPrecisionFormat");
        GLint* precision = GuestOutputArray<GLint>(
            host, args.Ptr(), 1, 1, "glGetShaderPrecisionFormat");
        if (!range || !precision) return;
        glGetShaderPrecisionFormat(shader_type, precision_type, range, precision);
    });
    manager.RegisterThunk("glGetShaderSource", [](Dynarmic::A32::Jit& jit,
                                             DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint shader = static_cast<GLuint>(args.U32());
        const GLsizei buffer_size = static_cast<GLsizei>(args.I32());
        const uint32_t length_address = args.Ptr();
        const uint32_t source_address = args.Ptr();
        bool length_valid = false;
        GLsizei* length = OptionalGuestOutput<GLsizei>(
            host, length_address, 1, "glGetShaderSource", &length_valid);
        if (!length_valid) return;
        GLchar* source = nullptr;
        if (buffer_size > 0) {
            source = GuestOutputArray<GLchar>(
                host, source_address, static_cast<size_t>(buffer_size), 1,
                "glGetShaderSource");
            if (!source) return;
        }
        glGetShaderSource(shader, buffer_size, length, source);
    });
    manager.RegisterThunk("glGetTexParameterfv", [](Dynarmic::A32::Jit& jit,
                                               DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        GLfloat* params = GuestOutputArray<GLfloat>(
            host, args.Ptr(), GetTextureParameterValueCount(pname), 1,
            "glGetTexParameterfv");
        if (!params) return;
        glGetTexParameterfv(target, pname, params);
    });
    manager.RegisterThunk("glGetTexParameteriv", [](Dynarmic::A32::Jit& jit,
                                               DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        GLint* params = GuestOutputArray<GLint>(
            host, args.Ptr(), GetTextureParameterValueCount(pname), 1,
            "glGetTexParameteriv");
        if (!params) return;
        glGetTexParameteriv(target, pname, params);
    });
    manager.RegisterThunk("glGetUniformfv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const GLint location = static_cast<GLint>(args.I32());
        // A single GLES2 uniform location contains at most one mat4 (16 scalars).
        GLfloat* params = GuestOutputArray<GLfloat>(
            host, args.Ptr(), 16, 1, "glGetUniformfv");
        if (!params) return;
        glGetUniformfv(program, location, params);
    });
    manager.RegisterThunk("glGetUniformiv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const GLint location = static_cast<GLint>(args.I32());
        GLint* params = GuestOutputArray<GLint>(
            host, args.Ptr(), 16, 1, "glGetUniformiv");
        if (!params) return;
        glGetUniformiv(program, location, params);
    });
    manager.RegisterThunk("glGetVertexAttribfv", [](Dynarmic::A32::Jit& jit,
                                               DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        GLfloat* params = GuestOutputArray<GLfloat>(
            host, args.Ptr(), GetVertexAttribValueCount(pname), 1,
            "glGetVertexAttribfv");
        if (!params) return;
        glGetVertexAttribfv(index, pname, params);
    });
    manager.RegisterThunk("glGetVertexAttribiv", [](Dynarmic::A32::Jit& jit,
                                               DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        GLint* params = GuestOutputArray<GLint>(
            host, args.Ptr(), GetVertexAttribValueCount(pname), 1,
            "glGetVertexAttribiv");
        if (!params) return;
        glGetVertexAttribiv(index, pname, params);
    });
    manager.RegisterThunk("glGetVertexAttribPointerv", [](Dynarmic::A32::Jit& jit,
                                                     DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        uint32_t* guest_pointer = GuestOutputArray<uint32_t>(
            host, args.Ptr(), 1, 1, "glGetVertexAttribPointerv");
        if (!guest_pointer) return;
        void* host_pointer = nullptr;
        GLint array_buffer_binding = 0;
        glGetVertexAttribPointerv(index, pname, &host_pointer);
        glGetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING,
                            &array_buffer_binding);
        *guest_pointer = HostVertexPointerToGuest(
            host, host_pointer, array_buffer_binding);
    });

    manager.RegisterThunk("glIsBuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        jit.Regs()[0] = static_cast<uint32_t>(glIsBuffer(static_cast<GLuint>(args.U32())));
    });
    manager.RegisterThunk("glIsEnabled", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        jit.Regs()[0] = static_cast<uint32_t>(glIsEnabled(static_cast<GLenum>(args.U32())));
    });
    manager.RegisterThunk("glIsFramebuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        jit.Regs()[0] = static_cast<uint32_t>(
            glIsFramebuffer(static_cast<GLuint>(args.U32())));
    });
    manager.RegisterThunk("glIsProgram", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        jit.Regs()[0] = static_cast<uint32_t>(glIsProgram(static_cast<GLuint>(args.U32())));
    });
    manager.RegisterThunk("glIsRenderbuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        jit.Regs()[0] = static_cast<uint32_t>(
            glIsRenderbuffer(static_cast<GLuint>(args.U32())));
    });
    manager.RegisterThunk("glIsShader", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        jit.Regs()[0] = static_cast<uint32_t>(glIsShader(static_cast<GLuint>(args.U32())));
    });
    manager.RegisterThunk("glIsTexture", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        jit.Regs()[0] = static_cast<uint32_t>(glIsTexture(static_cast<GLuint>(args.U32())));
    });
    manager.RegisterThunk("glReleaseShaderCompiler", [](Dynarmic::A32::Jit&, DynarmicHost&) {
        glReleaseShaderCompiler();
    });
    manager.RegisterThunk("glSampleCoverage", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLfloat value = args.F32();
        const GLboolean invert = static_cast<GLboolean>(args.U32());
        glSampleCoverage(value, invert);
    });

    manager.RegisterThunk("glStencilFunc", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum function = static_cast<GLenum>(args.U32());
        const GLint reference = static_cast<GLint>(args.I32());
        const GLuint mask = static_cast<GLuint>(args.U32());
        glStencilFunc(function, reference, mask);
    });
    manager.RegisterThunk("glStencilFuncSeparate", [](Dynarmic::A32::Jit& jit,
                                                 DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum face = static_cast<GLenum>(args.U32());
        const GLenum function = static_cast<GLenum>(args.U32());
        const GLint reference = static_cast<GLint>(args.I32());
        const GLuint mask = static_cast<GLuint>(args.U32());
        glStencilFuncSeparate(face, function, reference, mask);
    });
    manager.RegisterThunk("glStencilMask", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        glStencilMask(static_cast<GLuint>(args.U32()));
    });
    manager.RegisterThunk("glStencilMaskSeparate", [](Dynarmic::A32::Jit& jit,
                                                 DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum face = static_cast<GLenum>(args.U32());
        const GLuint mask = static_cast<GLuint>(args.U32());
        glStencilMaskSeparate(face, mask);
    });
    manager.RegisterThunk("glStencilOp", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum fail = static_cast<GLenum>(args.U32());
        const GLenum depth_fail = static_cast<GLenum>(args.U32());
        const GLenum depth_pass = static_cast<GLenum>(args.U32());
        glStencilOp(fail, depth_fail, depth_pass);
    });
    manager.RegisterThunk("glStencilOpSeparate", [](Dynarmic::A32::Jit& jit,
                                               DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum face = static_cast<GLenum>(args.U32());
        const GLenum stencil_fail = static_cast<GLenum>(args.U32());
        const GLenum depth_fail = static_cast<GLenum>(args.U32());
        const GLenum depth_pass = static_cast<GLenum>(args.U32());
        glStencilOpSeparate(face, stencil_fail, depth_fail, depth_pass);
    });
    manager.RegisterThunk("glTexParameterfv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        const GLfloat* params = GuestInputArray<GLfloat>(
            host, args.Ptr(), GetTextureParameterValueCount(pname), 1,
            "glTexParameterfv");
        if (!params) return;
        glTexParameterfv(target, pname, params);
    });

    manager.RegisterThunk("glUniform1fv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t value_address = args.Ptr();
        bool valid = false;
        const GLfloat* value = CountedGuestValues<GLfloat>(
            host, value_address, count, 1, "glUniform1fv", &valid);
        if (!valid) return;
        glUniform1fv(location, count, value);
    });
    manager.RegisterThunk("glUniform1iv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t value_address = args.Ptr();
        bool valid = false;
        const GLint* value = CountedGuestValues<GLint>(
            host, value_address, count, 1, "glUniform1iv", &valid);
        if (!valid) return;
        glUniform1iv(location, count, value);
    });
    manager.RegisterThunk("glUniform2fv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t value_address = args.Ptr();
        bool valid = false;
        const GLfloat* value = CountedGuestValues<GLfloat>(
            host, value_address, count, 2, "glUniform2fv", &valid);
        if (!valid) return;
        glUniform2fv(location, count, value);
    });
    manager.RegisterThunk("glUniform2i", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLint x = static_cast<GLint>(args.I32());
        const GLint y = static_cast<GLint>(args.I32());
        glUniform2i(location, x, y);
    });
    manager.RegisterThunk("glUniform2iv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t value_address = args.Ptr();
        bool valid = false;
        const GLint* value = CountedGuestValues<GLint>(
            host, value_address, count, 2, "glUniform2iv", &valid);
        if (!valid) return;
        glUniform2iv(location, count, value);
    });
    manager.RegisterThunk("glUniform3i", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLint x = static_cast<GLint>(args.I32());
        const GLint y = static_cast<GLint>(args.I32());
        const GLint z = static_cast<GLint>(args.I32());
        glUniform3i(location, x, y, z);
    });
    manager.RegisterThunk("glUniform3iv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t value_address = args.Ptr();
        bool valid = false;
        const GLint* value = CountedGuestValues<GLint>(
            host, value_address, count, 3, "glUniform3iv", &valid);
        if (!valid) return;
        glUniform3iv(location, count, value);
    });
    manager.RegisterThunk("glUniform4i", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLint x = static_cast<GLint>(args.I32());
        const GLint y = static_cast<GLint>(args.I32());
        const GLint z = static_cast<GLint>(args.I32());
        const GLint w = static_cast<GLint>(args.I32());
        glUniform4i(location, x, y, z, w);
    });
    manager.RegisterThunk("glUniform4iv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t value_address = args.Ptr();
        bool valid = false;
        const GLint* value = CountedGuestValues<GLint>(
            host, value_address, count, 4, "glUniform4iv", &valid);
        if (!valid) return;
        glUniform4iv(location, count, value);
    });
    manager.RegisterThunk("glUniformMatrix2fv", [](Dynarmic::A32::Jit& jit,
                                              DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const GLboolean transpose = static_cast<GLboolean>(args.U32());
        const uint32_t value_address = args.Ptr();
        bool valid = false;
        const GLfloat* value = CountedGuestValues<GLfloat>(
            host, value_address, count, 4, "glUniformMatrix2fv", &valid);
        if (!valid) return;
        glUniformMatrix2fv(location, count, transpose, value);
    });
    manager.RegisterThunk("glValidateProgram", [](Dynarmic::A32::Jit& jit,
                                             DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        glValidateProgram(static_cast<GLuint>(args.U32()));
    });

    manager.RegisterThunk("glVertexAttrib1f", [](Dynarmic::A32::Jit& jit,
                                           DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLfloat x = args.F32();
        glVertexAttrib1f(index, x);
    });
    manager.RegisterThunk("glVertexAttrib1fv", [](Dynarmic::A32::Jit& jit,
                                            DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLfloat* values = GuestInputArray<GLfloat>(
            host, args.Ptr(), 1, 1, "glVertexAttrib1fv");
        if (!values) return;
        glVertexAttrib1fv(index, values);
    });
    manager.RegisterThunk("glVertexAttrib2f", [](Dynarmic::A32::Jit& jit,
                                           DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLfloat x = args.F32();
        const GLfloat y = args.F32();
        glVertexAttrib2f(index, x, y);
    });
    manager.RegisterThunk("glVertexAttrib2fv", [](Dynarmic::A32::Jit& jit,
                                            DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLfloat* values = GuestInputArray<GLfloat>(
            host, args.Ptr(), 1, 2, "glVertexAttrib2fv");
        if (!values) return;
        glVertexAttrib2fv(index, values);
    });
    manager.RegisterThunk("glVertexAttrib3f", [](Dynarmic::A32::Jit& jit,
                                           DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLfloat x = args.F32();
        const GLfloat y = args.F32();
        const GLfloat z = args.F32();
        glVertexAttrib3f(index, x, y, z);
    });
    manager.RegisterThunk("glVertexAttrib3fv", [](Dynarmic::A32::Jit& jit,
                                            DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLfloat* values = GuestInputArray<GLfloat>(
            host, args.Ptr(), 1, 3, "glVertexAttrib3fv");
        if (!values) return;
        glVertexAttrib3fv(index, values);
    });
    manager.RegisterThunk("glVertexAttrib4f", [](Dynarmic::A32::Jit& jit,
                                           DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLfloat x = args.F32();
        const GLfloat y = args.F32();
        const GLfloat z = args.F32();
        const GLfloat w = args.F32();
        glVertexAttrib4f(index, x, y, z, w);
    });
    manager.RegisterThunk("glVertexAttrib4fv", [](Dynarmic::A32::Jit& jit,
                                             DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLfloat* values = GuestInputArray<GLfloat>(
            host, args.Ptr(), 1, 4, "glVertexAttrib4fv");
        if (!values) return;
        glVertexAttrib4fv(index, values);
    });

    // Keep the advertised GLES 3.0 surface complete. Unity resolves these as a
    // capability table; returning null for isolated core functions disables otherwise
    // valid shader, UBO, program-binary, instancing, and 3D-texture paths.
    manager.RegisterThunk("glFinish", [](Dynarmic::A32::Jit&, DynarmicHost&) {
        glFinish();
    });
    manager.RegisterThunk("glGetFramebufferAttachmentParameteriv",
        [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
            Arm32CallArgs args(jit, host.GetGuestMemoryBase());
            const GLenum target = static_cast<GLenum>(args.U32());
            const GLenum attachment = static_cast<GLenum>(args.U32());
            const GLenum pname = static_cast<GLenum>(args.U32());
            GLint* params = GuestOutputArray<GLint>(
                host, args.Ptr(), 1, 1, "glGetFramebufferAttachmentParameteriv");
            if (params) {
                glGetFramebufferAttachmentParameteriv(target, attachment, pname, params);
            }
        });
    manager.RegisterThunk("glTexParameteriv", [](Dynarmic::A32::Jit& jit,
                                                   DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        const GLint* params = GuestInputArray<GLint>(
            host, args.Ptr(), GetTextureParameterValueCount(pname), 1,
            "glTexParameteriv");
        if (params) glTexParameteriv(target, pname, params);
    });

    manager.RegisterThunk("glGenQueries", [](Dynarmic::A32::Jit& jit,
                                               DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t ids_address = args.Ptr();
        GLuint* ids = count > 0 ? GuestOutputArray<GLuint>(
            host, ids_address, static_cast<size_t>(count), 1, "glGenQueries") : nullptr;
        if (count <= 0 || ids) glGenQueries(count, ids);
    });
    manager.RegisterThunk("glDeleteQueries", [](Dynarmic::A32::Jit& jit,
                                                  DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        bool valid = false;
        const GLuint* ids = CountedGuestValues<GLuint>(
            host, args.Ptr(), count, 1, "glDeleteQueries", &valid);
        if (valid) glDeleteQueries(count, ids);
    });
    manager.RegisterThunk("glBeginQuery", [](Dynarmic::A32::Jit& jit,
                                               DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        glBeginQuery(static_cast<GLenum>(args.U32()), static_cast<GLuint>(args.U32()));
    });
    manager.RegisterThunk("glEndQuery", [](Dynarmic::A32::Jit& jit,
                                             DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        glEndQuery(static_cast<GLenum>(args.U32()));
    });

    manager.RegisterThunk("glBeginTransformFeedback", [](Dynarmic::A32::Jit& jit,
                                                            DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        glBeginTransformFeedback(static_cast<GLenum>(args.U32()));
    });
    manager.RegisterThunk("glEndTransformFeedback", [](Dynarmic::A32::Jit&,
                                                          DynarmicHost&) {
        glEndTransformFeedback();
    });
    manager.RegisterThunk("glBindTransformFeedback", [](Dynarmic::A32::Jit& jit,
                                                           DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        glBindTransformFeedback(static_cast<GLenum>(args.U32()),
                                static_cast<GLuint>(args.U32()));
    });
    manager.RegisterThunk("glGenTransformFeedbacks", [](Dynarmic::A32::Jit& jit,
                                                           DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t ids_address = args.Ptr();
        GLuint* ids = count > 0 ? GuestOutputArray<GLuint>(
            host, ids_address, static_cast<size_t>(count), 1,
            "glGenTransformFeedbacks") : nullptr;
        if (count <= 0 || ids) glGenTransformFeedbacks(count, ids);
    });
    manager.RegisterThunk("glDeleteTransformFeedbacks", [](Dynarmic::A32::Jit& jit,
                                                              DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        bool valid = false;
        const GLuint* ids = CountedGuestValues<GLuint>(
            host, args.Ptr(), count, 1, "glDeleteTransformFeedbacks", &valid);
        if (valid) glDeleteTransformFeedbacks(count, ids);
    });
    manager.RegisterThunk("glTransformFeedbackVaryings", [](Dynarmic::A32::Jit& jit,
                                                               DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t varyings_address = args.Ptr();
        const GLenum buffer_mode = static_cast<GLenum>(args.U32());
        std::vector<const GLchar*> varyings;
        if (!GuestGlNameArray(host, varyings_address, count,
                              "glTransformFeedbackVaryings", &varyings)) {
            return;
        }
        glTransformFeedbackVaryings(
            program, count, count > 0 ? varyings.data() : nullptr, buffer_mode);
    });

    manager.RegisterThunk("glBlitFramebuffer", [](Dynarmic::A32::Jit& jit,
                                                    DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint src_x0 = static_cast<GLint>(args.I32());
        const GLint src_y0 = static_cast<GLint>(args.I32());
        const GLint src_x1 = static_cast<GLint>(args.I32());
        const GLint src_y1 = static_cast<GLint>(args.I32());
        const GLint dst_x0 = static_cast<GLint>(args.I32());
        const GLint dst_y0 = static_cast<GLint>(args.I32());
        const GLint dst_x1 = static_cast<GLint>(args.I32());
        const GLint dst_y1 = static_cast<GLint>(args.I32());
        const GLbitfield mask = static_cast<GLbitfield>(args.U32());
        const GLenum filter = static_cast<GLenum>(args.U32());
        glBlitFramebuffer(src_x0, src_y0, src_x1, src_y1,
                          dst_x0, dst_y0, dst_x1, dst_y1, mask, filter);
    });
    manager.RegisterThunk("glRenderbufferStorageMultisample",
        [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
            Arm32CallArgs args(jit, host.GetGuestMemoryBase());
            const GLenum target = static_cast<GLenum>(args.U32());
            const GLsizei samples = static_cast<GLsizei>(args.I32());
            const GLenum internal_format = static_cast<GLenum>(args.U32());
            const GLsizei width = static_cast<GLsizei>(args.I32());
            const GLsizei height = static_cast<GLsizei>(args.I32());
            glRenderbufferStorageMultisample(
                target, samples, internal_format, width, height);
        });
    manager.RegisterThunk("glFramebufferTextureLayer", [](Dynarmic::A32::Jit& jit,
                                                             DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLenum attachment = static_cast<GLenum>(args.U32());
        const GLuint texture = static_cast<GLuint>(args.U32());
        const GLint level = static_cast<GLint>(args.I32());
        const GLint layer = static_cast<GLint>(args.I32());
        glFramebufferTextureLayer(target, attachment, texture, level, layer);
    });
    manager.RegisterThunk("glReadBuffer", [](Dynarmic::A32::Jit& jit,
                                               DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        glReadBuffer(static_cast<GLenum>(args.U32()));
    });

    manager.RegisterThunk("glDrawArraysInstanced", [](Dynarmic::A32::Jit& jit,
                                                         DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum mode = static_cast<GLenum>(args.U32());
        const GLint first = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const GLsizei instance_count = static_cast<GLsizei>(args.I32());
        glDrawArraysInstanced(mode, first, count, instance_count);
    });
    manager.RegisterThunk("glDrawElementsInstanced", [](Dynarmic::A32::Jit& jit,
                                                           DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum mode = static_cast<GLenum>(args.U32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const GLenum type = static_cast<GLenum>(args.U32());
        const uint32_t indices_address = args.Ptr();
        const GLsizei instance_count = static_cast<GLsizei>(args.I32());

        const void* indices = nullptr;
        if (GetBoundGlBuffer(GL_ELEMENT_ARRAY_BUFFER) != 0) {
            indices = reinterpret_cast<const void*>(
                static_cast<uintptr_t>(indices_address));
        } else if (count > 0 && instance_count > 0) {
            size_t element_size = 0;
            switch (type) {
                case GL_UNSIGNED_BYTE: element_size = sizeof(GLubyte); break;
                case GL_UNSIGNED_SHORT: element_size = sizeof(GLushort); break;
                case GL_UNSIGNED_INT: element_size = sizeof(GLuint); break;
                default:
                    // Preserve the driver's GL_INVALID_ENUM path without exposing an
                    // unchecked guest pointer to the host.
                    glDrawElementsInstanced(
                        mode, count, type, nullptr, instance_count);
                    return;
            }
            size_t bytes = 0;
            if (!CheckedArrayBytes(static_cast<size_t>(count), 1, element_size, &bytes) ||
                !GuestRangeValid(host.GetGuestMemorySize(), indices_address, bytes)) {
                LOGE("glDrawElementsInstanced rejected guest indices: address=0x%08X count=%d type=0x%X",
                     indices_address, static_cast<int>(count), type);
                return;
            }
            indices = GuestSpan(host.GetGuestMemoryBase(), host.GetGuestMemorySize(),
                                indices_address, bytes);
        }
        glDrawElementsInstanced(mode, count, type, indices, instance_count);
    });
    auto make_draw_elements_base_vertex = [](bool prefer_ext) {
      return [prefer_ext](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum mode = static_cast<GLenum>(args.U32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const GLenum type = static_cast<GLenum>(args.U32());
        const uint32_t indices_address = args.Ptr();
        const GLint base_vertex = static_cast<GLint>(args.I32());

        const void* indices = nullptr;
        if (!ResolveDrawElementIndices(host, count, type, indices_address,
                                       &indices, "glDrawElementsBaseVertex")) {
            return;
        }
        const auto function = ResolveDrawElementsBaseVertexProc(prefer_ext);
        if (function) {
            function(mode, count, type, indices, base_vertex);
        } else if (base_vertex == 0) {
            glDrawElements(mode, count, type, indices);
        } else {
            LOGE("Missing native glDrawElementsBaseVertex entry point");
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
        }
      };
    };
    manager.RegisterThunk("glDrawElementsBaseVertexOES",
                          make_draw_elements_base_vertex(false));
    manager.RegisterThunk("glDrawElementsBaseVertexEXT",
                          make_draw_elements_base_vertex(true));

    auto make_draw_elements_instanced_base_vertex = [](bool prefer_ext) {
      return [prefer_ext](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum mode = static_cast<GLenum>(args.U32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const GLenum type = static_cast<GLenum>(args.U32());
        const uint32_t indices_address = args.Ptr();
        const GLsizei instance_count = static_cast<GLsizei>(args.I32());
        const GLint base_vertex = static_cast<GLint>(args.I32());

        const void* indices = nullptr;
        if (instance_count > 0 &&
            !ResolveDrawElementIndices(host, count, type, indices_address,
                                       &indices,
                                       "glDrawElementsInstancedBaseVertex")) {
            return;
        }
        const auto function = ResolveDrawElementsInstancedBaseVertexProc(prefer_ext);
        if (function) {
            function(mode, count, type, indices, instance_count, base_vertex);
        } else if (base_vertex == 0) {
            glDrawElementsInstanced(mode, count, type, indices, instance_count);
        } else {
            LOGE("Missing native glDrawElementsInstancedBaseVertex entry point");
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
        }
      };
    };
    manager.RegisterThunk("glDrawElementsInstancedBaseVertexOES",
                          make_draw_elements_instanced_base_vertex(false));
    manager.RegisterThunk("glDrawElementsInstancedBaseVertexEXT",
                          make_draw_elements_instanced_base_vertex(true));
    manager.RegisterThunk("glCopyBufferSubData", [](Dynarmic::A32::Jit& jit,
                                                      DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum read_target = static_cast<GLenum>(args.U32());
        const GLenum write_target = static_cast<GLenum>(args.U32());
        const GLintptr read_offset = static_cast<GLintptr>(args.I32());
        const GLintptr write_offset = static_cast<GLintptr>(args.I32());
        const GLsizeiptr size = static_cast<GLsizeiptr>(args.I32());
        glCopyBufferSubData(read_target, write_target, read_offset, write_offset, size);
    });

    manager.RegisterThunk("glTexImage3D", [](Dynarmic::A32::Jit& jit,
                                              DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLint level = static_cast<GLint>(args.I32());
        const GLint internal_format = static_cast<GLint>(args.I32());
        const GLsizei width = static_cast<GLsizei>(args.I32());
        const GLsizei height = static_cast<GLsizei>(args.I32());
        const GLsizei depth = static_cast<GLsizei>(args.I32());
        const GLint border = static_cast<GLint>(args.I32());
        const GLenum format = static_cast<GLenum>(args.U32());
        const GLenum type = static_cast<GLenum>(args.U32());
        const uint32_t pixels_address = args.Ptr();

        GLint pbo = 0;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &pbo);
        const void* pixels = nullptr;
        if ((pbo != 0 || pixels_address != 0) &&
            !ResolveGlPixels(host, width, height, depth, format, type,
                             pixels_address, &pixels, true)) {
            return;
        }
        glTexImage3D(target, level, internal_format, width, height, depth,
                     border, format, type, pixels);
    });
    manager.RegisterThunk("glCompressedTexSubImage3D",
        [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
            Arm32CallArgs args(jit, host.GetGuestMemoryBase());
            const GLenum target = static_cast<GLenum>(args.U32());
            const GLint level = static_cast<GLint>(args.I32());
            const GLint x_offset = static_cast<GLint>(args.I32());
            const GLint y_offset = static_cast<GLint>(args.I32());
            const GLint z_offset = static_cast<GLint>(args.I32());
            const GLsizei width = static_cast<GLsizei>(args.I32());
            const GLsizei height = static_cast<GLsizei>(args.I32());
            const GLsizei depth = static_cast<GLsizei>(args.I32());
            const GLenum format = static_cast<GLenum>(args.U32());
            const GLsizei image_size = static_cast<GLsizei>(args.I32());
            const uint32_t data_address = args.Ptr();

            GLint pbo = 0;
            glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &pbo);
            const void* data = nullptr;
            if (pbo != 0) {
                data = reinterpret_cast<const void*>(static_cast<uintptr_t>(data_address));
            } else {
                bool valid = false;
                data = CountedGuestBytes(
                    host, data_address, image_size, "glCompressedTexSubImage3D", &valid);
                if (!valid) return;
            }
            glCompressedTexSubImage3D(target, level, x_offset, y_offset, z_offset,
                                      width, height, depth, format, image_size, data);
        });

    manager.RegisterThunk("glGetIntegeri_v", [](Dynarmic::A32::Jit& jit,
                                                  DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLuint index = static_cast<GLuint>(args.U32());
        GLint* data = GuestOutputArray<GLint>(
            host, args.Ptr(), 1, 1, "glGetIntegeri_v");
        if (data) glGetIntegeri_v(target, index, data);
    });
    manager.RegisterThunk("glGetInternalformativ", [](Dynarmic::A32::Jit& jit,
                                                         DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLenum internal_format = static_cast<GLenum>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        const GLsizei buffer_size = static_cast<GLsizei>(args.I32());
        const uint32_t params_address = args.Ptr();
        GLint* params = buffer_size > 0 ? GuestOutputArray<GLint>(
            host, params_address, static_cast<size_t>(buffer_size), 1,
            "glGetInternalformativ") : nullptr;
        if (buffer_size <= 0 || params) {
            glGetInternalformativ(target, internal_format, pname, buffer_size, params);
        }
    });

    manager.RegisterThunk("glGetProgramBinary", [](Dynarmic::A32::Jit& jit,
                                                      DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const GLsizei buffer_size = static_cast<GLsizei>(args.I32());
        const uint32_t length_address = args.Ptr();
        const uint32_t binary_format_address = args.Ptr();
        const uint32_t binary_address = args.Ptr();

        bool valid = false;
        GLsizei* length = OptionalGuestOutput<GLsizei>(
            host, length_address, 1, "glGetProgramBinary", &valid);
        if (!valid) return;
        GLenum* binary_format = GuestOutputArray<GLenum>(
            host, binary_format_address, 1, 1, "glGetProgramBinary");
        if (!binary_format) return;
        void* binary = buffer_size > 0 ? GuestOutputArray<uint8_t>(
            host, binary_address, static_cast<size_t>(buffer_size), 1,
            "glGetProgramBinary") : nullptr;
        if (buffer_size > 0 && !binary) return;
        glGetProgramBinary(program, buffer_size, length, binary_format, binary);
    });
    manager.RegisterThunk("glProgramBinary", [](Dynarmic::A32::Jit& jit,
                                                   DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const GLenum binary_format = static_cast<GLenum>(args.U32());
        const uint32_t binary_address = args.Ptr();
        const GLsizei length = static_cast<GLsizei>(args.I32());
        bool valid = false;
        const void* binary = CountedGuestBytes(
            host, binary_address, length, "glProgramBinary", &valid);
        if (valid) glProgramBinary(program, binary_format, binary, length);
    });

    manager.RegisterThunk("glGenSamplers", [](Dynarmic::A32::Jit& jit,
                                                DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t samplers_address = args.Ptr();
        GLuint* samplers = count > 0 ? GuestOutputArray<GLuint>(
            host, samplers_address, static_cast<size_t>(count), 1,
            "glGenSamplers") : nullptr;
        if (count <= 0 || samplers) glGenSamplers(count, samplers);
    });
    manager.RegisterThunk("glDeleteSamplers", [](Dynarmic::A32::Jit& jit,
                                                   DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        bool valid = false;
        const GLuint* samplers = CountedGuestValues<GLuint>(
            host, args.Ptr(), count, 1, "glDeleteSamplers", &valid);
        if (valid) glDeleteSamplers(count, samplers);
    });
    manager.RegisterThunk("glSamplerParameteri", [](Dynarmic::A32::Jit& jit,
                                                       DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint sampler = static_cast<GLuint>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        const GLint param = static_cast<GLint>(args.I32());
        glSamplerParameteri(sampler, pname, param);
    });

    manager.RegisterThunk("glGetUniformIndices", [](Dynarmic::A32::Jit& jit,
                                                       DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t names_address = args.Ptr();
        const uint32_t indices_address = args.Ptr();
        std::vector<const GLchar*> names;
        if (!GuestGlNameArray(host, names_address, count,
                              "glGetUniformIndices", &names)) {
            return;
        }
        GLuint* indices = count > 0 ? GuestOutputArray<GLuint>(
            host, indices_address, static_cast<size_t>(count), 1,
            "glGetUniformIndices") : nullptr;
        if (count <= 0 || indices) {
            glGetUniformIndices(program, count, count > 0 ? names.data() : nullptr, indices);
        }
    });
    manager.RegisterThunk("glGetActiveUniformsiv", [](Dynarmic::A32::Jit& jit,
                                                         DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        const uint32_t indices_address = args.Ptr();
        const GLenum pname = static_cast<GLenum>(args.U32());
        const uint32_t params_address = args.Ptr();
        bool valid = false;
        const GLuint* indices = CountedGuestValues<GLuint>(
            host, indices_address, count, 1, "glGetActiveUniformsiv", &valid);
        if (!valid) return;
        GLint* params = count > 0 ? GuestOutputArray<GLint>(
            host, params_address, static_cast<size_t>(count), 1,
            "glGetActiveUniformsiv") : nullptr;
        if (count <= 0 || params) glGetActiveUniformsiv(program, count, indices, pname, params);
    });
    manager.RegisterThunk("glGetUniformBlockIndex", [](Dynarmic::A32::Jit& jit,
                                                          DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const char* name = GuestGlName(host, args.Ptr());
        jit.Regs()[0] = name ? glGetUniformBlockIndex(program, name) : GL_INVALID_INDEX;
    });
    manager.RegisterThunk("glGetActiveUniformBlockiv", [](Dynarmic::A32::Jit& jit,
                                                             DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = static_cast<GLuint>(args.U32());
        const GLuint block_index = static_cast<GLuint>(args.U32());
        const GLenum pname = static_cast<GLenum>(args.U32());
        const uint32_t params_address = args.Ptr();
        size_t value_count = 1;
        if (pname == GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES) {
            GLint active_uniforms = 0;
            glGetActiveUniformBlockiv(
                program, block_index, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS,
                &active_uniforms);
            if (active_uniforms > 0) value_count = static_cast<size_t>(active_uniforms);
        }
        GLint* params = GuestOutputArray<GLint>(
            host, params_address, value_count, 1, "glGetActiveUniformBlockiv");
        if (params) glGetActiveUniformBlockiv(program, block_index, pname, params);
    });
    manager.RegisterThunk("glGetActiveUniformBlockName",
        [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
            Arm32CallArgs args(jit, host.GetGuestMemoryBase());
            const GLuint program = static_cast<GLuint>(args.U32());
            const GLuint block_index = static_cast<GLuint>(args.U32());
            const GLsizei buffer_size = static_cast<GLsizei>(args.I32());
            const uint32_t length_address = args.Ptr();
            const uint32_t name_address = args.Ptr();
            bool valid = false;
            GLsizei* length = OptionalGuestOutput<GLsizei>(
                host, length_address, 1, "glGetActiveUniformBlockName", &valid);
            if (!valid) return;
            GLchar* name = buffer_size > 0 ? GuestOutputArray<GLchar>(
                host, name_address, static_cast<size_t>(buffer_size), 1,
                "glGetActiveUniformBlockName") : nullptr;
            if (buffer_size > 0 && !name) return;
            glGetActiveUniformBlockName(
                program, block_index, buffer_size, length, name);
        });

    manager.RegisterThunk("glUniform1uiv", [](Dynarmic::A32::Jit& jit,
                                                DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        bool valid = false;
        const GLuint* values = CountedGuestValues<GLuint>(
            host, args.Ptr(), count, 1, "glUniform1uiv", &valid);
        if (valid) glUniform1uiv(location, count, values);
    });
    manager.RegisterThunk("glUniform2uiv", [](Dynarmic::A32::Jit& jit,
                                                DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        bool valid = false;
        const GLuint* values = CountedGuestValues<GLuint>(
            host, args.Ptr(), count, 2, "glUniform2uiv", &valid);
        if (valid) glUniform2uiv(location, count, values);
    });
    manager.RegisterThunk("glUniform3uiv", [](Dynarmic::A32::Jit& jit,
                                                DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        bool valid = false;
        const GLuint* values = CountedGuestValues<GLuint>(
            host, args.Ptr(), count, 3, "glUniform3uiv", &valid);
        if (valid) glUniform3uiv(location, count, values);
    });
    manager.RegisterThunk("glUniform4uiv", [](Dynarmic::A32::Jit& jit,
                                                DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLint location = static_cast<GLint>(args.I32());
        const GLsizei count = static_cast<GLsizei>(args.I32());
        bool valid = false;
        const GLuint* values = CountedGuestValues<GLuint>(
            host, args.Ptr(), count, 4, "glUniform4uiv", &valid);
        if (valid) glUniform4uiv(location, count, values);
    });
    manager.RegisterThunk("glVertexAttribIPointer", [](Dynarmic::A32::Jit& jit,
                                                          DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint index = static_cast<GLuint>(args.U32());
        const GLint size = static_cast<GLint>(args.I32());
        const GLenum type = static_cast<GLenum>(args.U32());
        const GLsizei stride = static_cast<GLsizei>(args.I32());
        const uint32_t pointer_address = args.Ptr();
        const void* pointer = nullptr;
        if (GetBoundGlBuffer(GL_ARRAY_BUFFER) != 0) {
            pointer = reinterpret_cast<const void*>(
                static_cast<uintptr_t>(pointer_address));
        } else if (pointer_address != 0) {
            if (!GuestRangeValid(host.GetGuestMemorySize(), pointer_address, 1)) {
                LOGE("glVertexAttribIPointer rejected guest pointer: address=0x%08X",
                     pointer_address);
                return;
            }
            pointer = GuestSpan(host.GetGuestMemoryBase(), host.GetGuestMemorySize(),
                                pointer_address, 1);
        }
        glVertexAttribIPointer(index, size, type, stride, pointer);
    });

    // Unity selects GLES3 on this GPU; these are real driver calls, not success stubs.
    auto strings = std::make_shared<GlStringCache>();
    manager.RegisterThunk("glGetStringi", [strings](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLubyte* value = glGetStringi(jit.Regs()[0], jit.Regs()[1]);
        if (!value) { jit.Regs()[0] = 0; return; }
        const std::string text(reinterpret_cast<const char*>(value));
        std::lock_guard<std::mutex> lock(strings->mutex);
        auto found = strings->values.find(text);
        if (found == strings->values.end()) {
            const uint32_t address = host.GuestAlloc(text.size() + 1, 1);
            if (!address) { jit.Regs()[0] = 0; return; }
            memcpy(host.GetGuestMemoryBase() + address, text.c_str(), text.size() + 1);
            found = strings->values.emplace(text, address).first;
        }
        jit.Regs()[0] = found->second;
    });
    manager.RegisterThunk("glBlendFuncSeparate", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glBlendFuncSeparate(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3]);
    });
    manager.RegisterThunk("glBindAttribLocation", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* name = GuestGlName(host, jit.Regs()[2]);
        if (name) glBindAttribLocation(jit.Regs()[0], jit.Regs()[1], name);
    });
    manager.RegisterThunk("glBufferSubData", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLintptr offset = static_cast<int32_t>(jit.Regs()[1]);
        const GLsizeiptr size = static_cast<int32_t>(jit.Regs()[2]);
        bool valid = false;
        const void* data = CountedGuestBytes(host, jit.Regs()[3], size, "glBufferSubData", &valid);
        if (valid) glBufferSubData(jit.Regs()[0], offset, size, data);
    });
    manager.RegisterThunk("glMapBufferRange", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLenum target = static_cast<GLenum>(jit.Regs()[0]);
        const GLintptr offset = static_cast<int32_t>(jit.Regs()[1]);
        const GLsizeiptr length = static_cast<int32_t>(jit.Regs()[2]);
        const GLbitfield access = static_cast<GLbitfield>(jit.Regs()[3]);
        if (length > 0 && static_cast<uint64_t>(length) > host.GetGuestMemorySize()) {
            jit.Regs()[0] = 0;
            return;
        }
        std::lock_guard lock(s_guest_mapping_mutex);
        // Let GLES reject invalid arguments/double maps and retain its error.
        void* host_pointer = glMapBufferRange(target, offset, length, access);
        if (!host_pointer) { jit.Regs()[0] = 0; return; }
        const GLuint buffer = NativeBoundBuffer(target);
        const uint32_t guest_address = host.GuestAlloc(static_cast<size_t>(length), 4);
        if (!guest_address) {
            glUnmapBuffer(target);
            jit.Regs()[0] = 0;
            return;
        }
        if (access & GL_MAP_READ_BIT) {
            memcpy(host.GetGuestMemoryBase() + guest_address, host_pointer,
                   static_cast<size_t>(length));
        }
        const auto old = s_guest_mapped_buffers.find(host_pointer);
        if (old != s_guest_mapped_buffers.end()) host.GuestFree(old->second.guest_address);
        s_guest_mapped_buffers[host_pointer] = GuestMappedBuffer{
            host_pointer, guest_address, static_cast<size_t>(length), access,
            ContextGroupLocked(eglGetCurrentContext()), buffer};
        jit.Regs()[0] = guest_address;
    });
    manager.RegisterThunk("glGetBufferPointerv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t output = jit.Regs()[2];
        if (!host.IsGuestMemoryRangeAccessible(output, sizeof(uint32_t))) return;
        std::lock_guard lock(s_guest_mapping_mutex);
        void* const untouched = reinterpret_cast<void*>(UINTPTR_MAX);
        void* native = untouched;
        glGetBufferPointerv(jit.Regs()[0], jit.Regs()[1], &native);
        // An invalid pname must leave the guest output untouched, like GLES.
        if (native == untouched || jit.Regs()[1] != GL_BUFFER_MAP_POINTER) return;
        const auto it = s_guest_mapped_buffers.find(native);
        host.MemoryWrite32(output, it == s_guest_mapped_buffers.end() ? 0 : it->second.guest_address);
    });
    manager.RegisterThunk("glUnmapBuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLenum target = static_cast<GLenum>(jit.Regs()[0]);
        std::lock_guard lock(s_guest_mapping_mutex);
        const auto it = s_guest_mapped_buffers.find(NativeMappedPointer(target));
        if (it == s_guest_mapped_buffers.end()) {
            jit.Regs()[0] = static_cast<uint32_t>(glUnmapBuffer(target));
            return;
        }
        const GuestMappedBuffer mapping = it->second;
        if (host.IsGuestMemoryRangeAccessible(mapping.guest_address, mapping.length)) {
            bridge_gl::CopyOnUnmap(mapping.host_pointer,
                host.GetGuestMemoryBase() + mapping.guest_address, mapping.length, mapping.access);
        }
        const GLboolean result = glUnmapBuffer(target);
        host.GuestFree(mapping.guest_address);
        s_guest_mapped_buffers.erase(it);
        jit.Regs()[0] = static_cast<uint32_t>(result);
    });
    manager.RegisterThunk("glFlushMappedBufferRange", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLenum target = static_cast<GLenum>(jit.Regs()[0]);
        const GLintptr offset = static_cast<int32_t>(jit.Regs()[1]);
        const GLsizeiptr length = static_cast<int32_t>(jit.Regs()[2]);
        std::lock_guard lock(s_guest_mapping_mutex);
        const auto it = s_guest_mapped_buffers.find(NativeMappedPointer(target));
        if (it != s_guest_mapped_buffers.end()) {
            const auto& mapping = it->second;
            if (host.IsGuestMemoryRangeAccessible(mapping.guest_address, mapping.length)) {
                bridge_gl::CopyOnFlush(mapping.host_pointer,
                    host.GetGuestMemoryBase() + mapping.guest_address, mapping.length,
                    mapping.access, offset, length);
            }
        }
        glFlushMappedBufferRange(target, offset, length);
    });
    manager.RegisterThunk("glBindBufferBase", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const GLenum target = static_cast<GLenum>(jit.Regs()[0]);
        const GLuint buffer = jit.Regs()[2];
        glBindBufferBase(target, jit.Regs()[1], buffer);
        TrackGlBufferBinding(target, buffer);
    });
    manager.RegisterThunk("glBindSampler", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glBindSampler(jit.Regs()[0], jit.Regs()[1]);
    });
    manager.RegisterThunk("glGenVertexArrays", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLsizei count = static_cast<int32_t>(jit.Regs()[0]);
        if (count <= 0) { glGenVertexArrays(count, nullptr); return; }
        auto* arrays = GuestOutputArray<GLuint>(host, jit.Regs()[1], count, 1, "glGenVertexArrays");
        if (arrays) glGenVertexArrays(count, arrays);
    });
    manager.RegisterThunk("glDeleteVertexArrays", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLsizei count = static_cast<int32_t>(jit.Regs()[0]);
        bool valid = false;
        const auto* arrays = CountedGuestValues<GLuint>(host, jit.Regs()[1], count, 1, "glDeleteVertexArrays", &valid);
        if (valid) { glDeleteVertexArrays(count, arrays); RefreshGlBindings(); }
    });
    manager.RegisterThunk("glBindVertexArray", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glBindVertexArray(jit.Regs()[0]);
        RefreshGlBindings();
    });
    manager.RegisterThunk("glIsVertexArray", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = glIsVertexArray(jit.Regs()[0]);
    });
    manager.RegisterThunk("glDrawBuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLsizei count = static_cast<int32_t>(jit.Regs()[0]);
        bool valid = false;
        const auto* buffers = CountedGuestValues<GLenum>(host, jit.Regs()[1], count, 1, "glDrawBuffers", &valid);
        if (valid) glDrawBuffers(count, buffers);
    });
    manager.RegisterThunk("glTexStorage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glTexStorage2D(jit.Regs()[0], static_cast<int32_t>(jit.Regs()[1]), jit.Regs()[2],
                       static_cast<int32_t>(jit.Regs()[3]), static_cast<int32_t>(host.MemoryRead32(jit.Regs()[13])));
    });
    manager.RegisterThunk("glTexStorage3D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glTexStorage3D(jit.Regs()[0], static_cast<int32_t>(jit.Regs()[1]), jit.Regs()[2],
                       static_cast<int32_t>(jit.Regs()[3]), static_cast<int32_t>(host.MemoryRead32(jit.Regs()[13])),
                       static_cast<int32_t>(host.MemoryRead32(jit.Regs()[13] + 4)));
    });
    manager.RegisterThunk("glTexSubImage3D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = args.U32();
        const GLint level = args.I32(), x = args.I32(), y = args.I32(), z = args.I32();
        const GLsizei width = args.I32(), height = args.I32(), depth = args.I32();
        const GLenum format = args.U32(), type = args.U32();
        const uint32_t address = args.Ptr();
        const void* pixels = nullptr;
        if (ResolveGlPixels(host, width, height, depth, format, type, address, &pixels, true))
            glTexSubImage3D(target, level, x, y, z, width, height, depth, format, type, pixels);
    });
    manager.RegisterThunk("glFenceSync", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const GLenum condition = static_cast<GLenum>(jit.Regs()[0]);
        const GLbitfield flags = static_cast<GLbitfield>(jit.Regs()[1]);
        jit.Regs()[0] = 0;
        if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
            LOGE("glFenceSync rejected without current EGL context");
            return;
        }
        std::lock_guard<std::mutex> lock(g_guest_gl_sync_mutex);
        GuestGlSyncSlot* slot = FindEmptyGuestGlSyncSlotLocked();
        const uint32_t handle = slot ? AllocateGuestGlSyncHandleLocked() : 0;
        if (!slot || !handle) {
            LOGE("glFenceSync registry exhausted count=%zu", g_guest_gl_sync_count);
            return;
        }
        const GLsync host_sync = glFenceSync(condition, flags);
        if (!host_sync) {
            LOGE("glFenceSync host call returned null condition=0x%X flags=0x%X", condition, flags);
            return;
        }
        slot->guest_handle = handle;
        slot->host_sync = host_sync;
        ++g_guest_gl_sync_count;
        jit.Regs()[0] = handle;
    });
    manager.RegisterThunk("glClientWaitSync", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const uint32_t handle = args.U32();
        const GLbitfield flags = static_cast<GLbitfield>(args.U32());
        const GLuint64 timeout = static_cast<GLuint64>(args.U64());
        jit.Regs()[0] = GL_WAIT_FAILED;
        if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
            LOGE("glClientWaitSync rejected without current EGL context handle=0x%08X", handle);
            return;
        }
        std::lock_guard<std::mutex> lock(g_guest_gl_sync_mutex);
        GuestGlSyncSlot* slot = FindGuestGlSyncLocked(handle);
        if (!slot) {
            LOGE("glClientWaitSync rejected stale handle=0x%08X", handle);
            return;
        }
        const GLenum result = glClientWaitSync(slot->host_sync, flags, timeout);
        jit.Regs()[0] = result;
    });
    manager.RegisterThunk("glDeleteSync", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        const uint32_t handle = jit.Regs()[0];
        std::lock_guard<std::mutex> lock(g_guest_gl_sync_mutex);
        GuestGlSyncSlot* slot = FindGuestGlSyncLocked(handle);
        if (!slot) {
            LOGE("glDeleteSync rejected stale handle=0x%08X", handle);
            return;
        }
        if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
            LOGE("glDeleteSync retained without current EGL context handle=0x%08X", handle);
            return;
        }
        glDeleteSync(slot->host_sync);
        *slot = {};
        --g_guest_gl_sync_count;
    });
    manager.RegisterThunk("glInvalidateFramebuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLsizei count = static_cast<int32_t>(jit.Regs()[1]);
        bool valid = false;
        const auto* attachments = CountedGuestValues<GLenum>(
            host, jit.Regs()[2], count, 1, "glInvalidateFramebuffer", &valid);
        if (valid) glInvalidateFramebuffer(jit.Regs()[0], count, attachments);
    });
    manager.RegisterThunk("glDiscardFramebufferEXT", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        static auto function = reinterpret_cast<PFNGLDISCARDFRAMEBUFFEREXTPROC>(
            eglGetProcAddress("glDiscardFramebufferEXT"));
        const GLsizei count = static_cast<int32_t>(jit.Regs()[1]);
        bool valid = false;
        const auto* attachments = CountedGuestValues<GLenum>(
            host, jit.Regs()[2], count, 1, "glDiscardFramebufferEXT", &valid);
        if (!function) {
            LOGE("Missing native glDiscardFramebufferEXT");
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
            return;
        }
        if (valid) function(jit.Regs()[0], count, attachments);
    });
    manager.RegisterThunk("glRenderbufferStorageMultisampleEXT", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        static auto function = reinterpret_cast<PFNGLRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC>(eglGetProcAddress("glRenderbufferStorageMultisampleEXT"));
        if (!function) {
            LOGE("Missing native glRenderbufferStorageMultisampleEXT");
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
            return;
        }
        function(jit.Regs()[0], static_cast<int32_t>(jit.Regs()[1]), jit.Regs()[2],
                 static_cast<int32_t>(jit.Regs()[3]), static_cast<int32_t>(host.MemoryRead32(jit.Regs()[13])));
    });
    manager.RegisterThunk("glFramebufferTexture2DMultisampleEXT", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        static auto function = reinterpret_cast<PFNGLFRAMEBUFFERTEXTURE2DMULTISAMPLEEXTPROC>(eglGetProcAddress("glFramebufferTexture2DMultisampleEXT"));
        if (!function) {
            LOGE("Missing native glFramebufferTexture2DMultisampleEXT");
            jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
            return;
        }
        function(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3],
                 static_cast<int32_t>(host.MemoryRead32(jit.Regs()[13])),
                 static_cast<int32_t>(host.MemoryRead32(jit.Regs()[13] + 4)));
    });
    manager.RegisterThunk("glProgramParameteri", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glProgramParameteri(jit.Regs()[0], jit.Regs()[1], static_cast<int32_t>(jit.Regs()[2]));
    });
    manager.RegisterThunk("glGetProgramInterfaceiv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        auto* value = GuestOutputArray<GLint>(host, jit.Regs()[3], 1, 1, "glGetProgramInterfaceiv");
        if (value) glGetProgramInterfaceiv(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], value);
    });
    manager.RegisterThunk("glGetProgramResourceiv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLuint program = args.U32();
        const GLenum interface = args.U32();
        const GLuint index = args.U32();
        const GLsizei prop_count = args.I32();
        const uint32_t props_address = args.Ptr();
        const GLsizei capacity = args.I32();
        const uint32_t length_address = args.Ptr(), params_address = args.Ptr();
        bool valid = false;
        const auto* props = CountedGuestValues<GLenum>(host, props_address, prop_count, 1, "glGetProgramResourceiv", &valid);
        if (!valid) return;
        auto* length = OptionalGuestOutput<GLsizei>(host, length_address, 1, "glGetProgramResourceiv", &valid);
        if (!valid) return;
        auto* params = capacity > 0 ? GuestOutputArray<GLint>(host, params_address, capacity, 1, "glGetProgramResourceiv") : nullptr;
        if (capacity > 0 && !params) return;
        glGetProgramResourceiv(program, interface, index, prop_count, props, capacity, length, params);
    });
    manager.RegisterThunk("glGetProgramResourceName", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLsizei capacity = static_cast<int32_t>(jit.Regs()[3]);
        bool valid = false;
        auto* length = OptionalGuestOutput<GLsizei>(host, host.MemoryRead32(jit.Regs()[13]), 1, "glGetProgramResourceName", &valid);
        if (!valid) return;
        auto* name = capacity > 0 ? GuestOutputArray<GLchar>(host, host.MemoryRead32(jit.Regs()[13] + 4), capacity, 1, "glGetProgramResourceName") : nullptr;
        if (capacity > 0 && !name) return;
        glGetProgramResourceName(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], capacity, length, name);
    });
    manager.RegisterThunk("glUniformBlockBinding", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glUniformBlockBinding(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2]);
    });
    manager.RegisterThunk("glLabelObjectEXT", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        static auto function = reinterpret_cast<PFNGLLABELOBJECTEXTPROC>(eglGetProcAddress("glLabelObjectEXT"));
        const GLsizei length = static_cast<int32_t>(jit.Regs()[2]);
        const char* label = length > 0 ? GuestInputArray<char>(host, jit.Regs()[3], length, 1, "glLabelObjectEXT") : GuestGlName(host, jit.Regs()[3]);
        if (function && (label || !jit.Regs()[3])) function(jit.Regs()[0], jit.Regs()[1], length, label);
    });
    manager.RegisterThunk("glGetObjectLabelEXT", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        static auto function = reinterpret_cast<PFNGLGETOBJECTLABELEXTPROC>(eglGetProcAddress("glGetObjectLabelEXT"));
        const GLsizei capacity = static_cast<int32_t>(jit.Regs()[2]);
        bool valid = false;
        auto* length = OptionalGuestOutput<GLsizei>(host, jit.Regs()[3], 1, "glGetObjectLabelEXT", &valid);
        if (!valid) return;
        auto* label = capacity > 0 ? GuestOutputArray<GLchar>(host, host.MemoryRead32(jit.Regs()[13]), capacity, 1, "glGetObjectLabelEXT") : nullptr;
        if (capacity > 0 && !label) return;
        if (function) function(jit.Regs()[0], jit.Regs()[1], capacity, length, label);
        else { if (length) *length = 0; if (label) *label = 0; }
    });
}
