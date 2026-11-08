#include "thunk_manager.h"
#include "dynarmic_host.h"
#include "thunk_types.h"
#include <GLES2/gl2.h>
#include <GLES3/gl31.h>
#include <EGL/egl.h>
#include <android/log.h>
#include <chrono>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>


// Mirror of the current buffer bindings. Querying the driver with glGetIntegerv inside
// glDrawElements/glVertexAttribPointer cost a round-trip on every draw call.
static thread_local GLuint s_bound_array_buffer = 0;
static thread_local GLuint s_bound_element_buffer = 0;
static thread_local GLuint s_bound_framebuffer = 0;
static thread_local std::unordered_map<GLenum, GLuint> s_bound_buffers;

void TrackGlBufferBinding(GLenum target, GLuint buffer) {
    s_bound_buffers[target] = buffer;
    if (target == GL_ARRAY_BUFFER) s_bound_array_buffer = buffer;
    else if (target == GL_ELEMENT_ARRAY_BUFFER) s_bound_element_buffer = buffer;
}

GLuint GetBoundGlBuffer(GLenum target) {
    if (target == GL_ARRAY_BUFFER) return s_bound_array_buffer;
    if (target == GL_ELEMENT_ARRAY_BUFFER) return s_bound_element_buffer;
    const auto it = s_bound_buffers.find(target);
    return it == s_bound_buffers.end() ? 0 : it->second;
}

void ForgetGlBufferBindings(GLsizei count, const GLuint* buffers) {
    if (count <= 0 || !buffers) return;
    for (GLsizei i = 0; i < count; ++i) {
        const GLuint deleted = buffers[i];
        if (!deleted) continue;
        if (s_bound_array_buffer == deleted) s_bound_array_buffer = 0;
        if (s_bound_element_buffer == deleted) s_bound_element_buffer = 0;
        for (auto& binding : s_bound_buffers) {
            if (binding.second == deleted) binding.second = 0;
        }
    }
}

void RefreshGlBindings() {
    if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
        s_bound_array_buffer = s_bound_element_buffer = s_bound_framebuffer = 0;
        s_bound_buffers.clear();
        return;
    }
    GLint array = 0, elements = 0, framebuffer = 0;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &elements);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
    s_bound_array_buffer = array;
    s_bound_element_buffer = elements;
    s_bound_framebuffer = framebuffer;
    s_bound_buffers[GL_ARRAY_BUFFER] = s_bound_array_buffer;
    s_bound_buffers[GL_ELEMENT_ARRAY_BUFFER] = s_bound_element_buffer;
}

void RefreshGlContextBindings() {
    s_bound_buffers.clear();
    RefreshGlBindings();
    EGLint version = 0;
    if (eglGetCurrentContext() == EGL_NO_CONTEXT ||
        !eglQueryContext(eglGetCurrentDisplay(), eglGetCurrentContext(), EGL_CONTEXT_CLIENT_VERSION, &version) ||
        version < 3) return;
    // Only on eglMakeCurrent, never on each draw/VAO bind. ES2 must not query
    // ES3 enums, otherwise refreshing the cache injects GL_INVALID_ENUM.
    for (auto pair : {std::pair{GL_COPY_READ_BUFFER, GL_COPY_READ_BUFFER_BINDING},
                      std::pair{GL_COPY_WRITE_BUFFER, GL_COPY_WRITE_BUFFER_BINDING},
                      std::pair{GL_PIXEL_PACK_BUFFER, GL_PIXEL_PACK_BUFFER_BINDING},
                      std::pair{GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_UNPACK_BUFFER_BINDING},
                      std::pair{GL_UNIFORM_BUFFER, GL_UNIFORM_BUFFER_BINDING},
                      std::pair{GL_TRANSFORM_FEEDBACK_BUFFER, GL_TRANSFORM_FEEDBACK_BUFFER_BINDING}}) {
        GLint buffer = 0;
        glGetIntegerv(pair.second, &buffer);
        TrackGlBufferBinding(pair.first, buffer);
    }
}

void ForgetGuestBufferMappings(DynarmicHost& host, GLsizei count, const GLuint* buffers);

bool ResolveGlPixels(DynarmicHost& host, int width, int height, int depth,
                     uint32_t format, uint32_t type, uint32_t address, const void** pixels, bool volume);

// UI scaling, the virtual-resolution way: tell the guest the screen is
// s_virtual{Width,Height} and stretch the result here. The guest sizes its viewport
// in those units while the real default framebuffer is still the surface size, so
// the window-system framebuffer -- and only that one -- gets mapped back. Offscreen
// targets are allocated in the guest's own units and must not be touched, which is
// what the s_bound_framebuffer check is for.
// Leave s_virtualWidth/Height at 0 to disable the whole path.
static void ScaleViewportToSurface(GLint& x, GLint& y, GLsizei& w, GLsizei& h) {
    if (s_bound_framebuffer != 0) return;
    const uint32_t vw = ThunkManager::s_virtualWidth, vh = ThunkManager::s_virtualHeight;
    const uint32_t rw = ThunkManager::s_screenWidth, rh = ThunkManager::s_screenHeight;
    if (!vw || !vh || !rw || !rh || (vw == rw && vh == rh)) return;
    x = static_cast<GLint>((float)x * rw / vw);
    w = static_cast<GLsizei>((float)w * rw / vw);
    y = static_cast<GLint>((float)y * rh / vh);
    h = static_cast<GLsizei>((float)h * rh / vh);
}
#define LOG_TAG "Arm32Bridge_GL"
#include "bridge_log.h"

#if CMR_ANDROID_LOG_PRINT && !defined(BRIDGE_RELEASE_LOGS)
class ScopedSlowGlCall final {
public:
    ScopedSlowGlCall(const char* name_, uint64_t detail_ = 0)
            : name{name_}, detail{detail_}, started{std::chrono::steady_clock::now()} {}

    ~ScopedSlowGlCall() {
        const auto elapsed_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - started)
                        .count());
        if (elapsed_us >= 1000) {
            LOGI("slow GL call name=%s elapsed_us=%llu detail=%llu", name,
                 static_cast<unsigned long long>(elapsed_us),
                 static_cast<unsigned long long>(detail));
        }
    }

private:
    const char* name;
    uint64_t detail;
    std::chrono::steady_clock::time_point started;
};
#define PROFILE_SLOW_GL(name, detail) ScopedSlowGlCall slow_gl_call{name, detail}
#else
#define PROFILE_SLOW_GL(name, detail) do { } while (false)
#endif

void RegisterGles2CoreThunks(ThunkManager& manager);

void ThunkManager::RegisterGLThunks() {
    uint8_t* mem = m_host.GetGuestMemoryBase();

    struct GlStringCache {
        std::mutex mutex;
        std::unordered_map<std::string, uint32_t> values;
    };
    auto gl_strings = std::make_shared<GlStringCache>();

    RegisterThunk("glActiveTexture", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glActiveTexture(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glAttachShader", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glAttachShader(jit.Regs()[0], jit.Regs()[1]);
    });

    RegisterThunk("glBindBuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLenum target = static_cast<GLenum>(jit.Regs()[0]);
        const GLuint buffer = jit.Regs()[1];
        glBindBuffer(target, buffer);
        TrackGlBufferBinding(target, buffer);
    });

    RegisterThunk("glBindFramebuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        s_bound_framebuffer = jit.Regs()[1];
        glBindFramebuffer(static_cast<GLenum>(jit.Regs()[0]), jit.Regs()[1]);
    });

    RegisterThunk("glBindRenderbuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glBindRenderbuffer(static_cast<GLenum>(jit.Regs()[0]), jit.Regs()[1]);
    });

    RegisterThunk("glBindTexture", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glBindTexture(static_cast<GLenum>(jit.Regs()[0]), jit.Regs()[1]);
    });

    RegisterThunk("glBlendFunc", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glBlendFunc(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLenum>(jit.Regs()[1]));
    });

    RegisterThunk("glBufferData", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLenum target = static_cast<GLenum>(jit.Regs()[0]);
        GLsizeiptr size = static_cast<GLsizeiptr>(jit.Regs()[1]);
        const void* data = GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2]);
        GLenum usage = static_cast<GLenum>(jit.Regs()[3]);
        PROFILE_SLOW_GL("glBufferData", size > 0 ? static_cast<uint64_t>(size) : 0);
        glBufferData(target, size, data, usage);
    });

    RegisterThunk("glCheckFramebufferStatus", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glCheckFramebufferStatus(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glClear", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glClear(static_cast<GLbitfield>(jit.Regs()[0]));
    });

    RegisterThunk("glClearColor", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float r = *reinterpret_cast<float*>(&jit.Regs()[0]);
        float g = *reinterpret_cast<float*>(&jit.Regs()[1]);
        float b = *reinterpret_cast<float*>(&jit.Regs()[2]);
        float a = *reinterpret_cast<float*>(&jit.Regs()[3]);
        glClearColor(r, g, b, a);
    });

    RegisterThunk("glColorMask", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glColorMask(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3]);
    });

    RegisterThunk("glCompileShader", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLuint shader = jit.Regs()[0];
        PROFILE_SLOW_GL("glCompileShader", shader);
        glCompileShader(shader);
        GLint compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            char log[1024];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            LOGE("Shader %u compile failed: %s", shader, log);
        }
    });

    RegisterThunk("glCompressedTexImage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLenum target = jit.Regs()[0];
        GLint level = jit.Regs()[1];
        GLenum internalformat = jit.Regs()[2];
        GLsizei width = jit.Regs()[3];
        uint32_t sp = jit.Regs()[13];
        GLsizei height = host.MemoryRead32(sp);
        GLint border = host.MemoryRead32(sp + 4);
        GLsizei imageSize = host.MemoryRead32(sp + 8);
        uint32_t data_gaddr = host.MemoryRead32(sp + 12);
        GLint pbo = 0;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &pbo);
        const void* data = nullptr;
        if (imageSize < 0) return;
        if (pbo != 0) {
            data = reinterpret_cast<const void*>(static_cast<uintptr_t>(data_gaddr));
        } else if (imageSize != 0) {
            if (!GuestRangeValid(host.GetGuestMemorySize(), data_gaddr,
                                 static_cast<size_t>(imageSize))) return;
            data = GuestSpan(host.GetGuestMemoryBase(), host.GetGuestMemorySize(),
                             data_gaddr, static_cast<size_t>(imageSize));
        }
        PROFILE_SLOW_GL("glCompressedTexImage2D", static_cast<uint64_t>(imageSize));
        glCompressedTexImage2D(target, level, internalformat, width, height, border, imageSize, data);
    });

    RegisterThunk("glCreateProgram", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glCreateProgram();
    });

    RegisterThunk("glCreateShader", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glCreateShader(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glCullFace", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glCullFace(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glDeleteBuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLsizei count = static_cast<GLsizei>(jit.Regs()[0]);
        const auto* buffers = reinterpret_cast<const GLuint*>(
            GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]));
        if (count > 0 && !host.IsGuestMemoryRangeAccessible(jit.Regs()[1],
                static_cast<uint64_t>(count) * sizeof(GLuint))) return;
        ForgetGuestBufferMappings(host, count, buffers);
        glDeleteBuffers(count, buffers);
        ForgetGlBufferBindings(count, buffers);
    });

    RegisterThunk("glDeleteFramebuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteFramebuffers(jit.Regs()[0], reinterpret_cast<const GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glDeleteProgram", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteProgram(jit.Regs()[0]);
    });

    RegisterThunk("glDeleteRenderbuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteRenderbuffers(jit.Regs()[0], reinterpret_cast<const GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glDeleteShader", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteShader(jit.Regs()[0]);
    });

    RegisterThunk("glDeleteTextures", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteTextures(jit.Regs()[0], reinterpret_cast<const GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glDepthFunc", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDepthFunc(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glDepthMask", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDepthMask(static_cast<GLboolean>(jit.Regs()[0]));
    });

    RegisterThunk("glDisable", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDisable(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glDisableVertexAttribArray", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDisableVertexAttribArray(jit.Regs()[0]);
    });

    RegisterThunk("glDrawArrays", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDrawArrays(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLint>(jit.Regs()[1]), static_cast<GLsizei>(jit.Regs()[2]));
    });

    RegisterThunk("glDrawElements", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t indices_gaddr = jit.Regs()[3];
        const void* indices = (s_bound_element_buffer != 0) ?
            reinterpret_cast<const void*>(static_cast<uintptr_t>(indices_gaddr)) :
            GuestToHost(host.GetGuestMemoryBase(), indices_gaddr);
        glDrawElements(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLsizei>(jit.Regs()[1]),
                       static_cast<GLenum>(jit.Regs()[2]), indices);
    });

    RegisterThunk("glEnable", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glEnable(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glEnableVertexAttribArray", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glEnableVertexAttribArray(jit.Regs()[0]);
    });

    RegisterThunk("glFlush", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glFlush();
    });

    RegisterThunk("glFramebufferRenderbuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glFramebufferRenderbuffer(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3]);
    });

    RegisterThunk("glFramebufferTexture2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glFramebufferTexture2D(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3], host.MemoryRead32(jit.Regs()[13]));
    });

    RegisterThunk("glFrontFace", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glFrontFace(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glGenBuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGenBuffers(jit.Regs()[0], reinterpret_cast<GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGenFramebuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGenFramebuffers(jit.Regs()[0], reinterpret_cast<GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGenRenderbuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGenRenderbuffers(jit.Regs()[0], reinterpret_cast<GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGenTextures", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGenTextures(jit.Regs()[0], reinterpret_cast<GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGenerateMipmap", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        PROFILE_SLOW_GL("glGenerateMipmap", jit.Regs()[0]);
        glGenerateMipmap(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glGetActiveAttrib", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        GLsizei* length = reinterpret_cast<GLsizei*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3]));
        GLint* size = reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp)));
        GLenum* type = reinterpret_cast<GLenum*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 4)));
        GLchar* name = reinterpret_cast<GLchar*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 8)));
        glGetActiveAttrib(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], length, size, type, name);
    });

    RegisterThunk("glGetActiveUniform", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        GLsizei* length = reinterpret_cast<GLsizei*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3]));
        GLint* size = reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp)));
        GLenum* type = reinterpret_cast<GLenum*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 4)));
        GLchar* name = reinterpret_cast<GLchar*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 8)));
        glGetActiveUniform(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], length, size, type, name);
    });

    RegisterThunk("glGetAttribLocation", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glGetAttribLocation(jit.Regs()[0], reinterpret_cast<const GLchar*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGetError", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glGetError();
    });

    RegisterThunk("glGetIntegerv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGetIntegerv(static_cast<GLenum>(jit.Regs()[0]),
                      reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGetProgramInfoLog", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLsizei* length = jit.Regs()[2] ?
            reinterpret_cast<GLsizei*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])) : nullptr;
        glGetProgramInfoLog(jit.Regs()[0], jit.Regs()[1], length,
                           reinterpret_cast<GLchar*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3])));
    });

    RegisterThunk("glGetProgramiv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGetProgramiv(jit.Regs()[0], static_cast<GLenum>(jit.Regs()[1]),
                       reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])));
    });

    RegisterThunk("glGetRenderbufferParameteriv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGetRenderbufferParameteriv(jit.Regs()[0], jit.Regs()[1],
                                     reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])));
    });

    RegisterThunk("glGetShaderInfoLog", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLsizei* length = jit.Regs()[2] ?
            reinterpret_cast<GLsizei*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])) : nullptr;
        glGetShaderInfoLog(jit.Regs()[0], jit.Regs()[1], length,
                          reinterpret_cast<GLchar*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3])));
    });

    RegisterThunk("glGetShaderiv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGetShaderiv(jit.Regs()[0], static_cast<GLenum>(jit.Regs()[1]),
                      reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])));
    });

    // Query the current context every time, then cache only successfully copied
    // strings. Caching a pre-MakeCurrent NULL by enum permanently made a later
    // valid context look shader-incompatible and also leaked state across hosts.
    RegisterThunk("glGetString", [gl_strings](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLenum name = static_cast<GLenum>(jit.Regs()[0]);
        const GLubyte* value = glGetString(name);
        if (!value) {
            jit.Regs()[0] = 0;
            return;
        }

        const std::string text(reinterpret_cast<const char*>(value));
        std::lock_guard<std::mutex> lock(gl_strings->mutex);
        auto found = gl_strings->values.find(text);
        if (found == gl_strings->values.end()) {
            const uint32_t guest_addr = host.GuestAlloc(text.size() + 1, 1);
            if (!guest_addr) {
                jit.Regs()[0] = 0;
                return;
            }
            memcpy(host.GetGuestMemoryBase() + guest_addr, text.c_str(), text.size() + 1);
            found = gl_strings->values.emplace(text, guest_addr).first;
        }
        jit.Regs()[0] = found->second;
    });

    RegisterThunk("glGetUniformLocation", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glGetUniformLocation(jit.Regs()[0], reinterpret_cast<const GLchar*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glHint", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glHint(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLenum>(jit.Regs()[1]));
    });

    RegisterThunk("glLineWidth", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glLineWidth(*reinterpret_cast<float*>(&jit.Regs()[0]));
    });

    RegisterThunk("glLinkProgram", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLuint prog = jit.Regs()[0];
        PROFILE_SLOW_GL("glLinkProgram", prog);
        glLinkProgram(prog);
        GLint linked = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &linked);
        if (!linked) {
            char log[1024];
            glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
            LOGE("Program %u link failed: %s", prog, log);
        }
    });

    RegisterThunk("glPixelStorei", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glPixelStorei(static_cast<GLenum>(jit.Regs()[0]), jit.Regs()[1]);
    });

    RegisterThunk("glPolygonOffset", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glPolygonOffset(*reinterpret_cast<float*>(&jit.Regs()[0]), *reinterpret_cast<float*>(&jit.Regs()[1]));
    });

    RegisterThunk("glReadPixels", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        GLenum format = host.MemoryRead32(sp);
        GLenum type = host.MemoryRead32(sp + 4);
        void* pixels = GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 8));
        glReadPixels(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3], format, type, pixels);
    });

    RegisterThunk("glRenderbufferStorage", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glRenderbufferStorage(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3]);
    });

    RegisterThunk("glScissor", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLint x = jit.Regs()[0], y = jit.Regs()[1];
        GLsizei w = jit.Regs()[2], h = jit.Regs()[3];
        ScaleViewportToSurface(x, y, w, h);
        glScissor(x, y, w, h);
    });

    RegisterThunk("glShaderSource", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLuint shader = jit.Regs()[0];
        GLsizei count = jit.Regs()[1];
        uint32_t str_array_gaddr = jit.Regs()[2];
        uint32_t len_array_gaddr = jit.Regs()[3];

        std::vector<const char*> strings(count);
        for (GLsizei i = 0; i < count; ++i) {
            uint32_t str_gaddr = host.MemoryRead32(str_array_gaddr + i * 4);
            strings[i] = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), str_gaddr));
        }

        const GLint* lengths = len_array_gaddr ?
            reinterpret_cast<const GLint*>(GuestToHost(host.GetGuestMemoryBase(), len_array_gaddr)) : nullptr;

        glShaderSource(shader, count, strings.data(), lengths);
    });

    RegisterThunk("glTexImage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLenum target = jit.Regs()[0];
        GLint level = jit.Regs()[1];
        GLint internalformat = jit.Regs()[2];
        GLsizei width = jit.Regs()[3];
        uint32_t sp = jit.Regs()[13];
        GLsizei height = host.MemoryRead32(sp);
        GLint border = host.MemoryRead32(sp + 4);
        GLenum format = host.MemoryRead32(sp + 8);
        GLenum type = host.MemoryRead32(sp + 12);
        uint32_t pixels_gaddr = host.MemoryRead32(sp + 16);
        GLint pbo = 0;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &pbo);
        const void* pixels = nullptr;
        if ((pbo != 0 || pixels_gaddr != 0) &&
            !ResolveGlPixels(host, width, height, 1, format, type, pixels_gaddr, &pixels, false)) return;
        const uint64_t pixel_count = width > 0 && height > 0
                ? static_cast<uint64_t>(width) * static_cast<uint64_t>(height)
                : 0;
        PROFILE_SLOW_GL("glTexImage2D", pixel_count);
        glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
    });

    RegisterThunk("glTexSubImage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        Arm32CallArgs args(jit, host.GetGuestMemoryBase());
        const GLenum target = static_cast<GLenum>(args.U32());
        const GLint level = static_cast<GLint>(args.I32());
        const GLint xoffset = static_cast<GLint>(args.I32());
        const GLint yoffset = static_cast<GLint>(args.I32());
        const GLsizei width = static_cast<GLsizei>(args.I32());
        const GLsizei height = static_cast<GLsizei>(args.I32());
        const GLenum format = static_cast<GLenum>(args.U32());
        const GLenum type = static_cast<GLenum>(args.U32());
        const uint32_t pixels_addr = args.Ptr();
        const void* pixels = nullptr;
        if (!ResolveGlPixels(host, width, height, 1, format, type, pixels_addr, &pixels, false)) return;
        const uint64_t pixel_count = width > 0 && height > 0
                ? static_cast<uint64_t>(width) * static_cast<uint64_t>(height)
                : 0;
        PROFILE_SLOW_GL("glTexSubImage2D", pixel_count);
        glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels);
    });

    RegisterThunk("glTexParameterf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glTexParameterf(jit.Regs()[0], jit.Regs()[1], *reinterpret_cast<float*>(&jit.Regs()[2]));
    });

    RegisterThunk("glTexParameteri", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glTexParameteri(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2]);
    });

    RegisterThunk("glUniform1f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform1f(jit.Regs()[0], *reinterpret_cast<float*>(&jit.Regs()[1]));
    });

    RegisterThunk("glUniform1i", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform1i(jit.Regs()[0], jit.Regs()[1]);
    });

    RegisterThunk("glUniform2f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform2f(jit.Regs()[0], *reinterpret_cast<float*>(&jit.Regs()[1]), *reinterpret_cast<float*>(&jit.Regs()[2]));
    });

    RegisterThunk("glUniform3f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform3f(jit.Regs()[0], *reinterpret_cast<float*>(&jit.Regs()[1]),
                    *reinterpret_cast<float*>(&jit.Regs()[2]), *reinterpret_cast<float*>(&jit.Regs()[3]));
    });

    RegisterThunk("glUniform3fv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform3fv(jit.Regs()[0], jit.Regs()[1],
                     reinterpret_cast<const GLfloat*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])));
    });

    RegisterThunk("glUniform4f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        uint32_t w_bits = host.MemoryRead32(sp);
        float w = *reinterpret_cast<float*>(&w_bits);
        glUniform4f(jit.Regs()[0], *reinterpret_cast<float*>(&jit.Regs()[1]),
                    *reinterpret_cast<float*>(&jit.Regs()[2]), *reinterpret_cast<float*>(&jit.Regs()[3]), w);
    });

    RegisterThunk("glUniform4fv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform4fv(jit.Regs()[0], jit.Regs()[1],
                     reinterpret_cast<const GLfloat*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])));
    });

    RegisterThunk("glUniformMatrix3fv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniformMatrix3fv(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2],
                           reinterpret_cast<const GLfloat*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3])));
    });

    RegisterThunk("glUniformMatrix4fv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniformMatrix4fv(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2],
                           reinterpret_cast<const GLfloat*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3])));
    });

    RegisterThunk("glUseProgram", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUseProgram(jit.Regs()[0]);
    });

    RegisterThunk("glVertexAttribPointer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLuint index = jit.Regs()[0];
        GLint size = jit.Regs()[1];
        GLenum type = jit.Regs()[2];
        GLboolean normalized = jit.Regs()[3];
        uint32_t sp = jit.Regs()[13];
        GLsizei stride = host.MemoryRead32(sp);
        uint32_t ptr_gaddr = host.MemoryRead32(sp + 4);

        const void* ptr = (s_bound_array_buffer != 0) ?
            reinterpret_cast<const void*>(static_cast<uintptr_t>(ptr_gaddr)) :
            GuestToHost(host.GetGuestMemoryBase(), ptr_gaddr);

        glVertexAttribPointer(index, size, type, normalized, stride, ptr);
    });

    RegisterThunk("glViewport", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLint x = jit.Regs()[0], y = jit.Regs()[1];
        GLsizei w = jit.Regs()[2], h = jit.Regs()[3];
        ScaleViewportToSurface(x, y, w, h);
        glViewport(x, y, w, h);
    });

    RegisterGles2CoreThunks(*this);
}
