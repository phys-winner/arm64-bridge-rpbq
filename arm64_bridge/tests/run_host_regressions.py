"""Compile production helpers/fragments on the host; no Android device required.

OpenSL native interfaces and ADPF calls are mocked. Extracted functions are the
actual source text, not rewritten models. This does not test Android's HAL/JIT.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def between(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vcvars', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audio = (ROOT / 'src/thunks_opensles.cpp').read_text(encoding='utf-8')
    perf = (ROOT / 'src/performance_hints.cpp').read_text(encoding='utf-8')
    gl = (ROOT / 'src/thunks_gl_gles2_core.cpp').read_text(encoding='utf-8')
    fragments = '\n'.join([
        'namespace audio_test {',
        'using SLObjectItf = void*; using SLPlayItf = void*; using SLVolumeItf = void*;',
        'using SLAndroidSimpleBufferQueueItf = void*; using SLAndroidConfigurationItf = void*;',
        'using SLuint32 = uint32_t;',
        between(audio, '// Each guest audio player', '// Host OpenSL ES Buffer Queue Callback'),
        between(audio, 'static void HostBqPlayerCallback', 'static uint32_t AllocateGuestInterface'),
        between(audio, 'static void RegisterPlayerHandles', 'bool IID_Matches'),
        '}',
        'namespace gl_test {',
        'using GLenum = uint32_t; using GLuint = uint32_t; using GLbitfield = uint32_t; using GLsizei = int;',
        'using EGLContext = uintptr_t; constexpr EGLContext EGL_NO_CONTEXT = 0;',
        'EGLContext current_context = 0; EGLContext eglGetCurrentContext() { return current_context; }',
        'struct DynarmicHost { std::vector<uint32_t> freed; void GuestFree(uint32_t address) { freed.push_back(address); } };',
        between(gl, 'struct GuestMappedBuffer', 'void* NativeMappedPointer'),
        between(gl, 'void TrackGlContext(', 'void RefreshGlBindings();'),
        '}',
        'namespace perf_test {',
        'using pid_t = int;',
        'struct APerformanceHintManager {}; struct APerformanceHintSession {};',
        'constexpr size_t kMaxSessionThreads = 16; constexpr int64_t kTargetFrameDurationNs = 7500000;',
        'constexpr int RTLD_NOW = 0, RTLD_LOCAL = 0, SYS_tgkill = 1;',
        'void* dlopen(const char*, int) { return nullptr; }',
        'void* dlsym(void*, const char*) { return nullptr; }',
        'int gettid() { return 10; } int getpid() { return 20; }',
        'int syscall(int, int, int tid, int) { if (tid == 33) { errno = ESRCH; return -1; } return 0; }',
        between(perf, 'using GetManagerFn', 'void AddAuxiliaryThread'),
        between(perf, 'bool LoadApi', '}  // namespace'),
        between(perf, 'void BeginRenderFrame()', '}  // namespace bridge_perf'),
        '}',
    ])
    unit = (ROOT / 'tests/host_regressions.cpp').read_text(encoding='utf-8')
    unit = unit.replace('// PRODUCTION_FRAGMENTS', fragments)
    source = out / 'host_regressions.generated.cpp'
    source.write_text(unit, encoding='utf-8')
    exe = out / 'host_regressions.exe'
    command = out / 'compile_host.cmd'
    command.write_text(
        '@echo off\n'
        f'call "{args.vcvars}" >nul\n'
        'if errorlevel 1 exit /b 1\n'
        f'cl /nologo /std:c++20 /EHsc /O2 /W4 /I"{ROOT / "src"}" "{source}" '
        f'/Fo"{out / "host_regressions.obj"}" /Fe"{exe}"\n'
        'exit /b %errorlevel%\n', encoding='utf-8')
    compiled = subprocess.run(['cmd.exe', '/d', '/c', str(command)],
                              text=True, capture_output=True)
    (out / 'compile.log').write_text(compiled.stdout + compiled.stderr, encoding='utf-8')
    if compiled.returncode:
        print(compiled.stdout, compiled.stderr)
        raise SystemExit(compiled.returncode)
    result = subprocess.run([str(exe)], text=True, capture_output=True, timeout=60)
    (out / 'results.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    print(result.stdout, end='')
    print(result.stderr, end='')
    (out / 'results.json').write_text(json.dumps({
        'exit_code': result.returncode, 'android_device_tested': False,
        'executable': str(exe), 'stdout': result.stdout,
    }, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
