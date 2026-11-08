cmake_minimum_required(VERSION 3.22)
project(arm32_bridge LANGUAGES C CXX)

# Pass -DBRIDGE_TARGET=<name> to match the game's System.loadLibrary() call.
set(BRIDGE_TARGET main CACHE STRING "Bridge library target name")
# Release artifacts: compile out INFO-level bridge logging (bridge_log.h).
set(BRIDGE_RELEASE_LOGS OFF CACHE BOOL "Compile out INFO-level bridge logging")
option(CMR_ANDROID_LOG_PRINT
       "Enable __android_log_print logging in the bridge" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Size: applied before FetchContent so dynarmic/fmt/mcl inherit them. Only
# JNI_OnLoad has to be exported (everything else is registered at runtime), so
# hiding the rest drops the dynsym table and most of .rela.
add_compile_options(-fvisibility=hidden -fvisibility-inlines-hidden
                    -ffunction-sections -fdata-sections)
add_link_options(-Wl,--gc-sections -Wl,--icf=all -Wl,--exclude-libs,ALL)

# Boost is an external header dependency; never vendor it into this repository.
set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "External Boost 1.83.0 source directory")
find_path(Boost_INCLUDE_DIR boost/version.hpp HINTS "${BOOST_ROOT}"
          NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
file(STRINGS "${Boost_INCLUDE_DIR}/boost/version.hpp" boost_version_line
     REGEX "^#define BOOST_VERSION [0-9]+$")
if(NOT boost_version_line STREQUAL "#define BOOST_VERSION 108300")
    message(FATAL_ERROR "This bridge requires Boost 1.83.0 headers")
endif()
set(BOOST_INCLUDEDIR "${Boost_INCLUDE_DIR}")
set(Boost_NO_SYSTEM_PATHS ON)
set(Boost_NO_BOOST_CMAKE ON)
include_directories("${Boost_INCLUDE_DIR}")

# Configure Dynarmic.
# NOTE: the real option is DYNARMIC_FRONTENDS (a list), not DYNARMIC_ENABLE_A32/A64 --
# the old names were silently ignored, so A64 was being built too.
set(DYNARMIC_FRONTENDS "A32" CACHE STRING "" FORCE)
set(DYNARMIC_TESTS OFF CACHE BOOL "" FORCE)
set(DYNARMIC_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
set(DYNARMIC_USE_BUNDLED_EXTERNALS ON CACHE BOOL "" FORCE)  # cross-compiling: never pick up host libs

message(STATUS "Configuring pinned Dynarmic dependency...")
FetchContent_Declare(
    dynarmic
    GIT_REPOSITORY https://github.com/azahar-emu/dynarmic.git
    GIT_TAG        e77b1ba0b7da7cbe93021b01a663acfe7c4dd516
)
FetchContent_GetProperties(dynarmic)
if(NOT dynarmic_POPULATED)
    FetchContent_Populate(dynarmic)
endif()
# The pinned ARM64 backend does not honor Unsafe_IgnoreGlobalMonitor for A32,
# unlike its x64 backend. Apply the small idempotent parity patch before the
# dependency is configured so a clean/offline rebuild cannot silently regress.
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            "-DDYNARMIC_SOURCE_DIR=${dynarmic_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_dynarmic_arm64.cmake"
    RESULT_VARIABLE dynarmic_patch_result
)
if(NOT dynarmic_patch_result EQUAL 0)
    message(FATAL_ERROR "Failed to apply the pinned Dynarmic ARM64 exclusive-monitor patch")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            "-DDYNARMIC_SOURCE_DIR=${dynarmic_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_dynarmic_arm64_extras.cmake"
    RESULT_VARIABLE dynarmic_extras_patch_result
)
if(NOT dynarmic_extras_patch_result EQUAL 0)
    message(FATAL_ERROR "Failed to apply pinned Dynarmic ARM64 runtime patches")
endif()
add_subdirectory("${dynarmic_SOURCE_DIR}" "${dynarmic_BINARY_DIR}")

# UnityFS uses LZ4 for virtually every asset block.  Keep the decompressor
# pinned and compile it directly into the ARM64 bridge so those blocks do not
# have to run through the ARM32 JIT.
FetchContent_Declare(
    lz4
    GIT_REPOSITORY https://github.com/lz4/lz4.git
    GIT_TAG        ebb370ca83af193212df4dcbadcc5d87bc0de2f0 # v1.10.0
)
FetchContent_GetProperties(lz4)
if(NOT lz4_POPULATED)
    FetchContent_Populate(lz4)
endif()

set(BRIDGE_SOURCES
    src/bridge_log.cpp
    src/elf_loader.cpp
    src/dynarmic_host.cpp
    src/guest_signals.cpp
    src/guest_hotpaths.cpp
    src/performance_hints.cpp
    src/thunk_manager.cpp
    src/thunks_gl.cpp
    src/thunks_gl_gles2_core.cpp
    src/thunks_libc.cpp
    src/thunks_opensles.cpp   # drop this line if the game is not on FMOD Ex
    src/jni_fake_env.cpp
    src/jni_bridge.cpp
    ${lz4_SOURCE_DIR}/lib/lz4.c
)

add_library(${BRIDGE_TARGET} SHARED ${BRIDGE_SOURCES})

# ARMv7 NEON VMLA rounds the multiply before the add. Prevent ARM64 Clang from
# contracting the native skinning replacement into fused FMLA instructions.
set_source_files_properties(src/guest_hotpaths.cpp PROPERTIES
    COMPILE_OPTIONS "-ffp-contract=off"
)

target_include_directories(${BRIDGE_TARGET} PRIVATE
    src
    ${dynarmic_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIR}
    ${lz4_SOURCE_DIR}/lib
)

target_link_libraries(${BRIDGE_TARGET} PRIVATE
    dynarmic
    GLESv3
    EGL
    OpenSLES
    android
    m
    z
)

target_compile_definitions(${BRIDGE_TARGET} PRIVATE
    ANDROID
    __ANDROID__
    CMR_ANDROID_LOG_PRINT=$<BOOL:${CMR_ANDROID_LOG_PRINT}>
)
if(BRIDGE_RELEASE_LOGS)
    target_compile_definitions(${BRIDGE_TARGET} PRIVATE BRIDGE_RELEASE_LOGS=1)
endif()

# Keep every translation unit behind the same compile-time Android log gate,
# including sources that include <android/log.h> transitively.
target_compile_options(${BRIDGE_TARGET} PRIVATE
    -include "${CMAKE_CURRENT_SOURCE_DIR}/src/android_log_gate.h"
)

if(CMR_ANDROID_LOG_PRINT)
    target_link_libraries(${BRIDGE_TARGET} PRIVATE log)
endif()

# Android 15+ devices boot with 16 KB pages; a 4 KB-aligned LOAD segment trips
# the "app isn't 16 KB compatible" loader warning.
target_link_options(${BRIDGE_TARGET} PRIVATE
    "-Wl,-z,max-page-size=16384"
    "-Wl,-z,common-page-size=16384"
    "-Wl,-z,relro"
    "-Wl,-z,now"
)
