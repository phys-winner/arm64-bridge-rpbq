if(NOT DEFINED DYNARMIC_SOURCE_DIR)
    message(FATAL_ERROR "DYNARMIC_SOURCE_DIR is required")
endif()

set(target "${DYNARMIC_SOURCE_DIR}/src/dynarmic/backend/arm64/a32_address_space.cpp")
if(NOT EXISTS "${target}")
    message(FATAL_ERROR "Pinned Dynarmic ARM64 source is missing: ${target}")
endif()

file(READ "${target}" content)
string(FIND "${content}" "BRIDGE_ARM64_IGNORE_GLOBAL_MONITOR" already_patched)
if(NOT already_patched EQUAL -1)
    return()
endif()

set(old_include [=[#include "dynarmic/backend/arm64/a32_address_space.h"
]=])
set(new_include [=[#include "dynarmic/backend/arm64/a32_address_space.h"

#include <cstring>

#include "dynarmic/backend/x64/exclusive_monitor_friend.h"
]=])

set(old_read [=[    auto fn = [](const A32::UserConfig& conf, A32::VAddr vaddr) -> T {
        return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
]=])
set(new_read [=[    auto fn = [](const A32::UserConfig& conf, A32::VAddr vaddr) -> T {
        if (conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreGlobalMonitor)) {
            // BRIDGE_ARM64_IGNORE_GLOBAL_MONITOR: match Dynarmic's x64 path.
            // Each processor owns its reservation entry; the bridge performs
            // the eventual store with a host compare-exchange.
            const T value = (conf.callbacks->*callback)(vaddr);
            std::memcpy(GetExclusiveMonitorValuePointer(conf.global_monitor, conf.processor_id),
                        &value, sizeof(T));
            *GetExclusiveMonitorAddressPointer(conf.global_monitor, conf.processor_id) = vaddr;
            return value;
        }
        return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
]=])

set(old_write [=[    auto fn = [](const A32::UserConfig& conf, A32::VAddr vaddr, T value) -> u32 {
        return conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr,
]=])
set(new_write [=[    auto fn = [](const A32::UserConfig& conf, A32::VAddr vaddr, T value) -> u32 {
        if (conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreGlobalMonitor)) {
            const VAddr reservation =
                    *GetExclusiveMonitorAddressPointer(conf.global_monitor, conf.processor_id);
            if (reservation != vaddr) {
                return 1;
            }
            T expected;
            std::memcpy(&expected,
                        GetExclusiveMonitorValuePointer(conf.global_monitor, conf.processor_id),
                        sizeof(T));
            return (conf.callbacks->*callback)(vaddr, value, expected) ? 0 : 1;
        }
        return conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr,
]=])

string(FIND "${content}" "${old_include}" include_position)
if(include_position EQUAL -1)
    message(FATAL_ERROR "Pinned Dynarmic source no longer matches the include patch anchor")
endif()
string(REPLACE "${old_include}" "${new_include}" content "${content}")

string(FIND "${content}" "${old_read}" read_position)
if(read_position EQUAL -1)
    message(FATAL_ERROR "Pinned Dynarmic source no longer matches the read patch anchor")
endif()
string(REPLACE "${old_read}" "${new_read}" content "${content}")

string(FIND "${content}" "${old_write}" write_position)
if(write_position EQUAL -1)
    message(FATAL_ERROR "Pinned Dynarmic source no longer matches the write patch anchor")
endif()
string(REPLACE "${old_write}" "${new_write}" content "${content}")

file(WRITE "${target}" "${content}")
message(STATUS "Applied ARM64 A32 exclusive-monitor parity patch to pinned Dynarmic")
