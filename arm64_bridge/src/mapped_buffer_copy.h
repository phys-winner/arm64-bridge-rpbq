#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace bridge_gl {
constexpr uint32_t kMapWrite = 0x0002;
constexpr uint32_t kMapFlushExplicit = 0x0010;

inline void CopyOnUnmap(void* native, const void* guest, size_t size, uint32_t access) {
    if ((access & kMapWrite) && !(access & kMapFlushExplicit)) {
        std::memcpy(native, guest, size);
    }
}

inline void CopyOnFlush(void* native, const void* guest, size_t size, uint32_t access,
                        int64_t offset, int64_t length) {
    if ((access & (kMapWrite | kMapFlushExplicit)) != (kMapWrite | kMapFlushExplicit) ||
        offset < 0 || length <= 0 || uint64_t(offset) > size ||
        uint64_t(length) > size - uint64_t(offset)) return;
    std::memcpy(static_cast<uint8_t*>(native) + offset,
                static_cast<const uint8_t*>(guest) + offset, static_cast<size_t>(length));
}
} // namespace bridge_gl
