#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <utility>

// Dense, one-based IDs. Readers never lock or copy a std::function. Replaced
// entries remain alive until the owner shuts down all dispatching threads.
template <typename Entry, size_t Capacity>
class PublishedEntries {
public:
    const Entry* Read(size_t id) const noexcept {
        return id && id <= Capacity ? slots_[id - 1].load(std::memory_order_acquire) : nullptr;
    }

    void Publish(size_t id, Entry entry) {
        if (!id || id > Capacity) throw std::out_of_range("Thunk ID outside trampoline region");
        auto owned = std::make_unique<const Entry>(std::move(entry));
        const Entry* published = owned.get();
        std::lock_guard lock(mutex_);
        entries_.push_back(std::move(owned));
        slots_[id - 1].store(published, std::memory_order_release);
    }

private:
    std::array<std::atomic<const Entry*>, Capacity> slots_{};
    std::mutex mutex_;
    std::vector<std::unique_ptr<const Entry>> entries_;
};
