#pragma once

#include <cstdint>
#include <mutex>

// All access, including invoking the guest, is under mutex. Recursive locking
// permits a callback to clear, replace its registration, or destroy itself.
struct AudioCallbackDelivery {
    struct Ticket { uint32_t callback, context; uint64_t generation; };
    std::recursive_mutex mutex;

    Ticket Capture() const { return {callback_, context_, generation_}; }
    bool Current(const Ticket& ticket) const {
        return ticket.callback && ticket.generation == generation_;
    }
    void Cancel() { ++generation_; }
    void Register(uint32_t callback, uint32_t context) {
        Cancel();
        callback_ = callback;
        context_ = context;
    }

private:
    uint32_t callback_ = 0, context_ = 0;
    uint64_t generation_ = 0;
};
