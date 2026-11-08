#pragma once
#include <cstdint>

namespace Dynarmic::A32 { class Jit; }
class DynarmicHost;

// Software-delivered guest signals; host/ART signal handlers are never replaced.
// ARM32 bionic sigaction is four words: handler, mask, flags, restorer.
struct GuestSignalAction {
    uint32_t handler = 0;
    uint32_t mask = 0;
    uint32_t flags = 0;
    uint32_t restorer = 0;
};
static_assert(sizeof(GuestSignalAction) == 16);

void GuestSignalsReset();
int GuestSignalActionSet(int signum, const GuestSignalAction* action, GuestSignalAction* old);
void GuestSignalRegisterThread(int tid, Dynarmic::A32::Jit* base);
void GuestSignalAddThreadJit(Dynarmic::A32::Jit* jit);
void GuestSignalUnregisterThread(int tid, Dynarmic::A32::Jit* base);
int GuestSignalSend(int tid, int signum);
bool GuestSignalPending();
void GuestSignalDeliverPending(DynarmicHost& host);
void GuestSignalWait(DynarmicHost& host);
uint64_t GuestSignalMask();
void GuestSignalSetMask(uint64_t mask);
