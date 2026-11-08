#pragma once

#include <cstdint>
#include <sys/types.h>

namespace bridge_perf {

// Register long-lived threads which cooperate to produce a Unity frame. The
// Android Dynamic Performance Framework treats the set as one workload.
void NoteGraphicsThread(pid_t tid);
void NoteWorkerThread(pid_t tid);

// Thread placement remains scheduler-owned. These role hooks are retained so
// callers do not need an ABI change and ADPF can still group frame producers.
void TuneRenderThread();
void TuneGraphicsThread();
void TuneWorkerThread();
void TuneAudioThread();

// Lazily creates a 120 Hz performance-hint session on the render thread.
void BeginRenderFrame();
void EndRenderFrame(int64_t actual_duration_ns);
void Shutdown();

}  // namespace bridge_perf
