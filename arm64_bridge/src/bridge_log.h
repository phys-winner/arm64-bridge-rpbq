#pragma once
#include <android/log.h>

// Release builds compile out INFO-level bridge logging: the libc/file thunks
// emit thousands of lines per minute and logd becomes a measurable cost.
// ERROR lines remain available when AndroidLogPrint is explicitly enabled.
// Gate the wrapper as well as __android_log_print: otherwise no-log builds
// still call bridge_log::verbose() for every suppressed INFO message.
#if !CMR_ANDROID_LOG_PRINT

#define LOGI(...) ((void)0)
#define LOGE(...) ((void)0)
#define BRIDGE_GUEST_LOG_PRIORITY_OK(prio) (false)

#elif defined(BRIDGE_RELEASE_LOGS)

#define LOGI(...) do { if (false) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); } while (0)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
// Guest-forwarded stdout/stderr: pass only warnings and worse in release.
#define BRIDGE_GUEST_LOG_PRIORITY_OK(prio) ((prio) >= ANDROID_LOG_WARN)

#else

namespace bridge_log {
// Runtime escape hatch for development builds: `adb shell setprop debug.bridge.logs 0`
// silences INFO lines for the next process start.
bool verbose();
}

#define LOGI(...) do { if (bridge_log::verbose()) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); } while (0)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define BRIDGE_GUEST_LOG_PRIORITY_OK(prio) (true)

#endif
