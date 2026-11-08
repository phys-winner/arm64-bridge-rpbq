#include "bridge_log.h"

#if !defined(BRIDGE_RELEASE_LOGS)

#include <sys/system_properties.h>

namespace bridge_log {
bool verbose() {
    static const bool cached = [] {
        char value[PROP_VALUE_MAX] = {0};
        const int length = __system_property_get("debug.bridge.logs", value);
        return !(length > 0 && value[0] == '0');
    }();
    return cached;
}
}

#endif
