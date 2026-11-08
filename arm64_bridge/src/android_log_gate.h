#pragma once

#include <android/log.h>

#ifndef CMR_ANDROID_LOG_PRINT
# define CMR_ANDROID_LOG_PRINT 0
#endif

#if !CMR_ANDROID_LOG_PRINT
# undef __android_log_print
# define __android_log_print(...) ((void)0)
#endif
